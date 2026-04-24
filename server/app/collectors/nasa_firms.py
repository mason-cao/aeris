import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
DAY_RANGE = 1
FIRMS_RADIUS_KM = 100.0
FIRMS_SOURCES: tuple[str, ...] = (
    "VIIRS_NOAA20_NRT",
    "VIIRS_SNPP_NRT",
    "MODIS_NRT",
)

CONFIDENCE_MAP: dict[str, float] = {
    "l": 33.0,
    "low": 33.0,
    "n": 66.0,
    "nominal": 66.0,
    "h": 100.0,
    "high": 100.0,
}


def firms_area_coordinates() -> str:
    """Return FIRMS area coordinates: west,south,east,north."""
    return target_bounding_box(radius_km=FIRMS_RADIUS_KM).as_csv()


def parse_firms_csv(source: str, text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("<") or "Invalid MAP_KEY" in stripped:
        raise ValueError("FIRMS returned a non-CSV response")

    reader = csv.DictReader(io.StringIO(stripped))
    if not reader.fieldnames:
        return []

    return [{"source_dataset": source, **row} for row in reader]


def parse_acquisition_time(row: dict[str, Any]) -> datetime | None:
    date_value = str(row.get("acq_date", "")).strip()
    time_value = str(row.get("acq_time", "")).strip().zfill(4)
    if not date_value or len(time_value) != 4:
        return None

    try:
        return datetime.strptime(
            f"{date_value} {time_value}",
            "%Y-%m-%d %H%M",
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def parse_confidence(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None
    if text in CONFIDENCE_MAP:
        return CONFIDENCE_MAP[text]

    try:
        return float(text)
    except ValueError:
        return None


def parse_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def detection_entity_id(row: dict[str, Any], timestamp: datetime) -> str:
    source = row.get("source_dataset", "unknown")
    satellite = row.get("satellite", "unknown")
    lat = parse_float(row.get("latitude"))
    lon = parse_float(row.get("longitude"))
    lat_text = f"{lat:.5f}" if lat is not None else "unknown"
    lon_text = f"{lon:.5f}" if lon is not None else "unknown"
    ts_text = timestamp.strftime("%Y%m%d%H%M")
    return f"{source}:{satellite}:{lat_text}:{lon_text}:{ts_text}"


class NASAFIRMSCollector(BaseCollector):
    source_name = "nasa_firms"
    collect_interval_minutes = 180

    async def fetch(self) -> dict[str, Any]:
        """Fetch recent active fire detections from NASA FIRMS."""
        if not settings.firms_map_key:
            raise RuntimeError("FIRMS_MAP_KEY is required")

        client = await self._get_client()
        area = firms_area_coordinates()
        rows: list[dict[str, Any]] = []
        failures: list[str] = []
        successful_requests = 0

        for source in FIRMS_SOURCES:
            url = f"{API_BASE}/{settings.firms_map_key}/{source}/{area}/{DAY_RANGE}"
            try:
                response = await client.get(url)
                response.raise_for_status()
                rows.extend(parse_firms_csv(source, response.text))
                successful_requests += 1
            except (httpx.HTTPError, ValueError) as exc:
                failures.append(f"{source}: {exc}")
                logger.warning(
                    "NASA FIRMS fetch failed",
                    extra={"source_dataset": source, "error": str(exc)},
                )

        if successful_requests == 0:
            raise RuntimeError("All NASA FIRMS source requests failed")

        return {"detections": rows, "errors": failures}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        """Transform FIRMS fire detections into normalized DataPoints."""
        points: list[DataPointCreate] = []

        for row in raw_data.get("detections", []):
            points.extend(self._normalize_detection(row))

        return points

    def _normalize_detection(self, row: dict[str, Any]) -> list[DataPointCreate]:
        timestamp = parse_acquisition_time(row)
        lat = parse_float(row.get("latitude"))
        lon = parse_float(row.get("longitude"))
        if timestamp is None or lat is None or lon is None:
            return []

        source_entity_id = detection_entity_id(row, timestamp)
        points: list[DataPointCreate] = []

        metric_values = [
            ("fire_radiative_power", parse_float(row.get("frp")), "MW"),
            ("fire_confidence", parse_confidence(row.get("confidence")), "percent"),
            (
                "fire_brightness",
                parse_float(row.get("bright_ti4") or row.get("brightness")),
                "K",
            ),
        ]

        for metric, value, unit in metric_values:
            if value is None:
                continue
            points.append(
                DataPointCreate(
                    timestamp=timestamp,
                    lat=lat,
                    lon=lon,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source=self.source_name,
                    source_entity_id=source_entity_id,
                    raw_json=row,
                )
            )

        return points
