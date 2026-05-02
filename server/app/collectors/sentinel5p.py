import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
COLLECTION_NAME = "SENTINEL-5P"
LOOKBACK_HOURS = 48
RESULT_LIMIT = 200

# S5P L2 product codes (from filename) → AERIS metric prefix
PRODUCT_TYPE_MAP: dict[str, str] = {
    "NO2": "s5p_no2",
    "SO2": "s5p_so2",
    "CO": "s5p_co",
    "O3": "s5p_o3",
    "HCHO": "s5p_hcho",
    "CH4": "s5p_ch4",
    "AER_AI": "s5p_aer_ai",
}

PRODUCT_PATTERN = re.compile(r"S5P_\w+_L2__([A-Z0-9_]+?)_+\d{8}T")


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip().rstrip("Z")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def extract_product_code(name: str | None) -> str | None:
    if not name:
        return None
    match = PRODUCT_PATTERN.match(name)
    if not match:
        return None
    return match.group(1).rstrip("_")


def attribute_value(attributes: list[dict[str, Any]], name: str) -> Any:
    for attribute in attributes or []:
        if attribute.get("Name") == name:
            return attribute.get("Value")
    return None


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def area_polygon() -> str:
    bbox = target_bounding_box()
    return (
        f"POLYGON(({bbox.min_lon:.6f} {bbox.min_lat:.6f},"
        f"{bbox.max_lon:.6f} {bbox.min_lat:.6f},"
        f"{bbox.max_lon:.6f} {bbox.max_lat:.6f},"
        f"{bbox.min_lon:.6f} {bbox.max_lat:.6f},"
        f"{bbox.min_lon:.6f} {bbox.min_lat:.6f}))"
    )


def odata_filter(now: datetime | None = None) -> str:
    end = now or datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)
    iso_start = start.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    polygon = area_polygon()
    return (
        f"Collection/Name eq '{COLLECTION_NAME}' "
        f"and ContentDate/Start gt {iso_start} "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{polygon}')"
    )


class Sentinel5PCollector(BaseCollector):
    """Collects Sentinel-5P L2 granule metadata covering the target area.

    Records granule availability and metadata-level cloud cover. Extraction
    of column-density values requires NetCDF processing of the granule
    payload and is deferred to a follow-up.
    """

    source_name = "sentinel5p"
    collect_interval_minutes = 360

    async def fetch(self) -> dict[str, Any]:
        client = await self._get_client()
        params = {
            "$filter": odata_filter(),
            "$top": str(RESULT_LIMIT),
            "$orderby": "ContentDate/Start desc",
            "$expand": "Attributes",
        }

        response = await client.get(API_BASE, params=params)
        response.raise_for_status()
        return response.json()

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        records = raw_data.get("value", []) if isinstance(raw_data, dict) else []
        points: list[DataPointCreate] = []
        for record in records:
            points.extend(self._normalize_record(record))
        return points

    def _normalize_record(self, record: dict[str, Any]) -> list[DataPointCreate]:
        product_id = record.get("Id")
        name = record.get("Name")
        if not product_id or not name:
            return []

        product_code = extract_product_code(name)
        metric_prefix = PRODUCT_TYPE_MAP.get(product_code or "")
        if metric_prefix is None:
            logger.debug("Skipping unmapped S5P product: %s", name)
            return []

        content_date = record.get("ContentDate") or {}
        timestamp = parse_iso_datetime(content_date.get("Start"))
        if timestamp is None:
            return []

        attributes = record.get("Attributes") or []
        readings: list[tuple[str, float, str]] = [
            (f"{metric_prefix}_granule_available", 1.0, "count"),
        ]

        cloud_cover = safe_float(attribute_value(attributes, "cloudCover"))
        if cloud_cover is not None:
            readings.append((f"{metric_prefix}_cloud_cover", cloud_cover, "percent"))

        points: list[DataPointCreate] = []
        for metric, value, unit in readings:
            points.append(
                DataPointCreate(
                    timestamp=timestamp,
                    lat=settings.aeris_target_lat,
                    lon=settings.aeris_target_lon,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source=self.source_name,
                    source_entity_id=str(product_id),
                    raw_json=record,
                )
            )
        return points
