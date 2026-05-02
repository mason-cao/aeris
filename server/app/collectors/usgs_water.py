import logging
from datetime import datetime
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box, within_target_radius

logger = logging.getLogger(__name__)

API_BASE = "https://waterservices.usgs.gov/nwis/iv/"

PARAMETER_MAP: dict[str, tuple[str, str]] = {
    "00060": ("stream_flow", "ft3/s"),
    "00010": ("water_temperature", "degC"),
    "00095": ("specific_conductance", "uS/cm"),
    "00065": ("gauge_height", "ft"),
    "63680": ("turbidity", "FNU"),
}


def parse_usgs_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def usgs_bbox_param() -> str:
    """USGS bbox order is west,south,east,north."""
    bbox = target_bounding_box()
    return (
        f"{bbox.min_lon:.6f},{bbox.min_lat:.6f},"
        f"{bbox.max_lon:.6f},{bbox.max_lat:.6f}"
    )


class USGSWaterCollector(BaseCollector):
    source_name = "usgs_water"
    collect_interval_minutes = 60

    async def fetch(self) -> dict[str, Any]:
        client = await self._get_client()
        params = {
            "format": "json",
            "bBox": usgs_bbox_param(),
            "parameterCd": ",".join(PARAMETER_MAP),
            "siteStatus": "active",
        }

        response = await client.get(API_BASE, params=params)
        response.raise_for_status()
        return response.json()

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        points: list[DataPointCreate] = []
        time_series = (
            raw_data.get("value", {}).get("timeSeries", [])
            if isinstance(raw_data, dict)
            else []
        )

        for series in time_series:
            points.extend(self._normalize_series(series))

        return points

    def _normalize_series(self, series: dict[str, Any]) -> list[DataPointCreate]:
        source_info = series.get("sourceInfo") or {}
        geo = (source_info.get("geoLocation") or {}).get("geogLocation") or {}
        try:
            lat = float(geo.get("latitude"))
            lon = float(geo.get("longitude"))
        except (TypeError, ValueError):
            return []

        if not within_target_radius(lat, lon):
            return []

        site_codes = source_info.get("siteCode") or []
        site_code = site_codes[0].get("value") if site_codes else None
        if not site_code:
            return []

        variable = series.get("variable") or {}
        codes = variable.get("variableCode") or []
        parameter_code = codes[0].get("value") if codes else None
        mapping = PARAMETER_MAP.get(str(parameter_code))
        if mapping is None:
            logger.debug("Skipping unknown USGS parameter: %s", parameter_code)
            return []
        metric, unit = mapping

        points: list[DataPointCreate] = []
        for value_block in series.get("values") or []:
            for entry in value_block.get("value") or []:
                point = self._normalize_value(
                    entry,
                    site_code=str(site_code),
                    lat=lat,
                    lon=lon,
                    metric=metric,
                    unit=unit,
                    series=series,
                )
                if point is not None:
                    points.append(point)

        return points

    def _normalize_value(
        self,
        entry: dict[str, Any],
        *,
        site_code: str,
        lat: float,
        lon: float,
        metric: str,
        unit: str,
        series: dict[str, Any],
    ) -> DataPointCreate | None:
        raw_value = entry.get("value")
        if raw_value is None or str(raw_value).strip() in {"", "-999999"}:
            return None

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        timestamp = parse_usgs_datetime(entry.get("dateTime"))
        if timestamp is None:
            return None

        return DataPointCreate(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            metric=metric,
            value=value,
            unit=unit,
            source=self.source_name,
            source_entity_id=site_code,
            raw_json={"variable": series.get("variable"), "value": entry},
        )
