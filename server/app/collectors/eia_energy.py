import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

DEFAULT_RESPONDENT = "SOCO"
LOOKBACK_HOURS = 72

SERIES_MAP: dict[str, tuple[str, str]] = {
    "D": ("grid_demand", "MWh"),
    "DF": ("grid_demand_forecast", "MWh"),
    "NG": ("grid_net_generation", "MWh"),
    "TI": ("grid_total_interchange", "MWh"),
}


def parse_eia_period(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    formats = ("%Y-%m-%dT%H", "%Y-%m-%dT%H:%M", "%Y-%m-%d")
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def lookback_window(now: datetime | None = None) -> tuple[str, str]:
    end = now or datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=LOOKBACK_HOURS)
    return start.strftime("%Y-%m-%dT%H"), end.strftime("%Y-%m-%dT%H")


class EIAEnergyCollector(BaseCollector):
    source_name = "eia_energy"
    collect_interval_minutes = 60

    def __init__(self, respondent: str = DEFAULT_RESPONDENT, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.respondent = respondent

    async def fetch(self) -> dict[str, Any]:
        if not settings.eia_api_key:
            raise RuntimeError("EIA_API_KEY is required")

        client = await self._get_client()
        start, end = lookback_window()
        params: list[tuple[str, str]] = [
            ("api_key", settings.eia_api_key),
            ("frequency", "hourly"),
            ("data[0]", "value"),
            ("facets[respondent][]", self.respondent),
            ("start", start),
            ("end", end),
            ("sort[0][column]", "period"),
            ("sort[0][direction]", "desc"),
            ("length", "5000"),
        ]

        response = await client.get(API_BASE, params=params)
        response.raise_for_status()
        return response.json()

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        records = (
            raw_data.get("response", {}).get("data", [])
            if isinstance(raw_data, dict)
            else []
        )

        points: list[DataPointCreate] = []
        for record in records:
            point = self._normalize_record(record)
            if point is not None:
                points.append(point)

        return points

    def _normalize_record(self, record: dict[str, Any]) -> DataPointCreate | None:
        series_type = record.get("type")
        mapping = SERIES_MAP.get(str(series_type))
        if mapping is None:
            logger.debug("Skipping unknown EIA series type: %s", series_type)
            return None
        metric, unit = mapping

        timestamp = parse_eia_period(record.get("period"))
        if timestamp is None:
            return None

        raw_value = record.get("value")
        if raw_value is None:
            return None
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return None

        respondent = str(record.get("respondent") or self.respondent)
        return DataPointCreate(
            timestamp=timestamp,
            lat=settings.aeris_target_lat,
            lon=settings.aeris_target_lon,
            metric=metric,
            value=value,
            unit=unit,
            source=self.source_name,
            source_entity_id=f"{respondent}:{series_type}",
            raw_json=record,
        )
