import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.config import settings

logger = logging.getLogger(__name__)

# EPA AirNow parameter name → our normalized metric name
PARAMETER_MAP: dict[str, str] = {
    "PM2.5": "pm25",
    "PM10": "pm10",
    "OZONE": "ozone",
    "O3": "ozone",
    "NO2": "no2",
    "SO2": "so2",
    "CO": "co",
}

EPA_UNIT_MAP: dict[str, str] = {
    "UG/M3": "ug/m3",
    "UG/M^3": "ug/m3",
    "PPM": "ppm",
    "PPB": "ppb",
}

TIMEZONE_OFFSET_HOURS: dict[str, int] = {
    "UTC": 0,
    "GMT": 0,
    "EST": -5,
    "EDT": -4,
    "CST": -6,
    "CDT": -5,
    "MST": -7,
    "MDT": -6,
    "PST": -8,
    "PDT": -7,
    "AKST": -9,
    "AKDT": -8,
    "HST": -10,
}

API_BASE = "https://www.airnowapi.org/aq/observation/latLong/current/"


def normalize_epa_unit(unit: str | None) -> str:
    """Normalize EPA unit labels to the canonical units stored by AERIS."""
    if not unit:
        return "unknown"
    cleaned = unit.strip().upper()
    return EPA_UNIT_MAP.get(cleaned, cleaned.lower())


def parse_observation_timestamp(
    date_observed: str | None,
    hour_observed: Any,
    local_timezone: str | None,
) -> datetime | None:
    if not date_observed:
        return None

    try:
        parsed = datetime.strptime(date_observed.strip(), "%Y-%m-%d").replace(
            hour=int(hour_observed),
        )
    except (ValueError, TypeError):
        return None

    timezone_name = (local_timezone or "UTC").strip().upper()
    offset_hours = TIMEZONE_OFFSET_HOURS.get(timezone_name, 0)
    local_tz = timezone(timedelta(hours=offset_hours))
    return parsed.replace(tzinfo=local_tz).astimezone(timezone.utc)


class EPAAirNowCollector(BaseCollector):
    source_name = "epa_airnow"
    collect_interval_minutes = 60

    async def fetch(self) -> dict[str, Any]:
        """Fetch current observations from EPA AirNow for the target area."""
        client = await self._get_client()
        params = {
            "format": "application/json",
            "latitude": settings.aeris_target_lat,
            "longitude": settings.aeris_target_lon,
            "distance": int(settings.aeris_target_radius_km * 0.621371),  # km → miles
            "API_KEY": settings.airnow_api_key,
        }

        response = await client.get(API_BASE, params=params)
        response.raise_for_status()

        observations = response.json()
        logger.debug(
            "EPA AirNow returned %d observations",
            len(observations),
        )
        return {"observations": observations}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        """Transform EPA AirNow observations to normalized DataPoints."""
        points: list[DataPointCreate] = []

        for obs in raw_data.get("observations", []):
            param_name = str(obs.get("ParameterName", "")).strip().upper()
            metric = PARAMETER_MAP.get(param_name)
            if metric is None:
                logger.debug("Skipping unknown parameter: %s", param_name)
                continue

            raw_value = obs.get("Value")
            unit = normalize_epa_unit(obs.get("Unit"))
            if raw_value is None:
                raw_value = obs.get("AQI")
                unit = "AQI"
                metric = f"{metric}_aqi"
            if raw_value is None:
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue

            ts = parse_observation_timestamp(
                obs.get("DateObserved"),
                obs.get("HourObserved", 0),
                obs.get("LocalTimeZone"),
            )
            if ts is None:
                logger.warning("Could not parse timestamp for observation: %s", obs)
                continue

            points.append(
                DataPointCreate(
                    timestamp=ts,
                    lat=obs.get("Latitude", settings.aeris_target_lat),
                    lon=obs.get("Longitude", settings.aeris_target_lon),
                    metric=metric,
                    value=value,
                    unit=unit,
                    source=self.source_name,
                    source_entity_id=str(obs.get("ReportingArea", "unknown")),
                    raw_json=obs,
                )
            )

        return points
