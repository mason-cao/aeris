import logging
from datetime import datetime, timezone
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

API_BASE = "https://www.airnowapi.org/aq/observation/latLong/current/"


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
            param_name = obs.get("ParameterName", "")
            metric = PARAMETER_MAP.get(param_name)
            if metric is None:
                logger.debug("Skipping unknown parameter: %s", param_name)
                continue

            aqi_value = obs.get("AQI")
            if aqi_value is None:
                continue

            # Parse the observation timestamp
            date_observed = obs.get("DateObserved", "").strip()
            hour_observed = obs.get("HourObserved", 0)
            try:
                ts = datetime.strptime(date_observed, "%Y-%m-%d").replace(
                    hour=int(hour_observed),
                    tzinfo=timezone.utc,
                )
            except (ValueError, TypeError):
                logger.warning("Could not parse timestamp for observation: %s", obs)
                continue

            points.append(
                DataPointCreate(
                    timestamp=ts,
                    lat=obs.get("Latitude", settings.aeris_target_lat),
                    lon=obs.get("Longitude", settings.aeris_target_lon),
                    metric=metric,
                    value=float(aqi_value),
                    source=self.source_name,
                    raw_json=obs,
                )
            )

        return points
