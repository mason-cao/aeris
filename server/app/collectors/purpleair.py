import logging
from datetime import datetime, timezone
from typing import Any

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box, within_target_radius
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.purpleair.com/v1/sensors"
MAX_AGE_SECONDS = 7200

REQUEST_FIELDS: tuple[str, ...] = (
    "name",
    "last_seen",
    "latitude",
    "longitude",
    "location_type",
    "private",
    "pm2.5_atm",
    "pm10.0_atm",
    "humidity",
    "temperature",
    "confidence",
    "channel_flags",
)

FIELD_MAP: dict[str, tuple[str, str]] = {
    "pm2.5_atm": ("pm25", "ug/m3"),
    "pm10.0_atm": ("pm10", "ug/m3"),
    "humidity": ("humidity", "percent"),
    "temperature": ("temperature", "degF"),
}


def parse_last_seen(value: Any) -> datetime | None:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None

    if timestamp <= 0:
        return None

    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def is_outdoor_sensor(value: Any) -> bool:
    return value in (0, "0", None)


class PurpleAirCollector(BaseCollector):
    source_name = "purpleair"
    collect_interval_minutes = 15

    async def fetch(self) -> dict[str, Any]:
        """Fetch current outdoor PurpleAir sensors for the target area."""
        client = await self._get_client()
        bbox = target_bounding_box()
        params = {
            "fields": ",".join(REQUEST_FIELDS),
            "location_type": 0,
            "max_age": MAX_AGE_SECONDS,
            "nwlng": f"{bbox.nw_lon:.6f}",
            "nwlat": f"{bbox.nw_lat:.6f}",
            "selng": f"{bbox.se_lon:.6f}",
            "selat": f"{bbox.se_lat:.6f}",
        }
        headers = {"X-API-Key": settings.purpleair_api_key}

        response = await client.get(API_BASE, params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()

        logger.debug(
            "PurpleAir returned %d sensors",
            len(payload.get("data", [])),
        )
        return payload

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        """Transform PurpleAir columnar sensor rows to normalized DataPoints."""
        fields = raw_data.get("fields", [])
        points: list[DataPointCreate] = []

        for row in raw_data.get("data", []):
            sensor = dict(zip(fields, row))
            sensor_points = self._normalize_sensor(sensor)
            points.extend(sensor_points)

        return points

    def _normalize_sensor(self, sensor: dict[str, Any]) -> list[DataPointCreate]:
        sensor_index = sensor.get("sensor_index")
        if sensor_index is None:
            return []

        if not is_outdoor_sensor(sensor.get("location_type")):
            return []

        try:
            lat = float(sensor.get("latitude"))
            lon = float(sensor.get("longitude"))
        except (TypeError, ValueError):
            return []

        if not within_target_radius(lat, lon):
            return []

        timestamp = parse_last_seen(sensor.get("last_seen"))
        if timestamp is None:
            logger.warning("Could not parse PurpleAir last_seen for sensor: %s", sensor)
            return []

        points: list[DataPointCreate] = []
        for field_name, (metric, unit) in FIELD_MAP.items():
            raw_value = sensor.get(field_name)
            if raw_value is None:
                continue

            try:
                value = float(raw_value)
            except (TypeError, ValueError):
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
                    source_entity_id=str(sensor_index),
                    raw_json=sensor,
                )
            )

        return points
