import logging
import math
from datetime import datetime, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.openaq.org/v3"
LOCATIONS_LIMIT = 1000

PARAMETER_MAP: dict[str, str] = {
    "pm25": "pm25",
    "pm2.5": "pm25",
    "pm10": "pm10",
    "o3": "ozone",
    "ozone": "ozone",
    "no2": "no2",
    "so2": "so2",
    "co": "co",
    "bc": "bc",
}

UNIT_MAP: dict[str, str] = {
    "ug/m3": "ug/m3",
    "ug/m^3": "ug/m3",
    "ppm": "ppm",
    "ppb": "ppb",
}


def normalize_openaq_unit(unit: str | None) -> str:
    """Normalize OpenAQ unit labels to the canonical units stored by AERIS."""
    if not unit:
        return "unknown"

    cleaned = unit.strip().lower()
    cleaned = (
        cleaned.replace("\u00b5", "u")
        .replace("\u03bc", "u")
        .replace("\u00b3", "3")
        .replace(" ", "")
    )
    return UNIT_MAP.get(cleaned, cleaned)


def parse_openaq_datetime(value: str | None) -> datetime | None:
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def target_bbox() -> str:
    """Return OpenAQ bbox param: min lon, min lat, max lon, max lat."""
    lat = settings.aeris_target_lat
    lon = settings.aeris_target_lon
    radius_km = settings.aeris_target_radius_km

    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.01))

    return (
        f"{lon - lon_delta:.4f},{lat - lat_delta:.4f},"
        f"{lon + lon_delta:.4f},{lat + lat_delta:.4f}"
    )


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates."""
    earth_radius_km = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(math.sqrt(a))


def location_within_target_radius(location: dict[str, Any]) -> bool:
    coordinates = location.get("coordinates") or {}
    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")
    if lat is None or lon is None:
        return False

    return (
        distance_km(
            settings.aeris_target_lat,
            settings.aeris_target_lon,
            float(lat),
            float(lon),
        )
        <= settings.aeris_target_radius_km
    )


class OpenAQCollector(BaseCollector):
    source_name = "openaq"
    collect_interval_minutes = 60

    async def fetch(self) -> dict[str, Any]:
        """Fetch current OpenAQ sensors for the target area."""
        client = await self._get_client()
        headers = (
            {"X-API-Key": settings.openaq_api_key}
            if settings.openaq_api_key
            else {}
        )
        params = {
            "bbox": target_bbox(),
            "limit": LOCATIONS_LIMIT,
        }

        response = await client.get(f"{API_BASE}/locations", params=params, headers=headers)
        response.raise_for_status()
        payload = response.json()

        meta = payload.get("meta") or {}
        found = meta.get("found")
        if isinstance(found, int) and found > LOCATIONS_LIMIT:
            logger.warning(
                "OpenAQ locations response clipped",
                extra={"found": found, "limit": LOCATIONS_LIMIT},
            )

        locations = [
            location
            for location in payload.get("results", [])
            if location_within_target_radius(location)
        ]
        sensors_by_location_id: dict[str, list[dict[str, Any]]] = {}

        for location in locations:
            location_id = location.get("id")
            if location_id is None:
                continue

            try:
                sensors_response = await client.get(
                    f"{API_BASE}/locations/{location_id}/sensors",
                    headers=headers,
                )
                sensors_response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "OpenAQ sensor fetch failed",
                    extra={"location_id": location_id, "error": str(exc)},
                )
                continue

            sensors_by_location_id[str(location_id)] = sensors_response.json().get(
                "results", []
            )

        logger.debug(
            "OpenAQ returned %d locations and %d sensors",
            len(locations),
            sum(len(sensors) for sensors in sensors_by_location_id.values()),
        )
        return {
            "locations": locations,
            "sensors_by_location_id": sensors_by_location_id,
        }

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        """Transform OpenAQ sensor latest values to normalized DataPoints."""
        points: list[DataPointCreate] = []
        locations_by_id = {
            str(location.get("id")): location
            for location in raw_data.get("locations", [])
            if location.get("id") is not None
        }

        for location_id, sensors in raw_data.get("sensors_by_location_id", {}).items():
            location = locations_by_id.get(str(location_id), {})
            location_coordinates = location.get("coordinates") or {}

            for sensor in sensors:
                point = self._normalize_sensor(sensor, location_coordinates, location)
                if point is not None:
                    points.append(point)

        return points

    def _normalize_sensor(
        self,
        sensor: dict[str, Any],
        location_coordinates: dict[str, Any],
        location: dict[str, Any],
    ) -> DataPointCreate | None:
        parameter = sensor.get("parameter") or {}
        parameter_name = str(parameter.get("name", "")).lower()
        metric = PARAMETER_MAP.get(parameter_name)
        if metric is None:
            logger.debug("Skipping unknown OpenAQ parameter: %s", parameter_name)
            return None

        latest = sensor.get("latest")
        if latest is None:
            return None

        try:
            value = float(latest.get("value"))
        except (TypeError, ValueError):
            return None

        timestamp = parse_openaq_datetime((latest.get("datetime") or {}).get("utc"))
        if timestamp is None:
            logger.warning("Could not parse OpenAQ timestamp for sensor: %s", sensor)
            return None

        latest_coordinates = latest.get("coordinates") or {}
        lat = latest_coordinates.get("latitude", location_coordinates.get("latitude"))
        lon = latest_coordinates.get("longitude", location_coordinates.get("longitude"))
        if lat is None or lon is None:
            return None

        sensor_id = sensor.get("id")
        if sensor_id is None:
            return None

        return DataPointCreate(
            timestamp=timestamp,
            lat=float(lat),
            lon=float(lon),
            metric=metric,
            value=value,
            unit=normalize_openaq_unit(parameter.get("units")),
            source=self.source_name,
            source_entity_id=str(sensor_id),
            raw_json={"location": location, "sensor": sensor},
        )
