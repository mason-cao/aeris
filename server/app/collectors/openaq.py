import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box, within_target_radius
from app.collectors.ratelimit import AsyncRateLimiter, rate_limited_get
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.openaq.org/v3"
LOCATIONS_LIMIT = 1000
# A station has a handful of sensors; request a full page rather than trusting
# the API's default limit.
SENSORS_LIMIT = 100

# Stay well under OpenAQ's documented 60/min. One shared limiter per process:
# the collector and the API backfill spend from the same key's budget.
OPENAQ_MAX_REQUESTS_PER_MINUTE = 30
OPENAQ_LIMITER = AsyncRateLimiter(OPENAQ_MAX_REQUESTS_PER_MINUTE)

# The station topology barely changes; refetch it daily, not hourly.
LOCATIONS_CACHE_TTL_S = 24 * 3600.0
_locations_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def clear_locations_cache() -> None:
    _locations_cache.clear()

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


def location_within_target_radius(location: dict[str, Any]) -> bool:
    coordinates = location.get("coordinates") or {}
    lat = coordinates.get("latitude")
    lon = coordinates.get("longitude")
    if lat is None or lon is None:
        return False
    try:
        return within_target_radius(float(lat), float(lon))
    except (TypeError, ValueError):
        logger.debug("Skipping OpenAQ location with bad coordinates: %s/%s", lat, lon)
        return False


class OpenAQCollector(BaseCollector):
    source_name = "openaq"
    collect_interval_minutes = 60

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        super().__init__(http_client)
        self._limiter = rate_limiter or OPENAQ_LIMITER

    async def _target_locations(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        bbox = target_bounding_box().as_csv()
        cached = _locations_cache.get(bbox)
        if cached is not None and time.monotonic() - cached[0] < LOCATIONS_CACHE_TTL_S:
            return cached[1]

        response = await rate_limited_get(
            client,
            f"{API_BASE}/locations",
            limiter=self._limiter,
            params={"bbox": bbox, "limit": LOCATIONS_LIMIT},
            headers=headers,
        )
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
        _locations_cache[bbox] = (time.monotonic(), locations)
        return locations

    async def fetch(self) -> dict[str, Any]:
        """Fetch current OpenAQ sensors for the target area."""
        client = await self._get_client()
        headers = (
            {"X-API-Key": settings.openaq_api_key}
            if settings.openaq_api_key
            else {}
        )

        try:
            locations = await self._target_locations(client, headers)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code in (401, 403):
                logger.error(
                    "OpenAQ API key rejected (HTTP %s); check OPENAQ_API_KEY",
                    exc.response.status_code,
                )
            raise
        sensors_by_location_id: dict[str, list[dict[str, Any]]] = {}

        for location in locations:
            location_id = location.get("id")
            if location_id is None:
                continue

            try:
                sensors_response = await rate_limited_get(
                    client,
                    f"{API_BASE}/locations/{location_id}/sensors",
                    limiter=self._limiter,
                    params={"limit": SENSORS_LIMIT},
                    headers=headers,
                )
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

        age_s = (datetime.now(timezone.utc) - timestamp).total_seconds()
        if age_s > settings.openaq_max_reading_age_s:
            logger.debug(
                "Skipping stale OpenAQ reading for sensor %s (%s): %.0fs old",
                sensor.get("id"),
                parameter_name,
                age_s,
            )
            return None

        latest_coordinates = latest.get("coordinates") or {}
        lat = latest_coordinates.get("latitude", location_coordinates.get("latitude"))
        lon = latest_coordinates.get("longitude", location_coordinates.get("longitude"))
        if lat is None or lon is None:
            return None
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            logger.debug(
                "Skipping OpenAQ sensor %s with bad coordinates: %s/%s",
                sensor.get("id"),
                lat,
                lon,
            )
            return None

        sensor_id = sensor.get("id")
        if sensor_id is None:
            return None

        return DataPointCreate(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            metric=metric,
            value=value,
            unit=normalize_openaq_unit(parameter.get("units")),
            source=self.source_name,
            source_entity_id=str(sensor_id),
            raw_json={"location": location, "sensor": sensor},
        )
