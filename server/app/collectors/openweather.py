import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import offset_coordinate, within_target_radius
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = "https://api.openweathermap.org/data/2.5/weather"
GRID_DISTANCE_KM = 25.0


@dataclass(frozen=True)
class WeatherQueryPoint:
    point_id: str
    lat: float
    lon: float


FIELD_MAP: dict[tuple[str, str], tuple[str, str]] = {
    ("main", "temp"): ("temperature", "degC"),
    ("main", "humidity"): ("humidity", "percent"),
    ("main", "pressure"): ("pressure", "hPa"),
    ("wind", "speed"): ("wind_speed", "m/s"),
    ("wind", "deg"): ("wind_direction", "degree"),
    ("clouds", "all"): ("cloud_cover", "percent"),
}


def parse_observation_time(value: Any) -> datetime | None:
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return None

    if timestamp <= 0:
        return None

    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def weather_query_points() -> list[WeatherQueryPoint]:
    lat = settings.aeris_target_lat
    lon = settings.aeris_target_lon
    distance = min(GRID_DISTANCE_KM, settings.aeris_target_radius_km)
    offsets = {
        "center": (0.0, 0.0),
        "north": (distance, 0.0),
        "east": (0.0, distance),
        "south": (-distance, 0.0),
        "west": (0.0, -distance),
    }

    points: list[WeatherQueryPoint] = []
    for point_id, (north_km, east_km) in offsets.items():
        point_lat, point_lon = offset_coordinate(
            lat,
            lon,
            north_km=north_km,
            east_km=east_km,
        )
        if within_target_radius(point_lat, point_lon):
            points.append(WeatherQueryPoint(point_id, point_lat, point_lon))

    return points


def extract_precipitation(payload: dict[str, Any]) -> float:
    rain = payload.get("rain") or {}
    snow = payload.get("snow") or {}
    total = 0.0

    for bucket in (rain, snow):
        try:
            total += float(bucket.get("1h", 0.0) or 0.0)
        except (TypeError, ValueError):
            continue

    return total


class OpenWeatherCollector(BaseCollector):
    source_name = "openweather"
    collect_interval_minutes = 60

    async def fetch(self) -> dict[str, Any]:
        """Fetch current weather for target-area grid points."""
        client = await self._get_client()
        observations: list[dict[str, Any]] = []

        for point in weather_query_points():
            params = {
                "lat": f"{point.lat:.6f}",
                "lon": f"{point.lon:.6f}",
                "appid": settings.openweather_api_key,
                "units": "metric",
            }
            try:
                response = await client.get(API_BASE, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "OpenWeather fetch failed",
                    extra={"point_id": point.point_id, "error": str(exc)},
                )
                continue

            payload = response.json()
            observations.append(
                {
                    "point_id": point.point_id,
                    "requested_lat": point.lat,
                    "requested_lon": point.lon,
                    "payload": payload,
                }
            )

        if not observations:
            raise RuntimeError("OpenWeather returned no observations")

        return {"observations": observations}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        """Transform OpenWeather current weather responses into DataPoints."""
        points: list[DataPointCreate] = []

        for observation in raw_data.get("observations", []):
            points.extend(self._normalize_observation(observation))

        return points

    def _normalize_observation(
        self,
        observation: dict[str, Any],
    ) -> list[DataPointCreate]:
        payload = observation.get("payload") or {}
        point_id = observation.get("point_id")
        if not point_id:
            return []

        timestamp = parse_observation_time(payload.get("dt"))
        if timestamp is None:
            logger.warning("Could not parse OpenWeather dt: %s", payload)
            return []

        coordinates = payload.get("coord") or {}
        lat = coordinates.get("lat", observation.get("requested_lat"))
        lon = coordinates.get("lon", observation.get("requested_lon"))
        if lat is None or lon is None:
            return []

        try:
            point_lat = float(lat)
            point_lon = float(lon)
        except (TypeError, ValueError):
            return []

        points: list[DataPointCreate] = []
        for (section_name, field_name), (metric, unit) in FIELD_MAP.items():
            section = payload.get(section_name) or {}
            value = section.get(field_name)
            if value is None:
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue

            points.append(
                self._make_point(
                    timestamp,
                    point_lat,
                    point_lon,
                    str(point_id),
                    metric,
                    numeric_value,
                    unit,
                    observation,
                )
            )

        points.append(
            self._make_point(
                timestamp,
                point_lat,
                point_lon,
                str(point_id),
                "precipitation",
                extract_precipitation(payload),
                "mm/h",
                observation,
            )
        )

        return points

    def _make_point(
        self,
        timestamp: datetime,
        lat: float,
        lon: float,
        point_id: str,
        metric: str,
        value: float,
        unit: str,
        raw_json: dict[str, Any],
    ) -> DataPointCreate:
        return DataPointCreate(
            timestamp=timestamp,
            lat=lat,
            lon=lon,
            metric=metric,
            value=value,
            unit=unit,
            source=self.source_name,
            source_entity_id=f"grid:{point_id}",
            raw_json=raw_json,
        )
