import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import offset_coordinate, within_target_radius
from app.config import settings

logger = logging.getLogger(__name__)

API_BASE = (
    "https://api.tomtom.com/traffic/services/4/flowSegmentData"
    "/absolute/10/json"
)
GRID_DISTANCE_KM = 25.0


@dataclass(frozen=True)
class TrafficQueryPoint:
    point_id: str
    lat: float
    lon: float


def traffic_query_points() -> list[TrafficQueryPoint]:
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

    points: list[TrafficQueryPoint] = []
    for point_id, (north_km, east_km) in offsets.items():
        point_lat, point_lon = offset_coordinate(
            lat,
            lon,
            north_km=north_km,
            east_km=east_km,
        )
        if within_target_radius(point_lat, point_lon):
            points.append(TrafficQueryPoint(point_id, point_lat, point_lon))

    return points


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class TomTomTrafficCollector(BaseCollector):
    source_name = "tomtom_traffic"
    collect_interval_minutes = 15

    async def fetch(self) -> dict[str, Any]:
        if not settings.tomtom_api_key:
            raise RuntimeError("TOMTOM_API_KEY is required")

        client = await self._get_client()
        observations: list[dict[str, Any]] = []

        for point in traffic_query_points():
            params = {
                "point": f"{point.lat:.6f},{point.lon:.6f}",
                "unit": "KMPH",
                "key": settings.tomtom_api_key,
            }
            try:
                response = await client.get(API_BASE, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning(
                    "TomTom fetch failed",
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
            raise RuntimeError("TomTom returned no observations")

        return {"observations": observations}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        points: list[DataPointCreate] = []
        for observation in raw_data.get("observations", []):
            points.extend(self._normalize_observation(observation))
        return points

    def _normalize_observation(
        self,
        observation: dict[str, Any],
    ) -> list[DataPointCreate]:
        payload = observation.get("payload") or {}
        flow = payload.get("flowSegmentData") or {}
        if not flow:
            return []

        point_id = observation.get("point_id")
        if not point_id:
            return []

        lat = safe_float(observation.get("requested_lat"))
        lon = safe_float(observation.get("requested_lon"))
        if lat is None or lon is None:
            return []

        timestamp = datetime.now(tz=timezone.utc).replace(microsecond=0)

        current_speed = safe_float(flow.get("currentSpeed"))
        free_flow_speed = safe_float(flow.get("freeFlowSpeed"))
        confidence = safe_float(flow.get("confidence"))
        current_travel_time = safe_float(flow.get("currentTravelTime"))
        free_flow_travel_time = safe_float(flow.get("freeFlowTravelTime"))

        readings: list[tuple[str, float | None, str]] = [
            ("traffic_current_speed", current_speed, "km/h"),
            ("traffic_free_flow_speed", free_flow_speed, "km/h"),
            ("traffic_current_travel_time", current_travel_time, "s"),
            ("traffic_free_flow_travel_time", free_flow_travel_time, "s"),
            ("traffic_confidence", confidence, "ratio"),
        ]

        if current_speed is not None and free_flow_speed and free_flow_speed > 0:
            readings.append(
                ("traffic_speed_ratio", current_speed / free_flow_speed, "ratio")
            )

        points: list[DataPointCreate] = []
        for metric, value, unit in readings:
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
                    source_entity_id=f"grid:{point_id}",
                    raw_json=observation,
                )
            )

        return points
