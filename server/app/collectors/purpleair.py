"""PurpleAir low-cost optical PM2.5 — instrument-independent from regulatory PM.

PurpleAir sensors use optical Plantower units; pairing them against the
regulatory BAM/gravimetric PM2.5 already in OpenAQ gives a genuine
different-instrument-physics channel for the type-1 PM2.5 sub-claim (the
Barkjohn correction characterizes the link, so it's *partial* independence).

Paid, points-metered API ($1 / 100k points; sensor owners free for their own
sensors). The live ``/v1/sensors`` bbox call is cheap (one per hour); the
``/v1/sensors/{id}/history`` backfill is point-expensive, so it is hard-capped
against the free ``/v1/organization`` balance.

Verified live 2026-06-24:
* GET ``/v1/sensors`` with bbox params ``nwlng/nwlat/selng/selat`` + ``fields``
  + ``location_type`` + ``max_age``; ``X-API-Key`` header.
* Response is columnar: ``{"fields": [...], "data": [[...], ...]}``; rows map to
  fields by NAME (the API may reorder), ``last_seen`` is epoch-seconds UTC.
* History: ``{"fields": ["time_stamp", ...], "data": [[...]]}``, ~98 pts /
  sensor-day at hourly averages.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import target_bounding_box, within_target_radius
from app.collectors.ratelimit import AsyncRateLimiter, rate_limited_get
from app.config import settings

logger = logging.getLogger(__name__)

SOURCE_NAME = "purpleair"
API_BASE = "https://api.purpleair.com/v1"
SENSORS_ENDPOINT = f"{API_BASE}/sensors"
ORGANIZATION_ENDPOINT = f"{API_BASE}/organization"

# Live PM2.5 field on /v1/sensors; the historical analogue used by the backfill.
LIVE_PM_FIELD = "pm2.5"
HISTORY_PM_FIELD = "pm2.5_atm"
LIVE_FIELDS = "pm2.5,latitude,longitude,last_seen,name"

# Only return sensors that reported within the last hour (server-side), so a
# long-offline unit doesn't re-emit a stale value every run.
LIVE_MAX_AGE_S = 3600
# Belt-and-suspenders client-side staleness guard.
MAX_READING_AGE_S = 6 * 3600

# PurpleAir asks callers to poll no faster than once every 1-10 minutes; hourly
# collection is well within that. Shared by collector + backfill.
PURPLEAIR_MAX_REQUESTS_PER_MINUTE = 6
PURPLEAIR_LIMITER = AsyncRateLimiter(PURPLEAIR_MAX_REQUESTS_PER_MINUTE)


def _api_headers() -> dict[str, str]:
    return {"X-API-Key": settings.purpleair_api_key}


def parse_purpleair_rows(
    fields: list[str], data: list[list[Any]]
) -> list[dict[str, Any]]:
    """Zip columnar ``data`` rows against ``fields`` by name."""
    index = {name: i for i, name in enumerate(fields)}
    rows: list[dict[str, Any]] = []
    for row in data:
        if len(row) != len(fields):
            continue
        rows.append({name: row[i] for name, i in index.items()})
    return rows


def purpleair_sensors_to_points(
    fields: list[str],
    data: list[list[Any]],
    *,
    source_name: str,
    now: datetime,
    pm_field: str = LIVE_PM_FIELD,
) -> list[DataPointCreate]:
    """Latest-value sensor rows -> DataPointCreate, dropping stale/bad rows."""
    points: list[DataPointCreate] = []
    for row in parse_purpleair_rows(fields, data):
        sensor_index = row.get("sensor_index")
        last_seen = row.get("last_seen")
        pm = row.get(pm_field)
        lat = row.get("latitude")
        lon = row.get("longitude")
        if None in (sensor_index, last_seen, pm, lat, lon):
            continue
        try:
            lat = float(lat)
            lon = float(lon)
            value = float(pm)
            seen = int(last_seen)
        except (TypeError, ValueError):
            continue
        if not within_target_radius(lat, lon):
            continue
        timestamp = datetime.fromtimestamp(seen, tz=timezone.utc)
        if (now - timestamp).total_seconds() > MAX_READING_AGE_S:
            continue
        points.append(
            DataPointCreate(
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                metric="pm25",
                value=value,
                unit="ug/m3",
                source=source_name,
                source_entity_id=str(int(sensor_index)),
                raw_json=row,
            )
        )
    return points


def purpleair_history_to_points(
    fields: list[str],
    data: list[list[Any]],
    *,
    sensor_index: int,
    lat: float,
    lon: float,
    source_name: str,
    pm_field: str = HISTORY_PM_FIELD,
) -> list[DataPointCreate]:
    """Historical rows for one sensor -> DataPointCreate (no staleness guard)."""
    index = {name: i for i, name in enumerate(fields)}
    if "time_stamp" not in index or pm_field not in index:
        return []
    ts_i, pm_i = index["time_stamp"], index[pm_field]

    points: list[DataPointCreate] = []
    for row in data:
        if len(row) <= max(ts_i, pm_i):
            continue
        raw_ts, raw_pm = row[ts_i], row[pm_i]
        if raw_ts is None or raw_pm is None:
            continue
        try:
            timestamp = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
            value = float(raw_pm)
        except (TypeError, ValueError):
            continue
        points.append(
            DataPointCreate(
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                metric="pm25",
                value=value,
                unit="ug/m3",
                source=source_name,
                source_entity_id=str(sensor_index),
                raw_json={"sensor_index": sensor_index, "time_stamp": raw_ts, pm_field: value},
            )
        )
    return points


def sensors_request_params(*, fields: str, max_age: int | None = None) -> dict[str, Any]:
    """Bbox query params for the /v1/sensors endpoint over the target area."""
    bbox = target_bounding_box()
    params: dict[str, Any] = {
        "fields": fields,
        "location_type": 0,  # outdoor sensors only
        "nwlng": bbox.min_lon,
        "nwlat": bbox.max_lat,
        "selng": bbox.max_lon,
        "selat": bbox.min_lat,
    }
    if max_age is not None:
        params["max_age"] = max_age
    return params


class PurpleAirCollector(BaseCollector):
    source_name = SOURCE_NAME
    collect_interval_minutes = 60

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        super().__init__(http_client)
        self._limiter = rate_limiter or PURPLEAIR_LIMITER

    async def fetch(self) -> dict[str, Any]:
        if not settings.purpleair_api_key:
            raise RuntimeError("PURPLEAIR_API_KEY is not set")
        client = await self._get_client()
        response = await rate_limited_get(
            client,
            SENSORS_ENDPOINT,
            limiter=self._limiter,
            params=sensors_request_params(fields=LIVE_FIELDS, max_age=LIVE_MAX_AGE_S),
            headers=_api_headers(),
        )
        payload = response.json()
        return {"fields": payload.get("fields", []), "data": payload.get("data", [])}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        return purpleair_sensors_to_points(
            raw_data.get("fields", []),
            raw_data.get("data", []),
            source_name=self.source_name,
            now=datetime.now(timezone.utc),
        )
