"""ASOS/METAR surface observations via the Iowa Environmental Mesonet (IEM).

IEM redistributes the NOAA ASOS/AWOS/METAR feed as a free, key-less CSV
service. These are direct anemometer/thermometer observations, a measurement
process distinct from NWP-derived fields. They are not statistically
independent: GFS analyses assimilate METAR observations and OpenWeather blends
multiple weather inputs. Any quantitative independence claim therefore needs
an explicit residual-error analysis.

Two endpoints are used, both rate-limited through the shared ASOS limiter:

* station metadata — ``/geojson/network/TX_ASOS.geojson`` gives every Texas
  ASOS station's id + coordinates; we keep the ones inside the target radius.
* observations — ``/cgi-bin/request/asos.py`` returns comma-format CSV for a
  set of stations over a time window.

IEM throttles to 1 request/sec/IP ("literally no reason to hit this service
at a higher rate"); the limiter is set to 30/min (2s spacing) to stay well
under that and leave headroom for the backfill, which shares the same limiter.

The feed mixes sub-hourly wind obs (temperature reported as ``M``) with the
routine hourly METAR (carries wind + temp + humidity). normalize() downsamples
to one observation per (station, metric, clock-hour) — the latest valid reading
in each hour — so the cadence matches the existing hourly met sources and the
row count stays bounded. Metric names/units mirror OpenWeather
(``wind_direction``/degree, ``wind_speed``/m/s, ``temperature``/degC,
``humidity``/percent) so the channels are directly poolable.
"""

from __future__ import annotations

import csv
import io
import logging
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from app.collectors.base import BaseCollector, DataPointCreate
from app.collectors.geo import within_target_radius
from app.collectors.ratelimit import AsyncRateLimiter, rate_limited_get

logger = logging.getLogger(__name__)

API_BASE = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
NETWORK_GEOJSON_TMPL = (
    "https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
)
ASOS_NETWORK = "TX_ASOS"

# CSV fields requested from the service (order is irrelevant; parsed by name).
ASOS_DATA_FIELDS = ("drct", "sknt", "tmpf", "relh")

# Live collection looks back this many hours so a routine METAR (issued near
# :53) is always captured even if a scheduled run drifts off the hour.
LIVE_LOOKBACK_HOURS = 3

# IEM throttles to 1 req/s/IP. 30/min = 2s spacing keeps us comfortably under
# and is shared with the backfill so both spend from one budget.
ASOS_MAX_REQUESTS_PER_MINUTE = 30
ASOS_LIMITER = AsyncRateLimiter(ASOS_MAX_REQUESTS_PER_MINUTE)

# Station topology barely changes; refetch daily, not every run.
STATIONS_CACHE_TTL_S = 24 * 3600.0
_stations_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}

KNOTS_TO_MS = 0.514444

# Tokens the service uses for absent/trace values.
_MISSING_TOKENS = {"M", "", "T", "NA", "NULL", "NONE"}


def clear_stations_cache() -> None:
    _stations_cache.clear()


def fahrenheit_to_celsius(value: float) -> float:
    return (value - 32.0) * 5.0 / 9.0


# csv field -> (metric, unit, converter); names/units mirror openweather.py.
FIELD_MAP: dict[str, tuple[str, str, Callable[[float], float]]] = {
    "drct": ("wind_direction", "degree", lambda v: v),
    "sknt": ("wind_speed", "m/s", lambda v: v * KNOTS_TO_MS),
    "tmpf": ("temperature", "degC", fahrenheit_to_celsius),
    "relh": ("humidity", "percent", lambda v: v),
}


def parse_asos_datetime(value: str | None) -> datetime | None:
    """Parse the IEM ``valid`` column (UTC, ``YYYY-MM-DD HH:MM``)."""
    if not value:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def parse_asos_csv(text: str) -> list[dict[str, str]]:
    """Parse onlycomma ASOS output into row dicts; tolerant of empty input."""
    if not text or not text.strip():
        return []
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames or "station" not in reader.fieldnames:
        return []
    return [row for row in reader if row.get("station")]


def _missing(token: str | None) -> bool:
    return token is None or token.strip().upper() in _MISSING_TOKENS


def asos_rows_to_points(
    rows: list[dict[str, str]], *, source_name: str
) -> list[DataPointCreate]:
    """Downsample rows to one valid observation per (station, metric, hour).

    The latest valid reading in each clock-hour wins, so the routine hourly
    METAR (which carries every field) represents the hour. Each point keeps its
    true observation timestamp. Stations outside the target radius and rows with
    unparseable timestamps/coordinates/values are dropped, never raised.
    """
    # key -> (best_valid_dt, DataPointCreate)
    best: dict[tuple[str, str, datetime], tuple[datetime, DataPointCreate]] = {}

    for row in rows:
        station = (row.get("station") or "").strip()
        if not station:
            continue
        valid = parse_asos_datetime(row.get("valid"))
        if valid is None:
            continue
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not within_target_radius(lat, lon):
            continue

        hour = valid.replace(minute=0, second=0, microsecond=0)
        for field, (metric, unit, convert) in FIELD_MAP.items():
            token = row.get(field)
            if _missing(token):
                continue
            try:
                value = convert(float(token))
            except (TypeError, ValueError):
                continue

            key = (station, metric, hour)
            existing = best.get(key)
            if existing is not None and existing[0] >= valid:
                continue
            best[key] = (
                valid,
                DataPointCreate(
                    timestamp=valid,
                    lat=lat,
                    lon=lon,
                    metric=metric,
                    value=value,
                    unit=unit,
                    source=source_name,
                    source_entity_id=station,
                    raw_json={"row": row},
                ),
            )

    return [point for _, point in best.values()]


async def fetch_target_stations(
    client: httpx.AsyncClient,
    limiter: AsyncRateLimiter,
    *,
    network: str = ASOS_NETWORK,
) -> list[dict[str, Any]]:
    """Online stations of ``network`` inside the target radius (cached daily)."""
    cached = _stations_cache.get(network)
    if cached is not None and time.monotonic() - cached[0] < STATIONS_CACHE_TTL_S:
        return cached[1]

    response = await rate_limited_get(
        client,
        NETWORK_GEOJSON_TMPL.format(network=network),
        limiter=limiter,
    )
    payload = response.json()

    stations: list[dict[str, Any]] = []
    for feature in payload.get("features", []):
        props = feature.get("properties") or {}
        if props.get("online") is False:
            continue
        station_id = feature.get("id") or props.get("sid")
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        if not station_id or len(coords) < 2:
            continue
        try:
            lon = float(coords[0])
            lat = float(coords[1])
        except (TypeError, ValueError):
            continue
        if not within_target_radius(lat, lon):
            continue
        stations.append(
            {"id": str(station_id), "lat": lat, "lon": lon, "name": props.get("sname")}
        )

    _stations_cache[network] = (time.monotonic(), stations)
    return stations


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")


def asos_request_params(
    station_ids: list[str], *, since: datetime, until: datetime
) -> list[tuple[str, str]]:
    """Build the repeated-key query params for the asos.py CSV endpoint."""
    params: list[tuple[str, str]] = [("station", sid) for sid in station_ids]
    params += [("data", field) for field in ASOS_DATA_FIELDS]
    params += [
        ("sts", _iso_z(since)),
        ("ets", _iso_z(until)),
        ("tz", "UTC"),
        ("format", "onlycomma"),
        ("latlon", "yes"),
        ("missing", "M"),
        ("trace", "T"),
    ]
    return params


class ASOSCollector(BaseCollector):
    source_name = "asos"
    collect_interval_minutes = 60

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        super().__init__(http_client)
        self._limiter = rate_limiter or ASOS_LIMITER

    async def fetch(self) -> dict[str, Any]:
        """Fetch the last few hours of obs for in-radius Houston ASOS stations."""
        client = await self._get_client()
        stations = await fetch_target_stations(client, self._limiter)
        if not stations:
            logger.warning("ASOS: no online stations inside the target radius")
            return {"csv": ""}

        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=LIVE_LOOKBACK_HOURS)
        params = asos_request_params(
            [s["id"] for s in stations], since=since, until=until
        )
        response = await rate_limited_get(
            client, API_BASE, limiter=self._limiter, params=params
        )
        return {"csv": response.text, "station_count": len(stations)}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        rows = parse_asos_csv(raw_data.get("csv", ""))
        return asos_rows_to_points(rows, source_name=self.source_name)
