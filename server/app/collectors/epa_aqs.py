"""EPA Air Quality System (AQS) — certified historical ground chemistry.

AQS is the federal regulatory archive. It is the certified counterpart to the
preliminary TCEQ feed: same monitors, but quality-assured — and lagging "6
months or more" behind real time, so it can never feed the live freeze window.
Its role is the *demonstrable* error-independence analysis on quiet windows:
the satellite column (Sentinel-5P) vs. in-situ ground (AQS) pair for the exact
petrochemical species (NO2/SO2/CO) the Houston framing targets.

Backfill-only — there is no live collector (a 6-month-delayed source has no
"current" reading to poll). This module holds the pure request/parse helpers;
``EPAAQSBackfill`` in backfill.py does the network + persistence.

Verified API contract (https://aqs.epa.gov/aqsweb/documents/data_api.html and
EPA's official pyaqsapi client):
* base ``https://aqs.epa.gov/data/api``; auth via ``email`` + ``key`` query params
* ``sampleData/byBox`` params: param (<=5 codes), bdate/edate (YYYYMMDD, same
  calendar year), minlat/maxlat/minlon/maxlon
* response ``{"Header": [...], "Data": [...]}``; records carry
  ``sample_measurement``/``units_of_measure``/``parameter_code``/``date_gmt``/
  ``time_gmt``/``latitude``/``longitude``/``state_code``/``county_code``/
  ``site_number``/``poc``
* usage: <=10 requests/min, 5s between requests, serial
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.collectors.base import DataPointCreate
from app.collectors.geo import within_target_radius
from app.collectors.ratelimit import AsyncRateLimiter

logger = logging.getLogger(__name__)

SOURCE_NAME = "epa_aqs"

API_BASE = "https://aqs.epa.gov/data/api"
AQS_SAMPLE_ENDPOINT = f"{API_BASE}/sampleData/byBox"

# The petrochemical species the satellite-vs-ground independence pair targets.
AQS_PARAM_TO_METRIC: dict[str, str] = {
    "42602": "no2",
    "42401": "so2",
    "42101": "co",
}

# AQS allows <=5 param codes per request, so all three go in one call.
DEFAULT_AQS_PARAM_CODES: tuple[str, ...] = tuple(AQS_PARAM_TO_METRIC)

# Usage policy: <=10 requests/min, 5s between requests. 10/min = 6s spacing,
# serial. Shared as the module limiter so any AQS caller spends one budget.
AQS_MAX_REQUESTS_PER_MINUTE = 10
AQS_LIMITER = AsyncRateLimiter(AQS_MAX_REQUESTS_PER_MINUTE)

_UNIT_MAP: dict[str, str] = {
    "parts per billion": "ppb",
    "parts per million": "ppm",
}


def normalize_aqs_unit(unit: str | None) -> str:
    """Map AQS ``unit_of_measure`` strings to AERIS canonical units."""
    if not unit:
        return "unknown"
    cleaned = unit.strip().lower()
    if cleaned in _UNIT_MAP:
        return _UNIT_MAP[cleaned]
    if cleaned.startswith("micrograms/cubic meter"):
        return "ug/m3"
    return cleaned


def parse_aqs_datetime(date_gmt: str | None, time_gmt: str | None) -> datetime | None:
    """Combine AQS ``date_gmt`` + ``time_gmt`` into a UTC datetime."""
    if not date_gmt or not time_gmt:
        return None
    try:
        return datetime.strptime(
            f"{date_gmt.strip()} {time_gmt.strip()}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def aqs_year_chunks(since: datetime, until: datetime) -> list[tuple[str, str]]:
    """Split ``[since, until]`` into per-calendar-year ``(bdate, edate)`` strings.

    AQS requires ``edate`` to be in the same calendar year as ``bdate``.
    """
    if until < since:
        return []
    chunks: list[tuple[str, str]] = []
    for year in range(since.year, until.year + 1):
        year_start = datetime(year, 1, 1, tzinfo=timezone.utc)
        year_end = datetime(year, 12, 31, tzinfo=timezone.utc)
        bdate = max(since, year_start)
        edate = min(until, year_end)
        chunks.append((bdate.strftime("%Y%m%d"), edate.strftime("%Y%m%d")))
    return chunks


def aqs_records_to_points(
    records: list[dict[str, Any]], *, source_name: str
) -> list[DataPointCreate]:
    """Transform AQS ``Data`` records into DataPointCreate; drop bad rows."""
    points: list[DataPointCreate] = []
    for record in records:
        metric = AQS_PARAM_TO_METRIC.get(str(record.get("parameter_code")))
        if metric is None:
            continue
        measurement = record.get("sample_measurement")
        if measurement is None:
            continue
        try:
            value = float(measurement)
        except (TypeError, ValueError):
            continue
        timestamp = parse_aqs_datetime(record.get("date_gmt"), record.get("time_gmt"))
        if timestamp is None:
            continue
        try:
            lat = float(record["latitude"])
            lon = float(record["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        if not within_target_radius(lat, lon):
            continue

        poc = record.get("poc")
        poc_str = str(int(poc)) if isinstance(poc, (int, float)) else str(poc)
        entity = "-".join(
            [
                str(record.get("state_code", "")),
                str(record.get("county_code", "")),
                str(record.get("site_number", "")),
                poc_str,
            ]
        )

        points.append(
            DataPointCreate(
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                metric=metric,
                value=value,
                unit=normalize_aqs_unit(record.get("units_of_measure")),
                source=source_name,
                source_entity_id=entity,
                raw_json={"record": record},
            )
        )
    return points
