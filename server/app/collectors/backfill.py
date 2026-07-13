"""Historical backfill for the four AERIS data sources.

Each source has its own strategy because the available historical surface
differs sharply:

* **OpenAQ** — ``/v3/sensors/{id}/measurements`` with ``datetime_from`` /
  ``datetime_to``, paginated. The high-value backfill: ~720 points per
  sensor over 30 days, enough to anchor every detector.
* **Sentinel-5P** — backfill walks 48h catalog windows backward through the
  date range. With CDSE credentials set it also downloads granules for the
  mapped column products and extracts column densities, exactly like the
  scheduled collector; granules whose columns already exist in the DB are
  not re-downloaded, so an interrupted backfill resumes on re-run. Without
  credentials it degrades to catalog-only (availability + cloud cover) and
  says so loudly. Granules run hundreds of MB each — budget bandwidth.
* **NOAA GFS** — NOMADS retains roughly the last 10 days of cycles. The
  backfill walks past 6-hour cycles backward through the requested range
  and reuses the collector's ``_load_cycle`` GRIB-filter fetch + parse.
* **OpenWeather** — the free tier has no historical endpoint. This
  strategy is a documented no-op; the only path to OpenWeather history is
  scheduled snapshot collection going forward.

CLI::

    python -m app.collectors.backfill                          # all sources, 30 days
    python -m app.collectors.backfill --source openaq          # one source
    python -m app.collectors.backfill --days 7
    python -m app.collectors.backfill --since 2026-04-01 --until 2026-05-01
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import io
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Sequence

import httpx
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.asos import (
    API_BASE as ASOS_API_BASE,
    ASOS_LIMITER,
    asos_request_params,
    asos_rows_to_points,
    fetch_target_stations,
    parse_asos_csv,
)
from app.collectors.base import DataPointCreate
from app.collectors.epa_aqs import (
    AQS_LIMITER,
    AQS_REQUEST_TIMEOUT_S,
    AQS_SAMPLE_ENDPOINT,
    DEFAULT_AQS_PARAM_CODES,
    SOURCE_NAME as EPA_AQS_SOURCE_NAME,
    aqs_month_chunks,
    aqs_records_to_points,
)
from app.collectors.geo import target_bounding_box, within_target_radius
from app.collectors.purpleair import (
    API_BASE as PURPLEAIR_API_BASE,
    HISTORY_FIELDS,
    ORGANIZATION_ENDPOINT,
    PURPLEAIR_LIMITER,
    SENSORS_ENDPOINT as PURPLEAIR_SENSORS_ENDPOINT,
    parse_purpleair_rows,
    purpleair_history_to_points,
    sensors_request_params,
)
from app.collectors.openaq import (
    OPENAQ_LIMITER,
    PARAMETER_MAP,
    location_within_target_radius,
    normalize_openaq_unit,
    parse_openaq_datetime,
)
from app.collectors.ratelimit import (
    AsyncRateLimiter,
    rate_limited_get,
    rate_limited_post,
)
from app.collectors.tceq import (
    API_URL as TCEQ_API_URL,
    CST as TCEQ_CST,
    DEFAULT_BREAKER_THRESHOLD as TCEQ_BREAKER_THRESHOLD,
    TCEQ_LIMITER,
    TCEQ_SITES,
    USER_AGENT as TCEQ_USER_AGENT,
    tceq_html_to_points,
    user_date_post_data,
)
from app.collectors.sentinel5p import (
    COLUMN_PRODUCTS,
    Sentinel5PCollector,
    extract_product_code,
    fetch_access_token,
    ids_with_stored_columns,
    odata_filter,
)
from app.config import settings
from app.db.models import DataPoint


logger = logging.getLogger(__name__)


# Per-source defaults --------------------------------------------------

OPENAQ_API_BASE = "https://api.openaq.org/v3"
OPENAQ_LOCATIONS_LIMIT = 1000
OPENAQ_DEFAULT_PAGE_SIZE = 1000
OPENAQ_SENSOR_DELAY_S = 0.1  # gentle rate-limiting between sensors
# Hard ceiling on measurement pages per sensor: at the default page size this
# is ~1M points, far beyond any real sensor's history, so it only ever trips
# on a misbehaving endpoint that ignores `page` and returns a full page
# forever (which would otherwise spin the loop indefinitely).
OPENAQ_MAX_PAGES = 1000

SENTINEL_WINDOW_HOURS = 48

GFS_CYCLE_HOURS = (0, 6, 12, 18)
GFS_RETENTION_DAYS = 10  # NOMADS keeps ~10 days; older requests will 404


@dataclass
class BackfillResult:
    """Per-source backfill outcome."""

    source: str
    records: int = 0
    skipped: bool = False
    error: str | None = None
    notes: str | None = None
    duration_ms: float = 0.0


class BackfillStrategy(ABC):
    """Per-source historical loader interface."""

    source_name: str

    @abstractmethod
    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        """Fetch and persist historical DataPoints in ``[since, until]``."""


# Shared persistence helper -------------------------------------------


# Conservative bound-parameter caps per dialect. SQLite's portable cap is 999
# (SQLITE_MAX_VARIABLE_NUMBER on builds before 3.32 — newer builds raise it,
# but we target the floor so the chunk size is safe wherever CI/deploy runs);
# Postgres' wire protocol caps a statement at 65535 bound parameters.
_SQLITE_PARAM_CAP = 999
_POSTGRES_PARAM_CAP = 65535


def _insert_chunk_size(dialect: str, n_cols: int) -> int:
    """Max rows per multi-row INSERT so ``rows * n_cols`` stays under the cap."""
    cap = _POSTGRES_PARAM_CAP if dialect == "postgresql" else _SQLITE_PARAM_CAP
    return max(1, cap // max(1, n_cols))


async def _store_points(
    session: AsyncSession,
    points: Sequence[DataPointCreate],
) -> int:
    """Upsert via the data_points dedup index; mirrors BaseCollector._store.

    Inserts are chunked so a large backfill never exceeds the dialect's
    bound-parameter cap (a single ``VALUES`` over the whole list blows SQLite's
    999-variable limit at ~111 rows). All chunks share one transaction, so the
    store stays all-or-nothing as before.
    """
    if not points:
        return 0
    rows = [p.model_dump() for p in points]
    bind = session.get_bind()
    dialect = bind.dialect.name if bind is not None else ""
    dedup_cols = ["source", "metric", "source_entity_id", "timestamp"]
    chunk_size = _insert_chunk_size(dialect, len(rows[0]))

    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        if dialect == "postgresql":
            stmt = pg_insert(DataPoint).values(chunk).on_conflict_do_nothing(
                index_elements=dedup_cols
            )
        elif dialect == "sqlite":
            stmt = sqlite_insert(DataPoint).values(chunk).on_conflict_do_nothing(
                index_elements=dedup_cols
            )
        else:
            raise NotImplementedError(
                f"_store_points has no dedup strategy for dialect {dialect!r}"
            )
        await session.execute(stmt)

    await session.commit()
    return len(rows)


# OpenAQ ---------------------------------------------------------------


class OpenAQBackfill(BackfillStrategy):
    """Paginated historical fetch of OpenAQ sensor measurements."""

    source_name = "openaq"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        page_size: int = OPENAQ_DEFAULT_PAGE_SIZE,
        sensor_delay_s: float = OPENAQ_SENSOR_DELAY_S,
        max_pages: int = OPENAQ_MAX_PAGES,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self.page_size = page_size
        self.sensor_delay_s = sensor_delay_s
        self.max_pages = max_pages
        self._limiter = rate_limiter or OPENAQ_LIMITER

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        total_inserted = 0
        try:
            client = await self._client_or_default()
            headers = (
                {"X-API-Key": settings.openaq_api_key}
                if settings.openaq_api_key
                else {}
            )

            locations = await self._list_locations(client, headers)
            logger.info(
                "OpenAQ backfill: %d locations in target radius", len(locations)
            )

            for location in locations:
                location_id = location.get("id")
                if location_id is None:
                    continue

                sensors = await self._list_sensors(client, headers, location_id)
                for sensor in sensors:
                    parameter = (sensor.get("parameter") or {}).get("name", "").lower()
                    if PARAMETER_MAP.get(parameter) is None:
                        continue

                    points = await self._collect_sensor(
                        client, headers, sensor, location, since, until
                    )
                    inserted = await _store_points(session, points)
                    total_inserted += inserted

                    if self.sensor_delay_s > 0:
                        await asyncio.sleep(self.sensor_delay_s)
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.close()

        return BackfillResult(
            source=self.source_name,
            records=total_inserted,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    async def _list_locations(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        response = await rate_limited_get(
            client,
            f"{OPENAQ_API_BASE}/locations",
            limiter=self._limiter,
            params={
                "bbox": target_bounding_box().as_csv(),
                "limit": OPENAQ_LOCATIONS_LIMIT,
            },
            headers=headers,
        )
        results = response.json().get("results", []) or []
        return [loc for loc in results if location_within_target_radius(loc)]

    async def _list_sensors(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        location_id: Any,
    ) -> list[dict[str, Any]]:
        try:
            response = await rate_limited_get(
                client,
                f"{OPENAQ_API_BASE}/locations/{location_id}/sensors",
                limiter=self._limiter,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "OpenAQ sensor list failed",
                extra={"location_id": location_id, "error": str(exc)},
            )
            return []
        return response.json().get("results", []) or []

    async def _collect_sensor(
        self,
        client: httpx.AsyncClient,
        headers: dict[str, str],
        sensor: dict[str, Any],
        location: dict[str, Any],
        since: datetime,
        until: datetime,
    ) -> list[DataPointCreate]:
        sensor_id = sensor.get("id")
        if sensor_id is None:
            return []

        parameter_obj = sensor.get("parameter") or {}
        parameter_name = str(parameter_obj.get("name", "")).lower()
        metric = PARAMETER_MAP.get(parameter_name)
        if metric is None:
            return []
        unit = normalize_openaq_unit(parameter_obj.get("units"))
        location_coords = location.get("coordinates") or {}
        loc_lat = location_coords.get("latitude")
        loc_lon = location_coords.get("longitude")

        points: list[DataPointCreate] = []
        page = 1
        while True:
            if page > self.max_pages:
                logger.warning(
                    "OpenAQ measurements hit max_pages guard; stopping pagination",
                    extra={
                        "sensor_id": sensor_id,
                        "max_pages": self.max_pages,
                    },
                )
                break
            try:
                response = await rate_limited_get(
                    client,
                    f"{OPENAQ_API_BASE}/sensors/{sensor_id}/measurements",
                    limiter=self._limiter,
                    headers=headers,
                    params={
                        "datetime_from": _iso_z(since),
                        "datetime_to": _iso_z(until),
                        "limit": self.page_size,
                        "page": page,
                    },
                )
            except httpx.HTTPError as exc:
                logger.warning(
                    "OpenAQ measurements page failed",
                    extra={
                        "sensor_id": sensor_id,
                        "page": page,
                        "error": str(exc),
                    },
                )
                break

            results = response.json().get("results", []) or []
            if not results:
                break

            for record in results:
                ts = _measurement_timestamp(record)
                if ts is None:
                    continue
                try:
                    value = float(record.get("value"))
                except (TypeError, ValueError):
                    continue
                coords = record.get("coordinates") or {}
                lat = coords.get("latitude", loc_lat)
                lon = coords.get("longitude", loc_lon)
                if lat is None or lon is None:
                    continue
                points.append(
                    DataPointCreate(
                        timestamp=ts,
                        lat=float(lat),
                        lon=float(lon),
                        metric=metric,
                        value=value,
                        unit=unit,
                        source=self.source_name,
                        source_entity_id=str(sensor_id),
                        raw_json={
                            "location": location,
                            "sensor": sensor,
                            "measurement": record,
                        },
                    )
                )

            if len(results) < self.page_size:
                break
            page += 1

        return points


# OpenAQ S3 archive ----------------------------------------------------

# Public open-data bucket; keyless and intended for exactly this bulk use.
# One gzipped CSV per (location, day); objects land ~3-4 days after the
# measurement day, so recent days legitimately 404.
OPENAQ_ARCHIVE_BASE = "https://openaq-data-archive.s3.amazonaws.com"
OPENAQ_ARCHIVE_DELAY_S = 0.05
# Max archive objects fetched at once. The N(locations) x D(days) GETs are the
# bottleneck; fetching a bounded batch concurrently (then storing serially, as
# one AsyncSession can't be shared across concurrent writes) collapses the wall
# time without unbounding either connections or in-memory payloads.
OPENAQ_ARCHIVE_CONCURRENCY = 8


def archive_key(location_id: int, day: date) -> str:
    return (
        f"records/csv.gz/locationid={location_id}/year={day.year}/"
        f"month={day.month:02d}/location-{location_id}-{day:%Y%m%d}.csv.gz"
    )


def parse_archive_csv(payload: bytes) -> list[DataPointCreate]:
    """Map archive CSV rows to normalized DataPoints.

    Columns: location_id, sensors_id, location, datetime (ISO with offset),
    lat, lon, parameter, units, value. sensors_id shares the API collector's
    entity-id space, so the dedup index treats both paths as one source.
    """
    text = gzip.decompress(payload).decode("utf-8")
    points: list[DataPointCreate] = []
    for row in csv.DictReader(io.StringIO(text)):
        metric = PARAMETER_MAP.get(str(row.get("parameter", "")).lower())
        if metric is None:
            continue
        timestamp = parse_openaq_datetime(row.get("datetime"))
        if timestamp is None:
            continue
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
            value = float(row["value"])
        except (KeyError, TypeError, ValueError):
            continue
        if not within_target_radius(lat, lon):
            continue
        sensor_id = row.get("sensors_id")
        if not sensor_id:
            continue
        points.append(
            DataPointCreate(
                timestamp=timestamp,
                lat=lat,
                lon=lon,
                metric=metric,
                value=value,
                unit=normalize_openaq_unit(row.get("units")),
                source="openaq",
                source_entity_id=str(sensor_id),
                raw_json={"archive": dict(row)},
            )
        )
    return points


async def db_location_ids(session: AsyncSession) -> list[int]:
    """Distinct OpenAQ location ids from already-collected raw_json."""
    latest_per_entity = (
        select(func.max(DataPoint.id))
        .where(DataPoint.source == "openaq")
        .group_by(DataPoint.source_entity_id)
    )
    rows = await session.execute(
        select(DataPoint.raw_json).where(DataPoint.id.in_(latest_per_entity))
    )
    ids: set[int] = set()
    for raw in rows.scalars():
        location_id = ((raw or {}).get("location") or {}).get("id")
        if location_id is not None:
            ids.add(int(location_id))
    return sorted(ids)


class OpenAQArchiveBackfill(BackfillStrategy):
    """Bulk history from the OpenAQ S3 data archive instead of the API."""

    source_name = "openaq"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        location_ids: Sequence[int] | None = None,
        request_delay_s: float = OPENAQ_ARCHIVE_DELAY_S,
        max_concurrency: int = OPENAQ_ARCHIVE_CONCURRENCY,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self.location_ids = list(location_ids) if location_ids else None
        self.request_delay_s = request_delay_s
        self.max_concurrency = max(1, max_concurrency)

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        total_inserted = 0
        missing_days = 0
        try:
            location_ids = self.location_ids or await db_location_ids(session)
            if not location_ids:
                return BackfillResult(
                    source=self.source_name,
                    error=(
                        "no OpenAQ location ids known; collect once via the API "
                        "or pass --location-ids"
                    ),
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            days = _days_between(since, until)
            client = await self._client_or_default()
            urls = [
                f"{OPENAQ_ARCHIVE_BASE}/{archive_key(location_id, day)}"
                for location_id in location_ids
                for day in days
            ]

            # Fetch a bounded batch of objects concurrently, then parse + store
            # that batch serially: the GETs are the slow part and parallelize
            # safely, but the shared AsyncSession can't take concurrent writes,
            # and batching also caps how many object bodies are held at once.
            for batch_start in range(0, len(urls), self.max_concurrency):
                batch = urls[batch_start : batch_start + self.max_concurrency]
                fetched = await asyncio.gather(
                    *(self._fetch_archive_object(client, url) for url in batch)
                )
                for status, content in fetched:
                    if status == "missing":
                        missing_days += 1
                    elif status == "ok" and content is not None:
                        points = parse_archive_csv(content)
                        total_inserted += await _store_points(session, points)
                    # "error": already logged in the fetch, skip without counting.

            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                notes=(
                    f"archive: {len(location_ids)} locations x {len(days)} days, "
                    f"{missing_days} objects absent"
                ),
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.close()

    async def _fetch_archive_object(
        self, client: httpx.AsyncClient, url: str
    ) -> tuple[str, bytes | None]:
        """GET one archive object, tagging the outcome for the serial store.

        ``("ok", bytes)`` on a 2xx body, ``("missing", None)`` on a 404 (a day
        whose object hasn't landed yet), ``("error", None)`` on any transport
        error (logged here, skipped without counting as missing).
        """
        try:
            response = await client.get(url)
            if response.status_code == 404:
                return ("missing", None)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "OpenAQ archive object failed",
                extra={"url": url, "error": str(exc)},
            )
            return ("error", None)
        if self.request_delay_s > 0:
            await asyncio.sleep(self.request_delay_s)
        return ("ok", response.content)


def _days_between(since: datetime, until: datetime) -> list[date]:
    first = since.astimezone(timezone.utc).date()
    last = until.astimezone(timezone.utc).date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


# Sentinel-5P ---------------------------------------------------------


class Sentinel5PBackfill(BackfillStrategy):
    """Walk the Sentinel-5P catalog backward in 48h windows, extracting columns.

    Column extraction reuses the scheduled collector's download path and
    requires CDSE credentials; without them the backfill degrades to
    catalog-only. Granules whose column values are already in the DB are
    skipped before download, so re-running resumes where a previous run
    stopped. The bearer token is refreshed per window because granule
    downloads easily outlive a single CDSE token.
    """

    source_name = "sentinel5p"

    def __init__(
        self,
        collector: Sentinel5PCollector | None = None,
        *,
        window_hours: int = SENTINEL_WINDOW_HOURS,
        extract_columns: bool = True,
    ) -> None:
        self.collector = collector or Sentinel5PCollector()
        self.window_hours = window_hours
        self.extract_columns = extract_columns

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        total = 0
        columns_extracted = 0
        columns_skipped = 0
        failed_windows: list[str] = []
        window = timedelta(hours=self.window_hours)
        cursor = until

        creds_present = bool(settings.cdse_username and settings.cdse_password)
        do_columns = self.extract_columns and creds_present
        notes: str | None = None
        if self.extract_columns and not creds_present:
            notes = (
                "CDSE credentials not set; catalog-only (no column densities "
                "recovered — set CDSE_USERNAME/CDSE_PASSWORD and re-run)"
            )
            logger.warning("Sentinel-5P backfill: %s", notes)

        try:
            while cursor > since:
                # The collector itself calls ``odata_filter()`` with no args,
                # so the window is anchored here and the payload is shaped to
                # match what ``Sentinel5PCollector.normalize`` reads. Advance the
                # cursor before the work so a skipped window can never loop.
                window_end = cursor
                cursor -= window
                # Isolate each 48h window: a transient catalog/token failure on
                # one must not abort the multi-hour run (per-granule failures are
                # already isolated downstream; an idempotent re-run fills gaps).
                try:
                    catalog = await self._fetch_window(window_end)
                    raw: dict[str, Any] = catalog
                    if do_columns:
                        extracted, skipped = await self._extract_window_columns(
                            session, catalog
                        )
                        columns_extracted += len(extracted)
                        columns_skipped += skipped
                        raw = {**catalog, "extracted_columns": extracted}
                    points = self.collector.normalize(raw)
                    total += await _store_points(session, points)
                except Exception as exc:  # noqa: BLE001 - isolate one window
                    await session.rollback()
                    logger.warning(
                        "Sentinel-5P window failed",
                        extra={
                            "window_end": window_end.isoformat(),
                            "error": str(exc),
                        },
                    )
                    failed_windows.append(window_end.isoformat())
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total,
                error=f"{type(exc).__name__}: {exc}",
                notes=notes,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.collector.close()

        if do_columns:
            notes = (
                f"columns: {columns_extracted} granules extracted, "
                f"{columns_skipped} already in DB"
            )
        if failed_windows:
            suffix = (
                f"{len(failed_windows)} window(s) skipped (re-run to fill): "
                f"{', '.join(failed_windows)}"
            )
            notes = f"{notes}; {suffix}" if notes else suffix
        return BackfillResult(
            source=self.source_name,
            records=total,
            notes=notes,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    async def _fetch_window(self, window_end: datetime) -> dict[str, Any]:
        """Catalog fetch anchored at ``window_end`` instead of now."""
        from app.collectors.sentinel5p import API_BASE, RESULT_LIMIT

        client = await self.collector._get_client()
        params = {
            "$filter": odata_filter(window_end, self.window_hours),
            "$top": str(RESULT_LIMIT),
            "$orderby": "ContentDate/Start desc",
            "$expand": "Attributes",
        }
        response = await client.get(
            API_BASE, params=params, timeout=60.0, follow_redirects=True
        )
        response.raise_for_status()
        return {"value": response.json().get("value", []) or []}

    async def _extract_window_columns(
        self,
        session: AsyncSession,
        catalog: dict[str, Any],
    ) -> tuple[dict[str, float], int]:
        """Extract column densities for one window's not-yet-stored granules.

        Returns the ``extracted_columns`` mapping for ``normalize`` plus the
        count of granules skipped because their column row already exists.
        """
        candidates: list[dict[str, Any]] = []
        candidate_ids: list[str] = []
        for record in catalog.get("value", []):
            product_id = record.get("Id")
            code = extract_product_code(record.get("Name"))
            if not product_id or code not in COLUMN_PRODUCTS:
                continue
            candidates.append(record)
            candidate_ids.append(str(product_id))

        if not candidates:
            return {}, 0

        already_stored = await ids_with_stored_columns(session, candidate_ids)
        remaining = [
            record
            for record in candidates
            if str(record.get("Id")) not in already_stored
        ]
        if not remaining:
            return {}, len(candidates)

        client = await self.collector._get_client()
        token = await fetch_access_token(
            client, settings.cdse_username, settings.cdse_password
        )
        extracted = await self.collector._extract_columns(
            client, {"value": remaining}, token
        )
        return extracted, len(candidates) - len(remaining)


# NOAA GFS ------------------------------------------------------------


class NOAAGFSBackfill(BackfillStrategy):
    """Walk past GFS cycles within the NOMADS retention window."""

    source_name = "noaa_gfs"

    def __init__(self, retention_days: int = GFS_RETENTION_DAYS) -> None:
        self.retention_days = retention_days

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        from app.collectors.noaa_gfs import NOAAGFSCollector

        start = time.monotonic()
        collector = NOAAGFSCollector()

        oldest_reachable = datetime.now(timezone.utc) - timedelta(
            days=self.retention_days
        )
        effective_since = max(since, oldest_reachable)
        notes: str | None = None
        if effective_since > since:
            notes = (
                f"NOMADS retains ~{self.retention_days} days of cycles; "
                f"clamped 'since' from {since.date()} to {effective_since.date()}"
            )

        total = 0
        try:
            for cycle in _gfs_cycles_in_range(effective_since, until):
                try:
                    grid = await asyncio.wait_for(
                        collector._load_cycle(cycle),
                        timeout=60.0,
                    )
                except Exception as exc:
                    logger.warning(
                        "GFS backfill cycle unreachable",
                        extra={"cycle": cycle.isoformat(), "error": str(exc)},
                    )
                    continue
                if not grid:
                    continue
                raw = {
                    "cycle_time": cycle.isoformat(),
                    "cycle_date": cycle.strftime("%Y%m%d"),
                    "cycle_hour": cycle.hour,
                    "grid": grid,
                }
                points = collector.normalize(raw)
                total += await _store_points(session, points)
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total,
                error=f"{type(exc).__name__}: {exc}",
                notes=notes,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await collector.close()

        return BackfillResult(
            source=self.source_name,
            records=total,
            notes=notes,
            duration_ms=(time.monotonic() - start) * 1000,
        )


def _gfs_cycles_in_range(since: datetime, until: datetime) -> list[datetime]:
    """Enumerate GFS cycle times (00/06/12/18 UTC) within [since, until]."""
    cycles: list[datetime] = []
    cursor = until.replace(minute=0, second=0, microsecond=0)
    cursor = cursor.replace(hour=(cursor.hour // 6) * 6)
    while cursor >= since:
        cycles.append(cursor)
        cursor -= timedelta(hours=6)
    return cycles


# OpenWeather (no-op) -------------------------------------------------


class OpenWeatherBackfill(BackfillStrategy):
    """Documented no-op: the free OpenWeather tier has no historical API."""

    source_name = "openweather"

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        notes = (
            "OpenWeather free tier exposes only current-conditions endpoints; "
            "historical data requires the paid One Call API 3.0 Historical add-on. "
            "Use the scheduled collector to accumulate snapshots going forward."
        )
        return BackfillResult(
            source=self.source_name,
            records=0,
            skipped=True,
            notes=notes,
        )


# ASOS / IEM ----------------------------------------------------------


def _chunk_windows(
    since: datetime, until: datetime, chunk_days: int
) -> list[tuple[datetime, datetime]]:
    """Split ``[since, until]`` into windows of at most ``chunk_days`` each."""
    if until <= since:
        return []
    step = timedelta(days=max(1, chunk_days))
    windows: list[tuple[datetime, datetime]] = []
    cursor = since
    while cursor < until:
        end = min(cursor + step, until)
        windows.append((cursor, end))
        cursor = end
    return windows


class ASOSBackfill(BackfillStrategy):
    """Historical ASOS/METAR backfill via IEM, per station, chunked in time.

    One request per (station, time-chunk) keeps each CSV small and the request
    count low — friendly to IEM's 1 req/s/IP throttle, which the shared
    ASOS_LIMITER already enforces. Idempotent via the data_points dedup index.
    """

    source_name = "asos"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
        chunk_days: int = 30,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self._limiter = rate_limiter or ASOS_LIMITER
        self.chunk_days = max(1, chunk_days)

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        total_inserted = 0
        failed: list[str] = []
        try:
            client = await self._client_or_default()
            stations = await fetch_target_stations(client, self._limiter)
            if not stations:
                return BackfillResult(
                    source=self.source_name,
                    records=0,
                    skipped=True,
                    notes="no online ASOS stations inside the target radius",
                    duration_ms=(time.monotonic() - start) * 1000,
                )

            logger.info("ASOS backfill: %d stations in target radius", len(stations))
            for station in stations:
                for chunk_since, chunk_until in _chunk_windows(
                    since, until, self.chunk_days
                ):
                    params = asos_request_params(
                        [station["id"]], since=chunk_since, until=chunk_until
                    )
                    # Isolate each (station, chunk): a persistent failure on one
                    # must not abort the rest (transient blips already retry in
                    # rate_limited_get; an idempotent re-run fills any gap).
                    try:
                        response = await rate_limited_get(
                            client, ASOS_API_BASE, limiter=self._limiter, params=params
                        )
                        points = asos_rows_to_points(
                            parse_asos_csv(response.text), source_name=self.source_name
                        )
                        total_inserted += await _store_points(session, points)
                    except Exception as exc:  # noqa: BLE001 - isolate one request
                        await session.rollback()
                        logger.warning(
                            "ASOS chunk failed",
                            extra={
                                "station": station["id"],
                                "since": chunk_since.isoformat(),
                                "error": str(exc),
                            },
                        )
                        failed.append(station["id"])
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.close()

        notes = None
        if failed:
            notes = (
                f"{len(failed)} station-chunk(s) failed and were skipped "
                f"(re-run to fill): {', '.join(sorted(set(failed)))}"
            )
        return BackfillResult(
            source=self.source_name,
            records=total_inserted,
            notes=notes,
            duration_ms=(time.monotonic() - start) * 1000,
        )


# EPA AQS (historical, backfill-only) ---------------------------------


class EPAAQSBackfill(BackfillStrategy):
    """Historical NO2/SO2/CO from EPA AQS, chunked by calendar month.

    Backfill-only: it supplies a historical ground leg for satellite-versus-
    ground comparisons. The parser does not retain a certification-status
    field, so these rows are not asserted to be certified. Skips cleanly when
    credentials are unset so a default all-source run never errors. Idempotent
    via the data_points dedup index.
    """

    source_name = EPA_AQS_SOURCE_NAME

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self._limiter = rate_limiter or AQS_LIMITER

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=AQS_REQUEST_TIMEOUT_S)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        if not settings.aqs_email or not settings.aqs_api_key:
            return BackfillResult(
                source=self.source_name,
                records=0,
                skipped=True,
                notes=(
                    "EPA AQS backfill requires AQS_EMAIL and AQS_API_KEY in "
                    "server/.env (free signup at aqs.epa.gov/data/api/signup). "
                    "AQS is configured as a historical, backfill-only source."
                ),
                duration_ms=(time.monotonic() - start) * 1000,
            )

        total_inserted = 0
        failed_chunks: list[str] = []
        bbox = target_bounding_box()
        try:
            client = await self._client_or_default()
            for bdate, edate in aqs_month_chunks(since, until):
                params = {
                    "email": settings.aqs_email,
                    "key": settings.aqs_api_key,
                    "param": ",".join(DEFAULT_AQS_PARAM_CODES),
                    "bdate": bdate,
                    "edate": edate,
                    "minlat": f"{bbox.min_lat:.6f}",
                    "maxlat": f"{bbox.max_lat:.6f}",
                    "minlon": f"{bbox.min_lon:.6f}",
                    "maxlon": f"{bbox.max_lon:.6f}",
                }
                # Isolate each month: a transient failure or a bad body on one
                # chunk must not abandon the remaining months (idempotent re-run
                # fills the gap). The failed window is surfaced in notes.
                try:
                    response = await rate_limited_get(
                        client, AQS_SAMPLE_ENDPOINT, limiter=self._limiter, params=params
                    )
                    data = response.json().get("Data") or []
                    points = aqs_records_to_points(data, source_name=self.source_name)
                    total_inserted += await _store_points(session, points)
                except Exception as exc:  # noqa: BLE001 - isolate one month
                    await session.rollback()
                    logger.warning(
                        "EPA AQS chunk failed",
                        extra={"bdate": bdate, "edate": edate, "error": str(exc)},
                    )
                    failed_chunks.append(bdate)
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.close()

        notes = None
        if failed_chunks:
            notes = (
                f"{len(failed_chunks)} month-chunk(s) failed and were skipped "
                f"(re-run to fill): {', '.join(failed_chunks)}"
            )
        return BackfillResult(
            source=self.source_name,
            records=total_inserted,
            notes=notes,
            duration_ms=(time.monotonic() - start) * 1000,
        )


# TCEQ (preliminary near-real-time, scraped) --------------------------


def _tceq_cst_dates(since: datetime, until: datetime) -> list[date]:
    """Calendar dates (in CST) spanning ``[since, until]`` inclusive."""
    start = since.astimezone(TCEQ_CST).date()
    end = until.astimezone(TCEQ_CST).date()
    days: list[date] = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return days


class TCEQBackfill(BackfillStrategy):
    """Historical TCEQ CAMS NO2/SO2/CO via daily_summary.pl, per site per day.

    Reuses the collector's verified site registry + HTML parser. Heavily
    rate-limited, per-request isolated, and circuit-broken — the source is
    undocumented and fragile, so a sustained outage stops the run instead of
    hammering ~1190 (site, day) POSTs and risking a ban. POSTs go through
    ``rate_limited_post`` for 429 + transient-error retry. Idempotent via the
    data_points dedup index. TCEQ data is preliminary; EPA AQS provides the
    historical archive used for comparison.
    """

    source_name = "tceq"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
        sites: list[dict[str, Any]] | None = None,
        breaker_threshold: int = TCEQ_BREAKER_THRESHOLD,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self._limiter = rate_limiter or TCEQ_LIMITER
        self._sites = sites if sites is not None else TCEQ_SITES
        self._breaker_threshold = breaker_threshold

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        total_inserted = 0
        consecutive_failures = 0
        tripped = False
        try:
            client = await self._client_or_default()
            for site in self._sites:
                if tripped:
                    break
                for day in _tceq_cst_dates(since, until):
                    # rate_limited_post owns the limiter acquire + 429/transient
                    # retry. Isolate per request, but trip a breaker on a sustained
                    # streak so a down undocumented service is not hammered.
                    try:
                        response = await rate_limited_post(
                            client,
                            TCEQ_API_URL,
                            limiter=self._limiter,
                            data=user_date_post_data(site["select_site"], day),
                            headers={"User-Agent": TCEQ_USER_AGENT},
                        )
                    except Exception as exc:  # noqa: BLE001 - isolate one request
                        consecutive_failures += 1
                        logger.warning(
                            "TCEQ backfill request failed",
                            extra={
                                "aqs_cd": site.get("aqs_cd"),
                                "day": day.isoformat(),
                                "error": str(exc),
                            },
                        )
                        if consecutive_failures >= self._breaker_threshold:
                            tripped = True
                            logger.error(
                                "TCEQ backfill circuit breaker tripped after %d "
                                "consecutive failures; stopping",
                                consecutive_failures,
                            )
                            break
                        continue
                    consecutive_failures = 0
                    points = tceq_html_to_points(
                        response.text,
                        site=site,
                        source_name=self.source_name,
                        expected_date=day,
                    )
                    total_inserted += await _store_points(session, points)
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.close()

        notes = None
        if tripped:
            notes = (
                f"circuit breaker tripped after {consecutive_failures} consecutive "
                "failures; run incomplete (re-run to continue)"
            )
        return BackfillResult(
            source=self.source_name,
            records=total_inserted,
            notes=notes,
            duration_ms=(time.monotonic() - start) * 1000,
        )


# PurpleAir (paid; point-budget-capped) -------------------------------

PURPLEAIR_DEFAULT_POINT_CAP = 300_000  # ~$3 at $1/100k points
PURPLEAIR_DEFAULT_POINT_FLOOR = 50_000  # never spend the balance below this
PURPLEAIR_DEFAULT_AVERAGE_MINUTES = 60
PURPLEAIR_DEFAULT_WINDOW_DAYS = 14


class PurpleAirBackfill(BackfillStrategy):
    """Historical PurpleAir PM2.5, hard-capped against the points balance.

    Lists in-radius sensors, then walks each sensor's history in windows. Before
    each sensor it reads the free ``/organization`` balance and stops once spend
    crosses ``point_cap`` or the balance falls to ``point_floor`` — so a backfill
    can never run the account dry. Idempotent via the data_points dedup index.
    """

    source_name = "purpleair"

    def __init__(
        self,
        http_client: httpx.AsyncClient | None = None,
        *,
        rate_limiter: AsyncRateLimiter | None = None,
        point_cap: int = PURPLEAIR_DEFAULT_POINT_CAP,
        point_floor: int = PURPLEAIR_DEFAULT_POINT_FLOOR,
        average_minutes: int = PURPLEAIR_DEFAULT_AVERAGE_MINUTES,
        window_days: int = PURPLEAIR_DEFAULT_WINDOW_DAYS,
    ) -> None:
        self._client = http_client
        self._owns_client = http_client is None
        self._limiter = rate_limiter or PURPLEAIR_LIMITER
        self.point_cap = point_cap
        self.point_floor = point_floor
        self.average_minutes = average_minutes
        self.window_days = max(1, window_days)

    async def _client_or_default(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _remaining_points(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> int | None:
        response = await rate_limited_get(
            client, ORGANIZATION_ENDPOINT, limiter=self._limiter, headers=headers
        )
        value = response.json().get("remaining_points")
        return int(value) if isinstance(value, (int, float)) else None

    async def backfill(
        self,
        session: AsyncSession,
        *,
        since: datetime,
        until: datetime,
    ) -> BackfillResult:
        import time

        start = time.monotonic()
        if not settings.purpleair_api_key:
            return BackfillResult(
                source=self.source_name,
                records=0,
                skipped=True,
                notes="PurpleAir backfill requires PURPLEAIR_API_KEY in server/.env.",
                duration_ms=(time.monotonic() - start) * 1000,
            )

        headers = {"X-API-Key": settings.purpleair_api_key}
        total_inserted = 0
        capped = False
        try:
            client = await self._client_or_default()
            initial_points = await self._remaining_points(client, headers)

            listing = await rate_limited_get(
                client,
                PURPLEAIR_SENSORS_ENDPOINT,
                limiter=self._limiter,
                params=sensors_request_params(fields="sensor_index,latitude,longitude"),
                headers=headers,
            )
            payload = listing.json()
            sensors = [
                row
                for row in parse_purpleair_rows(
                    payload.get("fields", []), payload.get("data", [])
                )
                if row.get("latitude") is not None
                and row.get("longitude") is not None
                and within_target_radius(
                    float(row["latitude"]), float(row["longitude"])
                )
            ]
            logger.info("PurpleAir backfill: %d sensors in target radius", len(sensors))

            windows = _chunk_windows(since, until, self.window_days)
            for sensor in sensors:
                remaining = await self._remaining_points(client, headers)
                spent = (
                    initial_points - remaining
                    if initial_points is not None and remaining is not None
                    else 0
                )
                if (remaining is not None and remaining <= self.point_floor) or (
                    spent >= self.point_cap
                ):
                    capped = True
                    logger.warning(
                        "PurpleAir backfill stopping at point cap "
                        "(spent~%s, remaining~%s)",
                        spent,
                        remaining,
                    )
                    break

                sensor_index = int(sensor["sensor_index"])
                lat = float(sensor["latitude"])
                lon = float(sensor["longitude"])
                for window_start, window_end in windows:
                    response = await rate_limited_get(
                        client,
                        f"{PURPLEAIR_API_BASE}/sensors/{sensor_index}/history",
                        limiter=self._limiter,
                        params={
                            "start_timestamp": int(window_start.timestamp()),
                            "end_timestamp": int(window_end.timestamp()),
                            "average": self.average_minutes,
                            # ATM value + cf_1/RH extras for the Barkjohn
                            # correction (see purpleair.HISTORY_FIELDS).
                            "fields": HISTORY_FIELDS,
                        },
                        headers=headers,
                    )
                    body = response.json()
                    points = purpleair_history_to_points(
                        body.get("fields", []),
                        body.get("data", []),
                        sensor_index=sensor_index,
                        lat=lat,
                        lon=lon,
                        source_name=self.source_name,
                    )
                    total_inserted += await _store_points(session, points)
        except Exception as exc:
            return BackfillResult(
                source=self.source_name,
                records=total_inserted,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.monotonic() - start) * 1000,
            )
        finally:
            await self.close()

        notes = "stopped at point cap" if capped else None
        return BackfillResult(
            source=self.source_name,
            records=total_inserted,
            notes=notes,
            duration_ms=(time.monotonic() - start) * 1000,
        )


# Dispatcher ----------------------------------------------------------


def available_strategies(
    openaq_location_ids: Sequence[int] | None = None,
) -> list[BackfillStrategy]:
    """Default strategy set covering every backfillable source.

    OpenAQ history comes from the S3 archive, not the API: bulk pulls through
    the hosted API are what got the key suspended (2026-06-10). The API-based
    OpenAQBackfill remains available for explicit, small, rate-limited use.
    """
    return [
        OpenAQArchiveBackfill(location_ids=openaq_location_ids),
        Sentinel5PBackfill(),
        NOAAGFSBackfill(),
        OpenWeatherBackfill(),
        ASOSBackfill(),
        EPAAQSBackfill(),
        TCEQBackfill(),
        PurpleAirBackfill(),
    ]


async def run_backfill(
    session: AsyncSession,
    *,
    strategies: Sequence[BackfillStrategy],
    since: datetime,
    until: datetime,
) -> list[BackfillResult]:
    """Run each strategy in sequence; one failure does not abort the rest."""
    results: list[BackfillResult] = []
    for strategy in strategies:
        try:
            result = await strategy.backfill(session, since=since, until=until)
        except Exception as exc:
            result = BackfillResult(
                source=strategy.source_name,
                error=f"{type(exc).__name__}: {exc}",
            )
        results.append(result)
    return results


# CLI ----------------------------------------------------------------


_SOURCE_CHOICES = (
    "openaq",
    "openweather",
    "noaa_gfs",
    "sentinel5p",
    "asos",
    "epa_aqs",
    "tceq",
    "purpleair",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.collectors.backfill",
        description=(
            "Backfill historical DataPoints from configured sources. "
            "Idempotent via the data_points dedup index — safe to re-run."
        ),
    )
    parser.add_argument(
        "--source",
        choices=_SOURCE_CHOICES,
        default=None,
        help="Backfill one source instead of all available strategies.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Look back N days from --until (default 30). Ignored when --since is given.",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO date (or datetime); overrides --days.",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="ISO date (or datetime); defaults to now UTC.",
    )
    parser.add_argument(
        "--location-ids",
        default=None,
        help=(
            "Comma-separated OpenAQ location ids for the archive backfill. "
            "Defaults to the ids already present in the database."
        ),
    )
    return parser.parse_args(argv)


def _iso_z(dt: datetime) -> str:
    """Render UTC datetime as an OpenAQ-friendly ISO-8601 string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _measurement_timestamp(record: dict[str, Any]) -> datetime | None:
    """Pull the measurement timestamp from an OpenAQ /measurements record.

    The v3 ``/measurements`` endpoint emits ``period.datetimeFrom.utc``
    (start of the measurement window); the ``latest`` field on the
    /sensors endpoint, by contrast, uses ``datetime.utc``. Both shapes are
    tolerated so the same parser works against either source.
    """
    period = record.get("period") or {}
    from_dt = period.get("datetimeFrom") or {}
    if "utc" in from_dt:
        return parse_openaq_datetime(from_dt.get("utc"))
    legacy = record.get("datetime") or {}
    if "utc" in legacy:
        return parse_openaq_datetime(legacy.get("utc"))
    return None


def _resolve_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    until = _parse_dt(args.until) if args.until else datetime.now(timezone.utc)
    if args.since:
        since = _parse_dt(args.since)
    else:
        since = until - timedelta(days=args.days)
    return since, until


def _parse_dt(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _strategies_for(args: argparse.Namespace) -> list[BackfillStrategy]:
    location_ids = (
        [int(part) for part in args.location_ids.split(",") if part.strip()]
        if args.location_ids
        else None
    )
    strategies = available_strategies(openaq_location_ids=location_ids)
    if args.source is None:
        return strategies
    return [s for s in strategies if s.source_name == args.source]


def _format_result(r: BackfillResult) -> str:
    status = "skipped" if r.skipped else ("error" if r.error else "ok")
    parts = [r.source, status, f"records={r.records}", f"duration_ms={r.duration_ms:.0f}"]
    if r.error:
        parts.append(f"error={r.error}")
    if r.notes:
        parts.append(f"notes={r.notes}")
    return " | ".join(parts)


async def _amain(argv: list[str] | None = None) -> int:
    from app.collectors.logsetup import configure_logging
    from app.db.session import async_session, engine

    configure_logging()
    args = _parse_args(argv)
    since, until = _resolve_window(args)
    strategies = _strategies_for(args)

    print(
        f"Backfill window: {since.isoformat()} → {until.isoformat()} "
        f"({len(strategies)} strategy/strategies)"
    )

    async with async_session() as session:
        results = await run_backfill(
            session, strategies=strategies, since=since, until=until
        )

    for result in results:
        print(_format_result(result))

    await engine.dispose()
    return 0 if all((r.skipped or r.error is None) for r in results) else 1


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
