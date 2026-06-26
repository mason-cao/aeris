from datetime import date, datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from sqlalchemy import select

from app.collectors.backfill import (
    BackfillResult,
    BackfillStrategy,
    NOAAGFSBackfill,
    OpenAQBackfill,
    OpenWeatherBackfill,
    Sentinel5PBackfill,
    _parse_args,
    available_strategies,
    run_backfill,
)
from app.collectors.noaa_gfs import NOAAGFSCollector
from app.collectors.ratelimit import AsyncRateLimiter
from app.collectors.sentinel5p import Sentinel5PCollector
from app.config import settings
from app.db.models import DataPoint


def _fast_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(6_000_000)


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


# OpenAQ mock-response factories -----------------------------------------


def _location(loc_id: int, lat: float, lon: float) -> dict[str, Any]:
    return {
        "id": loc_id,
        "name": f"loc-{loc_id}",
        "coordinates": {"latitude": lat, "longitude": lon},
    }


def _sensor(sensor_id: int, parameter: str, unit: str = "ug/m3") -> dict[str, Any]:
    return {
        "id": sensor_id,
        "parameter": {"name": parameter, "units": unit},
    }


def _measurement(ts: datetime, value: float) -> dict[str, Any]:
    """Mirror OpenAQ v3's actual /measurements response shape.

    The timestamp lives at ``period.datetimeFrom.utc`` (start of the
    measurement window); there's also a ``period.datetimeTo`` and no
    per-measurement ``coordinates`` field — the caller falls back to the
    location's coordinates.
    """
    def iso_z(d: datetime) -> str:
        return d.isoformat().replace("+00:00", "Z")

    end = ts + timedelta(hours=1)
    return {
        "value": value,
        "period": {
            "label": "raw",
            "interval": "01:00:00",
            "datetimeFrom": {"utc": iso_z(ts)},
            "datetimeTo": {"utc": iso_z(end)},
        },
        "parameter": {"id": 2, "name": "pm25", "units": "ug/m3"},
    }


class _MockTransport(httpx.AsyncBaseTransport):
    """In-memory router for OpenAQ-style endpoints used by the OpenAQ strategy."""

    def __init__(self, *, routes: dict[str, Any]) -> None:
        self.routes = routes
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        query = dict(request.url.params)
        self.calls.append((path, query))
        payload = self.routes.get(path)
        if callable(payload):
            payload = payload(query)
        if payload is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=payload)


def _build_client(routes: dict[str, Any]) -> tuple[httpx.AsyncClient, _MockTransport]:
    transport = _MockTransport(routes=routes)
    return httpx.AsyncClient(transport=transport), transport


# Tests ------------------------------------------------------------------


def _dpc(ts: datetime, value: float):
    """A minimal distinct-keyed DataPointCreate for _store_points tests."""
    from app.collectors.base import DataPointCreate

    return DataPointCreate(
        timestamp=ts,
        lat=29.76,
        lon=-95.37,
        metric="pm25",
        value=value,
        unit="ug/m3",
        source="openaq",
        source_entity_id="sensor-1",
        raw_json=None,
    )


class TestStorePointsChunking:
    def test_insert_chunk_size_respects_param_caps(self) -> None:
        from app.collectors.backfill import _insert_chunk_size

        n_cols = 9
        sqlite_chunk = _insert_chunk_size("sqlite", n_cols)
        pg_chunk = _insert_chunk_size("postgresql", n_cols)
        # SQLite's portable 999-variable cap must never be exceeded by a batch.
        assert 1 <= sqlite_chunk
        assert sqlite_chunk * n_cols <= 999
        # Postgres' 65535 cap allows materially larger batches.
        assert pg_chunk * n_cols <= 65535
        assert pg_chunk > sqlite_chunk

    @pytest.mark.asyncio
    async def test_store_points_raises_for_unsupported_dialect(self) -> None:
        from app.collectors.backfill import _store_points

        class _Dialect:
            name = "mysql"

        class _Bind:
            dialect = _Dialect()

        class _Session:
            def get_bind(self):
                return _Bind()

            async def execute(self, stmt):  # pragma: no cover - must not run
                return None

            async def commit(self):  # pragma: no cover - must not run
                return None

        with pytest.raises(NotImplementedError, match="dialect"):
            await _store_points(_Session(), [_dpc(T0, 1.0)])

    @pytest.mark.asyncio
    async def test_stores_more_rows_than_one_sqlite_batch(self, db_session) -> None:
        from app.collectors.backfill import _insert_chunk_size, _store_points

        chunk = _insert_chunk_size("sqlite", 9)
        n = chunk * 2 + 5
        points = [_dpc(T0 + timedelta(minutes=i), float(i)) for i in range(n)]
        stored = await _store_points(db_session, points)
        assert stored == n
        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert len(rows) == n

    @pytest.mark.asyncio
    async def test_splits_large_insert_into_capped_chunks(
        self, db_session, monkeypatch
    ) -> None:
        from app.collectors import backfill as bf

        chunk = bf._insert_chunk_size("sqlite", 9)
        n = chunk * 2 + 5
        points = [_dpc(T0 + timedelta(minutes=i), float(i)) for i in range(n)]

        calls = {"n": 0}
        real = db_session.execute

        async def _counting(*args, **kwargs):
            calls["n"] += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr(db_session, "execute", _counting)
        await bf._store_points(db_session, points)
        # ceil((2*chunk + 5) / chunk) == 3 capped inserts, not one giant one.
        assert calls["n"] == 3


def _archive_csv_gz(
    n_rows: int, *, location_id: int, sensor_id: int, day: "date"
) -> bytes:
    """A gzipped OpenAQ-archive CSV for one (location, day) object."""
    import csv as _csv
    import gzip
    import io

    buf = io.StringIO()
    writer = _csv.DictWriter(
        buf,
        fieldnames=[
            "location_id", "sensors_id", "location", "datetime",
            "lat", "lon", "parameter", "units", "value",
        ],
    )
    writer.writeheader()
    for i in range(n_rows):
        ts = datetime(day.year, day.month, day.day, i % 24, tzinfo=timezone.utc)
        writer.writerow({
            "location_id": location_id,
            "sensors_id": sensor_id,
            "location": f"loc-{location_id}",
            "datetime": ts.isoformat(),
            "lat": "29.76",
            "lon": "-95.37",
            "parameter": "pm25",
            "units": "ug/m3",
            "value": str(10.0 + i),
        })
    return gzip.compress(buf.getvalue().encode("utf-8"))


class _FakeArchiveResponse:
    def __init__(self, status_code: int, content: bytes) -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "archive error", request=None, response=None  # type: ignore[arg-type]
            )


class _ConcurrencyTrackingClient:
    """Async client that records peak in-flight GETs (404 for unknown URLs)."""

    def __init__(self, content_by_url: dict[str, bytes]) -> None:
        self.content_by_url = content_by_url
        self.in_flight = 0
        self.max_in_flight = 0
        self.urls: list[str] = []

    async def get(self, url: str) -> _FakeArchiveResponse:
        import asyncio as _asyncio

        self.urls.append(url)
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        await _asyncio.sleep(0.005)
        self.in_flight -= 1
        content = self.content_by_url.get(url)
        if content is None:
            return _FakeArchiveResponse(404, b"")
        return _FakeArchiveResponse(200, content)

    async def aclose(self) -> None:
        pass


class TestOpenAQArchiveBackfill:
    @pytest.mark.asyncio
    async def test_fetches_concurrently_within_bound(self, db_session) -> None:
        from app.collectors.backfill import (
            OPENAQ_ARCHIVE_BASE,
            OpenAQArchiveBackfill,
            archive_key,
        )

        locs = [1, 2]
        days = [date(2026, 5, 1), date(2026, 5, 2), date(2026, 5, 3)]
        content_by_url = {
            f"{OPENAQ_ARCHIVE_BASE}/{archive_key(loc, day)}": _archive_csv_gz(
                2, location_id=loc, sensor_id=100 + loc, day=day
            )
            for loc in locs
            for day in days
        }
        client = _ConcurrencyTrackingClient(content_by_url)
        strategy = OpenAQArchiveBackfill(
            http_client=client,  # type: ignore[arg-type]
            location_ids=locs,
            request_delay_s=0.0,
            max_concurrency=4,
        )
        result = await strategy.backfill(
            db_session,
            since=datetime(2026, 5, 1, tzinfo=timezone.utc),
            until=datetime(2026, 5, 3, tzinfo=timezone.utc),
        )
        assert result.error is None
        # Serial code would peak at 1; the bound caps it at max_concurrency.
        assert 2 <= client.max_in_flight <= 4

    @pytest.mark.asyncio
    async def test_stores_all_objects_and_counts_missing(self, db_session) -> None:
        from app.collectors.backfill import (
            OPENAQ_ARCHIVE_BASE,
            OpenAQArchiveBackfill,
            archive_key,
        )

        locs = [1, 2]
        days = [date(2026, 5, 1), date(2026, 5, 2)]
        content_by_url: dict[str, bytes] = {}
        for loc in locs:
            for day in days:
                if loc == 2 and day == date(2026, 5, 2):
                    continue  # one object absent → 404
                url = f"{OPENAQ_ARCHIVE_BASE}/{archive_key(loc, day)}"
                content_by_url[url] = _archive_csv_gz(
                    3, location_id=loc, sensor_id=100 + loc, day=day
                )
        client = _ConcurrencyTrackingClient(content_by_url)
        strategy = OpenAQArchiveBackfill(
            http_client=client,  # type: ignore[arg-type]
            location_ids=locs,
            request_delay_s=0.0,
            max_concurrency=4,
        )
        result = await strategy.backfill(
            db_session,
            since=datetime(2026, 5, 1, tzinfo=timezone.utc),
            until=datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
        assert result.error is None
        # 3 present objects x 3 rows = 9 inserted; 1 object missing.
        assert result.records == 9
        assert "1 objects absent" in (result.notes or "")
        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert len(rows) == 9


class TestAvailableStrategies:
    def test_lists_all_collectors(self) -> None:
        names = {s.source_name for s in available_strategies()}
        assert names == {
            "openaq",
            "openweather",
            "noaa_gfs",
            "sentinel5p",
            "asos",
            "epa_aqs",
            "tceq",
            "purpleair",
        }


class TestOpenWeatherBackfill:
    @pytest.mark.asyncio
    async def test_returns_skipped_result_with_explanation(self, db_session) -> None:
        strategy = OpenWeatherBackfill()
        result = await strategy.backfill(
            db_session, since=T0 - timedelta(days=7), until=T0
        )
        assert isinstance(result, BackfillResult)
        assert result.source == "openweather"
        assert result.records == 0
        assert result.skipped is True
        assert result.notes is not None
        # Must explain WHY so the user understands the caveat
        assert "historical" in result.notes.lower()


class TestOpenAQBackfillSingleSensor:
    @pytest.mark.asyncio
    async def test_paginates_measurements_for_a_sensor(self, db_session) -> None:
        # Two pages of 2 measurements; backfill should insert all 4.
        measurements_page_1 = {
            "results": [
                _measurement(T0 + timedelta(hours=i), 10.0 + i) for i in range(2)
            ],
            "meta": {"page": 1, "limit": 2, "found": 4},
        }
        measurements_page_2 = {
            "results": [
                _measurement(T0 + timedelta(hours=2 + i), 12.0 + i) for i in range(2)
            ],
            "meta": {"page": 2, "limit": 2, "found": 4},
        }
        measurements_page_3: dict[str, Any] = {
            "results": [],
            "meta": {"page": 3, "limit": 2, "found": 4},
        }

        def measurements_router(query: dict[str, str]) -> dict[str, Any]:
            page = int(query.get("page", "1"))
            return {1: measurements_page_1, 2: measurements_page_2}.get(
                page, measurements_page_3
            )

        client, transport = _build_client(
            routes={
                "/v3/locations": {
                    "results": [_location(1, 29.76, -95.37)],
                    "meta": {"found": 1},
                },
                "/v3/locations/1/sensors": {
                    "results": [_sensor(100, "pm25")],
                    "meta": {"found": 1},
                },
                "/v3/sensors/100/measurements": measurements_router,
            },
        )
        try:
            strategy = OpenAQBackfill(http_client=client, rate_limiter=_fast_limiter(), page_size=2)
            result = await strategy.backfill(
                db_session,
                since=T0 - timedelta(days=1),
                until=T0 + timedelta(days=1),
            )
        finally:
            await client.aclose()

        assert result.source == "openaq"
        assert result.records == 4
        assert result.skipped is False

        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert len(rows) == 4
        assert {r.metric for r in rows} == {"pm25"}
        assert {r.source for r in rows} == {"openaq"}
        assert {r.source_entity_id for r in rows} == {"100"}

    @pytest.mark.asyncio
    async def test_measurements_pagination_is_bounded_by_max_pages(
        self, db_session
    ) -> None:
        # A misbehaving endpoint that returns a full page forever would spin the
        # `while True` loop indefinitely; max_pages caps the page count.
        page_size = 2

        def always_full(query: dict[str, str]) -> dict[str, Any]:
            page = int(query.get("page", "1"))
            base = (page - 1) * page_size
            return {
                "results": [
                    _measurement(T0 + timedelta(hours=base + i), 10.0 + base + i)
                    for i in range(page_size)
                ],
                "meta": {"page": page},
            }

        client, transport = _build_client(
            routes={
                "/v3/locations": {
                    "results": [_location(1, 29.76, -95.37)],
                    "meta": {"found": 1},
                },
                "/v3/locations/1/sensors": {
                    "results": [_sensor(100, "pm25")],
                    "meta": {"found": 1},
                },
                "/v3/sensors/100/measurements": always_full,
            },
        )
        try:
            strategy = OpenAQBackfill(
                http_client=client,
                rate_limiter=_fast_limiter(),
                page_size=page_size,
                max_pages=3,
            )
            await strategy.backfill(
                db_session,
                since=T0 - timedelta(days=1),
                until=T0 + timedelta(days=1),
            )
        finally:
            await client.aclose()

        measurement_calls = [
            path
            for path, _ in transport.calls
            if path == "/v3/sensors/100/measurements"
        ]
        assert len(measurement_calls) == 3

    @pytest.mark.asyncio
    async def test_filters_to_target_radius_locations_only(self, db_session) -> None:
        # Two locations: one inside Houston bbox, one in NYC. The strategy
        # must skip the NYC location entirely and not query its sensors.
        client, transport = _build_client(
            routes={
                "/v3/locations": {
                    "results": [
                        _location(1, 29.76, -95.37),   # Houston
                        _location(2, 40.71, -74.00),   # NYC
                    ],
                    "meta": {"found": 2},
                },
                "/v3/locations/1/sensors": {
                    "results": [_sensor(100, "pm25")],
                    "meta": {"found": 1},
                },
                "/v3/locations/2/sensors": {
                    "results": [_sensor(200, "pm25")],
                    "meta": {"found": 1},
                },
                "/v3/sensors/100/measurements": {
                    "results": [_measurement(T0, 10.0)],
                    "meta": {},
                },
                "/v3/sensors/200/measurements": {
                    "results": [_measurement(T0, 99.0)],
                    "meta": {},
                },
            },
        )
        try:
            strategy = OpenAQBackfill(http_client=client, rate_limiter=_fast_limiter(), page_size=100)
            await strategy.backfill(
                db_session,
                since=T0 - timedelta(days=1),
                until=T0 + timedelta(days=1),
            )
        finally:
            await client.aclose()

        # Only Houston's sensor 100 should be queried; NYC sensor 200 ignored.
        called_sensors = {
            path for path, _ in transport.calls if "/sensors/" in path and "/measurements" in path
        }
        assert called_sensors == {"/v3/sensors/100/measurements"}
        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert {float(r.value) for r in rows} == {10.0}

    @pytest.mark.asyncio
    async def test_date_range_passed_as_query_params(self, db_session) -> None:
        captured: dict[str, str] = {}

        def measurements_router(query: dict[str, str]) -> dict[str, Any]:
            captured.update(query)
            return {"results": [], "meta": {"page": 1, "found": 0}}

        client, _ = _build_client(
            routes={
                "/v3/locations": {
                    "results": [_location(1, 29.76, -95.37)],
                    "meta": {"found": 1},
                },
                "/v3/locations/1/sensors": {
                    "results": [_sensor(100, "pm25")],
                    "meta": {"found": 1},
                },
                "/v3/sensors/100/measurements": measurements_router,
            },
        )
        try:
            strategy = OpenAQBackfill(http_client=client, rate_limiter=_fast_limiter(), page_size=100)
            since = T0 - timedelta(days=30)
            until = T0
            await strategy.backfill(db_session, since=since, until=until)
        finally:
            await client.aclose()

        assert "datetime_from" in captured
        assert "datetime_to" in captured
        # ISO-8601 with UTC suffix
        assert captured["datetime_from"].startswith(since.strftime("%Y-%m-%d"))
        assert captured["datetime_to"].startswith(until.strftime("%Y-%m-%d"))

    @pytest.mark.asyncio
    async def test_skips_sensors_with_unsupported_parameter(self, db_session) -> None:
        # Parameter "foobar" is not in the OpenAQ PARAMETER_MAP; the sensor
        # should be skipped entirely (no measurements query for it).
        called_paths: list[str] = []

        def sensor_router(_query: dict[str, str]) -> dict[str, Any]:
            called_paths.append("measurements")
            return {"results": [], "meta": {"found": 0}}

        client, transport = _build_client(
            routes={
                "/v3/locations": {
                    "results": [_location(1, 29.76, -95.37)],
                    "meta": {"found": 1},
                },
                "/v3/locations/1/sensors": {
                    "results": [_sensor(100, "foobar")],
                    "meta": {"found": 1},
                },
                "/v3/sensors/100/measurements": sensor_router,
            },
        )
        try:
            strategy = OpenAQBackfill(http_client=client, rate_limiter=_fast_limiter(), page_size=100)
            result = await strategy.backfill(
                db_session,
                since=T0 - timedelta(days=1),
                until=T0,
            )
        finally:
            await client.aclose()

        assert result.records == 0
        # No call to the measurements endpoint
        assert called_paths == []

    @pytest.mark.asyncio
    async def test_idempotent_rerun_does_not_duplicate(self, db_session) -> None:
        # The strategy relies on the existing data_points unique index;
        # running it twice with the same response must end with the same row count.
        routes = {
            "/v3/locations": {
                "results": [_location(1, 29.76, -95.37)],
                "meta": {"found": 1},
            },
            "/v3/locations/1/sensors": {
                "results": [_sensor(100, "pm25")],
                "meta": {"found": 1},
            },
            "/v3/sensors/100/measurements": {
                "results": [_measurement(T0 + timedelta(hours=i), 10.0 + i) for i in range(3)],
                "meta": {"page": 1, "found": 3},
            },
        }
        client1, _ = _build_client(routes=routes)
        client2, _ = _build_client(routes=routes)
        try:
            r1 = await OpenAQBackfill(http_client=client1, rate_limiter=_fast_limiter(), page_size=100).backfill(
                db_session, since=T0 - timedelta(days=1), until=T0 + timedelta(days=1)
            )
        finally:
            await client1.aclose()
        try:
            await OpenAQBackfill(http_client=client2, rate_limiter=_fast_limiter(), page_size=100).backfill(
                db_session, since=T0 - timedelta(days=1), until=T0 + timedelta(days=1)
            )
        finally:
            await client2.aclose()

        assert r1.records == 3
        # Second run: API returns 3 rows but the dedup index prevents inserts.
        # The strategy reports "records seen", not "rows newly inserted" — that's
        # a fine interpretation for the user-facing summary.
        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert len(rows) == 3


class TestNOAAGFSBackfill:
    @pytest.mark.asyncio
    async def test_stores_points_from_each_cycle_via_load_cycle(
        self, db_session
    ) -> None:
        # Window floored to a 6h boundary so exactly one GFS cycle is in range.
        until = datetime.now(timezone.utc).replace(
            minute=0, second=0, microsecond=0
        )
        until = until.replace(hour=(until.hour // 6) * 6)
        since = until

        grid = [
            {
                "lat": 29.75,
                "lon": -95.37,
                "values": {
                    "gh_500": 5840.0,
                    "t_850": 295.0,
                    "u_10m": 3.4,
                    "v_10m": -1.2,
                    "surface_pressure": 101325.0,
                    "precipitable_water": 35.0,
                    "pbl_height": 850.0,
                },
            }
        ]

        with patch.object(
            NOAAGFSCollector, "_load_cycle", new=AsyncMock(return_value=grid)
        ):
            result = await NOAAGFSBackfill().backfill(
                db_session, since=since, until=until
            )

        assert result.error is None
        assert result.records == 7  # one cell x seven GFS variables

        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert len(rows) == 7
        assert {row.source for row in rows} == {"noaa_gfs"}


def _s5p_catalog_records() -> list[dict[str, Any]]:
    return [
        {
            "Id": "prod-no2-1",
            "Name": (
                "S5P_NRTI_L2__NO2____20260501T120000_20260501T123000"
                "_00001_03_020800_20260501T140000.nc"
            ),
            "ContentDate": {"Start": "2026-05-01T12:00:00.000Z"},
            "Attributes": [{"Name": "cloudCover", "Value": 12.5}],
        },
        {
            "Id": "prod-so2-1",
            "Name": (
                "S5P_NRTI_L2__SO2____20260501T120000_20260501T123000"
                "_00001_03_020800_20260501T140000.nc"
            ),
            "ContentDate": {"Start": "2026-05-01T12:00:00.000Z"},
            "Attributes": [{"Name": "cloudCover", "Value": 8.0}],
        },
    ]


def _s5p_collector() -> Sentinel5PCollector:
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.json = MagicMock(return_value={"value": _s5p_catalog_records()})
    client = MagicMock()
    client.get = AsyncMock(return_value=response)
    return Sentinel5PCollector(http_client=client)


def _recording_s5p_collector() -> tuple[Sentinel5PCollector, list[str]]:
    """Catalog-only collector that records each window's ``$filter`` string."""
    captured: list[str] = []
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.json = MagicMock(return_value={"value": []})

    async def fake_get(url, params=None, **kwargs):
        captured.append(params["$filter"])
        return response

    client = MagicMock()
    client.get = fake_get
    return Sentinel5PCollector(http_client=client), captured


def _window_failing_s5p_collector(fail_on_call: int) -> Sentinel5PCollector:
    """Catalog-only collector whose Nth catalog GET raises a transient error."""
    state = {"n": 0}
    response = MagicMock()
    response.raise_for_status = MagicMock(return_value=None)
    response.json = MagicMock(return_value={"value": _s5p_catalog_records()})

    async def fake_get(url, params=None, **kwargs):
        state["n"] += 1
        if state["n"] == fail_on_call:
            raise httpx.ConnectError("transient catalog blip")
        return response

    client = MagicMock()
    client.get = fake_get
    return Sentinel5PCollector(http_client=client)


class TestSentinel5PBackfill:
    @pytest.mark.asyncio
    async def test_one_failing_window_is_isolated(
        self, db_session, monkeypatch
    ) -> None:
        # A transient failure on one 48h window must not abort the whole
        # multi-window run; the rest still loads and the gap is in notes.
        monkeypatch.setattr(settings, "cdse_username", "")
        monkeypatch.setattr(settings, "cdse_password", "")
        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=120)  # three 48h windows

        result = await Sentinel5PBackfill(
            collector=_window_failing_s5p_collector(fail_on_call=2)
        ).backfill(db_session, since=since, until=until)

        assert result.error is None  # one bad window doesn't sink the run
        assert result.notes is not None and "window" in result.notes.lower()
        # Windows 1 and 3 processed (4 catalog records each); window 2 skipped.
        assert result.records == 8
    @pytest.mark.asyncio
    async def test_catalog_only_without_credentials(
        self, db_session, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "")
        monkeypatch.setattr(settings, "cdse_password", "")
        # since < until by 1h triggers exactly one 48h catalog window.
        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=1)

        result = await Sentinel5PBackfill(collector=_s5p_collector()).backfill(
            db_session, since=since, until=until
        )

        assert result.error is None
        # Catalog-only mode: each mapped product emits _granule_available
        # plus _cloud_cover — no column densities without a granule download.
        assert result.records == 4
        assert result.notes is not None and "CDSE credentials" in result.notes

        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        assert {row.metric for row in rows} == {
            "s5p_no2_granule_available",
            "s5p_no2_cloud_cover",
            "s5p_so2_granule_available",
            "s5p_so2_cloud_cover",
        }

    @pytest.mark.asyncio
    async def test_extracts_columns_with_credentials(
        self, db_session, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")
        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=1)

        collector = _s5p_collector()
        extract = AsyncMock(return_value={"prod-no2-1": 0.000123})
        monkeypatch.setattr(collector, "_extract_columns", extract)
        monkeypatch.setattr(
            "app.collectors.backfill.fetch_access_token",
            AsyncMock(return_value="tok"),
        )

        result = await Sentinel5PBackfill(collector=collector).backfill(
            db_session, since=since, until=until
        )

        assert result.error is None
        assert result.records == 5  # 2 availability + 2 cloud cover + 1 column
        assert result.notes is not None and "1 granules extracted" in result.notes

        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        columns = [row for row in rows if row.metric == "s5p_no2_column"]
        assert len(columns) == 1
        assert columns[0].value == pytest.approx(0.000123)
        # Both catalog records are column products; with nothing stored yet
        # the whole window goes to extraction.
        passed_catalog = extract.call_args.args[1]
        assert [r["Id"] for r in passed_catalog["value"]] == [
            "prod-no2-1",
            "prod-so2-1",
        ]

    @pytest.mark.asyncio
    async def test_skips_granules_whose_columns_are_already_stored(
        self, db_session, monkeypatch
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")
        until = datetime.now(timezone.utc)
        since = until - timedelta(hours=1)

        db_session.add(
            DataPoint(
                timestamp=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
                lat=29.7604,
                lon=-95.3698,
                metric="s5p_no2_column",
                value=0.0002,
                unit="mol/m^2",
                source="sentinel5p",
                source_entity_id="prod-no2-1",
            )
        )
        await db_session.commit()

        collector = _s5p_collector()
        extract = AsyncMock(return_value={"prod-so2-1": 0.0005})
        monkeypatch.setattr(collector, "_extract_columns", extract)
        monkeypatch.setattr(
            "app.collectors.backfill.fetch_access_token",
            AsyncMock(return_value="tok"),
        )

        result = await Sentinel5PBackfill(collector=collector).backfill(
            db_session, since=since, until=until
        )

        assert result.error is None
        assert result.notes is not None and "1 already in DB" in result.notes
        # The granule with a stored column never reaches the download path.
        passed_catalog = extract.call_args.args[1]
        assert [r["Id"] for r in passed_catalog["value"]] == ["prod-so2-1"]

        rows = (await db_session.execute(select(DataPoint))).scalars().all()
        so2_columns = [row for row in rows if row.metric == "s5p_so2_column"]
        assert len(so2_columns) == 1
        assert so2_columns[0].value == pytest.approx(0.0005)

    @pytest.mark.asyncio
    async def test_backfill_windows_are_distinct_and_bounded(
        self, db_session, monkeypatch
    ) -> None:
        # The backward walk must query a distinct, both-sides-bounded window
        # per step. window_hours=24 (non-default) also pins that _fetch_window
        # threads self.window_hours rather than hardcoding the 48h lookback.
        monkeypatch.setattr(settings, "cdse_username", "")
        monkeypatch.setattr(settings, "cdse_password", "")
        until = datetime(2026, 5, 10, 12, tzinfo=timezone.utc)
        since = until - timedelta(hours=72)  # three contiguous 24h windows
        collector, captured = _recording_s5p_collector()

        result = await Sentinel5PBackfill(
            collector=collector, window_hours=24
        ).backfill(db_session, since=since, until=until)

        assert result.error is None
        assert len(captured) == 3
        # Upper bounds step strictly backward; each lower bound tiles to the
        # previous upper bound — no gaps, no re-fetching the newest granules.
        assert "ContentDate/Start gt 2026-05-09T12:00:00.000Z" in captured[0]
        assert "ContentDate/Start le 2026-05-10T12:00:00.000Z" in captured[0]
        assert "ContentDate/Start gt 2026-05-08T12:00:00.000Z" in captured[1]
        assert "ContentDate/Start le 2026-05-09T12:00:00.000Z" in captured[1]
        assert "ContentDate/Start gt 2026-05-07T12:00:00.000Z" in captured[2]
        assert "ContentDate/Start le 2026-05-08T12:00:00.000Z" in captured[2]

    @pytest.mark.asyncio
    async def test_backfill_catalog_window_follows_redirect(
        self, db_session, monkeypatch
    ) -> None:
        # The backfill catalog GET hits the same CDSE endpoint as the live
        # collector and must follow a 30x or the whole window is lost.
        monkeypatch.setattr(settings, "cdse_username", "")
        monkeypatch.setattr(settings, "cdse_password", "")

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "moved=1" not in url:
                return httpx.Response(
                    301, json={"value": []}, headers={"location": url + "&moved=1"}
                )
            return httpx.Response(200, json={"value": _s5p_catalog_records()})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)
        until = datetime(2026, 5, 10, 12, tzinfo=timezone.utc)
        since = until - timedelta(hours=1)
        try:
            result = await Sentinel5PBackfill(collector=collector).backfill(
                db_session, since=since, until=until
            )
        finally:
            await client.aclose()

        assert result.error is None
        assert result.records == 4


class TestRunBackfillDispatcher:
    @pytest.mark.asyncio
    async def test_dispatches_to_named_strategies_only(self, db_session) -> None:
        class _StubStrategy(BackfillStrategy):
            source_name = "fake"

            def __init__(self) -> None:
                self.calls = 0

            async def backfill(self, session, *, since, until):
                self.calls += 1
                return BackfillResult(source=self.source_name, records=7)

        stub = _StubStrategy()
        results = await run_backfill(
            db_session,
            strategies=[stub],
            since=T0 - timedelta(days=1),
            until=T0,
        )
        assert len(results) == 1
        assert results[0].records == 7
        assert stub.calls == 1

    @pytest.mark.asyncio
    async def test_collects_results_from_all_strategies_even_if_one_fails(
        self, db_session
    ) -> None:
        class _OK(BackfillStrategy):
            source_name = "ok"

            async def backfill(self, session, *, since, until):
                return BackfillResult(source=self.source_name, records=3)

        class _Boom(BackfillStrategy):
            source_name = "boom"

            async def backfill(self, session, *, since, until):
                raise RuntimeError("simulated failure")

        results = await run_backfill(
            db_session,
            strategies=[_OK(), _Boom()],
            since=T0 - timedelta(days=1),
            until=T0,
        )
        by_source = {r.source: r for r in results}
        assert by_source["ok"].records == 3
        assert by_source["boom"].records == 0
        assert by_source["boom"].error is not None
        assert "simulated failure" in by_source["boom"].error


class TestCLIParsing:
    def test_defaults_run_all_sources_for_30_days(self) -> None:
        args = _parse_args([])
        assert args.source is None
        assert args.days == 30
        assert args.since is None
        assert args.until is None

    def test_source_filter(self) -> None:
        args = _parse_args(["--source", "openaq"])
        assert args.source == "openaq"

    def test_explicit_since_and_until(self) -> None:
        args = _parse_args(["--since", "2026-04-01", "--until", "2026-05-01"])
        assert args.since == "2026-04-01"
        assert args.until == "2026-05-01"

    def test_days_override(self) -> None:
        args = _parse_args(["--days", "7"])
        assert args.days == 7

    def test_invalid_source_rejected(self) -> None:
        # argparse choices keeps the user from typing a nonexistent source name
        with pytest.raises(SystemExit):
            _parse_args(["--source", "not_a_real_source"])
