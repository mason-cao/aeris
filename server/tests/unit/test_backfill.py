import json
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest
import pytest_asyncio
from sqlalchemy import delete, select

from app.collectors.backfill import (
    BackfillResult,
    BackfillStrategy,
    OpenAQBackfill,
    OpenWeatherBackfill,
    _parse_args,
    available_strategies,
    run_backfill,
)
from app.db.models import DataPoint


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture(autouse=True)
async def _clean_data_points(db_session):
    await db_session.execute(delete(DataPoint))
    await db_session.commit()
    yield


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
    end = ts + timedelta(hours=1)
    iso_z = lambda d: d.isoformat().replace("+00:00", "Z")
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


class TestAvailableStrategies:
    def test_lists_all_four_collectors(self) -> None:
        names = {s.source_name for s in available_strategies()}
        assert names == {"openaq", "openweather", "noaa_gfs", "sentinel5p"}


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
            strategy = OpenAQBackfill(http_client=client, page_size=2)
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
            strategy = OpenAQBackfill(http_client=client, page_size=100)
            result = await strategy.backfill(
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
            strategy = OpenAQBackfill(http_client=client, page_size=100)
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
            strategy = OpenAQBackfill(http_client=client, page_size=100)
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
            r1 = await OpenAQBackfill(http_client=client1, page_size=100).backfill(
                db_session, since=T0 - timedelta(days=1), until=T0 + timedelta(days=1)
            )
        finally:
            await client1.aclose()
        try:
            r2 = await OpenAQBackfill(http_client=client2, page_size=100).backfill(
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
