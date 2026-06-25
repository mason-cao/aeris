from datetime import datetime, timedelta, timezone

import httpx
import pytest
from sqlalchemy import func, select

from app.collectors.backfill import PurpleAirBackfill
from app.collectors.purpleair import (
    MAX_READING_AGE_S,
    PurpleAirCollector,
    purpleair_history_to_points,
    purpleair_sensors_to_points,
)
from app.collectors.ratelimit import AsyncRateLimiter
from app.config import settings
from app.db.models import DataPoint

NOW = datetime(2026, 6, 24, 18, 0, tzinfo=timezone.utc)
IN_LAT, IN_LON = 29.76, -95.37
FAR_LAT, FAR_LON = 40.0, -100.0

SENSOR_FIELDS = ["sensor_index", "last_seen", "name", "latitude", "longitude", "pm2.5"]


def fast_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(6_000_000)


def epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def sensor_row(sid: int, *, lat=IN_LAT, lon=IN_LON, pm=6.2, seen=None):
    seen = seen if seen is not None else epoch(NOW - timedelta(minutes=5))
    return [sid, seen, f"sensor {sid}", lat, lon, pm]


class TestSensorsToPoints:
    def test_maps_pm25_with_unit_and_entity(self) -> None:
        pts = purpleair_sensors_to_points(
            SENSOR_FIELDS, [sensor_row(2386)], source_name="purpleair", now=NOW
        )
        assert len(pts) == 1
        p = pts[0]
        assert p.metric == "pm25"
        assert p.unit == "ug/m3"
        assert p.value == pytest.approx(6.2)
        assert p.source == "purpleair"
        assert p.source_entity_id == "2386"
        assert p.lat == pytest.approx(IN_LAT)
        assert p.timestamp.tzinfo is not None

    def test_maps_by_field_name_not_position(self) -> None:
        # A different field order must still resolve correctly.
        fields = ["pm2.5", "latitude", "longitude", "last_seen", "sensor_index"]
        row = [9.9, IN_LAT, IN_LON, epoch(NOW - timedelta(minutes=2)), 55]
        pts = purpleair_sensors_to_points(fields, [row], source_name="purpleair", now=NOW)
        assert pts[0].value == pytest.approx(9.9)
        assert pts[0].source_entity_id == "55"

    def test_filters_out_of_radius(self) -> None:
        pts = purpleair_sensors_to_points(
            SENSOR_FIELDS, [sensor_row(1, lat=FAR_LAT, lon=FAR_LON)],
            source_name="purpleair", now=NOW,
        )
        assert pts == []

    def test_drops_stale_reading(self) -> None:
        old = epoch(NOW - timedelta(seconds=MAX_READING_AGE_S + 3600))
        pts = purpleair_sensors_to_points(
            SENSOR_FIELDS, [sensor_row(1, seen=old)], source_name="purpleair", now=NOW
        )
        assert pts == []

    def test_skips_missing_or_bad_values(self) -> None:
        rows = [
            [1, epoch(NOW), "a", IN_LAT, IN_LON, None],  # no pm
            [2, epoch(NOW), "b", None, IN_LON, 5.0],  # no lat
            [3, epoch(NOW), "c", IN_LAT, IN_LON, "NaNish"],  # bad pm
        ]
        assert purpleair_sensors_to_points(
            SENSOR_FIELDS, rows, source_name="purpleair", now=NOW
        ) == []

    def test_empty(self) -> None:
        assert purpleair_sensors_to_points(SENSOR_FIELDS, [], source_name="purpleair", now=NOW) == []


HISTORY_FIELDS = ["time_stamp", "pm2.5_alt", "pm2.5_atm"]


class TestHistoryToPoints:
    def test_maps_history_rows(self) -> None:
        ts = epoch(datetime(2026, 6, 20, 5, 0, tzinfo=timezone.utc))
        data = [[ts, 2.2, 3.6]]
        pts = purpleair_history_to_points(
            HISTORY_FIELDS, data, sensor_index=2386, lat=IN_LAT, lon=IN_LON,
            source_name="purpleair",
        )
        assert len(pts) == 1
        assert pts[0].metric == "pm25"
        assert pts[0].unit == "ug/m3"
        assert pts[0].value == pytest.approx(3.6)  # pm2.5_atm
        assert pts[0].source_entity_id == "2386"
        assert pts[0].timestamp == datetime(2026, 6, 20, 5, 0, tzinfo=timezone.utc)

    def test_missing_field_yields_nothing(self) -> None:
        assert purpleair_history_to_points(
            ["time_stamp", "humidity"], [[123, 50]], sensor_index=1, lat=IN_LAT,
            lon=IN_LON, source_name="purpleair",
        ) == []


class TestPurpleAirFetch:
    @pytest.mark.asyncio
    async def test_fetch_uses_bbox_and_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "test-key")
        seen = {}

        # last_seen relative to the real clock: normalize() applies its
        # staleness guard against datetime.now(), so a fixed timestamp would go
        # stale and drop the row once enough wall-clock passes.
        fresh = sensor_row(
            2386, seen=epoch(datetime.now(timezone.utc) - timedelta(minutes=1))
        )

        def handler(request: httpx.Request) -> httpx.Response:
            seen["key"] = request.headers.get("X-API-Key")
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json={"fields": SENSOR_FIELDS, "data": [fresh]})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = PurpleAirCollector(http_client=client, rate_limiter=fast_limiter())
        raw = await collector.fetch()
        pts = collector.normalize(raw)
        await client.aclose()

        assert seen["key"] == "test-key"
        assert {"nwlng", "nwlat", "selng", "selat"} <= set(seen["params"])
        assert len(pts) == 1

    @pytest.mark.asyncio
    async def test_fetch_requires_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "")
        collector = PurpleAirCollector(rate_limiter=fast_limiter())
        with pytest.raises(RuntimeError):
            await collector.fetch()


T0 = datetime(2026, 6, 20, tzinfo=timezone.utc)


async def _count(session) -> int:
    return await session.scalar(select(func.count()).select_from(DataPoint))


def _backfill_handler(remaining_seq):
    """Serve /sensors once, /organization from a queue, /history columnar."""
    state = {"org": list(remaining_seq), "history_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/organization"):
            val = state["org"].pop(0) if state["org"] else state["org_last"]
            state["org_last"] = val
            return httpx.Response(200, json={"remaining_points": val})
        if path.endswith("/sensors"):
            return httpx.Response(
                200,
                json={
                    "fields": ["sensor_index", "latitude", "longitude"],
                    "data": [[10, IN_LAT, IN_LON], [11, 29.75, -95.36]],
                },
            )
        # history
        state["history_calls"] += 1
        ts = epoch(T0 + timedelta(hours=1))
        return httpx.Response(
            200, json={"fields": HISTORY_FIELDS, "data": [[ts, 2.0, 4.0]]}
        )

    return handler, state


class TestPurpleAirBackfill:
    @pytest.mark.asyncio
    async def test_stores_history_points(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "k")
        handler, state = _backfill_handler([1_000_000] * 10)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = PurpleAirBackfill(
            http_client=client, rate_limiter=fast_limiter(),
            point_cap=500_000, window_days=30,
        )
        try:
            result = await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(days=1)
            )
        finally:
            await client.aclose()

        assert result.source == "purpleair"
        assert result.error is None
        assert result.records == 2  # 2 sensors x 1 history point
        assert await _count(db_session) == 2

    @pytest.mark.asyncio
    async def test_halts_when_point_cap_exceeded(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "k")
        # initial=100k; before sensor1 -> 100k (spent 0); before sensor2 -> 40k
        # (spent 60k >= cap 50k -> stop before sensor2).
        handler, state = _backfill_handler([100_000, 100_000, 40_000])
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = PurpleAirBackfill(
            http_client=client, rate_limiter=fast_limiter(),
            point_cap=50_000, window_days=30,
        )
        try:
            result = await strategy.backfill(
                db_session, since=T0, until=T0 + timedelta(days=1)
            )
        finally:
            await client.aclose()

        # Only the first sensor's history was fetched before the cap tripped.
        assert state["history_calls"] == 1
        assert result.notes is not None and "cap" in result.notes.lower()

    @pytest.mark.asyncio
    async def test_skipped_without_key(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "")
        strategy = PurpleAirBackfill(rate_limiter=fast_limiter())
        result = await strategy.backfill(db_session, since=T0, until=T0 + timedelta(days=1))
        assert result.skipped is True
        assert result.records == 0

    @pytest.mark.asyncio
    async def test_idempotent_rerun(self, db_session, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "k")
        handler, _ = _backfill_handler([1_000_000] * 20)
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        strategy = PurpleAirBackfill(
            http_client=client, rate_limiter=fast_limiter(), point_cap=900_000
        )
        try:
            await strategy.backfill(db_session, since=T0, until=T0 + timedelta(days=1))
            await strategy.backfill(db_session, since=T0, until=T0 + timedelta(days=1))
        finally:
            await client.aclose()

        assert await _count(db_session) == 2
