import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx
import pytest
from sqlalchemy import select

from app.collectors.base import BaseCollector, CollectionResult, DataPointCreate
from app.db.models import DataPoint, DataSource


class MockCollector(BaseCollector):
    """Concrete implementation for testing the abstract BaseCollector."""

    source_name = "test_source"
    collect_interval_minutes = 60

    def __init__(self, raw_data: dict | None = None, should_fail: bool = False) -> None:
        super().__init__()
        self._raw_data = raw_data or {"items": []}
        self._should_fail = should_fail

    async def fetch(self) -> dict[str, Any]:
        if self._should_fail:
            raise ConnectionError("API unavailable")
        return self._raw_data

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        return [
            DataPointCreate(
                timestamp=datetime.now(timezone.utc),
                lat=34.0,
                lon=-84.0,
                metric="test_metric",
                value=float(i),
                unit="unit",
                source=self.source_name,
                source_entity_id=f"entity-{i}",
                raw_json=item,
            )
            for i, item in enumerate(raw_data.get("items", []))
        ]


class TestDataPointCreate:
    def test_creates_valid_point(self) -> None:
        point = DataPointCreate(
            timestamp=datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            lat=29.7604,
            lon=-95.3698,
            metric="pm25",
            value=42.0,
            unit="ug/m3",
            source="openaq",
            source_entity_id="Houston",
        )
        assert point.metric == "pm25"
        assert point.unit == "ug/m3"
        assert point.source_entity_id == "Houston"
        assert point.raw_json is None

    def test_with_raw_json(self) -> None:
        point = DataPointCreate(
            timestamp=datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            lat=29.7604,
            lon=-95.3698,
            metric="pm25",
            value=42.0,
            unit="ug/m3",
            source="openaq",
            source_entity_id="Houston",
            raw_json={"extra": "data"},
        )
        assert point.raw_json == {"extra": "data"}


class TestCollectionResult:
    def test_successful_result(self) -> None:
        result = CollectionResult(
            source="test",
            success=True,
            record_count=5,
            duration_ms=123.4,
        )
        assert result.success is True
        assert result.errors == []

    def test_failed_result(self) -> None:
        result = CollectionResult(
            source="test",
            success=False,
            errors=["Connection refused"],
        )
        assert result.success is False
        assert len(result.errors) == 1


class TestMockCollectorNormalize:
    def test_normalize_empty(self) -> None:
        collector = MockCollector(raw_data={"items": []})
        points = collector.normalize({"items": []})
        assert len(points) == 0

    def test_normalize_multiple(self) -> None:
        raw = {"items": [{"a": 1}, {"b": 2}, {"c": 3}]}
        collector = MockCollector(raw_data=raw)
        points = collector.normalize(raw)
        assert len(points) == 3
        assert all(p.source == "test_source" for p in points)


class TestMockCollectorFetch:
    @pytest.mark.asyncio
    async def test_fetch_returns_data(self) -> None:
        raw = {"items": [{"value": 1}]}
        collector = MockCollector(raw_data=raw)
        result = await collector.fetch()
        assert result == raw
        await collector.close()

    @pytest.mark.asyncio
    async def test_fetch_raises_on_failure(self) -> None:
        collector = MockCollector(should_fail=True)
        with pytest.raises(ConnectionError):
            await collector.fetch()
        await collector.close()


class TestMockCollectorCollect:
    @pytest.mark.asyncio
    async def test_collect_stores_points_and_updates_source(self, db_session) -> None:
        raw = {"items": [{"value": 1}, {"value": 2}]}
        collector = MockCollector(raw_data=raw)

        result = await collector.collect(db_session, max_retries=1)

        assert result.success is True
        assert result.record_count == 2

        data_result = await db_session.execute(
            select(DataPoint).where(DataPoint.source == "test_source")
        )
        points = data_result.scalars().all()
        assert len(points) == 2
        assert {point.unit for point in points} == {"unit"}

        source_result = await db_session.execute(
            select(DataSource).where(DataSource.name == "test_source")
        )
        source = source_result.scalar_one()
        assert source.status == "active"
        assert source.error_count == 0

    @pytest.mark.asyncio
    async def test_collect_rolls_back_points_when_status_update_fails(
        self, db_session
    ) -> None:
        # Atomicity: the data insert and the source-status update must commit
        # together. A failing status write must not leave the attempt's points
        # orphaned in the store.
        class StatusFailCollector(MockCollector):
            async def _update_source_status(self, session, *, success) -> None:
                if success:
                    raise RuntimeError("status write failed")
                await super()._update_source_status(session, success=success)

        collector = StatusFailCollector(raw_data={"items": [{"value": 1}, {"value": 2}]})

        result = await collector.collect(db_session, max_retries=1)

        assert result.success is False
        data_result = await db_session.execute(
            select(DataPoint).where(DataPoint.source == "test_source")
        )
        assert data_result.scalars().all() == []

    @pytest.mark.asyncio
    async def test_collect_does_not_retry_on_4xx(self, db_session, monkeypatch) -> None:
        # A 4xx (e.g. a rejected API key) won't succeed on retry; collect must
        # break immediately instead of burning the 90s backoff budget.
        slept: list[float] = []

        async def _no_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        calls = {"n": 0}

        class AuthFailCollector(MockCollector):
            async def fetch(self) -> dict[str, Any]:
                calls["n"] += 1
                request = httpx.Request("GET", "https://example.test")
                raise httpx.HTTPStatusError(
                    "unauthorized",
                    request=request,
                    response=httpx.Response(401, request=request),
                )

        result = await AuthFailCollector().collect(db_session, max_retries=3)

        assert result.success is False
        assert calls["n"] == 1
        assert slept == []

    @pytest.mark.asyncio
    async def test_collect_retries_transient_error(self, db_session, monkeypatch) -> None:
        # A network blip is transient: collect should use every attempt.
        slept: list[float] = []

        async def _no_sleep(seconds: float) -> None:
            slept.append(seconds)

        monkeypatch.setattr(asyncio, "sleep", _no_sleep)
        calls = {"n": 0}

        class FlakyCollector(MockCollector):
            async def fetch(self) -> dict[str, Any]:
                calls["n"] += 1
                raise ConnectionError("network blip")

        result = await FlakyCollector().collect(db_session, max_retries=3)

        assert result.success is False
        assert calls["n"] == 3
        assert len(slept) == 2

    @pytest.mark.asyncio
    async def test_collect_failure_sets_error_status(self, db_session) -> None:
        # Refactor guard for the retries-exhausted path: a failing fetch marks
        # the source 'error' and stores nothing.
        collector = MockCollector(should_fail=True)

        result = await collector.collect(db_session, max_retries=1)

        assert result.success is False
        assert result.errors

        source_result = await db_session.execute(
            select(DataSource).where(DataSource.name == "test_source")
        )
        source = source_result.scalar_one()
        assert source.status == "error"
        assert source.error_count == 1

        data_result = await db_session.execute(
            select(DataPoint).where(DataPoint.source == "test_source")
        )
        assert data_result.scalars().all() == []


class TestStoreDialectDedup:
    @pytest.mark.asyncio
    async def test_store_ignores_duplicate_observation_on_sqlite(
        self, db_session
    ) -> None:
        # _store dedups on (source, metric, source_entity_id, timestamp) via the
        # dialect's on_conflict_do_nothing. A re-collection that re-sees the same
        # observation must be a silent no-op, not a UNIQUE violation, and must
        # not overwrite the stored value. No other test exercises this branch.
        collector = MockCollector()
        ts = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)

        def _point(value: float) -> DataPointCreate:
            return DataPointCreate(
                timestamp=ts,
                lat=34.0,
                lon=-84.0,
                metric="test_metric",
                value=value,
                unit="unit",
                source=collector.source_name,
                source_entity_id="entity-dup",
            )

        await collector._store(db_session, [_point(1.0)])
        await db_session.commit()
        # Second run re-sees the same key with a different value.
        await collector._store(db_session, [_point(2.0)])
        await db_session.commit()

        rows = (
            await db_session.execute(
                select(DataPoint).where(DataPoint.source == "test_source")
            )
        ).scalars().all()
        assert len(rows) == 1
        assert rows[0].value == 1.0  # first write wins; the duplicate is dropped
