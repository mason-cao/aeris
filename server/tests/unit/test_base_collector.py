from datetime import datetime, timezone
from typing import Any

import pytest

from app.collectors.base import BaseCollector, CollectionResult, DataPointCreate


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
                source=self.source_name,
                raw_json=item,
            )
            for i, item in enumerate(raw_data.get("items", []))
        ]


class TestDataPointCreate:
    def test_creates_valid_point(self) -> None:
        point = DataPointCreate(
            timestamp=datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            lat=34.0515,
            lon=-84.0713,
            metric="pm25",
            value=42.0,
            source="epa_airnow",
        )
        assert point.metric == "pm25"
        assert point.raw_json is None

    def test_with_raw_json(self) -> None:
        point = DataPointCreate(
            timestamp=datetime(2026, 4, 5, 12, 0, tzinfo=timezone.utc),
            lat=34.0515,
            lon=-84.0713,
            metric="pm25",
            value=42.0,
            source="epa_airnow",
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
