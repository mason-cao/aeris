from typing import Any

import pytest

from app.collectors.base import BaseCollector, CollectionResult, DataPointCreate
from app.collectors.run_all import exit_code, format_result, run_collectors


class RunnerCollector(BaseCollector):
    source_name = "runner_collector"
    collect_interval_minutes = 60

    def __init__(
        self,
        result: CollectionResult | None = None,
        *,
        should_raise: bool = False,
    ) -> None:
        super().__init__()
        self.result = result or CollectionResult(
            source=self.source_name,
            success=True,
            record_count=1,
            duration_ms=2.0,
        )
        self.should_raise = should_raise
        self.closed = False

    async def fetch(self) -> dict[str, Any]:
        return {}

    def normalize(self, raw_data: dict[str, Any]) -> list[DataPointCreate]:
        return []

    async def collect(self, session, max_retries: int = 3) -> CollectionResult:
        if self.should_raise:
            raise RuntimeError("boom")
        return self.result

    async def close(self) -> None:
        self.closed = True
        await super().close()


class TestRunCollectors:
    @pytest.mark.asyncio
    async def test_run_collectors_returns_results(self) -> None:
        collectors = [
            RunnerCollector(
                CollectionResult(
                    source="one",
                    success=True,
                    record_count=2,
                    duration_ms=4.0,
                )
            ),
            RunnerCollector(
                CollectionResult(
                    source="two",
                    success=True,
                    record_count=3,
                    duration_ms=5.0,
                )
            ),
        ]

        results = await run_collectors(None, collectors, max_retries=1)

        assert [result.source for result in results] == ["one", "two"]
        assert [result.record_count for result in results] == [2, 3]
        assert all(collector.closed for collector in collectors)

    @pytest.mark.asyncio
    async def test_run_collectors_isolates_unhandled_failures(self) -> None:
        collectors = [
            RunnerCollector(should_raise=True),
            RunnerCollector(
                CollectionResult(
                    source="after_failure",
                    success=True,
                    record_count=1,
                    duration_ms=1.0,
                )
            ),
        ]

        results = await run_collectors(None, collectors, max_retries=1)

        assert len(results) == 2
        assert results[0].source == "runner_collector"
        assert results[0].success is False
        assert "RuntimeError: boom" in results[0].errors
        assert results[1].source == "after_failure"
        assert results[1].success is True
        assert all(collector.closed for collector in collectors)


class TestRunAllFormatting:
    def test_format_result_success(self) -> None:
        result = CollectionResult(
            source="openaq",
            success=True,
            record_count=4,
            duration_ms=12.5,
        )

        assert format_result(result) == "openaq | ok | records=4 | duration_ms=12.5"

    def test_format_result_failure_includes_errors(self) -> None:
        result = CollectionResult(
            source="nasa_firms",
            success=False,
            record_count=0,
            duration_ms=9.0,
            errors=["missing key"],
        )

        assert format_result(result) == (
            "nasa_firms | failed | records=0 | duration_ms=9.0 | "
            "errors=missing key"
        )

    def test_exit_code_success(self) -> None:
        assert exit_code([CollectionResult(source="one", success=True)]) == 0

    def test_exit_code_failure(self) -> None:
        assert exit_code(
            [
                CollectionResult(source="one", success=True),
                CollectionResult(source="two", success=False),
            ]
        ) == 1
