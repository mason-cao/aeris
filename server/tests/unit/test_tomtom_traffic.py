from typing import Any

import httpx
import pytest

from app.collectors.tomtom_traffic import (
    TomTomTrafficCollector,
    TrafficQueryPoint,
    traffic_query_points,
)
from app.config import settings


@pytest.fixture
def collector() -> TomTomTrafficCollector:
    return TomTomTrafficCollector()


def make_flow(**overrides: Any) -> dict[str, Any]:
    flow = {
        "frc": "FRC0",
        "currentSpeed": 60,
        "freeFlowSpeed": 100,
        "currentTravelTime": 90,
        "freeFlowTravelTime": 60,
        "confidence": 0.95,
        "roadClosure": False,
    }
    flow.update(overrides)
    return {"flowSegmentData": flow}


def make_observation(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "point_id": "center",
        "requested_lat": 34.0515,
        "requested_lon": -84.0713,
        "payload": payload or make_flow(),
    }


class TestTomTomTrafficNormalize:
    def test_normalize_emits_speed_metrics(
        self, collector: TomTomTrafficCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        metrics = {point.metric for point in points}
        assert metrics == {
            "traffic_current_speed",
            "traffic_free_flow_speed",
            "traffic_current_travel_time",
            "traffic_free_flow_travel_time",
            "traffic_confidence",
            "traffic_speed_ratio",
        }

    def test_normalize_computes_speed_ratio(
        self, collector: TomTomTrafficCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})
        ratio = next(point for point in points if point.metric == "traffic_speed_ratio")

        assert ratio.value == pytest.approx(0.6)
        assert ratio.unit == "ratio"

    def test_normalize_skips_speed_ratio_without_free_flow(
        self, collector: TomTomTrafficCollector
    ) -> None:
        payload = make_flow(freeFlowSpeed=0)
        points = collector.normalize({"observations": [make_observation(payload)]})

        assert "traffic_speed_ratio" not in {point.metric for point in points}

    def test_normalize_handles_missing_flow_segment(
        self, collector: TomTomTrafficCollector
    ) -> None:
        observation = {
            "point_id": "center",
            "requested_lat": 34.0515,
            "requested_lon": -84.0713,
            "payload": {},
        }

        assert collector.normalize({"observations": [observation]}) == []

    def test_normalize_sets_grid_entity_id(
        self, collector: TomTomTrafficCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        assert all(point.source == "tomtom_traffic" for point in points)
        assert {point.source_entity_id for point in points} == {"grid:center"}

    def test_normalize_sets_timestamp_to_collection_time(
        self, collector: TomTomTrafficCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        assert all(point.timestamp.tzinfo is not None for point in points)


class TestTomTomTrafficFetch:
    @pytest.mark.asyncio
    async def test_fetch_requires_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "tomtom_api_key", "")
        collector = TomTomTrafficCollector()

        with pytest.raises(RuntimeError, match="TOMTOM_API_KEY"):
            await collector.fetch()

    @pytest.mark.asyncio
    async def test_fetch_queries_all_grid_points(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "tomtom_api_key", "test-key")
        seen_keys: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.params["unit"] == "KMPH"
            seen_keys.append(request.url.params["key"])
            return httpx.Response(200, json=make_flow())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = TomTomTrafficCollector(http_client=client)

        raw = await collector.fetch()

        assert len(raw["observations"]) == 5
        assert seen_keys == ["test-key"] * 5
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_raises_when_all_points_fail(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "tomtom_api_key", "test-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"error": "server"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = TomTomTrafficCollector(http_client=client)

        with pytest.raises(RuntimeError, match="no observations"):
            await collector.fetch()

        await client.aclose()


class TestTomTomTrafficHelpers:
    def test_traffic_query_points_returns_five_target_points(self) -> None:
        points = traffic_query_points()

        assert len(points) == 5
        assert all(isinstance(point, TrafficQueryPoint) for point in points)
        assert {point.point_id for point in points} == {
            "center",
            "north",
            "east",
            "south",
            "west",
        }
