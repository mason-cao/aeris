from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.openaq import (
    PARAMETER_MAP,
    OpenAQCollector,
    normalize_openaq_unit,
    parse_openaq_datetime,
)
from app.config import settings


@pytest.fixture
def collector() -> OpenAQCollector:
    return OpenAQCollector()


def make_location(location_id: int = 100) -> dict[str, Any]:
    return {
        "id": location_id,
        "name": "Suwanee Monitor",
        "coordinates": {"latitude": 34.0515, "longitude": -84.0713},
    }


def make_sensor(
    sensor_id: int,
    parameter: str,
    *,
    unit: str = "\u00b5g/m\u00b3",
    value: float | None = 10.5,
    timestamp: str = "2026-04-13T12:00:00Z",
) -> dict[str, Any]:
    latest = None
    if value is not None:
        latest = {
            "datetime": {"utc": timestamp, "local": "2026-04-13T08:00:00-04:00"},
            "value": value,
            "coordinates": {"latitude": 34.0515, "longitude": -84.0713},
        }

    return {
        "id": sensor_id,
        "name": f"{parameter} sensor",
        "parameter": {
            "id": sensor_id,
            "name": parameter,
            "units": unit,
            "displayName": parameter.upper(),
        },
        "latest": latest,
    }


def make_raw(sensors: list[dict[str, Any]]) -> dict[str, Any]:
    location = make_location()
    return {
        "locations": [location],
        "sensors_by_location_id": {str(location["id"]): sensors},
    }


class TestOpenAQNormalize:
    def test_normalize_maps_all_plan_parameters(
        self, collector: OpenAQCollector
    ) -> None:
        sensors = [
            make_sensor(1, "pm25"),
            make_sensor(2, "pm10"),
            make_sensor(3, "o3", unit="ppm"),
            make_sensor(4, "no2", unit="ppb"),
            make_sensor(5, "so2", unit="ppb"),
            make_sensor(6, "co", unit="ppm"),
            make_sensor(7, "bc"),
        ]

        points = collector.normalize(make_raw(sensors))

        assert len(points) == 7
        assert {p.metric for p in points} == {
            "pm25",
            "pm10",
            "ozone",
            "no2",
            "so2",
            "co",
            "bc",
        }

    def test_normalize_sets_source_and_entity_id(
        self, collector: OpenAQCollector
    ) -> None:
        points = collector.normalize(make_raw([make_sensor(12345, "pm25")]))

        assert points[0].source == "openaq"
        assert points[0].source_entity_id == "12345"

    def test_normalize_sets_unit(self, collector: OpenAQCollector) -> None:
        points = collector.normalize(make_raw([make_sensor(1, "pm25")]))

        assert points[0].unit == "ug/m3"

    def test_normalize_parses_timestamp(self, collector: OpenAQCollector) -> None:
        points = collector.normalize(make_raw([make_sensor(1, "pm25")]))

        assert points[0].timestamp == datetime(
            2026, 4, 13, 12, 0, tzinfo=timezone.utc
        )

    def test_normalize_preserves_raw_json(self, collector: OpenAQCollector) -> None:
        points = collector.normalize(make_raw([make_sensor(1, "pm25")]))

        assert points[0].raw_json is not None
        assert points[0].raw_json["location"]["name"] == "Suwanee Monitor"
        assert points[0].raw_json["sensor"]["id"] == 1

    def test_normalize_skips_unknown_parameter(
        self, collector: OpenAQCollector
    ) -> None:
        points = collector.normalize(make_raw([make_sensor(1, "pm4")]))

        assert points == []

    def test_normalize_skips_null_latest(self, collector: OpenAQCollector) -> None:
        sensor = make_sensor(1, "pm25", value=None)

        assert collector.normalize(make_raw([sensor])) == []

    def test_normalize_skips_bad_timestamp(
        self, collector: OpenAQCollector
    ) -> None:
        sensor = make_sensor(1, "pm25", timestamp="not-a-date")

        assert collector.normalize(make_raw([sensor])) == []

    def test_normalize_empty_response(self, collector: OpenAQCollector) -> None:
        points = collector.normalize({"locations": [], "sensors_by_location_id": {}})

        assert points == []


class TestOpenAQFetch:
    @pytest.mark.asyncio
    async def test_fetch_walks_locations_and_sensors(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "openaq_api_key", "test-key")
        location = make_location()
        sensor = make_sensor(1, "pm25")
        seen_paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            assert request.headers["X-API-Key"] == "test-key"

            if request.url.path == "/v3/locations":
                assert "bbox" in request.url.params
                return httpx.Response(
                    200,
                    json={"meta": {"found": 1}, "results": [location]},
                )
            if request.url.path == f"/v3/locations/{location['id']}/sensors":
                return httpx.Response(
                    200,
                    json={"meta": {"found": 1}, "results": [sensor]},
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client)

        raw = await collector.fetch()

        assert seen_paths == [
            "/v3/locations",
            f"/v3/locations/{location['id']}/sensors",
        ]
        assert raw["locations"] == [location]
        assert raw["sensors_by_location_id"][str(location["id"])] == [sensor]
        await client.aclose()


class TestOpenAQHelpers:
    def test_unit_map_covers_common_openaq_units(self) -> None:
        assert normalize_openaq_unit("\u00b5g/m\u00b3") == "ug/m3"
        assert normalize_openaq_unit("ug/m3") == "ug/m3"
        assert normalize_openaq_unit("ppm") == "ppm"
        assert normalize_openaq_unit("ppb") == "ppb"

    def test_parse_openaq_datetime_returns_utc(self) -> None:
        parsed = parse_openaq_datetime("2026-04-13T08:00:00-04:00")

        assert parsed == datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc)

    def test_parameter_map_covers_plan_metrics(self) -> None:
        assert set(PARAMETER_MAP) >= {"pm25", "pm10", "o3", "no2", "so2", "co", "bc"}
