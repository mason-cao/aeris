from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.noaa_weather import (
    FIELD_MAP,
    OpenWeatherCollector,
    WeatherQueryPoint,
    extract_precipitation,
    parse_observation_time,
    weather_query_points,
)
from app.config import settings


@pytest.fixture
def collector() -> OpenWeatherCollector:
    return OpenWeatherCollector()


def make_payload(**overrides: Any) -> dict[str, Any]:
    payload = {
        "coord": {"lon": -84.0713, "lat": 34.0515},
        "weather": [{"id": 501, "main": "Rain", "description": "moderate rain"}],
        "main": {
            "temp": 21.4,
            "pressure": 1012,
            "humidity": 58,
        },
        "wind": {
            "speed": 3.4,
            "deg": 220,
        },
        "rain": {"1h": 1.2},
        "snow": {"1h": 0.3},
        "clouds": {"all": 75},
        "dt": 1776081600,
        "id": 4225309,
        "name": "Suwanee",
    }
    payload.update(overrides)
    return payload


def make_observation(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "point_id": "center",
        "requested_lat": 34.0515,
        "requested_lon": -84.0713,
        "payload": payload or make_payload(),
    }


class TestOpenWeatherNormalize:
    def test_normalize_maps_metrics(self, collector: OpenWeatherCollector) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        assert len(points) == 7
        assert {point.metric for point in points} == {
            "temperature",
            "humidity",
            "pressure",
            "wind_speed",
            "wind_direction",
            "cloud_cover",
            "precipitation",
        }

    def test_normalize_sets_units(self, collector: OpenWeatherCollector) -> None:
        points = collector.normalize({"observations": [make_observation()]})
        units_by_metric = {point.metric: point.unit for point in points}

        assert units_by_metric == {
            "temperature": "degC",
            "humidity": "percent",
            "pressure": "hPa",
            "wind_speed": "m/s",
            "wind_direction": "degree",
            "cloud_cover": "percent",
            "precipitation": "mm/h",
        }

    def test_normalize_uses_observation_timestamp(
        self, collector: OpenWeatherCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        assert all(
            point.timestamp == datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc)
            for point in points
        )

    def test_normalize_sets_source_and_grid_entity_id(
        self, collector: OpenWeatherCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        assert all(point.source == "openweather" for point in points)
        assert {point.source_entity_id for point in points} == {"grid:center"}

    def test_normalize_combines_rain_and_snow(
        self, collector: OpenWeatherCollector
    ) -> None:
        points = collector.normalize({"observations": [make_observation()]})
        precipitation = next(point for point in points if point.metric == "precipitation")

        assert precipitation.value == 1.5

    def test_normalize_stores_zero_precipitation_when_absent(
        self, collector: OpenWeatherCollector
    ) -> None:
        payload = make_payload(rain=None, snow=None)
        points = collector.normalize({"observations": [make_observation(payload)]})
        precipitation = next(point for point in points if point.metric == "precipitation")

        assert precipitation.value == 0.0

    def test_normalize_preserves_raw_json(self, collector: OpenWeatherCollector) -> None:
        points = collector.normalize({"observations": [make_observation()]})

        assert points[0].raw_json is not None
        assert points[0].raw_json["payload"]["name"] == "Suwanee"
        assert points[0].raw_json["point_id"] == "center"

    def test_normalize_skips_bad_timestamp(
        self, collector: OpenWeatherCollector
    ) -> None:
        payload = make_payload(dt="bad")

        assert collector.normalize({"observations": [make_observation(payload)]}) == []

    def test_normalize_skips_non_numeric_metric(
        self, collector: OpenWeatherCollector
    ) -> None:
        payload = make_payload(main={"temp": "bad", "pressure": 1012, "humidity": 58})
        points = collector.normalize({"observations": [make_observation(payload)]})

        assert "temperature" not in {point.metric for point in points}
        assert "pressure" in {point.metric for point in points}


class TestOpenWeatherFetch:
    @pytest.mark.asyncio
    async def test_fetch_queries_all_grid_points(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "openweather_api_key", "test-key")
        seen_points: list[tuple[str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/data/2.5/weather"
            assert request.url.params["appid"] == "test-key"
            assert request.url.params["units"] == "metric"
            seen_points.append((request.url.params["lat"], request.url.params["lon"]))
            return httpx.Response(200, json=make_payload())

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenWeatherCollector(http_client=client)

        raw = await collector.fetch()

        assert len(raw["observations"]) == 5
        assert len(seen_points) == 5
        assert {obs["point_id"] for obs in raw["observations"]} == {
            "center",
            "north",
            "east",
            "south",
            "west",
        }
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_raises_when_all_points_fail(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "openweather_api_key", "test-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"message": "server error"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenWeatherCollector(http_client=client)

        with pytest.raises(RuntimeError, match="no observations"):
            await collector.fetch()

        await client.aclose()


class TestOpenWeatherHelpers:
    def test_parse_observation_time_returns_utc_datetime(self) -> None:
        assert parse_observation_time(1776081600) == datetime(
            2026, 4, 13, 12, 0, tzinfo=timezone.utc
        )

    def test_parse_observation_time_rejects_bad_values(self) -> None:
        assert parse_observation_time(None) is None
        assert parse_observation_time("bad") is None
        assert parse_observation_time(0) is None

    def test_extract_precipitation_combines_rain_and_snow(self) -> None:
        assert extract_precipitation({"rain": {"1h": 1.1}, "snow": {"1h": 0.4}}) == 1.5

    def test_weather_query_points_returns_five_target_points(self) -> None:
        points = weather_query_points()

        assert len(points) == 5
        assert all(isinstance(point, WeatherQueryPoint) for point in points)
        assert {point.point_id for point in points} == {
            "center",
            "north",
            "east",
            "south",
            "west",
        }

    def test_field_map_covers_plan_metrics(self) -> None:
        assert FIELD_MAP == {
            ("main", "temp"): ("temperature", "degC"),
            ("main", "humidity"): ("humidity", "percent"),
            ("main", "pressure"): ("pressure", "hPa"),
            ("wind", "speed"): ("wind_speed", "m/s"),
            ("wind", "deg"): ("wind_direction", "degree"),
            ("clouds", "all"): ("cloud_cover", "percent"),
        }
