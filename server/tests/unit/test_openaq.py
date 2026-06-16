import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pytest

from app.collectors.openaq import (
    PARAMETER_MAP,
    SENSORS_LIMIT,
    OpenAQCollector,
    clear_locations_cache,
    location_within_target_radius,
    normalize_openaq_unit,
    parse_openaq_datetime,
)
from app.collectors.ratelimit import AsyncRateLimiter
from app.config import settings


@pytest.fixture(autouse=True)
def _fresh_locations_cache():
    clear_locations_cache()
    yield
    clear_locations_cache()


def fast_limiter() -> AsyncRateLimiter:
    return AsyncRateLimiter(6_000_000)


@pytest.fixture
def collector() -> OpenAQCollector:
    return OpenAQCollector()


def make_location(location_id: int = 100) -> dict[str, Any]:
    return {
        "id": location_id,
        "name": "Houston Monitor",
        "coordinates": {"latitude": 29.7604, "longitude": -95.3698},
    }


def make_sensor(
    sensor_id: int,
    parameter: str,
    *,
    unit: str = "\u00b5g/m\u00b3",
    value: float | None = 10.5,
    timestamp: str | None = None,
) -> dict[str, Any]:
    # A healthy station's /latest is recent; default to "just now" so the age
    # guard keeps it. Tests covering staleness pass an explicit old timestamp.
    if timestamp is None:
        timestamp = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    latest = None
    if value is not None:
        latest = {
            "datetime": {"utc": timestamp, "local": "2026-04-13T08:00:00-04:00"},
            "value": value,
            "coordinates": {"latitude": 29.7604, "longitude": -95.3698},
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
        recent = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(
            minutes=1
        )
        ts = recent.isoformat().replace("+00:00", "Z")
        points = collector.normalize(make_raw([make_sensor(1, "pm25", timestamp=ts)]))

        assert points[0].timestamp == recent

    def test_normalize_preserves_raw_json(self, collector: OpenAQCollector) -> None:
        points = collector.normalize(make_raw([make_sensor(1, "pm25")]))

        assert points[0].raw_json is not None
        assert points[0].raw_json["location"]["name"] == "Houston Monitor"
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

    def test_normalize_drops_stale_reading(
        self, collector: OpenAQCollector
    ) -> None:
        # An offline station's /latest is months old; emitting it as a fresh
        # DataPoint every run feeds detection a stale-but-current-looking value
        # and pollutes the series date range.
        sensor = make_sensor(1, "pm25", timestamp="2020-01-01T00:00:00Z")

        assert collector.normalize(make_raw([sensor])) == []

    def test_normalize_keeps_recent_reading(
        self, collector: OpenAQCollector
    ) -> None:
        recent = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        points = collector.normalize(
            make_raw([make_sensor(1, "pm25", timestamp=recent)])
        )

        assert len(points) == 1

    def test_normalize_skips_sensor_with_bad_coordinates(
        self, collector: OpenAQCollector
    ) -> None:
        # One station reporting non-numeric coordinates must be skipped, not
        # abort the whole normalize pass with an uncaught float() error.
        bad = make_sensor(1, "pm25")
        bad["latest"]["coordinates"] = {"latitude": "N/A", "longitude": "N/A"}
        good = make_sensor(2, "pm25")

        points = collector.normalize(make_raw([bad, good]))

        assert len(points) == 1


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
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        raw = await collector.fetch()

        assert seen_paths == [
            "/v3/locations",
            f"/v3/locations/{location['id']}/sensors",
        ]
        assert raw["locations"] == [location]
        assert raw["sensors_by_location_id"][str(location["id"])] == [sensor]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_sensors_request_sets_explicit_limit(self, monkeypatch) -> None:
        # Don't rely on the API's default page size for a location's sensors;
        # request a full page explicitly.
        monkeypatch.setattr(settings, "openaq_api_key", "test-key")
        location = make_location()
        sensor = make_sensor(1, "pm25")
        seen_limit: list[str | None] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v3/locations":
                return httpx.Response(
                    200, json={"meta": {"found": 1}, "results": [location]}
                )
            if request.url.path == f"/v3/locations/{location['id']}/sensors":
                seen_limit.append(request.url.params.get("limit"))
                return httpx.Response(
                    200, json={"meta": {"found": 1}, "results": [sensor]}
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        await collector.fetch()

        assert seen_limit == [str(SENSORS_LIMIT)]
        await client.aclose()

    @pytest.mark.asyncio
    async def test_second_fetch_reuses_cached_locations(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "openaq_api_key", "test-key")
        location = make_location()
        locations_calls = 0
        sensors_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal locations_calls, sensors_calls
            if request.url.path == "/v3/locations":
                locations_calls += 1
                return httpx.Response(
                    200, json={"meta": {"found": 1}, "results": [location]}
                )
            sensors_calls += 1
            return httpx.Response(
                200, json={"meta": {"found": 1}, "results": [make_sensor(1, "pm25")]}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        await collector.fetch()
        await collector.fetch()

        # The topology is cached for a day; only the latest values re-poll.
        assert locations_calls == 1
        assert sensors_calls == 2
        await client.aclose()

    @pytest.mark.asyncio
    async def test_expired_cache_refetches_locations(self, monkeypatch) -> None:
        import app.collectors.openaq as openaq_module

        monkeypatch.setattr(settings, "openaq_api_key", "test-key")
        monkeypatch.setattr(openaq_module, "LOCATIONS_CACHE_TTL_S", 0.0)
        location = make_location()
        locations_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal locations_calls
            if request.url.path == "/v3/locations":
                locations_calls += 1
                return httpx.Response(
                    200, json={"meta": {"found": 1}, "results": [location]}
                )
            return httpx.Response(200, json={"meta": {"found": 0}, "results": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        await collector.fetch()
        await collector.fetch()

        assert locations_calls == 2
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_logs_clear_message_on_rejected_key(
        self, monkeypatch, caplog
    ) -> None:
        # A rejected key returns 401 on the first call; surface a clear log
        # instead of an opaque HTTPStatusError, and still propagate it.
        monkeypatch.setattr(settings, "openaq_api_key", "dead-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Invalid API key"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        with caplog.at_level(logging.ERROR, logger="app.collectors.openaq"):
            with pytest.raises(httpx.HTTPStatusError):
                await collector.fetch()

        assert any("key rejected" in r.getMessage().lower() for r in caplog.records)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_discovery_401_falls_back_to_cached_locations(
        self, monkeypatch, caplog
    ) -> None:
        # Once the topology is discovered, a later 401 on /v3/locations must not
        # take the source dark: reuse the cached locations and keep polling
        # sensors. TTL is forced to 0 so the second fetch re-attempts discovery
        # (and fails) instead of serving the still-fresh cache.
        import app.collectors.openaq as openaq_module

        monkeypatch.setattr(settings, "openaq_api_key", "test-key")
        monkeypatch.setattr(openaq_module, "LOCATIONS_CACHE_TTL_S", 0.0)
        location = make_location()
        locations_calls = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal locations_calls
            if request.url.path == "/v3/locations":
                locations_calls += 1
                if locations_calls == 1:
                    return httpx.Response(
                        200, json={"meta": {"found": 1}, "results": [location]}
                    )
                return httpx.Response(401, json={"message": "Invalid API key"})
            return httpx.Response(
                200, json={"meta": {"found": 1}, "results": [make_sensor(1, "pm25")]}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        await collector.fetch()  # warms the cache
        with caplog.at_level(logging.WARNING, logger="app.collectors.openaq"):
            raw = await collector.fetch()  # discovery 401 -> fall back to cache

        assert locations_calls == 2  # the failed re-discovery was attempted
        assert raw["locations"] == [location]  # served from the cache
        assert raw["sensors_by_location_id"][str(location["id"])]  # sensors walked
        assert any("reusing" in r.getMessage().lower() for r in caplog.records)
        await client.aclose()

    @pytest.mark.asyncio
    async def test_discovery_401_cold_cache_still_raises(self, monkeypatch) -> None:
        # With nothing cached to fall back on, a rejected key must fail the run
        # loudly rather than inventing an empty location set.
        monkeypatch.setattr(settings, "openaq_api_key", "dead-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"message": "Invalid API key"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = OpenAQCollector(http_client=client, rate_limiter=fast_limiter())

        with pytest.raises(httpx.HTTPStatusError):
            await collector.fetch()
        await client.aclose()


class TestOpenAQHelpers:
    def test_location_within_radius_skips_bad_coordinates(self) -> None:
        # Malformed coordinates must not raise an uncaught float() error that
        # aborts the whole location scan.
        bad = {"coordinates": {"latitude": "N/A", "longitude": "N/A"}}

        assert location_within_target_radius(bad) is False

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
