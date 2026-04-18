from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.purpleair import (
    FIELD_MAP,
    MAX_AGE_SECONDS,
    PurpleAirCollector,
    is_outdoor_sensor,
    parse_last_seen,
)
from app.config import settings


MINIMAL_FIELDS = [
    "sensor_index",
    "last_seen",
    "latitude",
    "longitude",
    "location_type",
    "pm2.5_atm",
]


@pytest.fixture
def collector() -> PurpleAirCollector:
    return PurpleAirCollector()


@pytest.fixture
def sample_api_response() -> dict[str, Any]:
    fields = [
        "sensor_index",
        "name",
        "last_seen",
        "latitude",
        "longitude",
        "location_type",
        "private",
        "pm2.5_atm",
        "pm10.0_atm",
        "humidity",
        "temperature",
        "confidence",
        "channel_flags",
    ]
    return {
        "api_version": "V1.0.0",
        "fields": fields,
        "data": [
            [
                12345,
                "Suwanee PA",
                1776081600,
                34.0515,
                -84.0713,
                0,
                0,
                11.2,
                18.4,
                44.0,
                72.5,
                98,
                0,
            ],
            [
                67890,
                "Nearby PA",
                1776081600,
                34.1515,
                -84.1713,
                0,
                0,
                9.5,
                None,
                41.0,
                None,
                92,
                0,
            ],
        ],
    }


class TestPurpleAirNormalize:
    def test_normalize_maps_metrics(
        self, collector: PurpleAirCollector, sample_api_response: dict[str, Any]
    ) -> None:
        points = collector.normalize(sample_api_response)

        assert len(points) == 6
        assert {point.metric for point in points} == {
            "pm25",
            "pm10",
            "humidity",
            "temperature",
        }

    def test_normalize_sets_units(
        self, collector: PurpleAirCollector, sample_api_response: dict[str, Any]
    ) -> None:
        points = collector.normalize(sample_api_response)
        units_by_metric = {point.metric: point.unit for point in points}

        assert units_by_metric["pm25"] == "ug/m3"
        assert units_by_metric["pm10"] == "ug/m3"
        assert units_by_metric["humidity"] == "percent"
        assert units_by_metric["temperature"] == "degF"

    def test_normalize_sets_source_and_entity_id(
        self, collector: PurpleAirCollector, sample_api_response: dict[str, Any]
    ) -> None:
        points = collector.normalize(sample_api_response)

        assert all(point.source == "purpleair" for point in points)
        assert {point.source_entity_id for point in points} == {"12345", "67890"}

    def test_normalize_uses_last_seen_timestamp(
        self, collector: PurpleAirCollector, sample_api_response: dict[str, Any]
    ) -> None:
        points = collector.normalize(sample_api_response)

        assert all(
            point.timestamp == datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc)
            for point in points
        )

    def test_normalize_preserves_raw_json(
        self, collector: PurpleAirCollector, sample_api_response: dict[str, Any]
    ) -> None:
        points = collector.normalize(sample_api_response)

        assert points[0].raw_json is not None
        assert points[0].raw_json["name"] == "Suwanee PA"
        assert points[0].raw_json["confidence"] == 98

    def test_normalize_skips_indoor_sensor(self, collector: PurpleAirCollector) -> None:
        raw = {
            "fields": MINIMAL_FIELDS,
            "data": [[12345, 1776081600, 34.0515, -84.0713, 1, 11.2]],
        }

        assert collector.normalize(raw) == []

    def test_normalize_skips_sensor_outside_target_radius(
        self, collector: PurpleAirCollector
    ) -> None:
        raw = {
            "fields": MINIMAL_FIELDS,
            "data": [[12345, 1776081600, 36.0, -84.0713, 0, 11.2]],
        }

        assert collector.normalize(raw) == []

    def test_normalize_skips_bad_timestamp(self, collector: PurpleAirCollector) -> None:
        raw = {
            "fields": MINIMAL_FIELDS,
            "data": [[12345, "not-a-timestamp", 34.0515, -84.0713, 0, 11.2]],
        }

        assert collector.normalize(raw) == []

    def test_normalize_skips_missing_metric_value(
        self, collector: PurpleAirCollector
    ) -> None:
        raw = {
            "fields": MINIMAL_FIELDS,
            "data": [[12345, 1776081600, 34.0515, -84.0713, 0, None]],
        }

        assert collector.normalize(raw) == []


class TestPurpleAirFetch:
    @pytest.mark.asyncio
    async def test_fetch_uses_bbox_fields_and_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "purpleair_api_key", "test-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/v1/sensors"
            assert request.headers["X-API-Key"] == "test-key"
            assert request.url.params["location_type"] == "0"
            assert request.url.params["max_age"] == str(MAX_AGE_SECONDS)
            assert "pm2.5_atm" in request.url.params["fields"]
            assert "pm10.0_atm" in request.url.params["fields"]
            assert "nwlng" in request.url.params
            assert "nwlat" in request.url.params
            assert "selng" in request.url.params
            assert "selat" in request.url.params
            return httpx.Response(200, json={"fields": ["sensor_index"], "data": []})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = PurpleAirCollector(http_client=client)

        raw = await collector.fetch()

        assert raw == {"fields": ["sensor_index"], "data": []}
        await client.aclose()


class TestPurpleAirHelpers:
    def test_parse_last_seen_returns_utc_datetime(self) -> None:
        assert parse_last_seen(1776081600) == datetime(
            2026, 4, 13, 12, 0, tzinfo=timezone.utc
        )

    def test_parse_last_seen_rejects_bad_values(self) -> None:
        assert parse_last_seen(None) is None
        assert parse_last_seen("not-a-timestamp") is None
        assert parse_last_seen(0) is None

    def test_is_outdoor_sensor(self) -> None:
        assert is_outdoor_sensor(0) is True
        assert is_outdoor_sensor("0") is True
        assert is_outdoor_sensor(1) is False

    def test_field_map_covers_plan_metrics(self) -> None:
        assert FIELD_MAP == {
            "pm2.5_atm": ("pm25", "ug/m3"),
            "pm10.0_atm": ("pm10", "ug/m3"),
            "humidity": ("humidity", "percent"),
            "temperature": ("temperature", "degF"),
        }
