from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.usgs_water import (
    PARAMETER_MAP,
    USGSWaterCollector,
    parse_usgs_datetime,
    usgs_bbox_param,
)


@pytest.fixture
def collector() -> USGSWaterCollector:
    return USGSWaterCollector()


def make_series(
    *,
    parameter_code: str = "00060",
    site_code: str = "02335000",
    lat: float = 34.05,
    lon: float = -84.07,
    values: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "sourceInfo": {
            "siteName": "CHATTAHOOCHEE RIVER",
            "siteCode": [{"value": site_code, "network": "NWIS", "agencyCode": "USGS"}],
            "geoLocation": {"geogLocation": {"latitude": lat, "longitude": lon}},
        },
        "variable": {
            "variableCode": [{"value": parameter_code}],
            "variableName": "Discharge, ft3/s",
            "unit": {"unitCode": "ft3/s"},
        },
        "values": [
            {
                "value": values
                or [
                    {
                        "value": "1240",
                        "qualifiers": ["P"],
                        "dateTime": "2026-04-30T12:00:00.000-04:00",
                    }
                ]
            }
        ],
    }


def make_payload(*series: dict[str, Any]) -> dict[str, Any]:
    return {"value": {"timeSeries": list(series)}}


class TestUSGSWaterNormalize:
    def test_normalize_maps_known_parameters(self, collector: USGSWaterCollector) -> None:
        payload = make_payload(
            make_series(parameter_code="00060"),
            make_series(parameter_code="00010"),
        )

        points = collector.normalize(payload)

        assert {point.metric for point in points} == {
            "stream_flow",
            "water_temperature",
        }

    def test_normalize_skips_unknown_parameter(
        self, collector: USGSWaterCollector
    ) -> None:
        payload = make_payload(make_series(parameter_code="99999"))

        assert collector.normalize(payload) == []

    def test_normalize_skips_sentinel_value(self, collector: USGSWaterCollector) -> None:
        payload = make_payload(
            make_series(
                values=[
                    {"value": "-999999", "dateTime": "2026-04-30T12:00:00.000-04:00"}
                ]
            )
        )

        assert collector.normalize(payload) == []

    def test_normalize_filters_outside_target_radius(
        self, collector: USGSWaterCollector
    ) -> None:
        payload = make_payload(make_series(lat=10.0, lon=10.0))

        assert collector.normalize(payload) == []

    def test_normalize_sets_source_and_entity(
        self, collector: USGSWaterCollector
    ) -> None:
        payload = make_payload(make_series(site_code="02335000"))

        points = collector.normalize(payload)

        assert points[0].source == "usgs_water"
        assert points[0].source_entity_id == "02335000"
        assert points[0].unit == "ft3/s"
        assert points[0].value == 1240.0

    def test_normalize_preserves_timezone_in_timestamp(
        self, collector: USGSWaterCollector
    ) -> None:
        payload = make_payload(make_series())

        points = collector.normalize(payload)

        assert points[0].timestamp.tzinfo is not None


class TestUSGSWaterFetch:
    @pytest.mark.asyncio
    async def test_fetch_queries_target_bbox(self) -> None:
        seen_params: dict[str, str] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.items():
                seen_params[key] = value
            return httpx.Response(200, json=make_payload(make_series()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = USGSWaterCollector(http_client=client)

        raw = await collector.fetch()

        assert seen_params["format"] == "json"
        assert seen_params["bBox"] == usgs_bbox_param()
        assert seen_params["siteStatus"] == "active"
        assert "00060" in seen_params["parameterCd"]
        assert raw["value"]["timeSeries"][0]["sourceInfo"]["siteName"]
        await client.aclose()


class TestUSGSWaterHelpers:
    def test_parameter_map_covers_core_metrics(self) -> None:
        assert PARAMETER_MAP["00060"] == ("stream_flow", "ft3/s")
        assert PARAMETER_MAP["00010"] == ("water_temperature", "degC")
        assert PARAMETER_MAP["00095"] == ("specific_conductance", "uS/cm")

    def test_parse_usgs_datetime_handles_iso_offset(self) -> None:
        result = parse_usgs_datetime("2026-04-30T12:00:00.000-04:00")

        assert result is not None
        assert result.tzinfo is not None

    def test_parse_usgs_datetime_returns_none_for_garbage(self) -> None:
        assert parse_usgs_datetime("not-a-date") is None
        assert parse_usgs_datetime(None) is None
