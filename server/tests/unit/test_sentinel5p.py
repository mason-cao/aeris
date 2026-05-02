from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.sentinel5p import (
    PRODUCT_TYPE_MAP,
    Sentinel5PCollector,
    extract_product_code,
    odata_filter,
    parse_iso_datetime,
)


@pytest.fixture
def collector() -> Sentinel5PCollector:
    return Sentinel5PCollector()


def make_record(
    *,
    product_id: str = "abc-123",
    name: str = "S5P_NRTI_L2__NO2____20260430T120000_20260430T123000_99999_03_020401_20260430T130000.nc",
    start: str = "2026-04-30T12:00:00.000Z",
    cloud_cover: float | None = 12.5,
) -> dict[str, Any]:
    attributes = []
    if cloud_cover is not None:
        attributes.append({"Name": "cloudCover", "Value": cloud_cover})
    return {
        "Id": product_id,
        "Name": name,
        "ContentDate": {"Start": start, "End": start},
        "Footprint": "geography'SRID=4326;POLYGON(...)'",
        "Attributes": attributes,
        "Online": True,
    }


def make_payload(*records: dict[str, Any]) -> dict[str, Any]:
    return {"value": list(records)}


class TestSentinel5PNormalize:
    def test_normalize_emits_availability_and_cloud_cover(
        self, collector: Sentinel5PCollector
    ) -> None:
        points = collector.normalize(make_payload(make_record()))

        metrics = {point.metric for point in points}
        assert metrics == {"s5p_no2_granule_available", "s5p_no2_cloud_cover"}

    def test_normalize_omits_cloud_cover_when_missing(
        self, collector: Sentinel5PCollector
    ) -> None:
        points = collector.normalize(make_payload(make_record(cloud_cover=None)))

        assert {point.metric for point in points} == {"s5p_no2_granule_available"}

    def test_normalize_skips_unmapped_product_type(
        self, collector: Sentinel5PCollector
    ) -> None:
        payload = make_payload(
            make_record(name="S5P_NRTI_L2__OTHER__20260430T120000_xx.nc")
        )

        assert collector.normalize(payload) == []

    def test_normalize_uses_target_coordinates(
        self, collector: Sentinel5PCollector
    ) -> None:
        from app.config import settings

        points = collector.normalize(make_payload(make_record()))

        for point in points:
            assert point.lat == settings.aeris_target_lat
            assert point.lon == settings.aeris_target_lon

    def test_normalize_sets_source_and_entity(
        self, collector: Sentinel5PCollector
    ) -> None:
        points = collector.normalize(make_payload(make_record(product_id="abc-123")))

        assert all(point.source == "sentinel5p" for point in points)
        assert {point.source_entity_id for point in points} == {"abc-123"}

    def test_normalize_parses_iso_timestamp(
        self, collector: Sentinel5PCollector
    ) -> None:
        points = collector.normalize(make_payload(make_record()))

        assert points[0].timestamp == datetime(
            2026, 4, 30, 12, tzinfo=timezone.utc
        )

    def test_normalize_handles_multiple_product_types(
        self, collector: Sentinel5PCollector
    ) -> None:
        payload = make_payload(
            make_record(
                product_id="no2-1",
                name="S5P_NRTI_L2__NO2____20260430T120000_xx.nc",
                cloud_cover=None,
            ),
            make_record(
                product_id="so2-1",
                name="S5P_NRTI_L2__SO2____20260430T120000_xx.nc",
                cloud_cover=None,
            ),
            make_record(
                product_id="co-1",
                name="S5P_NRTI_L2__CO_____20260430T120000_xx.nc",
                cloud_cover=None,
            ),
        )

        points = collector.normalize(payload)

        assert {point.metric for point in points} == {
            "s5p_no2_granule_available",
            "s5p_so2_granule_available",
            "s5p_co_granule_available",
        }


class TestSentinel5PFetch:
    @pytest.mark.asyncio
    async def test_fetch_sends_odata_filter(self) -> None:
        seen_params: dict[str, str] = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.items():
                seen_params[key] = value
            return httpx.Response(200, json=make_payload(make_record()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        raw = await collector.fetch()

        assert "Collection/Name eq 'SENTINEL-5P'" in seen_params["$filter"]
        assert "OData.CSC.Intersects" in seen_params["$filter"]
        assert seen_params["$expand"] == "Attributes"
        assert raw["value"][0]["Id"] == "abc-123"
        await client.aclose()


class TestSentinel5PHelpers:
    def test_product_type_map_covers_pollutants(self) -> None:
        assert PRODUCT_TYPE_MAP["NO2"] == "s5p_no2"
        assert PRODUCT_TYPE_MAP["SO2"] == "s5p_so2"
        assert PRODUCT_TYPE_MAP["CO"] == "s5p_co"

    def test_extract_product_code_handles_padded_names(self) -> None:
        assert (
            extract_product_code(
                "S5P_NRTI_L2__NO2____20260430T120000_xx.nc"
            )
            == "NO2"
        )
        assert (
            extract_product_code(
                "S5P_OFFL_L2__CO_____20260430T120000_xx.nc"
            )
            == "CO"
        )

    def test_extract_product_code_returns_none_for_garbage(self) -> None:
        assert extract_product_code("not-a-product-name") is None
        assert extract_product_code(None) is None

    def test_parse_iso_datetime_supports_zulu_suffix(self) -> None:
        assert parse_iso_datetime("2026-04-30T12:00:00.000Z") == datetime(
            2026, 4, 30, 12, tzinfo=timezone.utc
        )

    def test_odata_filter_includes_polygon_and_lookback(self) -> None:
        now = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

        text = odata_filter(now=now)

        assert "POLYGON" in text
        assert "2026-04-28T12:00:00.000Z" in text
