from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from app.collectors.geo import BoundingBox
from app.collectors.sentinel5p import (
    COLUMN_PRODUCTS,
    MIN_VALID_PIXELS,
    PRODUCT_QA_THRESHOLD,
    PRODUCT_TYPE_MAP,
    Sentinel5PCollector,
    extract_product_code,
    fetch_access_token,
    filter_and_mean,
    granule_download_url,
    odata_filter,
    parse_iso_datetime,
)
from app.config import settings


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


class TestSentinel5PColumnNormalize:
    def test_emits_column_metric_when_extracted_present(
        self, collector: Sentinel5PCollector
    ) -> None:
        raw = {
            **make_payload(make_record(product_id="no2-1")),
            "extracted_columns": {"no2-1": 5.4e-5},
        }

        points = collector.normalize(raw)

        column = next(p for p in points if p.metric == "s5p_no2_column")
        assert column.value == pytest.approx(5.4e-5)
        assert column.unit == "mol/m^2"
        assert column.source_entity_id == "no2-1"

    def test_no_column_metric_when_extracted_absent(
        self, collector: Sentinel5PCollector
    ) -> None:
        points = collector.normalize(make_payload(make_record(product_id="no2-1")))

        assert all(not p.metric.endswith("_column") for p in points)

    def test_no_column_metric_for_unmapped_column_product(
        self, collector: Sentinel5PCollector
    ) -> None:
        # O3 is in PRODUCT_TYPE_MAP (catalog-mapped) but NOT in COLUMN_PRODUCTS
        raw = {
            **make_payload(
                make_record(
                    product_id="o3-1",
                    name="S5P_NRTI_L2__O3_____20260430T120000_xx.nc",
                    cloud_cover=None,
                )
            ),
            "extracted_columns": {"o3-1": 9.9e-3},
        }

        points = collector.normalize(raw)

        assert {p.metric for p in points} == {"s5p_o3_granule_available"}

    def test_column_metric_does_not_replace_catalog_metrics(
        self, collector: Sentinel5PCollector
    ) -> None:
        raw = {
            **make_payload(make_record(product_id="no2-1")),
            "extracted_columns": {"no2-1": 5.4e-5},
        }

        points = collector.normalize(raw)

        assert {p.metric for p in points} == {
            "s5p_no2_granule_available",
            "s5p_no2_cloud_cover",
            "s5p_no2_column",
        }


class TestFilterAndMean:
    def _bbox(self) -> BoundingBox:
        return BoundingBox(min_lat=29.0, max_lat=30.5, min_lon=-96.0, max_lon=-94.5)

    def test_returns_mean_when_all_pixels_pass(self) -> None:
        result = filter_and_mean(
            values=[1.0, 2.0, 3.0] + [4.0] * 10,
            qa=[0.9] * 13,
            lats=[29.7] * 13,
            lons=[-95.3] * 13,
            bbox=self._bbox(),
            qa_threshold=0.75,
            min_pixels=10,
        )

        assert result == pytest.approx((1 + 2 + 3 + 40) / 13)

    def test_filters_by_qa_threshold(self) -> None:
        result = filter_and_mean(
            values=[1.0] * 10 + [99.0] * 5,
            qa=[0.9] * 10 + [0.3] * 5,
            lats=[29.7] * 15,
            lons=[-95.3] * 15,
            bbox=self._bbox(),
            qa_threshold=0.75,
            min_pixels=5,
        )

        assert result == pytest.approx(1.0)

    def test_filters_pixels_outside_bbox(self) -> None:
        result = filter_and_mean(
            values=[1.0] * 10 + [99.0] * 5,
            qa=[0.9] * 15,
            lats=[29.7] * 10 + [40.0] * 5,
            lons=[-95.3] * 15,
            bbox=self._bbox(),
            qa_threshold=0.75,
            min_pixels=5,
        )

        assert result == pytest.approx(1.0)

    def test_filters_nan_values(self) -> None:
        result = filter_and_mean(
            values=[1.0] * 10 + [float("nan")] * 5,
            qa=[0.9] * 15,
            lats=[29.7] * 15,
            lons=[-95.3] * 15,
            bbox=self._bbox(),
            qa_threshold=0.75,
            min_pixels=5,
        )

        assert result == pytest.approx(1.0)

    def test_returns_none_when_below_min_pixels(self) -> None:
        result = filter_and_mean(
            values=[1.0] * 3,
            qa=[0.9] * 3,
            lats=[29.7] * 3,
            lons=[-95.3] * 3,
            bbox=self._bbox(),
            qa_threshold=0.75,
            min_pixels=10,
        )

        assert result is None

    def test_returns_none_for_empty_inputs(self) -> None:
        assert (
            filter_and_mean(
                values=[],
                qa=[],
                lats=[],
                lons=[],
                bbox=self._bbox(),
                qa_threshold=0.75,
            )
            is None
        )

    def test_returns_none_for_length_mismatch(self) -> None:
        assert (
            filter_and_mean(
                values=[1.0, 2.0],
                qa=[0.9],
                lats=[29.7, 29.8],
                lons=[-95.3, -95.4],
                bbox=self._bbox(),
                qa_threshold=0.75,
            )
            is None
        )


class TestFetchAccessToken:
    @pytest.mark.asyncio
    async def test_returns_access_token_on_success(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.method == "POST"
            assert "openid-connect/token" in str(request.url)
            body = request.content.decode()
            assert "grant_type=password" in body
            assert "client_id=cdse-public" in body
            assert "username=alice" in body
            return httpx.Response(
                200, json={"access_token": "tok-123", "expires_in": 600}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            token = await fetch_access_token(client, "alice", "secret")
        finally:
            await client.aclose()

        assert token == "tok-123"

    @pytest.mark.asyncio
    async def test_raises_when_response_missing_access_token(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"foo": "bar"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(RuntimeError, match="missing access_token"):
                await fetch_access_token(client, "alice", "secret")
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_raises_for_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": "invalid_grant"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_access_token(client, "alice", "secret")
        finally:
            await client.aclose()


class TestSentinel5PFetch:
    @pytest.mark.asyncio
    async def test_fetch_sends_odata_filter(self) -> None:
        seen_params: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            for key, value in request.url.params.items():
                seen_params[key] = value
            return httpx.Response(200, json=make_payload(make_record()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        try:
            raw = await collector.fetch()
        finally:
            await client.aclose()

        assert "Collection/Name eq 'SENTINEL-5P'" in seen_params["$filter"]
        assert "OData.CSC.Intersects" in seen_params["$filter"]
        assert seen_params["$expand"] == "Attributes"
        assert raw["value"][0]["Id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_catalog_only_mode_when_cdse_creds_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "")
        monkeypatch.setattr(settings, "cdse_password", "")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=make_payload(make_record()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        try:
            raw = await collector.fetch()
        finally:
            await client.aclose()

        assert "extracted_columns" not in raw

    @pytest.mark.asyncio
    async def test_falls_back_to_catalog_when_token_fetch_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                return httpx.Response(401, json={"error": "invalid_grant"})
            return httpx.Response(200, json=make_payload(make_record()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        try:
            raw = await collector.fetch()
        finally:
            await client.aclose()

        assert "extracted_columns" not in raw
        assert raw["value"][0]["Id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_includes_extracted_columns_when_creds_succeed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                return httpx.Response(
                    200, json={"access_token": "tok-123", "expires_in": 600}
                )
            return httpx.Response(200, json=make_payload(make_record(product_id="no2-1")))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        async def fake_extract(
            self,
            client_arg: httpx.AsyncClient,
            product_id: str,
            product_code: str,
            token: str,
            bbox: BoundingBox,
        ) -> float | None:
            assert token == "tok-123"
            assert product_code == "NO2"
            return 4.2e-5

        with patch.object(
            Sentinel5PCollector, "_download_and_extract", fake_extract
        ):
            try:
                raw = await collector.fetch()
            finally:
                await client.aclose()

        assert raw["extracted_columns"] == {"no2-1": pytest.approx(4.2e-5)}

    @pytest.mark.asyncio
    async def test_skips_extraction_for_unmapped_column_products(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                return httpx.Response(
                    200, json={"access_token": "tok-123", "expires_in": 600}
                )
            return httpx.Response(
                200,
                json=make_payload(
                    make_record(
                        product_id="o3-1",
                        name="S5P_NRTI_L2__O3_____20260430T120000_xx.nc",
                    )
                ),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        calls: list[str] = []

        async def fake_extract(
            self,
            client_arg: httpx.AsyncClient,
            product_id: str,
            product_code: str,
            token: str,
            bbox: BoundingBox,
        ) -> float | None:
            calls.append(product_id)
            return 1.0

        with patch.object(
            Sentinel5PCollector, "_download_and_extract", fake_extract
        ):
            try:
                raw = await collector.fetch()
            finally:
                await client.aclose()

        assert calls == []
        assert raw.get("extracted_columns") == {}


class TestSentinel5PHelpers:
    def test_product_type_map_covers_pollutants(self) -> None:
        assert PRODUCT_TYPE_MAP["NO2"] == "s5p_no2"
        assert PRODUCT_TYPE_MAP["SO2"] == "s5p_so2"
        assert PRODUCT_TYPE_MAP["CO"] == "s5p_co"

    def test_column_products_match_qa_thresholds(self) -> None:
        assert set(COLUMN_PRODUCTS) == set(PRODUCT_QA_THRESHOLD)

    def test_no2_qa_threshold_is_strictest(self) -> None:
        assert PRODUCT_QA_THRESHOLD["NO2"] == 0.75
        for code in ("SO2", "CO", "HCHO"):
            assert PRODUCT_QA_THRESHOLD[code] == 0.5

    def test_min_valid_pixels_is_reasonable_floor(self) -> None:
        assert MIN_VALID_PIXELS >= 1

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

    def test_granule_download_url_targets_value_endpoint(self) -> None:
        url = granule_download_url("abc-123")

        assert url.endswith("Products(abc-123)/$value")
