import math
import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import httpx
import numpy as np
import pytest

from app.collectors.geo import BoundingBox
from app.collectors.sentinel5p import (
    COLUMN_PRODUCTS,
    MIN_VALID_PIXELS,
    PRODUCT_QA_THRESHOLD,
    PRODUCT_TYPE_MAP,
    Sentinel5PCollector,
    _load_granule_arrays,
    _normalize_qa,
    extract_product_code,
    fetch_access_token,
    filter_and_mean,
    granule_download_url,
    odata_filter,
    parse_iso_datetime,
)
from app.config import settings
from app.db.models import DataPoint


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

    def test_accepts_numpy_array_inputs(self) -> None:
        # _load_granule_arrays now hands numpy arrays straight through; the
        # masked mean over them must match the list path.
        result = filter_and_mean(
            values=np.array([1.0, 2.0, 3.0] + [4.0] * 10),
            qa=np.array([0.9] * 13),
            lats=np.array([29.7] * 13),
            lons=np.array([-95.3] * 13),
            bbox=self._bbox(),
            qa_threshold=0.75,
            min_pixels=10,
        )

        assert result == pytest.approx((1 + 2 + 3 + 40) / 13)

    def test_filters_nan_qa_pixels(self) -> None:
        result = filter_and_mean(
            values=[1.0] * 10 + [5.0] * 5,
            qa=[0.9] * 10 + [float("nan")] * 5,
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


class TestNormalizeQa:
    """qa_value is a 0-100 quality percentage with a 0.01 scale_factor; decoded
    it should be the [0, 1] fraction filter_and_mean compares against the
    qa_threshold. A granule whose values arrive unscaled (0-100) must be
    normalized down, or every low-quality pixel would clear a fractional
    threshold (10 >= 0.75) and pollute the column mean.
    """

    def test_fractional_qa_is_unchanged(self) -> None:
        assert _normalize_qa(np.array([0.0, 0.5, 0.75, 1.0])).tolist() == [
            0.0, 0.5, 0.75, 1.0
        ]

    def test_percentage_qa_is_scaled_to_fraction(self) -> None:
        assert _normalize_qa(np.array([0.0, 50.0, 75.0, 100.0])).tolist() == [
            0.0, 0.5, 0.75, 1.0
        ]

    def test_nan_does_not_defeat_the_scale_decision(self) -> None:
        # A NaN max would make `max() > 1` false and skip scaling; the decision
        # must use the finite maximum (100 here) so the array is still scaled.
        out = _normalize_qa(np.array([np.nan, 50.0, 100.0]))
        assert math.isnan(out[0])
        assert out[1:].tolist() == [0.5, 1.0]

    def test_empty_array_returns_empty(self) -> None:
        assert _normalize_qa(np.array([])).tolist() == []


def _write_granule(
    path: str,
    *,
    variable: str = "nitrogendioxide_tropospheric_column",
    column: list[list[float]],
    qa_packed: list[list[int]],
    lat: list[list[float]],
    lon: list[list[float]],
) -> None:
    """Write a minimal TROPOMI-shaped L2 granule (PRODUCT group, packed qa).

    Mirrors the real file the extractor reads: a PRODUCT group whose qa_value
    is a uint8 percentage with a 0.01 scale_factor, and a column variable with
    a _FillValue. xarray's mask_and_scale must decode both.
    """
    import netCDF4

    fill = np.float32(-1.0e30)
    with netCDF4.Dataset(path, "w") as ds:
        product = ds.createGroup("PRODUCT")
        product.createDimension("time", 1)
        product.createDimension("scanline", 2)
        product.createDimension("ground_pixel", 2)
        dims = ("time", "scanline", "ground_pixel")

        lat_v = product.createVariable("latitude", "f4", dims)
        lat_v[:] = np.array([lat], dtype="f4")
        lon_v = product.createVariable("longitude", "f4", dims)
        lon_v[:] = np.array([lon], dtype="f4")

        qa_v = product.createVariable("qa_value", "u1", dims)
        qa_v.scale_factor = np.float64(0.01)
        qa_v.set_auto_maskandscale(False)
        qa_v[:] = np.array([qa_packed], dtype="u1")

        col_v = product.createVariable(variable, "f4", dims, fill_value=fill)
        col_v.set_auto_mask(False)
        col_v[:] = np.array([column], dtype="f4")


class TestLoadGranuleArrays:
    """Parse a real netCDF granule, end to end, the way GRIB parsing does.

    Every other extraction test mocks _download_and_extract, so the real
    xarray decode (group lookup, mask_and_scale, qa normalization, ravel) was
    untested against an actual file.
    """

    def test_decodes_real_netcdf_granule_with_scale_and_fill(self, tmp_path) -> None:
        path = str(tmp_path / "granule.nc")
        fill = -1.0e30
        _write_granule(
            path,
            column=[[4.0e-5, 6.0e-5], [5.0e-5, fill]],
            qa_packed=[[75, 100], [20, 75]],
            lat=[[29.0, 29.5], [30.0, 30.5]],
            lon=[[-95.0, -95.5], [-96.0, -96.5]],
        )

        values, qa, lat, lon = _load_granule_arrays(path, "NO2")

        assert values.shape == (4,)
        # mask_and_scale turns the _FillValue pixel into NaN; the rest decode.
        assert np.isnan(values[3])
        assert values[:3] == pytest.approx([4.0e-5, 6.0e-5, 5.0e-5], rel=1e-4)
        # qa_value: uint8 percentage * 0.01 -> [0, 1] fraction.
        assert qa.tolist() == pytest.approx([0.75, 1.0, 0.2, 0.75])
        assert lat.tolist() == pytest.approx([29.0, 29.5, 30.0, 30.5])
        assert lon.tolist() == pytest.approx([-95.0, -95.5, -96.0, -96.5])

    def test_raises_when_expected_variable_absent(self, tmp_path) -> None:
        # An NO2 granule lacks the SO2 column variable: a clear error, not a
        # KeyError deep in xarray.
        path = str(tmp_path / "granule.nc")
        _write_granule(
            path,
            column=[[4.0e-5, 6.0e-5], [5.0e-5, 5.5e-5]],
            qa_packed=[[75, 100], [20, 75]],
            lat=[[29.0, 29.5], [30.0, 30.5]],
            lon=[[-95.0, -95.5], [-96.0, -96.5]],
        )

        with pytest.raises(RuntimeError, match="missing expected variable"):
            _load_granule_arrays(path, "SO2")


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
    async def test_fetch_catalog_follows_redirect(self) -> None:
        # CDSE can 30x the catalogue endpoint; the shared client defaults to
        # follow_redirects=False, so the catalog GET must opt in per-request
        # or every request silently fails on the redirect.
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "moved=1" not in url:
                return httpx.Response(
                    301,
                    json={"value": [{"Id": "REDIRECT-NOT-FOLLOWED"}]},
                    headers={"location": url + "&moved=1"},
                )
            return httpx.Response(200, json=make_payload(make_record()))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)
        try:
            catalog = await collector._fetch_catalog(client)
        finally:
            await client.aclose()

        assert catalog["value"][0]["Id"] == "abc-123"

    @pytest.mark.asyncio
    async def test_fetch_access_token_follows_redirect(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "moved=1" not in url:
                return httpx.Response(
                    307,
                    json={"access_token": "STALE"},
                    headers={"location": url + "?moved=1"},
                )
            return httpx.Response(200, json={"access_token": "tok-xyz"})

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        try:
            token = await fetch_access_token(client, "alice", "secret")
        finally:
            await client.aclose()

        assert token == "tok-xyz"

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

        text = odata_filter(window_end=now)

        assert "POLYGON" in text
        assert "2026-04-28T12:00:00.000Z" in text

    def test_odata_filter_bounds_window_on_both_sides(self) -> None:
        # Without an upper bound, $orderby desc + $top re-returns the newest
        # granules every window, so a backward backfill never reaches deep
        # history. The window must be closed on both ends.
        end = datetime(2026, 4, 30, 12, tzinfo=timezone.utc)

        text = odata_filter(end)

        assert "ContentDate/Start gt 2026-04-28T12:00:00.000Z" in text
        assert "ContentDate/Start le 2026-04-30T12:00:00.000Z" in text

    def test_granule_download_url_targets_value_endpoint(self) -> None:
        url = granule_download_url("abc-123")

        assert url.endswith("Products(abc-123)/$value")


class _StubStreamResponse:
    """Minimal stand-in for an httpx streaming response.

    httpx rewrites Content-Length to match a provided body, so a real
    MockTransport can't simulate a stream that under-delivers its declared
    length — this stub lets the body and Content-Length diverge.
    """

    def __init__(self, body: bytes, content_length: int | None) -> None:
        self.is_redirect = False
        self.headers: dict[str, str] = {}
        if content_length is not None:
            self.headers["content-length"] = str(content_length)
        self._body = body

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self):
        yield self._body


class _StubStreamClient:
    def __init__(self, response: _StubStreamResponse) -> None:
        self._response = response

    def stream(self, *args, **kwargs):
        response = self._response

        class _Ctx:
            async def __aenter__(self) -> _StubStreamResponse:
                return response

            async def __aexit__(self, *exc) -> bool:
                return False

        return _Ctx()


class TestDownloadGranule:
    @pytest.mark.asyncio
    async def test_follows_redirect_and_reattaches_auth_header(
        self, tmp_path
    ) -> None:
        # The $value endpoint 301s to a different host; httpx strips the
        # Authorization header across hosts, so the collector must follow
        # by hand and re-attach it — this is the bug that 401'd every
        # granule on the first credentialed run (2026-06-12).
        seen: list[tuple[str, str | None]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append((request.url.host, request.headers.get("authorization")))
            if request.url.host == "catalogue.dataspace.copernicus.eu":
                return httpx.Response(
                    301,
                    headers={"location": "https://download.example.eu/p/abc.nc"},
                )
            return httpx.Response(200, content=b"granule-bytes")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)
        path = tmp_path / "granule.nc"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT)
        try:
            await collector._download_granule(client, "abc-123", "tok-1", fd)
        finally:
            await client.aclose()

        assert path.read_bytes() == b"granule-bytes"
        assert seen[0] == ("catalogue.dataspace.copernicus.eu", "Bearer tok-1")
        assert seen[1] == ("download.example.eu", "Bearer tok-1")

    @pytest.mark.asyncio
    async def test_gives_up_after_redirect_limit(self, tmp_path) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                301, headers={"location": str(request.url) + "x"}
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)
        fd = os.open(tmp_path / "granule.nc", os.O_WRONLY | os.O_CREAT)
        try:
            with pytest.raises(RuntimeError, match="redirects"):
                await collector._download_granule(client, "abc-123", "tok", fd)
        finally:
            await client.aclose()

    @pytest.mark.asyncio
    async def test_raises_on_truncated_stream(self, tmp_path) -> None:
        # Server declares 100 bytes but delivers fewer; the partial .nc must
        # be rejected so the granule is re-fetched on the next run instead of
        # yielding a column mean from an incomplete swath.
        client = _StubStreamClient(
            _StubStreamResponse(b"short", content_length=100)
        )
        collector = Sentinel5PCollector(http_client=client)
        fd = os.open(tmp_path / "granule.nc", os.O_WRONLY | os.O_CREAT)

        with pytest.raises(RuntimeError, match="truncated"):
            await collector._download_granule(client, "abc-123", "tok", fd)

    @pytest.mark.asyncio
    async def test_accepts_stream_matching_content_length(
        self, tmp_path
    ) -> None:
        body = b"granule-bytes"
        client = _StubStreamClient(
            _StubStreamResponse(body, content_length=len(body))
        )
        collector = Sentinel5PCollector(http_client=client)
        path = tmp_path / "granule.nc"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT)

        await collector._download_granule(client, "abc-123", "tok", fd)

        assert path.read_bytes() == body

    @pytest.mark.asyncio
    async def test_refreshes_token_and_retries_on_401(
        self, tmp_path, monkeypatch
    ) -> None:
        # The token can expire mid-download (the download timeout can exceed
        # the token's remaining lifetime). A 401 must trigger one refresh and
        # retry, not silently drop the granule.
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")
        refreshes = {"n": 0}

        async def fake_refresh(client, user, password):
            refreshes["n"] += 1
            return "fresh-token"

        monkeypatch.setattr(
            "app.collectors.sentinel5p.fetch_access_token", fake_refresh
        )
        seen_auth: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_auth.append(request.headers.get("authorization"))
            if len(seen_auth) == 1:
                return httpx.Response(401)
            return httpx.Response(200, content=b"granule-bytes")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)
        path = tmp_path / "granule.nc"
        fd = os.open(path, os.O_WRONLY | os.O_CREAT)
        try:
            await collector._download_granule(client, "abc-123", "stale-token", fd)
        finally:
            await client.aclose()

        assert refreshes["n"] == 1
        assert seen_auth == ["Bearer stale-token", "Bearer fresh-token"]
        assert path.read_bytes() == b"granule-bytes"


class TestExtractColumnsTokenRefresh:
    @pytest.mark.asyncio
    async def test_refreshes_token_when_stale(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")
        # Age limit 0 forces a refresh before every granule, standing in for
        # downloads that outlive the ~10-minute CDSE token.
        monkeypatch.setattr("app.collectors.sentinel5p.TOKEN_MAX_AGE_S", 0.0)

        token_requests: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                token_requests.append(1)
                return httpx.Response(
                    200, json={"access_token": f"tok-{len(token_requests)}"}
                )
            return httpx.Response(404)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)
        catalog = make_payload(
            make_record(product_id="no2-1"),
            make_record(product_id="no2-2"),
        )

        used_tokens: list[str] = []

        async def fake_extract(
            self,
            client_arg: httpx.AsyncClient,
            product_id: str,
            product_code: str,
            token: str,
            bbox: BoundingBox,
        ) -> float | None:
            used_tokens.append(token)
            return 1.0e-4

        with patch.object(
            Sentinel5PCollector, "_download_and_extract", fake_extract
        ):
            try:
                extracted = await collector._extract_columns(
                    client, catalog, "tok-stale"
                )
            finally:
                await client.aclose()

        assert len(extracted) == 2
        assert used_tokens == ["tok-1", "tok-2"]


class TestCollectSkipsStoredColumns:
    @pytest.mark.asyncio
    async def test_collect_does_not_redownload_stored_granules(
        self, db_session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(settings, "cdse_username", "alice")
        monkeypatch.setattr(settings, "cdse_password", "secret")

        db_session.add(
            DataPoint(
                timestamp=datetime(2026, 4, 30, 12, 0, tzinfo=timezone.utc),
                lat=29.7604,
                lon=-95.3698,
                metric="s5p_no2_column",
                value=2.0e-4,
                unit="mol/m^2",
                source="sentinel5p",
                source_entity_id="no2-stored",
            )
        )
        await db_session.commit()

        def handler(request: httpx.Request) -> httpx.Response:
            if "openid-connect/token" in str(request.url):
                return httpx.Response(200, json={"access_token": "tok-123"})
            return httpx.Response(
                200,
                json=make_payload(
                    make_record(product_id="no2-stored"),
                    make_record(
                        product_id="no2-new", start="2026-04-30T13:00:00.000Z"
                    ),
                ),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = Sentinel5PCollector(http_client=client)

        downloaded: list[str] = []

        async def fake_extract(
            self,
            client_arg: httpx.AsyncClient,
            product_id: str,
            product_code: str,
            token: str,
            bbox: BoundingBox,
        ) -> float | None:
            downloaded.append(product_id)
            return 3.0e-4

        with patch.object(
            Sentinel5PCollector, "_download_and_extract", fake_extract
        ):
            try:
                result = await collector.collect(db_session, max_retries=1)
            finally:
                await client.aclose()

        assert result.success
        assert downloaded == ["no2-new"]
