from datetime import datetime, timezone
from typing import Any

import httpx
import pytest

from app.collectors.nasa_firms import (
    FIRMS_SOURCES,
    NASAFIRMSCollector,
    detection_entity_id,
    firms_area_coordinates,
    parse_acquisition_time,
    parse_confidence,
    parse_firms_csv,
)
from app.config import settings


VIIRS_CSV = (
    "latitude,longitude,bright_ti4,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_ti5,frp,daynight\n"
    "34.10000,-84.20000,331.4,0.5,0.4,2026-04-20,631,N20,VIIRS,"
    "n,2.0NRT,300.1,12.3,D\n"
)

MODIS_CSV = (
    "latitude,longitude,brightness,scan,track,acq_date,acq_time,satellite,"
    "instrument,confidence,version,bright_t31,frp,daynight\n"
    "34.20000,-84.30000,322.8,1.0,1.0,2026-04-20,1825,Terra,MODIS,"
    "87,6.1NRT,291.2,8.4,N\n"
)


@pytest.fixture
def collector() -> NASAFIRMSCollector:
    return NASAFIRMSCollector()


def sample_viirs_row() -> dict[str, Any]:
    return parse_firms_csv("VIIRS_NOAA20_NRT", VIIRS_CSV)[0]


def sample_modis_row() -> dict[str, Any]:
    return parse_firms_csv("MODIS_NRT", MODIS_CSV)[0]


class TestNASAFIRMSNormalize:
    def test_normalize_viirs_detection(self, collector: NASAFIRMSCollector) -> None:
        points = collector.normalize({"detections": [sample_viirs_row()]})

        assert len(points) == 3
        assert {point.metric for point in points} == {
            "fire_radiative_power",
            "fire_confidence",
            "fire_brightness",
        }
        assert {point.unit for point in points} == {"MW", "percent", "K"}

    def test_normalize_modis_detection(self, collector: NASAFIRMSCollector) -> None:
        points = collector.normalize({"detections": [sample_modis_row()]})
        confidence = next(point for point in points if point.metric == "fire_confidence")
        brightness = next(point for point in points if point.metric == "fire_brightness")

        assert confidence.value == 87.0
        assert brightness.value == 322.8

    def test_normalize_sets_timestamp_and_coordinates(
        self, collector: NASAFIRMSCollector
    ) -> None:
        points = collector.normalize({"detections": [sample_viirs_row()]})

        assert all(
            point.timestamp == datetime(2026, 4, 20, 6, 31, tzinfo=timezone.utc)
            for point in points
        )
        assert all(point.lat == 34.1 for point in points)
        assert all(point.lon == -84.2 for point in points)

    def test_normalize_sets_source_and_entity_id(
        self, collector: NASAFIRMSCollector
    ) -> None:
        points = collector.normalize({"detections": [sample_viirs_row()]})

        assert all(point.source == "nasa_firms" for point in points)
        assert len({point.source_entity_id for point in points}) == 1
        assert points[0].source_entity_id.startswith("VIIRS_NOAA20_NRT:N20:")

    def test_normalize_preserves_raw_json(self, collector: NASAFIRMSCollector) -> None:
        points = collector.normalize({"detections": [sample_viirs_row()]})

        assert points[0].raw_json is not None
        assert points[0].raw_json["instrument"] == "VIIRS"

    def test_normalize_skips_bad_timestamp(self, collector: NASAFIRMSCollector) -> None:
        row = sample_viirs_row()
        row["acq_time"] = "bad"

        assert collector.normalize({"detections": [row]}) == []

    def test_normalize_skips_missing_coordinates(
        self, collector: NASAFIRMSCollector
    ) -> None:
        row = sample_viirs_row()
        row["latitude"] = ""

        assert collector.normalize({"detections": [row]}) == []


class TestNASAFIRMSFetch:
    @pytest.mark.asyncio
    async def test_fetch_queries_all_sources(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "firms_map_key", "test-key")
        seen_paths: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            if "MODIS_NRT" in request.url.path:
                return httpx.Response(200, text="latitude,longitude\n")
            return httpx.Response(200, text=VIIRS_CSV)

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = NASAFIRMSCollector(http_client=client)

        raw = await collector.fetch()

        assert len(seen_paths) == len(FIRMS_SOURCES)
        assert all("/test-key/" in path for path in seen_paths)
        assert len(raw["detections"]) == 2
        assert raw["errors"] == []
        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_raises_when_map_key_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "firms_map_key", "")
        collector = NASAFIRMSCollector()

        with pytest.raises(RuntimeError, match="FIRMS_MAP_KEY"):
            await collector.fetch()

    @pytest.mark.asyncio
    async def test_fetch_raises_when_all_sources_fail(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "firms_map_key", "test-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="server error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = NASAFIRMSCollector(http_client=client)

        with pytest.raises(RuntimeError, match="All NASA FIRMS"):
            await collector.fetch()

        await client.aclose()

    @pytest.mark.asyncio
    async def test_fetch_allows_empty_successful_source(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "firms_map_key", "test-key")

        async def handler(request: httpx.Request) -> httpx.Response:
            if "MODIS_NRT" in request.url.path:
                return httpx.Response(200, text="latitude,longitude\n")
            return httpx.Response(500, text="server error")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = NASAFIRMSCollector(http_client=client)

        raw = await collector.fetch()

        assert raw["detections"] == []
        assert len(raw["errors"]) == 2
        await client.aclose()


class TestNASAFIRMSHelpers:
    def test_parse_firms_csv_adds_source_dataset(self) -> None:
        rows = parse_firms_csv("VIIRS_SNPP_NRT", VIIRS_CSV)

        assert len(rows) == 1
        assert rows[0]["source_dataset"] == "VIIRS_SNPP_NRT"

    def test_parse_firms_csv_rejects_non_csv_response(self) -> None:
        with pytest.raises(ValueError, match="non-CSV"):
            parse_firms_csv("VIIRS_SNPP_NRT", "<html>Invalid MAP_KEY</html>")

    def test_parse_acquisition_time_zero_pads_hour(self) -> None:
        row = sample_viirs_row()

        assert parse_acquisition_time(row) == datetime(
            2026, 4, 20, 6, 31, tzinfo=timezone.utc
        )

    def test_parse_confidence_handles_viirs_and_modis_values(self) -> None:
        assert parse_confidence("l") == 33.0
        assert parse_confidence("n") == 66.0
        assert parse_confidence("h") == 100.0
        assert parse_confidence("87") == 87.0
        assert parse_confidence("bad") is None

    def test_detection_entity_id_is_deterministic(self) -> None:
        row = sample_viirs_row()
        timestamp = parse_acquisition_time(row)
        assert timestamp is not None

        assert detection_entity_id(row, timestamp) == (
            "VIIRS_NOAA20_NRT:N20:34.10000:-84.20000:202604200631"
        )

    def test_firms_area_coordinates_uses_west_south_east_north(self) -> None:
        coords = firms_area_coordinates().split(",")

        assert len(coords) == 4
        west, south, east, north = [float(value) for value in coords]
        assert west < east
        assert south < north
