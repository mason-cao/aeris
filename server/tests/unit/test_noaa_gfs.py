from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.collectors.noaa_gfs import (
    CYCLE_FALLBACK,
    NOAAGFSCollector,
    VARIABLES,
    cycle_candidates,
    filter_params,
    from_gfs_longitude,
    parse_cycle_time,
)

FIXTURE = Path(__file__).parent / "fixtures" / "gfs_houston_sample.grib2"


@pytest.fixture
def collector() -> NOAAGFSCollector:
    return NOAAGFSCollector()


def make_grid_cell(
    *,
    lat: float = 29.75,
    lon: float = -95.37,
    **overrides: float,
) -> dict[str, Any]:
    values: dict[str, float] = {
        "gh_500": 5840.0,
        "t_850": 295.0,
        "u_10m": 3.4,
        "v_10m": -1.2,
        "surface_pressure": 101325.0,
        "precipitable_water": 35.0,
        "pbl_height": 850.0,
    }
    values.update(overrides)
    return {"lat": lat, "lon": lon, "values": values}


def make_raw(cells: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "cycle_time": "2026-05-13T12:00:00+00:00",
        "cycle_date": "20260513",
        "cycle_hour": 12,
        "grid": cells if cells is not None else [make_grid_cell()],
    }


class TestCycleCandidates:
    def test_returns_four_cycles_newest_first(self) -> None:
        now = datetime(2026, 5, 13, 14, 30, tzinfo=timezone.utc)

        cycles = cycle_candidates(now)

        assert len(cycles) == CYCLE_FALLBACK
        assert cycles[0] == datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
        assert cycles[1] == datetime(2026, 5, 13, 6, 0, tzinfo=timezone.utc)
        assert cycles[2] == datetime(2026, 5, 13, 0, 0, tzinfo=timezone.utc)
        assert cycles[3] == datetime(2026, 5, 12, 18, 0, tzinfo=timezone.utc)

    def test_floors_to_six_hour_boundary(self) -> None:
        now = datetime(2026, 5, 13, 5, 59, tzinfo=timezone.utc)

        cycles = cycle_candidates(now)

        assert cycles[0] == datetime(2026, 5, 13, 0, 0, tzinfo=timezone.utc)

    def test_handles_naive_datetimes_as_utc(self) -> None:
        cycles = cycle_candidates(datetime(2026, 5, 13, 13, 0))

        assert cycles[0] == datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


class TestFilterParams:
    def test_dir_and_file_encode_cycle_date_and_hour(self) -> None:
        cycle = datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)

        params = filter_params(cycle)

        assert params["dir"] == "/gfs.20260517/12/atmos"
        assert params["file"] == "gfs.t12z.pgrb2.0p25.f000"

    def test_pads_single_digit_cycle_hour(self) -> None:
        cycle = datetime(2026, 5, 17, 6, 0, tzinfo=timezone.utc)

        params = filter_params(cycle)

        assert params["dir"] == "/gfs.20260517/06/atmos"
        assert params["file"] == "gfs.t06z.pgrb2.0p25.f000"

    def test_requests_all_seven_gfs_variables(self) -> None:
        params = filter_params(datetime(2026, 5, 17, 0, 0, tzinfo=timezone.utc))

        for var in ("HGT", "TMP", "UGRD", "VGRD", "PRES", "PWAT", "HPBL"):
            assert params[f"var_{var}"] == "on"

    def test_requests_subregion_bounding_box(self) -> None:
        params = filter_params(datetime(2026, 5, 17, 0, 0, tzinfo=timezone.utc))

        assert params["subregion"] == ""
        for key in ("toplat", "bottomlat", "leftlon", "rightlon"):
            assert key in params
        assert float(params["toplat"]) > float(params["bottomlat"])
        assert float(params["rightlon"]) > float(params["leftlon"])


class TestFromGfsLongitude:
    def test_unwraps_above_180(self) -> None:
        assert from_gfs_longitude(264.6302) == pytest.approx(-95.3698)

    def test_passes_through_below_180(self) -> None:
        assert from_gfs_longitude(45.0) == 45.0


class TestParseCycleTime:
    def test_handles_zulu_suffix(self) -> None:
        parsed = parse_cycle_time("2026-05-13T12:00:00Z")

        assert parsed == datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)

    def test_handles_offset_suffix(self) -> None:
        parsed = parse_cycle_time("2026-05-13T12:00:00+00:00")

        assert parsed == datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)

    def test_assumes_utc_when_naive(self) -> None:
        parsed = parse_cycle_time("2026-05-13T12:00:00")

        assert parsed.tzinfo is timezone.utc


class TestParseGrib:
    """Parses a real NOMADS GRIB-filter subset captured for the Houston bbox."""

    def test_extracts_one_cell_per_grid_point(
        self, collector: NOAAGFSCollector
    ) -> None:
        grid = collector._parse_grib(FIXTURE.read_bytes())

        assert len(grid) == 56  # 7x8 subregion

    def test_every_cell_carries_all_seven_metrics(
        self, collector: NOAAGFSCollector
    ) -> None:
        grid = collector._parse_grib(FIXTURE.read_bytes())

        expected = {var.metric for var in VARIABLES}
        for cell in grid:
            assert set(cell["values"]) == expected

    def test_extracts_pbl_height_despite_unrecognized_short_name(
        self, collector: NOAAGFSCollector
    ) -> None:
        # HPBL is NCEP local parameter 196; eccodes labels it shortName='unknown',
        # so it must be matched by parameterCategory/parameterNumber instead.
        grid = collector._parse_grib(FIXTURE.read_bytes())

        pbl_values = [cell["values"]["pbl_height"] for cell in grid]
        assert pbl_values
        assert all(0.0 < value < 4000.0 for value in pbl_values)

    def test_values_are_raw_pre_conversion(
        self, collector: NOAAGFSCollector
    ) -> None:
        # normalize() applies unit conversion; the grid carries raw GRIB values.
        grid = collector._parse_grib(FIXTURE.read_bytes())

        cell = grid[0]
        assert 250.0 < cell["values"]["t_850"] < 320.0  # Kelvin, not Celsius
        assert 90000.0 < cell["values"]["surface_pressure"] < 105000.0  # Pa, not hPa

    def test_converts_longitudes_to_wgs84(
        self, collector: NOAAGFSCollector
    ) -> None:
        grid = collector._parse_grib(FIXTURE.read_bytes())

        assert all(-98.0 < cell["lon"] < -93.0 for cell in grid)
        assert all(28.0 < cell["lat"] < 31.5 for cell in grid)

    def test_returns_empty_for_non_grib_bytes(
        self, collector: NOAAGFSCollector
    ) -> None:
        assert collector._parse_grib(b"") == []


class TestFetchCycleGrib:
    def _collector_with_response(
        self, *, status_code: int, content: bytes
    ) -> tuple[NOAAGFSCollector, MagicMock]:
        response = MagicMock(status_code=status_code, content=content)
        client = MagicMock()
        client.get = AsyncMock(return_value=response)
        return NOAAGFSCollector(http_client=client), client

    async def test_returns_bytes_when_response_is_grib(self) -> None:
        collector, _ = self._collector_with_response(
            status_code=200, content=b"GRIB\x00\x01payload"
        )

        data = await collector._fetch_cycle_grib(
            datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        )

        assert data == b"GRIB\x00\x01payload"

    async def test_returns_none_when_response_is_html_error_page(self) -> None:
        collector, _ = self._collector_with_response(
            status_code=200, content=b"<!doctype html><html>retired</html>"
        )

        data = await collector._fetch_cycle_grib(
            datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        )

        assert data is None

    async def test_returns_none_on_non_200_status(self) -> None:
        collector, _ = self._collector_with_response(status_code=404, content=b"GRIB")

        data = await collector._fetch_cycle_grib(
            datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        )

        assert data is None

    async def test_requests_the_nomads_grib_filter(self) -> None:
        collector, client = self._collector_with_response(
            status_code=200, content=b"GRIB"
        )

        await collector._fetch_cycle_grib(
            datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
        )

        called_url = client.get.await_args.args[0]
        assert "filter_gfs_0p25.pl" in called_url

    async def test_follows_redirect_to_grib(self) -> None:
        # NOMADS can 30x the filter endpoint; without per-request redirect
        # following the cycle is silently dropped as a non-200 response.
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "moved=1" not in url:
                return httpx.Response(307, headers={"location": url + "&moved=1"})
            return httpx.Response(200, content=b"GRIB\x00\x01payload")

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        collector = NOAAGFSCollector(http_client=client)
        try:
            data = await collector._fetch_cycle_grib(
                datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
            )
        finally:
            await client.aclose()

        assert data == b"GRIB\x00\x01payload"


class TestLoadCycle:
    async def test_returns_empty_when_cycle_unavailable(
        self, collector: NOAAGFSCollector
    ) -> None:
        with patch.object(
            collector, "_fetch_cycle_grib", new=AsyncMock(return_value=None)
        ):
            grid = await collector._load_cycle(
                datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
            )

        assert grid == []

    async def test_parses_fetched_grib_bytes(
        self, collector: NOAAGFSCollector
    ) -> None:
        with patch.object(
            collector,
            "_fetch_cycle_grib",
            new=AsyncMock(return_value=FIXTURE.read_bytes()),
        ):
            grid = await collector._load_cycle(
                datetime(2026, 5, 17, 12, 0, tzinfo=timezone.utc)
            )

        assert len(grid) == 56


class TestNOAAGFSNormalize:
    def test_emits_one_point_per_variable_per_in_radius_cell(
        self, collector: NOAAGFSCollector
    ) -> None:
        points = collector.normalize(make_raw([make_grid_cell()]))

        assert len(points) == len(VARIABLES)
        assert {p.metric for p in points} == {var.metric for var in VARIABLES}

    def test_sets_source_and_grid_entity_id(
        self, collector: NOAAGFSCollector
    ) -> None:
        points = collector.normalize(make_raw([make_grid_cell(lat=29.75, lon=-95.37)]))

        assert all(p.source == "noaa_gfs" for p in points)
        assert {p.source_entity_id for p in points} == {"gfs:29.75,-95.37"}

    def test_sets_timestamp_from_cycle(self, collector: NOAAGFSCollector) -> None:
        points = collector.normalize(make_raw([make_grid_cell()]))

        assert all(
            p.timestamp == datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)
            for p in points
        )

    def test_converts_kelvin_to_celsius_for_temperature(
        self, collector: NOAAGFSCollector
    ) -> None:
        points = collector.normalize(make_raw([make_grid_cell(t_850=295.15)]))
        t_850 = next(p for p in points if p.metric == "t_850")

        assert t_850.value == pytest.approx(22.0)
        assert t_850.unit == "degC"

    def test_converts_pascals_to_hectopascals_for_surface_pressure(
        self, collector: NOAAGFSCollector
    ) -> None:
        points = collector.normalize(
            make_raw([make_grid_cell(surface_pressure=101325.0)])
        )
        sp = next(p for p in points if p.metric == "surface_pressure")

        assert sp.value == pytest.approx(1013.25)
        assert sp.unit == "hPa"

    def test_passes_through_height_wind_water_and_pbl(
        self, collector: NOAAGFSCollector
    ) -> None:
        points = collector.normalize(make_raw([make_grid_cell()]))
        by_metric = {p.metric: p for p in points}

        assert by_metric["gh_500"].value == 5840.0
        assert by_metric["gh_500"].unit == "m"
        assert by_metric["u_10m"].value == pytest.approx(3.4)
        assert by_metric["v_10m"].value == pytest.approx(-1.2)
        assert by_metric["precipitable_water"].value == 35.0
        assert by_metric["pbl_height"].value == 850.0

    def test_filters_cells_outside_target_radius(
        self, collector: NOAAGFSCollector
    ) -> None:
        cells = [
            make_grid_cell(lat=29.75, lon=-95.37),
            make_grid_cell(lat=40.0, lon=-95.37),
        ]

        points = collector.normalize(make_raw(cells))

        assert {p.source_entity_id for p in points} == {"gfs:29.75,-95.37"}

    def test_preserves_raw_value_and_grib_provenance_in_raw_json(
        self, collector: NOAAGFSCollector
    ) -> None:
        points = collector.normalize(make_raw([make_grid_cell(t_850=300.0)]))
        t_850 = next(p for p in points if p.metric == "t_850")

        assert t_850.raw_json is not None
        assert t_850.raw_json["raw_value"] == 300.0
        assert t_850.raw_json["gfs_variable"] == "t"
        assert t_850.raw_json["cycle_hour"] == 12

    def test_skips_unknown_metric_keys(self, collector: NOAAGFSCollector) -> None:
        cell = make_grid_cell()
        cell["values"]["bogus_metric"] = 42.0

        points = collector.normalize(make_raw([cell]))

        assert all(p.metric in {var.metric for var in VARIABLES} for p in points)

    def test_skips_none_values(self, collector: NOAAGFSCollector) -> None:
        cell = make_grid_cell()
        cell["values"]["t_850"] = None

        points = collector.normalize(make_raw([cell]))

        assert all(p.metric != "t_850" for p in points)

    def test_empty_grid_yields_no_points(self, collector: NOAAGFSCollector) -> None:
        assert collector.normalize(make_raw([])) == []

    def test_missing_cycle_time_yields_no_points(
        self, collector: NOAAGFSCollector
    ) -> None:
        raw = make_raw([make_grid_cell()])
        raw.pop("cycle_time")

        assert collector.normalize(raw) == []


class TestNOAAGFSFetch:
    async def test_returns_cycle_metadata_and_grid_on_success(self) -> None:
        collector = NOAAGFSCollector()
        grid = [make_grid_cell()]
        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)

        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector, "_load_cycle", new=AsyncMock(return_value=grid)
            ) as mock_load,
        ):
            mock_dt.now.return_value = fixed_now
            result = await collector.fetch()

        assert result["cycle_date"] == "20260513"
        assert result["cycle_hour"] == 12
        assert result["grid"] == grid
        mock_load.assert_awaited_once()

    async def test_walks_back_when_newest_cycle_fails(self) -> None:
        collector = NOAAGFSCollector()
        grid = [make_grid_cell()]
        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)

        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector,
                "_load_cycle",
                new=AsyncMock(side_effect=[RuntimeError("502"), grid]),
            ) as mock_load,
        ):
            mock_dt.now.return_value = fixed_now
            result = await collector.fetch()

        assert result["cycle_hour"] == 6
        assert mock_load.await_count == 2

    async def test_raises_when_all_cycles_fail(self) -> None:
        collector = NOAAGFSCollector()
        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)

        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector,
                "_load_cycle",
                new=AsyncMock(side_effect=RuntimeError("network down")),
            ),
        ):
            mock_dt.now.return_value = fixed_now

            with pytest.raises(RuntimeError, match="No GFS cycle reachable"):
                await collector.fetch()

    async def test_treats_empty_grid_as_cycle_failure(self) -> None:
        collector = NOAAGFSCollector()
        grid = [make_grid_cell()]
        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)

        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector,
                "_load_cycle",
                new=AsyncMock(side_effect=[[], grid]),
            ) as mock_load,
        ):
            mock_dt.now.return_value = fixed_now
            result = await collector.fetch()

        assert mock_load.await_count == 2
        assert result["cycle_hour"] == 6
