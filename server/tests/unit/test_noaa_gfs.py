from datetime import datetime, timezone
from typing import Any
from unittest.mock import patch

import pytest

from app.collectors.noaa_gfs import (
    CYCLE_FALLBACK,
    NOAAGFSCollector,
    VARIABLES,
    cycle_candidates,
    dataset_url,
    from_gfs_longitude,
    parse_cycle_time,
    to_gfs_longitude,
)


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


class TestDatasetUrl:
    def test_formats_cycle_into_path(self) -> None:
        cycle = datetime(2026, 5, 13, 6, 0, tzinfo=timezone.utc)

        url = dataset_url(cycle)

        assert url == (
            "https://nomads.ncep.noaa.gov/dods/gfs_0p25"
            "/gfs20260513/gfs_0p25_06z"
        )


class TestLongitudeConversion:
    def test_to_gfs_longitude_wraps_negative_to_0_360(self) -> None:
        assert to_gfs_longitude(-95.3698) == pytest.approx(264.6302)

    def test_to_gfs_longitude_leaves_positive_in_range(self) -> None:
        assert to_gfs_longitude(45.0) == 45.0

    def test_from_gfs_longitude_unwraps_above_180(self) -> None:
        assert from_gfs_longitude(264.6302) == pytest.approx(-95.3698)

    def test_from_gfs_longitude_passes_through_below_180(self) -> None:
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

    def test_preserves_raw_value_in_raw_json(self, collector: NOAAGFSCollector) -> None:
        points = collector.normalize(make_raw([make_grid_cell(t_850=300.0)]))
        t_850 = next(p for p in points if p.metric == "t_850")

        assert t_850.raw_json is not None
        assert t_850.raw_json["raw_value"] == 300.0
        assert t_850.raw_json["gfs_variable"] == "tmpprs"
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
    @pytest.mark.asyncio
    async def test_returns_cycle_metadata_and_grid_on_success(self) -> None:
        collector = NOAAGFSCollector()
        grid = [make_grid_cell()]

        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)
        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(collector, "_read_cycle", return_value=grid) as mock_read,
        ):
            mock_dt.now.return_value = fixed_now
            result = await collector.fetch()

        assert result["cycle_date"] == "20260513"
        assert result["cycle_hour"] == 12
        assert result["grid"] == grid
        mock_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_walks_back_when_newest_cycle_fails(self) -> None:
        collector = NOAAGFSCollector()
        grid = [make_grid_cell()]

        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)
        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector,
                "_read_cycle",
                side_effect=[RuntimeError("502"), grid],
            ) as mock_read,
        ):
            mock_dt.now.return_value = fixed_now
            result = await collector.fetch()

        assert result["cycle_hour"] == 6
        assert mock_read.call_count == 2

    @pytest.mark.asyncio
    async def test_raises_when_all_cycles_fail(self) -> None:
        collector = NOAAGFSCollector()
        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)

        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector,
                "_read_cycle",
                side_effect=RuntimeError("network down"),
            ),
        ):
            mock_dt.now.return_value = fixed_now

            with pytest.raises(RuntimeError, match="No GFS cycle reachable"):
                await collector.fetch()

    @pytest.mark.asyncio
    async def test_treats_empty_grid_as_cycle_failure(self) -> None:
        collector = NOAAGFSCollector()
        fixed_now = datetime(2026, 5, 13, 13, 0, tzinfo=timezone.utc)

        with (
            patch("app.collectors.noaa_gfs.datetime") as mock_dt,
            patch.object(
                collector,
                "_read_cycle",
                side_effect=[[], [make_grid_cell()]],
            ) as mock_read,
        ):
            mock_dt.now.return_value = fixed_now
            result = await collector.fetch()

        assert mock_read.call_count == 2
        assert result["cycle_hour"] == 6
