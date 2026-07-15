"""B2 label-free calm-wind cutoff empirics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.eval.calm_wind_empirics import (
    CalmWindEmpiricalReport,
    ChannelEmpirics,
    ComponentObservation,
    SpeedObservation,
    calm_cutoff,
    candidate_centers,
    describe_cutoffs,
    pair_gfs_components,
    render_markdown,
    window_cutoffs,
)


def test_calm_cutoff_uses_population_sd_and_includes_exact_zero() -> None:
    assert calm_cutoff([1.0, 3.0]) == 0.0

    distribution = describe_cutoffs([0.0, 1.0], resolution=0.5)

    assert distribution.fraction_nonpositive == 0.5
    assert distribution.fraction_below_resolution == 0.5


def test_calm_cutoff_handles_degenerate_and_below_floor_series() -> None:
    assert calm_cutoff([2.0, 2.0]) == 2.0
    assert calm_cutoff([]) is None
    assert calm_cutoff([2.0]) is None
    with pytest.raises(ValueError, match="non-negative"):
        calm_cutoff([-0.1, 1.0])


def test_candidate_centers_are_timezone_stable_and_require_full_windows() -> None:
    aware_start = datetime(2026, 6, 1, tzinfo=UTC)
    aware_end = datetime(2026, 6, 5, tzinfo=UTC)
    naive_start = aware_start.replace(tzinfo=None)
    naive_end = aware_end.replace(tzinfo=None)

    aware = candidate_centers(aware_start, aware_end)
    naive = candidate_centers(naive_start, naive_end)

    assert aware == naive
    assert aware[0] == datetime(2026, 6, 2, 12, tzinfo=UTC)
    assert aware[-1] == datetime(2026, 6, 3, 11, tzinfo=UTC)


def test_window_cutoffs_include_both_72_hour_endpoints_and_count_low_n() -> None:
    center = datetime(2026, 6, 3, 12, tzinfo=UTC)
    observations = [
        SpeedObservation(center - timedelta(hours=36), 1.0),
        SpeedObservation(center + timedelta(hours=36), 3.0),
    ]

    result = window_cutoffs(observations, [center])

    assert result.cutoffs == (0.0,)
    assert result.insufficient_windows == 0
    assert result.window_sample_sizes == (2,)

    insufficient = window_cutoffs(observations[:1], [center])
    assert insufficient.cutoffs == ()
    assert insufficient.insufficient_windows == 1
    assert insufficient.window_sample_sizes == (1,)


def test_gfs_pairing_requires_same_entity_and_timestamp() -> None:
    event = datetime(2026, 6, 3, 12, tzinfo=UTC)
    u_values = [
        ComponentObservation("cell-a", event, 3.0),
        ComponentObservation("cell-b", event, 8.0),
    ]
    v_values = [
        ComponentObservation("cell-a", event, 4.0),
        ComponentObservation("cell-b", event + timedelta(hours=1), 6.0),
    ]

    paired = pair_gfs_components(u_values, v_values)

    assert paired == (SpeedObservation(event, 5.0),)


def test_distribution_resolution_is_strict_and_can_be_not_applicable() -> None:
    applicable = describe_cutoffs([0.49, 0.5, 0.51], resolution=0.5)
    not_applicable = describe_cutoffs([0.49, 0.5, 0.51], resolution=None)

    assert applicable.fraction_below_resolution == pytest.approx(1 / 3)
    assert not_applicable.fraction_below_resolution is None


def test_markdown_escapes_gfs_channel_pipe() -> None:
    distribution = describe_cutoffs([0.5, 1.0], resolution=None)
    report = CalmWindEmpiricalReport(
        snapshot_sha256="abc",
        study_start="start",
        study_end_exclusive="end",
        candidate_window_count=1,
        channels=(
            ChannelEmpirics(
                channel="GFS |u,v|",
                resolution_mps=None,
                observation_count=2,
                candidate_window_count=1,
                insufficient_window_count=0,
                window_n_minimum=2,
                window_n_median=2.0,
                window_n_maximum=2,
                cutoff_distribution=distribution,
            ),
        ),
    )

    assert "GFS \\|u,v\\|" in render_markdown(report)
