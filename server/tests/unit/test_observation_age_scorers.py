"""B8 observation-age enforcement at nearest-in-time scorer legs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

import pytest

from app.llm.corroboration import (
    SILENT,
    SUPPORTING,
    aggregate_verdicts,
    score_atmospheric_trap,
    score_background_vs_event,
    score_chemistry,
    score_concentration_elevation,
    score_meteorological_state,
    score_point_source_attribution,
    score_temporal_pattern,
    score_transport_direction,
)


def _summary(metrics_by_source: Mapping[str, Mapping[str, dict[str, Any]]]) -> dict:
    return {
        "anomaly": {
            "timestamp": "2026-06-15T12:00:00+00:00",
            "lat": 29.7604,
            "lon": -95.3698,
        },
        "sources": {
            source: {"metrics": dict(metrics)}
            for source, metrics in metrics_by_source.items()
        },
    }


def _nearest(value: float, dt_minutes: object) -> dict[str, Any]:
    return {
        "nearest_in_time": {
            "t": "2026-06-15T12:00:00+00:00",
            "v": value,
            "dt_minutes": dt_minutes,
        }
    }


def test_concentration_exact_gate_votes_and_one_minute_past_is_silent() -> None:
    fresh = _summary({"tceq": {"no2": _nearest(82.0, 90.0)}})
    stale = _summary({"tceq": {"no2": _nearest(82.0, 91.0)}})

    fresh_verdicts, _ = score_concentration_elevation(
        "NO2 exceeded 80 ppb.", fresh
    )
    stale_verdicts, stale_note = score_concentration_elevation(
        "NO2 exceeded 80 ppb.", stale
    )

    assert fresh_verdicts["tceq"] == SUPPORTING
    assert stale_verdicts["tceq"] == SILENT
    assert aggregate_verdicts(stale_verdicts).evidence_n == 0
    assert "tceq: no2 age-gated SILENT" in stale_note
    assert "dt_minutes=91.0" in stale_note
    assert "gate_minutes=90.0" in stale_note
    assert "reason=stale" in stale_note


def test_missing_age_is_an_explicit_scorer_abstention() -> None:
    summary = _summary(
        {"tceq": {"no2": {"nearest_in_time": {"v": 82.0}}}}
    )

    verdicts, note = score_concentration_elevation(
        "NO2 exceeded 80 ppb.", summary
    )

    assert verdicts["tceq"] == SILENT
    assert "tceq: no2 age-gated SILENT" in note
    assert "dt_minutes=None" in note
    assert "gate_minutes=90.0" in note
    assert "reason=missing_or_invalid" in note


def _gfs_wind_summary(u_age: float, v_age: float) -> dict:
    return _summary(
        {
            "noaa_gfs": {
                "u_10m": _nearest(0.0, u_age),
                "v_10m": _nearest(4.0, v_age),
            }
        }
    )


def test_gfs_direction_requires_both_component_ages_within_gate() -> None:
    fresh_verdicts, _ = score_transport_direction(
        "Southerly winds transported the plume northward.",
        _gfs_wind_summary(360.0, 360.0),
    )
    stale_verdicts, stale_note = score_transport_direction(
        "Southerly winds transported the plume northward.",
        _gfs_wind_summary(360.0, 361.0),
    )

    assert fresh_verdicts["noaa_gfs"] == SUPPORTING
    assert stale_verdicts["noaa_gfs"] == SILENT
    assert "noaa_gfs: v_10m age-gated SILENT" in stale_note
    assert "dt_minutes=361.0" in stale_note
    assert "gate_minutes=360.0" in stale_note


@pytest.mark.parametrize(
    ("scorer", "claim"),
    [
        (score_transport_direction, "Westerly winds transported the plume."),
        (
            score_point_source_attribution,
            "A source at 29.7604 N, 95.4698 W drove the plume.",
        ),
    ],
)
def test_openweather_direction_age_gate_applies_to_both_direction_scorers(
    scorer: Callable[[str, Mapping[str, Any]], tuple[dict[str, int], str]],
    claim: str,
) -> None:
    fresh = _summary(
        {"openweather": {"wind_direction": _nearest(270.0, 90.0)}}
    )
    stale = _summary(
        {"openweather": {"wind_direction": _nearest(270.0, 91.0)}}
    )

    fresh_verdicts, _ = scorer(claim, fresh)
    stale_verdicts, stale_note = scorer(claim, stale)

    assert fresh_verdicts["openweather"] == SUPPORTING
    assert stale_verdicts["openweather"] == SILENT
    assert "openweather: wind_direction age-gated SILENT" in stale_note


def test_meteorological_gate_abstains_per_aspect_not_whole_source() -> None:
    summary = _summary(
        {
            "openweather": {
                "wind_speed": _nearest(1.0, 90.0),
                "temperature": _nearest(35.0, 91.0),
            }
        }
    )

    verdicts, note = score_meteorological_state(
        "Wind was 1 m/s and temperature was 35 C.", summary
    )

    assert verdicts["openweather"] == SUPPORTING
    assert "openweather: temperature age-gated SILENT" in note
    assert "dt_minutes=91.0" in note
    assert "gate_minutes=90.0" in note


def test_pbl_event_value_is_silent_past_the_gfs_gate() -> None:
    fresh = _summary({"noaa_gfs": {"pbl_height": _nearest(500.0, 360.0)}})
    stale = _summary({"noaa_gfs": {"pbl_height": _nearest(500.0, 361.0)}})

    fresh_verdicts, _ = score_atmospheric_trap("PBL height was 500 m.", fresh)
    stale_verdicts, stale_note = score_atmospheric_trap(
        "PBL height was 500 m.", stale
    )

    assert fresh_verdicts["noaa_gfs"] == SUPPORTING
    assert stale_verdicts["noaa_gfs"] == SILENT
    assert "noaa_gfs: pbl_height age-gated SILENT" in stale_note


def test_chemistry_nearest_leg_uses_sentinel_gate_but_keeps_window_mean() -> None:
    fresh_block = _nearest(8.0e-5, 720.0)
    fresh_block["value_range"] = {"mean": 4.0e-5}
    stale_block = _nearest(8.0e-5, 721.0)
    stale_block["value_range"] = {"mean": 4.0e-5}

    fresh_verdicts, _ = score_chemistry(
        "Elevated HCHO points to fresh VOCs.",
        _summary({"sentinel5p": {"s5p_hcho_column": fresh_block}}),
    )
    stale_verdicts, stale_note = score_chemistry(
        "Elevated HCHO points to fresh VOCs.",
        _summary({"sentinel5p": {"s5p_hcho_column": stale_block}}),
    )

    assert fresh_verdicts["sentinel5p"] == SUPPORTING
    assert stale_verdicts["sentinel5p"] == SILENT
    assert "sentinel5p: s5p_hcho_column age-gated SILENT" in stale_note
    assert "dt_minutes=721.0" in stale_note
    assert "gate_minutes=720.0" in stale_note


def test_window_trend_and_spatial_cv_ignore_nearest_age() -> None:
    trend_block = _nearest(4.0, 999.0)
    trend_block["entities"] = [
        {
            "entity_id": "trend",
            "series": [
                ["2026-06-15T09:00:00+00:00", 1.0],
                ["2026-06-15T10:00:00+00:00", 2.0],
                ["2026-06-15T11:00:00+00:00", 3.0],
                ["2026-06-15T12:00:00+00:00", 4.0],
            ],
        }
    ]
    trend_verdicts, _ = score_temporal_pattern(
        "NO2 levels rose before the event.",
        _summary({"tceq": {"no2": trend_block}}),
    )

    cv_block = _nearest(10.0, 999.0)
    cv_block["entities"] = [
        {
            "entity_id": f"station-{index}",
            "series": [
                [f"2026-06-15T{hour:02d}:00:00+00:00", 10.0 + index * 0.1]
                for hour in range(6)
            ],
        }
        for index in range(5)
    ]
    cv_verdicts, _ = score_background_vs_event(
        "Regional PM2.5 affected the area.",
        _summary({"tceq": {"pm25": cv_block}}),
    )

    assert trend_verdicts["tceq"] == SUPPORTING
    assert cv_verdicts["tceq"] == SUPPORTING
