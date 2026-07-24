"""B2 calm-wind guard shared by direction scorers and packets."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.llm.corroboration import (
    SILENT,
    SUPPORTING,
    DEFAULT_WIND_TOLERANCE,
    calm_wind_decision,
    calm_wind_manifest_payload,
    calm_wind_source_decisions,
    score_meteorological_state,
    score_point_source_attribution,
    score_transport_direction,
)


def _metric(
    nearest_value: float,
    values: list[tuple[str, float]],
    *,
    entity_id: str = "station-a",
    dt_minutes: float = 0.0,
) -> dict:
    return {
        "nearest_in_time": {
            "t": values[0][0],
            "v": nearest_value,
            "entity_id": entity_id,
            "dt_minutes": dt_minutes,
        },
        "entities": [
            {
                "entity_id": entity_id,
                "series": [[timestamp, value] for timestamp, value in values],
            }
        ],
    }


def _summary(sources: dict) -> dict:
    return {
        "anomaly": {"lat": 29.76, "lon": -95.37},
        "sources": {
            source: {"metrics": metrics}
            for source, metrics in sources.items()
        },
    }


def test_proposed_floor_uses_population_sd_and_strict_event_boundary() -> None:
    # {1, 3}: mean=2, pstdev=1, raw cutoff=0, proposed effective cutoff=1.5.
    below = calm_wind_decision("asos", [1.0, 3.0], 1.499)
    exact = calm_wind_decision("asos", [1.0, 3.0], 1.5)

    assert below.raw_cutoff_ms == 0.0
    assert below.effective_cutoff_ms == 1.5
    assert below.calm is True
    assert below.direction_votable is False
    assert exact.calm is False
    assert exact.direction_votable is True


def test_positive_raw_cutoff_works_without_a_floor() -> None:
    tolerance = replace(DEFAULT_WIND_TOLERANCE, calm_floor_ms=None)

    below = calm_wind_decision(
        "openweather",
        [4.0, 6.0],
        2.999,
        tolerance=tolerance,
    )
    exact = calm_wind_decision(
        "openweather",
        [4.0, 6.0],
        3.0,
        tolerance=tolerance,
    )

    assert below.raw_cutoff_ms == 3.0
    assert below.effective_cutoff_ms == 3.0
    assert below.calm is True
    assert exact.direction_votable is True


def test_nonpositive_raw_cutoff_without_floor_disables_loudly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tolerance = replace(DEFAULT_WIND_TOLERANCE, calm_floor_ms=None)

    decision = calm_wind_decision(
        "noaa_gfs",
        [1.0, 3.0],
        0.25,
        tolerance=tolerance,
    )

    assert decision.guard_enabled is False
    assert decision.direction_votable is True
    assert decision.calm is None
    assert decision.reason == "raw_cutoff_nonpositive_guard_disabled"
    assert "disabled" in caplog.text
    assert "noaa_gfs" in caplog.text


@pytest.mark.parametrize(
    ("window", "reason"),
    [
        ([], "insufficient_window"),
        ([2.0], "insufficient_window"),
        ([-1.0, 2.0], "invalid_window_speed"),
        ([float("nan"), 2.0], "invalid_window_speed"),
    ],
)
def test_unevaluable_windows_abstain(
    window: list[float],
    reason: str,
) -> None:
    decision = calm_wind_decision("asos", window, 1.0)

    assert decision.direction_votable is False
    assert decision.calm is None
    assert decision.reason == reason


def test_zero_spread_window_is_evaluable() -> None:
    below = calm_wind_decision("asos", [2.0, 2.0], 1.999)
    exact = calm_wind_decision("asos", [2.0, 2.0], 2.0)

    assert below.raw_cutoff_ms == 2.0
    assert below.calm is True
    assert exact.calm is False
    assert exact.direction_votable is True


def test_gfs_window_pairs_components_only_at_exact_entity_timestamp() -> None:
    t0 = "2026-06-15T00:00:00+00:00"
    t1 = "2026-06-15T06:00:00+00:00"
    t2 = "2026-06-15T12:00:00+00:00"
    u_block = _metric(3.0, [(t0, 3.0), (t1, 8.0), (t2, 0.0)], entity_id="cell-a")
    v_block = _metric(4.0, [(t0, 4.0), (t2, 5.0)], entity_id="cell-a")
    v_block["entities"].append(
        {"entity_id": "cell-b", "series": [[t1, 6.0]]}
    )

    decisions, _ = calm_wind_source_decisions(
        _summary({"noaa_gfs": {"u_10m": u_block, "v_10m": v_block}}),
        ("noaa_gfs",),
    )

    decision = decisions["noaa_gfs"]
    assert decision.window_n == 2
    assert decision.event_speed_ms == 5.0
    assert decision.raw_cutoff_ms == 5.0
    assert decision.direction_votable is True


def test_stale_event_speed_abstains_but_exact_age_boundary_is_used() -> None:
    values = [
        ("2026-06-15T00:00:00+00:00", 2.0),
        ("2026-06-15T01:00:00+00:00", 2.0),
    ]
    fresh_speed = _metric(1.0, values, dt_minutes=90.0)
    stale_speed = _metric(1.0, values, dt_minutes=91.0)

    fresh, _ = calm_wind_source_decisions(
        _summary({"openweather": {"wind_speed": fresh_speed}}),
        ("openweather",),
    )
    stale, notes = calm_wind_source_decisions(
        _summary({"openweather": {"wind_speed": stale_speed}}),
        ("openweather",),
    )

    assert fresh["openweather"].calm is True
    assert stale["openweather"].direction_votable is False
    assert stale["openweather"].reason == "missing_event_speed"
    assert "age-gated SILENT" in "; ".join(notes)


@pytest.mark.parametrize(
    ("scorer", "claim"),
    [
        (score_transport_direction, "Winds from the south carried the plume."),
        (
            score_point_source_attribution,
            "A source near 29.73 N, 95.22 W drove the plume.",
        ),
    ],
)
def test_below_cutoff_silences_both_direction_consumers(
    scorer,
    claim: str,
) -> None:
    times = [
        ("2026-06-15T00:00:00+00:00", 2.0),
        ("2026-06-15T01:00:00+00:00", 2.0),
    ]
    summary = _summary(
        {
            "openweather": {
                "wind_direction": _metric(180.0, times),
                "wind_speed": _metric(1.0, times),
            }
        }
    )

    verdicts, note = scorer(claim, summary)

    assert verdicts["openweather"] == SILENT
    assert "wind direction unstable under calm conditions" in note
    assert "event_speed=1.0" in note
    assert "effective_cutoff=2.0" in note


def test_calm_guard_does_not_change_stagnant_meteorological_rule() -> None:
    times = [
        ("2026-06-15T00:00:00+00:00", 2.0),
        ("2026-06-15T01:00:00+00:00", 2.0),
    ]
    summary = _summary(
        {"openweather": {"wind_speed": _metric(1.0, times)}}
    )

    verdicts, _ = score_meteorological_state(
        "Conditions were stagnant.",
        summary,
    )

    assert verdicts["openweather"] == SUPPORTING


def test_manifest_records_confirmed_status_and_disabled_behavior() -> None:
    payload = calm_wind_manifest_payload()

    assert payload["floor_ms"] == 1.5
    assert payload["floor_status"] == "bracco_confirmed"
    assert payload["bracco_amendment_confirmed"] is True
    assert payload["bracco_confirmation_date"] == "2026-07-24"
    assert payload["ship_status"] == "shipped_bracco_confirmed"
    assert payload["raw_nonpositive_without_floor"] == "disabled_loudly"
