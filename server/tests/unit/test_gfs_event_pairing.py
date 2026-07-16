"""A-9 exact timestamp pairing for nearest-event GFS wind components."""

from __future__ import annotations

import pytest

from app.llm.corroboration import (
    SILENT,
    SUPPORTING,
    calm_wind_source_decisions,
    score_meteorological_state,
    score_point_source_attribution,
    score_transport_direction,
)


MATCHED_TIMESTAMP = "2026-06-05T12:00:00+00:00"


def _gfs_summary(
    *,
    u_timestamp: str | None = MATCHED_TIMESTAMP,
    v_timestamp: str | None = MATCHED_TIMESTAMP,
    u_dt_minutes: float = 0.0,
    v_dt_minutes: float = 0.0,
    u_entity_id: str = "cell-a",
    v_entity_id: str = "cell-a",
) -> dict:
    series_timestamps = (
        "2026-06-05T12:00:00+00:00",
        "2026-06-05T18:00:00+00:00",
    )

    def component(
        value: float,
        timestamp: str | None,
        dt_minutes: float,
        entity_id: str,
    ) -> dict:
        nearest: dict[str, object] = {
            "v": value,
            "dt_minutes": dt_minutes,
            "entity_id": entity_id,
        }
        if timestamp is not None:
            nearest["t"] = timestamp
        return {
            "nearest_in_time": nearest,
            "entities": [
                {
                    "entity_id": "cell-a",
                    "series": [
                        [series_timestamp, value]
                        for series_timestamp in series_timestamps
                    ],
                }
            ],
        }

    return {
        "schema_version": 1,
        "anomaly": {
            "timestamp": "2026-06-05T12:00:00+00:00",
            "lat": 29.76,
            "lon": -95.37,
        },
        "sources": {
            "noaa_gfs": {
                "metrics": {
                    "u_10m": component(
                        0.0,
                        u_timestamp,
                        u_dt_minutes,
                        u_entity_id,
                    ),
                    "v_10m": component(
                        4.0,
                        v_timestamp,
                        v_dt_minutes,
                        v_entity_id,
                    ),
                }
            }
        },
    }


def test_transport_silences_one_microsecond_component_mismatch() -> None:
    summary = _gfs_summary(
        v_timestamp="2026-06-05T12:00:00.000001+00:00"
    )

    verdicts, note = score_transport_direction(
        "Southerly winds pushed the plume inland.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "u/v timestamp pairing SILENT" in note
    assert "mismatch" in note


def test_transport_accepts_naive_and_aware_same_utc_instant() -> None:
    summary = _gfs_summary(
        u_timestamp="2026-06-05T12:00:00",
        v_timestamp="2026-06-05T07:00:00-05:00",
    )

    verdicts, _ = score_transport_direction(
        "Southerly winds pushed the plume inland.", summary
    )

    assert verdicts["noaa_gfs"] == SUPPORTING


@pytest.mark.parametrize(
    ("u_timestamp", "v_timestamp", "reason"),
    (
        (None, MATCHED_TIMESTAMP, "missing"),
        ("not-a-timestamp", MATCHED_TIMESTAMP, "malformed"),
    ),
)
def test_transport_silences_missing_or_malformed_component_timestamp(
    u_timestamp: str | None,
    v_timestamp: str,
    reason: str,
) -> None:
    verdicts, note = score_transport_direction(
        "Southerly winds pushed the plume inland.",
        _gfs_summary(u_timestamp=u_timestamp, v_timestamp=v_timestamp),
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "u/v timestamp pairing SILENT" in note
    assert reason in note


def test_stale_component_keeps_the_pair_silent() -> None:
    verdicts, note = score_transport_direction(
        "Southerly winds pushed the plume inland.",
        _gfs_summary(v_dt_minutes=361.0),
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "age-gated SILENT" in note


def test_meteorological_state_silences_mismatched_gfs_speed() -> None:
    verdicts, note = score_meteorological_state(
        "Conditions were stagnant with barely any air movement.",
        _gfs_summary(v_timestamp="2026-06-05T18:00:00+00:00"),
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "u/v timestamp pairing SILENT" in note


def test_point_source_silences_mismatched_gfs_direction() -> None:
    verdicts, note = score_point_source_attribution(
        "Plume signature consistent with a refinery upset near 29.73N, -95.22W.",
        _gfs_summary(v_timestamp="2026-06-05T18:00:00+00:00"),
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "u/v timestamp pairing SILENT" in note


def test_calm_guard_marks_mismatched_event_speed_unevaluable() -> None:
    decisions, notes = calm_wind_source_decisions(
        _gfs_summary(v_timestamp="2026-06-05T18:00:00+00:00"),
        ("noaa_gfs",),
    )

    decision = decisions["noaa_gfs"]
    assert decision.event_speed_ms is None
    assert decision.direction_votable is False
    assert decision.reason == "missing_event_speed"
    assert any("u/v timestamp pairing SILENT" in note for note in notes)


def test_same_timestamp_different_nearest_entities_remains_timestamp_votable() -> None:
    verdicts, _ = score_transport_direction(
        "Southerly winds pushed the plume inland.",
        _gfs_summary(u_entity_id="cell-a", v_entity_id="cell-b"),
    )

    assert verdicts["noaa_gfs"] == SUPPORTING
