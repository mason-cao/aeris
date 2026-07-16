"""B17 nearest-event-entity concentration baselines."""

from __future__ import annotations

from typing import Any

import pytest

from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    SUPPORTING,
    qualitative_elevation_verdict,
    score_concentration_elevation,
)


def _entity(entity_id: str, rows: list[list[str | float]]) -> dict[str, Any]:
    return {
        "entity_id": entity_id,
        "distance_km": 1.0,
        "n_points": len(rows),
        "series": rows,
    }


def _rows(
    values: list[float],
    *,
    timezone_suffix: str = "+00:00",
    final_timestamp: str = "09:00:00",
) -> list[list[str | float]]:
    timestamps = ("06:00:00", "07:00:00", final_timestamp)
    return [
        [f"2026-06-05T{timestamp}{timezone_suffix}", value]
        for timestamp, value in zip(timestamps, values, strict=True)
    ]


def _summary(
    entities: list[dict[str, Any]],
    *,
    nearest_entity_id: str | None = "near",
    nearest_value: float = 20.0,
    anomaly_timezone_suffix: str = "+00:00",
) -> dict[str, Any]:
    nearest: dict[str, Any] = {
        "t": f"2026-06-05T12:00:00{anomaly_timezone_suffix}",
        "v": nearest_value,
        "dt_minutes": 0.0,
    }
    if nearest_entity_id is not None:
        nearest["entity_id"] = nearest_entity_id
    return {
        "schema_version": 1,
        "anomaly": {
            "timestamp": f"2026-06-05T12:00:00{anomaly_timezone_suffix}",
            "source": "noaa_gfs",
            "metric": "pbl_height",
        },
        "sources": {
            "tceq": {
                "metrics": {
                    "ozone": {
                        "unit": "ppb",
                        "nearest_in_time": nearest,
                        "entities": entities,
                    }
                }
            }
        },
    }


def _verdict(summary: dict[str, Any]) -> tuple[int, str]:
    verdicts, note = score_concentration_elevation("Ozone was elevated.", summary)
    return verdicts["tceq"], note


def test_station_match_prevents_network_spatial_confounding() -> None:
    summary = _summary(
        [
            _entity("near", _rows([10.0, 10.0, 10.0])),
            _entity("far", _rows([100.0, 100.0, 100.0])),
        ]
    )

    verdict, note = _verdict(summary)

    assert verdict == SUPPORTING
    assert "entity_id=near" in note
    assert "baseline_n=3" in note
    assert "network" not in note


def test_exact_three_hour_endpoint_is_included() -> None:
    verdict, note = _verdict(
        _summary([_entity("near", _rows([10.0, 10.0, 10.0]))])
    )

    assert verdict == SUPPORTING
    assert "baseline_n=3" in note


def test_one_second_inside_gap_is_excluded_and_below_floor_is_silent() -> None:
    verdict, note = _verdict(
        _summary(
            [
                _entity(
                    "near",
                    _rows(
                        [10.0, 10.0, 10.0],
                        final_timestamp="09:00:01",
                    ),
                )
            ]
        )
    )

    assert verdict == SILENT
    assert "baseline_n=2" in note
    assert "insufficient" in note


@pytest.mark.parametrize("nearest_entity_id", [None, "unknown"])
def test_missing_or_unknown_nearest_entity_never_falls_back_to_pool(
    nearest_entity_id: str | None,
) -> None:
    verdict, note = _verdict(
        _summary(
            [_entity("other", _rows([10.0, 10.0, 10.0]))],
            nearest_entity_id=nearest_entity_id,
        )
    )

    assert verdict == SILENT
    assert "no station-matched pre-anomaly baseline" in note


def test_zero_spread_station_baseline_remains_evaluable() -> None:
    verdict, _note = _verdict(
        _summary(
            [_entity("near", _rows([5.0, 5.0, 5.0]))],
            nearest_value=6.0,
        )
    )

    assert verdict == SUPPORTING


def test_malformed_matched_station_series_is_silent() -> None:
    rows = _rows([10.0, 10.0, 10.0])
    rows.append(["not-a-timestamp", 10.0])

    verdict, note = _verdict(_summary([_entity("near", rows)]))

    assert verdict == SILENT
    assert "malformed_station_series" in note


def test_b15_substitution_is_applied_after_station_matching() -> None:
    verdict, note = _verdict(
        _summary(
            [
                _entity("near", _rows([-2.0, 2.0, 2.0])),
                _entity("far", _rows([100.0, 100.0, 100.0])),
            ],
            nearest_value=2.4,
        )
    )

    assert verdict == SUPPORTING
    assert "entity_id=near" in note


def test_naive_and_aware_timestamps_produce_the_same_station_verdict() -> None:
    aware = _summary([_entity("near", _rows([10.0, 11.0, 9.0]))])
    naive = _summary(
        [
            _entity(
                "near",
                _rows([10.0, 11.0, 9.0], timezone_suffix=""),
            )
        ],
        anomaly_timezone_suffix="",
    )

    assert _verdict(aware) == _verdict(naive)


def test_absolute_claim_does_not_require_station_baseline() -> None:
    summary = _summary([], nearest_entity_id=None, nearest_value=20.0)

    verdicts, note = score_concentration_elevation(
        "Ozone exceeded 15 ppb.", summary
    )

    assert verdicts["tceq"] == SUPPORTING
    assert "baseline" not in note


def test_old_pooled_baseline_would_contradict_spatial_case() -> None:
    pooled_values = [10.0, 10.0, 10.0, 100.0, 100.0, 100.0]

    assert sum(pooled_values) / len(pooled_values) == 55.0
    assert qualitative_elevation_verdict(20.0, pooled_values) == CONTRADICTING
