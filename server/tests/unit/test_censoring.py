"""B15 concentration-baseline censoring and quantitative exclusion."""

from __future__ import annotations

from app.llm.corroboration import (
    SILENT,
    SUPPORTING,
    BaselineCensoringStrategy,
    baseline_censor_limit,
    censor_baseline_values,
    qualitative_elevation_verdict,
    score_claim,
    score_concentration_elevation,
)


def test_so2_limit_half_substitution_and_exact_boundary() -> None:
    limit = baseline_censor_limit("tceq", "so2")

    assert limit == 0.5
    assert censor_baseline_values(
        [-1.0, 0.2, 0.5, 0.8],
        limit=limit,
        strategy=BaselineCensoringStrategy.LIMIT_HALF,
    ) == [0.25, 0.25, 0.5, 0.8]
    assert censor_baseline_values(
        [-1.0, 0.2, 0.5, 0.8],
        limit=limit,
        strategy=BaselineCensoringStrategy.DELETE,
    ) == [0.5, 0.8]


def test_physical_zero_substitution_is_not_an_invented_mdl() -> None:
    limit = baseline_censor_limit("tceq", "no2")

    assert limit == 0.0
    assert censor_baseline_values(
        [-2.0, 0.0, 1.0],
        limit=limit,
        strategy=BaselineCensoringStrategy.LIMIT_HALF,
    ) == [0.0, 0.0, 1.0]
    assert censor_baseline_values(
        [-2.0, 0.0, 1.0],
        limit=limit,
        strategy=BaselineCensoringStrategy.DELETE,
    ) == [0.0, 1.0]
    assert baseline_censor_limit("sentinel5p", "s5p_no2_column") is None


def test_empty_and_all_censored_baselines_remain_explicit() -> None:
    assert (
        censor_baseline_values(
            [],
            limit=0.5,
            strategy=BaselineCensoringStrategy.LIMIT_HALF,
        )
        == []
    )
    assert censor_baseline_values(
        [-0.5, 0.1, 0.2],
        limit=0.5,
        strategy=BaselineCensoringStrategy.LIMIT_HALF,
    ) == [0.25, 0.25, 0.25]
    assert (
        censor_baseline_values(
            [-0.5, 0.1, 0.2],
            limit=0.5,
            strategy=BaselineCensoringStrategy.DELETE,
        )
        == []
    )


def test_qualitative_verdict_boundaries_include_sd_zero_and_empty() -> None:
    assert qualitative_elevation_verdict(2.0, [1.0, 1.0, 1.0]) == SUPPORTING
    assert qualitative_elevation_verdict(1.0, [1.0, 1.0, 1.0]) == -1
    assert qualitative_elevation_verdict(2.0, [0.0, 2.0]) == SILENT
    assert qualitative_elevation_verdict(2.000001, [0.0, 2.0]) == SUPPORTING
    assert qualitative_elevation_verdict(1.0, []) is None


def _so2_summary(nearest: float) -> dict:
    return {
        "anomaly": {"timestamp": "2026-06-05T05:00:00+00:00"},
        "sources": {
            "tceq": {
                "metrics": {
                    "so2": {
                        "unit": "ppb",
                        "nearest_in_time": {
                            "t": "2026-06-05T05:00:00+00:00",
                            "v": nearest,
                            "dt_minutes": 0.0,
                            "entity_id": "monitor",
                        },
                        "entities": [
                            {
                                "entity_id": "monitor",
                                "series": [
                                    ["2026-06-05T00:00:00+00:00", -0.2],
                                    ["2026-06-05T01:00:00+00:00", 0.2],
                                    ["2026-06-05T02:00:00+00:00", 0.5],
                                    ["2026-06-05T05:00:00+00:00", nearest],
                                ],
                            }
                        ],
                    }
                }
            }
        },
    }


def test_substitution_keeps_baseline_n_while_event_floor_stays_silent() -> None:
    supported, note = score_concentration_elevation(
        "SO2 was elevated.", _so2_summary(1.0)
    )
    below_floor, below_note = score_concentration_elevation(
        "SO2 was elevated.", _so2_summary(0.2)
    )

    assert supported["tceq"] == SUPPORTING
    assert "pre-anomaly baseline" in note
    assert below_floor["tceq"] == SILENT
    assert "below ground detection floor 0.5" in below_note


def test_so2_concentration_claim_is_explicitly_excluded_from_quantitative_use() -> None:
    so2 = score_claim("SO2 exceeded 1 ppb.", _so2_summary(1.0))
    no2 = score_claim("NO2 exceeded 1 ppb.", {})

    assert so2.quantitative_exclusion_reason == "so2_underpowered"
    assert so2.qualitative_only is False
    assert no2.quantitative_exclusion_reason is None
