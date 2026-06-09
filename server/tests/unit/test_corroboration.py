"""Phase 2 corroboration scorer — shared aggregator.

The aggregator turns per-source verdicts (+1 supporting / -1 contradicting /
0 silent) into the scalar ``corroboration_score`` and ``evidence_n`` that the
research analysis correlates against expert labels. Spec:
docs/specs/2026-05-21-corroboration-scorer-design.md.
"""

import math

from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    SUPPORTING,
    aggregate_verdicts,
    low_corroboration_flag,
    score_concentration_elevation,
)


def test_all_silent_returns_null_and_unverified():
    result = aggregate_verdicts({"openaq": SILENT, "noaa_gfs": SILENT})
    assert result.corroboration_score is None
    assert result.evidence_n == 0
    assert result.unverified is True


def test_empty_verdicts_is_unverified():
    result = aggregate_verdicts({})
    assert result.corroboration_score is None
    assert result.evidence_n == 0
    assert result.unverified is True


def test_all_supporting_scores_plus_one():
    result = aggregate_verdicts(
        {"openaq": SUPPORTING, "sentinel5p": SUPPORTING, "noaa_gfs": SUPPORTING}
    )
    assert result.corroboration_score == 1.0
    assert result.evidence_n == 3
    assert result.supporting == 3
    assert result.contradicting == 0
    assert result.unverified is False


def test_all_contradicting_scores_minus_one():
    result = aggregate_verdicts(
        {"openaq": CONTRADICTING, "openweather": CONTRADICTING}
    )
    assert result.corroboration_score == -1.0
    assert result.evidence_n == 2
    assert result.contradicting == 2


def test_mixed_verdicts_average_over_evidence_n():
    result = aggregate_verdicts(
        {
            "openaq": SUPPORTING,
            "sentinel5p": SILENT,
            "noaa_gfs": CONTRADICTING,
            "openweather": SUPPORTING,
        }
    )
    assert result.supporting == 2
    assert result.contradicting == 1
    assert result.evidence_n == 3
    assert math.isclose(result.corroboration_score, 1 / 3, rel_tol=1e-9)
    assert result.unverified is False


def test_silent_sources_excluded_from_evidence_n():
    result = aggregate_verdicts(
        {"openaq": SUPPORTING, "sentinel5p": SILENT, "noaa_gfs": SILENT}
    )
    assert result.evidence_n == 1
    assert result.corroboration_score == 1.0


def test_result_preserves_per_source_verdicts():
    verdicts = {"openaq": SUPPORTING, "noaa_gfs": CONTRADICTING}
    result = aggregate_verdicts(verdicts)
    assert result.per_source_verdicts == verdicts


def test_low_corroboration_flag_requires_strong_negative_and_two_sources():
    # Strongly contradicted across >= 2 independent sources -> flagged.
    assert low_corroboration_flag(-0.6, evidence_n=2) is True
    assert low_corroboration_flag(-1.0, evidence_n=3) is True
    # A lone contradicting source is too little evidence to flag.
    assert low_corroboration_flag(-1.0, evidence_n=1) is False
    # Weak disagreement is not a flag.
    assert low_corroboration_flag(-0.4, evidence_n=3) is False
    # No evidence at all -> never flagged.
    assert low_corroboration_flag(None, evidence_n=0) is False


# --- concentration_elevation (headline type 1: OpenAQ + Sentinel-5P) ---


def _summary_with(metrics_by_source: dict) -> dict:
    """A minimal enrichment summary carrying {source: {metric: {...}}}."""
    return {
        "schema_version": 1,
        "sources": {
            src: {"metrics": metrics}
            for src, metrics in metrics_by_source.items()
        },
    }


def test_concentration_threshold_claim_supported_by_openaq():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 60.0, "max": 85.0, "mean": 72.0},
                    "nearest_in_time": {"v": 82.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "Ground-level NO2 exceeded 80 ppb in the afternoon.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_concentration_threshold_claim_contradicted_by_openaq():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 8.0, "max": 15.0, "mean": 11.0},
                    "nearest_in_time": {"v": 12.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation("NO2 exceeded 80 ppb.", summary)
    assert verdicts["openaq"] == CONTRADICTING


def test_concentration_unmatched_pollutant_is_silent():
    summary = _summary_with(
        {
            "openaq": {
                "ozone": {
                    "unit": "ppb",
                    "value_range": {"min": 20.0, "max": 40.0, "mean": 30.0},
                    "nearest_in_time": {"v": 35.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation("NO2 exceeded 80 ppb.", summary)
    assert verdicts.get("openaq", SILENT) == SILENT


def test_concentration_qualitative_elevated_uses_window_baseline():
    summary = _summary_with(
        {
            "openaq": {
                "pm25": {
                    "unit": "ug/m3",
                    "value_range": {"min": 10.0, "max": 55.0, "mean": 20.0},
                    "nearest_in_time": {"v": 52.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "PM2.5 was elevated across the area.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_concentration_sentinel_column_supports_no2_claim():
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_no2_column": {
                    "unit": "mol/m^2",
                    "value_range": {"min": 4.0e-5, "max": 9.0e-5, "mean": 6.0e-5},
                    "nearest_in_time": {"v": 8.5e-5},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "Tropospheric NO2 was elevated.", summary
    )
    assert verdicts["sentinel5p"] == SUPPORTING
