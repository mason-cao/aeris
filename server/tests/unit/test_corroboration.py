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
