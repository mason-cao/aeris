"""Phase 2 — cross-source corroboration scorer.

For each Phase-1-grounded claim about an atmospheric anomaly, score it against
the agreement of the four data sources (OpenAQ, Sentinel-5P, NOAA GFS,
OpenWeather), which sense different facets of one shared physical state through
largely independent measurement processes. Design + claim taxonomy:
docs/specs/2026-05-21-corroboration-scorer-design.md.

This module currently provides the shared aggregator that collapses per-source
verdicts into the scalar ``corroboration_score`` + ``evidence_n``. The ten
per-claim-type scorers build on top of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

# Per-source verdict on a single claim. A source either supports the claim,
# contradicts it, or is silent (no data bearing on it within the window).
SUPPORTING = 1
CONTRADICTING = -1
SILENT = 0

# low_corroboration_flag threshold (memo: metadata signal, not a scoring gate).
_LOW_CORROBORATION_SCORE = -0.5
_LOW_CORROBORATION_MIN_EVIDENCE = 2


@dataclass(frozen=True)
class CorroborationResult:
    """Aggregated cross-source verdict for one claim.

    ``corroboration_score`` is ``None`` (never 0) when every source is silent,
    so "no evidence" is not conflated with "balanced evidence" downstream.
    ``evidence_n`` travels with the score because (score=+1, n=1) and
    (score=+1, n=3) carry different evidential weight.
    """

    corroboration_score: float | None
    evidence_n: int
    supporting: int
    contradicting: int
    unverified: bool
    per_source_verdicts: dict[str, int]


def aggregate_verdicts(per_source_verdicts: Mapping[str, int]) -> CorroborationResult:
    """Collapse per-source verdicts into a scalar score and evidence count.

    ``score = (supporting - contradicting) / evidence_n`` in [-1, +1];
    ``None`` when ``evidence_n == 0`` (every source silent).
    """
    supporting = sum(1 for v in per_source_verdicts.values() if v == SUPPORTING)
    contradicting = sum(1 for v in per_source_verdicts.values() if v == CONTRADICTING)
    evidence_n = supporting + contradicting
    verdicts = dict(per_source_verdicts)

    if evidence_n == 0:
        return CorroborationResult(
            corroboration_score=None,
            evidence_n=0,
            supporting=0,
            contradicting=0,
            unverified=True,
            per_source_verdicts=verdicts,
        )

    return CorroborationResult(
        corroboration_score=(supporting - contradicting) / evidence_n,
        evidence_n=evidence_n,
        supporting=supporting,
        contradicting=contradicting,
        unverified=False,
        per_source_verdicts=verdicts,
    )


def low_corroboration_flag(score: float | None, *, evidence_n: int) -> bool:
    """Phase 2 metadata flag: strongly contradicted across >= 2 sources.

    Not a gate — the raw ``corroboration_score`` is what the research analysis
    correlates against expert labels; this is a convenience signal for
    downstream product code.
    """
    if score is None:
        return False
    return (
        score <= _LOW_CORROBORATION_SCORE
        and evidence_n >= _LOW_CORROBORATION_MIN_EVIDENCE
    )
