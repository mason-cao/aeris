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

import re
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


# ---------------------------------------------------------------------------
# Headline claim type 1 — concentration_elevation (OpenAQ + Sentinel-5P)
# ---------------------------------------------------------------------------

# Claim pollutant aliases -> OpenAQ metric name. Word-boundary matched so short
# codes ("co", "bc") don't fire inside other words ("could", "across").
_POLLUTANT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bnitrogen dioxide\b", "no2"),
    (r"\bno2\b", "no2"),
    (r"\bozone\b", "ozone"),
    (r"\bo3\b", "ozone"),
    (r"\bpm2\.?5\b", "pm25"),
    (r"\bpm10\b", "pm10"),
    (r"\bsulfur dioxide\b", "so2"),
    (r"\bsulphur dioxide\b", "so2"),
    (r"\bso2\b", "so2"),
    (r"\bcarbon monoxide\b", "co"),
    (r"\bco\b", "co"),
    (r"\bblack carbon\b", "bc"),
    (r"\bbc\b", "bc"),
)

# OpenAQ species that also have a TROPOMI column product to cross-check against.
_SENTINEL_COLUMN: dict[str, str] = {
    "no2": "s5p_no2_column",
    "so2": "s5p_so2_column",
    "co": "s5p_co_column",
}

# Words that make a numeric claim a threshold ("exceeded 80") rather than a
# point value ("was 80"). Threshold claims are met by measured >= claimed.
_THRESHOLD_WORDS: tuple[str, ...] = (
    "exceed",
    "above",
    "over ",
    "surpass",
    "topped",
    "reached",
    "greater than",
    "more than",
    ">",
)


@dataclass(frozen=True)
class ConcentrationTolerance:
    """Draft tolerances for concentration_elevation (pending Dr. Bracco)."""

    # Qualitative "elevated": the value nearest the anomaly must exceed the
    # in-window baseline (mean) by at least this ratio.
    elevated_ratio: float = 1.0


DEFAULT_CONCENTRATION_TOLERANCE = ConcentrationTolerance()


def _resolve_pollutant(claim_text: str) -> tuple[str | None, str | None]:
    """(OpenAQ metric, Sentinel column metric) named in the claim, if any."""
    lowered = claim_text.lower()
    for pattern, metric in _POLLUTANT_PATTERNS:
        if re.search(pattern, lowered):
            return metric, _SENTINEL_COLUMN.get(metric)
    return None, None


def _threshold_value(claim_text: str) -> float | None:
    """The numeric threshold in an 'exceeded N' style claim, else None."""
    lowered = claim_text.lower()
    if not any(word in lowered for word in _THRESHOLD_WORDS):
        return None
    # Strip pollutant tokens first so digits inside names (NO2, PM2.5, O3) are
    # not mistaken for the threshold value.
    cleaned = lowered
    for pattern, _metric in _POLLUTANT_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def score_concentration_elevation(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: ConcentrationTolerance = DEFAULT_CONCENTRATION_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a 'pollutant was elevated' claim against OpenAQ + Sentinel-5P.

    v1 assumes the claim's unit matches the stored metric's unit (no
    ppb<->ug/m^3 conversion). On that assumption a threshold claim is supported
    when the value nearest the anomaly meets the threshold, and a qualitative
    "elevated" is supported when the nearest value exceeds the in-window mean.
    Returns ``(per_source_verdicts, evidence_summary)``.
    """
    openaq_metric, sentinel_metric = _resolve_pollutant(claim_text)
    threshold = _threshold_value(claim_text)
    relevant = {"openaq": openaq_metric, "sentinel5p": sentinel_metric}

    sources = summary.get("sources", {})
    verdicts: dict[str, int] = {}
    notes: list[str] = []

    for source, metric in relevant.items():
        if metric is None:
            continue
        data = sources.get(source, {}).get("metrics", {}).get(metric)
        if not data or data.get("nearest_in_time", {}).get("v") is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: no {metric} in window")
            continue

        nearest = data["nearest_in_time"]["v"]
        if threshold is not None:
            verdict = SUPPORTING if nearest >= threshold else CONTRADICTING
            notes.append(
                f"{source}: {metric} nearest={nearest} vs threshold={threshold}"
            )
        else:
            mean = data.get("value_range", {}).get("mean", nearest)
            baseline = mean * tolerance.elevated_ratio
            verdict = SUPPORTING if nearest > baseline else CONTRADICTING
            notes.append(f"{source}: {metric} nearest={nearest} vs baseline={mean}")
        verdicts[source] = verdict

    if openaq_metric is None and sentinel_metric is None:
        notes.append("no recognized pollutant in claim")
    return verdicts, "; ".join(notes)
