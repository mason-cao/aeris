import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.parser import ClaimDraft

# Phase 1 - retrieval-grounded factuality check (the CLAUDE.md-mandated
# hallucination gate). For each claim we ask the FActScore-style question: is
# this claim's content present in the retrieved enrichment context the model
# was given? Grounding requires lexical term overlap, a cited-source presence
# check, and numeric consistency: any quantity stated in the claim must be
# consistent with the context (with a compatible unit when both sides carry
# one). Without the numeric requirement, "no2 exceeded 80 ppb" grounds
# against a context reporting 30 ppb — which would silently inflate the
# Phase 1 -> Phase 2 delta. Phase 1 is independent of Phase 2 corroboration.
#
# Numeric consistency is threshold-aware: "exceeded 80" is supported by any
# compatible context value at/above 80 (the claim is true of the context),
# not just by a value within ±25% of 80 — treating threshold claims as point
# values rejects true claims and disagrees with the Phase 2 concentration
# scorer's >= semantics. "below 2" gets the mirrored <= rule.

GROUNDED = "grounded"
UNVERIFIED = "unverified"

# Aligned with the Phase 2 concentration_elevation tolerance (±25%) so the
# Phase 1 -> Phase 2 delta can't be attributed to a strictness gap.
NUMERIC_TOLERANCE = 0.25

# Threshold cue words. A quantity preceded by one of these in the claim is
# matched directionally instead of by point tolerance. Shared with the
# Phase 2 concentration scorer (via threshold_cues) so both phases read
# "exceeded N" the same way. Word-bounded so "stopped" is not "topped".
_OVER_CUE_RE = re.compile(
    r"\bexceed|\babove\b|\bover\s|\bsurpass|\btopped\b|\breached\b"
    r"|\bgreater than\b|\bmore than\b|>"
)
_UNDER_CUE_RE = re.compile(
    r"\bbelow\b|\bunder\s|\bless than\b|\bfewer than\b|\bat most\b|<"
)


def threshold_cues(text: str) -> list[tuple[int, str]]:
    """(offset, "over" | "under") for each threshold cue in lowercased text."""
    cues = [(m.start(), "over") for m in _OVER_CUE_RE.finditer(text)]
    cues += [(m.start(), "under") for m in _UNDER_CUE_RE.finditer(text)]
    return sorted(cues)

_TERM_RE = re.compile(r"[a-z0-9.\-µ/]+")

# Clock times and ISO dates are locators, not measurements; strip them before
# quantity extraction so "between 14:00 and 18:00" needs no numeric support.
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def strip_locators(text: str) -> str:
    """Blank out clock times and ISO dates, preserving character positions."""

    def blank(match: re.Match) -> str:
        return " " * len(match.group())

    return _DATE_RE.sub(blank, _TIME_RE.sub(blank, text))

_QUANT_RE = re.compile(
    r"(?<![\w.\-])(-?\d+(?:\.\d+)?)\s*"
    r"(µg/m³|µg/m3|ug/m3|m/s|°c|°f|hpa|ppb|ppm|mph|km|mm|%)?"
    r"(?![\w.])"
)

_UNIT_ALIASES = {
    "µg/m³": "ug/m3",
    "µg/m3": "ug/m3",
    "ug/m3": "ug/m3",
    "°c": "c",
    "°f": "f",
}

_STOPWORDS = frozenset(
    {
        "this",
        "that",
        "with",
        "were",
        "from",
        "into",
        "near",
        "over",
        "during",
        "hours",
        "data",
        "context",
        "reports",
        "about",
        "there",
        "their",
        "have",
        "been",
        "which",
        "when",
        "where",
        "level",
    }
)


@dataclass
class GroundingResult:
    verdict: str
    evidence_ref: dict | None


def _quantities(text: str) -> list[tuple[float, str | None, int]]:
    """(value, unit, character offset) for every quantity in the text."""
    cleaned = strip_locators(text.lower())
    quantities: list[tuple[float, str | None, int]] = []
    for match in _QUANT_RE.finditer(cleaned):
        raw_unit = match.group(2)
        unit = _UNIT_ALIASES.get(raw_unit, raw_unit) if raw_unit else None
        quantities.append((float(match.group(1)), unit, match.start()))
    return quantities


def _threshold_relation(cues: list[tuple[int, str]], position: int) -> str:
    """How the quantity at ``position`` relates to the measurement it cites.

    A threshold cue word anywhere before the quantity makes it directional
    ("exceeded 80" -> any value >= 80 supports). The nearest preceding cue
    wins so "stayed below 5 after topping 40" reads each number correctly.
    """
    relation = "approx"
    for offset, kind in cues:
        if offset >= position:
            break
        relation = kind
    return relation


def _supports(
    value: float,
    ctx_value: float,
    relation: str,
    tolerance: float,
) -> bool:
    """Whether one context quantity is consistent with one claim quantity."""
    if abs(value - ctx_value) <= tolerance * abs(ctx_value):
        return True
    if relation == "over":
        return ctx_value >= value
    if relation == "under":
        return ctx_value <= value
    return False


def _match_numbers(
    claim_text: str,
    claim_quantities: list[tuple[float, str | None, int]],
    context_quantities: list[tuple[float, str | None, int]],
    tolerance: float,
) -> list[dict] | None:
    """Pair each claim quantity with a consistent context quantity.

    Returns the matched pairs, or None if any claim quantity has no support.
    Units must agree when both sides state one; a bare number can support a
    united one. Point tolerance is relative to the context's measured value;
    threshold-worded quantities also accept any value past the threshold.
    """
    cues = threshold_cues(strip_locators(claim_text.lower()))
    matched: list[dict] = []
    for value, unit, position in claim_quantities:
        relation = _threshold_relation(cues, position)
        support: dict | None = None
        for ctx_value, ctx_unit, _ctx_position in context_quantities:
            if unit and ctx_unit and unit != ctx_unit:
                continue
            if _supports(value, ctx_value, relation, tolerance):
                support = {
                    "claim": value,
                    "context": ctx_value,
                    "unit": unit or ctx_unit,
                    "relation": relation,
                }
                break
        if support is None:
            return None
        matched.append(support)
    return matched


def _salient_terms(text: str) -> set[str]:
    terms: set[str] = set()
    for token in _TERM_RE.findall(text.lower()):
        if any(ch.isdigit() for ch in token):
            terms.add(token)
        elif len(token) >= 4 and token not in _STOPWORDS:
            terms.add(token)
    return terms


def check_grounding(
    claim_text: str,
    context_text: str,
    *,
    cited_sources: Sequence[str] | None = None,
    min_overlap: int = 2,
    numeric_tolerance: float = NUMERIC_TOLERANCE,
) -> GroundingResult:
    """Verify a claim against the context the model was shown."""
    matched = sorted(_salient_terms(claim_text) & _salient_terms(context_text))

    sources_present = True
    if cited_sources:
        context_lower = context_text.lower()
        sources_present = all(src.lower() in context_lower for src in cited_sources)

    claim_quantities = _quantities(claim_text)
    matched_numbers: list[dict] | None = []
    if claim_quantities:
        matched_numbers = _match_numbers(
            claim_text,
            claim_quantities,
            _quantities(context_text),
            numeric_tolerance,
        )

    if len(matched) >= min_overlap and sources_present and matched_numbers is not None:
        evidence_ref: dict = {"matched_terms": matched}
        if matched_numbers:
            evidence_ref["matched_numbers"] = matched_numbers
        return GroundingResult(GROUNDED, evidence_ref)
    return GroundingResult(UNVERIFIED, None)


def ground_claim_drafts(
    drafts: Sequence[ClaimDraft],
    context_text: str,
) -> list[GroundingResult]:
    """Run the Phase 1 grounding check over every extracted claim."""
    return [
        check_grounding(
            draft.claim_text,
            context_text,
            cited_sources=draft.cited_sources,
        )
        for draft in drafts
    ]
