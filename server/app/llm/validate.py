import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.parser import ClaimDraft

# Phase 1 - retrieval-grounded factuality check (the CLAUDE.md-mandated
# hallucination gate). For each claim we ask the FActScore-style question: is
# this claim's content present in the retrieved enrichment context the model
# was given? Grounding requires lexical term overlap, a cited-source presence
# check, and numeric consistency: any quantity stated in the claim must appear
# in the context within tolerance (with a compatible unit when both sides
# carry one). Without the numeric requirement, "no2 exceeded 80 ppb" grounds
# against a context reporting 30 ppb — which would silently inflate the
# Phase 1 -> Phase 2 delta. Phase 1 is independent of Phase 2 corroboration.

GROUNDED = "grounded"
UNVERIFIED = "unverified"

# Aligned with the Phase 2 concentration_elevation tolerance (±25%) so the
# Phase 1 -> Phase 2 delta can't be attributed to a strictness gap.
NUMERIC_TOLERANCE = 0.25

_TERM_RE = re.compile(r"[a-z0-9.\-µ/]+")

# Clock times and ISO dates are locators, not measurements; strip them before
# quantity extraction so "between 14:00 and 18:00" needs no numeric support.
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

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


def _quantities(text: str) -> list[tuple[float, str | None]]:
    cleaned = _DATE_RE.sub(" ", _TIME_RE.sub(" ", text.lower()))
    quantities: list[tuple[float, str | None]] = []
    for match in _QUANT_RE.finditer(cleaned):
        raw_unit = match.group(2)
        unit = _UNIT_ALIASES.get(raw_unit, raw_unit) if raw_unit else None
        quantities.append((float(match.group(1)), unit))
    return quantities


def _match_numbers(
    claim_quantities: list[tuple[float, str | None]],
    context_quantities: list[tuple[float, str | None]],
    tolerance: float,
) -> list[dict] | None:
    """Pair each claim quantity with a compatible context quantity.

    Returns the matched pairs, or None if any claim quantity has no support.
    Units must agree when both sides state one; a bare number can support a
    united one. Tolerance is relative to the context's measured value.
    """
    matched: list[dict] = []
    for value, unit in claim_quantities:
        support: dict | None = None
        for ctx_value, ctx_unit in context_quantities:
            if unit and ctx_unit and unit != ctx_unit:
                continue
            if abs(value - ctx_value) <= tolerance * abs(ctx_value):
                support = {
                    "claim": value,
                    "context": ctx_value,
                    "unit": unit or ctx_unit,
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
            claim_quantities, _quantities(context_text), numeric_tolerance
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
