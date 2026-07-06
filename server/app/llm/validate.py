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

# A relative band is a fraction of the reference magnitude, which collapses to
# zero width when the reference is ~0 — a legitimate 0 reading would then
# reject every value but an exact 0. The floor keeps a minimal absolute band
# around zero. It is a degeneracy guard, not a widener: only references within
# ~floor/pct of zero are affected, so real measurements behave as before.
TOLERANCE_FLOOR = 1e-9


def within_tolerance(
    value: float,
    reference: float,
    pct: float,
    *,
    floor: float = TOLERANCE_FLOOR,
) -> bool:
    """Whether ``value`` is within a relative ``pct`` band of ``reference``.

    The band is ``pct * |reference|`` floored at ``floor`` so a near-zero
    reference cannot produce a zero-width band that only an exact match clears.
    Shared by the Phase 1 grounding check (_supports) and the Phase 2
    concentration scorer so both read near-zero readings the same way.
    """
    return abs(value - reference) <= max(pct * abs(reference), floor)

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

# Causal connectives. A claim asserting one ("X caused Y", "elevated due to Z")
# states a *relation* on top of its factual content. The relation itself cannot
# be verified lexically: the enrichment context is machine-rendered numeric
# lines that never contain causal language, so the old causal-connective-in-
# context requirement rejected every causally-phrased claim — excluding exactly
# the explanation content steps 2-4 prompt for, and counting phrasing style as
# hallucination. Phase 1 therefore grounds a causal claim on its factual
# content (term overlap, citations, numeric consistency) and records
# ``causal=True`` on the result so the analysis reports grounded x causal cuts
# separately; the relation's physical validity is judged by the Phase 2
# claim-type scorers (transport, trap, secondary formation are causal-shaped)
# and by the expert labels.
_CAUSAL_RE = re.compile(
    r"\bcaus(?:e|es|ed|ing)\b|\bdue to\b|\bbecause\b|\bled to\b"
    r"|\blead(?:s|ing)? to\b|\bresult(?:ed|ing|s)? (?:in|of|from)\b"
    r"|\bdriven by\b|\bdrove\b|\battribut(?:ed|able) to\b"
    r"|\bresponsible for\b|\btriggered\b|\bowing to\b|\bstem(?:s|med)? from\b"
)

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

# A bare (unit-less) claim number carries no metric of its own, so the unit
# guard in _match_numbers cannot reject a same-magnitude context number of a
# different metric ("index hit 45" vs "humidity 45%"). We instead require a
# salient metric token to co-occur within this many characters of *both*
# numbers. Tokens carrying a digit (no2, pm25) count as metric names; the bare
# number itself is excluded — numeric equality is handled by _supports. This
# can still be fooled by a location/time word adjacent to both numbers, but in
# the real (long) enrichment context distinct metrics separate, and it is far
# less permissive than matching any number within tolerance.
_METRIC_WINDOW = 32

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
    # The claim asserts a causal relation (see _CAUSAL_RE). Metadata, not a
    # gate: persisted so grounded/unverified x causal/descriptive is reportable.
    causal: bool = False


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
    if within_tolerance(value, ctx_value, tolerance):
        return True
    if relation == "over":
        return ctx_value >= value
    if relation == "under":
        return ctx_value <= value
    return False


def _metric_terms_near(clean_text: str, position: int) -> set[str]:
    """Salient metric tokens within _METRIC_WINDOW chars of a number.

    ``clean_text`` is already lowercased and locator-stripped — the same string
    the quantity offsets index into. Pure-number tokens are excluded: numeric
    equality is _supports's job; here we compare *what* is being measured.
    """
    lo = position - _METRIC_WINDOW
    hi = position + _METRIC_WINDOW
    terms: set[str] = set()
    for match in _TERM_RE.finditer(clean_text):
        if match.end() <= lo or match.start() >= hi:
            continue
        token = match.group()
        if token in _STOPWORDS or not any(ch.isalpha() for ch in token):
            continue
        if len(token) >= 4 or any(ch.isdigit() for ch in token):
            terms.add(token)
    return terms


def _match_numbers(
    claim_text: str,
    claim_quantities: list[tuple[float, str | None, int]],
    context_text: str,
    context_quantities: list[tuple[float, str | None, int]],
    tolerance: float,
) -> list[dict] | None:
    """Pair each claim quantity with a consistent context quantity.

    Returns the matched pairs, or None if any claim quantity has no support.
    Units must agree when both sides state one; a bare number can support a
    united one only when a metric token co-occurs near both (else "index hit
    45" would ground against "humidity 45%"). Point tolerance is relative to
    the context's measured value; threshold-worded quantities also accept any
    value past the threshold.
    """
    claim_clean = strip_locators(claim_text.lower())
    context_clean = strip_locators(context_text.lower())
    cues = threshold_cues(claim_clean)
    matched: list[dict] = []
    for value, unit, position in claim_quantities:
        relation = _threshold_relation(cues, position)
        claim_metric = (
            _metric_terms_near(claim_clean, position) if unit is None else None
        )
        support: dict | None = None
        for ctx_value, ctx_unit, ctx_position in context_quantities:
            if unit and ctx_unit and unit != ctx_unit:
                continue
            if claim_metric is not None and not (
                claim_metric & _metric_terms_near(context_clean, ctx_position)
            ):
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


# Whole-token spelling variants a model may plausibly emit for each canonical
# source (prompt.SOURCE_NAMES). Keys must stay in lockstep with SOURCE_NAMES
# (asserted in tests). Aliases exist because _TERM_RE splits on underscores
# ("noaa_gfs" -> noaa + gfs, "epa_aqs" -> epa + aqs) and models write natural
# spellings ("Sentinel-5P", "purple air"); without them the citation check
# rejected claims citing 4 of the 8 sources the prompt shows, inflating the
# Phase 1 unverified rate for exactly the models that cite diligently.
SOURCE_ALIASES: dict[str, frozenset[str]] = {
    "openaq": frozenset({"openaq"}),
    "sentinel5p": frozenset({"sentinel5p", "sentinel-5p", "s5p", "tropomi"}),
    "gfs": frozenset({"gfs", "noaa"}),
    "openweather": frozenset({"openweather", "openweathermap"}),
    "tceq": frozenset({"tceq"}),
    "purpleair": frozenset({"purpleair", "purple"}),
    "asos": frozenset({"asos", "metar"}),
    "epa_aqs": frozenset({"aqs"}),
}


def _cited_source_grounded(source: str, context_tokens: set[str]) -> bool:
    """Whether a cited source names a known data source present in the context.

    A cited source counts only when an alias of a known AERIS source
    (``SOURCE_ALIASES``) appears as a whole token in the citation *and* an
    alias of that same source appears in the context. The raw
    `src.lower() in context` substring check this replaces let an empty string
    ground (it is a substring of everything) and let any co-occurring word
    ("afternoon") pose as a source. Token matching against the known aliases
    rejects both. A multi-token citation ("noaa gfs") is grounded by its
    recognized part.
    """
    citation_tokens = set(_TERM_RE.findall(source.lower()))
    return any(
        (aliases & citation_tokens) and (aliases & context_tokens)
        for aliases in SOURCE_ALIASES.values()
    )


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
        cited = [src for src in cited_sources if src and src.strip()]
        if not cited:
            # The claim asserted a citation but named only empty strings; a
            # malformed citation must not vacuously satisfy the source check.
            sources_present = False
        else:
            context_tokens = set(_TERM_RE.findall(context_text.lower()))
            sources_present = all(
                _cited_source_grounded(src, context_tokens) for src in cited
            )

    claim_quantities = _quantities(claim_text)
    matched_numbers: list[dict] | None = []
    if claim_quantities:
        matched_numbers = _match_numbers(
            claim_text,
            claim_quantities,
            context_text,
            _quantities(context_text),
            numeric_tolerance,
        )

    causal = _CAUSAL_RE.search(claim_text.lower()) is not None

    if (
        len(matched) >= min_overlap
        and sources_present
        and matched_numbers is not None
    ):
        evidence_ref: dict = {"matched_terms": matched}
        if matched_numbers:
            evidence_ref["matched_numbers"] = matched_numbers
        return GroundingResult(GROUNDED, evidence_ref, causal=causal)
    return GroundingResult(UNVERIFIED, None, causal=causal)


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
