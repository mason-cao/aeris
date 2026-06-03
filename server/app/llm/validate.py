import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.llm.parser import ClaimDraft

# Phase 1 - retrieval-grounded factuality check (the CLAUDE.md-mandated
# hallucination gate). For each claim we ask the FActScore-style question: is
# this claim's content present in the retrieved enrichment context the model
# was given? This v1 uses lexical term overlap plus a cited-source presence
# check; the interface (verdict + evidence_ref) is stable so the matching
# strategy can be upgraded (entity- or statement-level) without downstream
# changes. Phase 1 is independent of Phase 2 corroboration.

GROUNDED = "grounded"
UNVERIFIED = "unverified"

_TERM_RE = re.compile(r"[a-z0-9.\-µ/]+")

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
) -> GroundingResult:
    """Verify a claim against the context the model was shown."""
    matched = sorted(_salient_terms(claim_text) & _salient_terms(context_text))

    sources_present = True
    if cited_sources:
        context_lower = context_text.lower()
        sources_present = all(src.lower() in context_lower for src in cited_sources)

    if len(matched) >= min_overlap and sources_present:
        return GroundingResult(GROUNDED, {"matched_terms": matched})
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
