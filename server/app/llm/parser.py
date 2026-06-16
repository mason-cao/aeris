from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.llm.prompt import MAX_CLAIMS_PER_STEP


@dataclass
class ClaimDraft:
    """A single extracted claim awaiting Phase 1 grounding and Phase 2 scoring."""

    claim_text: str
    step_index: int
    cited_sources: list[str] = field(default_factory=list)


def extract_claim_drafts(step_responses: Sequence[BaseModel]) -> list[ClaimDraft]:
    """Flatten the per-step claims into ordered ClaimDrafts.

    step_index is 1-based over the reasoning-chain step order. Blank statements
    are dropped and each step is capped at MAX_CLAIMS_PER_STEP defensively, even
    though the prompt already requests the cap.
    """
    drafts: list[ClaimDraft] = []
    for step_index, response in enumerate(step_responses, start=1):
        kept = 0
        for claim in getattr(response, "claims", []):
            statement = (getattr(claim, "statement", "") or "").strip()
            if not statement:
                continue
            drafts.append(
                ClaimDraft(
                    claim_text=statement,
                    step_index=step_index,
                    cited_sources=list(getattr(claim, "cited_sources", []) or []),
                )
            )
            kept += 1
            if kept >= MAX_CLAIMS_PER_STEP:
                break
    return drafts
