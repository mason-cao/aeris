from app.llm.parser import ClaimDraft, extract_claim_drafts
from app.llm.prompt import (
    MAX_CLAIMS_PER_STEP,
    ReasoningStepResponse,
    StepClaim,
    SynthesisResponse,
)


def _resp(*statements: str) -> ReasoningStepResponse:
    return ReasoningStepResponse(
        summary="s",
        claims=[StepClaim(statement=text, cited_sources=["openaq"]) for text in statements],
    )


class TestExtractClaimDrafts:
    def test_one_draft_per_claim_with_step_index(self) -> None:
        responses = [
            _resp("O3 elevated"),
            _resp("morning NOx present"),
            _resp("winds stagnant"),
            SynthesisResponse(
                final_narrative="n",
                stated_confidence=0.5,
                claims=[StepClaim(statement="local photochemical event", cited_sources=["gfs"])],
            ),
        ]

        drafts = extract_claim_drafts(responses)

        assert all(isinstance(d, ClaimDraft) for d in drafts)
        assert [d.claim_text for d in drafts] == [
            "O3 elevated",
            "morning NOx present",
            "winds stagnant",
            "local photochemical event",
        ]
        assert [d.step_index for d in drafts] == [1, 2, 3, 4]
        assert drafts[-1].cited_sources == ["gfs"]

    def test_truncates_to_cap_per_step(self) -> None:
        too_many = _resp(*[f"claim {i}" for i in range(MAX_CLAIMS_PER_STEP + 2)])

        drafts = extract_claim_drafts([too_many])

        assert len(drafts) == MAX_CLAIMS_PER_STEP

    def test_skips_blank_statements(self) -> None:
        resp = ReasoningStepResponse(
            summary="s",
            claims=[
                StepClaim(statement="real claim"),
                StepClaim(statement="   "),
                StepClaim(statement=""),
            ],
        )

        drafts = extract_claim_drafts([resp])

        assert len(drafts) == 1
        assert drafts[0].claim_text == "real claim"

    def test_claim_object_missing_statement_is_skipped(self) -> None:
        # .claims is getattr-guarded, but a malformed claim item lacking
        # .statement used to AttributeError; it must be skipped like a blank.
        class _Resp:
            claims = [object()]

        assert extract_claim_drafts([_Resp()]) == []

    def test_claim_missing_cited_sources_defaults_to_empty(self) -> None:
        class _Claim:
            statement = "real claim"

        class _Resp:
            claims = [_Claim()]

        drafts = extract_claim_drafts([_Resp()])
        assert len(drafts) == 1
        assert drafts[0].cited_sources == []
