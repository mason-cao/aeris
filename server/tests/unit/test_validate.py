from app.llm.parser import ClaimDraft
from app.llm.validate import (
    GROUNDED,
    UNVERIFIED,
    GroundingResult,
    check_grounding,
    ground_claim_drafts,
)


CONTEXT = (
    "openaq reports ozone elevated during the afternoon hours; "
    "gfs winds light and southerly; openweather clear skies"
)


class TestCheckGrounding:
    def test_claim_present_in_context_is_grounded(self) -> None:
        result = check_grounding(
            "Ground-level ozone was elevated in the afternoon",
            CONTEXT,
            cited_sources=["openaq"],
        )

        assert isinstance(result, GroundingResult)
        assert result.verdict == GROUNDED
        assert result.evidence_ref is not None
        assert "ozone" in result.evidence_ref["matched_terms"]

    def test_fabricated_claim_is_unverified(self) -> None:
        result = check_grounding(
            "A refinery explosion released benzene downtown",
            CONTEXT,
            cited_sources=[],
        )

        assert result.verdict == UNVERIFIED
        assert result.evidence_ref is None

    def test_claim_citing_absent_source_is_unverified(self) -> None:
        # Terms overlap, but the cited source does not appear in the context.
        result = check_grounding(
            "ozone was elevated in the afternoon",
            CONTEXT,
            cited_sources=["sentinel5p"],
        )

        assert result.verdict == UNVERIFIED


class TestGroundClaimDrafts:
    def test_returns_verdict_per_draft(self) -> None:
        drafts = [
            ClaimDraft(
                claim_text="ozone elevated in the afternoon",
                step_index=1,
                cited_sources=["openaq"],
            ),
            ClaimDraft(
                claim_text="a refinery explosion released benzene",
                step_index=2,
                cited_sources=[],
            ),
        ]

        results = ground_claim_drafts(drafts, CONTEXT)

        assert [r.verdict for r in results] == [GROUNDED, UNVERIFIED]
