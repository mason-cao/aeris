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


class TestNumericGrounding:
    def test_numeric_claim_contradicted_by_context_is_unverified(self) -> None:
        # Vocabulary overlaps, but the claimed magnitude is far off the
        # context's measurement — lexical overlap alone must not ground it.
        result = check_grounding(
            "no2 exceeded 80 ppb in the afternoon",
            "openaq reports no2 at 30 ppb elevated during the afternoon hours",
            cited_sources=["openaq"],
        )

        assert result.verdict == UNVERIFIED
        assert result.evidence_ref is None

    def test_numeric_claim_within_tolerance_is_grounded(self) -> None:
        result = check_grounding(
            "no2 reached 78 ppb in the afternoon",
            "openaq reports no2 peaked at 80 ppb during the afternoon hours",
            cited_sources=["openaq"],
        )

        assert result.verdict == GROUNDED
        assert result.evidence_ref is not None
        assert result.evidence_ref["matched_numbers"] == [
            {"claim": 78.0, "context": 80.0, "unit": "ppb"}
        ]

    def test_unit_mismatch_is_unverified(self) -> None:
        # Same magnitude, different unit — not the same measurement.
        result = check_grounding(
            "no2 reached 80 ppb in the afternoon",
            "openaq reports no2 elevated at 80 µg/m3 during the afternoon hours",
            cited_sources=["openaq"],
        )

        assert result.verdict == UNVERIFIED

    def test_time_of_day_tokens_are_not_measurements(self) -> None:
        # 14:00/18:00 are clock times, not quantities needing numeric support.
        result = check_grounding(
            "ozone was elevated between 14:00 and 18:00",
            CONTEXT,
            cited_sources=["openaq"],
        )

        assert result.verdict == GROUNDED


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
