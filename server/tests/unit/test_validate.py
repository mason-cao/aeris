from app.llm.parser import ClaimDraft
from app.llm.validate import (
    GROUNDED,
    TOLERANCE_FLOOR,
    UNVERIFIED,
    GroundingResult,
    _supports,
    check_grounding,
    ground_claim_drafts,
    within_tolerance,
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
            {"claim": 78.0, "context": 80.0, "unit": "ppb", "relation": "over"}
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


class TestThresholdGrounding:
    """Threshold-worded quantities match directionally, not by point tolerance.

    "exceeded 80" is a true statement about a context reporting 120; treating
    it as a point value would reject the claim Phase 2's concentration scorer
    accepts, skewing the Phase 1 -> Phase 2 delta.
    """

    def test_true_over_threshold_claim_is_grounded(self) -> None:
        result = check_grounding(
            "no2 exceeded 80 ppb at the downtown monitor",
            "openaq no2 downtown monitor: nearest to anomaly 120 ppb",
        )

        assert result.verdict == GROUNDED
        assert result.evidence_ref is not None
        assert result.evidence_ref["matched_numbers"] == [
            {"claim": 80.0, "context": 120.0, "unit": "ppb", "relation": "over"}
        ]

    def test_false_over_threshold_claim_is_unverified(self) -> None:
        # Measured 30 neither reaches 80 nor sits within point tolerance.
        result = check_grounding(
            "no2 exceeded 80 ppb at the downtown monitor",
            "openaq no2 downtown monitor: nearest to anomaly 30 ppb",
        )

        assert result.verdict == UNVERIFIED

    def test_true_under_threshold_claim_is_grounded(self) -> None:
        result = check_grounding(
            "wind speeds stayed below 5 m/s at the downtown monitor",
            "openweather wind downtown monitor: nearest to anomaly 1.5 m/s",
        )

        assert result.verdict == GROUNDED

    def test_false_under_threshold_claim_is_unverified(self) -> None:
        result = check_grounding(
            "wind speeds stayed below 5 m/s at the downtown monitor",
            "openweather wind downtown monitor: nearest to anomaly 9 m/s",
        )

        assert result.verdict == UNVERIFIED

    def test_quantity_before_threshold_word_keeps_point_semantics(self) -> None:
        # The 3 m/s wind value precedes "exceeded", so it must still match a
        # context measurement; the 80 after it gets threshold semantics.
        result = check_grounding(
            "winds of 3 m/s while no2 exceeded 80 ppb downtown",
            "openweather downtown winds 12 m/s; openaq no2 downtown 120 ppb",
        )

        assert result.verdict == UNVERIFIED


class TestUnitlessNumericGrounding:
    """A bare (unit-less) claim number must not ground against an arbitrary
    context number of a different metric that happens to share a magnitude.
    Without a metric check, "index hit 45" passes the hallucination gate on a
    context reporting "humidity 45%", silently inflating the Phase 1 -> Phase 2
    delta.
    """

    def test_unitless_claim_does_not_ground_against_unrelated_metric(self) -> None:
        # "index hit 45" carries no unit; the only same-magnitude context number
        # is humidity's 45%, a different measurement. The metrics near each
        # number share no token, so it must not ground.
        result = check_grounding(
            "the air quality index hit 45 in the afternoon",
            "openaq reports no2 at 30 ppb in the afternoon; "
            "gfs winds light and southerly; "
            "openweather relative humidity measured 45%",
            cited_sources=["openweather"],
        )

        assert result.verdict == UNVERIFIED
        assert result.evidence_ref is None

    def test_unitless_claim_grounds_when_metric_token_shared(self) -> None:
        # A bare number still grounds when its metric co-occurs with the context
        # number — "no2 ... 80" against an openaq no2 reading — so the fix does
        # not over-reject legitimate unit-less threshold claims.
        result = check_grounding(
            "no2 exceeded 80 at the downtown monitor",
            "openaq downtown no2 nearest to anomaly reached 120 ppb",
            cited_sources=["openaq"],
        )

        assert result.verdict == GROUNDED
        assert result.evidence_ref is not None


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


class TestWithinTolerance:
    def test_normal_magnitude_relative_band(self) -> None:
        # Within 25% of the reference passes; outside it fails. Unchanged from
        # the previous inline expression for any non-degenerate reference.
        assert within_tolerance(78.0, 80.0, 0.25) is True
        assert within_tolerance(120.0, 80.0, 0.25) is False

    def test_zero_reference_does_not_collapse_to_exact_match(self) -> None:
        # The bug: a relative band around a ~zero reference is zero-width, so a
        # value that is also essentially zero wrongly mismatches. The floor
        # keeps a minimal absolute band so the degenerate case behaves.
        assert within_tolerance(0.0, 0.0, 0.25) is True
        assert within_tolerance(5e-10, 0.0, 0.25) is True

    def test_floor_does_not_widen_band_for_real_magnitudes(self) -> None:
        # The floor is a degeneracy guard only: a genuinely different value
        # near a zero reference must still fail. We do not ground "0.5" on "0".
        assert within_tolerance(0.5, 0.0, 0.25) is False
        assert within_tolerance(2e-9, 0.0, 0.25) is False

    def test_floor_is_configurable(self) -> None:
        assert within_tolerance(2e-9, 0.0, 0.25, floor=1e-8) is True
        assert TOLERANCE_FLOOR == 1e-9

    def test_supports_zero_reading_routes_through_floor(self) -> None:
        # _supports is one of the two rewired sites; a near-zero reading that
        # used to mismatch a near-zero claim now grounds.
        assert _supports(0.0, 0.0, "approx", 0.25) is True
        assert _supports(5e-10, 0.0, "approx", 0.25) is True
        assert _supports(0.5, 0.0, "approx", 0.25) is False
