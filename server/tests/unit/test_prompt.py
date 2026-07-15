import math

import pytest
from pydantic import ValidationError

from app.llm.prompt import (
    MAX_CLAIMS_PER_STEP,
    SOURCE_NAMES,
    STEP_SEQUENCE,
    ReasoningStep,
    ReasoningStepResponse,
    SynthesisResponse,
    build_step_prompt,
    response_schema,
)


ANOMALY = "O3 at 92 ppb, 2026-07-16 15:00 CT, near the Ship Channel"
ENRICHMENT = "openaq o3 max 0.092 ppm; gfs wind 1.2 m/s southerly; openweather clear"


class TestStepSequence:
    def test_four_steps_in_reasoning_order(self) -> None:
        assert STEP_SEQUENCE == (
            ReasoningStep.PHYSICAL_SIGNATURE,
            ReasoningStep.CANDIDATE_CAUSES,
            ReasoningStep.EVIDENCE_EVALUATION,
            ReasoningStep.SYNTHESIS,
        )

    def test_response_schema_is_synthesis_only_for_last_step(self) -> None:
        assert response_schema(ReasoningStep.SYNTHESIS) is SynthesisResponse
        for step in STEP_SEQUENCE[:-1]:
            assert response_schema(step) is ReasoningStepResponse


class TestBuildStepPrompt:
    @pytest.mark.parametrize("step", STEP_SEQUENCE)
    def test_every_step_requires_atomic_self_contained_claims(
        self, step: ReasoningStep
    ) -> None:
        prompt = build_step_prompt(
            step,
            anomaly_text=ANOMALY,
            enrichment_text=ENRICHMENT,
        )

        atomic = "one atomic, self-contained statement"
        lowered = prompt.lower()
        assert atomic in lowered
        assert "name its subject explicitly" in lowered
        assert "antecedent outside the claim" in lowered
        assert "independent assertions" in lowered
        assert lowered.index(atomic) < lowered.index("respond only with json")

    def test_first_step_includes_anomaly_and_context_but_no_prior_reasoning(self) -> None:
        prompt = build_step_prompt(
            ReasoningStep.PHYSICAL_SIGNATURE,
            anomaly_text=ANOMALY,
            enrichment_text=ENRICHMENT,
        )
        assert ANOMALY in prompt
        assert ENRICHMENT in prompt
        assert "JSON" in prompt
        assert str(MAX_CLAIMS_PER_STEP) in prompt
        assert "REASONING SO FAR" not in prompt

    def test_later_step_threads_prior_summaries(self) -> None:
        prompt = build_step_prompt(
            ReasoningStep.CANDIDATE_CAUSES,
            anomaly_text=ANOMALY,
            enrichment_text=ENRICHMENT,
            prior_summaries={
                ReasoningStep.PHYSICAL_SIGNATURE: "afternoon ozone spike, light winds"
            },
        )
        assert "REASONING SO FAR" in prompt
        assert "afternoon ozone spike, light winds" in prompt

    def test_framing_names_every_collected_source(self) -> None:
        # The model may cite only what SYSTEM_FRAMING names; a source shown in
        # the data context but missing here forces mis-attribution or a
        # citation-gate failure (tceq/purpleair/asos were missing pre-fix).
        prompt = build_step_prompt(
            ReasoningStep.PHYSICAL_SIGNATURE,
            anomaly_text=ANOMALY,
            enrichment_text=ENRICHMENT,
        )
        for name in SOURCE_NAMES:
            assert name in prompt, name

    def test_research_framing_not_legal_attribution(self) -> None:
        prompt = build_step_prompt(
            ReasoningStep.SYNTHESIS,
            anomaly_text=ANOMALY,
            enrichment_text=ENRICHMENT,
        )
        assert "not legal attribution" in prompt.lower()

    def test_synthesis_prompt_requests_confidence_and_narrative(self) -> None:
        prompt = build_step_prompt(
            ReasoningStep.SYNTHESIS,
            anomaly_text=ANOMALY,
            enrichment_text=ENRICHMENT,
        )
        assert "final_narrative" in prompt
        assert "stated_confidence" in prompt


class TestSynthesisConfidenceBounds:
    """stated_confidence is a probability in [0, 1]. An out-of-range or
    non-finite value (a model emitting 5, -1, or NaN) must be rejected at parse
    time rather than flowing into the corroboration confidence analysis.
    """

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    def test_in_range_confidence_accepted(self, value: float) -> None:
        response = SynthesisResponse(final_narrative="x", stated_confidence=value)

        assert response.stated_confidence == value

    @pytest.mark.parametrize("value", [5.0, -1.0, 1.0001, math.nan, math.inf])
    def test_out_of_range_confidence_rejected(self, value: float) -> None:
        with pytest.raises(ValidationError):
            SynthesisResponse(final_narrative="x", stated_confidence=value)
