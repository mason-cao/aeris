from app.llm.prompt import (
    MAX_CLAIMS_PER_STEP,
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
