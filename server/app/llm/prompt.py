from enum import Enum

from pydantic import BaseModel, Field

# Centralized prompt templates and step-output schemas for the 4-step reasoning
# chain. Per project convention, no prompt strings live outside this module.

MAX_CLAIMS_PER_STEP = 3

SOURCE_NAMES = ("openaq", "sentinel5p", "gfs", "openweather")


class ReasoningStep(str, Enum):
    PHYSICAL_SIGNATURE = "physical_signature"
    CANDIDATE_CAUSES = "candidate_causes"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    SYNTHESIS = "synthesis"


STEP_SEQUENCE: tuple[ReasoningStep, ...] = (
    ReasoningStep.PHYSICAL_SIGNATURE,
    ReasoningStep.CANDIDATE_CAUSES,
    ReasoningStep.EVIDENCE_EVALUATION,
    ReasoningStep.SYNTHESIS,
)


class StepClaim(BaseModel):
    statement: str
    cited_sources: list[str] = Field(default_factory=list)


class ReasoningStepResponse(BaseModel):
    summary: str
    claims: list[StepClaim] = Field(default_factory=list)


class SynthesisResponse(BaseModel):
    final_narrative: str
    stated_confidence: float
    claims: list[StepClaim] = Field(default_factory=list)


def response_schema(step: ReasoningStep) -> type[BaseModel]:
    if step is ReasoningStep.SYNTHESIS:
        return SynthesisResponse
    return ReasoningStepResponse


SYSTEM_FRAMING = (
    "You are an atmospheric scientist explaining an air-quality anomaly in "
    "Houston, Texas. Treat this as scientific hypothesis generation for a "
    "research evaluation, not legal attribution: name the likely physical "
    "causes even when the evidence is partial. When you reference a data "
    "source, cite it by one of these names: " + ", ".join(SOURCE_NAMES) + "."
)


_STEP_INSTRUCTIONS: dict[ReasoningStep, str] = {
    ReasoningStep.PHYSICAL_SIGNATURE: (
        "Step 1 of 4 - physical signature. Describe what the data shows about "
        "this anomaly: which pollutant, how elevated, when, and where. State "
        "only what is observable in the data provided."
    ),
    ReasoningStep.CANDIDATE_CAUSES: (
        "Step 2 of 4 - candidate causes. List the most plausible physical "
        "causes of the signature described above."
    ),
    ReasoningStep.EVIDENCE_EVALUATION: (
        "Step 3 of 4 - evidence evaluation. For each candidate cause, weigh the "
        "supporting and contradicting evidence found in the data."
    ),
    ReasoningStep.SYNTHESIS: (
        "Step 4 of 4 - synthesis. Commit to the single most likely explanation, "
        "write a short final narrative, and state your confidence from 0 to 1."
    ),
}


def _schema_instruction(step: ReasoningStep) -> str:
    cap = f" Include at most {MAX_CLAIMS_PER_STEP} claims."
    if step is ReasoningStep.SYNTHESIS:
        return (
            'Respond ONLY with JSON of the form {"final_narrative": string, '
            '"stated_confidence": number from 0 to 1, "claims": '
            '[{"statement": string, "cited_sources": [string]}]}.' + cap
        )
    return (
        'Respond ONLY with JSON of the form {"summary": string, "claims": '
        '[{"statement": string, "cited_sources": [string]}]}.' + cap
    )


def build_step_prompt(
    step: ReasoningStep,
    *,
    anomaly_text: str,
    enrichment_text: str,
    prior_summaries: dict[ReasoningStep, str] | None = None,
) -> str:
    prior_summaries = prior_summaries or {}
    parts = [
        SYSTEM_FRAMING,
        "",
        "ANOMALY:",
        anomaly_text,
        "",
        "DATA CONTEXT:",
        enrichment_text,
    ]
    if prior_summaries:
        parts += ["", "REASONING SO FAR:"]
        for prior in STEP_SEQUENCE:
            if prior in prior_summaries:
                parts.append(f"- {prior.value}: {prior_summaries[prior]}")
    parts += ["", _STEP_INSTRUCTIONS[step], "", _schema_instruction(step)]
    return "\n".join(parts)
