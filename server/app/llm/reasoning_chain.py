from dataclasses import dataclass

from pydantic import BaseModel

from app.llm.client_base import GenerationResult, LLMClient
from app.llm.prompt import (
    STEP_SEQUENCE,
    ReasoningStep,
    build_step_prompt,
    response_schema,
)


@dataclass
class StepResult:
    step: ReasoningStep
    response: BaseModel  # ReasoningStepResponse or SynthesisResponse
    generation: GenerationResult


@dataclass
class ReasoningChainResult:
    steps: list[StepResult]
    final_narrative: str
    stated_confidence: float
    total_latency_ms: float
    total_prompt_tokens: int | None
    total_completion_tokens: int | None


def _sum_optional(values: list[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return sum(present) if present else None


async def run_reasoning_chain(
    client: LLMClient,
    *,
    anomaly_text: str,
    enrichment_text: str,
) -> ReasoningChainResult:
    """Run the 4-step reasoning chain, threading each step's summary forward."""
    steps: list[StepResult] = []
    prior_summaries: dict[ReasoningStep, str] = {}

    for step in STEP_SEQUENCE:
        prompt = build_step_prompt(
            step,
            anomaly_text=anomaly_text,
            enrichment_text=enrichment_text,
            prior_summaries=prior_summaries,
        )
        generation = await client.generate(prompt, response_schema(step))
        response = generation.parsed
        steps.append(StepResult(step=step, response=response, generation=generation))
        prior_summaries[step] = getattr(response, "summary", None) or getattr(
            response, "final_narrative", ""
        )

    synthesis = steps[-1].response
    return ReasoningChainResult(
        steps=steps,
        final_narrative=synthesis.final_narrative,
        stated_confidence=synthesis.stated_confidence,
        total_latency_ms=round(sum(s.generation.latency_ms for s in steps), 1),
        total_prompt_tokens=_sum_optional([s.generation.prompt_tokens for s in steps]),
        total_completion_tokens=_sum_optional(
            [s.generation.completion_tokens for s in steps]
        ),
    )
