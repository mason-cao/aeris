import json

import pytest
from pydantic import BaseModel

from app.llm.client_base import LLMClient, RawCompletion
from app.llm.prompt import STEP_SEQUENCE
from app.llm.reasoning_chain import ReasoningChainResult, run_reasoning_chain


class ScriptedClient(LLMClient):
    """LLMClient that returns scripted raw JSON, one per call, recording prompts."""

    model_name = "scripted"
    model_version = "v1"

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self._texts = list(texts)
        self.prompts: list[str] = []

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        self.prompts.append(prompt)
        return RawCompletion(
            text=self._texts.pop(0), prompt_tokens=10, completion_tokens=5
        )


def _step(summary: str, claims: list[dict]) -> str:
    return json.dumps({"summary": summary, "claims": claims})


def _synthesis(narrative: str, confidence: float, claims: list[dict]) -> str:
    return json.dumps(
        {
            "final_narrative": narrative,
            "stated_confidence": confidence,
            "claims": claims,
        }
    )


def _full_script() -> list[str]:
    return [
        _step("afternoon ozone spike, light winds", [{"statement": "O3 elevated", "cited_sources": ["openaq"]}]),
        _step("likely photochemical formation", [{"statement": "morning NOx present", "cited_sources": ["openaq"]}]),
        _step("met supports photochemistry", [{"statement": "winds were stagnant", "cited_sources": ["gfs"]}]),
        _synthesis("Photochemical ozone event.", 0.72, [{"statement": "local photochemical event", "cited_sources": ["openaq", "openweather"]}]),
    ]


class TestRunReasoningChain:
    @pytest.mark.asyncio
    async def test_runs_four_steps_in_order(self) -> None:
        client = ScriptedClient(_full_script())

        result = await run_reasoning_chain(
            client, anomaly_text="O3 92 ppb", enrichment_text="openaq o3 max 0.092"
        )

        assert isinstance(result, ReasoningChainResult)
        assert [s.step for s in result.steps] == list(STEP_SEQUENCE)
        assert len(client.prompts) == 4

    @pytest.mark.asyncio
    async def test_final_narrative_and_confidence_from_synthesis(self) -> None:
        client = ScriptedClient(_full_script())

        result = await run_reasoning_chain(
            client, anomaly_text="O3 92 ppb", enrichment_text="openaq o3 max 0.092"
        )

        assert result.final_narrative == "Photochemical ozone event."
        assert result.stated_confidence == pytest.approx(0.72)

    @pytest.mark.asyncio
    async def test_prior_step_summary_threaded_into_later_prompt(self) -> None:
        client = ScriptedClient(_full_script())

        await run_reasoning_chain(
            client, anomaly_text="O3 92 ppb", enrichment_text="openaq o3 max 0.092"
        )

        # Second prompt (candidate_causes) must carry the first step's summary.
        assert "afternoon ozone spike, light winds" in client.prompts[1]
        assert "afternoon ozone spike, light winds" not in client.prompts[0]

    @pytest.mark.asyncio
    async def test_aggregates_latency_and_tokens(self) -> None:
        client = ScriptedClient(_full_script())

        result = await run_reasoning_chain(
            client, anomaly_text="O3 92 ppb", enrichment_text="openaq o3 max 0.092"
        )

        assert result.total_prompt_tokens == 40
        assert result.total_completion_tokens == 20
        assert result.total_latency_ms >= 0
