import pytest
from pydantic import BaseModel

from app.llm.client_base import (
    GenerationResult,
    LLMClient,
    LLMParseError,
    RawCompletion,
)


class _Attribution(BaseModel):
    cause: str
    confidence: float


class MockLLMClient(LLMClient):
    """Concrete LLMClient returning scripted raw completions, for testing."""

    model_name = "mock-model"
    model_version = "v1"

    def __init__(self, responses: list[RawCompletion]) -> None:
        super().__init__()
        self._responses = list(responses)
        self.calls = 0

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        self.calls += 1
        return self._responses.pop(0)


def _completion(text: str) -> RawCompletion:
    return RawCompletion(text=text, prompt_tokens=10, completion_tokens=5)


class TestGenerate:
    @pytest.mark.asyncio
    async def test_parses_valid_json_into_schema(self) -> None:
        client = MockLLMClient(
            [_completion('{"cause": "photochemical", "confidence": 0.8}')]
        )

        result = await client.generate("prompt", _Attribution)

        assert isinstance(result, GenerationResult)
        assert isinstance(result.parsed, _Attribution)
        assert result.parsed.cause == "photochemical"
        assert result.parsed.confidence == pytest.approx(0.8)
        assert result.attempts == 1
        assert result.model_name == "mock-model"
        assert result.model_version == "v1"

    @pytest.mark.asyncio
    async def test_captures_tokens_and_latency(self) -> None:
        client = MockLLMClient(
            [_completion('{"cause": "transport", "confidence": 0.3}')]
        )

        result = await client.generate("prompt", _Attribution)

        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_rejects_nonpositive_max_attempts(self) -> None:
        # max_attempts<1 means the retry loop never runs; rather than masquerade
        # as a parse failure ("after 0 attempts"), reject the bad argument and
        # never touch the model.
        client = MockLLMClient([_completion('{"cause": "x", "confidence": 0.5}')])

        with pytest.raises(ValueError, match="max_attempts"):
            await client.generate("prompt", _Attribution, max_attempts=0)

        assert client.calls == 0

    @pytest.mark.asyncio
    async def test_latency_accumulates_across_retries(self) -> None:
        # The reported latency is the eval's per-cell cost; a failed retry's
        # time must count, not just the final successful attempt.
        ticks = iter([0.0, 0.5, 0.5, 1.2])  # attempt1: 500ms, attempt2: 700ms
        client = MockLLMClient(
            [
                _completion("not json"),
                _completion('{"cause": "x", "confidence": 0.5}'),
            ]
        )

        result = await client.generate(
            "prompt", _Attribution, max_attempts=2, clock=lambda: next(ticks)
        )

        assert result.attempts == 2
        assert result.latency_ms == pytest.approx(1200.0)

    @pytest.mark.asyncio
    async def test_retries_once_on_invalid_json_then_succeeds(self) -> None:
        client = MockLLMClient(
            [
                _completion("not json at all"),
                _completion('{"cause": "stagnation", "confidence": 0.5}'),
            ]
        )

        result = await client.generate("prompt", _Attribution)

        assert result.parsed.cause == "stagnation"
        assert result.attempts == 2
        assert client.calls == 2

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self) -> None:
        client = MockLLMClient(
            [_completion("nope"), _completion("still not json")]
        )

        with pytest.raises(LLMParseError):
            await client.generate("prompt", _Attribution, max_attempts=2)

        assert client.calls == 2

    @pytest.mark.asyncio
    async def test_missing_required_field_is_a_parse_failure(self) -> None:
        client = MockLLMClient([_completion('{"cause": "no confidence here"}')])

        with pytest.raises(LLMParseError):
            await client.generate("prompt", _Attribution, max_attempts=1)
