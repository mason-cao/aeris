import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


class RawCompletion(BaseModel):
    """Raw model output plus usage metadata from a single model call."""

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass
class GenerationResult:
    """A structured generation: the parsed schema instance plus call metadata."""

    parsed: BaseModel
    raw_text: str
    model_name: str
    model_version: str | None
    latency_ms: float
    attempts: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class LLMParseError(Exception):
    """Raised when a model's output cannot be parsed into the requested schema."""


class LLMClient(ABC):
    """Abstract base class for all LLM clients (local and cloud).

    Subclasses implement _complete(): one call to the model returning raw text
    and token counts. generate() wraps it with JSON-schema parsing, retry on
    parse failure, and latency capture, mirroring the BaseCollector discipline.
    """

    model_name: str
    model_version: str | None = None

    def __init__(self, http_client: httpx.AsyncClient | None = None) -> None:
        self._client = http_client
        self._owns_client = http_client is None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        """Call the model once and return raw text plus token counts.

        ``schema`` is the pydantic model the caller will parse the output
        into; backends that support schema-constrained decoding (OpenAI
        ``json_schema``, Gemini ``responseJsonSchema``) pass it to the API,
        others may ignore it.
        """

    async def generate(
        self,
        prompt: str,
        schema: type[BaseModel],
        max_attempts: int = 2,
    ) -> GenerationResult:
        """Generate structured output validated against schema, retrying on parse failure."""
        errors: list[str] = []

        for attempt in range(1, max_attempts + 1):
            start = time.monotonic()
            completion = await self._complete(prompt, schema)
            latency_ms = (time.monotonic() - start) * 1000

            try:
                parsed = schema.model_validate_json(completion.text)
            except ValidationError as e:
                errors.append(f"Attempt {attempt}: {type(e).__name__}")
                logger.warning(
                    "LLM output failed schema validation",
                    extra={
                        "model": self.model_name,
                        "attempt": attempt,
                        "schema": schema.__name__,
                    },
                )
                continue

            return GenerationResult(
                parsed=parsed,
                raw_text=completion.text,
                model_name=self.model_name,
                model_version=self.model_version,
                latency_ms=round(latency_ms, 1),
                attempts=attempt,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        raise LLMParseError(
            f"{self.model_name} failed to produce valid {schema.__name__} "
            f"after {max_attempts} attempts: {errors}"
        )
