import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Collection
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

# A freeze sweep is hundreds of sequential cloud calls; one transient blip must
# not drop a cell. Retry only server-side transients — 4xx (bad key, bad
# request, exhausted quota) fail identically on retry, so they propagate.
RETRYABLE_STATUS: frozenset[int] = frozenset({500, 502, 503, 504})
DEFAULT_MAX_HTTP_RETRIES = 2
DEFAULT_HTTP_BACKOFF_BASE_SECONDS = 1.0


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
    max_http_retries: int = DEFAULT_MAX_HTTP_RETRIES
    http_backoff_base_seconds: float = DEFAULT_HTTP_BACKOFF_BASE_SECONDS

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

    def _backoff_seconds(self, attempt: int) -> float:
        return self.http_backoff_base_seconds * (2**attempt)

    async def _send_with_retry(
        self,
        send: Callable[[], Awaitable[httpx.Response]],
        *,
        retry_statuses: Collection[int] = RETRYABLE_STATUS,
        server_delay: Callable[[httpx.Response], float | None] | None = None,
    ) -> httpx.Response:
        """Send one request with bounded exponential backoff on transients.

        Retries transport errors / timeouts and any response whose status is in
        ``retry_statuses``, up to ``max_http_retries`` times. ``server_delay``,
        if given, may return a server-advised wait (e.g. Gemini RetryInfo) that
        overrides the backoff for a status retry. Returns the final response —
        the caller still validates its status — or re-raises the transport
        error once retries are exhausted.
        """
        for attempt in range(self.max_http_retries + 1):
            try:
                response = await send()
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt == self.max_http_retries:
                    raise
                delay = self._backoff_seconds(attempt)
                logger.warning(
                    "LLM request transport error, retrying",
                    extra={
                        "model": self.model_name,
                        "attempt": attempt + 1,
                        "delay_s": delay,
                        "error": type(exc).__name__,
                    },
                )
                await asyncio.sleep(delay)
                continue

            if response.status_code in retry_statuses and attempt < self.max_http_retries:
                advised = server_delay(response) if server_delay is not None else None
                delay = advised if advised is not None else self._backoff_seconds(attempt)
                logger.warning(
                    "LLM request failed, retrying",
                    extra={
                        "model": self.model_name,
                        "attempt": attempt + 1,
                        "status": response.status_code,
                        "delay_s": delay,
                    },
                )
                await asyncio.sleep(delay)
                continue

            return response

        raise AssertionError("retry loop exited without returning")  # pragma: no cover

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
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> GenerationResult:
        """Generate structured output validated against schema, retrying on parse failure."""
        if max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
        errors: list[str] = []
        # Accumulate across attempts: a failed retry still costs wall-clock time
        # the eval pays for, so the reported latency must include it.
        total_latency_ms = 0.0

        for attempt in range(1, max_attempts + 1):
            start = clock()
            completion = await self._complete(prompt, schema)
            total_latency_ms += (clock() - start) * 1000

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
                latency_ms=round(total_latency_ms, 1),
                attempts=attempt,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )

        raise LLMParseError(
            f"{self.model_name} failed to produce valid {schema.__name__} "
            f"after {max_attempts} attempts: {errors}"
        )
