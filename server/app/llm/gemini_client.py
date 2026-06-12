import asyncio
import logging
import re

import httpx
from pydantic import BaseModel

from app.config import settings
from app.llm.client_base import LLMClient, RawCompletion

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# The cloud baseline (spec's "Gemini 3 Thinking" has no API id). Flash
# rather than pro: pro has zero free-tier quota, flash runs free — decided
# 2026-06-11, note the substitution in the methodology.
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_REQUEST_TIMEOUT = 120.0
# Free-tier Gemini quotas are per-minute, so 429s are expected during eval
# sweeps; retried here (unlike OpenAI) because the API tells us how long to
# wait via RetryInfo.
MAX_429_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0

_RETRY_DELAY_RE = re.compile(r"^([0-9.]+)s$")


class GeminiClient(LLMClient):
    """LLMClient backed by the Gemini ``generateContent`` REST API.

    Cloud comparison baseline only (Gemini 3 Thinking). Decoding is
    constrained via ``responseJsonSchema``; thought parts are dropped from
    the response so only the answer text reaches the JSON parser.
    ``model_version`` is filled in from the ``modelVersion`` the API reports.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ) -> None:
        super().__init__(http_client=http_client)
        self.model_name = model
        self.model_version = None
        key = api_key if api_key is not None else settings.google_api_key
        if not key:
            raise ValueError(
                "Google API key not configured; set GOOGLE_API_KEY in server/.env"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout

    @staticmethod
    def _retry_delay(response: httpx.Response, attempt: int) -> float:
        """Server-advised RetryInfo delay, or exponential backoff without it."""
        try:
            details = response.json()["error"]["details"]
            for detail in details:
                match = _RETRY_DELAY_RE.match(detail.get("retryDelay", ""))
                if match:
                    return float(match.group(1))
        except (ValueError, KeyError, TypeError):
            pass
        return BACKOFF_BASE_SECONDS * (2**attempt)

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        client = await self._get_client()
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema.model_json_schema(),
                # Pinned for eval reproducibility, matching the Ollama client.
                "temperature": 0.0,
            },
        }
        for attempt in range(MAX_429_RETRIES + 1):
            response = await client.post(
                f"{self._base_url}/models/{self.model_name}:generateContent",
                headers={"x-goog-api-key": self._api_key},
                json=payload,
                timeout=self._request_timeout,
            )
            if response.status_code != 429 or attempt == MAX_429_RETRIES:
                break
            delay = self._retry_delay(response, attempt)
            logger.warning(
                "Gemini rate limited, retrying",
                extra={"model": self.model_name, "attempt": attempt + 1, "delay_s": delay},
            )
            await asyncio.sleep(delay)
        response.raise_for_status()
        data = response.json()
        if data.get("modelVersion"):
            self.model_version = data["modelVersion"]
        parts = data["candidates"][0]["content"].get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        usage = data.get("usageMetadata", {})
        return RawCompletion(
            text=text,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=usage.get("candidatesTokenCount"),
        )
