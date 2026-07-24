import re

import httpx
from pydantic import BaseModel

from app.config import settings
from app.llm.client_base import LLMClient, LLMParseError, RawCompletion

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
# The cloud baseline (spec's "Gemini 3 Thinking" has no API id). Flash
# rather than pro: pro has zero free-tier quota, flash runs free — decided
# 2026-06-11, note the substitution in the methodology.
# GA, not preview: gemini-3-flash-preview 503'd / read-timed-out (>120s) on
# every real structured-output request in the 2026-06-18 dry-run, while a
# trivial request returned 200. gemini-3.5-flash (GA) completes the same
# chain (~77s). Switched 2026-06-18 — record the GA substitution too.
# 3.5 -> 3.6 on 2026-07-24, after billing moved the project to the standard
# tier (the free tier's 20 requests/day/model killed B19 iteration 002):
# same $1.50/MTok input, output $7.50 vs $9.00; verified serving
# schema-constrained JSON that day. Iterations 001/002 ran 3.5-flash, so
# gemini columns are not comparable across that boundary — record this
# substitution in the methodology with the other two.
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_REQUEST_TIMEOUT = 120.0
# Free-tier Gemini quotas are per-minute, so 429s are expected during eval
# sweeps; retried here (unlike OpenAI) because the API tells us how long to
# wait via RetryInfo. 5xx/timeouts are retried too (the preview model 503'd on
# every real request; the GA model is reliable, but this is cheap insurance).
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2.0
RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_RETRY_DELAY_RE = re.compile(r"^([0-9.]+)s$")


class GeminiClient(LLMClient):
    """LLMClient backed by the Gemini ``generateContent`` REST API.

    Cloud comparison baseline only (Gemini 3 Thinking). Decoding is
    constrained via ``responseJsonSchema``; thought parts are dropped from
    the response so only the answer text reaches the JSON parser.
    ``completion_tokens`` records billable output usage: visible candidate
    tokens plus separately reported thinking tokens.
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

    max_http_retries = MAX_RETRIES
    http_backoff_base_seconds = BACKOFF_BASE_SECONDS

    @staticmethod
    def _server_advised_delay(response: httpx.Response) -> float | None:
        """RetryInfo ``retryDelay`` in seconds if the body carries one, else None."""
        try:
            details = response.json()["error"]["details"]
            for detail in details:
                match = _RETRY_DELAY_RE.match(detail.get("retryDelay", ""))
                if match:
                    return float(match.group(1))
        except (ValueError, KeyError, TypeError):
            pass
        return None

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
        response = await self._send_with_retry(
            lambda: client.post(
                f"{self._base_url}/models/{self.model_name}:generateContent",
                headers={"x-goog-api-key": self._api_key},
                json=payload,
                timeout=self._request_timeout,
            ),
            retry_statuses=RETRYABLE_STATUS,
            server_delay=self._server_advised_delay,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("modelVersion"):
            self.model_version = data["modelVersion"]
        candidates = data.get("candidates") or []
        content = candidates[0].get("content", {}) if candidates else {}
        parts = content.get("parts", [])
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text:
            reason = (
                candidates[0].get("finishReason")
                if candidates
                else data.get("promptFeedback", {}).get("blockReason")
            )
            raise LLMParseError(
                f"{self.model_name} returned no usable content (reason={reason})"
            )
        usage = data.get("usageMetadata", {})
        candidate_tokens = usage.get("candidatesTokenCount")
        raw_thought_tokens = usage.get("thoughtsTokenCount", 0)
        if (
            type(candidate_tokens) is int
            and candidate_tokens >= 0
            and type(raw_thought_tokens) is int
            and raw_thought_tokens >= 0
        ):
            billable_output_tokens = candidate_tokens + raw_thought_tokens
        else:
            billable_output_tokens = None
        return RawCompletion(
            text=text,
            prompt_tokens=usage.get("promptTokenCount"),
            completion_tokens=billable_output_tokens,
        )
