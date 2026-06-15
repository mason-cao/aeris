import httpx
from pydantic import BaseModel

from app.config import settings
from app.llm.client_base import LLMClient, LLMParseError, RawCompletion

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.4"
# Thinking-tier responses routinely exceed the 30s collector timeout.
DEFAULT_REQUEST_TIMEOUT = 120.0


def _strict_json_schema(node: object) -> object:
    """Rewrite a pydantic JSON schema for OpenAI strict mode.

    Strict decoding requires every object to be closed
    (``additionalProperties: false``) and every property required; pydantic
    leaves defaulted fields optional, which strict mode rejects outright.
    """
    if isinstance(node, dict):
        out = {key: _strict_json_schema(value) for key, value in node.items()}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out
    if isinstance(node, list):
        return [_strict_json_schema(value) for value in node]
    return node


class GPTClient(LLMClient):
    """LLMClient backed by the OpenAI chat completions API.

    Cloud comparison baseline only (GPT-5.4 Standard Thinking). Requests use
    ``response_format: json_schema`` so decoding is constrained to the target
    pydantic schema; ``model_version`` is filled in from the resolved model
    snapshot the API reports (e.g. ``gpt-5.4-2026-03-11``) after the first
    call, so explanation rows record exactly which snapshot answered.
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
        key = api_key if api_key is not None else settings.openai_api_key
        if not key:
            raise ValueError(
                "OpenAI API key not configured; set OPENAI_API_KEY in server/.env"
            )
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                # No temperature/top_p here: the thinking tier rejects
                # sampling overrides, so GPT runs at the API default — record
                # that asymmetry in the methodology note alongside the pinned
                # temperature=0 on the Ollama and Gemini clients.
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": _strict_json_schema(schema.model_json_schema()),
                        "strict": True,
                    },
                },
            },
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if data.get("model"):
            self.model_version = data["model"]
        usage = data.get("usage", {})
        choices = data.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            finish = choices[0].get("finish_reason") if choices else None
            raise LLMParseError(
                f"{self.model_name} returned no usable content "
                f"(finish_reason={finish})"
            )
        return RawCompletion(
            text=content,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )
