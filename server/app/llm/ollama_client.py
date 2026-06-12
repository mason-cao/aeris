import httpx
from pydantic import BaseModel

from app.llm.client_base import LLMClient, RawCompletion

DEFAULT_BASE_URL = "http://localhost:11434"
# Local 8B generation runs well past the 30s collector timeout; allow longer.
DEFAULT_REQUEST_TIMEOUT = 120.0
# Pinned for the eval: Ollama's default temperature (0.8) makes every sweep
# unreproducible and confounds the calibration comparison across models.
DEFAULT_TEMPERATURE = 0.0


class OllamaClient(LLMClient):
    """LLMClient backed by a local Ollama server's HTTP API.

    Uses Ollama's JSON mode (`format: "json"`) so the response body is
    parseable into the requested pydantic schema by the base generate().
    """

    def __init__(
        self,
        model: str = "llama3:8b",
        base_url: str = DEFAULT_BASE_URL,
        http_client: httpx.AsyncClient | None = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        super().__init__(http_client=http_client)
        self.model_name = model
        self.model_version = None
        self._base_url = base_url.rstrip("/")
        self._request_timeout = request_timeout
        self._temperature = temperature

    # Schema-constrained decoding stays off here: per the phase plan the
    # local model runs plain JSON mode, so its parse-failure rate is a
    # measured result rather than masked by the decoder.
    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        client = await self._get_client()
        response = await client.post(
            f"{self._base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": self._temperature},
            },
            timeout=self._request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        return RawCompletion(
            text=data.get("response", ""),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )
