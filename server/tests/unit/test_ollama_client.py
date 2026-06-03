import json

import httpx
import pytest
from pydantic import BaseModel

from app.llm.client_base import LLMClient
from app.llm.ollama_client import OllamaClient


class _Attribution(BaseModel):
    cause: str
    confidence: float


def _client_with(handler, **kwargs) -> OllamaClient:
    transport = httpx.MockTransport(handler)
    return OllamaClient(http_client=httpx.AsyncClient(transport=transport), **kwargs)


class TestOllamaClient:
    def test_is_an_llm_client(self) -> None:
        assert issubclass(OllamaClient, LLMClient)

    @pytest.mark.asyncio
    async def test_complete_posts_to_generate_endpoint_and_parses_usage(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["method"] = request.method
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "response": '{"cause": "photochemical", "confidence": 0.8}',
                    "prompt_eval_count": 42,
                    "eval_count": 17,
                },
            )

        client = _client_with(handler, model="llama3:8b")
        raw = await client._complete("explain this anomaly")

        assert captured["method"] == "POST"
        assert captured["path"] == "/api/generate"
        assert captured["body"]["model"] == "llama3:8b"
        assert captured["body"]["format"] == "json"
        assert captured["body"]["stream"] is False
        assert raw.text == '{"cause": "photochemical", "confidence": 0.8}'
        assert raw.prompt_tokens == 42
        assert raw.completion_tokens == 17
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_returns_structured_result(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "response": '{"cause": "transport", "confidence": 0.4}',
                    "prompt_eval_count": 100,
                    "eval_count": 20,
                },
            )

        client = _client_with(handler, model="llama3:8b")
        result = await client.generate("prompt", _Attribution)

        assert result.parsed.cause == "transport"
        assert result.model_name == "llama3:8b"
        assert result.prompt_tokens == 100
        assert result.completion_tokens == 20
        await client.close()

    @pytest.mark.asyncio
    async def test_uses_configured_base_url(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            return httpx.Response(200, json={"response": "{}"})

        client = _client_with(
            handler, model="llama3:8b", base_url="http://server.local:11434"
        )
        await client._complete("p")

        assert captured["url"] == "http://server.local:11434/api/generate"
        await client.close()

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="model not loaded")

        client = _client_with(handler, model="llama3:8b")
        with pytest.raises(httpx.HTTPStatusError):
            await client._complete("p")
        await client.close()
