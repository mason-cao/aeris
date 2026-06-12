import json

import httpx
import pytest
from pydantic import BaseModel

from app.config import settings
from app.llm.client_base import LLMClient
from app.llm.gpt_client import GPTClient


class _Attribution(BaseModel):
    cause: str
    confidence: float


def _client_with(handler, **kwargs) -> GPTClient:
    transport = httpx.MockTransport(handler)
    kwargs.setdefault("api_key", "sk-test")
    return GPTClient(http_client=httpx.AsyncClient(transport=transport), **kwargs)


def _chat_response(content: str, **extra) -> dict:
    return {
        "model": "gpt-5.4-2026-03-11",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 17},
        **extra,
    }


class TestGPTClient:
    def test_is_an_llm_client(self) -> None:
        assert issubclass(GPTClient, LLMClient)

    def test_requires_an_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "openai_api_key", "")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            GPTClient()

    def test_falls_back_to_settings_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "openai_api_key", "sk-from-env")
        client = GPTClient()
        assert client._api_key == "sk-from-env"

    @pytest.mark.asyncio
    async def test_complete_posts_schema_constrained_chat_request(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["auth"] = request.headers.get("authorization")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200, json=_chat_response('{"cause": "transport", "confidence": 0.4}')
            )

        client = _client_with(handler)
        raw = await client._complete("explain this anomaly", _Attribution)

        assert captured["path"] == "/v1/chat/completions"
        assert captured["auth"] == "Bearer sk-test"
        body = captured["body"]
        assert body["model"] == "gpt-5.4"
        assert body["messages"] == [
            {"role": "user", "content": "explain this anomaly"}
        ]
        assert body["response_format"]["type"] == "json_schema"
        json_schema = body["response_format"]["json_schema"]
        assert json_schema["name"] == "_Attribution"
        # Strict mode: schema closed, every property required, properties kept.
        assert json_schema["strict"] is True
        sent = json_schema["schema"]
        assert sent["additionalProperties"] is False
        assert set(sent["required"]) == set(sent["properties"])
        assert sent["properties"] == _Attribution.model_json_schema()["properties"]
        assert raw.text == '{"cause": "transport", "confidence": 0.4}'
        assert raw.prompt_tokens == 42
        assert raw.completion_tokens == 17
        await client.close()

    @pytest.mark.asyncio
    async def test_records_resolved_model_snapshot_as_version(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response("{}"))

        client = _client_with(handler)
        assert client.model_version is None
        await client._complete("p", _Attribution)
        assert client.model_version == "gpt-5.4-2026-03-11"
        await client.close()

    @pytest.mark.asyncio
    async def test_generate_returns_structured_result(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_chat_response('{"cause": "photochemical", "confidence": 0.8}'),
            )

        client = _client_with(handler)
        result = await client.generate("prompt", _Attribution)

        assert result.parsed.cause == "photochemical"
        assert result.model_name == "gpt-5.4"
        assert result.prompt_tokens == 42
        assert result.completion_tokens == 17
        await client.close()

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "bad key"}})

        client = _client_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._complete("p", _Attribution)
        await client.close()
