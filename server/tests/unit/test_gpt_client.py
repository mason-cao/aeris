import json

import httpx
import pytest
from pydantic import BaseModel

from app.config import settings
from app.llm.client_base import LLMClient, LLMParseError
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


def _no_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.llm.client_base.asyncio.sleep", fake_sleep)
    return slept


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
    async def test_complete_raises_parse_error_on_empty_choices(self) -> None:
        # A safety-filtered 200 can return no choices; must surface as
        # LLMParseError (harness parse_failure) not an IndexError crash.
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "gpt-5.4", "choices": []})

        client = _client_with(handler)
        with pytest.raises(LLMParseError):
            await client._complete("p", _Attribution)
        await client.close()

    @pytest.mark.asyncio
    async def test_complete_raises_parse_error_on_null_content(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "gpt-5.4",
                    "choices": [
                        {"message": {"role": "assistant", "content": None},
                         "finish_reason": "content_filter"}
                    ],
                },
            )

        client = _client_with(handler)
        with pytest.raises(LLMParseError):
            await client._complete("p", _Attribution)
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


class TestGPTRetry:
    """A freeze sweep is ~200 sequential calls; one transient blip must not
    drop a cell. gpt_client had no retry — these pin the bounded backoff."""

    @pytest.mark.asyncio
    async def test_retries_5xx_then_succeeds(self, monkeypatch) -> None:
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(503, json={"error": {"message": "overloaded"}})
            return httpx.Response(
                200, json=_chat_response('{"cause": "x", "confidence": 0.5}')
            )

        client = _client_with(handler)
        raw = await client._complete("p", _Attribution)

        assert raw.text == '{"cause": "x", "confidence": 0.5}'
        assert calls["n"] == 2
        assert slept == [1.0]
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_on_timeout_then_succeeds(self, monkeypatch) -> None:
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("read timed out", request=request)
            return httpx.Response(
                200, json=_chat_response('{"cause": "x", "confidence": 0.5}')
            )

        client = _client_with(handler)
        raw = await client._complete("p", _Attribution)

        assert raw.text == '{"cause": "x", "confidence": 0.5}'
        assert calls["n"] == 2
        assert slept == [1.0]
        await client.close()

    @pytest.mark.asyncio
    async def test_persistent_5xx_raises_after_bounded_retries(
        self, monkeypatch
    ) -> None:
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(500, json={"error": {"message": "boom"}})

        client = _client_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._complete("p", _Attribution)

        assert calls["n"] == 3  # initial try + 2 retries
        assert slept == [1.0, 2.0]
        await client.close()

    @pytest.mark.asyncio
    async def test_structural_429_is_not_retried(self, monkeypatch) -> None:
        # The dry-run 429 was insufficient_quota (account/billing), not a rate
        # burst — backoff cannot fix it, so gpt does not retry 429.
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(
                429, json={"error": {"message": "insufficient_quota"}}
            )

        client = _client_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._complete("p", _Attribution)

        assert calls["n"] == 1
        assert slept == []
        await client.close()
