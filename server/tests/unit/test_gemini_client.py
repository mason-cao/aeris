import json

import httpx
import pytest
from pydantic import BaseModel

from app.config import settings
from app.llm.client_base import LLMClient
from app.llm.gemini_client import GeminiClient


class _Attribution(BaseModel):
    cause: str
    confidence: float


def _client_with(handler, **kwargs) -> GeminiClient:
    transport = httpx.MockTransport(handler)
    kwargs.setdefault("api_key", "g-test")
    return GeminiClient(http_client=httpx.AsyncClient(transport=transport), **kwargs)


def _gemini_response(parts: list[dict]) -> dict:
    return {
        "modelVersion": "gemini-3-thinking-0601",
        "candidates": [{"content": {"role": "model", "parts": parts}}],
        "usageMetadata": {"promptTokenCount": 42, "candidatesTokenCount": 17},
    }


def _no_sleep(monkeypatch) -> list[float]:
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr("app.llm.gemini_client.asyncio.sleep", fake_sleep)
    return slept


class TestGeminiClient:
    def test_is_an_llm_client(self) -> None:
        assert issubclass(GeminiClient, LLMClient)

    def test_requires_an_api_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "google_api_key", "")
        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            GeminiClient()

    def test_falls_back_to_settings_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "google_api_key", "g-from-env")
        client = GeminiClient()
        assert client._api_key == "g-from-env"

    @pytest.mark.asyncio
    async def test_complete_posts_schema_constrained_generate_request(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["key_header"] = request.headers.get("x-goog-api-key")
            captured["body"] = json.loads(request.content)
            return httpx.Response(
                200,
                json=_gemini_response(
                    [{"text": '{"cause": "transport", "confidence": 0.4}'}]
                ),
            )

        client = _client_with(handler)
        raw = await client._complete("explain this anomaly", _Attribution)

        assert captured["path"] == "/v1beta/models/gemini-3-flash-preview:generateContent"
        assert captured["key_header"] == "g-test"
        body = captured["body"]
        assert body["contents"] == [
            {"parts": [{"text": "explain this anomaly"}]}
        ]
        config = body["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"] == _Attribution.model_json_schema()
        assert raw.text == '{"cause": "transport", "confidence": 0.4}'
        assert raw.prompt_tokens == 42
        assert raw.completion_tokens == 17
        await client.close()

    @pytest.mark.asyncio
    async def test_skips_thought_parts_and_joins_text_parts(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_gemini_response(
                    [
                        {"text": "internal reasoning", "thought": True},
                        {"text": '{"cause": "tra'},
                        {"text": 'nsport", "confidence": 0.4}'},
                    ]
                ),
            )

        client = _client_with(handler)
        raw = await client._complete("p", _Attribution)
        assert raw.text == '{"cause": "transport", "confidence": 0.4}'
        await client.close()

    @pytest.mark.asyncio
    async def test_records_model_version(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_gemini_response([{"text": "{}"}]))

        client = _client_with(handler)
        assert client.model_version is None
        await client._complete("p", _Attribution)
        assert client.model_version == "gemini-3-thinking-0601"
        await client.close()

    @pytest.mark.asyncio
    async def test_retries_429_with_server_advised_delay(self, monkeypatch) -> None:
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(
                    429,
                    json={
                        "error": {
                            "code": 429,
                            "details": [
                                {
                                    "@type": "type.googleapis.com/google.rpc.RetryInfo",
                                    "retryDelay": "7s",
                                }
                            ],
                        }
                    },
                )
            return httpx.Response(200, json=_gemini_response([{"text": "{}"}]))

        client = _client_with(handler)
        raw = await client._complete("p", _Attribution)

        assert raw.text == "{}"
        assert calls["n"] == 2
        assert slept == [7.0]
        await client.close()

    @pytest.mark.asyncio
    async def test_429_without_retry_info_backs_off_exponentially(
        self, monkeypatch
    ) -> None:
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] <= 2:
                return httpx.Response(429, json={"error": {"code": 429}})
            return httpx.Response(200, json=_gemini_response([{"text": "{}"}]))

        client = _client_with(handler)
        raw = await client._complete("p", _Attribution)

        assert raw.text == "{}"
        assert slept == [2.0, 4.0]
        await client.close()

    @pytest.mark.asyncio
    async def test_persistent_429_raises_after_max_attempts(self, monkeypatch) -> None:
        slept = _no_sleep(monkeypatch)
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            return httpx.Response(429, json={"error": {"code": 429}})

        client = _client_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._complete("p", _Attribution)

        assert calls["n"] == 4  # initial try + 3 retries
        assert len(slept) == 3
        await client.close()

    @pytest.mark.asyncio
    async def test_raises_on_non_429_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, json={"error": {"message": "bad schema"}})

        client = _client_with(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._complete("p", _Attribution)
        await client.close()
