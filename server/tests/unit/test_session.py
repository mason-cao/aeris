import asyncio
from unittest.mock import AsyncMock

import pytest

from app.db import session as session_module


class TestEngineLifecycle:
    """engine_lifecycle() must release the asyncpg pool for CLI entry points.

    Each CLI runs ``asyncio.run(_amain())``; without disposing the engine the
    pool outlives the loop and shutdown logs 'Event loop is closed'.
    """

    def test_disposes_engine_on_normal_exit(self, monkeypatch) -> None:
        fake = AsyncMock()
        monkeypatch.setattr(session_module, "engine", fake)

        async def run() -> None:
            async with session_module.engine_lifecycle():
                pass

        asyncio.run(run())
        fake.dispose.assert_awaited_once()

    def test_disposes_engine_when_body_raises(self, monkeypatch) -> None:
        fake = AsyncMock()
        monkeypatch.setattr(session_module, "engine", fake)

        async def run() -> None:
            async with session_module.engine_lifecycle():
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            asyncio.run(run())
        fake.dispose.assert_awaited_once()

    def test_yields_the_engine(self, monkeypatch) -> None:
        fake = AsyncMock()
        monkeypatch.setattr(session_module, "engine", fake)

        async def run():
            async with session_module.engine_lifecycle() as eng:
                return eng

        assert asyncio.run(run()) is fake
