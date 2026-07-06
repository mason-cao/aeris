import asyncio
from collections.abc import Coroutine
from unittest.mock import AsyncMock

import pytest

from app.db import session as session_module


def _run(coro: Coroutine):
    """Run ``coro`` on a fresh loop, like the CLIs' ``asyncio.run`` — minus
    the policy side effect.

    ``asyncio.run`` registers its loop with the event-loop policy and, on
    close, calls ``set_event_loop(None)`` — clobbering the policy's pointer to
    the session-scoped loop pytest-asyncio installed for the wider suite. Any
    later ``loop_scope="session"`` test (the asyncpg-pinned integration tests)
    then dies with 'There is no current event loop'. Supplying a
    ``loop_factory`` gives the identical fresh-loop-per-call lifecycle these
    tests exercise while leaving the policy untouched.
    """
    with asyncio.Runner(loop_factory=asyncio.new_event_loop) as runner:
        return runner.run(coro)


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

        _run(run())
        fake.dispose.assert_awaited_once()

    def test_disposes_engine_when_body_raises(self, monkeypatch) -> None:
        fake = AsyncMock()
        monkeypatch.setattr(session_module, "engine", fake)

        async def run() -> None:
            async with session_module.engine_lifecycle():
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            _run(run())
        fake.dispose.assert_awaited_once()

    def test_yields_the_engine(self, monkeypatch) -> None:
        fake = AsyncMock()
        monkeypatch.setattr(session_module, "engine", fake)

        async def run():
            async with session_module.engine_lifecycle() as eng:
                return eng

        assert _run(run()) is fake
