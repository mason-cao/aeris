from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=(settings.aeris_env == "development"),
    pool_size=5,
    max_overflow=10,
)

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with async_session() as session:
        yield session


@asynccontextmanager
async def engine_lifecycle():
    """Dispose the process-wide async engine on exit.

    CLI entry points run ``asyncio.run(_amain())``; the asyncpg pool must be
    released before that event loop closes, otherwise shutdown logs 'Event
    loop is closed'. FastAPI uses ``get_session`` and disposes via the app
    lifespan instead, so this is for the CLI ``_amain`` bodies. Yields the
    engine for callers that want it.
    """
    try:
        yield engine
    finally:
        await engine.dispose()
