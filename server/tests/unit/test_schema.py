import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.schema import create_tables


@pytest.mark.asyncio
async def test_create_tables_supports_sqlite_without_timescaledb(tmp_path) -> None:
    db_path = tmp_path / "schema.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)

    try:
        await create_tables(engine)

        async with engine.connect() as conn:
            result = await conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' ORDER BY name"
                )
            )
            tables = {row[0] for row in result}
    finally:
        await engine.dispose()

    assert {"data_points", "data_sources"} <= tables
