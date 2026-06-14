import pytest
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.db.schema import configure_postgres_compatibility, create_tables


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


class _FakeDB:
    """Doubles as the engine (``begin()``) and connection (``execute()``).

    Raises on any statement containing ``fail_substring`` to simulate a
    Postgres failure (e.g. the dedup index cannot be created because a dirty
    dev DB already holds duplicate rows) without needing a real Postgres.
    """

    def __init__(self, fail_substring: str | None = None) -> None:
        self.fail_substring = fail_substring
        self.executed: list[str] = []

    async def execute(self, statement) -> None:
        sql = str(statement)
        self.executed.append(sql)
        if self.fail_substring and self.fail_substring in sql:
            raise SQLAlchemyError("simulated duplicate-row failure")

    def begin(self) -> "_FakeTxn":
        return _FakeTxn(self)


class _FakeTxn:
    def __init__(self, conn: _FakeDB) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeDB:
        return self._conn

    async def __aexit__(self, *exc) -> bool:
        return False


@pytest.mark.asyncio
async def test_compat_dedup_index_failure_does_not_propagate() -> None:
    # A dirty dev DB makes the dedup index creation fail; that must be caught
    # so it neither crashes startup nor rolls back table creation. The
    # compatibility columns must still be applied.
    db = _FakeDB(fail_substring="uq_data_points_dedup")

    await configure_postgres_compatibility(db)

    joined = " ".join(db.executed)
    assert "ADD COLUMN IF NOT EXISTS unit" in joined
    assert "source_entity_id" in joined
    assert "uq_data_points_dedup" in joined
