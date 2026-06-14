from collections.abc import AsyncGenerator

import pytest_asyncio
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.models import Base


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_engine(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("aeris-db") / "aeris.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}", echo=False)

    # SQLite leaves FK enforcement off by default; cascade behavior would not match Postgres without this.
    @event.listens_for(engine.sync_engine, "connect")
    def _enable_sqlite_fk(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # pysqlite/aiosqlite drives transactions itself by default, which breaks the
    # SAVEPOINT nesting db_session relies on for isolation. Hand BEGIN back to
    # SQLAlchemy so the per-test outer transaction and its nested savepoints
    # commit/roll back as expected.
    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_defer_begin_to_sqlalchemy(dbapi_conn, _connection_record):
        dbapi_conn.isolation_level = None

    @event.listens_for(engine.sync_engine, "begin")
    def _sqlite_emit_begin(conn):
        conn.exec_driver_sql("BEGIN")

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    # Run each test inside an outer transaction that is rolled back at teardown.
    # join_transaction_mode="create_savepoint" routes the session's own
    # commit()/rollback() through a SAVEPOINT, so code under test can commit
    # freely while every write stays quarantined to this test.
    connection = await test_engine.connect()
    transaction = await connection.begin()
    session = AsyncSession(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    try:
        yield session
    finally:
        await session.close()
        if transaction.is_active:
            await transaction.rollback()
        await connection.close()
