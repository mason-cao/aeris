import logging

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import Base

logger = logging.getLogger(__name__)


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables and enable TimescaleDB hypertable on data_points."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    if engine.dialect.name == "postgresql":
        await configure_postgres_compatibility(engine)
        await configure_timescale(engine)


async def configure_postgres_compatibility(engine: AsyncEngine) -> None:
    """Keep existing development databases compatible with the current ORM.

    Runs outside the create_all transaction so a failure here cannot roll back
    table creation or crash startup. The dedup index is created in its own
    transaction because CREATE UNIQUE INDEX guards index *existence*, not
    duplicate *data* — on a dirty dev DB that already holds duplicate rows it
    fails, which is caught and logged rather than aborting startup.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "ALTER TABLE data_points "
                "ADD COLUMN IF NOT EXISTS unit VARCHAR(32) NOT NULL DEFAULT 'unknown'"
            )
        )
        await conn.execute(
            text(
                "ALTER TABLE data_points "
                "ADD COLUMN IF NOT EXISTS source_entity_id "
                "VARCHAR(128) NOT NULL DEFAULT 'unknown'"
            )
        )
        logger.info("data_points compatibility columns configured")

    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_data_points_dedup "
                    'ON data_points (source, metric, source_entity_id, "timestamp")'
                )
            )
            logger.info("data_points dedup index configured")
    except SQLAlchemyError as exc:
        logger.warning(
            "data_points dedup index skipped (likely duplicate rows): %s", exc
        )


async def configure_timescale(engine: AsyncEngine) -> None:
    """Enable TimescaleDB if available without rolling back base tables."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
            logger.info("TimescaleDB extension enabled")

            await conn.execute(
                text(
                    "SELECT create_hypertable('data_points', 'timestamp', "
                    "if_not_exists => TRUE, migrate_data => TRUE)"
                )
            )
            logger.info("data_points hypertable configured")
    except SQLAlchemyError as exc:
        logger.warning("TimescaleDB setup skipped: %s", exc)


async def drop_tables(engine: AsyncEngine) -> None:
    """Drop all tables. Use only in tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("All tables dropped")
