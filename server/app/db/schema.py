import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app.db.models import Base

logger = logging.getLogger(__name__)


async def create_tables(engine: AsyncEngine) -> None:
    """Create all tables and enable TimescaleDB hypertable on data_points."""
    async with engine.begin() as conn:
        # Enable TimescaleDB extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb"))
        logger.info("TimescaleDB extension enabled")

        # Create all ORM tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

        # Keep existing development databases compatible with the current ORM.
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
        await conn.execute(
            text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_data_points_dedup "
                'ON data_points (source, metric, source_entity_id, "timestamp")'
            )
        )
        logger.info("data_points compatibility columns and dedup index configured")

        # Convert data_points to a TimescaleDB hypertable (idempotent)
        await conn.execute(
            text(
                "SELECT create_hypertable('data_points', 'timestamp', "
                "if_not_exists => TRUE, migrate_data => TRUE)"
            )
        )
        logger.info("data_points hypertable configured")


async def drop_tables(engine: AsyncEngine) -> None:
    """Drop all tables. Use only in tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        logger.info("All tables dropped")
