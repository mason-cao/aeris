"""Copy an AERIS SQLite snapshot into the configured (Postgres/Timescale) DB.

Option A of the pre-freeze DB-engine decision: the Acer keeps collecting on
SQLite; the freeze-day analysis pipeline runs on Postgres + TimescaleDB. This
one-shot loader moves the collected rows across so the hypertable holds them.

The copy goes through the ORM ``Table`` objects, not raw SQL, so the ``GUID``
type decorator translates CHAR(36) <-> native UUID and the ``JSON`` columns
(de)serialize per dialect automatically. ``create_tables`` on the Postgres
target builds the ``data_points`` hypertable (empty) before any rows land, so
inserts route straight into chunks.

    DATABASE_URL=postgresql+asyncpg://aeris:aeris@localhost:5433/aeris \
        python -m app.db.migrate --source-sqlite /path/to/aeris.db --reset
"""

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, inspect, select
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import settings
from app.db.models import Base
from app.db.schema import create_tables, drop_tables

logger = logging.getLogger(__name__)

_UTC = timezone.utc


def _to_utc(value: object) -> object:
    """Normalize datetimes to UTC; leave everything else untouched.

    Snapshot timestamps are stored naive (UTC wall-clock). Postgres
    ``timestamptz`` needs an aware value, so naive inputs are assumed UTC and
    aware inputs are converted — both land at the same instant.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC)
    return value


async def _count_rows(engine: AsyncEngine, table) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(select(func.count()).select_from(table))
        return int(result.scalar_one())


async def _source_table_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as conn:
        names = await conn.run_sync(lambda sync: inspect(sync).get_table_names())
    return set(names)


async def copy_table(
    source_engine: AsyncEngine,
    target_engine: AsyncEngine,
    table,
    *,
    chunk_size: int = 1000,
) -> int:
    """Stream every row of ``table`` from source to target, in chunks."""
    columns = list(table.columns)
    copied = 0
    async with source_engine.connect() as sconn:
        result = await sconn.stream(table.select())
        async with target_engine.begin() as tconn:
            async for partition in result.partitions(chunk_size):
                rows = [
                    {col.name: _to_utc(row._mapping[col.name]) for col in columns}
                    for row in partition
                ]
                if rows:
                    await tconn.execute(table.insert(), rows)
                    copied += len(rows)
    return copied


async def migrate_db(
    *,
    source_url: str,
    target_url: str,
    reset: bool = False,
    chunk_size: int = 1000,
) -> dict[str, int]:
    """Copy all tables present in the source into the target DB.

    With ``reset`` the target schema is dropped and rebuilt first; without it,
    a populated target is refused so a re-run cannot silently double-load.
    """
    source_engine = create_async_engine(source_url)
    target_engine = create_async_engine(target_url)
    try:
        if reset:
            await drop_tables(target_engine)
        # Builds tables on any dialect; on Postgres also enables the
        # TimescaleDB extension + converts data_points to a hypertable.
        await create_tables(target_engine)

        existing = await _count_rows(target_engine, Base.metadata.tables["data_points"])
        if existing and not reset:
            raise RuntimeError(
                f"target data_points is not empty ({existing} rows); "
                "pass reset=True (or --reset) to wipe and reload"
            )

        present = await _source_table_names(source_engine)
        counts: dict[str, int] = {}
        for table in Base.metadata.sorted_tables:
            if table.name not in present:
                logger.info("source has no %s table; skipping", table.name)
                counts[table.name] = 0
                continue
            counts[table.name] = await copy_table(
                source_engine, target_engine, table, chunk_size=chunk_size
            )
            logger.info("copied %s: %d rows", table.name, counts[table.name])
        return counts
    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an AERIS SQLite snapshot into the configured DATABASE_URL "
            "(Postgres/TimescaleDB analysis DB)."
        )
    )
    parser.add_argument(
        "--source-sqlite",
        required=True,
        help="Path to the source SQLite aeris.db snapshot.",
    )
    parser.add_argument(
        "--target-url",
        default=None,
        help="Target async DB URL (default: settings.database_url / DATABASE_URL).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop & recreate target tables first (wipes the analysis DB).",
    )
    parser.add_argument("--chunk-size", type=int, default=1000)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)

    source_path = Path(args.source_sqlite).expanduser().resolve()
    if not source_path.exists():
        _build_parser().error(f"source SQLite not found: {source_path}")
    source_url = f"sqlite+aiosqlite:///{source_path}"
    target_url = args.target_url or settings.database_url

    counts = await migrate_db(
        source_url=source_url,
        target_url=target_url,
        reset=args.reset,
        chunk_size=args.chunk_size,
    )
    total = sum(counts.values())
    print(f"Migrated {total} rows into {target_url}")
    for name, n in counts.items():
        if n:
            print(f"  {name}: {n}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
