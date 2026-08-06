"""Load a collector or analysis SQLite DB into the eval Postgres store.

Production collection runs on the Acer into SQLite; the eval runs on Postgres +
TimescaleDB. This copies rows across dialect-safely: SQLite drops tzinfo, so
timestamps are re-anchored to UTC before they land in the ``TIMESTAMPTZ``
columns, and ids round-trip through the ``GUID`` type.

``data_points`` is the default and the only table the collector source carries.
The derived tables (``anomalies``, ``enrichment_records``) are opt-in via
``--tables`` and exist for one specific reason: **anomaly ids are
``uuid.uuid4``, so re-running detection against Postgres would mint entirely
new ids and invalidate a frozen fixture that names 50 of them.** Once a freeze
is committed, the derived rows must be copied with their ids intact, never
regenerated. Explanations, claims, and expert labels are deliberately not
copyable here — those are the eval's outputs and belong to whichever store
produced them.
"""

import argparse
import asyncio
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import Table, func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.config import settings
from app.db.models import Anomaly, DataPoint, EnrichmentRecord


class CorruptSourceError(Exception):
    """The source holds ids that won't round-trip and must be repaired first."""


@dataclass
class LoadReport:
    total_source: int
    inserted: int
    skipped: int


async def _check_source_ids(source_engine: AsyncEngine) -> None:
    """Fail loudly on a source whose ids won't parse as UUIDs.

    On SQLite a numeric-looking id can be stored with REAL/INTEGER affinity (the
    ``Inf`` corruption that the ``GUID`` type now prevents for new rows). Reading
    such a row through the ORM raises an opaque ``ValueError`` mid-stream, so we
    detect it up front and point at the documented repair before writing anything.
    """
    if source_engine.dialect.name != "sqlite":
        return
    async with source_engine.connect() as conn:
        bad = (
            await conn.execute(
                text("SELECT count(*) FROM data_points WHERE typeof(id) <> 'text'")
            )
        ).scalar_one()
    if bad:
        raise CorruptSourceError(
            f"{bad} source row(s) have a non-text id — SQLite numeric-affinity "
            "corruption (see GUID in app/db/models.py). Repair before loading:\n"
            "    UPDATE data_points SET id = lower(hex(randomblob(16))) "
            "WHERE typeof(id) <> 'text';"
        )


def _aware_utc(value: datetime | None) -> datetime | None:
    """Re-anchor a SQLite-sourced naive datetime to UTC.

    SQLite stores ``DateTime(timezone=True)`` values without an offset, so they
    read back naive. Inserting a naive datetime into Postgres ``TIMESTAMPTZ``
    would have it interpreted in the session timezone — the silent corruption
    this load must avoid. Already-aware values pass through untouched.
    """
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


async def load_data_points(
    source_engine: AsyncEngine,
    dest_engine: AsyncEngine,
    *,
    since: datetime | None = None,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> LoadReport:
    """Copy ``data_points`` rows from the SQLite source into Postgres.

    Re-anchors tz-bearing columns to UTC and skips rows already present (matched
    on the ``uq_data_points_dedup`` business key), so the load is idempotent and
    safe to re-run. Rows are streamed and inserted in ``batch_size`` chunks to
    stay under the Postgres bind-parameter cap. ``since`` restricts to rows at or
    after that instant; ``dry_run`` counts without writing.
    """
    await _check_source_ids(source_engine)

    table = DataPoint.__table__
    stmt = select(table).order_by(table.c.timestamp, table.c.id)
    if since is not None:
        stmt = stmt.where(table.c.timestamp >= since)

    total = 0
    inserted = 0
    async with source_engine.connect() as src:
        result = await src.stream(stmt)
        async for partition in result.partitions(batch_size):
            rows = []
            for row in partition:
                values = dict(row._mapping)
                values["timestamp"] = _aware_utc(values["timestamp"])
                values["collected_at"] = _aware_utc(values["collected_at"])
                rows.append(values)
            total += len(rows)
            if dry_run or not rows:
                continue
            insert_stmt = pg_insert(table).values(rows).on_conflict_do_nothing(
                constraint="uq_data_points_dedup"
            )
            async with dest_engine.begin() as dest:
                inserted += (await dest.execute(insert_stmt)).rowcount or 0

    return LoadReport(
        total_source=total, inserted=inserted, skipped=total - inserted
    )


@dataclass(frozen=True)
class TableSpec:
    """One copyable table: its timestamp columns and a safe insert batch size."""

    name: str
    table: Table
    tz_columns: tuple[str, ...]
    order_by: tuple[str, ...]
    batch_size: int


# enrichment_records carries ~674 KB of JSON per row, so a 1000-row batch would
# build a ~670 MB statement. Keep its batch small; the others are narrow rows.
TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        "data_points",
        DataPoint.__table__,
        ("timestamp", "collected_at"),
        ("timestamp", "id"),
        1000,
    ),
    TableSpec(
        "anomalies",
        Anomaly.__table__,
        ("timestamp", "detected_at"),
        ("timestamp", "id"),
        500,
    ),
    TableSpec(
        "enrichment_records",
        EnrichmentRecord.__table__,
        ("context_window_start", "context_window_end", "created_at"),
        ("id",),
        25,
    ),
)
TABLES_BY_NAME: dict[str, TableSpec] = {spec.name: spec for spec in TABLE_SPECS}

# Copy order is fixed by the foreign keys, not by the order the user lists them.
TABLE_ORDER: tuple[str, ...] = tuple(spec.name for spec in TABLE_SPECS)


async def load_table(
    spec: TableSpec,
    source_engine: AsyncEngine,
    dest_engine: AsyncEngine,
    *,
    since: datetime | None = None,
    batch_size: int | None = None,
    dry_run: bool = False,
    conflict: Callable[[object], object] | None = None,
) -> LoadReport:
    """Stream one table from SQLite into Postgres, preserving ids.

    Skips rows already present by primary key, so a re-run is idempotent and an
    interrupted load resumes cleanly.
    """
    table = spec.table
    stmt = select(table).order_by(*(table.c[col] for col in spec.order_by))
    if since is not None and "timestamp" in table.c:
        stmt = stmt.where(table.c.timestamp >= since)

    size = batch_size or spec.batch_size
    total = 0
    inserted = 0
    async with source_engine.connect() as src:
        result = await src.stream(stmt)
        async for partition in result.partitions(size):
            rows = []
            for row in partition:
                values = dict(row._mapping)
                for column in spec.tz_columns:
                    values[column] = _aware_utc(values[column])
                rows.append(values)
            total += len(rows)
            if dry_run or not rows:
                continue
            insert_stmt = pg_insert(table).values(rows)
            insert_stmt = (
                conflict(insert_stmt)
                if conflict is not None
                else insert_stmt.on_conflict_do_nothing(index_elements=["id"])
            )
            async with dest_engine.begin() as dest:
                inserted += (await dest.execute(insert_stmt)).rowcount or 0

    return LoadReport(
        total_source=total, inserted=inserted, skipped=total - inserted
    )


async def count_rows(engine: AsyncEngine, spec: TableSpec) -> int:
    async with engine.connect() as conn:
        return (
            await conn.execute(select(func.count()).select_from(spec.table))
        ).scalar_one()


async def count_by_source_metric(
    engine: AsyncEngine, *, since: datetime | None = None
) -> dict[tuple[str, str], int]:
    """Row counts per ``(source, metric)`` — used to reconcile source vs dest."""
    table = DataPoint.__table__
    stmt = (
        select(table.c.source, table.c.metric, func.count().label("n"))
        .group_by(table.c.source, table.c.metric)
    )
    if since is not None:
        stmt = stmt.where(table.c.timestamp >= since)
    async with engine.connect() as conn:
        result = await conn.execute(stmt)
        return {(row.source, row.metric): row.n for row in result}


def _parse_since(value: str) -> datetime:
    return _aware_utc(datetime.fromisoformat(value))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.load_store",
        description=(
            "Load collector SQLite data_points into the eval Postgres store "
            "(set by DATABASE_URL). Idempotent: safe to re-run."
        ),
    )
    parser.add_argument(
        "--from",
        dest="source_path",
        required=True,
        help="Path to the collector SQLite database (e.g. a copy of the Acer aeris.db).",
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        metavar="DATE",
        help="Only load rows at or after this date/datetime, interpreted as UTC.",
    )
    parser.add_argument(
        "--tables",
        default="data_points",
        help=(
            "Comma-separated tables to copy from "
            f"{{{','.join(TABLE_ORDER)}}} (default: data_points). Derived "
            "tables are copied with their ids intact so a committed freeze "
            "keeps resolving; copy order follows the foreign keys regardless "
            "of the order given."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the per-table default batch size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count source rows and reconcile without writing.",
    )
    return parser.parse_args(argv)


def _parse_tables(raw: str) -> tuple[TableSpec, ...]:
    requested = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in requested if name not in TABLES_BY_NAME]
    if unknown:
        raise ValueError(
            f"unknown table(s) {unknown}; choose from {list(TABLE_ORDER)}"
        )
    if not requested:
        raise ValueError("--tables must name at least one table")
    # Foreign keys, not user order: anomalies must land before enrichment.
    return tuple(
        TABLES_BY_NAME[name] for name in TABLE_ORDER if name in set(requested)
    )


def _print_source_metric_table(
    src_counts: dict[tuple[str, str], int],
    dest_counts: dict[tuple[str, str], int],
) -> None:
    print(f"{'source/metric':<36}{'src':>8}{'dest':>8}")
    for key in sorted(set(src_counts) | set(dest_counts)):
        src = src_counts.get(key, 0)
        dest = dest_counts.get(key, 0)
        flag = "" if src == dest else "  <-- mismatch"
        print(f"{key[0] + '/' + key[1]:<36}{src:>8}{dest:>8}{flag}")


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    source_path = os.path.abspath(args.source_path)
    if not os.path.isfile(source_path):
        print(f"source database not found: {source_path}", file=sys.stderr)
        return 2
    try:
        specs = _parse_tables(args.tables)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    source_engine = create_async_engine(f"sqlite+aiosqlite:///{source_path}")
    dest_engine = create_async_engine(settings.database_url)
    prefix = "DRY RUN — " if args.dry_run else ""
    try:
        await _check_source_ids(source_engine)
        for spec in specs:
            if spec.name == "data_points":
                report = await load_data_points(
                    source_engine,
                    dest_engine,
                    since=args.since,
                    batch_size=args.batch_size or spec.batch_size,
                    dry_run=args.dry_run,
                )
            else:
                report = await load_table(
                    spec,
                    source_engine,
                    dest_engine,
                    since=args.since,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                )
            dest_total = await count_rows(dest_engine, spec)
            print(
                f"{prefix}{spec.name}: source rows {report.total_source}  "
                f"inserted {report.inserted}  "
                f"skipped (already present) {report.skipped}  "
                f"dest total {dest_total}"
            )
        if any(spec.name == "data_points" for spec in specs):
            src_counts = await count_by_source_metric(
                source_engine, since=args.since
            )
            dest_counts = await count_by_source_metric(
                dest_engine, since=args.since
            )
            _print_source_metric_table(src_counts, dest_counts)
    except CorruptSourceError as exc:
        print(str(exc), file=sys.stderr)
        return 3
    finally:
        await source_engine.dispose()
        await dest_engine.dispose()

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
