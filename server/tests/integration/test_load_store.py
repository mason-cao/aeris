"""Acer SQLite -> eval Postgres raw-data loader.

The collector box writes ``data_points`` to SQLite, which drops tzinfo; the
eval store is Postgres ``TIMESTAMPTZ``. These tests pin the two failure modes
that would silently corrupt the eval window: a timestamp landing in the wrong
zone, and a numeric-looking UUID id failing to round-trip.

The Postgres tests run against a throwaway ``aeris_loadtest`` database whose
session timezone is forced to ``America/Chicago`` — not UTC — so a loader that
forgot to re-anchor naive timestamps would shift the stored instant and the
assertions would catch it.
"""

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio
from sqlalchemy import delete, insert, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.db.models import Base, DataPoint
from app.eval.load_store import (
    TABLE_SPECS,
    CorruptSourceError,
    _amain,
    _aware_utc,
    _check_source_ids,
    _parse_args,
    _parse_since,
    _parse_tables,
    count_by_source_metric,
    load_data_points,
)

TEST_DB = "aeris_loadtest"

# A perfectly valid uuid4 whose undashed hex is digits plus a single 'e' — the
# shape SQLite reads as scientific notation; one such id once landed as Inf in
# production. See tests/unit/test_guid_type.py.
NUMERIC_LOOKING = uuid.UUID("12345678-9012-4456-8e78-901234567890")
WINDOW_TS = datetime(2026, 6, 15, 13, 0, tzinfo=timezone.utc)


class TestAwareUtc:
    def test_attaches_utc_to_naive(self) -> None:
        out = _aware_utc(datetime(2026, 6, 15, 13, 0))
        assert out == datetime(2026, 6, 15, 13, 0, tzinfo=timezone.utc)
        assert out.tzinfo is timezone.utc

    def test_passes_existing_tz_through_unchanged(self) -> None:
        aware = datetime(2026, 6, 15, 13, 0, tzinfo=timezone.utc)
        assert _aware_utc(aware) is aware

    def test_none_passes_through(self) -> None:
        assert _aware_utc(None) is None


_TERMINATE = (
    f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
    f"WHERE datname = '{TEST_DB}' AND pid <> pg_backend_pid()"
)


def _admin_engine():
    url = make_url(settings.database_url).set(database="postgres")
    return create_async_engine(url, isolation_level="AUTOCOMMIT")


async def _admin_exec(*statements: str) -> None:
    admin = _admin_engine()
    try:
        async with admin.connect() as conn:
            for stmt in statements:
                await conn.execute(text(stmt))
    finally:
        await admin.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def pg_engine():
    try:
        await _admin_exec(
            _TERMINATE,
            f'DROP DATABASE IF EXISTS "{TEST_DB}"',
            f'CREATE DATABASE "{TEST_DB}"',
            # Force a non-UTC session tz so a naive-timestamp load would misbehave.
            f"ALTER DATABASE \"{TEST_DB}\" SET timezone TO 'America/Chicago'",
        )
    except (OSError, SQLAlchemyError) as exc:
        pytest.skip(f"local Postgres not reachable: {exc}")

    url = make_url(settings.database_url).set(database=TEST_DB)
    engine = create_async_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()

    await _admin_exec(_TERMINATE, f'DROP DATABASE IF EXISTS "{TEST_DB}"')


@pytest_asyncio.fixture
async def dest_engine(pg_engine):
    async with pg_engine.begin() as conn:
        await conn.execute(delete(DataPoint.__table__))
    return pg_engine


@pytest_asyncio.fixture
async def source_engine(tmp_path):
    db_path = tmp_path / "acer.sqlite3"
    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


def _point(**overrides) -> dict:
    base = dict(
        id=uuid.uuid4(),
        timestamp=WINDOW_TS,
        lat=29.76,
        lon=-95.37,
        metric="pm25",
        value=42.0,
        unit="ug/m3",
        source="openaq",
        source_entity_id="c403",
        raw_json={"station": "C403"},
        collected_at=WINDOW_TS,
    )
    base.update(overrides)
    return base


async def _seed(engine, rows: list[dict]) -> None:
    async with engine.begin() as conn:
        await conn.execute(insert(DataPoint.__table__), rows)


async def _fetch_all(engine) -> list:
    async with engine.connect() as conn:
        return (await conn.execute(select(DataPoint.__table__))).all()


class TestRoundTrip:
    # asyncpg pins connections to their creating loop; run on the session loop
    # so these tests share the loop the session-scoped engine fixtures use.
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_loads_uuid_and_utc_timestamp(
        self, source_engine, dest_engine
    ) -> None:
        await _seed(
            source_engine,
            [_point(id=NUMERIC_LOOKING, timestamp=WINDOW_TS, collected_at=WINDOW_TS)],
        )

        report = await load_data_points(source_engine, dest_engine)

        assert report.inserted == 1
        assert report.total_source == 1
        assert report.skipped == 0

        rows = await _fetch_all(dest_engine)
        assert len(rows) == 1
        row = rows[0]
        # UUID round-trip: comes back as the exact uuid, typed.
        assert row.id == NUMERIC_LOOKING
        assert isinstance(row.id, uuid.UUID)
        # Timestamp integrity: same instant, in UTC, despite the Chicago session.
        assert row.timestamp == WINDOW_TS
        assert row.timestamp.utcoffset() == timedelta(0)
        assert row.collected_at == WINDOW_TS
        # Payload survives.
        assert row.raw_json == {"station": "C403"}
        assert row.metric == "pm25"
        assert row.value == 42.0


class TestDedup:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_rerun_inserts_nothing(self, source_engine, dest_engine) -> None:
        await _seed(source_engine, [_point()])

        first = await load_data_points(source_engine, dest_engine)
        second = await load_data_points(source_engine, dest_engine)

        assert first.inserted == 1
        assert second.total_source == 1
        assert second.inserted == 0
        assert second.skipped == 1
        assert len(await _fetch_all(dest_engine)) == 1

    async def test_skips_business_key_already_present_under_different_id(
        self, source_engine, dest_engine
    ) -> None:
        # A row the Mac collected itself: same business key, different uuid.
        business_key = dict(
            timestamp=WINDOW_TS,
            metric="pm25",
            source="openaq",
            source_entity_id="c403",
        )
        existing_id = uuid.uuid4()
        await _seed(dest_engine, [_point(id=existing_id, value=10.0, **business_key)])
        await _seed(source_engine, [_point(id=uuid.uuid4(), value=99.0, **business_key)])

        report = await load_data_points(source_engine, dest_engine)

        assert report.inserted == 0
        assert report.skipped == 1
        rows = await _fetch_all(dest_engine)
        assert len(rows) == 1
        # The pre-existing row is kept verbatim — do-nothing, not do-update.
        assert rows[0].id == existing_id
        assert rows[0].value == 10.0


class TestFiltering:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_since_excludes_earlier_rows(
        self, source_engine, dest_engine
    ) -> None:
        await _seed(
            source_engine,
            [
                _point(
                    id=uuid.uuid4(),
                    source_entity_id="old",
                    timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc),
                ),
                _point(id=uuid.uuid4(), source_entity_id="new", timestamp=WINDOW_TS),
            ],
        )

        report = await load_data_points(
            source_engine,
            dest_engine,
            since=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )

        assert report.total_source == 1
        assert report.inserted == 1
        rows = await _fetch_all(dest_engine)
        assert {row.source_entity_id for row in rows} == {"new"}

    async def test_loads_all_rows_across_batches(
        self, source_engine, dest_engine
    ) -> None:
        points = [
            _point(
                id=uuid.uuid4(),
                source_entity_id=f"s{i}",
                timestamp=WINDOW_TS + timedelta(minutes=i),
            )
            for i in range(5)
        ]
        await _seed(source_engine, points)

        report = await load_data_points(source_engine, dest_engine, batch_size=2)

        assert report.total_source == 5
        assert report.inserted == 5
        assert len(await _fetch_all(dest_engine)) == 5

    async def test_dry_run_writes_nothing(self, source_engine, dest_engine) -> None:
        await _seed(
            source_engine,
            [_point(), _point(id=uuid.uuid4(), source_entity_id="b")],
        )

        report = await load_data_points(source_engine, dest_engine, dry_run=True)

        assert report.total_source == 2
        assert report.inserted == 0
        assert len(await _fetch_all(dest_engine)) == 0


class TestReconciliation:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_counts_grouped_by_source_and_metric(self, dest_engine) -> None:
        await _seed(
            dest_engine,
            [
                _point(id=uuid.uuid4(), source="openaq", metric="pm25",
                       source_entity_id="a"),
                _point(id=uuid.uuid4(), source="openaq", metric="pm25",
                       source_entity_id="b"),
                _point(id=uuid.uuid4(), source="openaq", metric="o3",
                       source_entity_id="a"),
                _point(id=uuid.uuid4(), source="noaa_gfs", metric="u_10m",
                       source_entity_id="g"),
            ],
        )

        counts = await count_by_source_metric(dest_engine)

        assert counts == {
            ("openaq", "pm25"): 2,
            ("openaq", "o3"): 1,
            ("noaa_gfs", "u_10m"): 1,
        }

    async def test_counts_respect_since_on_sqlite(self, source_engine) -> None:
        await _seed(
            source_engine,
            [
                _point(id=uuid.uuid4(), source_entity_id="old",
                       timestamp=datetime(2026, 5, 1, tzinfo=timezone.utc)),
                _point(id=uuid.uuid4(), source_entity_id="new", timestamp=WINDOW_TS),
            ],
        )

        counts = await count_by_source_metric(
            source_engine, since=datetime(2026, 6, 1, tzinfo=timezone.utc)
        )

        assert counts == {("openaq", "pm25"): 1}


class TestCli:
    def test_parse_since_date_anchors_to_utc(self) -> None:
        assert _parse_since("2026-06-01") == datetime(2026, 6, 1, tzinfo=timezone.utc)

    def test_parse_since_datetime_anchors_to_utc(self) -> None:
        assert _parse_since("2026-06-01T06:30") == datetime(
            2026, 6, 1, 6, 30, tzinfo=timezone.utc
        )

    def test_defaults(self) -> None:
        args = _parse_args(["--from", "acer.db"])
        assert args.source_path == "acer.db"
        assert args.since is None
        # None means "use each table's own default": a 1000-row batch is right
        # for narrow data_points rows and would build a ~670 MB statement on
        # enrichment_records, whose summaries average 674 KB.
        assert args.batch_size is None
        assert args.tables == "data_points"
        assert args.dry_run is False

    def test_per_table_batch_defaults_scale_with_row_width(self) -> None:
        by_name = {spec.name: spec for spec in TABLE_SPECS}

        assert by_name["data_points"].batch_size == 1000
        assert by_name["enrichment_records"].batch_size == 25
        assert (
            by_name["enrichment_records"].batch_size
            < by_name["anomalies"].batch_size
            < by_name["data_points"].batch_size
        )

    def test_tables_are_copied_in_foreign_key_order_not_user_order(self) -> None:
        # anomalies must land before the enrichment rows that reference them,
        # however the flag is written.
        specs = _parse_tables("enrichment_records,data_points,anomalies")

        assert [spec.name for spec in specs] == [
            "data_points",
            "anomalies",
            "enrichment_records",
        ]

    def test_unknown_table_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown table"):
            _parse_tables("data_points,explanations")

    def test_empty_table_list_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one table"):
            _parse_tables(" , ")

    def test_flags(self) -> None:
        args = _parse_args(
            ["--from", "acer.db", "--since", "2026-06-01",
             "--batch-size", "250", "--dry-run"]
        )
        assert args.since == datetime(2026, 6, 1, tzinfo=timezone.utc)
        assert args.batch_size == 250
        assert args.dry_run is True


def _write_numeric_affinity_source(path) -> None:
    """A SQLite data_points table mirroring the Acer's old bare-UUID id column.

    The ``id UUID`` DDL gives NUMERIC affinity, so a numeric-looking undashed
    hex id is coerced to a REAL (the ``Inf`` corruption) while a dashed id stays
    text — exactly the pre-2026-06-12 production state.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE data_points ("
            "id UUID NOT NULL, timestamp TEXT NOT NULL, lat REAL NOT NULL, "
            "lon REAL NOT NULL, metric VARCHAR(64) NOT NULL, value REAL NOT NULL, "
            "unit VARCHAR(32) NOT NULL, source VARCHAR(64) NOT NULL, "
            "source_entity_id VARCHAR(128) NOT NULL, raw_json JSON, "
            "collected_at TEXT NOT NULL, PRIMARY KEY (id, timestamp))"
        )
        rows = [
            (str(uuid.uuid4()), "2026-06-15 13:00:00.000000", 29.76, -95.37,
             "pm25", 42.0, "ug/m3", "openaq", "good", None,
             "2026-06-15 13:00:00.000000"),
            (NUMERIC_LOOKING.hex, "2026-06-15 14:00:00.000000", 29.76, -95.37,
             "pm25", 43.0, "ug/m3", "openaq", "bad", None,
             "2026-06-15 14:00:00.000000"),
        ]
        conn.executemany(
            "INSERT INTO data_points VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()
    finally:
        conn.close()


class TestSourceGuard:
    pytestmark = pytest.mark.asyncio(loop_scope="session")

    async def test_rejects_numeric_affinity_id(self, tmp_path) -> None:
        path = tmp_path / "corrupt.sqlite3"
        _write_numeric_affinity_source(path)
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        try:
            with pytest.raises(CorruptSourceError) as exc_info:
                await _check_source_ids(engine)
            message = str(exc_info.value)
            assert "1" in message
            assert "typeof(id)" in message  # cites the repair
        finally:
            await engine.dispose()

    async def test_accepts_clean_source(self, source_engine) -> None:
        await _seed(source_engine, [_point(id=NUMERIC_LOOKING)])
        await _check_source_ids(source_engine)  # dashed text id — no raise

    async def test_load_rejects_corrupt_source_without_writing(
        self, tmp_path, dest_engine
    ) -> None:
        path = tmp_path / "corrupt.sqlite3"
        _write_numeric_affinity_source(path)
        engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
        try:
            with pytest.raises(CorruptSourceError):
                await load_data_points(engine, dest_engine)
            assert await _fetch_all(dest_engine) == []  # nothing partially loaded
        finally:
            await engine.dispose()

    async def test_cli_exits_nonzero_on_corrupt_source(self, tmp_path) -> None:
        path = tmp_path / "corrupt.sqlite3"
        _write_numeric_affinity_source(path)
        # The guard runs before any dest interaction, so the configured store is
        # never touched (engine is created lazily and never queried).
        exit_code = await _amain(["--from", str(path)])
        assert exit_code == 3
