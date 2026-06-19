from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.migrate import _to_utc, migrate_db
from app.db.models import Anomaly, DataPoint, DataSource, EnrichmentRecord
from app.db.schema import create_tables


def test_to_utc_naive_assumed_utc() -> None:
    naive = datetime(2026, 6, 1, 12, 0, 0)
    assert _to_utc(naive) == datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_to_utc_aware_converted_to_utc() -> None:
    aware = datetime(2026, 6, 1, 7, 0, 0, tzinfo=timezone(timedelta(hours=-5)))
    assert _to_utc(aware) == datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_to_utc_passthrough_non_datetime() -> None:
    assert _to_utc("x") == "x"
    assert _to_utc(None) is None


async def _seed(url: str) -> None:
    engine = create_async_engine(url)
    await create_tables(engine)
    async with AsyncSession(engine, expire_on_commit=False) as session:
        anomaly = Anomaly(
            timestamp=datetime(2026, 6, 1, 12, tzinfo=timezone.utc),
            lat=29.7,
            lon=-95.3,
            metric="pm25",
            source="openaq",
            value=80.0,
            methods_triggered=["zscore", "stl"],
            severity="severe",
        )
        session.add(anomaly)
        await session.flush()
        session.add(
            EnrichmentRecord(
                anomaly_id=anomaly.id,
                context_window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
                context_window_end=datetime(2026, 6, 2, tzinfo=timezone.utc),
                cross_source_summary_json={"sources": ["openaq"]},
            )
        )
        session.add_all(
            [
                DataPoint(
                    timestamp=datetime(2026, 6, 1, 10),  # naive, like the snapshot
                    lat=29.7,
                    lon=-95.3,
                    metric="pm25",
                    value=12.5,
                    unit="ug/m3",
                    source="openaq",
                    source_entity_id="s1",
                    raw_json={"a": 1},
                    collected_at=datetime(2026, 6, 1, 11),
                ),
                DataPoint(
                    timestamp=datetime(2026, 6, 1, 11),
                    lat=29.7,
                    lon=-95.3,
                    metric="ozone",
                    value=0.04,
                    unit="ppm",
                    source="openaq",
                    source_entity_id="s2",
                    raw_json=None,
                    collected_at=datetime(2026, 6, 1, 12),
                ),
            ]
        )
        session.add(
            DataSource(
                name="openaq", source_type="openaq", status="active", error_count=0
            )
        )
        await session.commit()
    await engine.dispose()


async def test_migrate_copies_all_tables_in_fk_order(tmp_path) -> None:
    src = f"sqlite+aiosqlite:///{tmp_path / 'src.db'}"
    dst = f"sqlite+aiosqlite:///{tmp_path / 'dst.db'}"
    await _seed(src)

    counts = await migrate_db(source_url=src, target_url=dst, reset=True)

    assert counts["data_points"] == 2
    assert counts["data_sources"] == 1
    assert counts["anomalies"] == 1
    assert counts["enrichment_records"] == 1

    engine = create_async_engine(dst)
    async with AsyncSession(engine) as session:
        assert (
            await session.execute(select(func.count()).select_from(DataPoint))
        ).scalar() == 2
        dp = (
            await session.execute(
                select(DataPoint).where(DataPoint.metric == "pm25")
            )
        ).scalar_one()
        assert dp.value == 12.5
        assert dp.raw_json == {"a": 1}
        # naive source timestamp preserved as the same UTC instant
        assert dp.timestamp.replace(tzinfo=None) == datetime(2026, 6, 1, 10)
        anomaly = (await session.execute(select(Anomaly))).scalar_one()
        assert anomaly.methods_triggered == ["zscore", "stl"]
        enrichment = (await session.execute(select(EnrichmentRecord))).scalar_one()
        assert enrichment.anomaly_id == anomaly.id  # FK survived
    await engine.dispose()


async def test_migrate_refuses_nonempty_target_without_reset(tmp_path) -> None:
    src = f"sqlite+aiosqlite:///{tmp_path / 'src.db'}"
    dst = f"sqlite+aiosqlite:///{tmp_path / 'dst.db'}"
    await _seed(src)
    await migrate_db(source_url=src, target_url=dst, reset=True)

    try:
        await migrate_db(source_url=src, target_url=dst, reset=False)
    except RuntimeError as exc:
        assert "not empty" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError on non-empty target")
