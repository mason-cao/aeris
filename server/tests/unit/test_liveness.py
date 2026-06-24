from datetime import datetime, timedelta, timezone

from app.db.models import DataPoint
from app.monitoring.liveness import (
    DEFAULT_BUDGETS_MINUTES,
    SourceLiveness,
    check_liveness,
    format_source,
)

NOW = datetime(2026, 6, 16, 12, 0, tzinfo=timezone.utc)


def _point(source: str, collected_at: datetime, entity: str = "e1") -> DataPoint:
    return DataPoint(
        timestamp=collected_at,
        lat=29.76,
        lon=-95.37,
        metric="pm25",
        value=1.0,
        unit="ug/m3",
        source=source,
        source_entity_id=entity,
        collected_at=collected_at,
    )


async def _insert(session, *points: DataPoint) -> None:
    session.add_all(points)
    await session.commit()


async def test_all_sources_fresh_is_healthy(db_session):
    recent = NOW - timedelta(minutes=10)
    await _insert(
        db_session,
        _point("openaq", recent),
        _point("openweather", recent),
        _point("noaa_gfs", recent),
        _point("sentinel5p", recent),
        _point("asos", recent),
        _point("tceq", recent),
        _point("purpleair", recent),
    )

    report = await check_liveness(db_session, now=NOW)

    assert report.healthy
    assert report.stale_sources == ()
    assert {s.source for s in report.sources} == set(DEFAULT_BUDGETS_MINUTES)


async def test_source_past_budget_is_stale(db_session):
    fresh = NOW - timedelta(minutes=10)
    await _insert(
        db_session,
        _point("openaq", NOW - timedelta(minutes=200)),  # budget 180m
        _point("openweather", fresh),
        _point("noaa_gfs", fresh),
        _point("sentinel5p", fresh),
        _point("asos", fresh),
        _point("tceq", fresh),
        _point("purpleair", fresh),
    )

    report = await check_liveness(db_session, now=NOW)

    assert not report.healthy
    assert [s.source for s in report.stale_sources] == ["openaq"]


async def test_source_with_no_rows_is_stale(db_session):
    await _insert(db_session, _point("openaq", NOW - timedelta(minutes=10)))

    report = await check_liveness(db_session, now=NOW)

    stale = {s.source for s in report.stale_sources}
    assert "openweather" in stale and "noaa_gfs" in stale and "sentinel5p" in stale
    openweather = next(s for s in report.sources if s.source == "openweather")
    assert openweather.age_minutes is None
    assert openweather.last_collected_at is None


async def test_orbital_sparsity_tolerated_for_sentinel5p(db_session):
    # A 24h-old reading is well within Sentinel-5P's multi-day budget but would
    # be stale for an hourly source -- the per-source budgets are the point.
    day_old = NOW - timedelta(hours=24)
    await _insert(
        db_session,
        _point("sentinel5p", day_old),
        _point("openaq", day_old),
    )

    report = await check_liveness(
        db_session, now=NOW, sources=("sentinel5p", "openaq")
    )

    by_source = {s.source: s for s in report.sources}
    assert not by_source["sentinel5p"].stale
    assert by_source["openaq"].stale


async def test_only_requested_source_is_checked(db_session):
    await _insert(db_session, _point("openaq", NOW - timedelta(minutes=10)))

    report = await check_liveness(db_session, now=NOW, sources=("openaq",))

    assert [s.source for s in report.sources] == ["openaq"]
    assert report.healthy


async def test_override_budget_collapses_to_one_value(db_session):
    await _insert(db_session, _point("sentinel5p", NOW - timedelta(minutes=200)))

    # The generous default (4320m) would pass; a 180m override flips it stale.
    report = await check_liveness(
        db_session,
        now=NOW,
        sources=("sentinel5p",),
        budgets={"sentinel5p": 180.0},
    )

    assert not report.healthy
    assert report.stale_sources[0].source == "sentinel5p"


def test_source_liveness_stale_property():
    assert SourceLiveness("openaq", None, None, 180.0).stale is True
    assert SourceLiveness("openaq", NOW, 181.0, 180.0).stale is True
    assert SourceLiveness("openaq", NOW, 179.0, 180.0).stale is False


def test_format_source_renders_stale_and_missing():
    missing = format_source(SourceLiveness("openweather", None, None, 180.0))
    assert "openweather" in missing and "STALE" in missing
    assert "age=no-data" in missing and "last=never" in missing

    ok = format_source(SourceLiveness("openaq", NOW, 12.0, 180.0))
    assert "ok" in ok and "age=12m" in ok and "budget=180m" in ok
