"""Guards the db_session fixture's transactional isolation.

These two tests must run in definition order: the first commits a probe row,
the second asserts it is gone. A non-isolating fixture (commit persists to the
shared engine) fails the second; a transactional fixture rolls the commit back
with its outer transaction and the second sees a clean table.
"""

from datetime import datetime, timezone

from sqlalchemy import func, select

from app.db.models import DataPoint

_PROBE_SOURCE = "iso-probe"


def _probe_point() -> DataPoint:
    return DataPoint(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        lat=29.76,
        lon=-95.37,
        metric="pm25",
        value=1.0,
        unit="ug/m3",
        source=_PROBE_SOURCE,
        source_entity_id="probe-1",
    )


async def _probe_count(session) -> int:
    return (
        await session.execute(
            select(func.count())
            .select_from(DataPoint)
            .where(DataPoint.source == _PROBE_SOURCE)
        )
    ).scalar_one()


async def test_committed_rows_are_visible_within_the_same_test(db_session):
    db_session.add(_probe_point())
    await db_session.commit()
    assert await _probe_count(db_session) == 1


async def test_committed_rows_do_not_leak_into_the_next_test(db_session):
    assert await _probe_count(db_session) == 0
