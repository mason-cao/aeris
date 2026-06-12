import sqlite3
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql, sqlite

from app.db.models import DataPoint, GUID


# A perfectly valid uuid4 whose undashed hex is digits plus a single 'e' —
# the ~1-in-a-million shape SQLite reads as scientific notation. The real
# production data_points table had exactly one of these stored as Inf.
NUMERIC_LOOKING = uuid.UUID("12345678-9012-4456-8e78-901234567890")


class TestGUIDBindAndResult:
    def test_binds_dashed_string_on_sqlite(self) -> None:
        bound = GUID().process_bind_param(NUMERIC_LOOKING, sqlite.dialect())
        assert bound == str(NUMERIC_LOOKING)
        assert isinstance(bound, str) and len(bound) == 36 and "-" in bound

    def test_normalizes_legacy_undashed_string_binds(self) -> None:
        bound = GUID().process_bind_param(NUMERIC_LOOKING.hex, sqlite.dialect())
        assert bound == str(NUMERIC_LOOKING)

    def test_passes_value_through_for_postgres(self) -> None:
        bound = GUID().process_bind_param(NUMERIC_LOOKING, postgresql.dialect())
        assert bound is NUMERIC_LOOKING

    def test_reads_both_dashed_and_legacy_undashed_rows(self) -> None:
        guid = GUID()
        dialect = sqlite.dialect()
        assert guid.process_result_value(str(NUMERIC_LOOKING), dialect) == (
            NUMERIC_LOOKING
        )
        assert guid.process_result_value(NUMERIC_LOOKING.hex, dialect) == (
            NUMERIC_LOOKING
        )
        assert guid.process_result_value(None, dialect) is None


class TestSQLiteAffinityRegression:
    """Document the corruption mode the dashed format defends against.

    The production table was created with the bare ``UUID`` DDL keyword,
    which SQLite gives NUMERIC affinity. These assertions pin the SQLite
    behavior itself so the dashed-bind fix can't silently regress.
    """

    def test_undashed_numeric_hex_is_coerced_to_real(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id UUID NOT NULL)")
        conn.execute("INSERT INTO t VALUES (?)", (NUMERIC_LOOKING.hex,))
        stored_type, value = conn.execute(
            "SELECT typeof(id), id FROM t"
        ).fetchone()
        assert stored_type == "real"
        assert value == float("inf")  # exponent 78... overflows float

    def test_dashed_form_survives_numeric_affinity(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id UUID NOT NULL)")
        conn.execute("INSERT INTO t VALUES (?)", (str(NUMERIC_LOOKING),))
        stored_type, value = conn.execute(
            "SELECT typeof(id), id FROM t"
        ).fetchone()
        assert stored_type == "text"
        assert value == str(NUMERIC_LOOKING)


class TestGUIDOrmRoundtrip:
    @pytest.mark.asyncio
    async def test_datapoint_id_roundtrips_through_sqlite(
        self, db_session
    ) -> None:
        point = DataPoint(
            id=NUMERIC_LOOKING,
            timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
            lat=29.7604,
            lon=-95.3698,
            metric="pm25",
            value=12.0,
            unit="ug/m3",
            source="openaq",
            source_entity_id="guid-roundtrip",
        )
        db_session.add(point)
        await db_session.commit()
        db_session.expunge_all()

        row = (
            await db_session.execute(
                select(DataPoint).where(
                    DataPoint.source_entity_id == "guid-roundtrip"
                )
            )
        ).scalar_one()
        assert row.id == NUMERIC_LOOKING
        assert isinstance(row.id, uuid.UUID)
