import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, select, text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.models import Anomaly, EnrichmentRecord


REPO_ROOT = Path(__file__).resolve().parents[2]

# A valid uuid4 whose undashed hex is digits plus a single 'e' — the shape a
# NUMERIC-affinity column coerces to REAL/Inf on SQLite (the production bug).
NUMERIC_LOOKING = uuid.UUID("12345678-9012-4456-8e78-901234567890")
PRE_A6_REVISION = "a5d7e9c2b614"
PRE_B18_REVISION = "c6f2a9d4e817"


def _insert_anomaly(connection: Connection, anomaly_id: uuid.UUID) -> None:
    connection.execute(
        text(
            "INSERT INTO anomalies "
            "(id, timestamp, lat, lon, metric, source, value, "
            "methods_triggered, severity) VALUES "
            "(:id, '2026-07-15 12:00:00', 29.76, -95.37, 'pm25', "
            "'openaq', 12.0, :methods, 'moderate')"
        ),
        {"id": anomaly_id.hex, "methods": '["zscore"]'},
    )


def _insert_explanation(
    connection: Connection,
    *,
    row_id: uuid.UUID,
    anomaly_id: uuid.UUID,
    model_name: str,
) -> None:
    connection.execute(
        text(
            "INSERT INTO explanations "
            "(id, anomaly_id, model_name, reasoning_steps_json, "
            "final_narrative) VALUES "
            "(:id, :anomaly_id, :model_name, :steps, 'synthetic')"
        ),
        {
            "id": row_id.hex,
            "anomaly_id": anomaly_id.hex,
            "model_name": model_name,
            "steps": "{}",
        },
    )


def _insert_expert_label(
    connection: Connection,
    *,
    row_id: uuid.UUID,
    anomaly_id: uuid.UUID,
    labeler: str,
) -> None:
    connection.execute(
        text(
            "INSERT INTO expert_labels (id, anomaly_id, labeler) "
            "VALUES (:id, :anomaly_id, :labeler)"
        ),
        {
            "id": row_id.hex,
            "anomaly_id": anomaly_id.hex,
            "labeler": labeler,
        },
    )


def _alembic_config(db_url: str) -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", db_url)
    return cfg


@pytest.fixture
def sqlite_url(tmp_path):
    db_path = tmp_path / "alembic_test.db"
    return f"sqlite:///{db_path}"


@pytest.mark.parametrize(
    "table",
    [
        "data_points",
        "data_sources",
        "anomalies",
        "enrichment_records",
        "explanations",
        "claims",
        "expert_labels",
    ],
)
def test_upgrade_head_creates_expected_table(sqlite_url, table) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        assert table in set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_upgrade_head_creates_claims_fk_to_explanations(sqlite_url) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        fks = inspect(engine).get_foreign_keys("claims")
        assert any(
            fk["referred_table"] == "explanations"
            and fk["referred_columns"] == ["id"]
            for fk in fks
        ), f"expected FK claims.explanation_id -> explanations.id, got {fks!r}"
    finally:
        engine.dispose()


def test_upgrade_head_creates_nullable_legacy_citation_outcome(sqlite_url) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        columns = {column["name"]: column for column in inspect(engine).get_columns("claims")}
        assert columns["citation_outcome"]["nullable"] is True
    finally:
        engine.dispose()


def test_b18_upgrade_adds_nullable_legacy_provenance_and_downgrades(
    sqlite_url: str,
) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        columns = {
            column["name"]: column
            for column in inspect(engine).get_columns("anomalies")
        }
        assert columns["source_entity_id"]["nullable"] is True
        assert columns["detector_availability_json"]["nullable"] is True
    finally:
        engine.dispose()

    command.downgrade(cfg, PRE_B18_REVISION)
    engine = create_engine(sqlite_url)
    try:
        columns = {
            column["name"] for column in inspect(engine).get_columns("anomalies")
        }
        assert "source_entity_id" not in columns
        assert "detector_availability_json" not in columns
    finally:
        engine.dispose()


def test_b18_upgrade_preserves_legacy_anomaly_with_null_provenance(
    sqlite_url: str,
) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, PRE_B18_REVISION)
    anomaly_id = uuid.uuid4()
    engine = create_engine(sqlite_url)
    try:
        with engine.begin() as connection:
            _insert_anomaly(connection, anomaly_id)
    finally:
        engine.dispose()

    command.upgrade(cfg, "head")
    engine = create_engine(sqlite_url)
    try:
        with engine.connect() as connection:
            row = connection.execute(
                text(
                    "SELECT source_entity_id, detector_availability_json "
                    "FROM anomalies WHERE id = :id"
                ),
                {"id": anomaly_id.hex},
            ).one()
        assert row == (None, None)
    finally:
        engine.dispose()


def test_upgrade_head_creates_enrichment_fk_to_anomalies(sqlite_url) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        fks = inspect(engine).get_foreign_keys("enrichment_records")
        assert any(
            fk["referred_table"] == "anomalies" and fk["referred_columns"] == ["id"]
            for fk in fks
        ), f"expected FK enrichment_records.anomaly_id -> anomalies.id, got {fks!r}"
    finally:
        engine.dispose()


def test_upgrade_head_id_column_resists_numeric_affinity(sqlite_url) -> None:
    # The migrated data_points.id must store the pathological undashed-hex id as
    # text. A NUMERIC-affinity id column (bare ``UUID`` DDL) coerces it to
    # REAL/Inf — the corruption GUID/create_all already prevents, which the
    # migrations must match.
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO data_points "
                    "(id, timestamp, lat, lon, metric, value, unit, source, "
                    "source_entity_id) VALUES "
                    "(:id, '2026-06-01 12:00:00', 29.76, -95.37, 'pm25', 12.0, "
                    "'ug/m3', 'openaq', 'affinity-probe')"
                ),
                {"id": NUMERIC_LOOKING.hex},
            )
            stored_type, value = conn.execute(
                text("SELECT typeof(id), id FROM data_points")
            ).fetchone()
        assert stored_type == "text"
        assert value == NUMERIC_LOOKING.hex
    finally:
        engine.dispose()


def test_upgrade_head_orm_uuid_roundtrips_pk_and_fk(sqlite_url) -> None:
    # The raw-SQL affinity test covers data_points.id; this covers the ORM
    # path the app actually uses against the *migrated* schema (not
    # create_all): the GUID TypeDecorator must round-trip a pathological UUID
    # primary key and a UUID foreign key. If migrations and models drift, this
    # breaks where create_all-based tests can't see it.
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        with Session(engine) as session:
            session.add(
                Anomaly(
                    id=NUMERIC_LOOKING,
                    timestamp=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
                    lat=29.76,
                    lon=-95.37,
                    metric="pm25",
                    source="openaq",
                    value=85.0,
                    methods_triggered=["zscore"],
                    severity="severe",
                )
            )
            session.add(
                EnrichmentRecord(
                    anomaly_id=NUMERIC_LOOKING,
                    context_window_start=datetime(2026, 6, 1, 6, 0, tzinfo=timezone.utc),
                    context_window_end=datetime(2026, 6, 1, 18, 0, tzinfo=timezone.utc),
                    cross_source_summary_json={"sources": {}},
                )
            )
            session.commit()

        with Session(engine) as session:
            anomaly = session.execute(select(Anomaly)).scalar_one()
            assert isinstance(anomaly.id, uuid.UUID)
            assert anomaly.id == NUMERIC_LOOKING

            record = session.execute(select(EnrichmentRecord)).scalar_one()
            assert isinstance(record.anomaly_id, uuid.UUID)
            assert record.anomaly_id == NUMERIC_LOOKING
            # FK resolves back to the parent row through the migrated schema.
            assert record.anomaly.id == NUMERIC_LOOKING
    finally:
        engine.dispose()


def test_upgrade_head_creates_integrity_unique_constraints(sqlite_url: str) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        explanation_constraints = {
            constraint["name"]: tuple(constraint["column_names"])
            for constraint in inspect(engine).get_unique_constraints("explanations")
        }
        label_constraints = {
            constraint["name"]: tuple(constraint["column_names"])
            for constraint in inspect(engine).get_unique_constraints("expert_labels")
        }
        assert explanation_constraints["uq_explanations_anomaly_model"] == (
            "anomaly_id",
            "model_name",
        )
        assert label_constraints["uq_expert_labels_anomaly_labeler"] == (
            "anomaly_id",
            "labeler",
        )
    finally:
        engine.dispose()


@pytest.mark.parametrize("record_type", ("explanation", "expert_label"))
def test_migrated_schema_rejects_exact_duplicate_integrity_pair(
    sqlite_url: str,
    record_type: str,
) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")
    anomaly_id = uuid.uuid4()
    engine = create_engine(sqlite_url)
    try:
        with engine.begin() as connection:
            _insert_anomaly(connection, anomaly_id)
            if record_type == "explanation":
                _insert_explanation(
                    connection,
                    row_id=uuid.uuid4(),
                    anomaly_id=anomaly_id,
                    model_name="llama3:8b",
                )
            else:
                _insert_expert_label(
                    connection,
                    row_id=uuid.uuid4(),
                    anomaly_id=anomaly_id,
                    labeler="bracco",
                )

        with pytest.raises(IntegrityError):
            with engine.begin() as connection:
                if record_type == "explanation":
                    _insert_explanation(
                        connection,
                        row_id=uuid.uuid4(),
                        anomaly_id=anomaly_id,
                        model_name="llama3:8b",
                    )
                else:
                    _insert_expert_label(
                        connection,
                        row_id=uuid.uuid4(),
                        anomaly_id=anomaly_id,
                        labeler="bracco",
                    )

        with engine.begin() as connection:
            if record_type == "explanation":
                _insert_explanation(
                    connection,
                    row_id=uuid.uuid4(),
                    anomaly_id=anomaly_id,
                    model_name="LLAMA3:8B",
                )
                count = connection.execute(
                    text("SELECT COUNT(*) FROM explanations")
                ).scalar_one()
            else:
                _insert_expert_label(
                    connection,
                    row_id=uuid.uuid4(),
                    anomaly_id=anomaly_id,
                    labeler="Bracco",
                )
                count = connection.execute(
                    text("SELECT COUNT(*) FROM expert_labels")
                ).scalar_one()
        assert count == 2
    finally:
        engine.dispose()


def test_a6_upgrade_lists_legacy_duplicates_without_mutation(
    sqlite_url: str,
) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, PRE_A6_REVISION)
    anomaly_id = uuid.uuid4()
    engine = create_engine(sqlite_url)
    try:
        with engine.begin() as connection:
            _insert_anomaly(connection, anomaly_id)
            for _index in range(2):
                _insert_explanation(
                    connection,
                    row_id=uuid.uuid4(),
                    anomaly_id=anomaly_id,
                    model_name="llama3:8b",
                )
                _insert_expert_label(
                    connection,
                    row_id=uuid.uuid4(),
                    anomaly_id=anomaly_id,
                    labeler="bracco",
                )
            explanations_before = connection.execute(
                text("SELECT * FROM explanations ORDER BY id")
            ).all()
            labels_before = connection.execute(
                text("SELECT * FROM expert_labels ORDER BY id")
            ).all()

        with pytest.raises(RuntimeError) as exc_info:
            command.upgrade(cfg, "head")

        message = str(exc_info.value)
        assert "A-6 uniqueness preflight failed" in message
        assert "explanations" in message
        assert f"anomaly_id={anomaly_id.hex}" in message
        assert "model_name='llama3:8b'" in message
        assert "expert_labels" in message
        assert "labeler='bracco'" in message
        assert message.count("count=2") == 2

        with engine.connect() as connection:
            explanations_after = connection.execute(
                text("SELECT * FROM explanations ORDER BY id")
            ).all()
            labels_after = connection.execute(
                text("SELECT * FROM expert_labels ORDER BY id")
            ).all()
            revision = connection.execute(
                text("SELECT version_num FROM alembic_version")
            ).scalar_one()
        assert explanations_after == explanations_before
        assert labels_after == labels_before
        assert revision == PRE_A6_REVISION
        assert {
            constraint["name"]
            for constraint in inspect(engine).get_unique_constraints("explanations")
        } == set()
        assert {
            constraint["name"]
            for constraint in inspect(engine).get_unique_constraints("expert_labels")
        } == set()
    finally:
        engine.dispose()


def test_a6_downgrade_drops_only_constraints_and_preserves_rows(
    sqlite_url: str,
) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")
    anomaly_id = uuid.uuid4()
    engine = create_engine(sqlite_url)
    try:
        with engine.begin() as connection:
            _insert_anomaly(connection, anomaly_id)
            _insert_explanation(
                connection,
                row_id=uuid.uuid4(),
                anomaly_id=anomaly_id,
                model_name="llama3:8b",
            )
            _insert_expert_label(
                connection,
                row_id=uuid.uuid4(),
                anomaly_id=anomaly_id,
                labeler="mason",
            )
            explanations_before = connection.execute(
                text("SELECT * FROM explanations ORDER BY id")
            ).all()
            labels_before = connection.execute(
                text("SELECT * FROM expert_labels ORDER BY id")
            ).all()

        command.downgrade(cfg, PRE_A6_REVISION)

        assert {
            constraint["name"]
            for constraint in inspect(engine).get_unique_constraints("explanations")
        } == set()
        assert {
            constraint["name"]
            for constraint in inspect(engine).get_unique_constraints("expert_labels")
        } == set()
        with engine.connect() as connection:
            assert connection.execute(
                text("SELECT * FROM explanations ORDER BY id")
            ).all() == explanations_before
            assert connection.execute(
                text("SELECT * FROM expert_labels ORDER BY id")
            ).all() == labels_before
    finally:
        engine.dispose()


def test_downgrade_base_leaves_no_app_tables(sqlite_url) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")

    engine = create_engine(sqlite_url)
    try:
        tables = set(inspect(engine).get_table_names())
        # alembic_version is kept by Alembic's own bookkeeping
        assert tables - {"alembic_version"} == set(), (
            f"downgrade left app tables behind: {tables}"
        )
    finally:
        engine.dispose()
