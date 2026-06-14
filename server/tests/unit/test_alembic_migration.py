import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text


REPO_ROOT = Path(__file__).resolve().parents[2]

# A valid uuid4 whose undashed hex is digits plus a single 'e' — the shape a
# NUMERIC-affinity column coerces to REAL/Inf on SQLite (the production bug).
NUMERIC_LOOKING = uuid.UUID("12345678-9012-4456-8e78-901234567890")


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
