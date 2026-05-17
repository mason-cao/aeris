from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


REPO_ROOT = Path(__file__).resolve().parents[2]


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
    ["data_points", "data_sources", "anomalies", "enrichment_records"],
)
def test_upgrade_head_creates_expected_table(sqlite_url, table) -> None:
    cfg = _alembic_config(sqlite_url)
    command.upgrade(cfg, "head")

    engine = create_engine(sqlite_url)
    try:
        assert table in set(inspect(engine).get_table_names())
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
