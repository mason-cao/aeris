from pathlib import Path

from app.config import Settings


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_env_example_uses_async_postgres_driver() -> None:
    env_example = (REPO_ROOT / "server" / ".env.example").read_text()

    assert "DATABASE_URL=postgresql+asyncpg://" in env_example


def test_settings_normalizes_plain_postgres_url_for_async_engine() -> None:
    settings = Settings(database_url="postgresql://aeris:aeris@localhost:5432/aeris")

    assert (
        settings.database_url
        == "postgresql+asyncpg://aeris:aeris@localhost:5432/aeris"
    )


def test_docker_compose_mounts_non_ha_timescale_data_directory() -> None:
    docker_compose = (REPO_ROOT / "server" / "docker-compose.yml").read_text()

    assert "timescale/timescaledb:latest-pg16" in docker_compose
    assert "/var/lib/postgresql/data" in docker_compose
    assert "/home/postgres/pgdata/data" not in docker_compose


def test_test_database_fixture_uses_temp_path() -> None:
    conftest = (REPO_ROOT / "server" / "tests" / "conftest.py").read_text()

    assert "tmp_path_factory" in conftest
    assert "test_aeris.db" not in conftest
