"""Claim inventory for the B14 subset selector."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from app.eval.claim_inventory import (
    ClaimInventoryError,
    build_inventory,
    main,
)

SNAPSHOT = "1e50e007cc8f60da1fd5ce4588aced05b186172817e757f9948a9adad06be557"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_database(path: Path, rows: list[tuple[str, str, str, str, int, str]]) -> None:
    """Seed a minimal packet-source database.

    Each row is (anomaly_id, source, metric, model_name, step_index, claim_text).
    """
    connection = sqlite3.connect(path)
    try:
        connection.executescript(
            """
            CREATE TABLE anomalies (id TEXT PRIMARY KEY, source TEXT, metric TEXT);
            CREATE TABLE explanations (
                id TEXT PRIMARY KEY, anomaly_id TEXT, model_name TEXT
            );
            CREATE TABLE claims (
                id TEXT PRIMARY KEY,
                explanation_id TEXT,
                step_index INTEGER,
                claim_text TEXT
            );
            """
        )
        anomalies: dict[str, tuple[str, str]] = {}
        explanations: dict[tuple[str, str], str] = {}
        for index, (anomaly_id, source, metric, model, step, text) in enumerate(rows):
            anomalies[anomaly_id] = (source, metric)
            key = (anomaly_id, model)
            if key not in explanations:
                explanation_id = f"e{len(explanations)}"
                explanations[key] = explanation_id
                connection.execute(
                    "INSERT INTO explanations VALUES (?, ?, ?)",
                    (explanation_id, anomaly_id, model),
                )
            connection.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?)",
                (f"c{index}", explanations[key], step, text),
            )
        for anomaly_id, (source, metric) in anomalies.items():
            connection.execute(
                "INSERT INTO anomalies VALUES (?, ?, ?)", (anomaly_id, source, metric)
            )
        connection.commit()
    finally:
        connection.close()


def _write_fixture(path: Path, anomaly_ids: list[str]) -> None:
    path.write_text(
        json.dumps({"snapshot_sha256": SNAPSHOT, "anomaly_ids": anomaly_ids}),
        encoding="utf-8",
    )


def test_inventory_collects_claims_and_strata_in_presentation_order(
    tmp_path: Path,
) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(
        database,
        [
            ("a0", "tceq", "so2", "zeta", 1, "z-first"),
            ("a0", "tceq", "so2", "alpha", 2, "a-second"),
            ("a0", "tceq", "so2", "alpha", 1, "a-first"),
            ("a1", "openaq", "ozone", "alpha", 1, "other"),
        ],
    )
    _write_fixture(fixture, ["a0", "a1"])

    inventory = build_inventory(
        database, expected_sha256=_sha256(database), anomaly_set=fixture
    )

    # Model name then step index, exactly as collect_claim_groups orders them.
    assert inventory["claims_by_anomaly"]["a0"] == ["a-first", "a-second", "z-first"]
    assert inventory["strata_by_anomaly"] == {
        "a0": "tceq/so2",
        "a1": "openaq/ozone",
    }
    assert inventory["stratum_count"] == 2
    assert inventory["raw_claim_count"] == 4
    assert inventory["provenance"]["fixture_snapshot_sha256"] == SNAPSHOT


def test_inventory_counts_duplicates_raw_and_unique_separately(tmp_path: Path) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(
        database,
        [
            ("a0", "tceq", "so2", "alpha", 1, "same"),
            ("a0", "tceq", "so2", "beta", 1, "same"),
            ("a0", "tceq", "so2", "beta", 2, "other"),
        ],
    )
    _write_fixture(fixture, ["a0"])

    inventory = build_inventory(
        database, expected_sha256=_sha256(database), anomaly_set=fixture
    )

    assert inventory["raw_claim_count"] == 3
    assert inventory["unique_claim_count"] == 2


def test_inventory_follows_the_fixture_order_not_the_database_order(
    tmp_path: Path,
) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(
        database,
        [
            ("a0", "tceq", "so2", "alpha", 1, "first"),
            ("a1", "openaq", "ozone", "alpha", 1, "second"),
        ],
    )
    _write_fixture(fixture, ["a1", "a0"])

    inventory = build_inventory(
        database, expected_sha256=_sha256(database), anomaly_set=fixture
    )

    assert list(inventory["claims_by_anomaly"]) == ["a1", "a0"]


def test_hash_mismatch_refuses_before_reading(tmp_path: Path) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(database, [("a0", "tceq", "so2", "alpha", 1, "text")])
    _write_fixture(fixture, ["a0"])

    with pytest.raises(ClaimInventoryError, match="SHA-256 mismatch"):
        build_inventory(database, expected_sha256="0" * 64, anomaly_set=fixture)


def test_anomaly_absent_from_database_is_an_error(tmp_path: Path) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(database, [("a0", "tceq", "so2", "alpha", 1, "text")])
    _write_fixture(fixture, ["a0", "missing"])

    with pytest.raises(ClaimInventoryError, match="anomaly absent from database"):
        build_inventory(
            database, expected_sha256=_sha256(database), anomaly_set=fixture
        )


def test_frozen_anomaly_without_claims_is_an_error(tmp_path: Path) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(database, [("a0", "tceq", "so2", "alpha", 1, "text")])
    connection = sqlite3.connect(database)
    connection.execute("INSERT INTO anomalies VALUES ('a1', 'openaq', 'ozone')")
    connection.commit()
    connection.close()
    _write_fixture(fixture, ["a0", "a1"])

    with pytest.raises(ClaimInventoryError, match="carry no claims: a1"):
        build_inventory(
            database, expected_sha256=_sha256(database), anomaly_set=fixture
        )


@pytest.mark.parametrize(
    ("fixture_body", "message"),
    [
        ({"anomaly_ids": ["a0"]}, "snapshot_sha256"),
        ({"snapshot_sha256": SNAPSHOT}, "nonempty array"),
        ({"snapshot_sha256": SNAPSHOT, "anomaly_ids": []}, "nonempty array"),
        (
            {"snapshot_sha256": SNAPSHOT, "anomaly_ids": ["a0", "a0"]},
            "duplicate fixture anomaly ID",
        ),
        ({"snapshot_sha256": SNAPSHOT, "anomaly_ids": [""]}, "nonempty string"),
    ],
)
def test_malformed_fixture_is_rejected(
    tmp_path: Path, fixture_body: dict[str, object], message: str
) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    _build_database(database, [("a0", "tceq", "so2", "alpha", 1, "text")])
    fixture.write_text(json.dumps(fixture_body), encoding="utf-8")

    with pytest.raises(ClaimInventoryError, match=message):
        build_inventory(
            database, expected_sha256=_sha256(database), anomaly_set=fixture
        )


def test_cli_output_is_byte_identical_across_runs(tmp_path: Path) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    _build_database(
        database,
        [
            ("a0", "tceq", "so2", "alpha", 1, "one"),
            ("a1", "openaq", "ozone", "alpha", 1, "two"),
        ],
    )
    _write_fixture(fixture, ["a0", "a1"])
    arguments = [
        "--database",
        str(database),
        "--expected-sha256",
        _sha256(database),
        "--anomaly-set",
        str(fixture),
    ]

    assert main([*arguments, "--output", str(first)]) == 0
    assert main([*arguments, "--output", str(second)]) == 0

    assert first.read_bytes() == second.read_bytes()


def test_cli_failure_leaves_output_absent(tmp_path: Path) -> None:
    database = tmp_path / "packets.db"
    fixture = tmp_path / "eval.json"
    output = tmp_path / "out.json"
    _build_database(database, [("a0", "tceq", "so2", "alpha", 1, "text")])
    _write_fixture(fixture, ["a0"])

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--database",
                str(database),
                "--expected-sha256",
                "0" * 64,
                "--anomaly-set",
                str(fixture),
                "--output",
                str(output),
            ]
        )

    assert not output.exists()
