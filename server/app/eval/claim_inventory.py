"""Claim inventory for the B14 subset selector.

Reads the hashed, read-only packet-source database and emits the exact claim
texts and stratum label for every frozen anomaly, in the same order the packet
generator and the labeling CLI present them.

CLI: ``python -m app.eval.claim_inventory --database <packets.db>
--expected-sha256 <sha256> --anomaly-set fixtures/eval50.json --output <out.json>``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

CHUNK_BYTES = 1024 * 1024


class ClaimInventoryError(ValueError):
    """Raised when inventory inputs violate the declared provenance rules."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_anomaly_ids(anomaly_set: Path) -> tuple[list[str], str]:
    try:
        fixture = json.loads(anomaly_set.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ClaimInventoryError(f"cannot read fixture {anomaly_set}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ClaimInventoryError(
            f"invalid JSON in fixture {anomaly_set}: {exc}"
        ) from exc
    if not isinstance(fixture, dict):
        raise ClaimInventoryError("fixture root must be an object")

    snapshot = fixture.get("snapshot_sha256")
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise ClaimInventoryError("fixture must carry a snapshot_sha256 string")

    anomaly_ids = fixture.get("anomaly_ids")
    if not isinstance(anomaly_ids, list) or not anomaly_ids:
        raise ClaimInventoryError("fixture anomaly_ids must be a nonempty array")

    seen: set[str] = set()
    validated: list[str] = []
    for rank, anomaly_id in enumerate(anomaly_ids, start=1):
        if not isinstance(anomaly_id, str) or not anomaly_id.strip():
            raise ClaimInventoryError(
                f"fixture anomaly ID at rank {rank} must be a nonempty string"
            )
        if anomaly_id in seen:
            raise ClaimInventoryError(f"duplicate fixture anomaly ID: {anomaly_id}")
        seen.add(anomaly_id)
        validated.append(anomaly_id)
    return validated, snapshot


def _claim_texts(
    connection: sqlite3.Connection, anomaly_id: str
) -> list[str]:
    """Exact claim texts for one anomaly, in label_cli presentation order.

    The ordering mirrors ``collect_claim_groups`` — model name then step index —
    so the inventory counts the same units the labeler is shown.
    """
    rows = connection.execute(
        """
        SELECT claims.claim_text
        FROM claims
        JOIN explanations ON claims.explanation_id = explanations.id
        WHERE explanations.anomaly_id = ?
        ORDER BY explanations.model_name, claims.step_index, claims.id
        """,
        (anomaly_id,),
    ).fetchall()
    return [str(row[0]) for row in rows]


def _stratum(connection: sqlite3.Connection, anomaly_id: str) -> str:
    row = connection.execute(
        "SELECT source, metric FROM anomalies WHERE id = ?",
        (anomaly_id,),
    ).fetchone()
    if row is None:
        raise ClaimInventoryError(f"anomaly absent from database: {anomaly_id}")
    source, metric = row
    if not source or not metric:
        raise ClaimInventoryError(
            f"anomaly {anomaly_id} is missing a source or metric"
        )
    return f"{source}/{metric}"


def build_inventory(
    database_path: Path,
    *,
    expected_sha256: str,
    anomaly_set: Path,
) -> dict[str, object]:
    """Collect claim texts and strata for every frozen anomaly."""
    resolved = database_path.resolve()
    if not resolved.exists():
        raise ClaimInventoryError(f"database does not exist: {resolved}")

    observed_sha256 = _sha256(resolved)
    if observed_sha256 != expected_sha256:
        raise ClaimInventoryError(
            f"database SHA-256 mismatch: {observed_sha256} != {expected_sha256}"
        )

    anomaly_ids, snapshot_sha256 = _frozen_anomaly_ids(anomaly_set)

    connection = sqlite3.connect(f"file:{resolved}?mode=ro&immutable=1", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        claims_by_anomaly: dict[str, list[str]] = {}
        strata_by_anomaly: dict[str, str] = {}
        empty: list[str] = []
        for anomaly_id in anomaly_ids:
            strata_by_anomaly[anomaly_id] = _stratum(connection, anomaly_id)
            texts = _claim_texts(connection, anomaly_id)
            if not texts:
                empty.append(anomaly_id)
            claims_by_anomaly[anomaly_id] = texts
    finally:
        connection.close()

    if empty:
        raise ClaimInventoryError(
            "frozen anomalies carry no claims: " + ", ".join(empty)
        )

    total_raw = sum(len(texts) for texts in claims_by_anomaly.values())
    total_unique = sum(
        len(dict.fromkeys(texts)) for texts in claims_by_anomaly.values()
    )
    return {
        "schema_version": 1,
        "provenance": {
            "database_sha256": observed_sha256,
            "fixture_snapshot_sha256": snapshot_sha256,
            "claim_order": "explanations.model_name, claims.step_index, claims.id",
        },
        "anomaly_count": len(anomaly_ids),
        "raw_claim_count": total_raw,
        "unique_claim_count": total_unique,
        "stratum_count": len(set(strata_by_anomaly.values())),
        "claims_by_anomaly": claims_by_anomaly,
        "strata_by_anomaly": strata_by_anomaly,
    }


def write_inventory(path: Path, inventory: dict[str, object]) -> None:
    """Write the inventory atomically so a crash cannot leave a partial file."""
    payload = json.dumps(inventory, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.claim_inventory",
        description=(
            "Emit the B14 claim inventory and stratum labels for the frozen "
            "anomaly set."
        ),
    )
    parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="read-only packet-source SQLite database",
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="recorded packet-source database SHA-256",
    )
    parser.add_argument(
        "--anomaly-set",
        required=True,
        type=Path,
        help="frozen fixture holding the authoritative ordered anomaly_ids",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the claim-inventory CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        inventory = build_inventory(
            args.database,
            expected_sha256=args.expected_sha256,
            anomaly_set=args.anomaly_set,
        )
        write_inventory(args.output, inventory)
    except (OSError, sqlite3.Error, ClaimInventoryError) as exc:
        _argument_error(parser, str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
