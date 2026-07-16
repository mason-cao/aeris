"""Deterministic complete-anomaly selection under a unique-claim cap."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn

DEFAULT_CLAIM_CAP = 200


class SubsetSelectionError(ValueError):
    """Raised when selector inputs violate the declared B14 protocol."""


def _validate_claim_cap(claim_cap: object) -> int:
    if type(claim_cap) is not int or claim_cap <= 0:
        raise SubsetSelectionError("claim cap must be a positive integer")
    return claim_cap


def _validate_anomaly_ids(anomaly_ids: object) -> list[str]:
    if not isinstance(anomaly_ids, list):
        raise SubsetSelectionError("fixture anomaly_ids must be an array")

    validated: list[str] = []
    seen: set[str] = set()
    for rank, anomaly_id in enumerate(anomaly_ids, start=1):
        if not isinstance(anomaly_id, str) or not anomaly_id.strip():
            raise SubsetSelectionError(
                f"fixture anomaly ID at rank {rank} must be a nonempty string"
            )
        if anomaly_id in seen:
            raise SubsetSelectionError(f"duplicate anomaly ID: {anomaly_id}")
        seen.add(anomaly_id)
        validated.append(anomaly_id)
    return validated


def _validate_claims_by_anomaly(
    claims_by_anomaly: object,
    anomaly_ids: Sequence[str],
) -> dict[str, list[str]]:
    if not isinstance(claims_by_anomaly, dict):
        raise SubsetSelectionError("claims_by_anomaly must be an object")
    for anomaly_id in claims_by_anomaly:
        if not isinstance(anomaly_id, str) or not anomaly_id.strip():
            raise SubsetSelectionError(
                "every claims_by_anomaly key must be a nonempty string"
            )

    expected = set(anomaly_ids)
    supplied = set(claims_by_anomaly)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing:
        raise SubsetSelectionError(
            f"missing anomaly inventories: {', '.join(missing)}"
        )
    if extra:
        raise SubsetSelectionError(f"extra anomaly inventories: {', '.join(extra)}")

    validated: dict[str, list[str]] = {}
    for anomaly_id in anomaly_ids:
        claims = claims_by_anomaly[anomaly_id]
        if not isinstance(claims, list) or not claims:
            raise SubsetSelectionError(
                f"claims for {anomaly_id} must be a nonempty claim array"
            )
        for position, claim in enumerate(claims, start=1):
            if not isinstance(claim, str) or not claim.strip():
                raise SubsetSelectionError(
                    f"claim {position} for {anomaly_id} must be a nonempty string"
                )
        validated[anomaly_id] = list(claims)
    return validated


def build_subset_manifest(
    anomaly_ids: object,
    claims_by_anomaly: object,
    *,
    claim_cap: object = DEFAULT_CLAIM_CAP,
) -> dict[str, object]:
    """Select the maximal complete-anomaly prefix that does not exceed the cap."""
    cap = _validate_claim_cap(claim_cap)
    ranked_ids = _validate_anomaly_ids(anomaly_ids)
    inventories = _validate_claims_by_anomaly(claims_by_anomaly, ranked_ids)

    selected_ids: list[str] = []
    audit: list[dict[str, object]] = []
    selected_claim_count = 0
    stopped_before: str | None = None

    for rank, anomaly_id in enumerate(ranked_ids, start=1):
        claims = inventories[anomaly_id]
        unique_claim_count = len(dict.fromkeys(claims))
        prospective_count = selected_claim_count + unique_claim_count
        decision = "include" if prospective_count <= cap else "stop_before"
        audit.append(
            {
                "rank": rank,
                "anomaly_id": anomaly_id,
                "raw_claim_count": len(claims),
                "unique_claim_count": unique_claim_count,
                "cumulative_unique_claims_before": selected_claim_count,
                "prospective_unique_claim_count": prospective_count,
                "decision": decision,
            }
        )
        if decision == "stop_before":
            stopped_before = anomaly_id
            break
        selected_ids.append(anomaly_id)
        selected_claim_count = prospective_count

    stopped = stopped_before is not None
    return {
        "schema_version": 1,
        "protocol": {
            "claim_cap": cap,
            "cap_unit": "unique exact claim texts within each anomaly",
            "deduplication": (
                "exact text within anomaly; no normalization; "
                "cross-anomaly texts count separately"
            ),
            "rank_source": "fixture.anomaly_ids",
            "stopping_rule": (
                "ordered complete-anomaly prefix; stop before first anomaly "
                "that would exceed cap"
            ),
        },
        "available_ranked_anomaly_count": len(ranked_ids),
        "inspected_anomaly_count": len(audit),
        "uninspected_ranked_anomaly_count": len(ranked_ids) - len(audit),
        "selected_anomaly_ids": selected_ids,
        "selected_unique_claim_count": selected_claim_count,
        "stopped_before_anomaly_id": stopped_before,
        "stop_reason": (
            "next complete anomaly would exceed claim cap"
            if stopped
            else "ranked input exhausted"
        ),
        "audit": audit,
    }


def _load_json(path: Path, description: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SubsetSelectionError(f"cannot read {description} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SubsetSelectionError(
            f"invalid JSON in {description} {path}: {exc}"
        ) from exc


def _required_object(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise SubsetSelectionError(f"{description} root must be an object")
    return value


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
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
        description=(
            "Select a deterministic complete-anomaly prefix under the B14 "
            "unique-claim cap."
        )
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        required=True,
        help="Freeze-style JSON containing authoritative ordered anomaly_ids.",
    )
    parser.add_argument(
        "--claims",
        type=Path,
        required=True,
        help="JSON containing claims_by_anomaly arrays of exact claim text.",
    )
    parser.add_argument(
        "--cap",
        type=int,
        default=DEFAULT_CLAIM_CAP,
        help=(
            "Positive unique-claim cap (default: proposed 200; confirmation "
            "remains external to this tool)."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the B14 selector CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        fixture = _required_object(_load_json(args.fixture, "fixture"), "fixture")
        inventory = _required_object(
            _load_json(args.claims, "claim inventory"), "claim inventory"
        )
        manifest = build_subset_manifest(
            fixture.get("anomaly_ids"),
            inventory.get("claims_by_anomaly"),
            claim_cap=args.cap,
        )
        _write_manifest(args.output, manifest)
    except (OSError, SubsetSelectionError) as exc:
        _argument_error(parser, str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
