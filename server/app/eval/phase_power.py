"""B19-anchored execution wrapper for the pre-registered B16/P7 power grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

import app.eval.phase_analysis as phase_analysis_module
from app.eval.phase_analysis import (
    HEADLINE_TYPES,
    QUALITATIVE_ONLY_TYPES,
    PowerSimulationConfig,
    run_power_simulation,
    write_manifest,
)

CLAIM_CAP_STATUS = "proposed_pending_bracco_confirmation"
DESIGN_SOURCE = "B19 funnel dry-run"
IMPLEMENTATION_SOURCES = (
    ("app/eval/phase_analysis.py", Path(phase_analysis_module.__file__)),
    ("app/eval/phase_power.py", Path(__file__)),
)


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _array(value: object, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array")
    return value


def _string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _integer(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return value


def _string_array(value: object, field_name: str) -> list[str]:
    result = [
        _string(item, f"{field_name} item")
        for item in _array(value, field_name)
    ]
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} must contain unique values")
    return result


def _sha256(value: object, field_name: str) -> str:
    text = _string(value, field_name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a 64-character SHA-256")
    return text


def _provenance(
    value: object,
    field_name: str,
) -> dict[str, object]:
    source = _mapping(value, field_name)
    if source.get("disposable_b19_not_official") is not True:
        raise ValueError(f"{field_name} must identify a disposable B19 run")
    iteration = _integer(source.get("iteration"), f"{field_name}.iteration")
    if iteration < 1:
        raise ValueError(f"{field_name}.iteration must be positive")
    git_commit = _string(source.get("git_commit"), f"{field_name}.git_commit")
    if len(git_commit) != 40 or any(
        character not in "0123456789abcdef" for character in git_commit.lower()
    ):
        raise ValueError(f"{field_name}.git_commit must be a full Git object ID")
    return {
        "db_copy_sha256": _sha256(
            source.get("db_copy_sha256"), f"{field_name}.db_copy_sha256"
        ),
        "disposable_b19_not_official": True,
        "git_commit": git_commit.lower(),
        "iteration": iteration,
        "selected_anomaly_ids": _string_array(
            source.get("selected_anomaly_ids"),
            f"{field_name}.selected_anomaly_ids",
        ),
    }


def _validated_run(
    payload: object,
    report: object,
) -> tuple[list[str], list[Mapping[str, object]], dict[str, object]]:
    payload_root = _mapping(payload, "B19 payload")
    report_root = _mapping(report, "B19 report")
    if payload_root.get("schema_version") != 1:
        raise ValueError("B19 payload schema_version must equal 1")
    if report_root.get("schema_version") != 1:
        raise ValueError("B19 report schema_version must equal 1")

    report_inputs = _mapping(payload_root.get("report_inputs"), "report_inputs")
    payload_provenance = _provenance(
        report_inputs.get("provenance"), "payload provenance"
    )
    report_provenance = _provenance(
        report_root.get("provenance"), "report provenance"
    )
    if payload_provenance != report_provenance:
        raise ValueError("B19 payload/report provenance mismatch")
    selected_ids = list(payload_provenance["selected_anomaly_ids"])
    if len(selected_ids) != 5:
        raise ValueError("B19 power design requires exactly five selected anomalies")

    payload_selection = _mapping(
        _mapping(payload_root.get("pipeline"), "pipeline").get("selection"),
        "pipeline.selection",
    )
    report_selection = _mapping(report_root.get("selection"), "report.selection")
    selection_surfaces = (
        _string_array(
            payload_selection.get("selected_anomaly_ids"),
            "pipeline selected_anomaly_ids",
        ),
        _string_array(
            report_selection.get("selected_anomaly_ids"),
            "report selected_anomaly_ids",
        ),
    )
    if any(surface != selected_ids for surface in selection_surfaces):
        raise ValueError("B19 selected-anomaly order mismatch")
    if _sha256(payload_root.get("database_sha256"), "payload database_sha256") != str(
        payload_provenance["db_copy_sha256"]
    ):
        raise ValueError("B19 payload database/provenance hash mismatch")

    go_no_go = _mapping(report_root.get("go_no_go"), "go_no_go")
    if go_no_go.get("status") != "go":
        raise ValueError("B19 report status must be go")
    if _array(go_no_go.get("hard_stops"), "go_no_go.hard_stops"):
        raise ValueError("B19 report must have zero hard stops")
    if _array(go_no_go.get("review_items"), "go_no_go.review_items"):
        raise ValueError("B19 report must have zero triggered review items")
    cell_audit = _mapping(report_root.get("cell_audit"), "cell_audit")
    if (
        cell_audit.get("expected_cells") != 15
        or cell_audit.get("completed_cells") != 15
        or _array(cell_audit.get("missing_cells"), "cell_audit.missing_cells")
        or _array(
            cell_audit.get("unexpected_cells"), "cell_audit.unexpected_cells"
        )
    ):
        raise ValueError("B19 report must contain exactly 15 completed cells")

    raw_claims = _array(report_inputs.get("claims"), "report_inputs.claims")
    claims = [_mapping(claim, f"claim {index}") for index, claim in enumerate(raw_claims)]
    counting_units = _mapping(report_root.get("counting_units"), "counting_units")
    if counting_units.get("claim_rows") != len(claims):
        raise ValueError("B19 claim-row count disagrees with finalized report")
    return selected_ids, claims, payload_provenance


def _normalized_claims(
    claims: Sequence[Mapping[str, object]],
    selected_ids: Sequence[str],
) -> list[dict[str, object]]:
    selected_set = set(selected_ids)
    seen_claim_ids: set[str] = set()
    normalized: list[dict[str, object]] = []
    for index, claim in enumerate(claims, start=1):
        claim_id = _string(claim.get("claim_id"), f"claim {index}.claim_id")
        if claim_id in seen_claim_ids:
            raise ValueError(f"duplicate B19 claim ID: {claim_id}")
        seen_claim_ids.add(claim_id)
        anomaly_id = _string(
            claim.get("anomaly_id"), f"claim {claim_id}.anomaly_id"
        )
        if anomaly_id not in selected_set:
            raise ValueError(f"B19 claim lies outside selected anomalies: {claim_id}")
        grounding = _string(
            claim.get("grounding_verdict"),
            f"claim {claim_id}.grounding_verdict",
        )
        if grounding not in {"grounded", "unverified"}:
            raise ValueError(
                f"claim {claim_id}.grounding_verdict must be grounded or unverified"
            )
        score = claim.get("corroboration_score")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise ValueError(
                    f"claim {claim_id}.corroboration_score must be finite or null"
                )
            score = float(score)
            if not math.isfinite(score) or not -1.0 <= score <= 1.0:
                raise ValueError(
                    f"claim {claim_id}.corroboration_score must be in [-1, 1]"
                )
        normalized.append(
            {
                "anomaly_id": anomaly_id,
                "claim_id": claim_id,
                "claim_text": _string(
                    claim.get("claim_text"), f"claim {claim_id}.claim_text"
                ),
                "claim_type": _string(
                    claim.get("claim_type"), f"claim {claim_id}.claim_type"
                ),
                "corroboration_score": score,
                "grounding_verdict": grounding,
            }
        )
    return sorted(
        normalized,
        key=lambda row: (
            selected_ids.index(str(row["anomaly_id"])),
            str(row["claim_text"]),
            str(row["claim_id"]),
        ),
    )


def derive_b19_power_design(
    payload: object,
    report: object,
    *,
    claim_cap: int,
) -> dict[str, object]:
    """Derive the label-free power populations from accepted B19 artifacts."""
    if type(claim_cap) is not int or claim_cap < 1:
        raise ValueError("claim_cap must be a positive integer")
    selected_ids, raw_claims, provenance = _validated_run(payload, report)
    claims = _normalized_claims(raw_claims, selected_ids)
    claims_by_anomaly: dict[str, list[dict[str, object]]] = defaultdict(list)
    for claim in claims:
        claims_by_anomaly[str(claim["anomaly_id"])].append(claim)
    missing = [anomaly_id for anomaly_id in selected_ids if not claims_by_anomaly[anomaly_id]]
    if missing:
        raise ValueError(f"B19 selected anomalies without claims: {missing}")

    packet_counts = [
        len(
            {
                str(claim["claim_text"])
                for claim in claims_by_anomaly[anomaly_id]
            }
        )
        for anomaly_id in selected_ids
    ]
    selected_subset: list[str] = []
    selected_total = 0
    first_excluded: str | None = None
    for anomaly_id, count in zip(selected_ids, packet_counts, strict=True):
        if selected_total + count > claim_cap:
            first_excluded = anomaly_id
            break
        selected_subset.append(anomaly_id)
        selected_total += count

    agreement_sizes: list[int] = []
    spearman_sizes: list[int] = []
    type_counts: Counter[str] = Counter()
    eligible_type_counts: Counter[str] = Counter()
    for anomaly_id in selected_subset:
        grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
        for claim in claims_by_anomaly[anomaly_id]:
            grouped[str(claim["claim_text"])].append(claim)
            type_counts[str(claim["claim_type"])] += 1
        agreement_count = 0
        for claim_text, decision_claims in sorted(grouped.items()):
            qualitative = {
                str(claim["claim_type"]) in QUALITATIVE_ONLY_TYPES
                for claim in decision_claims
            }
            if len(qualitative) > 1:
                raise ValueError(
                    "conflicting quantitative/qualitative types for decision "
                    f"{anomaly_id}/{claim_text}"
                )
            agreement_count += qualitative == {False}
        agreement_sizes.append(agreement_count)

        eligible = [
            claim
            for claim in claims_by_anomaly[anomaly_id]
            if claim["claim_type"] in HEADLINE_TYPES
            and claim["grounding_verdict"] == "grounded"
            and claim["corroboration_score"] is not None
        ]
        spearman_sizes.append(len(eligible))
        eligible_type_counts.update(str(claim["claim_type"]) for claim in eligible)

    unique_unit_count = sum(packet_counts)
    counting_units = _mapping(_mapping(report, "B19 report").get("counting_units"), "counting_units")
    if counting_units.get("unique_anomaly_exact_text") != unique_unit_count:
        raise ValueError("B19 unique-text count disagrees with finalized report")
    return {
        "source_iteration": provenance["iteration"],
        "claim_cap": claim_cap,
        "claim_cap_unit": "unique (anomaly_id, exact claim_text)",
        "claim_cap_status": CLAIM_CAP_STATUS,
        "b19_selected_anomaly_ids": selected_ids,
        "packet_unique_claim_counts": packet_counts,
        "selected_anomaly_ids": selected_subset,
        "selected_unique_claim_total": selected_total,
        "first_excluded_anomaly_id": first_excluded,
        "agreement_cluster_sizes": agreement_sizes,
        "agreement_decision_count": sum(agreement_sizes),
        "spearman_cluster_sizes": spearman_sizes,
        "spearman_eligible_claim_count": sum(spearman_sizes),
        "spearman_contributing_anomaly_count": sum(
            count > 0 for count in spearman_sizes
        ),
        "spearman_confirmatory_floor_met": (
            sum(spearman_sizes) >= 20
            and sum(count > 0 for count in spearman_sizes) >= 5
        ),
        "claim_type_counts": dict(sorted(type_counts.items())),
        "spearman_eligible_type_counts": dict(
            sorted(eligible_type_counts.items())
        ),
    }


def build_power_manifest(
    payload: object,
    report: object,
    *,
    payload_sha256: str,
    report_sha256: str,
    claim_cap: int,
) -> dict[str, object]:
    """Build the canonical B19-anchored production power manifest."""
    payload_digest = _sha256(payload_sha256, "payload_sha256")
    report_digest = _sha256(report_sha256, "report_sha256")
    design = derive_b19_power_design(payload, report, claim_cap=claim_cap)
    implementation = _implementation_provenance()
    agreement_sizes = tuple(int(size) for size in design["agreement_cluster_sizes"])
    spearman_sizes = tuple(int(size) for size in design["spearman_cluster_sizes"])
    config = PowerSimulationConfig(
        cluster_sizes=agreement_sizes,
        spearman_cluster_sizes=spearman_sizes,
        design_source=DESIGN_SOURCE,
    )
    simulation = run_power_simulation(config)
    return {
        "schema_version": 1,
        "pre_registration_status": "pre-registered 2026-07-16",
        "run_kind": "B19-anchored production power",
        "inputs": {
            "b19_payload_sha256": payload_digest,
            "b19_report_sha256": report_digest,
        },
        "implementation": implementation,
        "design": design,
        "simulation": simulation,
    }


def _load_json(path: Path, field_name: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read {field_name} {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {field_name} {path}: {exc}") from exc


def _file_sha256(path: Path, field_name: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"cannot hash {field_name} {path}: {exc}") from exc
    return digest.hexdigest()


def _implementation_provenance() -> dict[str, object]:
    combined = hashlib.sha256()
    source_hashes: dict[str, str] = {}
    for source_name, source_path in IMPLEMENTATION_SOURCES:
        try:
            content = source_path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"cannot read power implementation source {source_path}: {exc}"
            ) from exc
        source_hashes[source_name] = hashlib.sha256(content).hexdigest()
        combined.update(source_name.encode("utf-8"))
        combined.update(b"\0")
        combined.update(content)
        combined.update(b"\0")
    return {
        "source_sha256": source_hashes,
        "combined_sha256": combined.hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the B19-anchored pre-registered B16/P7 power grid."
    )
    parser.add_argument("--b19-payload", type=Path, required=True)
    parser.add_argument("--b19-report", type=Path, required=True)
    parser.add_argument("--claim-cap", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic B19-anchored power CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_power_manifest(
            _load_json(args.b19_payload, "B19 payload"),
            _load_json(args.b19_report, "B19 report"),
            payload_sha256=_file_sha256(args.b19_payload, "B19 payload"),
            report_sha256=_file_sha256(args.b19_report, "B19 report"),
            claim_cap=args.claim_cap,
        )
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as exc:
        _argument_error(parser, str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
