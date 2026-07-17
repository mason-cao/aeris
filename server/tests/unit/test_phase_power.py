from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.eval.phase_power import (
    build_power_manifest,
    derive_b19_power_design,
    main,
)


def _claim(
    anomaly_id: str,
    index: int,
    *,
    claim_type: str = "unclassified",
    grounded: bool = False,
    scored: bool = False,
) -> dict[str, object]:
    return {
        "anomaly_id": anomaly_id,
        "claim_id": f"{anomaly_id}-c{index}",
        "claim_text": f"{anomaly_id} claim {index}",
        "claim_type": claim_type,
        "grounding_verdict": "grounded" if grounded else "unverified",
        "corroboration_score": 0.25 if scored else None,
        "model": f"m{index % 3}",
    }


def _b19_inputs() -> tuple[dict[str, object], dict[str, object]]:
    anomaly_ids = [f"a{index}" for index in range(5)]
    qualitative_counts = [1, 0, 0, 0, 2]
    eligible_counts = [7, 1, 3, 4, 0]
    claims: list[dict[str, object]] = []
    for anomaly_id, qualitative_count, eligible_count in zip(
        anomaly_ids, qualitative_counts, eligible_counts, strict=True
    ):
        for index in range(36):
            if index < qualitative_count:
                claim_type = "point_source_attribution"
                grounded = True
                scored = True
            elif index < qualitative_count + eligible_count:
                claim_type = "concentration_elevation"
                grounded = True
                scored = True
            else:
                claim_type = "unclassified"
                grounded = False
                scored = False
            claims.append(
                _claim(
                    anomaly_id,
                    index,
                    claim_type=claim_type,
                    grounded=grounded,
                    scored=scored,
                )
            )
    provenance = {
        "db_copy_sha256": "b" * 64,
        "disposable_b19_not_official": True,
        "git_commit": "c" * 40,
        "iteration": 1,
        "selected_anomaly_ids": anomaly_ids,
    }
    payload = {
        "schema_version": 1,
        "database_sha256": "b" * 64,
        "pipeline": {
            "selection": {
                "selected_anomaly_ids": anomaly_ids,
            }
        },
        "report_inputs": {
            "claims": claims,
            "provenance": copy.deepcopy(provenance),
        },
    }
    report = {
        "schema_version": 1,
        "provenance": copy.deepcopy(provenance),
        "selection": {"selected_anomaly_ids": anomaly_ids},
        "counting_units": {
            "claim_rows": 180,
            "unique_anomaly_exact_text": 180,
        },
        "cell_audit": {
            "expected_cells": 15,
            "completed_cells": 15,
            "missing_cells": [],
            "unexpected_cells": [],
        },
        "go_no_go": {
            "status": "go",
            "hard_stops": [],
            "review_items": [],
        },
    }
    return payload, report


def test_b19_design_uses_declared_distinct_populations() -> None:
    payload, report = _b19_inputs()

    design = derive_b19_power_design(payload, report, claim_cap=200)

    assert design["selected_anomaly_ids"] == [f"a{index}" for index in range(5)]
    assert design["packet_unique_claim_counts"] == [36, 36, 36, 36, 36]
    assert design["selected_unique_claim_total"] == 180
    assert design["agreement_cluster_sizes"] == [35, 36, 36, 36, 34]
    assert design["agreement_decision_count"] == 177
    assert design["spearman_cluster_sizes"] == [7, 1, 3, 4, 0]
    assert design["spearman_eligible_claim_count"] == 15
    assert design["spearman_contributing_anomaly_count"] == 4
    assert design["spearman_confirmatory_floor_met"] is False
    assert design["claim_cap_status"] == "proposed_pending_bracco_confirmation"


def test_claim_cap_includes_exact_boundary_and_stops_without_substitution() -> None:
    payload, report = _b19_inputs()

    exact = derive_b19_power_design(payload, report, claim_cap=180)
    stopped = derive_b19_power_design(payload, report, claim_cap=179)

    assert exact["selected_unique_claim_total"] == 180
    assert len(exact["selected_anomaly_ids"]) == 5
    assert stopped["selected_unique_claim_total"] == 144
    assert stopped["selected_anomaly_ids"] == ["a0", "a1", "a2", "a3"]
    assert stopped["first_excluded_anomaly_id"] == "a4"


def test_b19_design_rejects_conflicting_types_for_one_decision() -> None:
    payload, report = _b19_inputs()
    duplicate = copy.deepcopy(payload["report_inputs"]["claims"][0])
    duplicate["claim_id"] = "duplicate"
    duplicate["claim_type"] = "concentration_elevation"
    payload["report_inputs"]["claims"].append(duplicate)
    report["counting_units"]["claim_rows"] = 181

    with pytest.raises(ValueError, match="conflicting quantitative/qualitative"):
        derive_b19_power_design(payload, report, claim_cap=200)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("go_no_go", "status"), "stop"),
        (("cell_audit", "completed_cells"), 14),
        (("provenance", "iteration"), 2),
    ],
)
def test_b19_design_rejects_unaccepted_or_mismatched_run(
    path: tuple[str, str],
    value: object,
) -> None:
    payload, report = _b19_inputs()
    report[path[0]][path[1]] = value

    with pytest.raises(ValueError):
        derive_b19_power_design(payload, report, claim_cap=200)


def test_power_manifest_records_hashes_and_separate_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, report = _b19_inputs()
    captured: dict[str, object] = {}

    def fake_simulation(config: object) -> dict[str, object]:
        captured["config"] = config
        return {"schema_version": 1, "official_monte_carlo": True}

    monkeypatch.setattr("app.eval.phase_power.run_power_simulation", fake_simulation)
    manifest = build_power_manifest(
        payload,
        report,
        payload_sha256="d" * 64,
        report_sha256="e" * 64,
        claim_cap=200,
    )

    config = captured["config"]
    assert config.cluster_sizes == (35, 36, 36, 36, 34)
    assert config.spearman_cluster_sizes == (7, 1, 3, 4, 0)
    assert config.outer_replicates == 2_000
    assert config.inner_bootstrap_resamples == 2_000
    assert manifest["inputs"]["b19_payload_sha256"] == "d" * 64
    assert manifest["inputs"]["b19_report_sha256"] == "e" * 64
    implementation = manifest["implementation"]
    assert set(implementation["source_sha256"]) == {
        "app/eval/phase_analysis.py",
        "app/eval/phase_power.py",
    }
    assert len(implementation["combined_sha256"]) == 64


def test_cli_is_byte_deterministic_and_failure_leaves_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload, report = _b19_inputs()
    payload_path = tmp_path / "payload.json"
    report_path = tmp_path / "report.json"
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    monkeypatch.setattr(
        "app.eval.phase_power.run_power_simulation",
        lambda _config: {"schema_version": 1, "official_monte_carlo": True},
    )

    for output_path in (first_path, second_path):
        assert (
            main(
                [
                    "--b19-payload",
                    str(payload_path),
                    "--b19-report",
                    str(report_path),
                    "--claim-cap",
                    "200",
                    "--output",
                    str(output_path),
                ]
            )
            == 0
        )
    assert first_path.read_bytes() == second_path.read_bytes()

    report["cell_audit"]["completed_cells"] = 14
    report_path.write_text(json.dumps(report), encoding="utf-8")
    failed_path = tmp_path / "failed.json"
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--b19-payload",
                str(payload_path),
                "--b19-report",
                str(report_path),
                "--claim-cap",
                "200",
                "--output",
                str(failed_path),
            ]
        )
    assert not failed_path.exists()
