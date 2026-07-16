from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.subset_selection import (
    SubsetSelectionError,
    build_subset_manifest,
    main,
)


def test_exact_cap_boundary_includes_complete_anomaly() -> None:
    manifest = build_subset_manifest(
        ["a", "b"],
        {"a": ["a1", "a2"], "b": ["b1"]},
        claim_cap=3,
    )

    assert manifest["selected_anomaly_ids"] == ["a", "b"]
    assert manifest["selected_unique_claim_count"] == 3
    assert manifest["stopped_before_anomaly_id"] is None
    assert manifest["stop_reason"] == "ranked input exhausted"
    assert [row["decision"] for row in manifest["audit"]] == [
        "include",
        "include",
    ]


def test_first_crossing_stops_ranked_prefix_without_later_substitution() -> None:
    manifest = build_subset_manifest(
        ["a", "b", "c"],
        {
            "a": ["a1", "a2"],
            "b": ["b1", "b2"],
            "c": ["c1"],
        },
        claim_cap=3,
    )

    assert manifest["selected_anomaly_ids"] == ["a"]
    assert manifest["selected_unique_claim_count"] == 2
    assert manifest["stopped_before_anomaly_id"] == "b"
    assert manifest["stop_reason"] == "next complete anomaly would exceed claim cap"
    assert manifest["inspected_anomaly_count"] == 2
    assert manifest["uninspected_ranked_anomaly_count"] == 1
    assert manifest["audit"][1] == {
        "rank": 2,
        "anomaly_id": "b",
        "raw_claim_count": 2,
        "unique_claim_count": 2,
        "cumulative_unique_claims_before": 2,
        "prospective_unique_claim_count": 4,
        "decision": "stop_before",
    }


def test_exact_text_dedup_is_within_anomaly_only() -> None:
    manifest = build_subset_manifest(
        ["a", "b"],
        {
            "a": ["same", "same", " Same "],
            "b": ["same"],
        },
        claim_cap=3,
    )

    assert manifest["selected_anomaly_ids"] == ["a", "b"]
    assert manifest["selected_unique_claim_count"] == 3
    assert [row["raw_claim_count"] for row in manifest["audit"]] == [3, 1]
    assert [row["unique_claim_count"] for row in manifest["audit"]] == [2, 1]


def test_first_anomaly_over_cap_produces_empty_prefix() -> None:
    manifest = build_subset_manifest(
        ["a", "b"],
        {"a": ["a1", "a2"], "b": ["b1"]},
        claim_cap=1,
    )

    assert manifest["selected_anomaly_ids"] == []
    assert manifest["selected_unique_claim_count"] == 0
    assert manifest["stopped_before_anomaly_id"] == "a"
    assert manifest["inspected_anomaly_count"] == 1
    assert manifest["uninspected_ranked_anomaly_count"] == 1


def test_empty_ranked_fixture_is_valid() -> None:
    manifest = build_subset_manifest([], {}, claim_cap=200)

    assert manifest["available_ranked_anomaly_count"] == 0
    assert manifest["selected_anomaly_ids"] == []
    assert manifest["audit"] == []
    assert manifest["stop_reason"] == "ranked input exhausted"


@pytest.mark.parametrize(
    ("anomaly_ids", "claims_by_anomaly", "claim_cap", "message"),
    [
        (["a"], {"a": ["claim"]}, 0, "positive integer"),
        (["a"], {"a": ["claim"]}, True, "positive integer"),
        (["a", "a"], {"a": ["claim"]}, 200, "duplicate anomaly ID"),
        ([""], {"": ["claim"]}, 200, "nonempty string"),
        (["a"], {}, 200, "missing anomaly inventories: a"),
        (
            ["a"],
            {"a": ["claim"], "extra": ["claim"]},
            200,
            "extra anomaly inventories: extra",
        ),
        (["a"], {"a": []}, 200, "nonempty claim array"),
        (["a"], {"a": [""]}, 200, "nonempty string"),
        (["a"], {"a": ["claim", 1]}, 200, "nonempty string"),
    ],
)
def test_malformed_or_incomplete_input_is_rejected(
    anomaly_ids: object,
    claims_by_anomaly: object,
    claim_cap: object,
    message: str,
) -> None:
    with pytest.raises(SubsetSelectionError, match=message):
        build_subset_manifest(  # type: ignore[arg-type]
            anomaly_ids,
            claims_by_anomaly,
            claim_cap=claim_cap,
        )


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_cli_output_is_byte_identical_and_auditable(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    claims_path = tmp_path / "claims.json"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    _write_json(fixture_path, {"anomaly_ids": ["a", "b", "c"]})
    _write_json(
        claims_path,
        {
            "schema_version": 1,
            "claims_by_anomaly": {
                "a": ["a1", "a1"],
                "b": ["b1", "b2"],
                "c": ["c1"],
            },
        },
    )

    assert main(
        [
            "--fixture",
            str(fixture_path),
            "--claims",
            str(claims_path),
            "--cap",
            "2",
            "--output",
            str(first_output),
        ]
    ) == 0
    assert main(
        [
            "--fixture",
            str(fixture_path),
            "--claims",
            str(claims_path),
            "--cap",
            "2",
            "--output",
            str(second_output),
        ]
    ) == 0

    assert first_output.read_bytes() == second_output.read_bytes()
    payload = json.loads(first_output.read_text(encoding="utf-8"))
    assert payload["protocol"] == {
        "claim_cap": 2,
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
    }
    assert payload["selected_anomaly_ids"] == ["a"]
    assert payload["stopped_before_anomaly_id"] == "b"


def test_cli_validation_failure_leaves_output_absent(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    claims_path = tmp_path / "claims.json"
    output_path = tmp_path / "selection.json"
    _write_json(fixture_path, {"anomaly_ids": ["a"]})
    _write_json(claims_path, {"claims_by_anomaly": {}})

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--fixture",
                str(fixture_path),
                "--claims",
                str(claims_path),
                "--output",
                str(output_path),
            ]
        )

    assert not output_path.exists()
