from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.subset_selection import (
    SubsetSelectionError,
    build_subset_manifest,
    main,
)


def test_round_robin_visits_every_stratum_before_deepening_any() -> None:
    manifest = build_subset_manifest(
        ["a1", "a2", "b1", "c1"],
        {"a1": ["x"], "a2": ["x"], "b1": ["x"], "c1": ["x"]},
        {"a1": "A", "a2": "A", "b1": "B", "c1": "C"},
        claim_cap=4,
    )

    assert manifest["selected_anomaly_ids"] == ["a1", "b1", "c1", "a2"]
    assert manifest["tier_one_anomaly_ids"] == ["a1", "b1", "c1"]
    assert manifest["tier_two_anomaly_ids"] == ["a2"]
    assert manifest["stratum_count"] == 3
    assert manifest["selected_stratum_count"] == 3


def test_stratum_precedence_is_first_appearance_in_rank_order() -> None:
    manifest = build_subset_manifest(
        ["b1", "a1", "b2", "a2"],
        {"b1": ["x"], "a1": ["x"], "b2": ["x"], "a2": ["x"]},
        {"b1": "B", "a1": "A", "b2": "B", "a2": "A"},
        claim_cap=4,
    )

    # B appears first in the ranked input, so B leads every cycle.
    assert manifest["selected_anomaly_ids"] == ["b1", "a1", "b2", "a2"]
    assert [row["cycle"] for row in manifest["audit"]] == [1, 1, 2, 2]


def test_cycle_is_carried_not_derived_when_strata_are_uneven() -> None:
    manifest = build_subset_manifest(
        ["a1", "a2", "a3", "b1"],
        {"a1": ["x"], "a2": ["x"], "a3": ["x"], "b1": ["x"]},
        {"a1": "A", "a2": "A", "a3": "A", "b1": "B"},
        claim_cap=4,
    )

    # Cycle 3 holds only a3 because stratum B is exhausted. A positional
    # formula would have called it cycle 2.
    assert manifest["selected_anomaly_ids"] == ["a1", "b1", "a2", "a3"]
    assert [row["cycle"] for row in manifest["audit"]] == [1, 1, 2, 3]
    assert manifest["tier_one_anomaly_ids"] == ["a1", "b1"]
    assert manifest["tier_two_anomaly_ids"] == ["a2", "a3"]


def test_exact_cap_boundary_includes_complete_anomaly() -> None:
    manifest = build_subset_manifest(
        ["a", "b"],
        {"a": ["a1", "a2"], "b": ["b1"]},
        {"a": "A", "b": "B"},
        claim_cap=3,
    )

    assert manifest["selected_anomaly_ids"] == ["a", "b"]
    assert manifest["selected_unique_claim_count"] == 3
    assert manifest["stopped_before_anomaly_id"] is None
    assert manifest["stop_reason"] == "ranked input exhausted"
    assert [row["decision"] for row in manifest["audit"]] == ["include", "include"]


def test_first_crossing_stops_traversal_without_later_substitution() -> None:
    manifest = build_subset_manifest(
        ["a", "b", "c"],
        {"a": ["a1", "a2"], "b": ["b1", "b2"], "c": ["c1"]},
        {"a": "A", "b": "B", "c": "C"},
        claim_cap=3,
    )

    # c would fit inside the remaining budget; taking it would be cherry-picking.
    assert manifest["selected_anomaly_ids"] == ["a"]
    assert manifest["selected_unique_claim_count"] == 2
    assert manifest["stopped_before_anomaly_id"] == "b"
    assert manifest["stop_reason"] == "next complete anomaly would exceed claim cap"
    assert manifest["inspected_anomaly_count"] == 2
    assert manifest["uninspected_ranked_anomaly_count"] == 1
    assert manifest["audit"][1] == {
        "traversal_position": 2,
        "cycle": 1,
        "tier": 1,
        "rank": 2,
        "stratum": "B",
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
        {"a": ["same", "same", " Same "], "b": ["same"]},
        {"a": "A", "b": "B"},
        claim_cap=3,
    )

    assert manifest["selected_anomaly_ids"] == ["a", "b"]
    assert manifest["selected_unique_claim_count"] == 3
    assert [row["raw_claim_count"] for row in manifest["audit"]] == [3, 1]
    assert [row["unique_claim_count"] for row in manifest["audit"]] == [2, 1]


def test_first_anomaly_over_cap_produces_empty_selection() -> None:
    manifest = build_subset_manifest(
        ["a", "b"],
        {"a": ["a1", "a2"], "b": ["b1"]},
        {"a": "A", "b": "B"},
        claim_cap=1,
    )

    assert manifest["selected_anomaly_ids"] == []
    assert manifest["tier_one_anomaly_ids"] == []
    assert manifest["selected_unique_claim_count"] == 0
    assert manifest["stopped_before_anomaly_id"] == "a"
    assert manifest["inspected_anomaly_count"] == 1
    assert manifest["uninspected_ranked_anomaly_count"] == 1


def test_empty_ranked_fixture_is_valid() -> None:
    manifest = build_subset_manifest([], {}, {}, claim_cap=430)

    assert manifest["available_ranked_anomaly_count"] == 0
    assert manifest["stratum_count"] == 0
    assert manifest["selected_anomaly_ids"] == []
    assert manifest["audit"] == []
    assert manifest["stop_reason"] == "ranked input exhausted"


def test_single_stratum_degrades_to_the_ranked_prefix() -> None:
    manifest = build_subset_manifest(
        ["a", "b", "c"],
        {"a": ["a1"], "b": ["b1"], "c": ["c1"]},
        {"a": "A", "b": "A", "c": "A"},
        claim_cap=2,
    )

    assert manifest["selected_anomaly_ids"] == ["a", "b"]
    assert [row["cycle"] for row in manifest["audit"]] == [1, 2, 3]
    assert manifest["tier_one_anomaly_ids"] == ["a"]


@pytest.mark.parametrize(
    ("anomaly_ids", "claims_by_anomaly", "strata_by_anomaly", "claim_cap", "message"),
    [
        (["a"], {"a": ["claim"]}, {"a": "A"}, 0, "positive integer"),
        (["a"], {"a": ["claim"]}, {"a": "A"}, True, "positive integer"),
        (["a", "a"], {"a": ["claim"]}, {"a": "A"}, 430, "duplicate anomaly ID"),
        ([""], {"": ["claim"]}, {"": "A"}, 430, "nonempty string"),
        (["a"], {}, {"a": "A"}, 430, "missing anomaly inventories: a"),
        (
            ["a"],
            {"a": ["claim"], "extra": ["claim"]},
            {"a": "A"},
            430,
            "extra anomaly inventories: extra",
        ),
        (["a"], {"a": []}, {"a": "A"}, 430, "nonempty claim array"),
        (["a"], {"a": [""]}, {"a": "A"}, 430, "nonempty string"),
        (["a"], {"a": ["claim", 1]}, {"a": "A"}, 430, "nonempty string"),
        (["a"], {"a": ["claim"]}, {}, 430, "missing anomaly strata: a"),
        (
            ["a"],
            {"a": ["claim"]},
            {"a": "A", "extra": "B"},
            430,
            "extra anomaly strata: extra",
        ),
        (["a"], {"a": ["claim"]}, {"a": ""}, 430, "stratum for a"),
        (["a"], {"a": ["claim"]}, {"a": 1}, 430, "stratum for a"),
        (["a"], {"a": ["claim"]}, [], 430, "strata_by_anomaly must be an object"),
    ],
)
def test_malformed_or_incomplete_input_is_rejected(
    anomaly_ids: object,
    claims_by_anomaly: object,
    strata_by_anomaly: object,
    claim_cap: object,
    message: str,
) -> None:
    with pytest.raises(SubsetSelectionError, match=message):
        build_subset_manifest(  # type: ignore[arg-type]
            anomaly_ids,
            claims_by_anomaly,
            strata_by_anomaly,
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
            "claims_by_anomaly": {"a": ["a1", "a1"], "b": ["b1", "b2"], "c": ["c1"]},
            "strata_by_anomaly": {"a": "A", "b": "B", "c": "C"},
        },
    )

    arguments = [
        "--fixture",
        str(fixture_path),
        "--claims",
        str(claims_path),
        "--cap",
        "2",
    ]
    assert main([*arguments, "--output", str(first_output)]) == 0
    assert main([*arguments, "--output", str(second_output)]) == 0

    assert first_output.read_bytes() == second_output.read_bytes()
    payload = json.loads(first_output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["protocol"] == {
        "claim_cap": 2,
        "cap_unit": "unique exact claim texts within each anomaly",
        "deduplication": (
            "exact text within anomaly; no normalization; "
            "cross-anomaly texts count separately"
        ),
        "rank_source": "fixture.anomaly_ids",
        "stratum_source": "inventory.strata_by_anomaly",
        "stratum_precedence": "first appearance in the ranked input",
        "traversal": (
            "stratified round-robin; one anomaly per stratum per cycle, "
            "rank order preserved within each stratum"
        ),
        "stopping_rule": (
            "complete-anomaly traversal; stop before the first anomaly "
            "that would exceed cap, with no later substitution"
        ),
        "tiers": (
            "tier 1 is cycle 1, one anomaly per stratum; tier 2 is every "
            "later cycle and is optional per labeler"
        ),
    }
    # a dedups to one claim and fits; b needs two more and crosses the cap of 2.
    assert payload["selected_anomaly_ids"] == ["a"]
    assert payload["stopped_before_anomaly_id"] == "b"


def test_cli_validation_failure_leaves_output_absent(tmp_path: Path) -> None:
    fixture_path = tmp_path / "fixture.json"
    claims_path = tmp_path / "claims.json"
    output_path = tmp_path / "selection.json"
    _write_json(fixture_path, {"anomaly_ids": ["a"]})
    _write_json(claims_path, {"claims_by_anomaly": {}, "strata_by_anomaly": {}})

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
