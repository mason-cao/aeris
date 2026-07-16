"""B3/D4 conservative variable-pruning screen."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval.pruning_screen import (
    CandidateStatistics,
    PruningThresholds,
    build_manifest,
    decide_pruning,
    main,
    render_markdown,
    write_manifest,
)
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256


START = datetime(2026, 6, 1, tzinfo=UTC)
THRESHOLDS = PruningThresholds(bootstrap_resamples=199)


def _candidate(
    name: str,
    values: list[float | None],
    *,
    mechanism_relevant: bool = False,
) -> dict[str, object]:
    return {
        "name": name,
        "source": "noaa_gfs",
        "metric": name,
        "unit": "m" if name == "gh_500" else "mm",
        "physical_mechanism_relevant": mechanism_relevant,
        "physical_mechanism_rationale": (
            "Known atmospheric mechanism retained."
            if mechanism_relevant
            else "No direct mechanism identified in the pre-screen review."
        ),
        "values": values,
    }


def _payload(
    *,
    outcomes: list[float | None] | None = None,
    candidates: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    y = outcomes or [1.0, 4.0, 2.0, 8.0, 5.0, 7.0, 3.0, 6.0]
    anchors = [
        {
            "anchor_id": f"anchor-{index}",
            "timestamp": (START + timedelta(hours=index)).isoformat(),
            "outcome_value": value,
        }
        for index, value in enumerate(y)
    ]
    return {
        "schema_version": 1,
        "input_kind": "synthetic",
        "snapshot_sha256": LOCKED_SNAPSHOT_SHA256,
        "study_window": {
            "start": "2026-06-01T00:00:00Z",
            "end_exclusive": "2026-07-13T00:00:00Z",
        },
        "anchor_population": "Synthetic hourly anchors for unit testing only.",
        "outcome_definition": "Synthetic concentration-event behavior.",
        "anchors": anchors,
        "candidates": candidates
        or [
            _candidate("gh_500", list(range(1, 9))),
            _candidate("precipitable_water", list(range(8, 0, -1))),
        ],
    }


def _stats(
    *,
    rho: float = 0.049,
    p_value: float = 0.20,
    ci_low: float = -0.1,
    ci_high: float = 0.0,
    evaluable: bool = True,
    reason: str | None = None,
) -> CandidateStatistics:
    return CandidateStatistics(
        eligible_pair_count=20,
        rho=rho,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        evaluable=evaluable,
        unevaluable_reason=reason,
    )


def test_decision_drops_only_when_every_conservative_condition_holds() -> None:
    decision = decide_pruning(
        _stats(),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )

    assert decision.decision == "drop"
    assert decision.nonsignificant is True
    assert decision.negligible is True
    assert decision.ci_covers_zero is True
    assert decision.no_physical_mechanism is True


def test_exact_decision_boundaries_are_keep_conservative() -> None:
    at_p = decide_pruning(
        _stats(p_value=0.20),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )
    below_p = decide_pruning(
        _stats(p_value=0.199999),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )
    at_rho = decide_pruning(
        _stats(rho=0.05),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )
    endpoint_zero = decide_pruning(
        _stats(ci_low=0.0, ci_high=0.1),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )
    excludes_zero = decide_pruning(
        _stats(ci_low=0.001, ci_high=0.1),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )

    assert at_p.decision == "drop"
    assert below_p.decision == "keep"
    assert at_rho.decision == "keep"
    assert endpoint_zero.decision == "drop"
    assert excludes_zero.decision == "keep"


def test_mechanism_and_unevaluable_statistics_always_keep() -> None:
    mechanism = decide_pruning(
        _stats(),
        physical_mechanism_relevant=True,
        thresholds=THRESHOLDS,
    )
    unevaluable = decide_pruning(
        _stats(evaluable=False, reason="eligible n < 4"),
        physical_mechanism_relevant=False,
        thresholds=THRESHOLDS,
    )

    assert mechanism.decision == "keep"
    assert mechanism.reason == "declared physical mechanism relevance"
    assert unevaluable.decision == "keep"
    assert unevaluable.reason == "unevaluable: eligible n < 4"


def test_planted_monotonic_positive_and_negative_rho_are_recovered() -> None:
    outcomes = [float(value) for value in range(1, 9)]
    manifest = build_manifest(
        _payload(
            outcomes=outcomes,
            candidates=[
                _candidate("gh_500", outcomes),
                _candidate("precipitable_water", list(reversed(outcomes))),
            ],
        ),
        thresholds=THRESHOLDS,
    )
    rows = {row["name"]: row for row in manifest["variables"]}

    assert rows["gh_500"]["rho"] == pytest.approx(1.0)
    assert rows["precipitable_water"]["rho"] == pytest.approx(-1.0)
    assert rows["gh_500"]["decision"] == "keep"
    assert rows["precipitable_water"]["decision"] == "keep"


def test_spearman_uses_average_ranks_for_ties() -> None:
    manifest = build_manifest(
        _payload(
            outcomes=[1.0, 2.0, 2.0, 3.0],
            candidates=[
                _candidate("gh_500", [1.0, 1.0, 2.0, 3.0]),
                _candidate("precipitable_water", [3.0, 2.0, 1.0, 1.0]),
            ],
        ),
        thresholds=THRESHOLDS,
    )

    assert manifest["variables"][0]["rho"] == pytest.approx(5 / 6)


def test_missing_pairs_are_excluded_and_counted() -> None:
    outcomes = [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    manifest = build_manifest(
        _payload(
            outcomes=outcomes,
            candidates=[
                _candidate("gh_500", [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0]),
                _candidate(
                    "precipitable_water",
                    [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
                ),
            ],
        ),
        thresholds=THRESHOLDS,
    )
    rows = {row["name"]: row for row in manifest["variables"]}

    assert rows["gh_500"]["input_pair_count"] == 8
    assert rows["gh_500"]["eligible_pair_count"] == 6
    assert rows["gh_500"]["missing_pair_count"] == 2
    assert rows["precipitable_water"]["eligible_pair_count"] == 7
    assert rows["precipitable_water"]["missing_pair_count"] == 1


@pytest.mark.parametrize(
    ("values", "reason"),
    [
        ([None] * 8, "eligible n < 4"),
        ([1.0] * 8, "constant variable series"),
    ],
)
def test_all_missing_and_constant_candidates_keep(
    values: list[float | None], reason: str
) -> None:
    manifest = build_manifest(
        _payload(
            candidates=[
                _candidate("gh_500", values),
                _candidate("precipitable_water", list(range(1, 9))),
            ]
        ),
        thresholds=THRESHOLDS,
    )
    row = manifest["variables"][0]

    assert row["decision"] == "keep"
    assert row["reason"] == f"unevaluable: {reason}"


def test_candidate_seed_and_output_are_independent_of_input_order() -> None:
    first_payload = _payload()
    second_payload = copy.deepcopy(first_payload)
    second_payload["candidates"] = list(reversed(second_payload["candidates"]))

    first = build_manifest(first_payload, thresholds=THRESHOLDS)
    second = build_manifest(second_payload, thresholds=THRESHOLDS)

    assert first == second
    assert [row["name"] for row in first["variables"]] == [
        "gh_500",
        "precipitable_water",
    ]
    assert first["variables"][0]["bootstrap_seed"] != first["variables"][1][
        "bootstrap_seed"
    ]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload.update(snapshot_sha256="0" * 64),
            "canonical snapshot",
        ),
        (
            lambda payload: payload["study_window"].update(
                end_exclusive="2026-07-12T00:00:00Z"
            ),
            "exact declared study window",
        ),
        (
            lambda payload: payload["candidates"].pop(),
            "missing required candidates: precipitable_water",
        ),
        (
            lambda payload: payload["candidates"].append(
                copy.deepcopy(payload["candidates"][0])
            ),
            "duplicate candidate name: gh_500",
        ),
        (
            lambda payload: payload["anchors"].append(
                copy.deepcopy(payload["anchors"][0])
            ),
            "duplicate anchor ID: anchor-0",
        ),
        (
            lambda payload: payload["candidates"][0].update(values=[1.0]),
            "one value per anchor",
        ),
        (
            lambda payload: payload["candidates"][0]["values"].__setitem__(
                0, float("nan")
            ),
            "finite numeric or null",
        ),
    ],
)
def test_manifest_protocol_validation_fails_loudly(
    mutate: Callable[[dict[str, object]], None], message: str
) -> None:
    payload = _payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        build_manifest(payload, thresholds=THRESHOLDS)


def test_real_screen_input_is_blocked_pending_ratification() -> None:
    payload = _payload()
    payload["input_kind"] = "real"

    with pytest.raises(ValueError, match="real pruning screen is blocked"):
        build_manifest(payload, thresholds=THRESHOLDS)


def test_invalid_timestamp_and_blank_mechanism_rationale_fail() -> None:
    timestamp = _payload()
    timestamp["anchors"][0]["timestamp"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="invalid anchor timestamp"):
        build_manifest(timestamp, thresholds=THRESHOLDS)

    rationale = _payload()
    rationale["candidates"][0]["physical_mechanism_rationale"] = ""
    with pytest.raises(ValueError, match="nonempty physical-mechanism rationale"):
        build_manifest(rationale, thresholds=THRESHOLDS)


def test_json_and_markdown_are_byte_deterministic(tmp_path: Path) -> None:
    manifest = build_manifest(_payload(), thresholds=THRESHOLDS)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_manifest(manifest, first)
    write_manifest(manifest, second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == manifest
    assert render_markdown(manifest) == render_markdown(manifest)
    assert "| gh_500 | noaa_gfs | gh_500 |" in render_markdown(manifest)
    assert manifest["threshold_status"] == "declared — pending Mason ratification"


def test_cli_validation_failure_leaves_output_absent(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "output.json"
    payload = _payload()
    payload["snapshot_sha256"] = "0" * 64
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )

    assert not output_path.exists()
