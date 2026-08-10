"""B16/P7 pre-registered clustered phase-analysis machinery."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from app.eval.phase_analysis import (
    AnalysisThresholds,
    OverlapPair,
    PowerSimulationConfig,
    aggregate_expert_validity,
    aggregate_machine_metrics,
    analyze_agreement,
    analyze_negative_controls,
    analyze_sign_mapped_kappa,
    analyze_spearman,
    build_analysis_manifest,
    build_decision_overlap,
    cluster_bootstrap_interval,
    cohen_kappa,
    holm_bonferroni,
    holm_significant,
    main,
    naive_bootstrap_interval,
    parse_failure_accounting,
    power_kappa_bootstrap_interval,
    power_spearman_bootstrap_interval,
    run_power_simulation,
    sign_mapped_verdict,
    spawn_substream_seeds,
    wilcoxon_signed_rank,
    write_manifest,
)


FAST = AnalysisThresholds(
    bootstrap_resamples=199,
    wilcoxon_monte_carlo_resamples=999,
)


def _claim(
    claim_id: str,
    anomaly_id: str,
    model: str,
    text: str,
    *,
    claim_type: str = "concentration_elevation",
    grounding_verdict: str = "grounded",
    score: float | None = 0.5,
    evidence_n: int = 2,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "anomaly_id": anomaly_id,
        "model": model,
        "claim_text": text,
        "claim_type": claim_type,
        "grounding_verdict": grounding_verdict,
        "corroboration_score": score,
        "evidence_n": evidence_n,
    }


def _label(labeler: str, claim_id: str, verdict: str | None) -> dict[str, object]:
    return {"labeler": labeler, "claim_id": claim_id, "verdict": verdict}


def test_seed_substreams_are_sorted_unique_and_order_independent() -> None:
    first = spawn_substream_seeds(["zeta", "alpha", "middle"])
    second = spawn_substream_seeds(["middle", "zeta", "alpha"])

    assert first == second
    assert list(first) == ["alpha", "middle", "zeta"]
    assert len(set(first.values())) == 3


def test_decision_overlap_deduplicates_fanout_and_ignores_rehearsal_labels() -> None:
    claims = [
        _claim("c1", "a1", "m1", "same text"),
        _claim("c2", "a1", "m2", "same text"),
        _claim("c3", "a1", "m3", "same text "),
    ]
    labels = [
        _label("mason", "c1", "valid"),
        _label("mason", "c2", "valid"),
        _label("bracco", "c1", "invalid"),
        _label("bracco", "c2", "invalid"),
        _label("bracco", "c3", "unsure"),
        _label("rehearsal-mason", "c1", "unsure"),
    ]

    overlap = build_decision_overlap(claims, labels)

    assert len(overlap.pairs) == 1
    assert overlap.pairs[0].claim_text == "same text"
    assert overlap.pairs[0].mason_verdict == "valid"
    assert overlap.pairs[0].bracco_verdict == "invalid"
    assert overlap.missing_by_labeler == {"bracco": 0, "mason": 1}
    assert overlap.excluded_missing_either == 1


def test_conflicting_fanout_verdicts_fail_loudly() -> None:
    claims = [
        _claim("c1", "a1", "m1", "same text"),
        _claim("c2", "a1", "m2", "same text"),
    ]
    labels = [
        _label("mason", "c1", "valid"),
        _label("mason", "c2", "invalid"),
    ]

    with pytest.raises(ValueError, match="conflicting fanned-out verdicts"):
        build_decision_overlap(claims, labels)


def test_planted_kappa_is_recovered_with_marginals() -> None:
    rng = np.random.default_rng(20260716)
    categories = np.asarray(["valid", "invalid", "unsure"])
    probabilities = np.asarray([0.6, 0.25, 0.15])
    first = rng.choice(categories, size=2_000, p=probabilities)
    keep = rng.random(2_000) < 0.7
    independent = rng.choice(categories, size=2_000, p=probabilities)
    second = np.where(keep, first, independent)

    result = cohen_kappa(first.tolist(), second.tolist())

    assert result.kappa == pytest.approx(0.7, abs=0.05)
    assert result.observed_agreement is not None
    assert sum(result.first_marginals.values()) == pytest.approx(1.0)
    assert sum(result.second_marginals.values()) == pytest.approx(1.0)
    assert result.pair_count == 2_000


def test_zero_overlap_is_a_hard_error_and_constant_marginal_is_undefined() -> None:
    with pytest.raises(ValueError, match="zero overlap pairs"):
        cohen_kappa([], [])

    result = cohen_kappa(["valid"] * 5, ["valid"] * 5)
    assert result.kappa is None
    assert result.undefined_reason == "constant marginal distribution"


def test_agreement_primary_keeps_unsure_and_sensitivity_excludes_it() -> None:
    claims: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    verdicts = [
        ("valid", "valid"),
        ("invalid", "invalid"),
        ("unsure", "unsure"),
        ("valid", "invalid"),
        ("unsure", "valid"),
    ]
    for index, (mason, bracco) in enumerate(verdicts):
        claim_id = f"c{index}"
        claims.append(_claim(claim_id, f"a{index}", "m1", f"text {index}"))
        labels.extend(
            [_label("mason", claim_id, mason), _label("bracco", claim_id, bracco)]
        )

    result = analyze_agreement(
        build_decision_overlap(claims, labels),
        bootstrap_resamples=FAST.bootstrap_resamples,
        seeds=spawn_substream_seeds(
            ["agreement:primary", "agreement:unsure_excluded"]
        ),
    )

    assert result["primary"]["pair_count"] == 5
    assert result["unsure_excluded"]["pair_count"] == 3
    assert result["primary"]["categories"] == ["valid", "invalid", "unsure"]
    assert result["unsure_excluded"]["categories"] == ["valid", "invalid"]


def test_cluster_bootstrap_refuses_one_cluster() -> None:
    records = [{"anomaly_id": "a1", "value": value} for value in (0.0, 1.0)]
    result = cluster_bootstrap_interval(
        records,
        cluster_key=lambda row: str(row["anomaly_id"]),
        statistic=lambda rows: float(np.mean([row["value"] for row in rows])),
        n_resamples=99,
        seed=7,
    )

    assert result.point_estimate == pytest.approx(0.5)
    assert result.ci_available is False
    assert result.refusal_reason == "cluster bootstrap requires at least 2 anomalies"


def test_more_than_five_percent_undefined_replicates_suppresses_ci() -> None:
    records = [
        {"anomaly_id": "a1", "value": 0.0},
        {"anomaly_id": "a2", "value": 1.0},
    ]

    def both_clusters_required(rows: list[dict[str, object]]) -> float | None:
        if len({row["anomaly_id"] for row in rows}) < 2:
            return None
        return float(np.mean([row["value"] for row in rows]))

    result = cluster_bootstrap_interval(
        records,
        cluster_key=lambda row: str(row["anomaly_id"]),
        statistic=both_clusters_required,
        n_resamples=199,
        seed=8,
    )

    assert result.undefined_fraction > 0.05
    assert result.ci_available is False
    assert result.refusal_reason == "more than 5% bootstrap replicates undefined"


def test_power_kappa_bootstrap_matches_general_cluster_bootstrap() -> None:
    pairs = [
        OverlapPair("a1", "c1", "valid", "valid"),
        OverlapPair("a1", "c2", "valid", "invalid"),
        OverlapPair("a1", "c3", "unsure", "unsure"),
        OverlapPair("a2", "c4", "invalid", "invalid"),
        OverlapPair("a2", "c5", "invalid", "valid"),
        OverlapPair("a3", "c6", "valid", "valid"),
        OverlapPair("a3", "c7", "valid", "valid"),
    ]

    def kappa_statistic(sample: list[OverlapPair]) -> float | None:
        return cohen_kappa(
            [pair.mason_verdict for pair in sample],
            [pair.bracco_verdict for pair in sample],
        ).kappa

    reference = cluster_bootstrap_interval(
        pairs,
        cluster_key=lambda pair: pair.anomaly_id,
        statistic=kappa_statistic,
        n_resamples=999,
        seed=84,
    )
    optimized = power_kappa_bootstrap_interval(
        pairs,
        n_resamples=999,
        seed=84,
    )

    assert optimized.point_estimate == pytest.approx(reference.point_estimate)
    assert optimized.ci_low == pytest.approx(reference.ci_low)
    assert optimized.ci_high == pytest.approx(reference.ci_high)
    assert optimized.defined_replicates == reference.defined_replicates
    assert optimized.undefined_replicates == reference.undefined_replicates
    assert optimized.ci_available is reference.ci_available


def test_power_spearman_bootstrap_matches_general_with_ties_and_unsure() -> None:
    records = [
        {"anomaly_id": "a1", "score": -0.5, "verdict": "invalid"},
        {"anomaly_id": "a1", "score": 0.0, "verdict": "unsure"},
        {"anomaly_id": "a1", "score": 0.5, "verdict": "valid"},
        {"anomaly_id": "a2", "score": -0.5, "verdict": "invalid"},
        {"anomaly_id": "a2", "score": 0.2, "verdict": "valid"},
        {"anomaly_id": "a3", "score": 0.8, "verdict": "valid"},
    ]
    reference = analyze_spearman(
        records,
        n_resamples=999,
        seed=91,
        min_claims=1,
        min_anomalies=1,
    )["bootstrap"]
    optimized = power_spearman_bootstrap_interval(
        records,
        n_resamples=999,
        seed=91,
    )

    assert isinstance(reference, dict)
    assert optimized.point_estimate == pytest.approx(reference["point_estimate"])
    assert optimized.ci_low == pytest.approx(reference["ci_low"])
    assert optimized.ci_high == pytest.approx(reference["ci_high"])
    assert optimized.defined_replicates == reference["defined_replicates"]
    assert optimized.undefined_replicates == reference["undefined_replicates"]
    assert optimized.ci_available is reference["ci_available"]


def test_exactly_five_percent_undefined_replicates_keeps_ci() -> None:
    records = [
        {"anomaly_id": f"a{index}", "value": float(index)} for index in range(3)
    ]

    def at_least_two_clusters(rows: list[dict[str, object]]) -> float | None:
        if len({row["anomaly_id"] for row in rows}) < 2:
            return None
        return float(np.mean([row["value"] for row in rows]))

    result = cluster_bootstrap_interval(
        records,
        cluster_key=lambda row: str(row["anomaly_id"]),
        statistic=at_least_two_clusters,
        n_resamples=20,
        seed=1,
    )

    assert result.undefined_fraction == pytest.approx(0.05)
    assert result.ci_available is True


def test_planted_clustering_mean_ci_is_wider_than_naive_over_25_replicates() -> None:
    cluster_widths: list[float] = []
    naive_widths: list[float] = []
    for replicate in range(25):
        rng = np.random.default_rng(10_000 + replicate)
        cluster_effects = rng.normal(size=40)
        records: list[dict[str, object]] = []
        for anomaly_index, effect in enumerate(cluster_effects):
            probability = 1.0 / (1.0 + math.exp(-1.3 * effect))
            for value in rng.binomial(1, probability, size=12):
                records.append(
                    {
                        "anomaly_id": f"a{anomaly_index}",
                        "value": float(value),
                    }
                )
        def statistic(rows: list[dict[str, object]]) -> float:
            return float(np.mean([float(row["value"]) for row in rows]))

        clustered = cluster_bootstrap_interval(
            records,
            cluster_key=lambda row: str(row["anomaly_id"]),
            statistic=statistic,
            n_resamples=10_000,
            seed=20_000 + replicate,
        )
        naive = naive_bootstrap_interval(
            records,
            statistic=statistic,
            n_resamples=10_000,
            seed=20_000 + replicate,
        )
        assert clustered.ci_low is not None and clustered.ci_high is not None
        assert naive.ci_low is not None and naive.ci_high is not None
        cluster_widths.append(clustered.ci_high - clustered.ci_low)
        naive_widths.append(naive.ci_high - naive.ci_low)

    assert np.mean(cluster_widths) > np.mean(naive_widths)


def test_expert_validity_uses_claim_units_and_declared_unsure_sensitivity() -> None:
    claims = [
        _claim("c1", "a1", "m1", "duplicate"),
        _claim("c2", "a1", "m1", "duplicate"),
        _claim("c3", "a1", "m1", "unsure"),
        _claim(
            "c4",
            "a1",
            "m1",
            "qualitative",
            claim_type="chemistry",
        ),
        _claim("c5", "a1", "m2", "only unsure"),
    ]
    labels = [
        _label("mason", "c1", "valid"),
        _label("mason", "c2", "invalid"),
        _label("mason", "c3", "unsure"),
        _label("mason", "c4", "valid"),
        _label("mason", "c5", "unsure"),
    ]

    rows = aggregate_expert_validity(claims, labels, labeler="mason")
    by_model = {row["model"]: row for row in rows}

    assert by_model["m1"]["n_valid"] == 1
    assert by_model["m1"]["n_invalid"] == 1
    assert by_model["m1"]["n_unsure"] == 1
    assert by_model["m1"]["validity_rate"] == pytest.approx(0.5)
    assert by_model["m1"]["unsure_as_invalid_rate"] == pytest.approx(1 / 3)
    assert by_model["m2"]["validity_rate"] is None
    assert by_model["m2"]["unsure_as_invalid_rate"] == pytest.approx(0.0)


def test_missing_mason_label_is_not_replaced_by_bracco() -> None:
    claims = [_claim("c1", "a1", "m1", "one")]
    labels = [_label("bracco", "c1", "valid")]

    row = aggregate_expert_validity(claims, labels, labeler="mason")[0]

    assert row["n_missing"] == 1
    assert row["validity_rate"] is None


def test_machine_aggregation_includes_empty_cells_and_evidence_distribution() -> None:
    claims = [
        _claim("c1", "a1", "m1", "one", score=0.5, evidence_n=2),
        _claim(
            "c2",
            "a1",
            "m1",
            "two",
            grounding_verdict="unverified",
            score=None,
            evidence_n=0,
        ),
        _claim(
            "c3",
            "a1",
            "m1",
            "qualitative",
            claim_type="point_source_attribution",
        ),
    ]

    rows = aggregate_machine_metrics(claims, ["a1", "a2"], ["m1"])

    assert rows[0]["grounded_rate"] == pytest.approx(0.5)
    assert rows[0]["mean_corroboration_score"] == pytest.approx(0.5)
    assert rows[0]["evidence_n_distribution"] == {"0": 1, "2": 1}
    assert rows[1]["anomaly_id"] == "a2"
    assert rows[1]["grounded_rate"] is None


def test_wilcoxon_exact_enumeration_matches_scipy_on_tie_free_case() -> None:
    first = [1.0, 2.0, 3.0, 4.0, 5.0]
    second = [0.0] * 5
    ours = wilcoxon_signed_rank(first, second, thresholds=FAST, seed=10)
    expected = stats.wilcoxon(
        first,
        second,
        alternative="two-sided",
        zero_method="pratt",
        method="exact",
    )

    assert ours.p_value == pytest.approx(float(expected.pvalue))
    assert ours.method == "complete sign enumeration"
    assert ours.randomization_assignment_count == 32
    assert ours.monte_carlo_plus_one_correction is False


def test_wilcoxon_declared_degenerate_and_monte_carlo_behavior() -> None:
    all_zero = wilcoxon_signed_rank([1.0] * 6, [1.0] * 6, thresholds=FAST, seed=1)
    too_small = wilcoxon_signed_rank(
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        thresholds=FAST,
        seed=1,
    )
    first = list(range(1, 22))
    second = [0.0] * 21
    monte_carlo_a = wilcoxon_signed_rank(first, second, thresholds=FAST, seed=2)
    monte_carlo_b = wilcoxon_signed_rank(first, second, thresholds=FAST, seed=2)

    assert all_zero.p_value == 1.0
    assert all_zero.reason == "all paired differences are zero"
    assert too_small.p_value is None
    assert too_small.reason == "fewer than 5 complete pairs; descriptive only"
    assert monte_carlo_a == monte_carlo_b
    assert monte_carlo_a.method == "deterministic Monte Carlo sign flips"
    assert monte_carlo_a.randomization_assignment_count == 999
    assert monte_carlo_a.monte_carlo_plus_one_correction is True


def test_wilcoxon_pratt_ranks_include_mixed_zeros_and_average_ties() -> None:
    result = wilcoxon_signed_rank(
        [0.0, 1.0, -2.0, 2.0, -3.0],
        [0.0] * 5,
        thresholds=FAST,
        seed=3,
    )

    assert result.statistic == pytest.approx(5.5)
    assert result.nonzero_difference_count == 4
    assert result.randomization_assignment_count == 16


def test_holm_bonferroni_adjusts_within_one_family() -> None:
    adjusted = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})

    assert adjusted == {
        "a": pytest.approx(0.03),
        "b": pytest.approx(0.06),
        "c": pytest.approx(0.06),
    }
    assert holm_significant(0.049999) is True
    assert holm_significant(0.05) is False


def test_kappa_wording_gate_is_inclusive_at_exactly_point_six() -> None:
    first = ["valid"] * 5 + ["invalid"] * 5
    second = ["valid"] * 4 + ["invalid", "valid"] + ["invalid"] * 4
    claims: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for index, (mason, bracco) in enumerate(zip(first, second, strict=True)):
        claim_id = f"c{index}"
        claims.append(_claim(claim_id, f"a{index}", "m1", f"text {index}"))
        labels.extend(
            [_label("mason", claim_id, mason), _label("bracco", claim_id, bracco)]
        )
    analysis = analyze_agreement(
        build_decision_overlap(claims, labels),
        bootstrap_resamples=49,
    )

    assert analysis["primary"]["kappa"] == pytest.approx(0.6)
    assert analysis["primary"]["wording"] == "expert-labeled"


def test_spearman_matches_scipy_and_exact_minimum_is_confirmatory() -> None:
    records: list[dict[str, object]] = []
    scores = [float(index % 7) for index in range(20)]
    verdicts = ["valid" if index % 3 else "invalid" for index in range(20)]
    for index, (score, verdict) in enumerate(zip(scores, verdicts, strict=True)):
        records.append(
            {
                "anomaly_id": f"a{index % 5}",
                "score": score,
                "verdict": verdict,
            }
        )
    result = analyze_spearman(records, n_resamples=199, seed=11)
    expected = stats.spearmanr(
        scores,
        [1.0 if verdict == "valid" else 0.0 for verdict in verdicts],
    )

    assert result["rho"] == pytest.approx(float(expected.statistic))
    assert result["unclustered_scipy_p_value_diagnostic"] == pytest.approx(
        float(expected.pvalue)
    )
    assert result["eligible_claim_count"] == 20
    assert result["anomaly_count"] == 5
    assert result["confirmatory"] is True


@pytest.mark.parametrize(
    ("scores", "verdicts", "reason"),
    [
        ([1.0] * 20, ["valid", "invalid"] * 10, "constant score vector"),
        (list(range(20)), ["valid"] * 20, "constant verdict vector"),
    ],
)
def test_spearman_constant_vectors_are_declared_undefined(
    scores: list[float], verdicts: list[str], reason: str
) -> None:
    records = [
        {"anomaly_id": f"a{index % 5}", "score": score, "verdict": verdict}
        for index, (score, verdict) in enumerate(zip(scores, verdicts, strict=True))
    ]

    result = analyze_spearman(records, n_resamples=99, seed=12)

    assert result["rho"] is None
    assert result["undefined_reason"] == reason


def test_parse_failure_accounting_distinguishes_recovery_final_fail_and_error() -> None:
    events = [
        {"model": "m2", "anomaly_id": "a1", "error": "bad JSON"},
        {"model": "m2", "anomaly_id": "a1", "error": "bad JSON again"},
        {"model": "m2", "anomaly_id": "a2", "error": "bad JSON"},
    ]
    result = parse_failure_accounting(
        ["a1", "a2", "a3"],
        ["m1", "m2"],
        explanation_cells=[
            ("a1", "m1"),
            ("a2", "m1"),
            ("a3", "m1"),
            ("a1", "m2"),
        ],
        parse_failure_events=events,
        error_cells=[("a3", "m2")],
    )
    m2 = result["models"][1]

    assert result["planned_cell_count"] == 6
    assert m2["parse_failure_event_count"] == 3
    assert m2["recovered_parse_event_cell_count"] == 1
    assert m2["final_parse_failed_cell_count"] == 1
    assert m2["non_parse_error_cell_count"] == 1
    assert m2["final_failed_cell_count"] == 2
    assert m2["final_failed_cell_rate"] == pytest.approx(2 / 3)
    assert m2["attempts_per_completed_cell"] == pytest.approx(3.0)
    assert "plain-JSON" in result["decoding_asymmetry_note"]


def test_small_power_simulation_is_deterministic_and_marked_synthetic() -> None:
    config = PowerSimulationConfig(
        cluster_sizes=(6, 6, 6, 6, 6),
        iccs=(0.15,),
        valid_prevalences=(0.6,),
        unsure_prevalences=(0.1,),
        true_kappas=(0.6,),
        latent_rhos=(0.6,),
        outer_replicates=5,
        inner_bootstrap_resamples=19,
        design_source="synthetic-test",
    )

    first = run_power_simulation(config)
    second = run_power_simulation(config)

    assert first == second
    assert first["design_source"] == "synthetic-test"
    assert first["official_monte_carlo"] is False
    assert len(first["kappa_grid"]) == 1
    assert len(first["spearman_grid"]) == 1
    assert isinstance(first["minimum_detectable_effects"][0]["fallback_required"], bool)


def test_power_simulation_uses_distinct_agreement_and_spearman_populations() -> None:
    config = PowerSimulationConfig(
        cluster_sizes=(6, 5),
        spearman_cluster_sizes=(2, 0),
        iccs=(0.15,),
        valid_prevalences=(0.6,),
        unsure_prevalences=(0.1,),
        true_kappas=(0.6,),
        latent_rhos=(0.6,),
        outer_replicates=2,
        inner_bootstrap_resamples=3,
        design_source="synthetic-test",
    )

    result = run_power_simulation(config)

    assert result["agreement_cluster_sizes"] == [6, 5]
    assert result["spearman_cluster_sizes"] == [2, 0]
    assert result["agreement_decision_count"] == 11
    assert result["spearman_eligible_claim_count"] == 2
    assert result["spearman_contributing_anomaly_count"] == 1


def test_power_simulation_rejects_an_all_zero_spearman_population() -> None:
    with pytest.raises(ValueError, match="spearman_cluster_sizes"):
        PowerSimulationConfig(
            cluster_sizes=(6, 5),
            spearman_cluster_sizes=(0, 0),
        )


def _manifest_payload() -> dict[str, object]:
    anomaly_ids = [f"a{index}" for index in range(5)]
    models = ["m1", "m2", "m3"]
    claims: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    explanations: list[dict[str, str]] = []
    claim_index = 0
    for anomaly_index, anomaly_id in enumerate(anomaly_ids):
        for model_index, model in enumerate(models):
            explanations.append({"anomaly_id": anomaly_id, "model": model})
            for local_index in range(2):
                claim_id = f"c{claim_index}"
                claim_index += 1
                text = f"claim {anomaly_id} {model} {local_index}"
                score = float(
                    ((anomaly_index + model_index + local_index) % 5) - 2
                ) / 2.0
                claims.append(_claim(claim_id, anomaly_id, model, text, score=score))
                verdict = "valid" if score >= 0 else "invalid"
                labels.append(_label("mason", claim_id, verdict))
                labels.append(_label("bracco", claim_id, verdict))
    return {
        "schema_version": 1,
        "fixture": {"anomaly_ids": anomaly_ids, "models": models},
        "claims": claims,
        "labels": labels,
        "explanations": explanations,
        "error_cells": [],
    }


def test_manifest_and_json_output_are_byte_deterministic(tmp_path: Path) -> None:
    payload = _manifest_payload()
    reversed_payload = copy.deepcopy(payload)
    reversed_payload["claims"] = list(reversed(reversed_payload["claims"]))
    reversed_payload["labels"] = list(reversed(reversed_payload["labels"]))
    first = build_analysis_manifest(payload, [], thresholds=FAST)
    second = build_analysis_manifest(reversed_payload, [], thresholds=FAST)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    write_manifest(first, first_path)
    write_manifest(second, second_path)

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert json.loads(first_path.read_text(encoding="utf-8")) == first
    assert "generated_at" not in first


def test_cli_validation_failure_leaves_output_absent(tmp_path: Path) -> None:
    input_path = tmp_path / "input.json"
    sidecar_path = tmp_path / "failures.jsonl"
    output_path = tmp_path / "output.json"
    payload = _manifest_payload()
    payload["fixture"]["models"] = ["m1", "m2"]
    input_path.write_text(json.dumps(payload), encoding="utf-8")
    sidecar_path.write_text("", encoding="utf-8")

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--input",
                str(input_path),
                "--parse-failures",
                str(sidecar_path),
                "--output",
                str(output_path),
            ]
        )

    assert not output_path.exists()


def test_empty_analysis_claim_population_fails_loudly() -> None:
    payload = _manifest_payload()
    payload["claims"] = []
    payload["labels"] = []

    with pytest.raises(ValueError, match="zero overlap pairs"):
        build_analysis_manifest(payload, [], thresholds=FAST)


# The statistics this analysis declares it will run. A change here is a change
# to the pre-registered analysis plan, so the list is asserted rather than
# discovered — that is what stops a document claiming a statistic the code
# does not implement.
DECLARED_ANALYSIS_IDS = (
    "agreement:primary",
    "agreement:unsure_excluded",
    "exploratory:sign_mapped_kappa",
    "negative_control:claim_length_spearman",
    "negative_control:corroboration_sign",
    "negative_control:majority_label",
    "spearman:exploratory:claim_type:concentration_elevation",
    "spearman:exploratory:evidence_n:1",
    "spearman:exploratory:evidence_n:>=2",
    "spearman:exploratory:model:m1",
    "spearman:exploratory:model:m2",
    "spearman:exploratory:model:m3",
    "spearman:primary",
    "spearman:unsure_as_invalid",
    "wilcoxon:expert_validity:m1 vs m2",
    "wilcoxon:expert_validity:m1 vs m3",
    "wilcoxon:expert_validity:m2 vs m3",
    "wilcoxon:grounded_rate:m1 vs m2",
    "wilcoxon:grounded_rate:m1 vs m3",
    "wilcoxon:grounded_rate:m2 vs m3",
    "wilcoxon:mean_corroboration:m1 vs m2",
    "wilcoxon:mean_corroboration:m1 vs m3",
    "wilcoxon:mean_corroboration:m2 vs m3",
)


def test_manifest_declares_exactly_the_expected_statistics() -> None:
    manifest = build_analysis_manifest(_manifest_payload(), [], thresholds=FAST)

    assert tuple(sorted(manifest["seed_substreams"])) == DECLARED_ANALYSIS_IDS


def test_manifest_reports_every_declared_statistic_block() -> None:
    manifest = build_analysis_manifest(_manifest_payload(), [], thresholds=FAST)

    assert set(manifest["spearman"]) == {
        "primary_headline_pooled",
        "unsure_as_invalid_sensitivity",
        "exploratory",
    }
    assert manifest["exploratory_sign_mapped_kappa"]["exploratory_only"] is True
    assert set(manifest["negative_controls"]) >= {
        "majority_label",
        "corroboration_sign",
        "accuracy_margin_over_majority",
        "claim_length_spearman",
    }


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.0, "valid"),
        (0.25, "valid"),
        (-1.0, "invalid"),
        (-0.25, "invalid"),
        (0.0, "unsure"),
        (-0.0, "unsure"),
    ],
)
def test_sign_mapped_verdict_boundaries(score: float, expected: str) -> None:
    assert sign_mapped_verdict(score) == expected


def _sign_record(
    anomaly_id: str,
    claim_id: str,
    score: float | None,
    verdict: str | None,
    text: str = "a claim",
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "anomaly_id": anomaly_id,
        "model": "m1",
        "claim_text": text,
        "claim_type": "concentration_elevation",
        "grounding_verdict": "grounded",
        "score": score,
        "evidence_n": 1,
        "verdict": verdict,
    }


def test_sign_mapped_kappa_excludes_unscored_and_unlabeled() -> None:
    records = [
        _sign_record("a0", "c0", 0.5, "valid"),
        _sign_record("a0", "c1", -0.5, "invalid"),
        _sign_record("a1", "c2", None, "valid"),
        _sign_record("a1", "c3", 0.5, None),
    ]

    result = analyze_sign_mapped_kappa(records, n_resamples=99)

    assert result["excluded_unscored"] == 1
    assert result["excluded_unlabeled"] == 1
    assert result["pair_count"] == 2
    assert result["exploratory_only"] is True


def test_sign_mapped_kappa_handles_no_usable_claims() -> None:
    result = analyze_sign_mapped_kappa(
        [_sign_record("a0", "c0", None, "valid")], n_resamples=99
    )

    assert result["pair_count"] == 0
    assert result["kappa"] is None
    assert result["undefined_reason"] == "zero scored and labeled claims"
    assert result["bootstrap"] is None


def test_margin_is_positive_when_corroboration_beats_the_trivial_baseline() -> None:
    records = [
        _sign_record("a0", "c0", 0.5, "valid"),
        _sign_record("a0", "c1", -0.5, "invalid"),
        _sign_record("a1", "c2", -0.5, "invalid"),
        _sign_record("a1", "c3", 0.5, "valid"),
    ]

    result = analyze_negative_controls(records, n_resamples=99)

    # The sign map is right on every claim; a constant predictor cannot be,
    # because the labels are evenly split.
    assert result["corroboration_sign"]["accuracy"] == pytest.approx(1.0)
    assert result["majority_label"]["accuracy"] == pytest.approx(0.5)
    assert result["accuracy_margin_over_majority"] == pytest.approx(0.5)


def test_margin_is_negative_when_the_trivial_baseline_wins() -> None:
    records = [
        _sign_record("a0", "c0", -0.5, "valid"),
        _sign_record("a0", "c1", -0.5, "valid"),
        _sign_record("a1", "c2", -0.5, "valid"),
        _sign_record("a1", "c3", 0.5, "invalid"),
    ]

    result = analyze_negative_controls(records, n_resamples=99)

    # This is the failure the control exists to expose: corroboration adds
    # nothing over always answering the modal label.
    assert result["corroboration_sign"]["accuracy"] == pytest.approx(0.0)
    assert result["majority_label"]["predicted_verdict"] == "valid"
    assert result["majority_label"]["accuracy"] == pytest.approx(0.75)
    assert result["accuracy_margin_over_majority"] == pytest.approx(-0.75)


def test_majority_label_control_breaks_ties_deterministically() -> None:
    records = [
        _sign_record("a0", "c0", 0.5, "valid"),
        _sign_record("a0", "c1", -0.5, "invalid"),
    ]

    first = analyze_negative_controls(records, n_resamples=99)
    second = analyze_negative_controls(list(reversed(records)), n_resamples=99)

    assert (
        first["majority_label"]["predicted_verdict"]
        == second["majority_label"]["predicted_verdict"]
        == "valid"
    )


def test_claim_length_control_substitutes_length_for_score() -> None:
    records = [
        _sign_record("a0", "c0", 0.5, "valid", text="x" * 10),
        _sign_record("a0", "c1", 0.5, "invalid", text="x" * 200),
        _sign_record("a1", "c2", None, "valid", text="x" * 999),
    ]

    result = analyze_negative_controls(records, n_resamples=99)
    control = result["claim_length_spearman"]

    assert control["substitution"] == (
        "claim character count replaces the corroboration score"
    )
    # The unscored claim is dropped so the control sees the same claims
    # corroboration is judged on.
    assert control["eligible_claim_count"] == 2


def test_negative_controls_require_every_declared_seed() -> None:
    with pytest.raises(ValueError, match="missing negative-control seeds"):
        analyze_negative_controls(
            [_sign_record("a0", "c0", 0.5, "valid")],
            n_resamples=99,
            seeds={"negative_control:majority_label": 1},
        )
