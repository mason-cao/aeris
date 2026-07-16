"""Pre-registered B16/P7 clustered evaluation analysis."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, NoReturn, TypeVar

import numpy as np
from scipy import stats

MASTER_SEED: Final = 20_260_716
OFFICIAL_LABELERS: Final = ("bracco", "mason")
PRIMARY_ANALYSIS_LABELER: Final = "mason"
VERDICTS: Final = ("valid", "invalid", "unsure")
HEADLINE_TYPES: Final = frozenset(
    {
        "concentration_elevation",
        "transport_direction",
        "meteorological_state",
    }
)
QUALITATIVE_ONLY_TYPES: Final = frozenset(
    {"chemistry", "point_source_attribution"}
)
DECODING_ASYMMETRY_NOTE: Final = (
    "The local model uses plain-JSON decoding while cloud models use "
    "schema-constrained decoding; parse-failure rates reflect that known "
    "asymmetry."
)

_Record = TypeVar("_Record")


@dataclass(frozen=True)
class AnalysisThresholds:
    """Pre-registered B16 inference and resampling settings."""

    master_seed: int = MASTER_SEED
    confidence_level: float = 0.95
    bootstrap_resamples: int = 10_000
    bootstrap_undefined_limit: float = 0.05
    kappa_wording_gate: float = 0.60
    alpha: float = 0.05
    wilcoxon_min_pairs: int = 5
    wilcoxon_enumeration_max_pairs: int = 20
    wilcoxon_monte_carlo_resamples: int = 100_000
    spearman_min_claims: int = 20
    spearman_min_anomalies: int = 5

    def __post_init__(self) -> None:
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if type(self.bootstrap_resamples) is not int or self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be a positive integer")
        if not 0.0 <= self.bootstrap_undefined_limit <= 1.0:
            raise ValueError(
                "bootstrap_undefined_limit must be between zero and one"
            )
        if not 0.0 <= self.kappa_wording_gate <= 1.0:
            raise ValueError("kappa_wording_gate must be between zero and one")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if type(self.wilcoxon_min_pairs) is not int or self.wilcoxon_min_pairs < 1:
            raise ValueError("wilcoxon_min_pairs must be a positive integer")
        if (
            type(self.wilcoxon_enumeration_max_pairs) is not int
            or self.wilcoxon_enumeration_max_pairs < self.wilcoxon_min_pairs
        ):
            raise ValueError(
                "wilcoxon_enumeration_max_pairs must be at least "
                "wilcoxon_min_pairs"
            )
        if (
            type(self.wilcoxon_monte_carlo_resamples) is not int
            or self.wilcoxon_monte_carlo_resamples < 1
        ):
            raise ValueError(
                "wilcoxon_monte_carlo_resamples must be a positive integer"
            )
        if type(self.spearman_min_claims) is not int or self.spearman_min_claims < 1:
            raise ValueError("spearman_min_claims must be a positive integer")
        if (
            type(self.spearman_min_anomalies) is not int
            or self.spearman_min_anomalies < 1
        ):
            raise ValueError("spearman_min_anomalies must be a positive integer")


@dataclass(frozen=True)
class PowerSimulationConfig:
    """Pre-registered hierarchical P7 Monte Carlo design."""

    cluster_sizes: tuple[int, ...]
    iccs: tuple[float, ...] = (0.0, 0.15, 0.3)
    valid_prevalences: tuple[float, ...] = (0.5, 0.6, 0.7)
    unsure_prevalences: tuple[float, ...] = (0.1, 0.2)
    true_kappas: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8)
    latent_rhos: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6)
    outer_replicates: int = 2_000
    inner_bootstrap_resamples: int = 2_000
    master_seed: int = MASTER_SEED
    design_source: str = "B19 funnel dry-run"

    def __post_init__(self) -> None:
        if not self.cluster_sizes or any(
            type(size) is not int or size < 1 for size in self.cluster_sizes
        ):
            raise ValueError("cluster_sizes must contain positive integers")
        for icc in self.iccs:
            if not 0.0 <= icc < 1.0:
                raise ValueError("every ICC must be in [0, 1)")
        for value in (*self.true_kappas, *self.latent_rhos):
            if not 0.0 <= value <= 1.0:
                raise ValueError("kappa/rho grid values must be in [0, 1]")
        for valid in self.valid_prevalences:
            if not 0.0 < valid < 1.0:
                raise ValueError("valid prevalence must be between zero and one")
        for unsure in self.unsure_prevalences:
            if not 0.0 <= unsure < 1.0:
                raise ValueError("unsure prevalence must be in [0, 1)")
            if any(valid + unsure >= 1.0 for valid in self.valid_prevalences):
                raise ValueError("valid + unsure prevalence must be below one")
        if type(self.outer_replicates) is not int or self.outer_replicates < 1:
            raise ValueError("outer_replicates must be a positive integer")
        if (
            type(self.inner_bootstrap_resamples) is not int
            or self.inner_bootstrap_resamples < 1
        ):
            raise ValueError(
                "inner_bootstrap_resamples must be a positive integer"
            )
        if type(self.master_seed) is not int or self.master_seed < 0:
            raise ValueError("master_seed must be a nonnegative integer")
        if not isinstance(self.design_source, str) or not self.design_source.strip():
            raise ValueError("design_source must be a nonempty string")


@dataclass(frozen=True)
class OverlapPair:
    """One exact-text decision labeled by both official labelers."""

    anomaly_id: str
    claim_text: str
    mason_verdict: str
    bracco_verdict: str


@dataclass(frozen=True)
class DecisionOverlap:
    """Mason–Bracco decision overlap and missingness accounting."""

    pairs: tuple[OverlapPair, ...]
    decision_unit_count: int
    missing_by_labeler: dict[str, int]
    excluded_missing_either: int
    excluded_qualitative_decision_units: int


@dataclass(frozen=True)
class KappaResult:
    """Unweighted nominal Cohen's kappa and required diagnostics."""

    pair_count: int
    categories: tuple[str, ...]
    kappa: float | None
    observed_agreement: float | None
    expected_agreement: float | None
    first_marginals: dict[str, float]
    second_marginals: dict[str, float]
    undefined_reason: str | None


@dataclass(frozen=True)
class BootstrapResult:
    """A deterministic percentile interval with degeneracy accounting."""

    point_estimate: float | None
    ci_low: float | None
    ci_high: float | None
    confidence_level: float
    requested_replicates: int
    defined_replicates: int
    undefined_replicates: int
    undefined_fraction: float
    cluster_count: int | None
    ci_available: bool
    refusal_reason: str | None


@dataclass(frozen=True)
class WilcoxonResult:
    """Two-sided paired signed-rank result under the declared exact rule."""

    pair_count: int
    dropped_pair_count: int
    nonzero_difference_count: int
    statistic: float | None
    p_value: float | None
    method: str | None
    randomization_assignment_count: int | None
    monte_carlo_plus_one_correction: bool
    alternative: str
    zero_method: str
    tie_method: str
    inferential: bool
    reason: str | None


def spawn_substream_seeds(
    analysis_ids: Sequence[str],
    *,
    master_seed: int = MASTER_SEED,
) -> dict[str, int]:
    """Spawn deterministic, order-independent substreams in sorted ID order."""
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer")
    validated: list[str] = []
    seen: set[str] = set()
    for analysis_id in analysis_ids:
        if not isinstance(analysis_id, str) or not analysis_id.strip():
            raise ValueError("analysis IDs must be nonempty strings")
        if analysis_id in seen:
            raise ValueError(f"duplicate analysis ID: {analysis_id}")
        seen.add(analysis_id)
        validated.append(analysis_id)
    ordered = sorted(validated)
    children = np.random.SeedSequence(master_seed).spawn(len(ordered))
    return {
        analysis_id: int(child.generate_state(1, dtype=np.uint64)[0])
        for analysis_id, child in zip(ordered, children, strict=True)
    }


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _finite_or_none(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be finite numeric or null")
    return result


def _normalize_claims(claims: object) -> list[dict[str, object]]:
    if not isinstance(claims, list):
        raise ValueError("claims must be an array")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for position, raw_claim in enumerate(claims, start=1):
        claim = _required_mapping(raw_claim, f"claim {position}")
        claim_id = _required_string(claim.get("claim_id"), "claim_id")
        if claim_id in seen:
            raise ValueError(f"duplicate claim ID: {claim_id}")
        seen.add(claim_id)
        grounding = _required_string(
            claim.get("grounding_verdict"),
            f"grounding_verdict for claim {claim_id}",
        )
        if grounding not in {"grounded", "unverified"}:
            raise ValueError(
                f"grounding_verdict for claim {claim_id} must be grounded or "
                "unverified"
            )
        evidence_n = claim.get("evidence_n")
        if type(evidence_n) is not int or evidence_n < 0:
            raise ValueError(f"evidence_n for claim {claim_id} must be nonnegative")
        score = _finite_or_none(
            claim.get("corroboration_score"),
            f"corroboration_score for claim {claim_id}",
        )
        if score is not None and not -1.0 <= score <= 1.0:
            raise ValueError(
                f"corroboration_score for claim {claim_id} must be in [-1, 1]"
            )
        normalized.append(
            {
                "claim_id": claim_id,
                "anomaly_id": _required_string(
                    claim.get("anomaly_id"), f"anomaly_id for claim {claim_id}"
                ),
                "model": _required_string(
                    claim.get("model"), f"model for claim {claim_id}"
                ),
                "claim_text": _required_string(
                    claim.get("claim_text"), f"claim_text for claim {claim_id}"
                ),
                "claim_type": _required_string(
                    claim.get("claim_type"), f"claim_type for claim {claim_id}"
                ),
                "grounding_verdict": grounding,
                "corroboration_score": score,
                "evidence_n": evidence_n,
            }
        )
    return sorted(
        normalized,
        key=lambda row: (
            str(row["anomaly_id"]),
            str(row["model"]),
            str(row["claim_id"]),
        ),
    )


def _normalize_labels(
    labels: object,
    claim_ids: set[str],
) -> list[dict[str, object]]:
    if not isinstance(labels, list):
        raise ValueError("labels must be an array")
    normalized: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for position, raw_label in enumerate(labels, start=1):
        label = _required_mapping(raw_label, f"label {position}")
        labeler = _required_string(label.get("labeler"), "labeler")
        claim_id = _required_string(label.get("claim_id"), "label claim_id")
        if claim_id not in claim_ids:
            raise ValueError(f"label references unknown claim ID: {claim_id}")
        key = (labeler, claim_id)
        if key in seen:
            raise ValueError(f"duplicate label record: {labeler}/{claim_id}")
        seen.add(key)
        verdict = label.get("verdict")
        if verdict is not None and verdict not in VERDICTS:
            raise ValueError(
                f"verdict for {labeler}/{claim_id} must be valid, invalid, "
                "unsure, or null"
            )
        normalized.append(
            {"labeler": labeler, "claim_id": claim_id, "verdict": verdict}
        )
    return sorted(
        normalized,
        key=lambda row: (str(row["labeler"]), str(row["claim_id"])),
    )


def _normalized_claims_and_labels(
    claims: object,
    labels: object,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized_claims = _normalize_claims(claims)
    normalized_labels = _normalize_labels(
        labels,
        {str(claim["claim_id"]) for claim in normalized_claims},
    )
    return normalized_claims, normalized_labels


def build_decision_overlap(
    claims: object,
    labels: object,
) -> DecisionOverlap:
    """Build exact-text decision units without fan-out multiplication."""
    normalized_claims, normalized_labels = _normalized_claims_and_labels(
        claims, labels
    )
    claim_by_id = {
        str(claim["claim_id"]): claim for claim in normalized_claims
    }
    label_by_key = {
        (str(label["labeler"]), str(label["claim_id"])): label["verdict"]
        for label in normalized_labels
        if label["labeler"] in OFFICIAL_LABELERS
    }
    decision_claims: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for claim in claim_by_id.values():
        decision_claims[
            (str(claim["anomaly_id"]), str(claim["claim_text"]))
        ].append(claim)

    missing = {labeler: 0 for labeler in OFFICIAL_LABELERS}
    pairs: list[OverlapPair] = []
    excluded_missing = 0
    qualitative_exclusions = 0
    quantitative_decision_count = 0
    for decision_key in sorted(decision_claims):
        grouped_claims = decision_claims[decision_key]
        qualitative_flags = {
            str(claim["claim_type"]) in QUALITATIVE_ONLY_TYPES
            for claim in grouped_claims
        }
        if len(qualitative_flags) > 1:
            raise ValueError(
                "fanned-out exact claim text has conflicting quantitative/"
                f"qualitative types: {decision_key[0]}/{decision_key[1]}"
            )
        if qualitative_flags == {True}:
            qualitative_exclusions += 1
            continue
        quantitative_decision_count += 1

        verdict_by_labeler: dict[str, str | None] = {}
        for labeler in OFFICIAL_LABELERS:
            verdicts = {
                str(verdict)
                for claim in grouped_claims
                if (
                    verdict := label_by_key.get((labeler, str(claim["claim_id"])))
                )
                is not None
            }
            if len(verdicts) > 1:
                raise ValueError(
                    "conflicting fanned-out verdicts for "
                    f"{labeler}/{decision_key[0]}/{decision_key[1]}"
                )
            verdict_by_labeler[labeler] = next(iter(verdicts), None)
            if verdict_by_labeler[labeler] is None:
                missing[labeler] += 1

        mason = verdict_by_labeler["mason"]
        bracco = verdict_by_labeler["bracco"]
        if mason is None or bracco is None:
            excluded_missing += 1
            continue
        pairs.append(
            OverlapPair(
                anomaly_id=decision_key[0],
                claim_text=decision_key[1],
                mason_verdict=mason,
                bracco_verdict=bracco,
            )
        )

    return DecisionOverlap(
        pairs=tuple(pairs),
        decision_unit_count=quantitative_decision_count,
        missing_by_labeler=missing,
        excluded_missing_either=excluded_missing,
        excluded_qualitative_decision_units=qualitative_exclusions,
    )


def cohen_kappa(
    first: Sequence[str],
    second: Sequence[str],
    *,
    categories: Sequence[str] = VERDICTS,
) -> KappaResult:
    """Compute unweighted nominal Cohen's kappa without hidden weighting."""
    if len(first) != len(second):
        raise ValueError("kappa label vectors must have equal length")
    if not first:
        raise ValueError("zero overlap pairs")
    category_tuple = tuple(categories)
    if not category_tuple or len(set(category_tuple)) != len(category_tuple):
        raise ValueError("kappa categories must be unique and nonempty")
    allowed = set(category_tuple)
    if any(value not in allowed for value in (*first, *second)):
        raise ValueError("kappa vectors contain a verdict outside the categories")

    pair_count = len(first)
    first_counts = Counter(first)
    second_counts = Counter(second)
    first_marginals = {
        category: first_counts[category] / pair_count for category in category_tuple
    }
    second_marginals = {
        category: second_counts[category] / pair_count for category in category_tuple
    }
    observed = sum(
        first_value == second_value
        for first_value, second_value in zip(first, second, strict=True)
    ) / pair_count
    expected = sum(
        first_marginals[category] * second_marginals[category]
        for category in category_tuple
    )
    denominator = 1.0 - expected
    if math.isclose(denominator, 0.0, rel_tol=0.0, abs_tol=1e-15):
        return KappaResult(
            pair_count=pair_count,
            categories=category_tuple,
            kappa=None,
            observed_agreement=observed,
            expected_agreement=expected,
            first_marginals=first_marginals,
            second_marginals=second_marginals,
            undefined_reason="constant marginal distribution",
        )
    return KappaResult(
        pair_count=pair_count,
        categories=category_tuple,
        kappa=(observed - expected) / denominator,
        observed_agreement=observed,
        expected_agreement=expected,
        first_marginals=first_marginals,
        second_marginals=second_marginals,
        undefined_reason=None,
    )


def _statistic_value(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def cluster_bootstrap_interval(
    records: Sequence[_Record],
    *,
    cluster_key: Callable[[_Record], str],
    statistic: Callable[[list[_Record]], float | None],
    n_resamples: int = 10_000,
    seed: int = MASTER_SEED,
    confidence_level: float = 0.95,
    undefined_limit: float = 0.05,
) -> BootstrapResult:
    """Percentile-bootstrap a statistic by complete anomaly clusters."""
    observed = list(records)
    if not observed:
        raise ValueError("cluster bootstrap input is empty")
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if not 0.0 <= undefined_limit <= 1.0:
        raise ValueError("undefined_limit must be between zero and one")
    point = _statistic_value(statistic(observed))
    groups: dict[str, list[_Record]] = defaultdict(list)
    for record in observed:
        key = cluster_key(record)
        if not isinstance(key, str) or not key:
            raise ValueError("cluster keys must be nonempty strings")
        groups[key].append(record)
    ordered_groups = [groups[key] for key in sorted(groups)]
    cluster_count = len(ordered_groups)
    if point is None:
        return BootstrapResult(
            point_estimate=None,
            ci_low=None,
            ci_high=None,
            confidence_level=confidence_level,
            requested_replicates=n_resamples,
            defined_replicates=0,
            undefined_replicates=0,
            undefined_fraction=0.0,
            cluster_count=cluster_count,
            ci_available=False,
            refusal_reason="point statistic undefined",
        )
    if cluster_count < 2:
        return BootstrapResult(
            point_estimate=point,
            ci_low=None,
            ci_high=None,
            confidence_level=confidence_level,
            requested_replicates=n_resamples,
            defined_replicates=0,
            undefined_replicates=0,
            undefined_fraction=0.0,
            cluster_count=cluster_count,
            ci_available=False,
            refusal_reason="cluster bootstrap requires at least 2 anomalies",
        )

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    undefined = 0
    for _ in range(n_resamples):
        sampled: list[_Record] = []
        for index in rng.integers(0, cluster_count, size=cluster_count):
            sampled.extend(ordered_groups[int(index)])
        estimate = _statistic_value(statistic(sampled))
        if estimate is None:
            undefined += 1
        else:
            estimates.append(estimate)
    undefined_fraction = undefined / n_resamples
    if undefined_fraction > undefined_limit:
        return BootstrapResult(
            point_estimate=point,
            ci_low=None,
            ci_high=None,
            confidence_level=confidence_level,
            requested_replicates=n_resamples,
            defined_replicates=len(estimates),
            undefined_replicates=undefined,
            undefined_fraction=undefined_fraction,
            cluster_count=cluster_count,
            ci_available=False,
            refusal_reason="more than 5% bootstrap replicates undefined",
        )
    if not estimates:
        return BootstrapResult(
            point_estimate=point,
            ci_low=None,
            ci_high=None,
            confidence_level=confidence_level,
            requested_replicates=n_resamples,
            defined_replicates=0,
            undefined_replicates=undefined,
            undefined_fraction=undefined_fraction,
            cluster_count=cluster_count,
            ci_available=False,
            refusal_reason="no defined bootstrap replicates",
        )
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(
        np.asarray(estimates),
        [tail, 1.0 - tail],
        method="linear",
    )
    return BootstrapResult(
        point_estimate=point,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        requested_replicates=n_resamples,
        defined_replicates=len(estimates),
        undefined_replicates=undefined,
        undefined_fraction=undefined_fraction,
        cluster_count=cluster_count,
        ci_available=True,
        refusal_reason=None,
    )


def naive_bootstrap_interval(
    records: Sequence[_Record],
    *,
    statistic: Callable[[list[_Record]], float | None],
    n_resamples: int = 10_000,
    seed: int = MASTER_SEED,
    confidence_level: float = 0.95,
    undefined_limit: float = 0.05,
) -> BootstrapResult:
    """Naive row bootstrap retained only as the declared clustering diagnostic."""
    observed = list(records)
    if not observed:
        raise ValueError("naive bootstrap input is empty")
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    point = _statistic_value(statistic(observed))
    if point is None:
        return BootstrapResult(
            point_estimate=None,
            ci_low=None,
            ci_high=None,
            confidence_level=confidence_level,
            requested_replicates=n_resamples,
            defined_replicates=0,
            undefined_replicates=0,
            undefined_fraction=0.0,
            cluster_count=None,
            ci_available=False,
            refusal_reason="point statistic undefined",
        )
    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    undefined = 0
    row_count = len(observed)
    for _ in range(n_resamples):
        sample = [
            observed[int(index)]
            for index in rng.integers(0, row_count, row_count)
        ]
        estimate = _statistic_value(statistic(sample))
        if estimate is None:
            undefined += 1
        else:
            estimates.append(estimate)
    undefined_fraction = undefined / n_resamples
    if undefined_fraction > undefined_limit or not estimates:
        return BootstrapResult(
            point_estimate=point,
            ci_low=None,
            ci_high=None,
            confidence_level=confidence_level,
            requested_replicates=n_resamples,
            defined_replicates=len(estimates),
            undefined_replicates=undefined,
            undefined_fraction=undefined_fraction,
            cluster_count=None,
            ci_available=False,
            refusal_reason=(
                "more than 5% bootstrap replicates undefined"
                if undefined_fraction > undefined_limit
                else "no defined bootstrap replicates"
            ),
        )
    tail = (1.0 - confidence_level) / 2.0
    low, high = np.quantile(
        np.asarray(estimates),
        [tail, 1.0 - tail],
        method="linear",
    )
    return BootstrapResult(
        point_estimate=point,
        ci_low=float(low),
        ci_high=float(high),
        confidence_level=confidence_level,
        requested_replicates=n_resamples,
        defined_replicates=len(estimates),
        undefined_replicates=undefined,
        undefined_fraction=undefined_fraction,
        cluster_count=None,
        ci_available=True,
        refusal_reason=None,
    )


def _kappa_for_pairs(
    pairs: list[OverlapPair], categories: tuple[str, ...]
) -> float | None:
    if not pairs:
        return None
    result = cohen_kappa(
        [pair.mason_verdict for pair in pairs],
        [pair.bracco_verdict for pair in pairs],
        categories=categories,
    )
    return result.kappa


def _agreement_block(
    pairs: list[OverlapPair],
    *,
    categories: tuple[str, ...],
    bootstrap_resamples: int,
    seed: int,
) -> dict[str, object]:
    if not pairs:
        return {
            "pair_count": 0,
            "categories": list(categories),
            "kappa": None,
            "observed_agreement": None,
            "expected_agreement": None,
            "mason_marginals": {category: 0.0 for category in categories},
            "bracco_marginals": {category: 0.0 for category in categories},
            "undefined_reason": "zero decidable overlap pairs",
            "wording": None,
            "bootstrap": None,
        }
    estimate = cohen_kappa(
        [pair.mason_verdict for pair in pairs],
        [pair.bracco_verdict for pair in pairs],
        categories=categories,
    )
    interval = cluster_bootstrap_interval(
        pairs,
        cluster_key=lambda pair: pair.anomaly_id,
        statistic=lambda sample: _kappa_for_pairs(sample, categories),
        n_resamples=bootstrap_resamples,
        seed=seed,
    )
    wording = None
    if estimate.kappa is not None:
        wording = "expert-labeled" if estimate.kappa >= 0.60 else "expert-audited"
    return {
        "pair_count": estimate.pair_count,
        "categories": list(estimate.categories),
        "kappa": estimate.kappa,
        "observed_agreement": estimate.observed_agreement,
        "expected_agreement": estimate.expected_agreement,
        "mason_marginals": estimate.first_marginals,
        "bracco_marginals": estimate.second_marginals,
        "undefined_reason": estimate.undefined_reason,
        "wording": wording,
        "bootstrap": asdict(interval),
    }


def analyze_agreement(
    overlap: DecisionOverlap,
    *,
    bootstrap_resamples: int = 10_000,
    seeds: Mapping[str, int] | None = None,
) -> dict[str, object]:
    """Run primary V/I/U kappa and the D6 unsure-exclusion sensitivity."""
    if not overlap.pairs:
        raise ValueError("zero overlap pairs")
    required_ids = ("agreement:primary", "agreement:unsure_excluded")
    seed_map = (
        dict(seeds)
        if seeds is not None
        else spawn_substream_seeds(required_ids)
    )
    missing_seeds = [
        analysis_id for analysis_id in required_ids if analysis_id not in seed_map
    ]
    if missing_seeds:
        raise ValueError("missing agreement seeds: " + ", ".join(missing_seeds))
    pairs = list(overlap.pairs)
    sensitivity_pairs = [
        pair
        for pair in pairs
        if pair.mason_verdict != "unsure" and pair.bracco_verdict != "unsure"
    ]
    return {
        "decision_unit_count": overlap.decision_unit_count,
        "missing_by_labeler": overlap.missing_by_labeler,
        "excluded_missing_either": overlap.excluded_missing_either,
        "excluded_qualitative_decision_units": (
            overlap.excluded_qualitative_decision_units
        ),
        "primary": _agreement_block(
            pairs,
            categories=VERDICTS,
            bootstrap_resamples=bootstrap_resamples,
            seed=seed_map["agreement:primary"],
        ),
        "unsure_excluded": _agreement_block(
            sensitivity_pairs,
            categories=("valid", "invalid"),
            bootstrap_resamples=bootstrap_resamples,
            seed=seed_map["agreement:unsure_excluded"],
        ),
    }


def aggregate_expert_validity(
    claims: object,
    labels: object,
    *,
    labeler: str = PRIMARY_ANALYSIS_LABELER,
) -> list[dict[str, object]]:
    """Aggregate claim-unit V/I/U counts into per-anomaly model cells."""
    if labeler not in OFFICIAL_LABELERS:
        raise ValueError("expert-validity labeler must be mason or bracco")
    normalized_claims, normalized_labels = _normalized_claims_and_labels(
        claims, labels
    )
    verdict_by_claim = {
        str(label["claim_id"]): label["verdict"]
        for label in normalized_labels
        if label["labeler"] == labeler
    }
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for claim in normalized_claims:
        if claim["claim_type"] in QUALITATIVE_ONLY_TYPES:
            continue
        key = (str(claim["anomaly_id"]), str(claim["model"]))
        verdict = verdict_by_claim.get(str(claim["claim_id"]))
        grouped[key][str(verdict) if verdict is not None else "missing"] += 1
    rows: list[dict[str, object]] = []
    for anomaly_id, model in sorted(grouped):
        counts = grouped[(anomaly_id, model)]
        valid = counts["valid"]
        invalid = counts["invalid"]
        unsure = counts["unsure"]
        missing = counts["missing"]
        decided_denominator = valid + invalid
        sensitivity_denominator = decided_denominator + unsure
        rows.append(
            {
                "anomaly_id": anomaly_id,
                "model": model,
                "n_valid": valid,
                "n_invalid": invalid,
                "n_unsure": unsure,
                "n_missing": missing,
                "validity_rate": (
                    valid / decided_denominator if decided_denominator else None
                ),
                "unsure_as_invalid_rate": (
                    valid / sensitivity_denominator
                    if sensitivity_denominator
                    else None
                ),
            }
        )
    return rows


def _validated_unique_strings(values: object, field_name: str) -> list[str]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field_name} must be a nonempty array")
    validated: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = _required_string(value, field_name)
        if item in seen:
            raise ValueError(f"duplicate {field_name} value: {item}")
        seen.add(item)
        validated.append(item)
    return sorted(validated)


def aggregate_machine_metrics(
    claims: object,
    anomaly_ids: object,
    models: object,
) -> list[dict[str, object]]:
    """Build label-free machine-side cells over every planned fixture cell."""
    normalized_claims = _normalize_claims(claims)
    anomalies = _validated_unique_strings(anomaly_ids, "anomaly_ids")
    model_names = _validated_unique_strings(models, "models")
    allowed_cells = set(itertools.product(anomalies, model_names))
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for claim in normalized_claims:
        key = (str(claim["anomaly_id"]), str(claim["model"]))
        if key not in allowed_cells:
            raise ValueError(f"claim lies outside the planned fixture cell: {key}")
        if claim["claim_type"] not in QUALITATIVE_ONLY_TYPES:
            grouped[key].append(claim)
    rows: list[dict[str, object]] = []
    for anomaly_id, model in sorted(allowed_cells):
        cell_claims = grouped[(anomaly_id, model)]
        grounded = sum(
            claim["grounding_verdict"] == "grounded" for claim in cell_claims
        )
        scores = [
            float(claim["corroboration_score"])
            for claim in cell_claims
            if claim["corroboration_score"] is not None
        ]
        evidence_counts = Counter(int(claim["evidence_n"]) for claim in cell_claims)
        rows.append(
            {
                "anomaly_id": anomaly_id,
                "model": model,
                "quantitative_claim_count": len(cell_claims),
                "grounded_rate": (
                    grounded / len(cell_claims) if cell_claims else None
                ),
                "mean_corroboration_score": (
                    float(np.mean(scores)) if scores else None
                ),
                "evidence_n_distribution": {
                    str(value): evidence_counts[value]
                    for value in sorted(evidence_counts)
                },
            }
        )
    return rows


def wilcoxon_signed_rank(
    first: Sequence[float | None],
    second: Sequence[float | None],
    *,
    thresholds: AnalysisThresholds = AnalysisThresholds(),
    seed: int = MASTER_SEED,
) -> WilcoxonResult:
    """Run the declared Pratt/average-rank two-sided paired test."""
    if len(first) != len(second):
        raise ValueError("paired Wilcoxon vectors must have equal length")
    pairs: list[tuple[float, float]] = []
    for left, right in zip(first, second, strict=True):
        if left is None or right is None:
            continue
        left_value = _finite_or_none(left, "Wilcoxon value")
        right_value = _finite_or_none(right, "Wilcoxon value")
        assert left_value is not None and right_value is not None
        pairs.append((left_value, right_value))
    pair_count = len(pairs)
    dropped = len(first) - pair_count
    if pair_count == 0:
        return WilcoxonResult(
            pair_count=0,
            dropped_pair_count=dropped,
            nonzero_difference_count=0,
            statistic=None,
            p_value=None,
            method=None,
            randomization_assignment_count=None,
            monte_carlo_plus_one_correction=False,
            alternative="two-sided",
            zero_method="Pratt",
            tie_method="average ranks",
            inferential=False,
            reason="fewer than 5 complete pairs; descriptive only",
        )
    differences = np.asarray([left - right for left, right in pairs], dtype=float)
    nonzero = differences != 0.0
    nonzero_count = int(np.count_nonzero(nonzero))
    if nonzero_count == 0:
        return WilcoxonResult(
            pair_count=pair_count,
            dropped_pair_count=dropped,
            nonzero_difference_count=0,
            statistic=0.0,
            p_value=1.0,
            method="degenerate all-zero convention",
            randomization_assignment_count=None,
            monte_carlo_plus_one_correction=False,
            alternative="two-sided",
            zero_method="Pratt",
            tie_method="average ranks",
            inferential=pair_count >= thresholds.wilcoxon_min_pairs,
            reason="all paired differences are zero",
        )
    ranks = stats.rankdata(np.abs(differences), method="average")
    signed_ranks = ranks[nonzero]
    observed_positive = float(np.sum(signed_ranks[differences[nonzero] > 0.0]))
    total_rank = float(np.sum(signed_ranks))
    observed_statistic = min(observed_positive, total_rank - observed_positive)
    if pair_count < thresholds.wilcoxon_min_pairs:
        return WilcoxonResult(
            pair_count=pair_count,
            dropped_pair_count=dropped,
            nonzero_difference_count=nonzero_count,
            statistic=observed_statistic,
            p_value=None,
            method=None,
            randomization_assignment_count=None,
            monte_carlo_plus_one_correction=False,
            alternative="two-sided",
            zero_method="Pratt",
            tie_method="average ranks",
            inferential=False,
            reason="fewer than 5 complete pairs; descriptive only",
        )

    tolerance = 1e-12
    if pair_count <= thresholds.wilcoxon_enumeration_max_pairs:
        assignment_count = 1 << nonzero_count
        extreme = 0
        for assignment in range(assignment_count):
            positive = sum(
                rank
                for index, rank in enumerate(signed_ranks)
                if assignment & (1 << index)
            )
            statistic_value = min(positive, total_rank - positive)
            if statistic_value <= observed_statistic + tolerance:
                extreme += 1
        p_value = extreme / assignment_count
        method = "complete sign enumeration"
        plus_one_correction = False
    else:
        rng = np.random.default_rng(seed)
        extreme = 0
        remaining = thresholds.wilcoxon_monte_carlo_resamples
        batch_size = 10_000
        while remaining:
            current = min(batch_size, remaining)
            signs = rng.integers(0, 2, size=(current, nonzero_count), dtype=np.int8)
            positive = np.sum(signs * signed_ranks, axis=1)
            sampled_statistics = np.minimum(positive, total_rank - positive)
            extreme += int(
                np.count_nonzero(sampled_statistics <= observed_statistic + tolerance)
            )
            remaining -= current
        p_value = (extreme + 1) / (
            thresholds.wilcoxon_monte_carlo_resamples + 1
        )
        method = "deterministic Monte Carlo sign flips"
        assignment_count = thresholds.wilcoxon_monte_carlo_resamples
        plus_one_correction = True
    return WilcoxonResult(
        pair_count=pair_count,
        dropped_pair_count=dropped,
        nonzero_difference_count=nonzero_count,
        statistic=observed_statistic,
        p_value=float(p_value),
        method=method,
        randomization_assignment_count=assignment_count,
        monte_carlo_plus_one_correction=plus_one_correction,
        alternative="two-sided",
        zero_method="Pratt",
        tie_method="average ranks",
        inferential=True,
        reason=None,
    )


def holm_bonferroni(
    p_values: Mapping[str, float | None],
) -> dict[str, float | None]:
    """Adjust one pre-declared comparison family with Holm–Bonferroni."""
    finite: list[tuple[str, float]] = []
    for name, value in p_values.items():
        if value is None:
            continue
        numeric = float(value)
        if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
            raise ValueError(f"invalid p-value for {name}: {value}")
        finite.append((name, numeric))
    finite.sort(key=lambda item: (item[1], item[0]))
    count = len(finite)
    adjusted: dict[str, float | None] = {name: None for name in sorted(p_values)}
    running = 0.0
    for index, (name, value) in enumerate(finite):
        running = max(running, min(1.0, (count - index) * value))
        adjusted[name] = running
    return adjusted


def holm_significant(
    adjusted_p_value: float | None,
    *,
    alpha: float = 0.05,
) -> bool:
    """Apply the pre-registered strict Holm boundary; equality does not reject."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    if adjusted_p_value is None:
        return False
    value = float(adjusted_p_value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("adjusted p-value must be in [0, 1] or null")
    return value < alpha


def _spearman_value(records: list[dict[str, object]]) -> float | None:
    scores = np.asarray([float(record["score"]) for record in records])
    labels = np.asarray([float(record["label_value"]) for record in records])
    if np.unique(scores).size < 2 or np.unique(labels).size < 2:
        return None
    result = stats.spearmanr(scores, labels, alternative="two-sided")
    return _statistic_value(result.statistic)


def analyze_spearman(
    records: Sequence[Mapping[str, object]],
    *,
    include_unsure_as_invalid: bool = False,
    n_resamples: int = 10_000,
    seed: int = MASTER_SEED,
    min_claims: int = 20,
    min_anomalies: int = 5,
) -> dict[str, object]:
    """Analyze score–label association with anomaly-clustered uncertainty."""
    normalized: list[dict[str, object]] = []
    missing = 0
    excluded_unsure = 0
    for position, raw_record in enumerate(records, start=1):
        record = _required_mapping(raw_record, f"Spearman record {position}")
        anomaly_id = _required_string(
            record.get("anomaly_id"), "Spearman anomaly_id"
        )
        score = _finite_or_none(record.get("score"), "Spearman score")
        verdict = record.get("verdict")
        if score is None or verdict is None:
            missing += 1
            continue
        if verdict not in VERDICTS:
            raise ValueError("Spearman verdict must be valid, invalid, unsure, or null")
        if verdict == "unsure" and not include_unsure_as_invalid:
            excluded_unsure += 1
            continue
        normalized.append(
            {
                "anomaly_id": anomaly_id,
                "score": score,
                "label_value": 1.0 if verdict == "valid" else 0.0,
            }
        )
    normalized.sort(
        key=lambda row: (
            str(row["anomaly_id"]),
            float(row["score"]),
            float(row["label_value"]),
        )
    )
    claim_count = len(normalized)
    anomaly_count = len({str(record["anomaly_id"]) for record in normalized})
    confirmatory_eligibility = (
        claim_count >= min_claims and anomaly_count >= min_anomalies
    )
    base = {
        "eligible_claim_count": claim_count,
        "anomaly_count": anomaly_count,
        "missing_count": missing,
        "excluded_unsure_count": excluded_unsure,
        "unsure_coding": (
            "invalid" if include_unsure_as_invalid else "excluded"
        ),
    }
    if not normalized:
        return {
            **base,
            "rho": None,
            "unclustered_scipy_p_value_diagnostic": None,
            "confirmatory": False,
            "undefined_reason": "no eligible claims",
            "bootstrap": None,
        }
    scores = np.asarray([float(record["score"]) for record in normalized])
    labels = np.asarray([float(record["label_value"]) for record in normalized])
    if np.unique(scores).size < 2:
        return {
            **base,
            "rho": None,
            "unclustered_scipy_p_value_diagnostic": None,
            "confirmatory": False,
            "undefined_reason": "constant score vector",
            "bootstrap": None,
        }
    if np.unique(labels).size < 2:
        return {
            **base,
            "rho": None,
            "unclustered_scipy_p_value_diagnostic": None,
            "confirmatory": False,
            "undefined_reason": "constant verdict vector",
            "bootstrap": None,
        }
    result = stats.spearmanr(scores, labels, alternative="two-sided")
    interval = cluster_bootstrap_interval(
        normalized,
        cluster_key=lambda record: str(record["anomaly_id"]),
        statistic=_spearman_value,
        n_resamples=n_resamples,
        seed=seed,
    )
    return {
        **base,
        "rho": float(result.statistic),
        "unclustered_scipy_p_value_diagnostic": float(result.pvalue),
        "confirmatory": confirmatory_eligibility,
        "undefined_reason": None,
        "bootstrap": asdict(interval),
    }


def _normalized_cells(
    cells: Sequence[tuple[str, str] | Mapping[str, object]],
    field_name: str,
) -> list[tuple[str, str]]:
    normalized: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw_cell in cells:
        if isinstance(raw_cell, Mapping):
            anomaly_id = _required_string(
                raw_cell.get("anomaly_id"), f"{field_name} anomaly_id"
            )
            model = _required_string(raw_cell.get("model"), f"{field_name} model")
        elif isinstance(raw_cell, tuple) and len(raw_cell) == 2:
            anomaly_id = _required_string(raw_cell[0], f"{field_name} anomaly_id")
            model = _required_string(raw_cell[1], f"{field_name} model")
        else:
            raise ValueError(f"{field_name} cells must be pairs or objects")
        cell = (anomaly_id, model)
        if cell in seen:
            raise ValueError(f"duplicate {field_name} cell: {cell}")
        seen.add(cell)
        normalized.append(cell)
    return sorted(normalized)


def parse_failure_accounting(
    anomaly_ids: object,
    models: object,
    *,
    explanation_cells: Sequence[tuple[str, str] | Mapping[str, object]],
    parse_failure_events: Sequence[Mapping[str, object]],
    error_cells: Sequence[tuple[str, str] | Mapping[str, object]],
) -> dict[str, object]:
    """Separate persisted parse retries, final missing cells, and other errors."""
    anomalies = _validated_unique_strings(anomaly_ids, "anomaly_ids")
    model_names = _validated_unique_strings(models, "models")
    planned = set(itertools.product(anomalies, model_names))
    explained = set(_normalized_cells(explanation_cells, "explanation"))
    errors = set(_normalized_cells(error_cells, "error"))
    if not explained <= planned:
        raise ValueError("explanation cells contain an unplanned cell")
    if not errors <= planned:
        raise ValueError("error cells contain an unplanned cell")
    if explained & errors:
        raise ValueError("a cell cannot be both explained and a final error")

    events_by_cell: Counter[tuple[str, str]] = Counter()
    for position, raw_event in enumerate(parse_failure_events, start=1):
        event = _required_mapping(raw_event, f"parse-failure event {position}")
        cell = (
            _required_string(event.get("anomaly_id"), "parse event anomaly_id"),
            _required_string(event.get("model"), "parse event model"),
        )
        if cell not in planned:
            raise ValueError(f"parse-failure event contains an unplanned cell: {cell}")
        _required_string(event.get("error"), "parse event error")
        events_by_cell[cell] += 1
    final_missing = planned - explained
    event_cells = set(events_by_cell)
    ambiguous_final = (event_cells & final_missing) & errors
    if ambiguous_final:
        raise ValueError(
            "final cells cannot be classified as both parse and non-parse errors"
        )

    model_rows: list[dict[str, object]] = []
    for model in model_names:
        model_planned = {(anomaly_id, model) for anomaly_id in anomalies}
        model_explained = model_planned & explained
        model_missing = model_planned - explained
        model_event_cells = model_planned & event_cells
        recovered_event_cells = model_event_cells & explained
        final_parse_cells = (model_event_cells & model_missing) - errors
        model_errors = model_planned & errors
        unclassified = model_missing - final_parse_cells - model_errors
        recovered_event_count = sum(
            events_by_cell[cell] for cell in recovered_event_cells
        )
        attempts_per_completed = (
            (len(model_explained) + recovered_event_count) / len(model_explained)
            if model_explained
            else None
        )
        model_rows.append(
            {
                "model": model,
                "planned_cell_count": len(model_planned),
                "completed_cell_count": len(model_explained),
                "final_failed_cell_count": len(model_missing),
                "final_failed_cell_rate": len(model_missing) / len(model_planned),
                "parse_failure_event_count": sum(
                    events_by_cell[cell] for cell in model_event_cells
                ),
                "parse_failure_event_cell_count": len(model_event_cells),
                "recovered_parse_event_cell_count": len(recovered_event_cells),
                "final_parse_failed_cell_count": len(final_parse_cells),
                "non_parse_error_cell_count": len(model_errors),
                "unclassified_missing_cell_count": len(unclassified),
                "attempts_per_completed_cell": attempts_per_completed,
            }
        )
    return {
        "planned_cell_count": len(planned),
        "models": model_rows,
        "decoding_asymmetry_note": DECODING_ASYMMETRY_NOTE,
    }


def _category_from_latent(
    latent: np.ndarray,
    *,
    valid_prevalence: float,
    unsure_prevalence: float,
) -> np.ndarray:
    invalid_prevalence = 1.0 - valid_prevalence - unsure_prevalence
    invalid_cut = stats.norm.ppf(invalid_prevalence)
    unsure_cut = stats.norm.ppf(invalid_prevalence + unsure_prevalence)
    return np.where(
        latent < invalid_cut,
        "invalid",
        np.where(latent < unsure_cut, "unsure", "valid"),
    )


def _simulate_hierarchical_records(
    cluster_sizes: tuple[int, ...],
    *,
    icc: float,
    valid_prevalence: float,
    unsure_prevalence: float,
    latent_rho: float,
    rng: np.random.Generator,
) -> tuple[list[str], list[str], list[dict[str, object]]]:
    first_labels: list[str] = []
    anomaly_ids: list[str] = []
    score_records: list[dict[str, object]] = []
    for anomaly_index, cluster_size in enumerate(cluster_sizes):
        anomaly_id = f"a{anomaly_index}"
        random_intercept = float(rng.normal())
        residual = rng.normal(size=cluster_size)
        latent = math.sqrt(icc) * random_intercept + math.sqrt(1.0 - icc) * residual
        categories = _category_from_latent(
            latent,
            valid_prevalence=valid_prevalence,
            unsure_prevalence=unsure_prevalence,
        )
        scores = latent_rho * latent + math.sqrt(1.0 - latent_rho**2) * rng.normal(
            size=cluster_size
        )
        for category, score in zip(categories, scores, strict=True):
            verdict = str(category)
            first_labels.append(verdict)
            anomaly_ids.append(anomaly_id)
            score_records.append(
                {
                    "anomaly_id": anomaly_id,
                    "score": float(score),
                    "verdict": verdict,
                }
            )
    return first_labels, anomaly_ids, score_records


def _second_labeler_with_target_kappa(
    first: Sequence[str],
    *,
    true_kappa: float,
    valid_prevalence: float,
    unsure_prevalence: float,
    rng: np.random.Generator,
) -> list[str]:
    probabilities = [
        valid_prevalence,
        1.0 - valid_prevalence - unsure_prevalence,
        unsure_prevalence,
    ]
    independent = rng.choice(
        np.asarray(["valid", "invalid", "unsure"]),
        size=len(first),
        p=probabilities,
    )
    retained = rng.random(len(first)) < true_kappa
    return [
        first_value if keep else str(independent_value)
        for first_value, keep, independent_value in zip(
            first, retained, independent, strict=True
        )
    ]


def run_power_simulation(config: PowerSimulationConfig) -> dict[str, object]:
    """Run the pre-registered hierarchical power grid without official labels."""
    kappa_specs = sorted(
        itertools.product(
            config.iccs,
            config.valid_prevalences,
            config.unsure_prevalences,
            config.true_kappas,
        )
    )
    spearman_specs = sorted(
        itertools.product(
            config.iccs,
            config.valid_prevalences,
            config.unsure_prevalences,
            config.latent_rhos,
        )
    )
    analysis_ids = [
        f"power:kappa:{icc}:{valid}:{unsure}:{kappa}"
        for icc, valid, unsure, kappa in kappa_specs
    ] + [
        f"power:spearman:{icc}:{valid}:{unsure}:{rho}"
        for icc, valid, unsure, rho in spearman_specs
    ]
    seeds = spawn_substream_seeds(analysis_ids, master_seed=config.master_seed)
    kappa_rows: list[dict[str, object]] = []
    for icc, valid, unsure, true_kappa in kappa_specs:
        analysis_id = f"power:kappa:{icc}:{valid}:{unsure}:{true_kappa}"
        rng = np.random.default_rng(seeds[analysis_id])
        point_gate_count = 0
        defined_point_count = 0
        widths: list[float] = []
        available_intervals = 0
        for _ in range(config.outer_replicates):
            first, anomaly_ids, _ = _simulate_hierarchical_records(
                config.cluster_sizes,
                icc=icc,
                valid_prevalence=valid,
                unsure_prevalence=unsure,
                latent_rho=0.0,
                rng=rng,
            )
            second = _second_labeler_with_target_kappa(
                first,
                true_kappa=true_kappa,
                valid_prevalence=valid,
                unsure_prevalence=unsure,
                rng=rng,
            )
            pairs = [
                OverlapPair(
                    anomaly_id=anomaly_id,
                    claim_text=f"synthetic-{index}",
                    mason_verdict=first_value,
                    bracco_verdict=second_value,
                )
                for index, (anomaly_id, first_value, second_value) in enumerate(
                    zip(anomaly_ids, first, second, strict=True)
                )
            ]
            point = cohen_kappa(first, second)
            if point.kappa is not None:
                defined_point_count += 1
                point_gate_count += point.kappa >= 0.60
            interval = cluster_bootstrap_interval(
                pairs,
                cluster_key=lambda pair: pair.anomaly_id,
                statistic=lambda sample: _kappa_for_pairs(sample, VERDICTS),
                n_resamples=config.inner_bootstrap_resamples,
                seed=int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)),
            )
            if interval.ci_available:
                assert interval.ci_low is not None and interval.ci_high is not None
                available_intervals += 1
                widths.append(interval.ci_high - interval.ci_low)
        kappa_rows.append(
            {
                "icc": icc,
                "valid_prevalence": valid,
                "unsure_prevalence": unsure,
                "true_kappa": true_kappa,
                "outer_replicates": config.outer_replicates,
                "defined_point_replicates": defined_point_count,
                "probability_point_estimate_at_least_0_60": (
                    point_gate_count / config.outer_replicates
                ),
                "available_interval_replicates": available_intervals,
                "mean_ci_width": float(np.mean(widths)) if widths else None,
            }
        )

    spearman_rows: list[dict[str, object]] = []
    for icc, valid, unsure, rho in spearman_specs:
        analysis_id = f"power:spearman:{icc}:{valid}:{unsure}:{rho}"
        rng = np.random.default_rng(seeds[analysis_id])
        exclusions = 0
        available = 0
        for _ in range(config.outer_replicates):
            _, _, records = _simulate_hierarchical_records(
                config.cluster_sizes,
                icc=icc,
                valid_prevalence=valid,
                unsure_prevalence=unsure,
                latent_rho=rho,
                rng=rng,
            )
            analysis = analyze_spearman(
                records,
                n_resamples=config.inner_bootstrap_resamples,
                seed=int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)),
                min_claims=1,
                min_anomalies=1,
            )
            bootstrap = analysis["bootstrap"]
            if not isinstance(bootstrap, Mapping) or not bootstrap["ci_available"]:
                continue
            low = float(bootstrap["ci_low"])
            high = float(bootstrap["ci_high"])
            available += 1
            exclusions += low > 0.0 or high < 0.0
        spearman_rows.append(
            {
                "icc": icc,
                "valid_prevalence": valid,
                "unsure_prevalence": unsure,
                "latent_rho": rho,
                "outer_replicates": config.outer_replicates,
                "available_interval_replicates": available,
                "power_ci_excludes_zero": exclusions / config.outer_replicates,
            }
        )

    mde_rows: list[dict[str, object]] = []
    for icc, valid, unsure in sorted(
        itertools.product(
            config.iccs, config.valid_prevalences, config.unsure_prevalences
        )
    ):
        relevant = [
            row
            for row in spearman_rows
            if row["icc"] == icc
            and row["valid_prevalence"] == valid
            and row["unsure_prevalence"] == unsure
        ]
        reaching = [
            float(row["latent_rho"])
            for row in relevant
            if float(row["power_ci_excludes_zero"]) >= 0.80
        ]
        power_at_rho_0_6 = next(
            (
                float(row["power_ci_excludes_zero"])
                for row in relevant
                if math.isclose(float(row["latent_rho"]), 0.6)
            ),
            None,
        )
        mde_rows.append(
            {
                "icc": icc,
                "valid_prevalence": valid,
                "unsure_prevalence": unsure,
                "minimum_detectable_rho": min(reaching) if reaching else None,
                "power_at_rho_0_6": power_at_rho_0_6,
                "fallback_required": (
                    power_at_rho_0_6 is not None and power_at_rho_0_6 < 0.80
                ),
            }
        )
    return {
        "schema_version": 1,
        "design_source": config.design_source,
        "official_monte_carlo": config.design_source == "B19 funnel dry-run",
        "master_seed": config.master_seed,
        "seed_substreams": seeds,
        "cluster_sizes": list(config.cluster_sizes),
        "outer_replicates": config.outer_replicates,
        "inner_bootstrap_resamples": config.inner_bootstrap_resamples,
        "dgm": (
            "Gaussian hierarchical latent validity with anomaly random "
            "intercept; categorical thresholds preserve declared marginal "
            "prevalence; second labeler retains the first label with true-"
            "kappa probability and otherwise draws independently from the "
            "declared margins; score has declared latent Gaussian rho."
        ),
        "kappa_grid": kappa_rows,
        "spearman_grid": spearman_rows,
        "minimum_detectable_effects": mde_rows,
    }


def _comparison_family(
    cells: Sequence[Mapping[str, object]],
    *,
    field_name: str,
    family_name: str,
    anomaly_ids: Sequence[str],
    models: Sequence[str],
    thresholds: AnalysisThresholds,
    seeds: Mapping[str, int],
) -> list[dict[str, object]]:
    values = {
        (str(row["anomaly_id"]), str(row["model"])): row.get(field_name)
        for row in cells
    }
    comparisons: list[dict[str, object]] = []
    raw_p_values: dict[str, float | None] = {}
    for first_model, second_model in itertools.combinations(sorted(models), 2):
        comparison_name = f"{first_model} vs {second_model}"
        first_values = [
            values.get((anomaly_id, first_model)) for anomaly_id in anomaly_ids
        ]
        second_values = [
            values.get((anomaly_id, second_model)) for anomaly_id in anomaly_ids
        ]
        analysis_id = f"wilcoxon:{family_name}:{comparison_name}"
        result = wilcoxon_signed_rank(
            first_values,
            second_values,
            thresholds=thresholds,
            seed=seeds[analysis_id],
        )
        raw_p_values[comparison_name] = result.p_value
        comparisons.append(
            {
                "comparison": comparison_name,
                **asdict(result),
                "complete_pair_count": result.pair_count,
                "dropped_anomaly_count": result.dropped_pair_count,
            }
        )
    adjusted = holm_bonferroni(raw_p_values)
    for row in comparisons:
        row["holm_adjusted_p_value"] = adjusted[str(row["comparison"])]
        adjusted_p = row["holm_adjusted_p_value"]
        row["reject_at_alpha_0_05"] = holm_significant(
            adjusted_p,
            alpha=thresholds.alpha,
        )
    return comparisons


def _primary_label_records(
    claims: list[dict[str, object]],
    labels: list[dict[str, object]],
) -> list[dict[str, object]]:
    verdict_by_claim = {
        str(label["claim_id"]): label["verdict"]
        for label in labels
        if label["labeler"] == PRIMARY_ANALYSIS_LABELER
    }
    records: list[dict[str, object]] = []
    for claim in claims:
        if claim["claim_type"] in QUALITATIVE_ONLY_TYPES:
            continue
        records.append(
            {
                "claim_id": claim["claim_id"],
                "anomaly_id": claim["anomaly_id"],
                "model": claim["model"],
                "claim_type": claim["claim_type"],
                "grounding_verdict": claim["grounding_verdict"],
                "score": claim["corroboration_score"],
                "evidence_n": claim["evidence_n"],
                "verdict": verdict_by_claim.get(str(claim["claim_id"])),
            }
        )
    return records


def _spearman_input(
    records: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "anomaly_id": record["anomaly_id"],
            "score": record["score"],
            "verdict": record["verdict"],
        }
        for record in records
        if record["grounding_verdict"] == "grounded"
        and record["score"] is not None
    ]


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _normalize_manifest_input(
    payload: object,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    list[dict[str, object]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    root = _required_mapping(payload, "analysis input root")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must equal 1")
    fixture = _required_mapping(root.get("fixture"), "fixture")
    anomaly_ids = _validated_unique_strings(
        fixture.get("anomaly_ids"), "anomaly_ids"
    )
    models = _validated_unique_strings(fixture.get("models"), "models")
    if len(models) != 3:
        raise ValueError("B16 requires exactly 3 planned models")
    claims, labels = _normalized_claims_and_labels(
        root.get("claims"), root.get("labels")
    )
    planned = set(itertools.product(anomaly_ids, models))
    for claim in claims:
        cell = (str(claim["anomaly_id"]), str(claim["model"]))
        if cell not in planned:
            raise ValueError(f"claim lies outside the planned fixture cell: {cell}")
    raw_explanations = root.get("explanations")
    if not isinstance(raw_explanations, list):
        raise ValueError("explanations must be an array")
    explanations = _normalized_cells(raw_explanations, "explanation")
    raw_errors = root.get("error_cells")
    if not isinstance(raw_errors, list):
        raise ValueError("error_cells must be an array")
    errors = _normalized_cells(raw_errors, "error")
    normalized = {
        "schema_version": 1,
        "fixture": {"anomaly_ids": anomaly_ids, "models": models},
        "claims": claims,
        "labels": labels,
        "explanations": [
            {"anomaly_id": anomaly_id, "model": model}
            for anomaly_id, model in explanations
        ],
        "error_cells": [
            {"anomaly_id": anomaly_id, "model": model}
            for anomaly_id, model in errors
        ],
    }
    return normalized, claims, labels, explanations, errors


def build_analysis_manifest(
    payload: object,
    parse_failure_events: Sequence[Mapping[str, object]],
    *,
    thresholds: AnalysisThresholds = AnalysisThresholds(),
) -> dict[str, object]:
    """Build the deterministic pre-registered phase-analysis manifest."""
    normalized, claims, labels, explanations, error_cells = _normalize_manifest_input(
        payload
    )
    fixture = normalized["fixture"]
    assert isinstance(fixture, Mapping)
    anomaly_ids = list(fixture["anomaly_ids"])
    models = list(fixture["models"])
    assert all(isinstance(value, str) for value in anomaly_ids)
    assert all(isinstance(value, str) for value in models)

    score_records = _primary_label_records(claims, labels)
    exploratory_groups: dict[str, list[dict[str, object]]] = {}
    headline = [
        record for record in score_records if record["claim_type"] in HEADLINE_TYPES
    ]
    for model in models:
        exploratory_groups[f"model:{model}"] = [
            record for record in headline if record["model"] == model
        ]
    for claim_type in sorted(
        {str(record["claim_type"]) for record in score_records}
    ):
        exploratory_groups[f"claim_type:{claim_type}"] = [
            record for record in score_records if record["claim_type"] == claim_type
        ]
    exploratory_groups["evidence_n:1"] = [
        record for record in headline if record["evidence_n"] == 1
    ]
    exploratory_groups["evidence_n:>=2"] = [
        record for record in headline if int(record["evidence_n"]) >= 2
    ]

    analysis_ids = ["agreement:primary", "agreement:unsure_excluded"]
    for family_name in (
        "expert_validity",
        "grounded_rate",
        "mean_corroboration",
    ):
        for first_model, second_model in itertools.combinations(models, 2):
            analysis_ids.append(
                f"wilcoxon:{family_name}:{first_model} vs {second_model}"
            )
    analysis_ids.extend(["spearman:primary", "spearman:unsure_as_invalid"])
    analysis_ids.extend(
        f"spearman:exploratory:{name}" for name in sorted(exploratory_groups)
    )
    seeds = spawn_substream_seeds(
        analysis_ids,
        master_seed=thresholds.master_seed,
    )

    overlap = build_decision_overlap(claims, labels)
    agreement = analyze_agreement(
        overlap,
        bootstrap_resamples=thresholds.bootstrap_resamples,
        seeds=seeds,
    )
    expert_cells = aggregate_expert_validity(
        claims, labels, labeler=PRIMARY_ANALYSIS_LABELER
    )
    machine_cells = aggregate_machine_metrics(claims, anomaly_ids, models)
    wilcoxon_families = {
        "expert_validity": _comparison_family(
            expert_cells,
            field_name="validity_rate",
            family_name="expert_validity",
            anomaly_ids=anomaly_ids,
            models=models,
            thresholds=thresholds,
            seeds=seeds,
        ),
        "grounded_rate": _comparison_family(
            machine_cells,
            field_name="grounded_rate",
            family_name="grounded_rate",
            anomaly_ids=anomaly_ids,
            models=models,
            thresholds=thresholds,
            seeds=seeds,
        ),
        "mean_corroboration": _comparison_family(
            machine_cells,
            field_name="mean_corroboration_score",
            family_name="mean_corroboration",
            anomaly_ids=anomaly_ids,
            models=models,
            thresholds=thresholds,
            seeds=seeds,
        ),
    }
    primary_spearman = analyze_spearman(
        _spearman_input(headline),
        n_resamples=thresholds.bootstrap_resamples,
        seed=seeds["spearman:primary"],
        min_claims=thresholds.spearman_min_claims,
        min_anomalies=thresholds.spearman_min_anomalies,
    )
    unsure_sensitivity = analyze_spearman(
        _spearman_input(headline),
        include_unsure_as_invalid=True,
        n_resamples=thresholds.bootstrap_resamples,
        seed=seeds["spearman:unsure_as_invalid"],
        min_claims=thresholds.spearman_min_claims,
        min_anomalies=thresholds.spearman_min_anomalies,
    )
    exploratory_spearman: list[dict[str, object]] = []
    for name in sorted(exploratory_groups):
        result = analyze_spearman(
            _spearman_input(exploratory_groups[name]),
            n_resamples=thresholds.bootstrap_resamples,
            seed=seeds[f"spearman:exploratory:{name}"],
            min_claims=thresholds.spearman_min_claims,
            min_anomalies=thresholds.spearman_min_anomalies,
        )
        result["stratum"] = name
        result["analysis_status"] = "exploratory"
        exploratory_spearman.append(result)

    parse_accounting = parse_failure_accounting(
        anomaly_ids,
        models,
        explanation_cells=explanations,
        parse_failure_events=parse_failure_events,
        error_cells=error_cells,
    )
    normalized_events = sorted(
        [
            {
                "model": _required_string(event.get("model"), "parse event model"),
                "anomaly_id": _required_string(
                    event.get("anomaly_id"), "parse event anomaly_id"
                ),
                "error": _required_string(event.get("error"), "parse event error"),
            }
            for event in parse_failure_events
        ],
        key=lambda event: (
            event["model"],
            event["anomaly_id"],
            event["error"],
        ),
    )
    input_digest = hashlib.sha256(
        _canonical_json(
            {"analysis_input": normalized, "parse_failure_events": normalized_events}
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "pre_registration_status": "pre-registered 2026-07-16",
        "official_labelers": list(OFFICIAL_LABELERS),
        "primary_analysis_labeler": PRIMARY_ANALYSIS_LABELER,
        "confirmatory_claim_types": sorted(HEADLINE_TYPES),
        "excluded_qualitative_claim_types": sorted(QUALITATIVE_ONLY_TYPES),
        "input_sha256": input_digest,
        "thresholds": asdict(thresholds),
        "seed_substreams": seeds,
        "agreement": agreement,
        "expert_validity_cells": expert_cells,
        "machine_cells": machine_cells,
        "wilcoxon_families": wilcoxon_families,
        "spearman": {
            "primary_headline_pooled": primary_spearman,
            "unsure_as_invalid_sensitivity": unsure_sensitivity,
            "exploratory": exploratory_spearman,
        },
        "parse_failures": parse_accounting,
        "power": {
            "status": "not run",
            "reason": "B19 label-free design composition is not yet available",
        },
    }


def _write_text(path: Path, text: str) -> None:
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_manifest(manifest: Mapping[str, object], path: Path) -> None:
    """Write byte-deterministic analysis JSON atomically."""
    _write_text(
        path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read analysis input {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in analysis input {path}: {exc}") from exc


def _load_jsonl(path: Path) -> list[Mapping[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"cannot read parse-failure sidecar {path}: {exc}") from exc
    events: list[Mapping[str, object]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid JSON in parse-failure sidecar line {line_number}: {exc}"
            ) from exc
        events.append(_required_mapping(event, f"parse-failure line {line_number}"))
    return events


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pre-registered B16 clustered phase analysis."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--parse-failures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the deterministic B16 analysis CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = build_analysis_manifest(
            _load_json(args.input),
            _load_jsonl(args.parse_failures),
        )
        write_manifest(manifest, args.output)
    except (OSError, ValueError) as exc:
        _argument_error(parser, str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
