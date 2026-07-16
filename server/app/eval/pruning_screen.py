"""Deterministic B3/D4 conservative variable-pruning screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from statistics import fmean, median
from typing import Any, Final, NoReturn

import numpy as np
import scipy
from scipy import stats

from app.collectors.geo import distance_km
from app.config import settings
from app.llm.corroboration import phase2_metric_owners
from app.provenance.nomination import series_is_nomination_eligible
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    STUDY_END_EXCLUSIVE,
    STUDY_START,
)
from app.provenance.purpleair_qc import purpleair_reading_is_eligible

THRESHOLD_STATUS: Final = "ratified by Mason — 2026-07-16"
REQUIRED_CANDIDATES: Final = frozenset({"gh_500", "precipitable_water"})
SYNTHETIC_INPUT_KIND: Final = "synthetic"
POLLUTANT_GRID: Final = (
    ("openaq", "ozone"),
    ("openaq", "pm10"),
    ("openaq", "pm25"),
    ("tceq", "co"),
    ("tceq", "no2"),
    ("tceq", "so2"),
)
LAGS_HOURS: Final = (0, 6, 12, 24)
D4_STATISTICAL_CAVEATS: Final = (
    "uncorrected multiplicity is keep-biased under the all-cells rule",
    "autocorrelation can inflate significance and is keep-biased here",
    "iid-bootstrap confidence intervals may be too narrow and are keep-biased here",
)
PRUNING_FIXTURE_PATH: Final = (
    Path(__file__).parent / "fixtures" / "pruning_screen.run-001.json"
)
MECHANISM_FIXTURE_PATH: Final = (
    Path(__file__).parent / "fixtures" / "pruning_mechanisms.run-001.json"
)


@dataclass(frozen=True)
class PruningThresholds:
    """Pre-declared D4 thresholds and bootstrap settings."""

    alpha: float = 0.20
    negligible_abs_rho: float = 0.05
    confidence_level: float = 0.80
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_716
    min_pairs: int = 100

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be between zero and one")
        if not 0.0 <= self.negligible_abs_rho <= 1.0:
            raise ValueError("negligible_abs_rho must be between zero and one")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        if not math.isclose(
            self.confidence_level,
            1.0 - self.alpha,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("confidence_level must equal 1 - alpha")
        if type(self.bootstrap_resamples) is not int or self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be a positive integer")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("bootstrap_seed must be a nonnegative integer")
        if type(self.min_pairs) is not int or self.min_pairs < 2:
            raise ValueError("min_pairs must be an integer of at least two")


@dataclass(frozen=True)
class CandidateStatistics:
    """Paired association statistics for one candidate variable."""

    eligible_pair_count: int
    rho: float | None
    p_value: float | None
    ci_low: float | None
    ci_high: float | None
    evaluable: bool
    unevaluable_reason: str | None


@dataclass(frozen=True)
class PruningDecision:
    """Conservative D4 conjunction and resulting keep/drop decision."""

    decision: str
    reason: str
    nonsignificant: bool | None
    negligible: bool | None
    ci_covers_zero: bool | None
    no_physical_mechanism: bool


@dataclass(frozen=True)
class HourlyObservation:
    """One finite, in-radius row available to the D4 hourly construction."""

    source: str
    metric: str
    entity_id: str
    timestamp: datetime
    value: float
    unit: str
    nomination_eligible: bool = True


@dataclass(frozen=True)
class MechanismAssessment:
    """Mason's verbatim mechanism veto for one statistically droppable metric."""

    relevant: bool
    ambiguous: bool
    assessment: str

    def __post_init__(self) -> None:
        if type(self.relevant) is not bool or type(self.ambiguous) is not bool:
            raise ValueError("mechanism relevant/ambiguous fields must be boolean")
        if not isinstance(self.assessment, str) or not self.assessment.strip():
            raise ValueError("mechanism assessment must be nonempty")


@dataclass(frozen=True)
class SnapshotPruningInput:
    """Hash-verified, finite in-radius rows and rendered inventory."""

    snapshot_sha256: str
    observations: tuple[HourlyObservation, ...]
    rendered_metrics: tuple[dict[str, str], ...]
    input_row_count: int
    finite_in_radius_row_count: int
    quality_excluded_row_count: int


def decide_pruning(
    candidate: CandidateStatistics,
    *,
    physical_mechanism_relevant: bool,
    thresholds: PruningThresholds = PruningThresholds(),
) -> PruningDecision:
    """Apply the pre-declared four-part drop conjunction."""
    no_physical_mechanism = not physical_mechanism_relevant
    if not candidate.evaluable:
        reason = candidate.unevaluable_reason or "undefined statistic"
        return PruningDecision(
            decision="keep",
            reason=f"unevaluable: {reason}",
            nonsignificant=None,
            negligible=None,
            ci_covers_zero=None,
            no_physical_mechanism=no_physical_mechanism,
        )

    if any(
        value is None
        for value in (
            candidate.rho,
            candidate.p_value,
            candidate.ci_low,
            candidate.ci_high,
        )
    ):
        return PruningDecision(
            decision="keep",
            reason="unevaluable: undefined statistic",
            nonsignificant=None,
            negligible=None,
            ci_covers_zero=None,
            no_physical_mechanism=no_physical_mechanism,
        )

    rho = float(candidate.rho)
    p_value = float(candidate.p_value)
    ci_low = float(candidate.ci_low)
    ci_high = float(candidate.ci_high)
    if not all(math.isfinite(value) for value in (rho, p_value, ci_low, ci_high)):
        return PruningDecision(
            decision="keep",
            reason="unevaluable: nonfinite statistic",
            nonsignificant=None,
            negligible=None,
            ci_covers_zero=None,
            no_physical_mechanism=no_physical_mechanism,
        )

    nonsignificant = p_value >= thresholds.alpha
    negligible = abs(rho) < thresholds.negligible_abs_rho
    ci_covers_zero = ci_low <= 0.0 <= ci_high
    should_drop = (
        nonsignificant
        and negligible
        and ci_covers_zero
        and no_physical_mechanism
    )
    if should_drop:
        reason = "all conservative D4 conditions satisfied"
    elif physical_mechanism_relevant:
        reason = "declared physical mechanism relevance"
    else:
        failed_conditions: list[str] = []
        if not nonsignificant:
            failed_conditions.append("p < alpha")
        if not negligible:
            failed_conditions.append("abs(rho) is not below bound")
        if not ci_covers_zero:
            failed_conditions.append("confidence interval excludes zero")
        reason = "keep condition: " + "; ".join(failed_conditions)

    return PruningDecision(
        decision="drop" if should_drop else "keep",
        reason=reason,
        nonsignificant=nonsignificant,
        negligible=negligible,
        ci_covers_zero=ci_covers_zero,
        no_physical_mechanism=no_physical_mechanism,
    )


def build_metric_scope(
    rendered_metrics: Sequence[Mapping[str, object]],
    *,
    scorer_owners: Mapping[tuple[str, str], Sequence[str]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Separate rendered metrics into code-owned exemptions and D4 candidates."""
    owners = scorer_owners if scorer_owners is not None else phase2_metric_owners()
    normalized_owners: dict[tuple[str, str], tuple[str, ...]] = {}
    for raw_key, raw_functions in owners.items():
        if (
            not isinstance(raw_key, tuple)
            or len(raw_key) != 2
            or not all(isinstance(item, str) and item for item in raw_key)
        ):
            raise ValueError("scorer metric owner key must be (source, metric)")
        functions = tuple(sorted(set(raw_functions)))
        if not functions or not all(
            isinstance(function, str) and function for function in functions
        ):
            raise ValueError(f"scorer metric {raw_key!r} has no code owner")
        normalized_owners[raw_key] = functions

    rendered_units: dict[tuple[str, str], str] = {}
    for position, raw in enumerate(rendered_metrics, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"rendered metric {position} must be an object")
        source = _required_nonempty_string(
            raw.get("source"), f"rendered metric {position} source"
        )
        metric = _required_nonempty_string(
            raw.get("metric"), f"rendered metric {position} metric"
        )
        unit = _required_nonempty_string(
            raw.get("unit"), f"rendered metric {position} unit"
        )
        key = (source, metric)
        prior_unit = rendered_units.get(key)
        if prior_unit is not None and prior_unit != unit:
            raise ValueError(
                f"multiple rendered units for {source}/{metric}: "
                f"{prior_unit}, {unit}"
            )
        rendered_units[key] = unit

    rendered: list[dict[str, object]] = []
    candidates: list[dict[str, object]] = []
    exempt: list[dict[str, object]] = []
    for key in sorted(rendered_units):
        source, metric = key
        row: dict[str, object] = {
            "source": source,
            "metric": metric,
            "unit": rendered_units[key],
            "renderer_owner": "app.llm.explain.render_enrichment_text",
            "owners": list(normalized_owners.get(key, ())),
        }
        rendered.append(row)
        if key in normalized_owners:
            exempt.append(dict(row))
        else:
            candidates.append(dict(row))

    scorer_inventory = [
        {
            "source": source,
            "metric": metric,
            "owners": list(normalized_owners[(source, metric)]),
            "rendered_in_snapshot": (source, metric) in rendered_units,
        }
        for source, metric in sorted(normalized_owners)
    ]
    return {
        "rendered": rendered,
        "scorer_owned": scorer_inventory,
        "exempt_rendered": exempt,
        "candidates": candidates,
    }


def _utc_hour(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        normalized = timestamp.replace(tzinfo=UTC)
    else:
        normalized = timestamp.astimezone(UTC)
    return normalized.replace(minute=0, second=0, microsecond=0)


def build_hourly_series(
    observations: Sequence[HourlyObservation],
    *,
    pollutant_keys: set[tuple[str, str]],
    candidate_keys: set[tuple[str, str]],
) -> tuple[
    dict[tuple[str, str], dict[datetime, float]],
    dict[tuple[str, str], dict[datetime, float]],
]:
    """Build entity-balanced hourly y medians and candidate x means."""
    overlap = pollutant_keys & candidate_keys
    if overlap:
        raise ValueError(f"pollutant/candidate scope overlap: {sorted(overlap)}")
    entity_values: dict[
        tuple[tuple[str, str], datetime, str], list[float]
    ] = defaultdict(list)
    selected_keys = pollutant_keys | candidate_keys
    for observation in observations:
        key = (observation.source, observation.metric)
        if key not in selected_keys:
            continue
        if key in pollutant_keys and not observation.nomination_eligible:
            continue
        if not observation.entity_id:
            raise ValueError(f"missing entity ID for {key!r}")
        numeric = float(observation.value)
        if not math.isfinite(numeric):
            raise ValueError(f"nonfinite hourly observation for {key!r}")
        entity_values[(key, _utc_hour(observation.timestamp), observation.entity_id)].append(
            numeric
        )

    across_entities: dict[
        tuple[tuple[str, str], datetime], list[float]
    ] = defaultdict(list)
    for (key, hour, _entity), values in sorted(entity_values.items()):
        across_entities[(key, hour)].append(float(median(values)))

    pollutants = {key: {} for key in sorted(pollutant_keys)}
    candidates = {key: {} for key in sorted(candidate_keys)}
    for (key, hour), values in sorted(across_entities.items()):
        if key in pollutant_keys:
            pollutants[key][hour] = float(median(values))
        elif key in candidate_keys:
            candidates[key][hour] = float(fmean(values))
    return pollutants, candidates


def deseasonalize_hourly(
    series: Mapping[datetime, float],
) -> dict[datetime, float]:
    """Subtract each series' window-wide UTC hour-of-day mean."""
    by_hour: dict[int, list[float]] = defaultdict(list)
    normalized: dict[datetime, float] = {}
    for raw_timestamp, raw_value in series.items():
        timestamp = _utc_hour(raw_timestamp)
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError("deseasonalization series contains a nonfinite value")
        if timestamp in normalized:
            raise ValueError(f"duplicate hourly timestamp: {timestamp.isoformat()}")
        normalized[timestamp] = value
        by_hour[timestamp.hour].append(value)
    means = {hour: fmean(values) for hour, values in by_hour.items()}
    return {
        timestamp: value - means[timestamp.hour]
        for timestamp, value in sorted(normalized.items())
    }


def pair_leading_series(
    pollutant: Mapping[datetime, float],
    candidate: Mapping[datetime, float],
    *,
    lag_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair candidate x(t-lag) with pollutant y(t)."""
    if lag_hours not in LAGS_HOURS:
        raise ValueError(f"undeclared candidate lead: {lag_hours}")
    x_values: list[float] = []
    y_values: list[float] = []
    lag = timedelta(hours=lag_hours)
    normalized_candidate = {
        _utc_hour(timestamp): float(value) for timestamp, value in candidate.items()
    }
    for raw_timestamp, raw_value in sorted(pollutant.items()):
        timestamp = _utc_hour(raw_timestamp)
        candidate_value = normalized_candidate.get(timestamp - lag)
        if candidate_value is None:
            continue
        pollutant_value = float(raw_value)
        if not math.isfinite(candidate_value) or not math.isfinite(pollutant_value):
            continue
        x_values.append(candidate_value)
        y_values.append(pollutant_value)
    return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_observation_timestamp(raw: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid snapshot observation timestamp: {raw!r}") from exc
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def load_snapshot_inputs(
    database_path: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> SnapshotPruningInput:
    """Load the D4 population read-only with unconditional pre/post hashing."""
    resolved = database_path.resolve()
    before_hash = _file_sha256(resolved)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != {expected_sha256}"
        )
    start = datetime.fromisoformat(STUDY_START.replace("Z", "+00:00"))
    end = datetime.fromisoformat(STUDY_END_EXCLUSIVE.replace("Z", "+00:00"))
    connection: sqlite3.Connection | None = None
    observations: list[HourlyObservation] = []
    rendered: set[tuple[str, str, str]] = set()
    input_rows = 0
    finite_in_radius_rows = 0
    quality_excluded_rows = 0
    try:
        connection = sqlite3.connect(
            f"file:{resolved}?mode=ro&immutable=1",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
        rows = connection.execute(
            """
            SELECT source, metric, source_entity_id, timestamp, value, unit, lat, lon
            FROM data_points
            WHERE timestamp >= ? AND timestamp < ?
            ORDER BY source, metric, timestamp, source_entity_id
            """,
            (
                start.strftime("%Y-%m-%d %H:%M:%S"),
                end.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
        for raw_row in rows:
            input_rows += 1
            source, metric, entity_id, raw_timestamp, value, unit, lat, lon = raw_row
            source = _required_nonempty_string(source, "snapshot source")
            metric = _required_nonempty_string(metric, "snapshot metric")
            entity_id = _required_nonempty_string(
                entity_id, f"snapshot {source}/{metric} entity ID"
            )
            unit = _required_nonempty_string(
                unit, f"snapshot {source}/{metric} unit"
            )
            timestamp = _parse_observation_timestamp(raw_timestamp)
            try:
                numeric_value = float(value)
                numeric_lat = float(lat)
                numeric_lon = float(lon)
            except (TypeError, ValueError):
                quality_excluded_rows += 1
                continue
            if not all(
                math.isfinite(item)
                for item in (numeric_value, numeric_lat, numeric_lon)
            ):
                quality_excluded_rows += 1
                continue
            if distance_km(
                anchor_lat,
                anchor_lon,
                numeric_lat,
                numeric_lon,
            ) > radius_km:
                continue
            if (
                source == "purpleair"
                and metric == "pm25"
                and not purpleair_reading_is_eligible(entity_id, timestamp)
            ):
                quality_excluded_rows += 1
                continue
            finite_in_radius_rows += 1
            rendered.add((source, metric, unit))
            observations.append(
                HourlyObservation(
                    source=source,
                    metric=metric,
                    entity_id=entity_id,
                    timestamp=timestamp,
                    value=numeric_value,
                    unit=unit,
                    nomination_eligible=series_is_nomination_eligible(
                        source,
                        metric,
                        entity_id,
                    ),
                )
            )
    finally:
        if connection is not None:
            connection.close()
        after_hash = _file_sha256(resolved)
        if after_hash != expected_sha256:
            raise RuntimeError(
                f"snapshot SHA-256 mismatch after read: {after_hash} != {expected_sha256}"
            )
    observations.sort(
        key=lambda row: (
            row.source,
            row.metric,
            row.timestamp,
            row.entity_id,
            row.value,
        )
    )
    rendered_metrics = tuple(
        {"source": source, "metric": metric, "unit": unit}
        for source, metric, unit in sorted(rendered)
    )
    return SnapshotPruningInput(
        snapshot_sha256=after_hash,
        observations=tuple(observations),
        rendered_metrics=rendered_metrics,
        input_row_count=input_rows,
        finite_in_radius_row_count=finite_in_radius_rows,
        quality_excluded_row_count=quality_excluded_rows,
    )


def _required_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _required_nonempty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _finite_or_null(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be finite numeric or null")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite numeric or null")
    return numeric


def _normalized_timestamp(value: object, anchor_id: str) -> str:
    raw = _required_nonempty_string(value, f"timestamp for anchor {anchor_id}")
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid anchor timestamp for {anchor_id}: {raw}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    parsed = parsed.astimezone(UTC)
    start = datetime.fromisoformat(STUDY_START.replace("Z", "+00:00"))
    end = datetime.fromisoformat(STUDY_END_EXCLUSIVE.replace("Z", "+00:00"))
    if not start <= parsed < end:
        raise ValueError(
            f"anchor timestamp for {anchor_id} lies outside the declared study window"
        )
    return parsed.isoformat().replace("+00:00", "Z")


def _normalize_input(payload: object) -> dict[str, object]:
    root = _required_mapping(payload, "input root")
    if root.get("schema_version") != 1:
        raise ValueError("schema_version must equal 1")
    if root.get("input_kind") != SYNTHETIC_INPUT_KIND:
        raise ValueError(
            "real pruning screen is blocked pending Mason ratification; "
            "input_kind must be synthetic"
        )
    if root.get("snapshot_sha256") != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("input must identify the canonical snapshot SHA-256")

    window = _required_mapping(root.get("study_window"), "study_window")
    if (
        window.get("start") != STUDY_START
        or window.get("end_exclusive") != STUDY_END_EXCLUSIVE
    ):
        raise ValueError("input must use the exact declared study window")

    anchor_population = _required_nonempty_string(
        root.get("anchor_population"), "anchor_population"
    )
    outcome_definition = _required_nonempty_string(
        root.get("outcome_definition"), "outcome_definition"
    )

    raw_anchors = root.get("anchors")
    if not isinstance(raw_anchors, list) or not raw_anchors:
        raise ValueError("anchors must be a nonempty array")
    anchors: list[dict[str, object]] = []
    anchor_ids: set[str] = set()
    for position, raw_anchor in enumerate(raw_anchors, start=1):
        anchor = _required_mapping(raw_anchor, f"anchor {position}")
        anchor_id = _required_nonempty_string(
            anchor.get("anchor_id"), f"anchor ID at position {position}"
        )
        if anchor_id in anchor_ids:
            raise ValueError(f"duplicate anchor ID: {anchor_id}")
        anchor_ids.add(anchor_id)
        anchors.append(
            {
                "anchor_id": anchor_id,
                "timestamp": _normalized_timestamp(anchor.get("timestamp"), anchor_id),
                "outcome_value": _finite_or_null(
                    anchor.get("outcome_value"),
                    f"outcome_value for anchor {anchor_id}",
                ),
            }
        )

    raw_candidates = root.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a nonempty array")
    candidates: list[dict[str, object]] = []
    candidate_names: set[str] = set()
    for position, raw_candidate in enumerate(raw_candidates, start=1):
        candidate = _required_mapping(raw_candidate, f"candidate {position}")
        name = _required_nonempty_string(
            candidate.get("name"), f"candidate name at position {position}"
        )
        if name in candidate_names:
            raise ValueError(f"duplicate candidate name: {name}")
        candidate_names.add(name)

        values = candidate.get("values")
        if not isinstance(values, list) or len(values) != len(anchors):
            raise ValueError(f"candidate {name} must supply one value per anchor")
        mechanism_relevant = candidate.get("physical_mechanism_relevant")
        if type(mechanism_relevant) is not bool:
            raise ValueError(
                f"candidate {name} physical_mechanism_relevant must be boolean"
            )
        rationale = _required_nonempty_string(
            candidate.get("physical_mechanism_rationale"),
            f"candidate {name} nonempty physical-mechanism rationale",
        )
        candidates.append(
            {
                "name": name,
                "source": _required_nonempty_string(
                    candidate.get("source"), f"candidate {name} source"
                ),
                "metric": _required_nonempty_string(
                    candidate.get("metric"), f"candidate {name} metric"
                ),
                "unit": _required_nonempty_string(
                    candidate.get("unit"), f"candidate {name} unit"
                ),
                "physical_mechanism_relevant": mechanism_relevant,
                "physical_mechanism_rationale": rationale,
                "values": [
                    _finite_or_null(value, f"value {index} for candidate {name}")
                    for index, value in enumerate(values)
                ],
            }
        )

    missing_candidates = sorted(REQUIRED_CANDIDATES - candidate_names)
    if missing_candidates:
        raise ValueError(
            "missing required candidates: " + ", ".join(missing_candidates)
        )
    candidates.sort(key=lambda candidate: str(candidate["name"]))

    return {
        "schema_version": 1,
        "input_kind": SYNTHETIC_INPUT_KIND,
        "snapshot_sha256": LOCKED_SNAPSHOT_SHA256,
        "study_window": {
            "start": STUDY_START,
            "end_exclusive": STUDY_END_EXCLUSIVE,
        },
        "anchor_population": anchor_population,
        "outcome_definition": outcome_definition,
        "anchors": anchors,
        "candidates": candidates,
    }


def _candidate_seed(base_seed: int, candidate_name: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{candidate_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _spearman_statistic(
    first: np.ndarray,
    second: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray | float:
    """Vectorized Spearman rho with average ranks for SciPy bootstrap batches."""
    first_ranks = stats.rankdata(first, method="average", axis=axis)
    second_ranks = stats.rankdata(second, method="average", axis=axis)
    first_centered = first_ranks - np.mean(first_ranks, axis=axis, keepdims=True)
    second_centered = second_ranks - np.mean(
        second_ranks,
        axis=axis,
        keepdims=True,
    )
    numerator = np.sum(first_centered * second_centered, axis=axis)
    denominator = np.sqrt(
        np.sum(first_centered**2, axis=axis)
        * np.sum(second_centered**2, axis=axis)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = numerator / denominator
    if np.ndim(result) == 0:
        return float(result)
    return result


def candidate_statistics(
    outcomes: Sequence[float | None],
    values: Sequence[float | None],
    *,
    rng: np.random.Generator,
    thresholds: PruningThresholds,
) -> CandidateStatistics:
    """Compute one declared Spearman/paired-BCa cell."""
    pairs = [
        (outcome, value)
        for outcome, value in zip(outcomes, values, strict=True)
        if outcome is not None and value is not None
    ]
    pair_count = len(pairs)
    if pair_count < thresholds.min_pairs:
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=None,
            p_value=None,
            ci_low=None,
            ci_high=None,
            evaluable=False,
            unevaluable_reason=f"eligible n < {thresholds.min_pairs}",
        )

    outcome_array = np.asarray([pair[0] for pair in pairs], dtype=float)
    value_array = np.asarray([pair[1] for pair in pairs], dtype=float)
    if np.unique(value_array).size < 2:
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=None,
            p_value=None,
            ci_low=None,
            ci_high=None,
            evaluable=False,
            unevaluable_reason="constant variable series",
        )
    if np.unique(outcome_array).size < 2:
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=None,
            p_value=None,
            ci_low=None,
            ci_high=None,
            evaluable=False,
            unevaluable_reason="constant outcome series",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = stats.spearmanr(
                value_array,
                outcome_array,
                alternative="two-sided",
            )
        rho = float(result.statistic)
        p_value = float(result.pvalue)
    except (TypeError, ValueError, FloatingPointError) as exc:
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=None,
            p_value=None,
            ci_low=None,
            ci_high=None,
            evaluable=False,
            unevaluable_reason=f"Spearman computation failed ({type(exc).__name__})",
        )
    if not math.isfinite(rho) or not math.isfinite(p_value):
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=rho if math.isfinite(rho) else None,
            p_value=p_value if math.isfinite(p_value) else None,
            ci_low=None,
            ci_high=None,
            evaluable=False,
            unevaluable_reason="undefined Spearman statistic",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            interval_result = stats.bootstrap(
                (value_array, outcome_array),
                _spearman_statistic,
                paired=True,
                vectorized=True,
                confidence_level=thresholds.confidence_level,
                n_resamples=thresholds.bootstrap_resamples,
                method="BCa",
                rng=rng,
                batch=256,
            )
        ci_low = float(interval_result.confidence_interval.low)
        ci_high = float(interval_result.confidence_interval.high)
    except (TypeError, ValueError, FloatingPointError, OverflowError) as exc:
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=rho,
            p_value=p_value,
            ci_low=None,
            ci_high=None,
            evaluable=False,
            unevaluable_reason=(
                f"paired BCa bootstrap failed ({type(exc).__name__})"
            ),
        )
    if not math.isfinite(ci_low) or not math.isfinite(ci_high):
        return CandidateStatistics(
            eligible_pair_count=pair_count,
            rho=rho,
            p_value=p_value,
            ci_low=ci_low if math.isfinite(ci_low) else None,
            ci_high=ci_high if math.isfinite(ci_high) else None,
            evaluable=False,
            unevaluable_reason="undefined paired BCa interval",
        )
    return CandidateStatistics(
        eligible_pair_count=pair_count,
        rho=rho,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        evaluable=True,
        unevaluable_reason=None,
    )


def _candidate_statistics(
    outcomes: Sequence[float | None],
    values: Sequence[float | None],
    *,
    seed: int,
    thresholds: PruningThresholds,
) -> CandidateStatistics:
    """Compatibility wrapper for the historical synthetic manifest."""
    return candidate_statistics(
        outcomes,
        values,
        rng=np.random.default_rng(seed),
        thresholds=thresholds,
    )


def evaluate_pruning_grid(
    pollutant_series: Mapping[tuple[str, str], Mapping[datetime, float]],
    candidate_series: Mapping[tuple[str, str], Mapping[datetime, float]],
    *,
    thresholds: PruningThresholds = PruningThresholds(),
) -> list[dict[str, object]]:
    """Evaluate every candidate across the declared six-by-four grid."""
    if set(pollutant_series) != set(POLLUTANT_GRID):
        missing = sorted(set(POLLUTANT_GRID) - set(pollutant_series))
        extra = sorted(set(pollutant_series) - set(POLLUTANT_GRID))
        raise ValueError(f"pollutant grid mismatch; missing={missing}, extra={extra}")
    if not candidate_series:
        raise ValueError("D4 candidate series inventory is empty")

    deseasonalized_pollutants = {
        key: deseasonalize_hourly(series)
        for key, series in sorted(pollutant_series.items())
    }
    deseasonalized_candidates = {
        key: deseasonalize_hourly(series)
        for key, series in sorted(candidate_series.items())
    }
    specifications = [
        (candidate, pollutant, lag)
        for candidate in sorted(deseasonalized_candidates)
        for pollutant in sorted(POLLUTANT_GRID)
        for lag in LAGS_HOURS
    ]
    children = np.random.SeedSequence(thresholds.bootstrap_seed).spawn(
        len(specifications)
    )
    cells: list[dict[str, object]] = []
    for spawn_index, (specification, child) in enumerate(
        zip(specifications, children, strict=True)
    ):
        candidate_key, pollutant_key, lag = specification
        x_values, y_values = pair_leading_series(
            deseasonalized_pollutants[pollutant_key],
            deseasonalized_candidates[candidate_key],
            lag_hours=lag,
        )
        cell = candidate_statistics(
            y_values,
            x_values,
            rng=np.random.default_rng(child),
            thresholds=thresholds,
        )
        nonsignificant = (
            None
            if cell.p_value is None
            else cell.p_value >= thresholds.alpha
        )
        negligible = (
            None
            if cell.rho is None
            else abs(cell.rho) < thresholds.negligible_abs_rho
        )
        ci_covers_zero = (
            None
            if cell.ci_low is None or cell.ci_high is None
            else cell.ci_low <= 0.0 <= cell.ci_high
        )
        statistical_drop_condition = bool(
            cell.evaluable
            and nonsignificant is True
            and negligible is True
            and ci_covers_zero is True
        )
        cells.append(
            {
                "candidate_source": candidate_key[0],
                "candidate_metric": candidate_key[1],
                "pollutant_source": pollutant_key[0],
                "pollutant_metric": pollutant_key[1],
                "lag_hours": lag,
                "eligible_pair_count": cell.eligible_pair_count,
                "rho": cell.rho,
                "p_value": cell.p_value,
                "ci_low": cell.ci_low,
                "ci_high": cell.ci_high,
                "evaluable": cell.evaluable,
                "unevaluable_reason": cell.unevaluable_reason,
                "nonsignificant": nonsignificant,
                "negligible": negligible,
                "ci_covers_zero": ci_covers_zero,
                "statistical_drop_condition": statistical_drop_condition,
                "bootstrap_substream": {
                    "master_seed": thresholds.bootstrap_seed,
                    "spawn_index": spawn_index,
                    "spawn_key": list(child.spawn_key),
                },
            }
        )
    return cells


def finalize_grid_decisions(
    cells: Sequence[Mapping[str, object]],
    *,
    assessments: Mapping[str, MechanismAssessment],
) -> dict[str, object]:
    """Apply the all-cells conjunction and Mason's mechanism veto."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for position, raw_cell in enumerate(cells, start=1):
        if not isinstance(raw_cell, Mapping):
            raise ValueError(f"grid cell {position} must be an object")
        source = _required_nonempty_string(
            raw_cell.get("candidate_source"), f"grid cell {position} candidate source"
        )
        metric = _required_nonempty_string(
            raw_cell.get("candidate_metric"), f"grid cell {position} candidate metric"
        )
        grouped[(source, metric)].append(dict(raw_cell))
    if not grouped:
        raise ValueError("grid has no candidate cells")

    expected_cells = {
        (source, metric, lag)
        for source, metric in POLLUTANT_GRID
        for lag in LAGS_HOURS
    }
    variables: list[dict[str, object]] = []
    mechanism_review_complete = True
    for source, metric in sorted(grouped):
        variable_cells = grouped[(source, metric)]
        identities = {
            (
                str(cell.get("pollutant_source")),
                str(cell.get("pollutant_metric")),
                cell.get("lag_hours"),
            )
            for cell in variable_cells
        }
        if identities != expected_cells or len(variable_cells) != len(expected_cells):
            raise ValueError(
                f"candidate {source}/{metric} does not have the exact 24-cell grid"
            )
        variable_cells.sort(
            key=lambda cell: (
                str(cell["pollutant_source"]),
                str(cell["pollutant_metric"]),
                int(cell["lag_hours"]),
            )
        )
        passing_count = sum(
            cell.get("statistical_drop_condition") is True
            for cell in variable_cells
        )
        inevaluable_count = sum(
            cell.get("evaluable") is not True for cell in variable_cells
        )
        statistical_pass = passing_count == len(expected_cells)
        assessment_key = f"{source}/{metric}"
        assessment = assessments.get(assessment_key) if statistical_pass else None
        if not statistical_pass:
            decision = "keep"
            reason = "at least one grid cell showed relevance or was inevaluable"
        elif assessment is None:
            mechanism_review_complete = False
            decision = "keep"
            reason = "required Mason mechanism assessment missing; ambiguity keeps"
        elif assessment.relevant:
            decision = "keep"
            reason = "standard atmospheric mechanism retained by Mason"
        elif assessment.ambiguous:
            decision = "keep"
            reason = "mechanism assessment ambiguous; keep by declaration"
        else:
            decision = "drop"
            reason = "all 24 statistical cells passed and mechanism veto passed"
        variables.append(
            {
                "source": source,
                "metric": metric,
                "statistical_cells_total": len(expected_cells),
                "statistical_cells_passing": passing_count,
                "inevaluable_cell_count": inevaluable_count,
                "all_statistical_cells_pass": statistical_pass,
                "mechanism_assessment": (
                    asdict(assessment) if assessment is not None else None
                ),
                "decision": decision,
                "reason": reason,
            }
        )
    return {
        "mechanism_review_complete": mechanism_review_complete,
        "variables": variables,
        "summary": {
            "candidate_count": len(variables),
            "keep_count": sum(row["decision"] == "keep" for row in variables),
            "drop_count": sum(row["decision"] == "drop" for row in variables),
            "statistical_pass_count": sum(
                row["all_statistical_cells_pass"] is True for row in variables
            ),
        },
    }


def _serialize_hourly_inputs(
    series: Mapping[tuple[str, str], Mapping[datetime, float]],
) -> list[dict[str, object]]:
    return [
        {
            "source": source,
            "metric": metric,
            "points": [
                [timestamp.isoformat().replace("+00:00", "Z"), value]
                for timestamp, value in sorted(series[(source, metric)].items())
            ],
        }
        for source, metric in sorted(series)
    ]


def build_snapshot_manifest(
    snapshot: SnapshotPruningInput,
    *,
    run_number: int,
    assessments: Mapping[str, MechanismAssessment],
    thresholds: PruningThresholds = PruningThresholds(),
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> dict[str, object]:
    """Build the complete schema-2 manifest from one immutable snapshot read."""
    if type(run_number) is not int or run_number < 1:
        raise ValueError("run number must be a positive integer")
    scope = build_metric_scope(snapshot.rendered_metrics)
    candidate_keys = {
        (str(row["source"]), str(row["metric"])) for row in scope["candidates"]
    }
    pollutant_keys = set(POLLUTANT_GRID)
    pollutant_series, candidate_series = build_hourly_series(
        snapshot.observations,
        pollutant_keys=pollutant_keys,
        candidate_keys=candidate_keys,
    )
    cells = evaluate_pruning_grid(
        pollutant_series,
        candidate_series,
        thresholds=thresholds,
    )
    decisions = finalize_grid_decisions(cells, assessments=assessments)
    candidate_units = {
        (str(row["source"]), str(row["metric"])): str(row["unit"])
        for row in scope["candidates"]
    }
    variables: list[dict[str, object]] = []
    for raw_row in decisions["variables"]:
        if not isinstance(raw_row, Mapping):
            raise ValueError("D4 decision row must be an object")
        row = dict(raw_row)
        key = (str(row["source"]), str(row["metric"]))
        row["unit"] = candidate_units[key]
        variables.append(row)
    drop_metric_keys = sorted(
        f"{row['source']}/{row['metric']}"
        for row in variables
        if row["decision"] == "drop"
    )
    nomination_rows = sum(
        observation.nomination_eligible
        and (observation.source, observation.metric) in pollutant_keys
        for observation in snapshot.observations
    )
    return {
        "schema_version": 2,
        "run_number": run_number,
        "threshold_status": THRESHOLD_STATUS,
        "real_screen_executed": True,
        "mechanism_review_complete": decisions["mechanism_review_complete"],
        "snapshot_sha256": snapshot.snapshot_sha256,
        "study_window": {
            "start": STUDY_START,
            "end_exclusive": STUDY_END_EXCLUSIVE,
        },
        "target": {
            "lat": anchor_lat,
            "lon": anchor_lon,
            "radius_km": radius_km,
        },
        "construction": {
            "entity_hour_collapse": "median",
            "pollutant_hour_aggregate": "median across eligible entities",
            "candidate_hour_aggregate": "arithmetic mean across in-radius entities",
            "deseasonalization": (
                "subtract each series' window-wide UTC-hour-of-day mean"
            ),
            "candidate_lead_hours": list(LAGS_HOURS),
            "lag_orientation": "candidate x(t-L) paired with pollutant y(t)",
            "pollutant_grid": [
                {"source": source, "metric": metric}
                for source, metric in POLLUTANT_GRID
            ],
        },
        "method": {
            "statistic": "two-sided Spearman rank correlation (average ties)",
            "confidence_interval": "80% paired iid BCa bootstrap",
            "scipy_version": scipy.__version__,
            "rng": "numpy SeedSequence.spawn in sorted cell order",
        },
        "thresholds": asdict(thresholds),
        "statistical_caveats": list(D4_STATISTICAL_CAVEATS),
        "expected_outcome_declared_before_run": (
            "zero drops likely; gh_500 has a synoptic ridging/subsidence to "
            "stagnation mechanism and precipitable_water has a moisture/washout "
            "mechanism; parameters must not be weakened to manufacture pruning"
        ),
        "input_provenance": {
            "snapshot_sha256": snapshot.snapshot_sha256,
            "database_rows_in_study_window": snapshot.input_row_count,
            "finite_b7_eligible_in_radius_rows": (
                snapshot.finite_in_radius_row_count
            ),
            "quality_excluded_rows": snapshot.quality_excluded_row_count,
            "nomination_eligible_pollutant_rows": nomination_rows,
        },
        "inventories": scope,
        "hourly_inputs": {
            "pollutants": _serialize_hourly_inputs(pollutant_series),
            "candidates": _serialize_hourly_inputs(candidate_series),
        },
        "cells": cells,
        "variables": variables,
        "drop_metric_keys": drop_metric_keys,
        "summary": decisions["summary"],
    }


def metric_is_retained(
    source: str,
    metric: str,
    *,
    manifest: Mapping[str, object] | None = None,
) -> bool:
    """Return the ratified prompt decision, refusing any exempt-metric drop."""
    if manifest is None:
        if not PRUNING_FIXTURE_PATH.is_file():
            return True
        manifest = load_pruning_fixture()
    if manifest.get("schema_version") != 2:
        raise ValueError("unsupported D4 pruning fixture schema")
    if manifest.get("real_screen_executed") is not True:
        raise ValueError("D4 pruning fixture is not a real executed screen")
    if manifest.get("mechanism_review_complete") is not True:
        raise ValueError("D4 pruning fixture mechanism review is incomplete")
    inventories = _required_mapping(
        manifest.get("inventories"), "D4 pruning inventories"
    )
    raw_exempt = inventories.get("exempt_rendered")
    if not isinstance(raw_exempt, list):
        raise ValueError("D4 pruning exempt inventory must be an array")
    exempt_keys: set[str] = set()
    for position, raw_row in enumerate(raw_exempt, start=1):
        row = _required_mapping(raw_row, f"D4 exempt metric {position}")
        exempt_source = _required_nonempty_string(
            row.get("source"), f"D4 exempt metric {position} source"
        )
        exempt_metric = _required_nonempty_string(
            row.get("metric"), f"D4 exempt metric {position} metric"
        )
        exempt_keys.add(f"{exempt_source}/{exempt_metric}")
    raw_drops = manifest.get("drop_metric_keys")
    if not isinstance(raw_drops, list) or not all(
        isinstance(item, str) and item for item in raw_drops
    ):
        raise ValueError("D4 pruning drop_metric_keys must be a string array")
    drop_keys = set(raw_drops)
    invalid = sorted(drop_keys & exempt_keys)
    if invalid:
        raise ValueError(
            "D4 attempted to drop scorer-exempt metric: " + ", ".join(invalid)
        )
    return f"{source}/{metric}" not in drop_keys


def write_numbered_manifest(
    manifest: Mapping[str, object],
    path: Path,
    *,
    run_number: int,
) -> None:
    """Exclusively write one canonical numbered real-screen artifact."""
    if type(run_number) is not int or run_number < 1:
        raise ValueError("run number must be a positive integer")
    if manifest.get("run_number") != run_number:
        raise ValueError("manifest run number does not match requested run number")
    token = f"run-{run_number:03d}"
    if token not in path.name:
        raise ValueError(f"numbered D4 output filename must contain {token}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    manifest,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                )
                + "\n"
            )
    except FileExistsError as exc:
        raise FileExistsError(
            f"numbered D4 artifact already exists and cannot be overwritten: {path}"
        ) from exc


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def build_manifest(
    payload: object,
    *,
    thresholds: PruningThresholds = PruningThresholds(),
) -> dict[str, object]:
    """Validate synthetic paired input and build the deterministic B3 manifest."""
    normalized = _normalize_input(payload)
    input_digest = hashlib.sha256(
        _canonical_json(normalized).encode("utf-8")
    ).hexdigest()
    anchors = normalized["anchors"]
    candidates = normalized["candidates"]
    assert isinstance(anchors, list)
    assert isinstance(candidates, list)
    outcomes = [anchor["outcome_value"] for anchor in anchors]

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        name = str(candidate["name"])
        seed = _candidate_seed(thresholds.bootstrap_seed, name)
        candidate_stats = _candidate_statistics(
            outcomes,
            candidate["values"],
            seed=seed,
            thresholds=thresholds,
        )
        decision = decide_pruning(
            candidate_stats,
            physical_mechanism_relevant=bool(
                candidate["physical_mechanism_relevant"]
            ),
            thresholds=thresholds,
        )
        rows.append(
            {
                "name": name,
                "source": candidate["source"],
                "metric": candidate["metric"],
                "unit": candidate["unit"],
                "physical_mechanism_relevant": candidate[
                    "physical_mechanism_relevant"
                ],
                "physical_mechanism_rationale": candidate[
                    "physical_mechanism_rationale"
                ],
                "input_pair_count": len(anchors),
                "eligible_pair_count": candidate_stats.eligible_pair_count,
                "missing_pair_count": (
                    len(anchors) - candidate_stats.eligible_pair_count
                ),
                "rho": candidate_stats.rho,
                "p_value": candidate_stats.p_value,
                "ci_low": candidate_stats.ci_low,
                "ci_high": candidate_stats.ci_high,
                "bootstrap_seed": seed,
                "evaluable": candidate_stats.evaluable,
                "unevaluable_reason": candidate_stats.unevaluable_reason,
                "nonsignificant": decision.nonsignificant,
                "negligible": decision.negligible,
                "ci_covers_zero": decision.ci_covers_zero,
                "no_physical_mechanism": decision.no_physical_mechanism,
                "decision": decision.decision,
                "reason": decision.reason,
            }
        )

    return {
        "schema_version": 1,
        "threshold_status": THRESHOLD_STATUS,
        "real_screen_executed": False,
        "method": {
            "statistic": "Spearman rank correlation (average ranks for ties)",
            "p_value": "two-sided large-sample scipy.stats.spearmanr",
            "confidence_interval": "paired BCa bootstrap",
            "scipy_version": scipy.__version__,
        },
        "thresholds": asdict(thresholds),
        "input_provenance": {
            "input_kind": SYNTHETIC_INPUT_KIND,
            "input_manifest_sha256": input_digest,
            "snapshot_sha256": LOCKED_SNAPSHOT_SHA256,
            "study_window": normalized["study_window"],
            "anchor_population": normalized["anchor_population"],
            "outcome_definition": normalized["outcome_definition"],
            "anchor_count": len(anchors),
        },
        "summary": {
            "candidate_count": len(rows),
            "keep_count": sum(row["decision"] == "keep" for row in rows),
            "drop_count": sum(row["decision"] == "drop" for row in rows),
            "unevaluable_count": sum(not bool(row["evaluable"]) for row in rows),
        },
        "variables": rows,
    }


def _format_number(value: object) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.12g}"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_markdown(manifest: Mapping[str, object]) -> str:
    """Render the canonical human-review table for a B3 manifest."""
    provenance = _required_mapping(
        manifest.get("input_provenance"), "manifest input_provenance"
    )
    raw_rows = manifest.get("variables")
    if not isinstance(raw_rows, list):
        raise ValueError("manifest variables must be an array")
    lines = [
        "# B3/D4 variable-pruning screen",
        "",
        f"Threshold status: {_markdown_cell(manifest.get('threshold_status'))}",
        "",
        (
            "Synthetic input SHA-256: `"
            f"{_markdown_cell(provenance.get('input_manifest_sha256'))}`"
        ),
        "",
        (
            "| Variable | Source | Metric | Unit | Eligible n | Missing n | "
            "rho | p | 80% BCa low | 80% BCa high | Mechanism relevant | "
            "Decision | Reason |"
        ),
        (
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"
        ),
    ]
    for raw_row in raw_rows:
        row = _required_mapping(raw_row, "manifest variable row")
        lines.append(
            "| "
            + " | ".join(
                [
                    _markdown_cell(row.get("name")),
                    _markdown_cell(row.get("source")),
                    _markdown_cell(row.get("metric")),
                    _markdown_cell(row.get("unit")),
                    _markdown_cell(row.get("eligible_pair_count")),
                    _markdown_cell(row.get("missing_pair_count")),
                    _format_number(row.get("rho")),
                    _format_number(row.get("p_value")),
                    _format_number(row.get("ci_low")),
                    _format_number(row.get("ci_high")),
                    _markdown_cell(row.get("physical_mechanism_relevant")),
                    _markdown_cell(row.get("decision")),
                    _markdown_cell(row.get("reason")),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


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
    """Write canonical, deterministic JSON atomically."""
    _write_text(
        path,
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a synthetic B3/D4 statistic fixture or execute the single "
            "ratified label-free snapshot grid screen."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--database", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", default=LOCKED_SNAPSHOT_SHA256)
    parser.add_argument("--run-number", type=int)
    parser.add_argument("--mechanism-assessments", type=Path)
    parser.add_argument("--anchor-lat", type=float, default=settings.aeris_target_lat)
    parser.add_argument("--anchor-lon", type=float, default=settings.aeris_target_lon)
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument(
        "--format",
        choices=("json", "markdown"),
        default="json",
    )
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"cannot read input {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in input {path}: {exc}") from exc


def load_mechanism_assessments(
    path: Path,
    *,
    run_number: int,
) -> dict[str, MechanismAssessment]:
    """Load Mason's exact, numbered D4 mechanism review input."""
    root = _required_mapping(_load_json(path), "mechanism assessment root")
    if root.get("schema_version") != 1:
        raise ValueError("mechanism assessment schema_version must equal 1")
    if root.get("run_number") != run_number:
        raise ValueError("mechanism assessment run number mismatch")
    raw_assessments = root.get("assessments")
    if not isinstance(raw_assessments, list):
        raise ValueError("mechanism assessments must be an array")
    assessments: dict[str, MechanismAssessment] = {}
    for position, raw_assessment in enumerate(raw_assessments, start=1):
        row = _required_mapping(raw_assessment, f"mechanism assessment {position}")
        source = _required_nonempty_string(
            row.get("source"), f"mechanism assessment {position} source"
        )
        metric = _required_nonempty_string(
            row.get("metric"), f"mechanism assessment {position} metric"
        )
        key = f"{source}/{metric}"
        if key in assessments:
            raise ValueError(f"duplicate mechanism assessment: {key}")
        relevant = row.get("relevant")
        ambiguous = row.get("ambiguous")
        if type(relevant) is not bool or type(ambiguous) is not bool:
            raise ValueError(
                f"mechanism assessment {key} relevant/ambiguous must be boolean"
            )
        assessment = _required_nonempty_string(
            row.get("assessment"), f"mechanism assessment {key} text"
        )
        assessments[key] = MechanismAssessment(
            relevant=relevant,
            ambiguous=ambiguous,
            assessment=assessment,
        )
    return assessments


@lru_cache(maxsize=1)
def load_pruning_fixture(
    path: Path = PRUNING_FIXTURE_PATH,
) -> dict[str, Any]:
    """Load the active real D4 artifact and prove it matches live scorer scope."""
    root = _required_mapping(_load_json(path), "D4 pruning fixture")
    fixture = dict(root)
    if fixture.get("schema_version") != 2:
        raise ValueError("unsupported D4 pruning fixture schema")
    if fixture.get("run_number") != 1:
        raise ValueError("active D4 pruning fixture must be run 1")
    if fixture.get("threshold_status") != THRESHOLD_STATUS:
        raise ValueError("D4 pruning fixture threshold status drifted")
    if fixture.get("real_screen_executed") is not True:
        raise ValueError("D4 pruning fixture is not a real executed screen")
    if fixture.get("mechanism_review_complete") is not True:
        raise ValueError("D4 pruning fixture mechanism review is incomplete")
    if fixture.get("snapshot_sha256") != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("D4 pruning fixture does not match the locked snapshot")
    if fixture.get("thresholds") != asdict(PruningThresholds()):
        raise ValueError("D4 pruning fixture thresholds differ from ratification")
    if fixture.get("statistical_caveats") != list(D4_STATISTICAL_CAVEATS):
        raise ValueError("D4 pruning fixture statistical caveats drifted")
    if fixture.get("study_window") != {
        "start": STUDY_START,
        "end_exclusive": STUDY_END_EXCLUSIVE,
    }:
        raise ValueError("D4 pruning fixture study window drifted")

    inventories = _required_mapping(
        fixture.get("inventories"), "D4 pruning inventories"
    )
    raw_scorer_owned = inventories.get("scorer_owned")
    if not isinstance(raw_scorer_owned, list):
        raise ValueError("D4 scorer-owned inventory must be an array")
    artifact_owners: dict[tuple[str, str], tuple[str, ...]] = {}
    for position, raw_row in enumerate(raw_scorer_owned, start=1):
        row = _required_mapping(raw_row, f"D4 scorer-owned metric {position}")
        source = _required_nonempty_string(
            row.get("source"), f"D4 scorer-owned metric {position} source"
        )
        metric = _required_nonempty_string(
            row.get("metric"), f"D4 scorer-owned metric {position} metric"
        )
        raw_owners = row.get("owners")
        if not isinstance(raw_owners, list) or not all(
            isinstance(owner, str) and owner for owner in raw_owners
        ):
            raise ValueError(f"D4 scorer-owned metric {source}/{metric} has no owners")
        key = (source, metric)
        if key in artifact_owners:
            raise ValueError(f"duplicate D4 scorer-owned metric: {source}/{metric}")
        artifact_owners[key] = tuple(raw_owners)
    if artifact_owners != phase2_metric_owners():
        raise ValueError(
            "live Phase-2 metric ownership differs from the executed D4 screen"
        )

    def inventory_keys(field_name: str) -> set[tuple[str, str]]:
        raw_rows = inventories.get(field_name)
        if not isinstance(raw_rows, list):
            raise ValueError(f"D4 {field_name} inventory must be an array")
        keys: set[tuple[str, str]] = set()
        for position, raw_row in enumerate(raw_rows, start=1):
            row = _required_mapping(raw_row, f"D4 {field_name} metric {position}")
            key = (
                _required_nonempty_string(
                    row.get("source"), f"D4 {field_name} metric {position} source"
                ),
                _required_nonempty_string(
                    row.get("metric"), f"D4 {field_name} metric {position} metric"
                ),
            )
            if key in keys:
                raise ValueError(f"duplicate D4 {field_name} metric: {key}")
            keys.add(key)
        return keys

    rendered_keys = inventory_keys("rendered")
    candidate_keys = inventory_keys("candidates")
    exempt_keys = inventory_keys("exempt_rendered")
    if candidate_keys & exempt_keys or candidate_keys | exempt_keys != rendered_keys:
        raise ValueError("D4 rendered candidate/exempt partition is malformed")
    if exempt_keys != rendered_keys & set(artifact_owners):
        raise ValueError("D4 exempt inventory differs from live scorer ownership")

    raw_cells = fixture.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 24 * len(candidate_keys):
        raise ValueError("D4 pruning fixture does not contain 24 cells per candidate")
    assessments = load_mechanism_assessments(
        MECHANISM_FIXTURE_PATH,
        run_number=1,
    )
    recomputed = finalize_grid_decisions(raw_cells, assessments=assessments)
    if fixture.get("variables") != [
        {
            **row,
            "unit": next(
                str(candidate["unit"])
                for candidate in inventories["candidates"]
                if candidate["source"] == row["source"]
                and candidate["metric"] == row["metric"]
            ),
        }
        for row in recomputed["variables"]
    ]:
        raise ValueError("D4 variable decisions do not reproduce from stored cells")
    expected_drops = sorted(
        f"{row['source']}/{row['metric']}"
        for row in recomputed["variables"]
        if row["decision"] == "drop"
    )
    if fixture.get("drop_metric_keys") != expected_drops:
        raise ValueError("D4 drop list does not reproduce from stored decisions")
    return fixture


def pruning_manifest_payload() -> dict[str, Any]:
    """Hash-link and embed the complete D4 run plus Mason's verbatim input."""
    screen = deepcopy(load_pruning_fixture())
    mechanism_root = _required_mapping(
        _load_json(MECHANISM_FIXTURE_PATH),
        "D4 mechanism fixture",
    )
    load_mechanism_assessments(MECHANISM_FIXTURE_PATH, run_number=1)
    return {
        "artifact": PRUNING_FIXTURE_PATH.name,
        "artifact_sha256": _file_sha256(PRUNING_FIXTURE_PATH),
        "mechanism_assessment_artifact": MECHANISM_FIXTURE_PATH.name,
        "mechanism_assessment_artifact_sha256": _file_sha256(
            MECHANISM_FIXTURE_PATH
        ),
        "mechanism_assessment_input": deepcopy(dict(mechanism_root)),
        "screen": screen,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the historical synthetic tool or numbered ratified snapshot screen."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.database is not None:
        if args.run_number is None:
            _argument_error(parser, "--run-number is required with --database")
        if args.mechanism_assessments is None:
            _argument_error(
                parser,
                "--mechanism-assessments is required with --database",
            )
        if args.format != "json":
            _argument_error(parser, "real D4 output must use canonical JSON")
        try:
            assessments = load_mechanism_assessments(
                args.mechanism_assessments,
                run_number=args.run_number,
            )
            snapshot = load_snapshot_inputs(
                args.database,
                expected_sha256=args.expected_sha256,
                anchor_lat=args.anchor_lat,
                anchor_lon=args.anchor_lon,
                radius_km=args.radius_km,
            )
            manifest = build_snapshot_manifest(
                snapshot,
                run_number=args.run_number,
                assessments=assessments,
                anchor_lat=args.anchor_lat,
                anchor_lon=args.anchor_lon,
                radius_km=args.radius_km,
            )
            if manifest["mechanism_review_complete"] is not True:
                missing = [
                    f"{row['source']}/{row['metric']}"
                    for row in manifest["variables"]
                    if row["all_statistical_cells_pass"] is True
                    and row["mechanism_assessment"] is None
                ]
                raise ValueError(
                    "Mason mechanism assessment required before output: "
                    + ", ".join(missing)
                )
        except (OSError, ValueError) as exc:
            _argument_error(parser, str(exc))
        write_numbered_manifest(manifest, args.output, run_number=args.run_number)
        return 0

    try:
        if args.run_number is not None or args.mechanism_assessments is not None:
            raise ValueError(
                "--run-number/--mechanism-assessments require --database"
            )
        manifest = build_manifest(_load_json(args.input))
        if args.format == "markdown":
            _write_text(args.output, render_markdown(manifest))
        else:
            write_manifest(manifest, args.output)
    except (OSError, ValueError) as exc:
        _argument_error(parser, str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
