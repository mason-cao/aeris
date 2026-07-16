"""Deterministic B3/D4 conservative variable-pruning screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, NoReturn

import numpy as np
import scipy
from scipy import stats

from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    STUDY_END_EXCLUSIVE,
    STUDY_START,
)

THRESHOLD_STATUS: Final = "declared — pending Mason ratification"
REQUIRED_CANDIDATES: Final = frozenset({"gh_500", "precipitable_water"})
SYNTHETIC_INPUT_KIND: Final = "synthetic"


@dataclass(frozen=True)
class PruningThresholds:
    """Pre-declared D4 thresholds and bootstrap settings."""

    alpha: float = 0.20
    negligible_abs_rho: float = 0.05
    confidence_level: float = 0.80
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 20_260_715
    min_pairs: int = 4

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


def _spearman_statistic(first: np.ndarray, second: np.ndarray) -> float:
    return float(stats.spearmanr(first, second, alternative="two-sided").statistic)


def _candidate_statistics(
    outcomes: Sequence[float | None],
    values: Sequence[float | None],
    *,
    seed: int,
    thresholds: PruningThresholds,
) -> CandidateStatistics:
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
                vectorized=False,
                confidence_level=thresholds.confidence_level,
                n_resamples=thresholds.bootstrap_resamples,
                method="BCa",
                rng=np.random.default_rng(seed),
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
            "Build a synthetic B3/D4 pruning-screen manifest. Real inputs are "
            "blocked pending Mason ratification."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
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


def main(argv: Sequence[str] | None = None) -> int:
    """Run the synthetic-only B3/D4 manifest CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
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
