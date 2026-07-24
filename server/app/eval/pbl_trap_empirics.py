"""Label-free B1 same-cell PBL-reference diagnostics on frozen SQLite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from app.llm.corroboration import (
    CONTRADICTING,
    DEFAULT_TRAP_TOLERANCE,
    SILENT,
    SUPPORTING,
    TrapTolerance,
)
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256

STUDY_START = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
STUDY_END_EXCLUSIVE = datetime(2026, 7, 13, 0, 0, tzinfo=UTC)
WINDOW_HALF_WIDTH = timedelta(hours=36)


@dataclass(frozen=True, order=True)
class PblObservation:
    entity_id: str
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class PblAnchorAssessment:
    reference_n: int
    reference_value_n: int
    reference_mean: float | None
    reference_pstdev: float | None
    support_threshold: float | None
    verdict: int
    silence_reason: str | None


@dataclass(frozen=True)
class PblTrapEmpiricalReport:
    snapshot_sha256: str
    study_start: str
    study_end_exclusive: str
    observation_count: int
    candidate_anchor_count: int
    reference_n_counts: tuple[tuple[int, int], ...]
    insufficient_distinct_day_count: int
    zero_spread_count: int
    evaluable_count: int
    supporting_count: int
    contradicting_count: int
    silent_count: int
    supporting_rate: float | None
    contradicting_rate: float | None
    silent_rate: float | None

    def to_dict(self) -> dict[str, Any]:
        denominator = self.candidate_anchor_count
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "study_start": self.study_start,
            "study_end_exclusive": self.study_end_exclusive,
            "observation_count": self.observation_count,
            "candidate_anchor_count": self.candidate_anchor_count,
            "reference_n_distribution": [
                {
                    "reference_n": reference_n,
                    "anchor_count": count,
                    "fraction": count / denominator if denominator else None,
                }
                for reference_n, count in self.reference_n_counts
            ],
            "insufficient_distinct_day_count": (
                self.insufficient_distinct_day_count
            ),
            "zero_spread_count": self.zero_spread_count,
            "evaluable_count": self.evaluable_count,
            "outcomes": {
                "supporting": {
                    "count": self.supporting_count,
                    "rate": self.supporting_rate,
                },
                "contradicting": {
                    "count": self.contradicting_count,
                    "rate": self.contradicting_rate,
                },
                "silent": {
                    "count": self.silent_count,
                    "rate": self.silent_rate,
                },
            },
            "tolerance": {
                "suppression_sigma": DEFAULT_TRAP_TOLERANCE.suppression_sigma,
                "min_same_hour_points": (
                    DEFAULT_TRAP_TOLERANCE.min_same_hour_points
                ),
                "sd_estimator": "population",
            },
        }


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalized_observation(observation: PblObservation) -> PblObservation:
    entity_id = str(observation.entity_id)
    if not entity_id:
        raise ValueError("PBL observation entity_id must be non-empty")
    value = float(observation.value)
    if not math.isfinite(value):
        raise ValueError("PBL observations must be finite")
    return PblObservation(entity_id, _ensure_utc(observation.timestamp), value)


def assess_anchor(
    anchor: PblObservation,
    observations: list[PblObservation] | tuple[PblObservation, ...],
    *,
    tolerance: TrapTolerance = DEFAULT_TRAP_TOLERANCE,
) -> PblAnchorAssessment:
    """Apply the declared R1 reference and verdict bands to one raw anchor."""
    event = _normalized_observation(anchor)
    window_start = event.timestamp - WINDOW_HALF_WIDTH
    window_end = event.timestamp + WINDOW_HALF_WIDTH
    references = [
        row
        for raw_row in observations
        for row in (_normalized_observation(raw_row),)
        if row.entity_id == event.entity_id
        and window_start <= row.timestamp <= window_end
        and row.timestamp.date() != event.timestamp.date()
        and row.timestamp.hour == event.timestamp.hour
    ]
    distinct_dates = {row.timestamp.date() for row in references}
    reference_n = len(distinct_dates)
    if reference_n < tolerance.min_same_hour_points:
        return PblAnchorAssessment(
            reference_n=reference_n,
            reference_value_n=len(references),
            reference_mean=None,
            reference_pstdev=None,
            support_threshold=None,
            verdict=SILENT,
            silence_reason="insufficient_distinct_days",
        )

    values = [row.value for row in references]
    reference_mean = statistics.fmean(values)
    reference_pstdev = statistics.pstdev(values)
    threshold = reference_mean - (
        tolerance.suppression_sigma * reference_pstdev
    )
    if reference_pstdev == 0.0:
        return PblAnchorAssessment(
            reference_n=reference_n,
            reference_value_n=len(references),
            reference_mean=reference_mean,
            reference_pstdev=reference_pstdev,
            support_threshold=threshold,
            verdict=SILENT,
            silence_reason="zero_spread",
        )
    if event.value <= threshold:
        verdict = SUPPORTING
        silence_reason = None
    elif event.value >= reference_mean:
        verdict = CONTRADICTING
        silence_reason = None
    else:
        verdict = SILENT
        silence_reason = "between_threshold_and_mean"
    return PblAnchorAssessment(
        reference_n=reference_n,
        reference_value_n=len(references),
        reference_mean=reference_mean,
        reference_pstdev=reference_pstdev,
        support_threshold=threshold,
        verdict=verdict,
        silence_reason=silence_reason,
    )


def build_report(
    observations: list[PblObservation] | tuple[PblObservation, ...],
    *,
    snapshot_sha256: str,
    study_start: datetime = STUDY_START,
    study_end_exclusive: datetime = STUDY_END_EXCLUSIVE,
    tolerance: TrapTolerance = DEFAULT_TRAP_TOLERANCE,
) -> PblTrapEmpiricalReport:
    """Aggregate R1 reference sizes and label-free verdict behavior."""
    start = _ensure_utc(study_start)
    end = _ensure_utc(study_end_exclusive)
    if end <= start:
        raise ValueError("study end must be after study start")
    normalized = tuple(
        sorted(
            _normalized_observation(row)
            for row in observations
            if start <= _ensure_utc(row.timestamp) < end
        )
    )
    seen: set[tuple[str, datetime]] = set()
    for row in normalized:
        key = (row.entity_id, row.timestamp)
        if key in seen:
            raise ValueError(f"duplicate PBL observation: {key}")
        seen.add(key)

    anchors = tuple(
        row
        for row in normalized
        if row.timestamp - WINDOW_HALF_WIDTH >= start
        and row.timestamp + WINDOW_HALF_WIDTH < end
    )
    assessments = tuple(
        assess_anchor(row, normalized, tolerance=tolerance)
        for row in anchors
    )
    reference_counts = Counter(result.reference_n for result in assessments)
    insufficient_count = sum(
        result.silence_reason == "insufficient_distinct_days"
        for result in assessments
    )
    zero_spread_count = sum(
        result.silence_reason == "zero_spread" for result in assessments
    )
    supporting_count = sum(
        result.verdict == SUPPORTING for result in assessments
    )
    contradicting_count = sum(
        result.verdict == CONTRADICTING for result in assessments
    )
    silent_count = sum(result.verdict == SILENT for result in assessments)
    denominator = len(anchors)

    def rate(count: int) -> float | None:
        return count / denominator if denominator else None

    return PblTrapEmpiricalReport(
        snapshot_sha256=snapshot_sha256,
        study_start=start.isoformat(),
        study_end_exclusive=end.isoformat(),
        observation_count=len(normalized),
        candidate_anchor_count=denominator,
        reference_n_counts=tuple(sorted(reference_counts.items())),
        insufficient_distinct_day_count=insufficient_count,
        zero_spread_count=zero_spread_count,
        evaluable_count=denominator - insufficient_count - zero_spread_count,
        supporting_count=supporting_count,
        contradicting_count=contradicting_count,
        silent_count=silent_count,
        supporting_rate=rate(supporting_count),
        contradicting_rate=rate(contradicting_count),
        silent_rate=rate(silent_count),
    )


def build_sigma_sweep(
    observations: list[PblObservation] | tuple[PblObservation, ...],
    *,
    snapshot_sha256: str,
    sigmas: tuple[float, ...] | list[float],
    study_start: datetime = STUDY_START,
    study_end_exclusive: datetime = STUDY_END_EXCLUSIVE,
) -> dict[str, Any]:
    """Per-sigma R1 outcome rates for the label-free threshold sweep.

    The contradiction band stays at the reference mean regardless of sigma
    (D3), so only support and silence trade off. The sweep is the "test
    different values" check Bracco delegated on 2026-07-24; it must run
    before any label exists.
    """
    if not sigmas:
        raise ValueError("at least one sigma value is required")
    cleaned = sorted({float(value) for value in sigmas})
    if any(not math.isfinite(value) or value <= 0.0 for value in cleaned):
        raise ValueError("sigma values must be finite and positive")

    reports = [
        build_report(
            observations,
            snapshot_sha256=snapshot_sha256,
            study_start=study_start,
            study_end_exclusive=study_end_exclusive,
            tolerance=replace(
                DEFAULT_TRAP_TOLERANCE, suppression_sigma=value
            ),
        )
        for value in cleaned
    ]
    first = reports[0]
    return {
        "snapshot_sha256": first.snapshot_sha256,
        "study_start": first.study_start,
        "study_end_exclusive": first.study_end_exclusive,
        "observation_count": first.observation_count,
        "candidate_anchor_count": first.candidate_anchor_count,
        "insufficient_distinct_day_count": (
            first.insufficient_distinct_day_count
        ),
        "zero_spread_count": first.zero_spread_count,
        "min_same_hour_points": DEFAULT_TRAP_TOLERANCE.min_same_hour_points,
        "sd_estimator": "population",
        "bands": (
            "support <= mean - sigma*pstdev; contradict >= mean; "
            "silent in between"
        ),
        "authority": (
            "sigma selection delegated to Mason in writing "
            "(Bracco reply 2026-07-24); label-free pre-freeze sweep"
        ),
        "sweep": [
            {
                "suppression_sigma": value,
                "evaluable_count": report.evaluable_count,
                "outcomes": report.to_dict()["outcomes"],
            }
            for value, report in zip(cleaned, reports, strict=True)
        ],
    }


def run_sigma_sweep(
    database_path: Path,
    *,
    expected_sha256: str,
    sigmas: tuple[float, ...] | list[float],
) -> dict[str, Any]:
    """Hash-verified read of the snapshot followed by the sigma sweep."""
    observations, verified_sha256 = _verified_observations(
        database_path,
        expected_sha256=expected_sha256,
    )
    return build_sigma_sweep(
        observations,
        snapshot_sha256=verified_sha256,
        sigmas=sigmas,
    )


def render_sweep_markdown(sweep: dict[str, Any]) -> str:
    def cell(outcome: dict[str, Any]) -> str:
        return f"{outcome['count']} ({_format_fraction(outcome['rate'])})"

    lines = [
        "| Sigma | Support | Contradict | Silent |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {row['suppression_sigma']} "
        f"| {cell(row['outcomes']['supporting'])} "
        f"| {cell(row['outcomes']['contradicting'])} "
        f"| {cell(row['outcomes']['silent'])} |"
        for row in sweep["sweep"]
    )
    lines.extend(
        (
            "",
            "| Diagnostic | Count |",
            "|---|---:|",
            f"| Stored PBL rows | {sweep['observation_count']} |",
            f"| Complete-window anchors | {sweep['candidate_anchor_count']} |",
            f"| Insufficient distinct-day references | "
            f"{sweep['insufficient_distinct_day_count']} |",
            f"| Zero-spread references | {sweep['zero_spread_count']} |",
        )
    )
    return "\n".join(lines)


def _snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_timestamp(value: str) -> datetime:
    return _ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _load_observations(connection: sqlite3.Connection) -> list[PblObservation]:
    rows = list(
        connection.execute(
            """
            SELECT source_entity_id, timestamp, value, unit
            FROM data_points
            WHERE source = 'noaa_gfs'
              AND metric = 'pbl_height'
              AND timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp, source_entity_id
            """,
            (
                STUDY_START.strftime("%Y-%m-%d %H:%M:%S"),
                STUDY_END_EXCLUSIVE.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    )
    units = {str(row["unit"]) for row in rows}
    if units != {"m"}:
        raise ValueError(f"noaa_gfs pbl_height units are not exactly m: {units}")
    return [
        PblObservation(
            entity_id=str(row["source_entity_id"]),
            timestamp=_parse_timestamp(str(row["timestamp"])),
            value=float(row["value"]),
        )
        for row in rows
    ]


def _verified_observations(
    database_path: Path,
    *,
    expected_sha256: str,
) -> tuple[list[PblObservation], str]:
    """Read the immutable snapshot and verify its hash before and after."""
    if expected_sha256 != LOCKED_SNAPSHOT_SHA256:
        raise ValueError(
            "expected SHA-256 is not the canonical locked snapshot hash: "
            f"{expected_sha256} != {LOCKED_SNAPSHOT_SHA256}"
        )
    resolved = database_path.resolve()
    before_hash = _snapshot_sha256(resolved)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != "
            f"{expected_sha256}"
        )
    connection = sqlite3.connect(
        f"file:{resolved}?mode=ro&immutable=1",
        uri=True,
    )
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        observations = _load_observations(connection)
    finally:
        connection.close()
    after_hash = _snapshot_sha256(resolved)
    if after_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch after read: {after_hash} != "
            f"{expected_sha256}"
        )
    return observations, after_hash


def run_empirics(
    database_path: Path,
    *,
    expected_sha256: str,
) -> PblTrapEmpiricalReport:
    """Hash-verified read of the snapshot followed by the R1 report."""
    observations, verified_sha256 = _verified_observations(
        database_path,
        expected_sha256=expected_sha256,
    )
    return build_report(observations, snapshot_sha256=verified_sha256)


def _format_fraction(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(report: PblTrapEmpiricalReport) -> str:
    denominator = report.candidate_anchor_count
    lines = [
        "| Reference n (distinct days) | Anchors | Fraction |",
        "|---:|---:|---:|",
    ]
    lines.extend(
        f"| {reference_n} | {count} | "
        f"{_format_fraction(count / denominator if denominator else None)} |"
        for reference_n, count in report.reference_n_counts
    )
    lines.extend(
        (
            "",
            "| Outcome | Anchors | Rate |",
            "|---|---:|---:|",
            f"| Support | {report.supporting_count} | "
            f"{_format_fraction(report.supporting_rate)} |",
            f"| Contradict | {report.contradicting_count} | "
            f"{_format_fraction(report.contradicting_rate)} |",
            f"| Silent | {report.silent_count} | "
            f"{_format_fraction(report.silent_rate)} |",
            "",
            "| Diagnostic | Count |",
            "|---|---:|",
            f"| Stored PBL rows | {report.observation_count} |",
            f"| Complete-window anchors | {report.candidate_anchor_count} |",
            f"| Evaluable positive-spread anchors | {report.evaluable_count} |",
            f"| Insufficient distinct-day references | "
            f"{report.insufficient_distinct_day_count} |",
            f"| Zero-spread references | {report.zero_spread_count} |",
        )
    )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.pbl_trap_empirics",
        description="Compute label-free B1 same-cell PBL reference behavior.",
    )
    parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="read-only frozen SQLite snapshot",
    )
    parser.add_argument(
        "--expected-sha256",
        required=True,
        help="canonical locked snapshot SHA-256",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    parser.add_argument(
        "--sigma-sweep",
        help=(
            "comma-separated suppression sigmas; runs the label-free "
            "threshold sweep instead of the single-sigma report"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write the sweep payload as JSON to this path",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.sigma_sweep is not None:
        sigmas = [
            float(part)
            for part in args.sigma_sweep.split(",")
            if part.strip()
        ]
        sweep = run_sigma_sweep(
            args.database,
            expected_sha256=args.expected_sha256,
            sigmas=sigmas,
        )
        if args.output is not None:
            args.output.write_text(
                json.dumps(sweep, indent=2, sort_keys=True) + "\n"
            )
        if args.format == "json":
            print(json.dumps(sweep, indent=2, sort_keys=True))
        else:
            print(render_sweep_markdown(sweep))
        return
    report = run_empirics(
        args.database,
        expected_sha256=args.expected_sha256,
    )
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
