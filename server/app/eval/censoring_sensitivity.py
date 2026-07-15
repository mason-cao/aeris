"""B15 label-free censoring sensitivity on an immutable SQLite snapshot."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from statistics import fmean
from typing import Any, Final

from app.collectors.geo import distance_km
from app.config import settings
from app.eval.calm_wind_empirics import (
    STUDY_END_EXCLUSIVE,
    STUDY_START,
    WINDOW_HALF_WIDTH,
    candidate_centers,
)
from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    SUPPORTING,
    BaselineCensoringStrategy,
    DEFAULT_CONCENTRATION_TOLERANCE,
    SO2_QUANTITATIVE_EXCLUSION_REASON,
    baseline_censor_limit,
    censor_baseline_values,
    qualitative_elevation_verdict,
)
from app.llm.observation_age import assess_observation_age
from app.provenance.openaq_pm25 import verified_monitor_entity_ids
from app.provenance.purpleair_qc import (
    LOCKED_SNAPSHOT_SHA256,
    excluded_purpleair_row_keys,
)

RELEVANT_SOURCES: Final = frozenset(
    {"openaq", "tceq", "epa_aqs", "purpleair", "sentinel5p"}
)
GROUND_CONCENTRATION_METRICS: Final = frozenset(
    {"bc", "co", "no2", "ozone", "pm10", "pm25", "so2"}
)
SENTINEL_COLUMN_METRICS: Final = frozenset(
    {"s5p_co_column", "s5p_no2_column", "s5p_so2_column"}
)
FIXTURE_PATH: Final = Path(__file__).parent / "fixtures" / "censoring_sensitivity.v1.json"


@dataclass(frozen=True)
class CensoringObservation:
    entity_id: str
    timestamp: datetime
    value: float
    distance_km: float


@dataclass(frozen=True)
class MetricCensoringSensitivity:
    source: str
    metric: str
    unit: str
    censor_limit: float | None
    replacement_value: float | None
    observation_count: int
    anchor_count: int
    baseline_observation_instances: int
    censored_observation_instances: int
    censored_fraction: float | None
    primary_evaluable_windows: int
    deletion_evaluable_windows: int
    paired_evaluable_windows: int
    mean_shift_minimum: float | None
    mean_shift_p50: float | None
    mean_shift_mean: float | None
    mean_shift_p95: float | None
    mean_shift_maximum: float | None
    deletion_mean_higher_fraction: float | None
    deletion_mean_equal_fraction: float | None
    deletion_mean_lower_fraction: float | None
    primary_supporting: int
    primary_contradicting: int
    primary_silent: int
    deletion_supporting: int
    deletion_contradicting: int
    deletion_silent: int
    changed_verdict_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CensoringSensitivityReport:
    schema_version: int
    snapshot_sha256: str
    study_start: str
    study_end_exclusive: str
    anchor_semantics: str
    anchor_count: int
    anchor_lat: float
    anchor_lon: float
    radius_km: float
    input_rows: int
    eligible_in_radius_rows: int
    quality_excluded_rows: int
    unit_assertion_passed: bool
    structurally_absent_sources: tuple[str, ...]
    rules: Mapping[str, Any]
    metrics: tuple[MetricCensoringSensitivity, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "snapshot_sha256": self.snapshot_sha256,
            "study_start": self.study_start,
            "study_end_exclusive": self.study_end_exclusive,
            "anchor_semantics": self.anchor_semantics,
            "anchor_count": self.anchor_count,
            "anchor_location": {
                "lat": self.anchor_lat,
                "lon": self.anchor_lon,
                "radius_km": self.radius_km,
            },
            "input_rows": self.input_rows,
            "eligible_in_radius_rows": self.eligible_in_radius_rows,
            "quality_excluded_rows": self.quality_excluded_rows,
            "unit_assertion_passed": self.unit_assertion_passed,
            "structurally_absent_sources": list(self.structurally_absent_sources),
            "rules": dict(self.rules),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def _is_relevant_metric(source: str, metric: str) -> bool:
    if source == "sentinel5p":
        return metric in SENTINEL_COLUMN_METRICS
    if source == "purpleair":
        return metric == "pm25"
    return source in RELEVANT_SOURCES and metric in GROUND_CONCENTRATION_METRICS


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=STUDY_START.tzinfo)
    return value.astimezone(STUDY_START.tzinfo)


def _parse_timestamp(raw: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid observation timestamp: {raw!r}") from exc
    return _ensure_utc(parsed)


def _percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def _nearest_event(
    prepared: Sequence[CensoringObservation],
    timestamps: Sequence[datetime],
    anchor: datetime,
) -> CensoringObservation | None:
    left = bisect.bisect_left(timestamps, anchor - WINDOW_HALF_WIDTH)
    right = bisect.bisect_right(timestamps, anchor + WINDOW_HALF_WIDTH)
    if left == right:
        return None
    return min(
        prepared[left:right],
        key=lambda row: (
            abs(row.timestamp - anchor),
            row.distance_km,
            row.entity_id,
        ),
    )


def _event_value(
    source: str,
    nearest: CensoringObservation | None,
    anchor: datetime,
    censor_limit: float | None,
) -> float | None:
    if nearest is None:
        return None
    dt_minutes = round(
        abs(nearest.timestamp - anchor).total_seconds() / 60.0,
        1,
    )
    if not assess_observation_age(source, dt_minutes).votes:
        return None
    if censor_limit is not None and nearest.value < censor_limit:
        return None
    return nearest.value


def _verdict(
    event_value: float | None,
    baseline_values: list[float],
) -> int:
    if event_value is None or len(baseline_values) < (
        DEFAULT_CONCENTRATION_TOLERANCE.min_baseline_points
    ):
        return SILENT
    verdict = qualitative_elevation_verdict(event_value, baseline_values)
    return SILENT if verdict is None else verdict


def _verdict_counts(verdicts: Sequence[int]) -> tuple[int, int, int]:
    return (
        sum(verdict == SUPPORTING for verdict in verdicts),
        sum(verdict == CONTRADICTING for verdict in verdicts),
        sum(verdict == SILENT for verdict in verdicts),
    )


def metric_censoring_sensitivity(
    *,
    source: str,
    metric: str,
    unit: str,
    observations: Sequence[CensoringObservation],
    anchors: Sequence[datetime],
) -> MetricCensoringSensitivity:
    """Compare declared substitution with pre-B15 deletion for one metric."""
    prepared = sorted(
        (
            CensoringObservation(
                entity_id=row.entity_id,
                timestamp=_ensure_utc(row.timestamp),
                value=float(row.value),
                distance_km=float(row.distance_km),
            )
            for row in observations
        ),
        key=lambda row: (row.timestamp, row.distance_km, row.entity_id),
    )
    timestamps = [row.timestamp for row in prepared]
    censor_limit = baseline_censor_limit(source, metric)
    baseline_instances = 0
    censored_instances = 0
    primary_evaluable = 0
    deletion_evaluable = 0
    mean_shifts: list[float] = []
    primary_verdicts: list[int] = []
    deletion_verdicts: list[int] = []
    gap = timedelta(hours=DEFAULT_CONCENTRATION_TOLERANCE.baseline_gap_h)
    min_points = DEFAULT_CONCENTRATION_TOLERANCE.min_baseline_points

    for raw_anchor in anchors:
        anchor = _ensure_utc(raw_anchor)
        left = bisect.bisect_left(timestamps, anchor - WINDOW_HALF_WIDTH)
        right = bisect.bisect_right(timestamps, anchor - gap)
        raw_baseline = [row.value for row in prepared[left:right]]
        baseline_instances += len(raw_baseline)
        if censor_limit is not None:
            censored_instances += sum(
                value < censor_limit for value in raw_baseline
            )
        primary = censor_baseline_values(
            raw_baseline,
            limit=censor_limit,
            strategy=BaselineCensoringStrategy.LIMIT_HALF,
        )
        deletion = censor_baseline_values(
            raw_baseline,
            limit=censor_limit,
            strategy=BaselineCensoringStrategy.DELETE,
        )
        primary_ready = len(primary) >= min_points
        deletion_ready = len(deletion) >= min_points
        primary_evaluable += int(primary_ready)
        deletion_evaluable += int(deletion_ready)
        if primary_ready and deletion_ready:
            mean_shifts.append(fmean(deletion) - fmean(primary))

        nearest = _nearest_event(prepared, timestamps, anchor)
        event_value = _event_value(source, nearest, anchor, censor_limit)
        primary_verdicts.append(_verdict(event_value, primary))
        deletion_verdicts.append(_verdict(event_value, deletion))

    sorted_shifts = sorted(mean_shifts)
    paired = len(sorted_shifts)
    primary_support, primary_contradict, primary_silent = _verdict_counts(
        primary_verdicts
    )
    deletion_support, deletion_contradict, deletion_silent = _verdict_counts(
        deletion_verdicts
    )
    higher = sum(shift > 0.0 and not math.isclose(shift, 0.0) for shift in mean_shifts)
    equal = sum(math.isclose(shift, 0.0, abs_tol=1e-12) for shift in mean_shifts)
    lower = paired - higher - equal

    return MetricCensoringSensitivity(
        source=source,
        metric=metric,
        unit=unit,
        censor_limit=censor_limit,
        replacement_value=(censor_limit / 2.0 if censor_limit is not None else None),
        observation_count=len(prepared),
        anchor_count=len(anchors),
        baseline_observation_instances=baseline_instances,
        censored_observation_instances=censored_instances,
        censored_fraction=(
            censored_instances / baseline_instances if baseline_instances else None
        ),
        primary_evaluable_windows=primary_evaluable,
        deletion_evaluable_windows=deletion_evaluable,
        paired_evaluable_windows=paired,
        mean_shift_minimum=sorted_shifts[0] if sorted_shifts else None,
        mean_shift_p50=_percentile(sorted_shifts, 0.50) if sorted_shifts else None,
        mean_shift_mean=fmean(sorted_shifts) if sorted_shifts else None,
        mean_shift_p95=_percentile(sorted_shifts, 0.95) if sorted_shifts else None,
        mean_shift_maximum=sorted_shifts[-1] if sorted_shifts else None,
        deletion_mean_higher_fraction=higher / paired if paired else None,
        deletion_mean_equal_fraction=equal / paired if paired else None,
        deletion_mean_lower_fraction=lower / paired if paired else None,
        primary_supporting=primary_support,
        primary_contradicting=primary_contradict,
        primary_silent=primary_silent,
        deletion_supporting=deletion_support,
        deletion_contradicting=deletion_contradict,
        deletion_silent=deletion_silent,
        changed_verdict_count=sum(
            primary != deletion
            for primary, deletion in zip(
                primary_verdicts,
                deletion_verdicts,
                strict=True,
            )
        ),
    )


def _snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_observations(
    connection: sqlite3.Connection,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> tuple[
    dict[tuple[str, str, str], list[CensoringObservation]],
    int,
    int,
    int,
]:
    placeholders = ",".join("?" for _ in RELEVANT_SOURCES)
    rows = connection.execute(
        f"""
        SELECT source, metric, source_entity_id, timestamp, value, unit, lat, lon
        FROM data_points
        WHERE source IN ({placeholders})
          AND timestamp >= ?
          AND timestamp < ?
        ORDER BY source, metric, timestamp, source_entity_id
        """,
        (
            *sorted(RELEVANT_SOURCES),
            STUDY_START.strftime("%Y-%m-%d %H:%M:%S"),
            STUDY_END_EXCLUSIVE.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    verified_openaq = verified_monitor_entity_ids()
    excluded_purpleair = excluded_purpleair_row_keys()
    grouped: dict[tuple[str, str, str], list[CensoringObservation]] = defaultdict(list)
    units: dict[tuple[str, str], set[str]] = defaultdict(set)
    input_rows = 0
    quality_excluded = 0
    in_radius = 0

    for source, metric, entity_id, raw_timestamp, value, unit, lat, lon in rows:
        source = str(source)
        metric = str(metric)
        if not _is_relevant_metric(source, metric):
            continue
        input_rows += 1
        timestamp = _parse_timestamp(raw_timestamp)
        entity_id = str(entity_id)
        if unit is None:
            raise ValueError(f"missing unit for {source}/{metric}")
        normalized_unit = str(unit).strip()
        if not normalized_unit:
            raise ValueError(f"missing unit for {source}/{metric}")
        units[(source, metric)].add(normalized_unit)
        if source == "openaq" and metric == "pm25" and entity_id not in verified_openaq:
            quality_excluded += 1
            continue
        if source == "purpleair" and metric == "pm25" and (
            entity_id,
            timestamp,
        ) in excluded_purpleair:
            quality_excluded += 1
            continue
        try:
            numeric_value = float(value)
            numeric_lat = float(lat)
            numeric_lon = float(lon)
        except (TypeError, ValueError):
            quality_excluded += 1
            continue
        if not all(
            math.isfinite(item) for item in (numeric_value, numeric_lat, numeric_lon)
        ):
            quality_excluded += 1
            continue
        if distance_km(anchor_lat, anchor_lon, numeric_lat, numeric_lon) > radius_km:
            continue
        in_radius += 1
        distance = distance_km(anchor_lat, anchor_lon, numeric_lat, numeric_lon)
        grouped[(source, metric, normalized_unit)].append(
            CensoringObservation(
                entity_id=entity_id,
                timestamp=timestamp,
                value=numeric_value,
                distance_km=distance,
            )
        )

    conflicts = [
        f"{source}/{metric}: {', '.join(sorted(metric_units))}"
        for (source, metric), metric_units in sorted(units.items())
        if len(metric_units) > 1
    ]
    if conflicts:
        raise ValueError("multiple units for " + "; ".join(conflicts))
    return dict(grouped), input_rows, in_radius, quality_excluded


def run_sensitivity(
    database_path: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> CensoringSensitivityReport:
    """Run B15 sensitivity with pre/post snapshot identity verification."""
    resolved = database_path.resolve()
    before_hash = _snapshot_sha256(resolved)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != {expected_sha256}"
        )
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{resolved}?mode=ro&immutable=1",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
        grouped, input_rows, in_radius, quality_excluded = _load_observations(
            connection,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            radius_km=radius_km,
        )
    finally:
        if connection is not None:
            connection.close()
        after_hash = _snapshot_sha256(resolved)
        if after_hash != expected_sha256:
            raise RuntimeError(
                f"snapshot SHA-256 mismatch after read: {after_hash} != {expected_sha256}"
            )

    anchors = candidate_centers(STUDY_START, STUDY_END_EXCLUSIVE)
    metrics = tuple(
        metric_censoring_sensitivity(
            source=source,
            metric=metric,
            unit=unit,
            observations=observations,
            anchors=anchors,
        )
        for (source, metric, unit), observations in sorted(grouped.items())
    )
    present_sources = {metric.source for metric in metrics}
    tolerance = DEFAULT_CONCENTRATION_TOLERANCE
    return CensoringSensitivityReport(
        schema_version=1,
        snapshot_sha256=after_hash,
        study_start=STUDY_START.isoformat(),
        study_end_exclusive=STUDY_END_EXCLUSIVE.isoformat(),
        anchor_semantics=(
            "B2/B8 UTC-hour centers; baseline endpoint-inclusive from "
            "anchor-36 h through anchor-3 h"
        ),
        anchor_count=len(anchors),
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
        input_rows=input_rows,
        eligible_in_radius_rows=in_radius,
        quality_excluded_rows=quality_excluded,
        unit_assertion_passed=True,
        structurally_absent_sources=tuple(
            sorted(set(RELEVANT_SOURCES) - present_sources)
        ),
        rules={
            "primary": BaselineCensoringStrategy.LIMIT_HALF.value,
            "alternative": BaselineCensoringStrategy.DELETE.value,
            "ground_so2_limit_ppb": tolerance.so2_ground_detection_limit_ppb,
            "sentinel_so2_limit_mol_m2": tolerance.so2_detection_limit_mol_m2,
            "other_ground_physical_bound": 0.0,
            "baseline_gap_hours": tolerance.baseline_gap_h,
            "minimum_baseline_points": tolerance.min_baseline_points,
            "elevated_sigma": tolerance.elevated_sigma,
            "so2_quantitative_exclusion_reason": (
                SO2_QUANTITATIVE_EXCLUSION_REASON
            ),
        },
        metrics=metrics,
    )


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _format_fraction(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def render_markdown(report: CensoringSensitivityReport) -> str:
    lines = [
        "Deletion - primary baseline and qualitative-verdict sensitivity:",
        "",
        "| Source | Metric | Unit | L | L/2 | Obs | Baseline instances | Censored | Fraction | Evaluable primary/delete/paired | Mean shift min/p50/mean/p95/max | Delete mean > / = / < | Primary S/C/0 | Delete S/C/0 | Changed |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in report.metrics:
        lines.append(
            "| "
            + " | ".join(
                (
                    metric.source,
                    metric.metric,
                    metric.unit,
                    _format_number(metric.censor_limit),
                    _format_number(metric.replacement_value),
                    str(metric.observation_count),
                    str(metric.baseline_observation_instances),
                    str(metric.censored_observation_instances),
                    _format_fraction(metric.censored_fraction),
                    f"{metric.primary_evaluable_windows}/{metric.deletion_evaluable_windows}/{metric.paired_evaluable_windows}",
                    "/".join(
                        _format_number(value)
                        for value in (
                            metric.mean_shift_minimum,
                            metric.mean_shift_p50,
                            metric.mean_shift_mean,
                            metric.mean_shift_p95,
                            metric.mean_shift_maximum,
                        )
                    ),
                    "/".join(
                        _format_fraction(value)
                        for value in (
                            metric.deletion_mean_higher_fraction,
                            metric.deletion_mean_equal_fraction,
                            metric.deletion_mean_lower_fraction,
                        )
                    ),
                    f"{metric.primary_supporting}/{metric.primary_contradicting}/{metric.primary_silent}",
                    f"{metric.deletion_supporting}/{metric.deletion_contradicting}/{metric.deletion_silent}",
                    str(metric.changed_verdict_count),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Structurally absent sources: "
            + (", ".join(report.structurally_absent_sources) or "none"),
            "Single-unit assertion: passed",
        ]
    )
    return "\n".join(lines)


def write_report(report: CensoringSensitivityReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def load_censoring_fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("censoring sensitivity fixture must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported censoring sensitivity fixture schema")
    if payload.get("snapshot_sha256") != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("censoring fixture does not match the locked snapshot")
    if payload.get("unit_assertion_passed") is not True:
        raise ValueError("censoring fixture did not pass the single-unit assertion")
    if not isinstance(payload.get("metrics"), list):
        raise ValueError("censoring sensitivity fixture metrics must be a list")
    return payload


def censoring_manifest_payload() -> dict[str, Any]:
    """Hash-link and embed the B15 declaration and sensitivity evidence."""
    payload = deepcopy(load_censoring_fixture())
    payload["artifact"] = FIXTURE_PATH.name
    payload["artifact_sha256"] = _snapshot_sha256(FIXTURE_PATH)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.censoring_sensitivity",
        description="Compare declared B15 baseline substitution with deletion.",
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--expected-sha256", default=LOCKED_SNAPSHOT_SHA256)
    parser.add_argument("--anchor-lat", type=float, default=settings.aeris_target_lat)
    parser.add_argument("--anchor-lon", type=float, default=settings.aeris_target_lon)
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_sensitivity(
        args.database,
        expected_sha256=args.expected_sha256,
        anchor_lat=args.anchor_lat,
        anchor_lon=args.anchor_lon,
        radius_km=args.radius_km,
    )
    if args.output is not None:
        write_report(report, args.output)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
