"""B17 label-free station-matched baseline diagnostics on frozen SQLite."""

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
from datetime import UTC, datetime, timedelta
from pathlib import Path
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
BASELINE_LOCALITY: Final = "nearest_event_entity"
FIXTURE_PATH: Final = (
    Path(__file__).parent / "fixtures" / "baseline_locality_empirics.v1.json"
)


@dataclass(frozen=True)
class BaselineObservation:
    entity_id: str
    timestamp: datetime
    value: float
    distance_km: float


@dataclass(frozen=True)
class MetricBaselineLocality:
    source: str
    metric: str
    unit: str
    observation_count: int
    anchor_count: int
    event_eligible_count: int
    pooled_evaluable_count: int
    matched_evaluable_count: int
    pooled_supporting: int
    pooled_contradicting: int
    pooled_silent: int
    matched_supporting: int
    matched_contradicting: int
    matched_silent: int
    pooled_support_rate: float | None
    pooled_contradict_rate: float | None
    pooled_silent_rate: float | None
    matched_support_rate: float | None
    matched_contradict_rate: float | None
    matched_silent_rate: float | None
    changed_verdict_count: int
    matched_baseline_n_minimum: int | None
    matched_baseline_n_p50: float | None
    matched_baseline_n_p95: float | None
    matched_baseline_n_maximum: int | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineLocalityReport:
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
    metrics: tuple[MetricBaselineLocality, ...]

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
            "structurally_absent_sources": list(
                self.structurally_absent_sources
            ),
            "rules": dict(self.rules),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(raw: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid observation timestamp: {raw!r}") from exc
    return _ensure_utc(parsed)


def _normalized_observation(
    observation: BaselineObservation,
) -> BaselineObservation:
    entity_id = str(observation.entity_id)
    if not entity_id:
        raise ValueError("baseline observation entity_id must be non-empty")
    value = float(observation.value)
    distance = float(observation.distance_km)
    if not math.isfinite(value):
        raise ValueError("baseline observation values must be finite")
    if not math.isfinite(distance) or distance < 0.0:
        raise ValueError(
            "baseline observation distances must be finite and nonnegative"
        )
    return BaselineObservation(
        entity_id=entity_id,
        timestamp=_ensure_utc(observation.timestamp),
        value=value,
        distance_km=distance,
    )


def _percentile(sorted_values: Sequence[int], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def _nearest_event(
    prepared: Sequence[BaselineObservation],
    timestamps: Sequence[datetime],
    anchor: datetime,
) -> BaselineObservation | None:
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
            row.timestamp,
        ),
    )


def _event_value(
    source: str,
    nearest: BaselineObservation | None,
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


def _baseline_values(
    observations: Sequence[BaselineObservation],
    *,
    entity_id: str | None,
    censor_limit: float | None,
) -> list[float]:
    raw_values = [
        row.value
        for row in observations
        if entity_id is None or row.entity_id == entity_id
    ]
    return censor_baseline_values(
        raw_values,
        limit=censor_limit,
        strategy=BaselineCensoringStrategy.LIMIT_HALF,
    )


def _verdict(event_value: float | None, baseline_values: list[float]) -> int:
    if event_value is None or len(baseline_values) < (
        DEFAULT_CONCENTRATION_TOLERANCE.min_baseline_points
    ):
        return SILENT
    verdict = qualitative_elevation_verdict(event_value, baseline_values)
    return SILENT if verdict is None else verdict


def _counts(verdicts: Sequence[int]) -> tuple[int, int, int]:
    return (
        sum(verdict == SUPPORTING for verdict in verdicts),
        sum(verdict == CONTRADICTING for verdict in verdicts),
        sum(verdict == SILENT for verdict in verdicts),
    )


def metric_baseline_locality(
    *,
    source: str,
    metric: str,
    unit: str,
    observations: Sequence[BaselineObservation],
    anchors: Sequence[datetime],
) -> MetricBaselineLocality:
    """Compare old pooled and declared matched baselines for one metric."""
    prepared = sorted(
        (_normalized_observation(row) for row in observations),
        key=lambda row: (row.timestamp, row.distance_km, row.entity_id),
    )
    seen: set[tuple[str, datetime]] = set()
    for row in prepared:
        key = (row.entity_id, row.timestamp)
        if key in seen:
            raise ValueError(f"duplicate baseline observation: {key}")
        seen.add(key)
    normalized_anchors = tuple(_ensure_utc(anchor) for anchor in anchors)
    timestamps = [row.timestamp for row in prepared]
    gap = timedelta(
        hours=DEFAULT_CONCENTRATION_TOLERANCE.baseline_gap_h
    )
    min_points = DEFAULT_CONCENTRATION_TOLERANCE.min_baseline_points
    censor_limit = baseline_censor_limit(source, metric)
    event_eligible = 0
    pooled_evaluable = 0
    matched_evaluable = 0
    matched_ns: list[int] = []
    pooled_verdicts: list[int] = []
    matched_verdicts: list[int] = []

    for anchor in normalized_anchors:
        left = bisect.bisect_left(
            timestamps,
            anchor - WINDOW_HALF_WIDTH,
        )
        baseline_right = bisect.bisect_right(timestamps, anchor - gap)
        baseline_rows = prepared[left:baseline_right]
        nearest = _nearest_event(prepared, timestamps, anchor)
        event_value = _event_value(source, nearest, anchor, censor_limit)
        event_eligible += int(event_value is not None)

        pooled = _baseline_values(
            baseline_rows,
            entity_id=None,
            censor_limit=censor_limit,
        )
        matched = _baseline_values(
            baseline_rows,
            entity_id=nearest.entity_id if nearest is not None else "",
            censor_limit=censor_limit,
        )
        pooled_evaluable += int(len(pooled) >= min_points)
        matched_evaluable += int(len(matched) >= min_points)
        matched_ns.append(len(matched))
        pooled_verdicts.append(_verdict(event_value, pooled))
        matched_verdicts.append(_verdict(event_value, matched))

    pooled_support, pooled_contradict, pooled_silent = _counts(
        pooled_verdicts
    )
    matched_support, matched_contradict, matched_silent = _counts(
        matched_verdicts
    )
    denominator = len(normalized_anchors)

    def rate(count: int) -> float | None:
        return count / denominator if denominator else None

    sorted_ns = sorted(matched_ns)
    return MetricBaselineLocality(
        source=source,
        metric=metric,
        unit=unit,
        observation_count=len(prepared),
        anchor_count=denominator,
        event_eligible_count=event_eligible,
        pooled_evaluable_count=pooled_evaluable,
        matched_evaluable_count=matched_evaluable,
        pooled_supporting=pooled_support,
        pooled_contradicting=pooled_contradict,
        pooled_silent=pooled_silent,
        matched_supporting=matched_support,
        matched_contradicting=matched_contradict,
        matched_silent=matched_silent,
        pooled_support_rate=rate(pooled_support),
        pooled_contradict_rate=rate(pooled_contradict),
        pooled_silent_rate=rate(pooled_silent),
        matched_support_rate=rate(matched_support),
        matched_contradict_rate=rate(matched_contradict),
        matched_silent_rate=rate(matched_silent),
        changed_verdict_count=sum(
            pooled != matched
            for pooled, matched in zip(
                pooled_verdicts,
                matched_verdicts,
                strict=True,
            )
        ),
        matched_baseline_n_minimum=sorted_ns[0] if sorted_ns else None,
        matched_baseline_n_p50=(
            _percentile(sorted_ns, 0.50) if sorted_ns else None
        ),
        matched_baseline_n_p95=(
            _percentile(sorted_ns, 0.95) if sorted_ns else None
        ),
        matched_baseline_n_maximum=sorted_ns[-1] if sorted_ns else None,
    )


def build_report(
    grouped: Mapping[
        tuple[str, str, str],
        Sequence[BaselineObservation],
    ],
    *,
    snapshot_sha256: str,
    anchors: Sequence[datetime],
    input_rows: int,
    eligible_in_radius_rows: int,
    quality_excluded_rows: int,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> BaselineLocalityReport:
    """Build the deterministic B17 report from eligible observations."""
    metrics = tuple(
        metric_baseline_locality(
            source=source,
            metric=metric,
            unit=unit,
            observations=observations,
            anchors=anchors,
        )
        for (source, metric, unit), observations in sorted(grouped.items())
    )
    tolerance = DEFAULT_CONCENTRATION_TOLERANCE
    present_sources = {metric.source for metric in metrics}
    return BaselineLocalityReport(
        schema_version=1,
        snapshot_sha256=snapshot_sha256,
        study_start=STUDY_START.isoformat(),
        study_end_exclusive=STUDY_END_EXCLUSIVE.isoformat(),
        anchor_semantics=(
            "B2/B8 UTC-hour centers; centered endpoint-inclusive 72-hour "
            "window; baseline through anchor-3 hours"
        ),
        anchor_count=len(anchors),
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
        input_rows=input_rows,
        eligible_in_radius_rows=eligible_in_radius_rows,
        quality_excluded_rows=quality_excluded_rows,
        unit_assertion_passed=True,
        structurally_absent_sources=tuple(
            sorted(RELEVANT_SOURCES - present_sources)
        ),
        rules={
            "baseline_locality": BASELINE_LOCALITY,
            "comparison": "all_entity_network_pool",
            "pooled_fallback": False,
            "event_tie_break": "absolute_time_then_distance",
            "event_age_gates": "app.llm.observation_age.SOURCE_MAX_AGE_MINUTES",
            "censoring": BaselineCensoringStrategy.LIMIT_HALF.value,
            "baseline_gap_hours": tolerance.baseline_gap_h,
            "minimum_baseline_points": tolerance.min_baseline_points,
            "elevated_sigma": tolerance.elevated_sigma,
            "rates_denominator": "all_anchors",
            "trigger_channel_demotion_applied": False,
        },
        metrics=metrics,
    )


def _is_relevant_metric(source: str, metric: str) -> bool:
    if source == "sentinel5p":
        return metric in SENTINEL_COLUMN_METRICS
    if source == "purpleair":
        return metric == "pm25"
    return source in RELEVANT_SOURCES and metric in GROUND_CONCENTRATION_METRICS


def _load_observations(
    connection: sqlite3.Connection,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> tuple[
    dict[tuple[str, str, str], list[BaselineObservation]],
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
    grouped: dict[tuple[str, str, str], list[BaselineObservation]] = (
        defaultdict(list)
    )
    units: dict[tuple[str, str], set[str]] = defaultdict(set)
    input_rows = 0
    quality_excluded = 0
    in_radius = 0

    for source, metric, entity_id, raw_timestamp, value, unit, lat, lon in rows:
        normalized_source = str(source)
        normalized_metric = str(metric)
        if not _is_relevant_metric(normalized_source, normalized_metric):
            continue
        input_rows += 1
        timestamp = _parse_timestamp(raw_timestamp)
        normalized_entity_id = str(entity_id)
        if unit is None or not str(unit).strip():
            raise ValueError(
                f"missing unit for {normalized_source}/{normalized_metric}"
            )
        normalized_unit = str(unit).strip()
        units[(normalized_source, normalized_metric)].add(normalized_unit)
        if (
            normalized_source == "openaq"
            and normalized_metric == "pm25"
            and normalized_entity_id not in verified_openaq
        ):
            quality_excluded += 1
            continue
        if (
            normalized_source == "purpleair"
            and normalized_metric == "pm25"
            and (normalized_entity_id, timestamp) in excluded_purpleair
        ):
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
            math.isfinite(item)
            for item in (numeric_value, numeric_lat, numeric_lon)
        ):
            quality_excluded += 1
            continue
        distance = distance_km(
            anchor_lat,
            anchor_lon,
            numeric_lat,
            numeric_lon,
        )
        if distance > radius_km:
            continue
        in_radius += 1
        grouped[(normalized_source, normalized_metric, normalized_unit)].append(
            BaselineObservation(
                entity_id=normalized_entity_id,
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_comparison(
    database_path: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> BaselineLocalityReport:
    """Run B17 with immutable pre/post snapshot identity verification."""
    resolved = database_path.resolve()
    before_hash = _sha256_file(resolved)
    if before_hash != expected_sha256:
        raise ValueError(
            "snapshot SHA-256 mismatch before read: "
            f"{before_hash} != {expected_sha256}"
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
        after_hash = _sha256_file(resolved)
        if after_hash != expected_sha256:
            raise RuntimeError(
                "snapshot SHA-256 mismatch after read: "
                f"{after_hash} != {expected_sha256}"
            )

    return build_report(
        grouped,
        snapshot_sha256=after_hash,
        anchors=candidate_centers(STUDY_START, STUDY_END_EXCLUSIVE),
        input_rows=input_rows,
        eligible_in_radius_rows=in_radius,
        quality_excluded_rows=quality_excluded,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )


def _format_fraction(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _format_number(value: float | int | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def render_markdown(report: BaselineLocalityReport) -> str:
    lines = [
        "Network-pooled versus nearest-event-entity baseline comparison:",
        "",
        "| Source | Metric | Unit | Obs | Anchors | Event eligible | Baseline evaluable pooled/matched | Pooled S/C/0 (support rate) | Matched S/C/0 (support rate) | Changed | Matched n min/p50/p95/max |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in report.metrics:
        lines.append(
            "| "
            + " | ".join(
                (
                    metric.source,
                    metric.metric,
                    metric.unit,
                    str(metric.observation_count),
                    str(metric.anchor_count),
                    str(metric.event_eligible_count),
                    f"{metric.pooled_evaluable_count}/{metric.matched_evaluable_count}",
                    (
                        f"{metric.pooled_supporting}/"
                        f"{metric.pooled_contradicting}/"
                        f"{metric.pooled_silent} "
                        f"({_format_fraction(metric.pooled_support_rate)})"
                    ),
                    (
                        f"{metric.matched_supporting}/"
                        f"{metric.matched_contradicting}/"
                        f"{metric.matched_silent} "
                        f"({_format_fraction(metric.matched_support_rate)})"
                    ),
                    str(metric.changed_verdict_count),
                    "/".join(
                        _format_number(value)
                        for value in (
                            metric.matched_baseline_n_minimum,
                            metric.matched_baseline_n_p50,
                            metric.matched_baseline_n_p95,
                            metric.matched_baseline_n_maximum,
                        )
                    ),
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


def write_report(report: BaselineLocalityReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_baseline_locality_fixture(
    path: Path = FIXTURE_PATH,
    *,
    expected_snapshot_sha256: str = LOCKED_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline locality fixture must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported baseline locality fixture schema")
    if payload.get("snapshot_sha256") != expected_snapshot_sha256:
        raise ValueError("baseline locality fixture snapshot mismatch")
    rules = payload.get("rules")
    if not isinstance(rules, dict) or rules.get("baseline_locality") != (
        BASELINE_LOCALITY
    ):
        raise ValueError("baseline locality fixture has the wrong rule")
    if payload.get("unit_assertion_passed") is not True:
        raise ValueError("baseline locality fixture failed its unit assertion")
    if not isinstance(payload.get("metrics"), list):
        raise ValueError("baseline locality fixture metrics must be a list")
    return payload


def baseline_locality_manifest_payload(
    path: Path = FIXTURE_PATH,
    *,
    expected_snapshot_sha256: str = LOCKED_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    """Hash-link and embed the B17 declaration and empirical evidence."""
    payload = deepcopy(
        load_baseline_locality_fixture(
            path,
            expected_snapshot_sha256=expected_snapshot_sha256,
        )
    )
    payload["artifact"] = path.name
    payload["artifact_sha256"] = _sha256_file(path)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.baseline_locality_empirics",
        description=(
            "Compare pooled and nearest-event-entity concentration baselines."
        ),
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--expected-sha256", default=LOCKED_SNAPSHOT_SHA256)
    parser.add_argument(
        "--anchor-lat",
        type=float,
        default=settings.aeris_target_lat,
    )
    parser.add_argument(
        "--anchor-lon",
        type=float,
        default=settings.aeris_target_lon,
    )
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_comparison(
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
