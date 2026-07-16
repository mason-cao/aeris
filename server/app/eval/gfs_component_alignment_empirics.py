"""Label-free A-9 GFS nearest-component alignment audit on frozen SQLite."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from app.collectors.geo import distance_km
from app.config import settings
from app.db.models import Anomaly, DataPoint
from app.detection.enrichment import EnrichmentConfig, build_cross_source_summary
from app.eval.calm_wind_empirics import (
    STUDY_END_EXCLUSIVE,
    STUDY_START,
    WINDOW_HALF_WIDTH,
    candidate_centers,
)
from app.llm import corroboration
from app.llm.observation_age import assess_observation_age
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256


@dataclass(frozen=True, order=True)
class GfsComponentObservation:
    metric: str
    entity_id: str
    timestamp: datetime
    value: float
    unit: str
    lat: float
    lon: float


@dataclass(frozen=True)
class GfsAnchorAssessment:
    both_values_present: bool
    both_fresh: bool
    timestamp_exact: bool
    timestamp_mismatch: bool
    missing_timestamp: bool
    malformed_timestamp: bool
    old_pair_eligible: bool
    new_pair_eligible: bool
    mismatch_minutes: float | None
    equal_timestamp_entity_mismatch: bool


@dataclass(frozen=True)
class NumericDistribution:
    count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    maximum: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "count": self.count,
            "minimum": self.minimum,
            "p50": self.p50,
            "p95": self.p95,
            "maximum": self.maximum,
        }


@dataclass(frozen=True)
class GfsComponentAlignmentReport:
    schema_version: int
    snapshot_sha256: str
    study_start: str
    study_end_exclusive: str
    anchor_semantics: str
    anchor_count: int
    anchor_lat: float
    anchor_lon: float
    radius_km: float
    input_row_count: int
    in_radius_row_count: int
    u_observation_count: int
    v_observation_count: int
    both_values_present_count: int
    both_fresh_count: int
    timestamp_exact_count: int
    timestamp_mismatch_count: int
    missing_timestamp_count: int
    malformed_timestamp_count: int
    old_pair_eligible_count: int
    new_pair_eligible_count: int
    changed_to_silent_count: int
    equal_timestamp_entity_mismatch_count: int
    mismatch_minutes: NumericDistribution

    def to_dict(self) -> dict[str, Any]:
        denominator = self.anchor_count

        def count_rate(count: int) -> dict[str, int | float | None]:
            return {
                "count": count,
                "rate": count / denominator if denominator else None,
            }

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
            "input_row_count": self.input_row_count,
            "in_radius_row_count": self.in_radius_row_count,
            "u_observation_count": self.u_observation_count,
            "v_observation_count": self.v_observation_count,
            "both_values_present": count_rate(self.both_values_present_count),
            "both_b8_fresh": count_rate(self.both_fresh_count),
            "timestamp_exact": count_rate(self.timestamp_exact_count),
            "timestamp_mismatch": count_rate(self.timestamp_mismatch_count),
            "missing_timestamp": count_rate(self.missing_timestamp_count),
            "malformed_timestamp": count_rate(self.malformed_timestamp_count),
            "old_pair_eligible": count_rate(self.old_pair_eligible_count),
            "new_pair_eligible": count_rate(self.new_pair_eligible_count),
            "changed_to_silent_count": self.changed_to_silent_count,
            "changed_to_silent_fraction": (
                self.changed_to_silent_count / denominator if denominator else None
            ),
            "equal_timestamp_entity_mismatch": count_rate(
                self.equal_timestamp_entity_mismatch_count
            ),
            "mismatch_minutes": self.mismatch_minutes.to_dict(),
            "gfs_age_gate_minutes": assess_observation_age(
                "noaa_gfs", 0.0
            ).gate_minutes,
        }


@dataclass(frozen=True)
class _SummaryPoint:
    timestamp: datetime
    lat: float
    lon: float
    metric: str
    value: float
    unit: str
    source: str
    source_entity_id: str


@dataclass(frozen=True)
class _SummaryAnchor:
    id: None
    timestamp: datetime
    lat: float
    lon: float
    metric: str
    source: str
    value: float
    severity: str


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(value: object) -> datetime:
    try:
        return _ensure_utc(
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        )
    except ValueError as exc:
        raise ValueError(f"invalid observation timestamp: {value!r}") from exc


def _normalized_observation(
    observation: GfsComponentObservation,
) -> GfsComponentObservation:
    if observation.metric not in {"u_10m", "v_10m"}:
        raise ValueError(f"unsupported GFS component metric: {observation.metric}")
    if not observation.entity_id:
        raise ValueError("GFS component entity_id must be non-empty")
    numeric = (
        float(observation.value),
        float(observation.lat),
        float(observation.lon),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("GFS component value and coordinates must be finite")
    return GfsComponentObservation(
        metric=observation.metric,
        entity_id=str(observation.entity_id),
        timestamp=_ensure_utc(observation.timestamp),
        value=numeric[0],
        unit=str(observation.unit),
        lat=numeric[1],
        lon=numeric[2],
    )


def _prepare_observations(
    observations: Sequence[GfsComponentObservation],
) -> tuple[GfsComponentObservation, ...]:
    normalized = tuple(sorted(_normalized_observation(row) for row in observations))
    seen: set[tuple[str, str, datetime]] = set()
    for row in normalized:
        key = (row.metric, row.entity_id, row.timestamp)
        if key in seen:
            raise ValueError(f"duplicate GFS component: {key}")
        seen.add(key)
    return normalized


def _summary_for_anchor(
    observations: Sequence[GfsComponentObservation],
    anchor_time: datetime,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> Mapping[str, Any]:
    anchor = _ensure_utc(anchor_time)
    window_start = anchor - WINDOW_HALF_WIDTH
    window_end = anchor + WINDOW_HALF_WIDTH
    points = [
        _SummaryPoint(
            timestamp=row.timestamp,
            lat=row.lat,
            lon=row.lon,
            metric=row.metric,
            value=row.value,
            unit=row.unit,
            source="noaa_gfs",
            source_entity_id=row.entity_id,
        )
        for row in observations
        if window_start <= row.timestamp <= window_end
    ]
    summary_anchor = _SummaryAnchor(
        id=None,
        timestamp=anchor,
        lat=anchor_lat,
        lon=anchor_lon,
        metric="label_free_empirical_anchor",
        source="label_free_empirical_anchor",
        value=0.0,
        severity="not_applicable",
    )
    return build_cross_source_summary(
        cast(Anomaly, summary_anchor),
        cast(Sequence[DataPoint], points),
        window_start=window_start,
        window_end=window_end,
        config=EnrichmentConfig(spatial_radius_km=radius_km),
    )


def _nearest_block(
    summary: Mapping[str, Any],
    metric: str,
) -> Mapping[str, Any] | None:
    nearest = (
        summary.get("sources", {})
        .get("noaa_gfs", {})
        .get("metrics", {})
        .get(metric, {})
        .get("nearest_in_time")
    )
    return nearest if isinstance(nearest, Mapping) else None


def _nearest_timestamp(
    nearest: Mapping[str, Any] | None,
) -> tuple[datetime | None, str]:
    raw_timestamp = nearest.get("t") if nearest is not None else None
    if raw_timestamp is None:
        return None, "missing"
    try:
        timestamp = datetime.fromisoformat(
            str(raw_timestamp).replace("Z", "+00:00")
        )
    except ValueError:
        return None, "malformed"
    return _ensure_utc(timestamp), "valid"


def _finite_value_present(nearest: Mapping[str, Any] | None) -> bool:
    if nearest is None or nearest.get("v") is None:
        return False
    try:
        return math.isfinite(float(nearest["v"]))
    except (TypeError, ValueError):
        return False


def _fresh(nearest: Mapping[str, Any] | None) -> bool:
    if not _finite_value_present(nearest):
        return False
    return assess_observation_age(
        "noaa_gfs",
        nearest.get("dt_minutes") if nearest is not None else None,
    ).votes


def _assess_summary(summary: Mapping[str, Any]) -> GfsAnchorAssessment:
    u_nearest = _nearest_block(summary, "u_10m")
    v_nearest = _nearest_block(summary, "v_10m")
    both_values = _finite_value_present(u_nearest) and _finite_value_present(
        v_nearest
    )
    both_fresh = _fresh(u_nearest) and _fresh(v_nearest)
    u_timestamp, u_status = _nearest_timestamp(u_nearest)
    v_timestamp, v_status = _nearest_timestamp(v_nearest)
    exact = (
        u_timestamp is not None
        and v_timestamp is not None
        and u_timestamp == v_timestamp
    )
    mismatch = (
        u_timestamp is not None
        and v_timestamp is not None
        and u_timestamp != v_timestamp
    )
    missing = "missing" in {u_status, v_status}
    malformed = not missing and "malformed" in {u_status, v_status}
    old_eligible = both_values and both_fresh
    new_eligible = old_eligible and exact
    mismatch_minutes = (
        abs((u_timestamp - v_timestamp).total_seconds()) / 60.0
        if mismatch and u_timestamp is not None and v_timestamp is not None
        else None
    )
    u_entity = u_nearest.get("entity_id") if u_nearest is not None else None
    v_entity = v_nearest.get("entity_id") if v_nearest is not None else None
    entity_mismatch = (
        exact
        and u_entity is not None
        and v_entity is not None
        and str(u_entity) != str(v_entity)
    )
    u_value, v_value, _ = corroboration._gfs_wind_components(summary)
    production_eligible = u_value is not None and v_value is not None
    if production_eligible != new_eligible:
        raise RuntimeError(
            "production GFS pair eligibility disagrees with A-9 audit: "
            f"{production_eligible} != {new_eligible}"
        )
    return GfsAnchorAssessment(
        both_values_present=both_values,
        both_fresh=both_fresh,
        timestamp_exact=exact,
        timestamp_mismatch=mismatch,
        missing_timestamp=missing,
        malformed_timestamp=malformed,
        old_pair_eligible=old_eligible,
        new_pair_eligible=new_eligible,
        mismatch_minutes=mismatch_minutes,
        equal_timestamp_entity_mismatch=entity_mismatch,
    )


def assess_anchor(
    observations: Sequence[GfsComponentObservation],
    anchor_time: datetime,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float = 50.0,
) -> GfsAnchorAssessment:
    """Run production enrichment and assess one A-9 anchor."""
    prepared = _prepare_observations(observations)
    summary = _summary_for_anchor(
        prepared,
        _ensure_utc(anchor_time),
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )
    return _assess_summary(summary)


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


def _distribution(values: Sequence[float]) -> NumericDistribution:
    ordered = sorted(values)
    return NumericDistribution(
        count=len(ordered),
        minimum=ordered[0] if ordered else None,
        p50=_percentile(ordered, 0.50) if ordered else None,
        p95=_percentile(ordered, 0.95) if ordered else None,
        maximum=ordered[-1] if ordered else None,
    )


def build_report(
    observations: Sequence[GfsComponentObservation],
    *,
    snapshot_sha256: str,
    anchors: Sequence[datetime] | None = None,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> GfsComponentAlignmentReport:
    """Aggregate old/new GFS event-pair eligibility over declared anchors."""
    prepared = _prepare_observations(observations)
    normalized_anchors = tuple(
        _ensure_utc(anchor)
        for anchor in (
            anchors
            if anchors is not None
            else candidate_centers(STUDY_START, STUDY_END_EXCLUSIVE)
        )
    )
    if len(set(normalized_anchors)) != len(normalized_anchors):
        raise ValueError("empirical anchors must be unique")

    chronological = tuple(sorted(prepared, key=lambda row: row.timestamp))
    timestamps = tuple(row.timestamp for row in chronological)
    assessments: list[GfsAnchorAssessment] = []
    for anchor in normalized_anchors:
        left = bisect.bisect_left(timestamps, anchor - WINDOW_HALF_WIDTH)
        right = bisect.bisect_right(timestamps, anchor + WINDOW_HALF_WIDTH)
        summary = _summary_for_anchor(
            chronological[left:right],
            anchor,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            radius_km=radius_km,
        )
        assessments.append(_assess_summary(summary))

    in_radius = tuple(
        row
        for row in prepared
        if distance_km(anchor_lat, anchor_lon, row.lat, row.lon) <= radius_km
    )
    return GfsComponentAlignmentReport(
        schema_version=1,
        snapshot_sha256=snapshot_sha256,
        study_start=STUDY_START.isoformat(),
        study_end_exclusive=STUDY_END_EXCLUSIVE.isoformat(),
        anchor_semantics=(
            "B2/B8 UTC-hour centers; centered endpoint-inclusive 72-hour "
            "context wholly inside the study interval"
        ),
        anchor_count=len(normalized_anchors),
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
        input_row_count=len(prepared),
        in_radius_row_count=len(in_radius),
        u_observation_count=sum(row.metric == "u_10m" for row in in_radius),
        v_observation_count=sum(row.metric == "v_10m" for row in in_radius),
        both_values_present_count=sum(
            row.both_values_present for row in assessments
        ),
        both_fresh_count=sum(row.both_fresh for row in assessments),
        timestamp_exact_count=sum(row.timestamp_exact for row in assessments),
        timestamp_mismatch_count=sum(
            row.timestamp_mismatch for row in assessments
        ),
        missing_timestamp_count=sum(row.missing_timestamp for row in assessments),
        malformed_timestamp_count=sum(
            row.malformed_timestamp for row in assessments
        ),
        old_pair_eligible_count=sum(
            row.old_pair_eligible for row in assessments
        ),
        new_pair_eligible_count=sum(
            row.new_pair_eligible for row in assessments
        ),
        changed_to_silent_count=sum(
            row.old_pair_eligible and not row.new_pair_eligible
            for row in assessments
        ),
        equal_timestamp_entity_mismatch_count=sum(
            row.equal_timestamp_entity_mismatch for row in assessments
        ),
        mismatch_minutes=_distribution(
            [
                row.mismatch_minutes
                for row in assessments
                if row.mismatch_minutes is not None
            ]
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
) -> list[GfsComponentObservation]:
    rows = connection.execute(
        """
        SELECT metric, source_entity_id, timestamp, value, unit, lat, lon
        FROM data_points
        WHERE source = 'noaa_gfs'
          AND metric IN ('u_10m', 'v_10m')
        ORDER BY metric, timestamp, source_entity_id
        """
    )
    observations: list[GfsComponentObservation] = []
    for metric, entity_id, raw_timestamp, value, unit, lat, lon in rows:
        timestamp = _parse_timestamp(raw_timestamp)
        if not STUDY_START <= timestamp < STUDY_END_EXCLUSIVE:
            continue
        observations.append(
            GfsComponentObservation(
                metric=str(metric),
                entity_id=str(entity_id),
                timestamp=timestamp,
                value=float(value),
                unit=str(unit),
                lat=float(lat),
                lon=float(lon),
            )
        )
    return observations


def run_empirics(
    database_path: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> GfsComponentAlignmentReport:
    """Read immutable SQLite and verify the locked hash before and after."""
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
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(
            f"file:{resolved}?mode=ro&immutable=1",
            uri=True,
        )
        connection.execute("PRAGMA query_only = ON")
        observations = _load_observations(connection)
    finally:
        if connection is not None:
            connection.close()
        after_hash = _snapshot_sha256(resolved)
        if after_hash != expected_sha256:
            raise RuntimeError(
                f"snapshot SHA-256 mismatch after read: {after_hash} != "
                f"{expected_sha256}"
            )
    return build_report(
        observations,
        snapshot_sha256=after_hash,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )


def _format_fraction(count: int, denominator: int) -> str:
    return "N/A" if not denominator else f"{count / denominator:.2%}"


def _format_number(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.6g}"


def render_markdown(report: GfsComponentAlignmentReport) -> str:
    denominator = report.anchor_count
    rows = (
        ("Both values present", report.both_values_present_count),
        ("Both B8-fresh", report.both_fresh_count),
        ("Exact timestamp", report.timestamp_exact_count),
        ("Timestamp mismatch", report.timestamp_mismatch_count),
        ("Missing timestamp", report.missing_timestamp_count),
        ("Malformed timestamp", report.malformed_timestamp_count),
        ("Old pair eligible", report.old_pair_eligible_count),
        ("New pair eligible", report.new_pair_eligible_count),
        ("Changed to silent", report.changed_to_silent_count),
        (
            "Equal timestamp / different entity",
            report.equal_timestamp_entity_mismatch_count,
        ),
    )
    lines = [
        "| Anchor diagnostic | Count | Fraction |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {name} | {count} | {_format_fraction(count, denominator)} |"
        for name, count in rows
    )
    distribution = report.mismatch_minutes
    lines.extend(
        (
            "",
            "| Distribution | min | p50 | p95 | max |",
            "|---|---:|---:|---:|---:|",
            "| Mismatch minutes | "
            f"{_format_number(distribution.minimum)} | "
            f"{_format_number(distribution.p50)} | "
            f"{_format_number(distribution.p95)} | "
            f"{_format_number(distribution.maximum)} |",
            "",
            "| Input diagnostic | Count |",
            "|---|---:|",
            f"| Raw u_10m rows | {report.u_observation_count} |",
            f"| Raw v_10m rows | {report.v_observation_count} |",
            f"| Relevant input rows | {report.input_row_count} |",
            f"| In-radius relevant rows | {report.in_radius_row_count} |",
            f"| Complete-window anchors | {report.anchor_count} |",
        )
    )
    return "\n".join(lines)


def write_report(report: GfsComponentAlignmentReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.gfs_component_alignment_empirics",
        description="Audit A-9 nearest-event GFS u/v timestamp alignment.",
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
    report = run_empirics(
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
