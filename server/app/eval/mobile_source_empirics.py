"""Label-free A-4 mobile-source local-day audit on frozen SQLite."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

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
from app.llm.corroboration import (
    CONTRADICTING,
    DEFAULT_SOURCE_TYPE_TOLERANCE,
    SILENT,
    SUPPORTING,
    score_emissions_source_type,
)
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    verified_monitor_entity_ids,
)
from app.provenance.purpleair_qc import purpleair_reading_is_eligible


GROUND_SOURCES: Final = ("openaq", "tceq", "epa_aqs")
RECOGNIZED_METRICS: Final = (
    "bc",
    "co",
    "no2",
    "ozone",
    "pm10",
    "pm25",
    "so2",
)
MOBILE_CLAIMS: Final = {
    "bc": "black carbon mobile traffic",
    "co": "carbon monoxide mobile traffic",
    "no2": "NO2 mobile traffic",
    "ozone": "ozone mobile traffic",
    "pm10": "PM10 mobile traffic",
    "pm25": "PM2.5 mobile traffic",
    "so2": "SO2 mobile traffic",
}
VERDICT_NAMES: Final = {
    CONTRADICTING: "contradicting",
    SILENT: "silent",
    SUPPORTING: "supporting",
}


@dataclass(frozen=True, order=True)
class MobileObservation:
    source: str
    metric: str
    entity_id: str
    timestamp: datetime
    value: float
    unit: str
    lat: float
    lon: float


@dataclass(frozen=True)
class MobilePairAssessment:
    anomaly_day_point_count: int
    whole_window_verdict: int
    anomaly_day_verdict: int


@dataclass(frozen=True)
class PointCountDistribution:
    frequency: tuple[tuple[int, int], ...]
    minimum: int | None
    p50: float | None
    p95: float | None
    maximum: int | None
    below_floor_count: int
    below_floor_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frequency": [
                {"point_count": point_count, "anchor_count": anchor_count}
                for point_count, anchor_count in self.frequency
            ],
            "minimum": self.minimum,
            "p50": self.p50,
            "p95": self.p95,
            "maximum": self.maximum,
            "below_floor_count": self.below_floor_count,
            "below_floor_fraction": self.below_floor_fraction,
        }


@dataclass(frozen=True)
class OutcomeCounts:
    supporting_count: int
    contradicting_count: int
    silent_count: int

    @property
    def total(self) -> int:
        return self.supporting_count + self.contradicting_count + self.silent_count

    def to_dict(self) -> dict[str, Any]:
        denominator = self.total

        def rate(count: int) -> float | None:
            return count / denominator if denominator else None

        return {
            "supporting": {
                "count": self.supporting_count,
                "rate": rate(self.supporting_count),
            },
            "contradicting": {
                "count": self.contradicting_count,
                "rate": rate(self.contradicting_count),
            },
            "silent": {
                "count": self.silent_count,
                "rate": rate(self.silent_count),
            },
        }


@dataclass(frozen=True)
class MobilePairEmpirics:
    source: str
    metric: str
    eligible_observation_count: int
    anomaly_day_point_counts: PointCountDistribution
    whole_window_outcomes: OutcomeCounts
    anomaly_day_outcomes: OutcomeCounts
    changed_verdict_count: int
    changed_verdict_fraction: float | None
    transition_counts: tuple[tuple[int, int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metric": self.metric,
            "eligible_observation_count": self.eligible_observation_count,
            "anomaly_day_point_counts": self.anomaly_day_point_counts.to_dict(),
            "whole_window_outcomes": self.whole_window_outcomes.to_dict(),
            "anomaly_day_outcomes": self.anomaly_day_outcomes.to_dict(),
            "changed_verdict_count": self.changed_verdict_count,
            "changed_verdict_fraction": self.changed_verdict_fraction,
            "changed_transitions": [
                {
                    "whole_window": VERDICT_NAMES[whole_window],
                    "anomaly_day": VERDICT_NAMES[anomaly_day],
                    "anchor_count": count,
                }
                for whole_window, anomaly_day, count in self.transition_counts
            ],
        }


@dataclass(frozen=True)
class MobileSourceEmpiricalReport:
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
    quality_eligible_row_count: int
    quality_excluded_row_count: int
    pairs: tuple[MobilePairEmpirics, ...]

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
            "input_row_count": self.input_row_count,
            "in_radius_row_count": self.in_radius_row_count,
            "quality_eligible_row_count": self.quality_eligible_row_count,
            "quality_excluded_row_count": self.quality_excluded_row_count,
            "tolerance": asdict(DEFAULT_SOURCE_TYPE_TOLERANCE),
            "pairs": [pair.to_dict() for pair in self.pairs],
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


def _is_relevant(observation: MobileObservation) -> bool:
    if observation.metric not in RECOGNIZED_METRICS:
        return False
    if observation.source == "purpleair":
        return observation.metric == "pm25"
    return observation.source in GROUND_SOURCES


def _quality_eligible(observation: MobileObservation) -> bool:
    if observation.source == "openaq" and observation.metric == "pm25":
        return observation.entity_id in verified_monitor_entity_ids()
    if observation.source == "purpleair" and observation.metric == "pm25":
        return purpleair_reading_is_eligible(
            observation.entity_id,
            observation.timestamp,
        )
    return True


def _normalized_observation(observation: MobileObservation) -> MobileObservation:
    if not observation.entity_id:
        raise ValueError("relevant observation entity_id must be non-empty")
    numeric = (
        float(observation.value),
        float(observation.lat),
        float(observation.lon),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("relevant observation value and coordinates must be finite")
    return MobileObservation(
        source=str(observation.source),
        metric=str(observation.metric),
        entity_id=str(observation.entity_id),
        timestamp=_ensure_utc(observation.timestamp),
        value=numeric[0],
        unit=str(observation.unit),
        lat=numeric[1],
        lon=numeric[2],
    )


def _prepare_observations(
    observations: Sequence[MobileObservation],
) -> tuple[MobileObservation, ...]:
    relevant = tuple(
        sorted(
            _normalized_observation(row)
            for row in observations
            if _is_relevant(row)
        )
    )
    seen: set[tuple[str, str, str, datetime]] = set()
    for row in relevant:
        key = (row.source, row.metric, row.entity_id, row.timestamp)
        if key in seen:
            raise ValueError(f"duplicate relevant observation: {key}")
        seen.add(key)
    return relevant


def _summary_for_anchor(
    observations: Sequence[MobileObservation],
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
            source=row.source,
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


def _mobile_verdict(series: Sequence[tuple[datetime, float]]) -> int:
    tolerance = DEFAULT_SOURCE_TYPE_TOLERANCE
    if len(series) < tolerance.min_points:
        return SILENT
    peak_timestamp, _ = max(series, key=lambda pair: pair[1])
    local_hour = (peak_timestamp.hour + corroboration._LOCAL_UTC_OFFSET_H) % 24
    return (
        SUPPORTING
        if tolerance.morning_start_h <= local_hour < tolerance.morning_end_h
        else CONTRADICTING
    )


def _assess_summary_pair(
    summary: Mapping[str, Any],
    anchor_time: datetime,
    *,
    source: str,
    metric: str,
) -> MobilePairAssessment:
    block = corroboration._metric_block(summary, source, metric)
    whole_window = corroboration._pooled_series(block)
    anomaly_day = corroboration._local_day_slice(whole_window, anchor_time)
    expected_new = _mobile_verdict(anomaly_day)
    verdicts, _ = score_emissions_source_type(MOBILE_CLAIMS[metric], summary)
    production_new = verdicts.get(source, SILENT)
    if production_new != expected_new:
        raise RuntimeError(
            "production mobile scorer disagrees with the declared local-day "
            f"construction for {source}/{metric}: {production_new} != "
            f"{expected_new}"
        )
    return MobilePairAssessment(
        anomaly_day_point_count=len(anomaly_day),
        whole_window_verdict=_mobile_verdict(whole_window),
        anomaly_day_verdict=production_new,
    )


def assess_pair(
    observations: Sequence[MobileObservation],
    anchor_time: datetime,
    *,
    source: str,
    metric: str,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float = 50.0,
) -> MobilePairAssessment:
    """Run production enrichment and compare one A-4 source/metric pair."""
    if metric not in MOBILE_CLAIMS:
        raise ValueError(f"unsupported mobile-source metric: {metric}")
    prepared = _prepare_observations(observations)
    anchor = _ensure_utc(anchor_time)
    summary = _summary_for_anchor(
        prepared,
        anchor,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )
    return _assess_summary_pair(
        summary,
        anchor,
        source=source,
        metric=metric,
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


def _point_count_distribution(values: Sequence[int]) -> PointCountDistribution:
    ordered = sorted(values)
    floor = DEFAULT_SOURCE_TYPE_TOLERANCE.min_points
    below = sum(value < floor for value in ordered)
    return PointCountDistribution(
        frequency=tuple(sorted(Counter(ordered).items())),
        minimum=ordered[0] if ordered else None,
        p50=_percentile(ordered, 0.50) if ordered else None,
        p95=_percentile(ordered, 0.95) if ordered else None,
        maximum=ordered[-1] if ordered else None,
        below_floor_count=below,
        below_floor_fraction=below / len(ordered) if ordered else None,
    )


def _outcome_counts(verdicts: Sequence[int]) -> OutcomeCounts:
    return OutcomeCounts(
        supporting_count=sum(verdict == SUPPORTING for verdict in verdicts),
        contradicting_count=sum(verdict == CONTRADICTING for verdict in verdicts),
        silent_count=sum(verdict == SILENT for verdict in verdicts),
    )


def build_report(
    observations: Sequence[MobileObservation],
    *,
    snapshot_sha256: str,
    anchors: Sequence[datetime] | None = None,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> MobileSourceEmpiricalReport:
    """Compare whole-window and anomaly-day mobile verdicts at all anchors."""
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

    in_radius = tuple(
        row
        for row in prepared
        if distance_km(anchor_lat, anchor_lon, row.lat, row.lon) <= radius_km
    )
    quality_eligible = tuple(row for row in in_radius if _quality_eligible(row))
    pairs = tuple(sorted({(row.source, row.metric) for row in quality_eligible}))
    eligible_counts = Counter((row.source, row.metric) for row in quality_eligible)

    chronological = tuple(sorted(prepared, key=lambda row: row.timestamp))
    timestamps = tuple(row.timestamp for row in chronological)
    by_pair: dict[tuple[str, str], list[MobilePairAssessment]] = defaultdict(list)
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
        for source, metric in pairs:
            by_pair[(source, metric)].append(
                _assess_summary_pair(
                    summary,
                    anchor,
                    source=source,
                    metric=metric,
                )
            )

    pair_reports: list[MobilePairEmpirics] = []
    for source, metric in pairs:
        assessments = by_pair[(source, metric)]
        whole_window = [row.whole_window_verdict for row in assessments]
        anomaly_day = [row.anomaly_day_verdict for row in assessments]
        transitions = Counter(
            (old, new)
            for old, new in zip(whole_window, anomaly_day, strict=True)
            if old != new
        )
        changed = sum(transitions.values())
        pair_reports.append(
            MobilePairEmpirics(
                source=source,
                metric=metric,
                eligible_observation_count=eligible_counts[(source, metric)],
                anomaly_day_point_counts=_point_count_distribution(
                    [row.anomaly_day_point_count for row in assessments]
                ),
                whole_window_outcomes=_outcome_counts(whole_window),
                anomaly_day_outcomes=_outcome_counts(anomaly_day),
                changed_verdict_count=changed,
                changed_verdict_fraction=(
                    changed / len(assessments) if assessments else None
                ),
                transition_counts=tuple(
                    (old, new, count)
                    for (old, new), count in sorted(transitions.items())
                ),
            )
        )

    return MobileSourceEmpiricalReport(
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
        quality_eligible_row_count=len(quality_eligible),
        quality_excluded_row_count=len(in_radius) - len(quality_eligible),
        pairs=tuple(pair_reports),
    )


def _snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_observations(connection: sqlite3.Connection) -> list[MobileObservation]:
    source_placeholders = ",".join("?" for _ in (*GROUND_SOURCES, "purpleair"))
    metric_placeholders = ",".join("?" for _ in RECOGNIZED_METRICS)
    rows = connection.execute(
        f"""
        SELECT source, metric, source_entity_id, timestamp, value, unit, lat, lon
        FROM data_points
        WHERE source IN ({source_placeholders})
          AND metric IN ({metric_placeholders})
        ORDER BY source, metric, timestamp, source_entity_id
        """,
        (*GROUND_SOURCES, "purpleair", *RECOGNIZED_METRICS),
    )
    observations: list[MobileObservation] = []
    for source, metric, entity_id, raw_timestamp, value, unit, lat, lon in rows:
        timestamp = _parse_timestamp(raw_timestamp)
        if not STUDY_START <= timestamp < STUDY_END_EXCLUSIVE:
            continue
        observation = MobileObservation(
            source=str(source),
            metric=str(metric),
            entity_id=str(entity_id),
            timestamp=timestamp,
            value=float(value),
            unit=str(unit),
            lat=float(lat),
            lon=float(lon),
        )
        if _is_relevant(observation):
            observations.append(observation)
    return observations


def run_empirics(
    database_path: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> MobileSourceEmpiricalReport:
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


def _format_fraction(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def _format_point(value: int | float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int) or value.is_integer():
        return str(int(value))
    return f"{value:.2f}"


def _outcome_cells(outcomes: OutcomeCounts) -> tuple[str, str, str]:
    payload = outcomes.to_dict()
    cells = [
        f"{payload[name]['count']} ({_format_fraction(payload[name]['rate'])})"
        for name in ("supporting", "contradicting", "silent")
    ]
    return cells[0], cells[1], cells[2]


def render_markdown(report: MobileSourceEmpiricalReport) -> str:
    lines = [
        "| Source | Metric | n min | p50 | p95 | n max | Below n=4 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for pair in report.pairs:
        counts = pair.anomaly_day_point_counts
        lines.append(
            f"| {pair.source} | {pair.metric} | "
            f"{_format_point(counts.minimum)} | {_format_point(counts.p50)} | "
            f"{_format_point(counts.p95)} | {_format_point(counts.maximum)} | "
            f"{counts.below_floor_count} "
            f"({_format_fraction(counts.below_floor_fraction)}) |"
        )
    for pair in report.pairs:
        lines.extend(
            (
                "",
                f"#### {pair.source}/{pair.metric}",
                "",
                "| Rule | Support | Contradict | Silent |",
                "|---|---:|---:|---:|",
            )
        )
        for name, outcomes in (
            ("Whole window", pair.whole_window_outcomes),
            ("Anomaly local day", pair.anomaly_day_outcomes),
        ):
            support, contradict, silent = _outcome_cells(outcomes)
            lines.append(f"| {name} | {support} | {contradict} | {silent} |")
        lines.append(
            f"Changed verdicts: {pair.changed_verdict_count} "
            f"({_format_fraction(pair.changed_verdict_fraction)})"
        )
    lines.extend(
        (
            "",
            "| Diagnostic | Count |",
            "|---|---:|",
            f"| Relevant input rows | {report.input_row_count} |",
            f"| In-radius relevant rows | {report.in_radius_row_count} |",
            f"| Quality-eligible rows | {report.quality_eligible_row_count} |",
            f"| B6/B7-excluded rows | {report.quality_excluded_row_count} |",
            f"| Complete-window anchors | {report.anchor_count} |",
            f"| Present source/metric pairs | {len(report.pairs)} |",
        )
    )
    return "\n".join(lines)


def write_report(report: MobileSourceEmpiricalReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.mobile_source_empirics",
        description="Audit A-4 anomaly-local-day mobile-source behavior.",
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
