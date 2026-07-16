"""Label-free A-3 secondary-formation coupling audit on frozen SQLite."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import fmean
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
from app.llm.corroboration import (
    CONTRADICTING,
    DEFAULT_SECONDARY_TOLERANCE,
    SILENT,
    SUPPORTING,
    score_secondary_formation,
)
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256


GROUND_SOURCES: Final = ("openaq", "tceq", "epa_aqs")
RELEVANT_SOURCE_METRICS: Final = frozenset(
    {
        *(
            source_metric
            for source in GROUND_SOURCES
            for source_metric in ((source, "ozone"), (source, "no2"))
        ),
        ("openweather", "cloud_cover"),
    }
)
LOCAL_UTC_OFFSET: Final = timedelta(hours=-5)


@dataclass(frozen=True, order=True)
class FormationObservation:
    source: str
    metric: str
    entity_id: str
    timestamp: datetime
    value: float
    unit: str
    lat: float
    lon: float


@dataclass(frozen=True)
class FormationAnchorAssessment:
    ozone_point_count: int
    no2_point_count: int
    cloud_mean: float | None
    lag_verdict: int
    former_insolation_verdict: int
    conditional_insolation_verdict: int


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
class SecondaryFormationEmpiricalReport:
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
    ozone_point_counts: PointCountDistribution
    no2_point_counts: PointCountDistribution
    cloud_available_count: int
    lag_outcomes: OutcomeCounts
    former_insolation_outcomes: OutcomeCounts
    conditional_insolation_outcomes: OutcomeCounts
    former_cloud_only_votes_silenced: int
    former_supporting_votes_silenced: int
    former_contradicting_votes_silenced: int

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
            "relevant_source_metrics": [
                {"source": source, "metric": metric}
                for source, metric in sorted(RELEVANT_SOURCE_METRICS)
            ],
            "tolerance": asdict(DEFAULT_SECONDARY_TOLERANCE),
            "ozone_point_counts": self.ozone_point_counts.to_dict(),
            "no2_point_counts": self.no2_point_counts.to_dict(),
            "cloud_available_count": self.cloud_available_count,
            "cloud_available_fraction": (
                self.cloud_available_count / self.anchor_count
                if self.anchor_count
                else None
            ),
            "lag_outcomes": self.lag_outcomes.to_dict(),
            "former_insolation_outcomes": self.former_insolation_outcomes.to_dict(),
            "conditional_insolation_outcomes": (
                self.conditional_insolation_outcomes.to_dict()
            ),
            "former_cloud_only_votes_silenced": (
                self.former_cloud_only_votes_silenced
            ),
            "former_supporting_votes_silenced": (
                self.former_supporting_votes_silenced
            ),
            "former_contradicting_votes_silenced": (
                self.former_contradicting_votes_silenced
            ),
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


def _is_relevant(observation: FormationObservation) -> bool:
    return (observation.source, observation.metric) in RELEVANT_SOURCE_METRICS


def _normalized_observation(
    observation: FormationObservation,
) -> FormationObservation:
    if not observation.entity_id:
        raise ValueError("relevant observation entity_id must be non-empty")
    numeric = (
        float(observation.value),
        float(observation.lat),
        float(observation.lon),
    )
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError("relevant observation value and coordinates must be finite")
    return FormationObservation(
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
    observations: Sequence[FormationObservation],
) -> tuple[FormationObservation, ...]:
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
    observations: Sequence[FormationObservation],
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
    config = EnrichmentConfig(spatial_radius_km=radius_km)
    return build_cross_source_summary(
        cast(Anomaly, summary_anchor),
        cast(Sequence[DataPoint], points),
        window_start=window_start,
        window_end=window_end,
        config=config,
    )


def _metric_block(
    summary: Mapping[str, Any],
    source: str,
    metric: str,
) -> Mapping[str, Any] | None:
    block = (
        summary.get("sources", {})
        .get(source, {})
        .get("metrics", {})
        .get(metric)
    )
    return block if isinstance(block, Mapping) else None


def _local_day_values(
    block: Mapping[str, Any] | None,
    anchor_time: datetime,
) -> list[float]:
    if block is None:
        return []
    local_day = (_ensure_utc(anchor_time) + LOCAL_UTC_OFFSET).date()
    values: list[float] = []
    for entity in block.get("entities", []):
        for raw_timestamp, raw_value in entity.get("series", []):
            timestamp = _parse_timestamp(raw_timestamp)
            if (timestamp + LOCAL_UTC_OFFSET).date() == local_day:
                values.append(float(raw_value))
    return values


def _cloud_mean(
    summary: Mapping[str, Any],
    anchor_time: datetime,
) -> float | None:
    block = _metric_block(summary, "openweather", "cloud_cover")
    local_values = _local_day_values(block, anchor_time)
    if local_values:
        return fmean(local_values)
    if block is None:
        return None
    raw_mean = block.get("value_range", {}).get("mean")
    return float(raw_mean) if raw_mean is not None else None


def _assess_summary(
    summary: Mapping[str, Any],
    anchor_time: datetime,
) -> FormationAnchorAssessment:
    ozone_count = sum(
        len(_local_day_values(_metric_block(summary, source, "ozone"), anchor_time))
        for source in GROUND_SOURCES
    )
    no2_count = sum(
        len(_local_day_values(_metric_block(summary, source, "no2"), anchor_time))
        for source in GROUND_SOURCES
    )
    cloud_mean = _cloud_mean(summary, anchor_time)
    verdicts, _ = score_secondary_formation("", summary)
    lag_votes = {
        verdicts.get(source, SILENT)
        for source in GROUND_SOURCES
        if verdicts.get(source, SILENT) != SILENT
    }
    if len(lag_votes) > 1:
        raise ValueError(f"ground lag sources disagree: {sorted(lag_votes)}")
    lag_verdict = next(iter(lag_votes), SILENT)
    if cloud_mean is None:
        former_verdict = SILENT
    elif cloud_mean <= DEFAULT_SECONDARY_TOLERANCE.clear_sky_max_cloud_pct:
        former_verdict = SUPPORTING
    else:
        former_verdict = CONTRADICTING
    return FormationAnchorAssessment(
        ozone_point_count=ozone_count,
        no2_point_count=no2_count,
        cloud_mean=cloud_mean,
        lag_verdict=lag_verdict,
        former_insolation_verdict=former_verdict,
        conditional_insolation_verdict=verdicts.get("openweather", SILENT),
    )


def assess_anchor(
    observations: Sequence[FormationObservation],
    anchor_time: datetime,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float = 50.0,
) -> FormationAnchorAssessment:
    """Run production enrichment and the A-3 scorer for one raw anchor."""
    prepared = _prepare_observations(observations)
    anchor = _ensure_utc(anchor_time)
    return _assess_prepared(
        prepared,
        anchor,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )


def _assess_prepared(
    observations: Sequence[FormationObservation],
    anchor_time: datetime,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> FormationAnchorAssessment:
    summary = _summary_for_anchor(
        observations,
        anchor_time,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )
    return _assess_summary(summary, anchor_time)


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
    floor = DEFAULT_SECONDARY_TOLERANCE.min_points
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
    observations: Sequence[FormationObservation],
    *,
    snapshot_sha256: str,
    anchors: Sequence[datetime] | None = None,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> SecondaryFormationEmpiricalReport:
    """Apply A-3 at every declared label-free complete-window anchor."""
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
    assessments: list[FormationAnchorAssessment] = []
    for anchor in normalized_anchors:
        left = bisect.bisect_left(timestamps, anchor - WINDOW_HALF_WIDTH)
        right = bisect.bisect_right(timestamps, anchor + WINDOW_HALF_WIDTH)
        assessments.append(
            _assess_prepared(
                chronological[left:right],
                anchor,
                anchor_lat=anchor_lat,
                anchor_lon=anchor_lon,
                radius_km=radius_km,
            )
        )
    lag_verdicts = [result.lag_verdict for result in assessments]
    former_verdicts = [result.former_insolation_verdict for result in assessments]
    conditional_verdicts = [
        result.conditional_insolation_verdict for result in assessments
    ]
    removed = [
        result
        for result in assessments
        if result.lag_verdict == SILENT
        and result.former_insolation_verdict != SILENT
        and result.conditional_insolation_verdict == SILENT
    ]
    return SecondaryFormationEmpiricalReport(
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
        in_radius_row_count=sum(
            distance_km(anchor_lat, anchor_lon, row.lat, row.lon) <= radius_km
            for row in prepared
        ),
        ozone_point_counts=_point_count_distribution(
            [result.ozone_point_count for result in assessments]
        ),
        no2_point_counts=_point_count_distribution(
            [result.no2_point_count for result in assessments]
        ),
        cloud_available_count=sum(
            result.cloud_mean is not None for result in assessments
        ),
        lag_outcomes=_outcome_counts(lag_verdicts),
        former_insolation_outcomes=_outcome_counts(former_verdicts),
        conditional_insolation_outcomes=_outcome_counts(conditional_verdicts),
        former_cloud_only_votes_silenced=len(removed),
        former_supporting_votes_silenced=sum(
            result.former_insolation_verdict == SUPPORTING for result in removed
        ),
        former_contradicting_votes_silenced=sum(
            result.former_insolation_verdict == CONTRADICTING for result in removed
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
) -> list[FormationObservation]:
    rows = connection.execute(
        """
        SELECT source, metric, source_entity_id, timestamp, value, unit, lat, lon
        FROM data_points
        WHERE (
            source IN ('openaq', 'tceq', 'epa_aqs')
            AND metric IN ('ozone', 'no2')
        ) OR (
            source = 'openweather'
            AND metric = 'cloud_cover'
        )
        ORDER BY source, metric, timestamp, source_entity_id
        """
    )
    observations: list[FormationObservation] = []
    for source, metric, entity_id, raw_timestamp, value, unit, lat, lon in rows:
        timestamp = _parse_timestamp(raw_timestamp)
        if not STUDY_START <= timestamp < STUDY_END_EXCLUSIVE:
            continue
        observations.append(
            FormationObservation(
                source=str(source),
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
) -> SecondaryFormationEmpiricalReport:
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
    return tuple(
        f"{payload[name]['count']} ({_format_fraction(payload[name]['rate'])})"
        for name in ("supporting", "contradicting", "silent")
    )  # type: ignore[return-value]


def render_markdown(report: SecondaryFormationEmpiricalReport) -> str:
    lines = [
        "| Local-day pooled leg | n min | p50 | p95 | n max | Below n=3 | Fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, distribution in (
        ("O3", report.ozone_point_counts),
        ("NO2", report.no2_point_counts),
    ):
        lines.append(
            f"| {name} | {_format_point(distribution.minimum)} | "
            f"{_format_point(distribution.p50)} | "
            f"{_format_point(distribution.p95)} | "
            f"{_format_point(distribution.maximum)} | "
            f"{distribution.below_floor_count} | "
            f"{_format_fraction(distribution.below_floor_fraction)} |"
        )
    lines.extend(
        (
            "",
            "| Leg / rule | Support | Contradict | Silent |",
            "|---|---:|---:|---:|",
        )
    )
    for name, outcomes in (
        ("Ground O3/NO2 lag", report.lag_outcomes),
        ("Former unconditional insolation", report.former_insolation_outcomes),
        ("Conditional insolation", report.conditional_insolation_outcomes),
    ):
        support, contradict, silent = _outcome_cells(outcomes)
        lines.append(f"| {name} | {support} | {contradict} | {silent} |")
    lines.extend(
        (
            "",
            "| Diagnostic | Count |",
            "|---|---:|",
            f"| Relevant input rows | {report.input_row_count} |",
            f"| In-radius relevant rows | {report.in_radius_row_count} |",
            f"| Complete-window anchors | {report.anchor_count} |",
            f"| Anchors with cloud evidence | {report.cloud_available_count} |",
            "| Former cloud-only votes silenced | "
            f"{report.former_cloud_only_votes_silenced} |",
            "| Former supporting votes silenced | "
            f"{report.former_supporting_votes_silenced} |",
            "| Former contradicting votes silenced | "
            f"{report.former_contradicting_votes_silenced} |",
        )
    )
    return "\n".join(lines)


def write_report(
    report: SecondaryFormationEmpiricalReport,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.secondary_formation_empirics",
        description="Audit A-3 lag-conditioned insolation without labels.",
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
