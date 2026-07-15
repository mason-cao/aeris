"""B8 label-free observation-age empirics on a read-only SQLite snapshot."""

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
from functools import lru_cache
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
from app.llm.observation_age import DEFAULT_OBSERVATION_AGE_GATES
from app.provenance.openaq_pm25 import verified_monitor_entity_ids
from app.provenance.purpleair_qc import (
    LOCKED_SNAPSHOT_SHA256,
    excluded_purpleair_row_keys,
)

HOURLY_SOURCES: Final = frozenset(
    {"openaq", "tceq", "purpleair", "asos", "openweather", "epa_aqs"}
)
HOURLY_STOP_FRACTION: Final = 0.20
FIXTURE_PATH: Final = Path(__file__).parent / "fixtures" / "observation_age_empirics.v1.json"


@dataclass(frozen=True)
class AgeObservation:
    entity_id: str
    timestamp: datetime
    lat: float
    lon: float


@dataclass(frozen=True)
class NearestObservationAge:
    entity_id: str
    timestamp: datetime
    distance_km: float
    dt_minutes: float


@dataclass(frozen=True)
class MetricAgeEmpirics:
    source: str
    metric: str
    gate_minutes: float
    observation_count: int
    anchor_count: int
    anchors_with_data: int
    anchors_without_data: int
    minimum: float | None
    p50: float | None
    p90: float | None
    p95: float | None
    p99: float | None
    maximum: float | None
    silenced_anchor_count: int
    silenced_fraction: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObservationAgeEmpiricalReport:
    schema_version: int
    snapshot_sha256: str
    study_start: str
    study_end_exclusive: str
    anchor_semantics: str
    anchor_count: int
    anchor_lat: float
    anchor_lon: float
    radius_km: float
    gates_minutes: Mapping[str, float]
    input_rows: int
    eligible_in_radius_rows: int
    quality_excluded_rows: int
    structurally_absent_sources: tuple[str, ...]
    metrics: tuple[MetricAgeEmpirics, ...]
    hourly_stop_fraction: float
    stop_rule_violations: tuple[str, ...]

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
            "gates_minutes": dict(sorted(self.gates_minutes.items())),
            "input_rows": self.input_rows,
            "eligible_in_radius_rows": self.eligible_in_radius_rows,
            "quality_excluded_rows": self.quality_excluded_rows,
            "structurally_absent_sources": list(self.structurally_absent_sources),
            "metrics": [metric.to_dict() for metric in self.metrics],
            "hourly_stop_fraction": self.hourly_stop_fraction,
            "stop_rule_violations": list(self.stop_rule_violations),
        }


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(raw: object) -> datetime:
    try:
        return _ensure_utc(datetime.fromisoformat(str(raw).replace("Z", "+00:00")))
    except ValueError as exc:
        raise ValueError(f"invalid observation timestamp: {raw!r}") from exc


def empirical_anchor_centers(
    study_start: datetime,
    study_end_exclusive: datetime,
) -> tuple[datetime, ...]:
    """Reuse the declared B2/P0b complete-window anchor construction."""
    return candidate_centers(study_start, study_end_exclusive)


def _prepared_observations(
    observations: Sequence[AgeObservation],
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> list[tuple[datetime, float, AgeObservation]]:
    prepared: list[tuple[datetime, float, AgeObservation]] = []
    for row in observations:
        timestamp = _ensure_utc(row.timestamp)
        if not all(math.isfinite(value) for value in (row.lat, row.lon)):
            continue
        distance = distance_km(anchor_lat, anchor_lon, row.lat, row.lon)
        if distance <= radius_km:
            prepared.append((timestamp, distance, row))
    prepared.sort(key=lambda item: (item[0], item[1], item[2].entity_id))
    return prepared


def _nearest_from_prepared(
    prepared: Sequence[tuple[datetime, float, AgeObservation]],
    timestamps: Sequence[datetime],
    anchor_time: datetime,
) -> NearestObservationAge | None:
    anchor = _ensure_utc(anchor_time)
    left = bisect.bisect_left(timestamps, anchor - WINDOW_HALF_WIDTH)
    right = bisect.bisect_right(timestamps, anchor + WINDOW_HALF_WIDTH)
    if left == right:
        return None
    timestamp, distance, row = min(
        prepared[left:right],
        key=lambda item: (abs(item[0] - anchor), item[1]),
    )
    return NearestObservationAge(
        entity_id=row.entity_id,
        timestamp=timestamp,
        distance_km=distance,
        dt_minutes=round(abs(timestamp - anchor).total_seconds() / 60.0, 1),
    )


def nearest_observation_age(
    observations: Sequence[AgeObservation],
    anchor_time: datetime,
    *,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> NearestObservationAge | None:
    """Nearest in-window observation, with enrichment-equivalent tie-breaking."""
    prepared = _prepared_observations(
        observations,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )
    return _nearest_from_prepared(
        prepared,
        [item[0] for item in prepared],
        anchor_time,
    )


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


def metric_age_empirics(
    *,
    source: str,
    metric: str,
    observations: Sequence[AgeObservation],
    anchors: Sequence[datetime],
    gate_minutes: float,
    anchor_lat: float,
    anchor_lon: float,
    radius_km: float,
) -> MetricAgeEmpirics:
    """Summarize nearest-observation ages for one source/metric."""
    prepared = _prepared_observations(
        observations,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
    )
    timestamps = [item[0] for item in prepared]
    ages = sorted(
        nearest.dt_minutes
        for anchor in anchors
        if (
            nearest := _nearest_from_prepared(prepared, timestamps, anchor)
        )
        is not None
    )
    anchor_count = len(anchors)
    with_data = len(ages)
    silenced = sum(age > gate_minutes for age in ages)
    return MetricAgeEmpirics(
        source=source,
        metric=metric,
        gate_minutes=gate_minutes,
        observation_count=len(prepared),
        anchor_count=anchor_count,
        anchors_with_data=with_data,
        anchors_without_data=anchor_count - with_data,
        minimum=ages[0] if ages else None,
        p50=_percentile(ages, 0.50) if ages else None,
        p90=_percentile(ages, 0.90) if ages else None,
        p95=_percentile(ages, 0.95) if ages else None,
        p99=_percentile(ages, 0.99) if ages else None,
        maximum=ages[-1] if ages else None,
        silenced_anchor_count=silenced,
        silenced_fraction=silenced / with_data if with_data else None,
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
) -> tuple[dict[tuple[str, str], list[AgeObservation]], int, int, int]:
    gates = DEFAULT_OBSERVATION_AGE_GATES.to_dict()
    placeholders = ",".join("?" for _ in gates)
    rows = connection.execute(
        f"""
        SELECT source, metric, source_entity_id, timestamp, value, lat, lon
        FROM data_points
        WHERE source IN ({placeholders})
          AND timestamp >= ?
          AND timestamp < ?
        ORDER BY source, metric, timestamp, source_entity_id
        """,
        (
            *sorted(gates),
            STUDY_START.strftime("%Y-%m-%d %H:%M:%S"),
            STUDY_END_EXCLUSIVE.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    verified_openaq = verified_monitor_entity_ids()
    excluded_purpleair = excluded_purpleair_row_keys()
    grouped: dict[tuple[str, str], list[AgeObservation]] = defaultdict(list)
    input_rows = 0
    quality_excluded = 0
    in_radius = 0
    for source, metric, entity_id, raw_timestamp, value, lat, lon in rows:
        input_rows += 1
        timestamp = _parse_timestamp(raw_timestamp)
        source = str(source)
        metric = str(metric)
        entity_id = str(entity_id)
        if source == "openaq" and metric == "pm25" and entity_id not in verified_openaq:
            quality_excluded += 1
            continue
        if source == "purpleair" and metric == "pm25" and (
            entity_id,
            timestamp,
        ) in excluded_purpleair:
            quality_excluded += 1
            continue
        numeric_values = (float(value), float(lat), float(lon))
        if not all(math.isfinite(item) for item in numeric_values):
            quality_excluded += 1
            continue
        if distance_km(anchor_lat, anchor_lon, float(lat), float(lon)) > radius_km:
            continue
        in_radius += 1
        grouped[(source, metric)].append(
            AgeObservation(
                entity_id=entity_id,
                timestamp=timestamp,
                lat=float(lat),
                lon=float(lon),
            )
        )
    return dict(grouped), input_rows, in_radius, quality_excluded


def run_empirics(
    database_path: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    anchor_lat: float = settings.aeris_target_lat,
    anchor_lon: float = settings.aeris_target_lon,
    radius_km: float = 50.0,
) -> ObservationAgeEmpiricalReport:
    """Run the B8 audit with pre/post snapshot hash verification."""
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

    anchors = empirical_anchor_centers(STUDY_START, STUDY_END_EXCLUSIVE)
    gates = DEFAULT_OBSERVATION_AGE_GATES.to_dict()
    metrics = tuple(
        metric_age_empirics(
            source=source,
            metric=metric,
            observations=observations,
            anchors=anchors,
            gate_minutes=gates[source],
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            radius_km=radius_km,
        )
        for (source, metric), observations in sorted(grouped.items())
    )
    present_sources = {metric.source for metric in metrics}
    absent_sources = tuple(sorted(set(gates) - present_sources))
    violations = tuple(
        f"{metric.source}/{metric.metric}={metric.silenced_fraction:.6f}"
        for metric in metrics
        if metric.source in HOURLY_SOURCES
        and metric.silenced_fraction is not None
        and metric.silenced_fraction > HOURLY_STOP_FRACTION
    )
    return ObservationAgeEmpiricalReport(
        schema_version=1,
        snapshot_sha256=after_hash,
        study_start=STUDY_START.isoformat(),
        study_end_exclusive=STUDY_END_EXCLUSIVE.isoformat(),
        anchor_semantics=(
            "B2/P0b UTC-hour centers with endpoint-inclusive +/-36 h contexts "
            "wholly inside the study interval"
        ),
        anchor_count=len(anchors),
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        radius_km=radius_km,
        gates_minutes=gates,
        input_rows=input_rows,
        eligible_in_radius_rows=in_radius,
        quality_excluded_rows=quality_excluded,
        structurally_absent_sources=absent_sources,
        metrics=metrics,
        hourly_stop_fraction=HOURLY_STOP_FRACTION,
        stop_rule_violations=violations,
    )


def _format_number(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.6g}"


def _format_fraction(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.2%}"


def render_markdown(report: ObservationAgeEmpiricalReport) -> str:
    lines = [
        "| Source | Metric | Gate min | Anchors | With data | No data | dt min | p50 | p90 | p95 | p99 | max | Silenced | Fraction |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for metric in report.metrics:
        lines.append(
            "| "
            + " | ".join(
                (
                    metric.source,
                    metric.metric,
                    _format_number(metric.gate_minutes),
                    str(metric.anchor_count),
                    str(metric.anchors_with_data),
                    str(metric.anchors_without_data),
                    _format_number(metric.minimum),
                    _format_number(metric.p50),
                    _format_number(metric.p90),
                    _format_number(metric.p95),
                    _format_number(metric.p99),
                    _format_number(metric.maximum),
                    str(metric.silenced_anchor_count),
                    _format_fraction(metric.silenced_fraction),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Structurally absent sources: "
            + (", ".join(report.structurally_absent_sources) or "none"),
            "Hourly >20% stop-rule violations: "
            + (", ".join(report.stop_rule_violations) or "none"),
        ]
    )
    return "\n".join(lines)


def write_report(report: ObservationAgeEmpiricalReport, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def load_observation_age_fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("observation-age empirical fixture must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported observation-age empirical fixture schema")
    if payload.get("snapshot_sha256") != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("observation-age fixture does not match the locked snapshot")
    if payload.get("gates_minutes") != DEFAULT_OBSERVATION_AGE_GATES.to_dict():
        raise ValueError("observation-age fixture gates differ from the declaration")
    if payload.get("anchor_count") != len(
        empirical_anchor_centers(STUDY_START, STUDY_END_EXCLUSIVE)
    ):
        raise ValueError("observation-age fixture anchor count differs from B2/P0b")
    if payload.get("stop_rule_violations") != []:
        raise ValueError("observation-age fixture has unresolved hourly stop violations")
    if not isinstance(payload.get("metrics"), list):
        raise ValueError("observation-age fixture metrics must be a list")
    return payload


def observation_age_manifest_payload() -> dict[str, Any]:
    """Hash-link and embed the declared B8 gates and empirical table."""
    payload = deepcopy(load_observation_age_fixture())
    payload["artifact"] = FIXTURE_PATH.name
    payload["artifact_sha256"] = _snapshot_sha256(FIXTURE_PATH)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.observation_age_empirics",
        description="Compute B8 nearest-observation age distributions without labels.",
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
    return 2 if report.stop_rule_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
