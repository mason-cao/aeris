"""B7 time-aware PurpleAir eligibility and frozen-snapshot QC artifact."""

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
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from statistics import fmean, median
from typing import Any, Final

from app.collectors.geo import distance_km

LOCKED_SNAPSHOT_SHA256: Final = (
    "8ec0bfacec592b50a31aafb9e80f61e886cfb48da030d595e89bdc0f53f9ea81"
)
STUDY_START: Final = datetime(2026, 6, 1, tzinfo=timezone.utc)
STUDY_END_EXCLUSIVE: Final = datetime(2026, 7, 13, tzinfo=timezone.utc)
JUNE_END_EXCLUSIVE: Final = datetime(2026, 7, 1, tzinfo=timezone.utc)
TARGET_LAT: Final = 29.7604
TARGET_LON: Final = -95.3698
TARGET_RADIUS_KM: Final = 50.0
FIXTURE_PATH: Final = (
    Path(__file__).parent / "fixtures" / "purpleair_time_aware_qc.v1.json"
)
AUDITED_SENSOR_IDS: Final = ("165203", "194469", "288282")

RowKey = tuple[str, datetime]
WindowKey = tuple[str, datetime]


@dataclass(frozen=True)
class PurpleAirQCParameters:
    """Mason-declared B7 parameters; changes require a dated protocol revision."""

    window_hours: int = 24
    center_step_hours: int = 1
    candidate_min_observations: int = 6
    peer_min_observations: int = 6
    minimum_peer_sensors: int = 10
    segment_ratio_threshold: float = 5.0
    segment_absolute_floor_ug_m3: float = 20.0
    saturation_ug_m3: float = 500.0
    network_extreme_median_ug_m3: float = 100.0


DEFAULT_QC_PARAMETERS: Final = PurpleAirQCParameters()


@dataclass(frozen=True)
class PurpleAirReading:
    entity_id: str
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class PurpleAirQCSegment:
    entity_id: str
    reason: str
    decision_start: datetime
    decision_end_inclusive: datetime
    covered_start: datetime
    covered_end_exclusive: datetime
    excluded_row_count: int


@dataclass(frozen=True)
class PurpleAirQCResult:
    parameters: PurpleAirQCParameters
    study_start: datetime
    study_end_exclusive: datetime
    exclusion_reasons: Mapping[RowKey, tuple[str, ...]]
    primary_excluded_windows: frozenset[WindowKey]
    unevaluated_windows: Mapping[WindowKey, str]
    total_window_count: int
    evaluated_window_count: int
    segments: tuple[PurpleAirQCSegment, ...]

    @property
    def excluded_row_count(self) -> int:
        return len(self.exclusion_reasons)

    @property
    def unevaluated_window_count(self) -> int:
        return len(self.unevaluated_windows)

    def is_eligible(self, entity_id: str, timestamp: datetime) -> bool:
        return (str(entity_id), _ensure_utc(timestamp)) not in self.exclusion_reasons


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _iso(timestamp: datetime) -> str:
    return _ensure_utc(timestamp).isoformat()


def _floor_hour(timestamp: datetime) -> datetime:
    return _ensure_utc(timestamp).replace(minute=0, second=0, microsecond=0)


def _hour_centers(start: datetime, end_exclusive: datetime) -> list[datetime]:
    center = _floor_hour(start)
    if center < start:
        center += timedelta(hours=1)
    centers: list[datetime] = []
    while center < end_exclusive:
        centers.append(center)
        center += timedelta(hours=1)
    return centers


def _window_bounds(
    center: datetime,
    start: datetime,
    end_exclusive: datetime,
    parameters: PurpleAirQCParameters,
) -> tuple[datetime, datetime]:
    half_window = timedelta(hours=parameters.window_hours / 2.0)
    return max(start, center - half_window), min(end_exclusive, center + half_window)


def _series_slice(
    readings: Sequence[PurpleAirReading],
    timestamps: Sequence[datetime],
    start: datetime,
    end_exclusive: datetime,
) -> Sequence[PurpleAirReading]:
    lower = bisect.bisect_left(timestamps, start)
    upper = bisect.bisect_left(timestamps, end_exclusive)
    return readings[lower:upper]


def _contiguous_groups(timestamps: Sequence[datetime]) -> list[list[datetime]]:
    groups: list[list[datetime]] = []
    for timestamp in sorted(set(timestamps)):
        if not groups or timestamp - groups[-1][-1] > timedelta(hours=1):
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    return groups


def evaluate_purpleair_qc(
    readings: Sequence[PurpleAirReading],
    study_start: datetime,
    study_end_exclusive: datetime,
    *,
    parameters: PurpleAirQCParameters = DEFAULT_QC_PARAMETERS,
) -> PurpleAirQCResult:
    """Evaluate the declared persistent-segment and saturation rules."""
    start = _ensure_utc(study_start)
    end_exclusive = _ensure_utc(study_end_exclusive)
    if end_exclusive <= start:
        raise ValueError("PurpleAir QC study interval must be non-empty")
    if parameters.window_hours <= 0 or parameters.window_hours % 2 != 0:
        raise ValueError("PurpleAir QC window_hours must be a positive even integer")

    normalized = sorted(
        (
            PurpleAirReading(
                entity_id=str(reading.entity_id),
                timestamp=_ensure_utc(reading.timestamp),
                value=float(reading.value),
            )
            for reading in readings
            if start <= _ensure_utc(reading.timestamp) < end_exclusive
            and math.isfinite(float(reading.value))
        ),
        key=lambda reading: (reading.entity_id, reading.timestamp),
    )
    by_entity: dict[str, list[PurpleAirReading]] = defaultdict(list)
    for reading in normalized:
        by_entity[reading.entity_id].append(reading)
    entity_ids = sorted(by_entity)
    entity_timestamps = {
        entity_id: [reading.timestamp for reading in entity_readings]
        for entity_id, entity_readings in by_entity.items()
    }
    centers = _hour_centers(start, end_exclusive)

    window_statistics: dict[datetime, dict[str, tuple[int, float]]] = {}
    for center in centers:
        window_start, window_end = _window_bounds(
            center, start, end_exclusive, parameters
        )
        statistics: dict[str, tuple[int, float]] = {}
        for entity_id in entity_ids:
            window_readings = _series_slice(
                by_entity[entity_id],
                entity_timestamps[entity_id],
                window_start,
                window_end,
            )
            if window_readings:
                statistics[entity_id] = (
                    len(window_readings),
                    median([reading.value for reading in window_readings])
                )
        window_statistics[center] = statistics

    primary_excluded_windows: set[WindowKey] = set()
    unevaluated_windows: dict[WindowKey, str] = {}
    evaluated_window_count = 0
    for center in centers:
        statistics = window_statistics[center]
        for entity_id in entity_ids:
            key = (entity_id, center)
            sensor_statistics = statistics.get(entity_id)
            if (
                sensor_statistics is None
                or sensor_statistics[0] < parameters.candidate_min_observations
            ):
                unevaluated_windows[key] = "candidate_observation_floor"
                continue
            sensor_median = sensor_statistics[1]
            peers = [
                peer_statistics[1]
                for peer_id, peer_statistics in statistics.items()
                if peer_id != entity_id
                and peer_statistics[0] >= parameters.peer_min_observations
            ]
            if len(peers) < parameters.minimum_peer_sensors:
                unevaluated_windows[key] = "peer_sensor_floor"
                continue
            network_median = float(median(peers))
            if not math.isfinite(network_median):
                unevaluated_windows[key] = "nonfinite_network_median"
                continue
            evaluated_window_count += 1
            ratio_exceeded = (
                network_median == 0.0
                or sensor_median / network_median
                >= parameters.segment_ratio_threshold
            )
            if (
                ratio_exceeded
                and sensor_median >= parameters.segment_absolute_floor_ug_m3
            ):
                primary_excluded_windows.add(key)

    reasons: dict[RowKey, set[str]] = defaultdict(set)
    primary_centers_by_entity: dict[str, list[datetime]] = defaultdict(list)
    for entity_id, center in sorted(primary_excluded_windows):
        primary_centers_by_entity[entity_id].append(center)
        window_start, window_end = _window_bounds(
            center, start, end_exclusive, parameters
        )
        for reading in _series_slice(
            by_entity[entity_id],
            entity_timestamps[entity_id],
            window_start,
            window_end,
        ):
            reasons[(entity_id, reading.timestamp)].add("persistent_segment")

    by_hour_entity: dict[datetime, dict[str, list[PurpleAirReading]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for reading in normalized:
        by_hour_entity[_floor_hour(reading.timestamp)][reading.entity_id].append(
            reading
        )
    saturation_times_by_entity: dict[str, list[datetime]] = defaultdict(list)
    for hour, hourly_entities in sorted(by_hour_entity.items()):
        hourly_medians = {
            entity_id: float(median([reading.value for reading in entity_readings]))
            for entity_id, entity_readings in hourly_entities.items()
        }
        for entity_id, entity_readings in hourly_entities.items():
            peers = [
                peer_median
                for peer_id, peer_median in hourly_medians.items()
                if peer_id != entity_id
            ]
            network_extreme = (
                len(peers) >= parameters.minimum_peer_sensors
                and float(median(peers))
                >= parameters.network_extreme_median_ug_m3
            )
            for reading in entity_readings:
                if (
                    reading.value >= parameters.saturation_ug_m3
                    and not network_extreme
                ):
                    reasons[(entity_id, reading.timestamp)].add("saturation")
                    saturation_times_by_entity[entity_id].append(reading.timestamp)

    segments: list[PurpleAirQCSegment] = []
    half_window = timedelta(hours=parameters.window_hours / 2.0)
    for entity_id, primary_centers in sorted(primary_centers_by_entity.items()):
        for group in _contiguous_groups(primary_centers):
            covered_start = max(start, group[0] - half_window)
            covered_end = min(end_exclusive, group[-1] + half_window)
            excluded_row_count = sum(
                1
                for reading in by_entity[entity_id]
                if covered_start <= reading.timestamp < covered_end
                and "persistent_segment"
                in reasons.get((entity_id, reading.timestamp), set())
            )
            segments.append(
                PurpleAirQCSegment(
                    entity_id=entity_id,
                    reason="persistent_segment",
                    decision_start=group[0],
                    decision_end_inclusive=group[-1],
                    covered_start=covered_start,
                    covered_end_exclusive=covered_end,
                    excluded_row_count=excluded_row_count,
                )
            )
    for entity_id, saturation_times in sorted(saturation_times_by_entity.items()):
        for group in _contiguous_groups(
            [_floor_hour(timestamp) for timestamp in saturation_times]
        ):
            row_times = [
                timestamp
                for timestamp in saturation_times
                if group[0] <= _floor_hour(timestamp) <= group[-1]
            ]
            segments.append(
                PurpleAirQCSegment(
                    entity_id=entity_id,
                    reason="saturation",
                    decision_start=group[0],
                    decision_end_inclusive=group[-1],
                    covered_start=min(row_times),
                    covered_end_exclusive=max(row_times) + timedelta(microseconds=1),
                    excluded_row_count=len(row_times),
                )
            )
    segments.sort(
        key=lambda segment: (
            segment.entity_id,
            segment.covered_start,
            segment.reason,
        )
    )

    frozen_reasons = {
        key: tuple(
            reason
            for reason in ("persistent_segment", "saturation")
            if reason in row_reasons
        )
        for key, row_reasons in reasons.items()
    }
    return PurpleAirQCResult(
        parameters=parameters,
        study_start=start,
        study_end_exclusive=end_exclusive,
        exclusion_reasons=frozen_reasons,
        primary_excluded_windows=frozenset(primary_excluded_windows),
        unevaluated_windows=unevaluated_windows,
        total_window_count=len(entity_ids) * len(centers),
        evaluated_window_count=evaluated_window_count,
        segments=tuple(segments),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_database_timestamp(raw: object) -> datetime:
    try:
        return _ensure_utc(datetime.fromisoformat(str(raw)))
    except ValueError as exc:
        raise ValueError(f"invalid PurpleAir timestamp: {raw!r}") from exc


def _load_snapshot_readings(
    database: Path,
    *,
    expected_sha256: str,
) -> tuple[list[PurpleAirReading], int]:
    before_hash = _sha256(database)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != {expected_sha256}"
        )
    readings: list[PurpleAirReading] = []
    queried_rows = 0
    connection = sqlite3.connect(f"{database.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(
            """
            SELECT source_entity_id, timestamp, value, lat, lon
            FROM data_points
            WHERE source = 'purpleair'
              AND metric = 'pm25'
              AND datetime(timestamp) >= datetime(?)
              AND datetime(timestamp) < datetime(?)
            ORDER BY source_entity_id, timestamp, id
            """,
            (_iso(STUDY_START), _iso(STUDY_END_EXCLUSIVE)),
        )
        for entity_id, raw_timestamp, raw_value, lat, lon in cursor:
            queried_rows += 1
            value = float(raw_value)
            if not math.isfinite(value):
                continue
            if distance_km(TARGET_LAT, TARGET_LON, float(lat), float(lon)) > TARGET_RADIUS_KM:
                continue
            readings.append(
                PurpleAirReading(
                    entity_id=str(entity_id),
                    timestamp=_parse_database_timestamp(raw_timestamp),
                    value=value,
                )
            )
    finally:
        connection.close()
        after_hash = _sha256(database)
        if after_hash != expected_sha256:
            raise RuntimeError(
                "snapshot SHA-256 mismatch after read: "
                f"{after_hash} != {expected_sha256}"
            )
    return readings, queried_rows


def _fraction(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _mean(readings: Sequence[PurpleAirReading]) -> float | None:
    return fmean(reading.value for reading in readings) if readings else None


def _serialized_segment(segment: PurpleAirQCSegment) -> dict[str, Any]:
    return {
        "entity_id": segment.entity_id,
        "reason": segment.reason,
        "decision_start": _iso(segment.decision_start),
        "decision_end_inclusive": _iso(segment.decision_end_inclusive),
        "covered_start": _iso(segment.covered_start),
        "covered_end_exclusive": _iso(segment.covered_end_exclusive),
        "excluded_row_count": segment.excluded_row_count,
    }


def _audited_sensor_summary(
    sensor_id: str,
    readings: Sequence[PurpleAirReading],
    result: PurpleAirQCResult,
) -> dict[str, Any]:
    sensor_readings = [reading for reading in readings if reading.entity_id == sensor_id]
    excluded = [
        reading
        for reading in sensor_readings
        if not result.is_eligible(reading.entity_id, reading.timestamp)
    ]
    retained = [
        reading
        for reading in sensor_readings
        if result.is_eligible(reading.entity_id, reading.timestamp)
    ]
    last_excluded = max((reading.timestamp for reading in excluded), default=None)
    retained_after = [
        reading
        for reading in retained
        if last_excluded is not None and reading.timestamp > last_excluded
    ]
    return {
        "entity_id": sensor_id,
        "rows": len(sensor_readings),
        "excluded_rows": len(excluded),
        "excluded_fraction": _fraction(len(excluded), len(sensor_readings)),
        "first_excluded": _iso(min(reading.timestamp for reading in excluded))
        if excluded
        else None,
        "last_excluded": _iso(last_excluded) if last_excluded else None,
        "retained_after_last_excluded": len(retained_after),
        "latest_retained": _iso(max(reading.timestamp for reading in retained))
        if retained
        else None,
    }


def derive_purpleair_qc_fixture(
    database: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    parameters: PurpleAirQCParameters = DEFAULT_QC_PARAMETERS,
) -> dict[str, Any]:
    """Derive the manifest-ready B7 artifact from the locked snapshot."""
    readings, queried_rows = _load_snapshot_readings(
        database, expected_sha256=expected_sha256
    )
    result = evaluate_purpleair_qc(
        readings,
        STUDY_START,
        STUDY_END_EXCLUSIVE,
        parameters=parameters,
    )
    excluded_keys = set(result.exclusion_reasons)
    retained = [
        reading
        for reading in readings
        if (reading.entity_id, reading.timestamp) not in excluded_keys
    ]
    june = [reading for reading in readings if reading.timestamp < JUNE_END_EXCLUSIVE]
    june_retained = [
        reading
        for reading in june
        if (reading.entity_id, reading.timestamp) not in excluded_keys
    ]
    unevaluated_by_reason: dict[str, int] = defaultdict(int)
    for reason in result.unevaluated_windows.values():
        unevaluated_by_reason[reason] += 1
    reason_counts = {
        reason: sum(
            1 for reasons in result.exclusion_reasons.values() if reason in reasons
        )
        for reason in ("persistent_segment", "saturation")
    }
    reading_by_key = {
        (reading.entity_id, reading.timestamp): reading for reading in readings
    }

    return {
        "schema_version": 1,
        "fixture_id": "purpleair-time-aware-qc-v1",
        "snapshot_sha256": expected_sha256,
        "study_window": {
            "start": _iso(STUDY_START),
            "end_exclusive": _iso(STUDY_END_EXCLUSIVE),
        },
        "target": {
            "lat": TARGET_LAT,
            "lon": TARGET_LON,
            "radius_km": TARGET_RADIUS_KM,
        },
        "source": "purpleair",
        "metric": "pm25",
        "stored_field": "raw ATM; no Barkjohn correction in this evaluation",
        "window_semantics": (
            "UTC-hour centers; [center-12h, center+12h), clamped at study edges"
        ),
        "parameters": asdict(parameters),
        "input": {
            "queried_rows": queried_rows,
            "in_radius_finite_rows": len(readings),
            "entities": len({reading.entity_id for reading in readings}),
        },
        "window_evaluation": {
            "total": result.total_window_count,
            "evaluated": result.evaluated_window_count,
            "unevaluated": result.unevaluated_window_count,
            "unevaluated_fraction": _fraction(
                result.unevaluated_window_count, result.total_window_count
            ),
            "unevaluated_by_reason": dict(sorted(unevaluated_by_reason.items())),
            "primary_excluded_windows": len(result.primary_excluded_windows),
        },
        "row_evaluation": {
            "total": len(readings),
            "excluded": result.excluded_row_count,
            "excluded_fraction": _fraction(result.excluded_row_count, len(readings)),
            "retained": len(retained),
            "reason_counts_nonexclusive": reason_counts,
        },
        "june_channel": {
            "before_rows": len(june),
            "after_rows": len(june_retained),
            "mean_before_ug_m3": _mean(june),
            "mean_after_ug_m3": _mean(june_retained),
        },
        "audited_sensors": [
            _audited_sensor_summary(sensor_id, readings, result)
            for sensor_id in AUDITED_SENSOR_IDS
        ],
        "segments": [_serialized_segment(segment) for segment in result.segments],
        "excluded_rows": [
            {
                "entity_id": entity_id,
                "timestamp": _iso(timestamp),
                "value": reading_by_key[(entity_id, timestamp)].value,
                "reasons": list(result.exclusion_reasons[(entity_id, timestamp)]),
            }
            for entity_id, timestamp in sorted(excluded_keys)
        ],
    }


def write_purpleair_qc_fixture(payload: Mapping[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


@lru_cache(maxsize=1)
def load_purpleair_qc_fixture() -> dict[str, Any]:
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture = dict(_as_mapping(payload, field_name="PurpleAir QC fixture"))
    if fixture.get("schema_version") != 1:
        raise ValueError("unsupported PurpleAir QC fixture schema")
    if fixture.get("snapshot_sha256") != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("PurpleAir QC fixture does not match the locked snapshot")
    if fixture.get("parameters") != asdict(DEFAULT_QC_PARAMETERS):
        raise ValueError("PurpleAir QC fixture parameters differ from the declaration")
    if fixture.get("stored_field") != (
        "raw ATM; no Barkjohn correction in this evaluation"
    ):
        raise ValueError("PurpleAir QC fixture does not preserve the raw ATM field")
    excluded_rows = fixture.get("excluded_rows")
    if not isinstance(excluded_rows, list):
        raise ValueError("PurpleAir QC fixture excluded_rows must be a list")
    return fixture


@lru_cache(maxsize=1)
def excluded_purpleair_row_keys() -> frozenset[RowKey]:
    keys: set[RowKey] = set()
    for raw_row in load_purpleair_qc_fixture()["excluded_rows"]:
        row = _as_mapping(raw_row, field_name="PurpleAir QC excluded row")
        entity_id = row.get("entity_id")
        timestamp = row.get("timestamp")
        if not isinstance(entity_id, str) or not isinstance(timestamp, str):
            raise ValueError("PurpleAir QC excluded row has invalid identity")
        key = (entity_id, _parse_database_timestamp(timestamp))
        if key in keys:
            raise ValueError(f"duplicate PurpleAir QC excluded row: {key!r}")
        keys.add(key)
    return frozenset(keys)


def purpleair_reading_is_eligible(entity_id: str, observed_at: datetime) -> bool:
    """Shared B7 predicate for detection, enrichment, and scoring paths."""
    return (str(entity_id), _ensure_utc(observed_at)) not in excluded_purpleair_row_keys()


def purpleair_qc_manifest_payload() -> dict[str, Any]:
    """Manifest evidence copied from, and hash-linked to, the B7 artifact."""
    fixture = load_purpleair_qc_fixture()
    copied_fields = (
        "fixture_id",
        "snapshot_sha256",
        "study_window",
        "target",
        "source",
        "metric",
        "stored_field",
        "window_semantics",
        "parameters",
        "input",
        "window_evaluation",
        "row_evaluation",
        "june_channel",
        "audited_sensors",
        "segments",
    )
    payload: dict[str, Any] = {
        "artifact": FIXTURE_PATH.name,
        "artifact_sha256": _sha256(FIXTURE_PATH),
    }
    for field_name in copied_fields:
        if field_name not in fixture:
            raise ValueError(
                f"PurpleAir QC fixture is missing manifest field {field_name!r}"
            )
        payload[field_name] = deepcopy(fixture[field_name])
    return payload


def format_purpleair_qc_report(payload: Mapping[str, Any]) -> str:
    """Render the empirical artifact fields as plan-ready Markdown."""
    june = _as_mapping(payload["june_channel"], field_name="june_channel")
    rows = _as_mapping(payload["row_evaluation"], field_name="row_evaluation")
    windows = _as_mapping(
        payload["window_evaluation"], field_name="window_evaluation"
    )
    lines = [
        "| Scope | Before rows | After rows | Mean before (ug/m3) | Mean after (ug/m3) |",
        "|---|---:|---:|---:|---:|",
        (
            f"| June 1-30 | {june['before_rows']} | {june['after_rows']} | "
            f"{float(june['mean_before_ug_m3']):.6f} | "
            f"{float(june['mean_after_ug_m3']):.6f} |"
        ),
        "",
        "| Row/window measure | Value |",
        "|---|---:|",
        f"| Study rows excluded | {rows['excluded']} / {rows['total']} ({float(rows['excluded_fraction']):.6%}) |",
        f"| Windows unevaluated | {windows['unevaluated']} / {windows['total']} ({float(windows['unevaluated_fraction']):.6%}) |",
        f"| Primary excluded windows | {windows['primary_excluded_windows']} |",
        "",
        "| Audited sensor | Rows | Excluded | First excluded | Last excluded | Retained after | Latest retained |",
        "|---|---:|---:|---|---|---:|---|",
    ]
    for raw_sensor in payload["audited_sensors"]:
        sensor = _as_mapping(raw_sensor, field_name="audited sensor")
        lines.append(
            f"| {sensor['entity_id']} | {sensor['rows']} | {sensor['excluded_rows']} | "
            f"{sensor['first_excluded'] or 'N/A'} | {sensor['last_excluded'] or 'N/A'} | "
            f"{sensor['retained_after_last_excluded']} | {sensor['latest_retained'] or 'N/A'} |"
        )
    lines.extend(
        [
            "",
            "| Sensor | Reason | Decision start | Decision end | Covered start | Covered end | Excluded rows |",
            "|---|---|---|---|---|---|---:|",
        ]
    )
    for raw_segment in payload["segments"]:
        segment = _as_mapping(raw_segment, field_name="PurpleAir QC segment")
        lines.append(
            f"| {segment['entity_id']} | {segment['reason']} | "
            f"{segment['decision_start']} | {segment['decision_end_inclusive']} | "
            f"{segment['covered_start']} | {segment['covered_end_exclusive']} | "
            f"{segment['excluded_row_count']} |"
        )
    return "\n".join(lines)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the B7 PurpleAir time-aware QC artifact."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--expected-sha256", default=LOCKED_SNAPSHOT_SHA256)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    payload = derive_purpleair_qc_fixture(
        args.database, expected_sha256=args.expected_sha256
    )
    if args.output is not None:
        write_purpleair_qc_fixture(payload, args.output)
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(format_purpleair_qc_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
