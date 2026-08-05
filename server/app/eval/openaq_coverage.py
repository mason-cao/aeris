"""C4′ label-free OpenAQ coverage audit on the locked SQLite snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn

from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    NON_MONITOR_SENSOR,
    UNMAPPABLE_ARCHIVE,
    VERIFIED_MONITOR,
    load_openaq_pm25_fixture,
)

from app.provenance.openaq_pm25 import (  # noqa: E402
    STUDY_END_EXCLUSIVE_AT as STUDY_END_EXCLUSIVE,
    STUDY_START_AT as STUDY_START,
)
_HOUR = timedelta(hours=1)
_DAY = timedelta(days=1)

_GroupKey = tuple[str, str, str, str | None]
_CellKey = tuple[str, str, datetime]


@dataclass(frozen=True)
class OpenAQObservation:
    """One stored OpenAQ row required by the coverage audit."""

    entity_id: str
    timestamp: datetime
    collected_at: datetime
    metric: str
    unit: str
    raw_json: object


@dataclass
class _EntityAccumulator:
    providers: set[str] = field(default_factory=set)
    monitor_flags: set[bool] = field(default_factory=set)
    instrument_sets: set[tuple[str, ...]] = field(default_factory=set)
    has_inline: bool = False
    has_archive: bool = False


@dataclass(frozen=True)
class _EntityMetadata:
    provider: str | None
    entity_class: str


@dataclass(frozen=True)
class _Cell:
    entity_id: str
    metric: str
    hour: datetime
    group: _GroupKey
    paths: frozenset[str]
    row_count: int


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(value: object, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc
    return _ensure_utc(parsed)


def _iso_z(value: datetime) -> str:
    return _ensure_utc(value).isoformat().replace("+00:00", "Z")


def _floor_hour(value: datetime) -> datetime:
    return _ensure_utc(value).replace(minute=0, second=0, microsecond=0)


def _as_mapping(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _monitor_flag(value: object, entity_id: str) -> bool:
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    raise ValueError(f"OpenAQ entity {entity_id} has invalid isMonitor metadata")


def _inline_metadata(
    raw_json: Mapping[str, Any], entity_id: str
) -> tuple[str, bool, tuple[str, ...]]:
    location = _as_mapping(
        raw_json.get("location"), f"OpenAQ entity {entity_id} location"
    )
    provider_payload = _as_mapping(
        location.get("provider"), f"OpenAQ entity {entity_id} provider"
    )
    provider = provider_payload.get("name")
    if not isinstance(provider, str) or not provider.strip():
        raise ValueError(f"OpenAQ entity {entity_id} has no provider name")
    is_monitor = _monitor_flag(location.get("isMonitor"), entity_id)
    raw_instruments = location.get("instruments")
    if not isinstance(raw_instruments, list) or not raw_instruments:
        raise ValueError(f"OpenAQ entity {entity_id} has no instrument metadata")
    instruments: list[str] = []
    for raw_instrument in raw_instruments:
        instrument = _as_mapping(
            raw_instrument, f"OpenAQ entity {entity_id} instrument"
        )
        name = instrument.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"OpenAQ entity {entity_id} has blank instrument name")
        instruments.append(name)
    return provider, is_monitor, tuple(sorted(set(instruments)))


def _normalize_observation(
    observation: OpenAQObservation,
    *,
    study_start: datetime,
    study_end_exclusive: datetime,
) -> tuple[OpenAQObservation, str]:
    if not isinstance(observation.entity_id, str) or not observation.entity_id.strip():
        raise ValueError("OpenAQ row must have a nonempty entity ID")
    if not isinstance(observation.metric, str) or not observation.metric.strip():
        raise ValueError(f"OpenAQ entity {observation.entity_id} has blank metric")
    if not isinstance(observation.unit, str) or not observation.unit.strip():
        raise ValueError(f"OpenAQ metric {observation.metric} has blank unit")
    timestamp = _ensure_utc(observation.timestamp)
    collected_at = _ensure_utc(observation.collected_at)
    if not study_start <= timestamp < study_end_exclusive:
        raise ValueError(
            f"OpenAQ timestamp {_iso_z(timestamp)} is outside declared study window"
        )
    raw_json = _as_mapping(observation.raw_json, "OpenAQ raw_json")
    paths = [name for name in ("location", "archive") if name in raw_json]
    if len(paths) != 1:
        raise ValueError(
            f"OpenAQ entity {observation.entity_id} row must have exactly one "
            "ingest path"
        )
    path = "inline" if paths[0] == "location" else "archive"
    _as_mapping(
        raw_json[paths[0]],
        f"OpenAQ entity {observation.entity_id} {paths[0]} payload",
    )
    return (
        OpenAQObservation(
            entity_id=observation.entity_id,
            timestamp=timestamp,
            collected_at=collected_at,
            metric=observation.metric,
            unit=observation.unit,
            raw_json=dict(raw_json),
        ),
        path,
    )


def _entity_metadata(
    observations: Sequence[tuple[OpenAQObservation, str]],
) -> dict[str, _EntityMetadata]:
    accumulators: dict[str, _EntityAccumulator] = defaultdict(_EntityAccumulator)
    for observation, path in observations:
        accumulator = accumulators[observation.entity_id]
        if path == "archive":
            accumulator.has_archive = True
            continue
        accumulator.has_inline = True
        provider, is_monitor, instruments = _inline_metadata(
            _as_mapping(observation.raw_json, "OpenAQ raw_json"),
            observation.entity_id,
        )
        accumulator.providers.add(provider)
        accumulator.monitor_flags.add(is_monitor)
        accumulator.instrument_sets.add(instruments)

    metadata: dict[str, _EntityMetadata] = {}
    for entity_id in sorted(accumulators):
        accumulator = accumulators[entity_id]
        if not accumulator.has_inline:
            metadata[entity_id] = _EntityMetadata(
                provider=None, entity_class=UNMAPPABLE_ARCHIVE
            )
            continue
        if len(accumulator.providers) != 1:
            raise ValueError(
                f"OpenAQ entity {entity_id} has conflicting provider metadata"
            )
        if len(accumulator.monitor_flags) != 1:
            raise ValueError(
                f"OpenAQ entity {entity_id} has conflicting isMonitor metadata"
            )
        provider = next(iter(accumulator.providers))
        is_monitor = next(iter(accumulator.monitor_flags))
        if (
            provider == "AirNow"
            and is_monitor
            and accumulator.instrument_sets == {("Government Monitor",)}
        ):
            entity_class = VERIFIED_MONITOR
        elif provider in {"AirGradient", "Clarity"} and not is_monitor:
            entity_class = NON_MONITOR_SENSOR
        else:
            raise ValueError(
                "OpenAQ entity "
                f"{entity_id} has an undeclared provider/monitor/instrument "
                f"combination: provider={provider!r}, is_monitor={is_monitor!r}, "
                f"instrument_sets={sorted(accumulator.instrument_sets)!r}"
            )
        metadata[entity_id] = _EntityMetadata(
            provider=provider, entity_class=entity_class
        )
    return metadata


def _unit_by_metric(
    observations: Sequence[tuple[OpenAQObservation, str]],
) -> dict[str, str]:
    units: dict[str, set[str]] = defaultdict(set)
    for observation, _ in observations:
        units[observation.metric].add(observation.unit)
    for metric, metric_units in sorted(units.items()):
        if len(metric_units) != 1:
            raise ValueError(
                f"OpenAQ metric {metric} has multiple units: {sorted(metric_units)!r}"
            )
    return {metric: next(iter(units[metric])) for metric in sorted(units)}


def _assert_b6_pm25_match(
    observations: Sequence[tuple[OpenAQObservation, str]],
    metadata: Mapping[str, _EntityMetadata],
    expected: Mapping[str, str],
) -> None:
    actual_ids = {
        observation.entity_id
        for observation, _ in observations
        if observation.metric == "pm25"
    }
    expected_ids = set(expected)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "OpenAQ PM2.5 entity-set drift from B6 fixture: "
            f"missing={missing!r}, extra={extra!r}"
        )
    mismatches = [
        (entity_id, expected[entity_id], metadata[entity_id].entity_class)
        for entity_id in sorted(actual_ids)
        if metadata[entity_id].entity_class != expected[entity_id]
    ]
    if mismatches:
        raise ValueError(f"OpenAQ PM2.5 class drift from B6 fixture: {mismatches!r}")


def _group_sort_key(group: _GroupKey) -> tuple[str, str, str, str]:
    metric, unit, entity_class, provider = group
    return metric, unit, entity_class, provider or ""


def _fraction(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _hours_between(start: datetime, end: datetime) -> int:
    seconds = (end - start).total_seconds()
    if seconds < 0 or seconds % 3600 != 0:
        raise ValueError("coverage endpoints must define nonnegative whole UTC hours")
    return int(seconds // 3600)


def _longest_internal_gap(cells: Sequence[_Cell]) -> int:
    hours_by_stream: dict[tuple[str, str], list[datetime]] = defaultdict(list)
    for cell in cells:
        hours_by_stream[(cell.entity_id, cell.metric)].append(cell.hour)
    longest = 0
    for hours in hours_by_stream.values():
        ordered = sorted(set(hours))
        for previous, current in zip(ordered, ordered[1:]):
            gap = _hours_between(previous + _HOUR, current)
            longest = max(longest, gap)
    return longest


def _summary(
    daily_rows: Sequence[Mapping[str, Any]],
    cells: Sequence[_Cell],
    *,
    entity_metric_count: int,
    unique_entity_count: int,
) -> dict[str, Any]:
    integer_fields = (
        "row_count",
        "observed_entity_hours",
        "roster_expected_entity_hours",
        "active_expected_entity_hours",
        "missing_active_entity_hours",
        "inline_contributed_entity_hours",
        "archive_contributed_entity_hours",
        "archive_only_entity_hours",
        "both_path_entity_hours",
        "overlap_row_count",
    )
    totals = {
        field: sum(int(row[field]) for row in daily_rows) for field in integer_fields
    }
    return {
        "entity_metric_count": entity_metric_count,
        "unique_entity_count": unique_entity_count,
        **totals,
        "roster_coverage_fraction": _fraction(
            totals["observed_entity_hours"], totals["roster_expected_entity_hours"]
        ),
        "active_coverage_fraction": _fraction(
            totals["observed_entity_hours"], totals["active_expected_entity_hours"]
        ),
        "days_with_observation": sum(
            int(row["observed_entity_hours"] > 0) for row in daily_rows
        ),
        "longest_internal_gap_hours": _longest_internal_gap(cells),
    }


def build_report(
    observations: Sequence[OpenAQObservation],
    *,
    snapshot_sha256: str,
    snapshot_max_collected_at: datetime,
    expected_pm25_classifications: Mapping[str, str],
    study_start: datetime = STUDY_START,
    study_end_exclusive: datetime = STUDY_END_EXCLUSIVE,
) -> dict[str, object]:
    """Build the deterministic C4′ report from already-loaded OpenAQ rows."""
    start = _ensure_utc(study_start)
    end = _ensure_utc(study_end_exclusive)
    if start != _floor_hour(start) or end != _floor_hour(end) or end <= start:
        raise ValueError("study endpoints must be increasing exact UTC hours")
    if not observations:
        raise ValueError("C4′ requires at least one OpenAQ row")

    normalized = [
        _normalize_observation(
            observation,
            study_start=start,
            study_end_exclusive=end,
        )
        for observation in observations
    ]
    normalized.sort(
        key=lambda item: (
            item[0].timestamp,
            item[0].entity_id,
            item[0].metric,
            item[0].unit,
            item[1],
            json.dumps(item[0].raw_json, sort_keys=True),
        )
    )
    unit_by_metric = _unit_by_metric(normalized)
    metadata = _entity_metadata(normalized)
    _assert_b6_pm25_match(
        normalized, metadata, expected_pm25_classifications
    )

    coverage_end = min(end, _floor_hour(snapshot_max_collected_at))
    if coverage_end <= start:
        raise ValueError("snapshot contains no completed study-window UTC hour")
    analyzed = [item for item in normalized if item[0].timestamp < coverage_end]
    trailing_count = len(normalized) - len(analyzed)
    if not analyzed:
        raise ValueError("no OpenAQ rows fall in completed coverage hours")

    cell_paths: dict[_CellKey, set[str]] = defaultdict(set)
    cell_row_counts: dict[_CellKey, int] = defaultdict(int)
    cell_groups: dict[_CellKey, _GroupKey] = {}
    for observation, path in analyzed:
        hour = _floor_hour(observation.timestamp)
        key = (observation.entity_id, observation.metric, hour)
        entity = metadata[observation.entity_id]
        group = (
            observation.metric,
            observation.unit,
            entity.entity_class,
            entity.provider,
        )
        prior_group = cell_groups.setdefault(key, group)
        if prior_group != group:
            raise ValueError(f"OpenAQ entity-hour has inconsistent group: {key!r}")
        cell_paths[key].add(path)
        cell_row_counts[key] += 1

    cells = [
        _Cell(
            entity_id=entity_id,
            metric=metric,
            hour=hour,
            group=cell_groups[(entity_id, metric, hour)],
            paths=frozenset(cell_paths[(entity_id, metric, hour)]),
            row_count=cell_row_counts[(entity_id, metric, hour)],
        )
        for entity_id, metric, hour in sorted(
            cell_paths, key=lambda key: (key[2], key[1], key[0])
        )
    ]
    groups = sorted({cell.group for cell in cells}, key=_group_sort_key)
    cells_by_group: dict[_GroupKey, list[_Cell]] = {
        group: [cell for cell in cells if cell.group == group] for group in groups
    }
    roster_by_group: dict[_GroupKey, set[str]] = {
        group: {cell.entity_id for cell in group_cells}
        for group, group_cells in cells_by_group.items()
    }
    spans: dict[tuple[str, str], tuple[datetime, datetime]] = {}
    for cell in cells:
        key = (cell.entity_id, cell.metric)
        if key not in spans:
            spans[key] = (cell.hour, cell.hour)
        else:
            first, last = spans[key]
            spans[key] = (min(first, cell.hour), max(last, cell.hour))

    daily_rows: list[dict[str, Any]] = []
    day_start = start
    while day_start < coverage_end:
        day_end = min(day_start + _DAY, coverage_end)
        completed_clock_hours = _hours_between(day_start, day_end)
        for group in groups:
            metric, unit, entity_class, provider = group
            group_cells = [
                cell
                for cell in cells_by_group[group]
                if day_start <= cell.hour < day_end
            ]
            roster = roster_by_group[group]
            active_expected = 0
            active_entities = 0
            for entity_id in roster:
                first, last = spans[(entity_id, metric)]
                active_start = max(first, day_start)
                active_end = min(last + _HOUR, day_end)
                expected = (
                    _hours_between(active_start, active_end)
                    if active_end > active_start
                    else 0
                )
                if expected:
                    active_entities += 1
                    active_expected += expected
            observed = len(group_cells)
            roster_expected = len(roster) * completed_clock_hours
            missing_active = active_expected - observed
            if not 0 <= observed <= active_expected <= roster_expected:
                raise ValueError(
                    "impossible OpenAQ coverage counts for "
                    f"{day_start.date()} {group!r}"
                )
            row_count = sum(cell.row_count for cell in group_cells)
            daily_rows.append(
                {
                    "date": day_start.date().isoformat(),
                    "metric": metric,
                    "unit": unit,
                    "entity_class": entity_class,
                    "provider": provider,
                    "completed_clock_hours": completed_clock_hours,
                    "roster_entities": len(roster),
                    "observed_entities": len(
                        {cell.entity_id for cell in group_cells}
                    ),
                    "active_entities": active_entities,
                    "row_count": row_count,
                    "observed_entity_hours": observed,
                    "roster_expected_entity_hours": roster_expected,
                    "roster_coverage_fraction": _fraction(observed, roster_expected),
                    "active_expected_entity_hours": active_expected,
                    "active_coverage_fraction": _fraction(observed, active_expected),
                    "missing_active_entity_hours": missing_active,
                    "inline_contributed_entity_hours": sum(
                        "inline" in cell.paths for cell in group_cells
                    ),
                    "archive_contributed_entity_hours": sum(
                        "archive" in cell.paths for cell in group_cells
                    ),
                    "archive_only_entity_hours": sum(
                        cell.paths == {"archive"} for cell in group_cells
                    ),
                    "both_path_entity_hours": sum(
                        cell.paths == {"inline", "archive"} for cell in group_cells
                    ),
                    "overlap_row_count": row_count - observed,
                }
            )
        day_start += _DAY

    group_summaries: list[dict[str, Any]] = []
    for group in groups:
        metric, unit, entity_class, provider = group
        group_daily = [
            row
            for row in daily_rows
            if (
                row["metric"],
                row["unit"],
                row["entity_class"],
                row["provider"],
            )
            == group
        ]
        group_cells = cells_by_group[group]
        summary = _summary(
            group_daily,
            group_cells,
            entity_metric_count=len(roster_by_group[group]),
            unique_entity_count=len(roster_by_group[group]),
        )
        group_summaries.append(
            {
                "metric": metric,
                "unit": unit,
                "entity_class": entity_class,
                "provider": provider,
                **summary,
            }
        )

    all_entity_metrics = {(cell.entity_id, cell.metric) for cell in cells}
    overall_summary = _summary(
        daily_rows,
        cells,
        entity_metric_count=len(all_entity_metrics),
        unique_entity_count=len({cell.entity_id for cell in cells}),
    )
    class_counts = []
    for entity_class, provider in sorted(
        {(value.entity_class, value.provider) for value in metadata.values()},
        key=lambda item: (item[0], item[1] or ""),
    ):
        class_counts.append(
            {
                "entity_class": entity_class,
                "provider": provider,
                "entities": sum(
                    value.entity_class == entity_class and value.provider == provider
                    for value in metadata.values()
                ),
            }
        )

    return {
        "schema_version": 1,
        "snapshot_sha256": snapshot_sha256,
        "source": "openaq",
        "study_window": {
            "start": _iso_z(start),
            "end_exclusive": _iso_z(end),
        },
        "coverage_window": {
            "start": _iso_z(start),
            "end_exclusive": _iso_z(coverage_end),
            "cutoff_rule": (
                "min(study end, UTC-hour floor of snapshot-wide max collected_at); "
                "end exclusive"
            ),
        },
        "coverage_unit": "distinct (source_entity_id, metric, UTC-hour)",
        "entity_class_rule": {
            "verified_monitor": (
                "AirNow AND isMonitor=true AND Government Monitor"
            ),
            "non_monitor_sensor": (
                "provider in [AirGradient, Clarity] AND isMonitor=false"
            ),
            "unmappable_archive": "archive-only entity without inline metadata",
            "archive_mapping": "exact source_entity_id equality",
        },
        "denominator_rules": {
            "roster": (
                "full-window group entity roster times completed clock hours"
            ),
            "active_span": (
                "each entity's first through last observed UTC-hour, inclusive"
            ),
        },
        "input_row_count": len(normalized),
        "analyzed_row_count": len(analyzed),
        "incomplete_trailing_hour_row_count": trailing_count,
        "unit_by_metric": unit_by_metric,
        "single_unit_assertion_passed": True,
        "b6_pm25_fixture_match": True,
        "entity_class_counts": class_counts,
        "overall_summary": overall_summary,
        "group_summaries": group_summaries,
        "daily_coverage": daily_rows,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fixture_pm25_classifications() -> dict[str, str]:
    return {
        str(entity["entity_id"]): str(entity["classification"])
        for entity in load_openaq_pm25_fixture()["entities"]
    }


def derive_report(
    database: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
    expected_pm25_classifications: Mapping[str, str] | None = None,
    study_start: datetime = STUDY_START,
    study_end_exclusive: datetime = STUDY_END_EXCLUSIVE,
) -> dict[str, object]:
    """Read the immutable SQLite snapshot and derive the C4′ report."""
    before_hash = _sha256(database)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != {expected_sha256}"
        )

    connection: sqlite3.Connection | None = None
    observations: list[OpenAQObservation] = []
    snapshot_max_collected_at: datetime | None = None
    try:
        uri = f"{database.resolve().as_uri()}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only = ON")
        raw_max_collected_at = connection.execute(
            "SELECT MAX(collected_at) FROM data_points"
        ).fetchone()[0]
        if raw_max_collected_at is None:
            raise ValueError("snapshot has no collected_at values")
        snapshot_max_collected_at = _parse_timestamp(
            raw_max_collected_at, "snapshot maximum collected_at"
        )
        rows = connection.execute(
            """
            SELECT source_entity_id, timestamp, collected_at, metric, unit, raw_json
            FROM data_points
            WHERE source = 'openaq'
              AND datetime(timestamp) >= datetime(?)
              AND datetime(timestamp) < datetime(?)
            """,
            (_iso_z(study_start), _iso_z(study_end_exclusive)),
        )
        for (
            entity_id,
            timestamp,
            collected_at,
            metric,
            unit,
            raw_json,
        ) in rows:
            try:
                payload = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"OpenAQ entity {entity_id} has invalid raw_json"
                ) from exc
            observations.append(
                OpenAQObservation(
                    entity_id=entity_id,
                    timestamp=_parse_timestamp(timestamp, "OpenAQ timestamp"),
                    collected_at=_parse_timestamp(
                        collected_at, "OpenAQ collected_at"
                    ),
                    metric=metric,
                    unit=unit,
                    raw_json=payload,
                )
            )
    finally:
        if connection is not None:
            connection.close()
        after_hash = _sha256(database)
        if after_hash != expected_sha256:
            raise RuntimeError(
                "snapshot SHA-256 mismatch after read: "
                f"{after_hash} != {expected_sha256}"
            )

    if snapshot_max_collected_at is None:
        raise ValueError("snapshot maximum collected_at was not resolved")
    expected = (
        _fixture_pm25_classifications()
        if expected_pm25_classifications is None
        else dict(expected_pm25_classifications)
    )
    return build_report(
        observations,
        snapshot_sha256=expected_sha256,
        snapshot_max_collected_at=snapshot_max_collected_at,
        expected_pm25_classifications=expected,
        study_start=study_start,
        study_end_exclusive=study_end_exclusive,
    )


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_report(report: Mapping[str, object], output: Path) -> None:
    """Write deterministic canonical JSON for a C4′ report."""
    _atomic_write_text(
        output,
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _percent(value: object) -> str:
    if value is None:
        return "N/A"
    return f"{100.0 * float(value):.2f}%"


def render_markdown(report: Mapping[str, object]) -> str:
    """Render deterministic group and per-day C4′ coverage tables."""
    coverage_window = _as_mapping(report["coverage_window"], "coverage_window")
    overall = _as_mapping(report["overall_summary"], "overall_summary")
    lines = [
        "# C4′ OpenAQ hourly coverage",
        "",
        f"- Snapshot SHA-256: `{report['snapshot_sha256']}`",
        (
            "- Completed-hour window: "
            f"`{coverage_window['start']}` to "
            f"`{coverage_window['end_exclusive']}` (end exclusive)"
        ),
        f"- Input/analyzed/trailing rows: {report['input_row_count']} / "
        f"{report['analyzed_row_count']} / "
        f"{report['incomplete_trailing_hour_row_count']}",
        f"- Observed entity-hours: {overall['observed_entity_hours']}",
        f"- Roster coverage: {_percent(overall['roster_coverage_fraction'])}",
        f"- Active-span coverage: {_percent(overall['active_coverage_fraction'])}",
        f"- Archive-only entity-hours: {overall['archive_only_entity_hours']}",
        f"- Longest internal gap: {overall['longest_internal_gap_hours']} h",
        "",
        "## Group summary",
        "",
        (
            "| Metric | Unit | Entity class | Provider | Entities | Rows | "
            "Observed entity-hours | Roster expected | Roster coverage | "
            "Active expected | Active coverage | Missing active | Archive-only | "
            "Both paths | Overlap rows | Longest gap h |"
        ),
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["group_summaries"]:
        provider = row["provider"] if row["provider"] is not None else "unmapped"
        lines.append(
            f"| {row['metric']} | {row['unit']} | {row['entity_class']} | "
            f"{provider} | {row['unique_entity_count']} | {row['row_count']} | "
            f"{row['observed_entity_hours']} | "
            f"{row['roster_expected_entity_hours']} | "
            f"{_percent(row['roster_coverage_fraction'])} | "
            f"{row['active_expected_entity_hours']} | "
            f"{_percent(row['active_coverage_fraction'])} | "
            f"{row['missing_active_entity_hours']} | "
            f"{row['archive_only_entity_hours']} | "
            f"{row['both_path_entity_hours']} | {row['overlap_row_count']} | "
            f"{row['longest_internal_gap_hours']} |"
        )
    lines += [
        "",
        "## Per-day coverage",
        "",
        (
            "| Date | Metric | Unit | Entity class | Provider | Clock h | "
            "Roster entities | Observed entities | Active entities | Rows | "
            "Observed entity-hours | Roster expected | Roster coverage | "
            "Active expected | Active coverage | Missing active | Inline cells | "
            "Archive cells | Archive-only | Both paths | Overlap rows |"
        ),
        (
            "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|---:|---:|---:|"
        ),
    ]
    for row in report["daily_coverage"]:
        provider = row["provider"] if row["provider"] is not None else "unmapped"
        lines.append(
            f"| {row['date']} | {row['metric']} | {row['unit']} | "
            f"{row['entity_class']} | {provider} | "
            f"{row['completed_clock_hours']} | {row['roster_entities']} | "
            f"{row['observed_entities']} | {row['active_entities']} | "
            f"{row['row_count']} | {row['observed_entity_hours']} | "
            f"{row['roster_expected_entity_hours']} | "
            f"{_percent(row['roster_coverage_fraction'])} | "
            f"{row['active_expected_entity_hours']} | "
            f"{_percent(row['active_coverage_fraction'])} | "
            f"{row['missing_active_entity_hours']} | "
            f"{row['inline_contributed_entity_hours']} | "
            f"{row['archive_contributed_entity_hours']} | "
            f"{row['archive_only_entity_hours']} | "
            f"{row['both_path_entity_hours']} | {row['overlap_row_count']} |"
        )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Derive the label-free C4′ OpenAQ hourly coverage report."
    )
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument(
        "--expected-sha256", default=LOCKED_SNAPSHOT_SHA256
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the C4′ immutable snapshot audit CLI."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = derive_report(
            args.database,
            expected_sha256=args.expected_sha256,
        )
        write_report(report, args.output)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as exc:
        _argument_error(parser, str(exc))
    if args.format == "json":
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(render_markdown(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
