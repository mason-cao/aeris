"""Label-free B2 calm-wind cutoff diagnostics on a read-only SQLite snapshot."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import sqlite3
import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

STUDY_START = datetime(2026, 6, 1, 0, 0, tzinfo=UTC)
STUDY_END_EXCLUSIVE = datetime(2026, 7, 13, 0, 0, tzinfo=UTC)
WINDOW_HALF_WIDTH = timedelta(hours=36)
REPORTING_RESOLUTIONS: dict[str, float | None] = {
    "ASOS": 0.514444,
    "GFS |u,v|": None,
    "OpenWeather": 0.01,
}


@dataclass(frozen=True, order=True)
class SpeedObservation:
    timestamp: datetime
    value: float


@dataclass(frozen=True, order=True)
class ComponentObservation:
    entity_id: str
    timestamp: datetime
    value: float


@dataclass(frozen=True)
class WindowCutoffResult:
    cutoffs: tuple[float, ...]
    window_sample_sizes: tuple[int, ...]
    insufficient_windows: int


@dataclass(frozen=True)
class CutoffDistribution:
    n: int
    minimum: float | None
    p05: float | None
    median: float | None
    mean: float | None
    p95: float | None
    maximum: float | None
    pstdev: float | None
    fraction_nonpositive: float | None
    fraction_below_resolution: float | None

    def to_dict(self) -> dict[str, int | float | None]:
        return {
            "n": self.n,
            "minimum": self.minimum,
            "p05": self.p05,
            "median": self.median,
            "mean": self.mean,
            "p95": self.p95,
            "maximum": self.maximum,
            "pstdev": self.pstdev,
            "fraction_nonpositive": self.fraction_nonpositive,
            "fraction_below_resolution": self.fraction_below_resolution,
        }


@dataclass(frozen=True)
class ChannelEmpirics:
    channel: str
    resolution_mps: float | None
    observation_count: int
    candidate_window_count: int
    insufficient_window_count: int
    window_n_minimum: int | None
    window_n_median: float | None
    window_n_maximum: int | None
    cutoff_distribution: CutoffDistribution

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "resolution_mps": self.resolution_mps,
            "observation_count": self.observation_count,
            "candidate_window_count": self.candidate_window_count,
            "insufficient_window_count": self.insufficient_window_count,
            "window_n_minimum": self.window_n_minimum,
            "window_n_median": self.window_n_median,
            "window_n_maximum": self.window_n_maximum,
            "cutoff_distribution": self.cutoff_distribution.to_dict(),
        }


@dataclass(frozen=True)
class CalmWindEmpiricalReport:
    snapshot_sha256: str
    study_start: str
    study_end_exclusive: str
    candidate_window_count: int
    channels: tuple[ChannelEmpirics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_sha256": self.snapshot_sha256,
            "study_start": self.study_start,
            "study_end_exclusive": self.study_end_exclusive,
            "candidate_window_count": self.candidate_window_count,
            "channels": [channel.to_dict() for channel in self.channels],
        }


def _ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _parse_timestamp(value: str) -> datetime:
    return _ensure_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _validate_speed(value: float) -> float:
    speed = float(value)
    if not math.isfinite(speed):
        raise ValueError("wind speeds must be finite")
    if speed < 0.0:
        raise ValueError("wind speeds must be non-negative")
    return speed


def calm_cutoff(values: list[float]) -> float | None:
    """Return mean - 2 population SD, or None below the declared n=2 floor."""
    speeds = [_validate_speed(value) for value in values]
    if len(speeds) < 2:
        return None
    return statistics.fmean(speeds) - 2.0 * statistics.pstdev(speeds)


def candidate_centers(
    study_start: datetime,
    study_end_exclusive: datetime,
) -> tuple[datetime, ...]:
    """Hourly centers whose endpoint-inclusive 72-hour windows are in range."""
    start = _ensure_utc(study_start)
    end = _ensure_utc(study_end_exclusive)
    if end <= start:
        return ()
    center = start + WINDOW_HALF_WIDTH
    centers: list[datetime] = []
    while center + WINDOW_HALF_WIDTH < end:
        centers.append(center)
        center += timedelta(hours=1)
    return tuple(centers)


def window_cutoffs(
    observations: list[SpeedObservation],
    centers: list[datetime] | tuple[datetime, ...],
) -> WindowCutoffResult:
    """Compute endpoint-inclusive cutoffs for sorted or unsorted observations."""
    normalized = sorted(
        SpeedObservation(_ensure_utc(row.timestamp), _validate_speed(row.value))
        for row in observations
    )
    timestamps = [row.timestamp for row in normalized]
    values = [row.value for row in normalized]
    cutoffs: list[float] = []
    sample_sizes: list[int] = []
    insufficient = 0
    for raw_center in centers:
        center = _ensure_utc(raw_center)
        left = bisect.bisect_left(timestamps, center - WINDOW_HALF_WIDTH)
        right = bisect.bisect_right(timestamps, center + WINDOW_HALF_WIDTH)
        sample = values[left:right]
        sample_sizes.append(len(sample))
        cutoff = calm_cutoff(sample)
        if cutoff is None:
            insufficient += 1
        else:
            cutoffs.append(cutoff)
    return WindowCutoffResult(tuple(cutoffs), tuple(sample_sizes), insufficient)


def pair_gfs_components(
    u_values: list[ComponentObservation],
    v_values: list[ComponentObservation],
) -> tuple[SpeedObservation, ...]:
    """Pair GFS components only at identical entity/timestamp keys."""
    u_by_key: dict[tuple[str, datetime], float] = {}
    v_by_key: dict[tuple[str, datetime], float] = {}
    for rows, target in ((u_values, u_by_key), (v_values, v_by_key)):
        for row in rows:
            timestamp = _ensure_utc(row.timestamp)
            value = float(row.value)
            if not math.isfinite(value):
                raise ValueError("GFS wind components must be finite")
            key = (row.entity_id, timestamp)
            if key in target:
                raise ValueError(f"duplicate GFS component key: {key}")
            target[key] = value
    paired = [
        SpeedObservation(timestamp=key[1], value=math.hypot(u_by_key[key], v_by_key[key]))
        for key in sorted(set(u_by_key) & set(v_by_key))
    ]
    return tuple(sorted(paired))


def _percentile(sorted_values: list[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] + fraction * (
        sorted_values[upper] - sorted_values[lower]
    )


def describe_cutoffs(
    cutoffs: list[float] | tuple[float, ...],
    *,
    resolution: float | None,
) -> CutoffDistribution:
    values = sorted(float(value) for value in cutoffs)
    if any(not math.isfinite(value) for value in values):
        raise ValueError("cutoffs must be finite")
    if not values:
        return CutoffDistribution(0, None, None, None, None, None, None, None, None, None)
    fraction_below = (
        None
        if resolution is None
        else sum(value < resolution for value in values) / len(values)
    )
    return CutoffDistribution(
        n=len(values),
        minimum=values[0],
        p05=_percentile(values, 0.05),
        median=statistics.median(values),
        mean=statistics.fmean(values),
        p95=_percentile(values, 0.95),
        maximum=values[-1],
        pstdev=statistics.pstdev(values),
        fraction_nonpositive=sum(value <= 0.0 for value in values) / len(values),
        fraction_below_resolution=fraction_below,
    )


def _snapshot_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _query_rows(
    connection: sqlite3.Connection,
    *,
    source: str,
    metric: str,
) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            """
            SELECT source_entity_id, timestamp, value, unit
            FROM data_points
            WHERE source = ?
              AND metric = ?
              AND timestamp >= ?
              AND timestamp < ?
            ORDER BY timestamp, source_entity_id
            """,
            (
                source,
                metric,
                STUDY_START.strftime("%Y-%m-%d %H:%M:%S"),
                STUDY_END_EXCLUSIVE.strftime("%Y-%m-%d %H:%M:%S"),
            ),
        )
    )


def _load_speed_channel(
    connection: sqlite3.Connection,
    *,
    source: str,
) -> list[SpeedObservation]:
    rows = _query_rows(connection, source=source, metric="wind_speed")
    units = {str(row["unit"]) for row in rows}
    if units != {"m/s"}:
        raise ValueError(f"{source} wind_speed units are not exactly m/s: {units}")
    return [
        SpeedObservation(_parse_timestamp(str(row["timestamp"])), _validate_speed(row["value"]))
        for row in rows
    ]


def _load_gfs_channel(connection: sqlite3.Connection) -> list[SpeedObservation]:
    components: dict[str, list[ComponentObservation]] = {}
    for metric in ("u_10m", "v_10m"):
        rows = _query_rows(connection, source="noaa_gfs", metric=metric)
        units = {str(row["unit"]) for row in rows}
        if units != {"m/s"}:
            raise ValueError(f"noaa_gfs {metric} units are not exactly m/s: {units}")
        components[metric] = [
            ComponentObservation(
                str(row["source_entity_id"]),
                _parse_timestamp(str(row["timestamp"])),
                float(row["value"]),
            )
            for row in rows
        ]
    return list(pair_gfs_components(components["u_10m"], components["v_10m"]))


def _channel_empirics(
    channel: str,
    observations: list[SpeedObservation],
    centers: tuple[datetime, ...],
) -> ChannelEmpirics:
    result = window_cutoffs(observations, centers)
    sample_sizes = list(result.window_sample_sizes)
    return ChannelEmpirics(
        channel=channel,
        resolution_mps=REPORTING_RESOLUTIONS[channel],
        observation_count=len(observations),
        candidate_window_count=len(centers),
        insufficient_window_count=result.insufficient_windows,
        window_n_minimum=min(sample_sizes) if sample_sizes else None,
        window_n_median=statistics.median(sample_sizes) if sample_sizes else None,
        window_n_maximum=max(sample_sizes) if sample_sizes else None,
        cutoff_distribution=describe_cutoffs(
            result.cutoffs,
            resolution=REPORTING_RESOLUTIONS[channel],
        ),
    )


def run_empirics(
    database_path: Path,
    *,
    expected_sha256: str,
) -> CalmWindEmpiricalReport:
    """Run all channel empirics with pre/post snapshot hash verification."""
    resolved = database_path.resolve()
    before_hash = _snapshot_sha256(resolved)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != {expected_sha256}"
        )
    uri = f"file:{resolved}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only = ON")
        observations = {
            "ASOS": _load_speed_channel(connection, source="asos"),
            "GFS |u,v|": _load_gfs_channel(connection),
            "OpenWeather": _load_speed_channel(connection, source="openweather"),
        }
    finally:
        connection.close()
    after_hash = _snapshot_sha256(resolved)
    if after_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch after read: {after_hash} != {expected_sha256}"
        )
    centers = candidate_centers(STUDY_START, STUDY_END_EXCLUSIVE)
    channels = tuple(
        _channel_empirics(channel, observations[channel], centers)
        for channel in ("ASOS", "GFS |u,v|", "OpenWeather")
    )
    return CalmWindEmpiricalReport(
        snapshot_sha256=after_hash,
        study_start=STUDY_START.isoformat(),
        study_end_exclusive=STUDY_END_EXCLUSIVE.isoformat(),
        candidate_window_count=len(centers),
        channels=channels,
    )


def _format_value(value: float | int | None, *, digits: int = 6) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def _format_fraction(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(report: CalmWindEmpiricalReport) -> str:
    lines = [
        "| Channel | Speed rows | Windows | Insufficient | Window n min/median/max | Resolution (m/s) | Cutoff min | p05 | Median | Mean | p95 | Max | Cutoff pstdev | Fraction <= 0 | Fraction < resolution |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for channel in report.channels:
        distribution = channel.cutoff_distribution
        sample_size = "/".join(
            (
                _format_value(channel.window_n_minimum),
                _format_value(channel.window_n_median),
                _format_value(channel.window_n_maximum),
            )
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    channel.channel.replace("|", "\\|"),
                    str(channel.observation_count),
                    str(channel.candidate_window_count),
                    str(channel.insufficient_window_count),
                    sample_size,
                    _format_value(channel.resolution_mps),
                    _format_value(distribution.minimum),
                    _format_value(distribution.p05),
                    _format_value(distribution.median),
                    _format_value(distribution.mean),
                    _format_value(distribution.p95),
                    _format_value(distribution.maximum),
                    _format_value(distribution.pstdev),
                    _format_fraction(distribution.fraction_nonpositive),
                    _format_fraction(distribution.fraction_below_resolution),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.calm_wind_empirics",
        description="Compute B2 calm-wind cutoff distributions without labels.",
    )
    parser.add_argument("--database", required=True, type=Path, help="read-only frozen SQLite snapshot")
    parser.add_argument("--expected-sha256", required=True, help="recorded snapshot SHA-256")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    report = run_empirics(args.database, expected_sha256=args.expected_sha256)
    if args.format == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
