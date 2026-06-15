"""CLI runner for the detection pipeline.

Loads ``DataPoint`` rows from the database, groups them into per-station
time-series, joins contemporaneous NOAA GFS aux features for IsolationForest
(10-m wind speed derived from ``u_10m``/``v_10m``, plus ``pbl_height``), runs
the :class:`DetectionEngine`, and persists the resulting :class:`Anomaly`
rows idempotently.

Usage::

    python -m app.detection.run                          # all data
    python -m app.detection.run --source openaq          # filter by source
    python -m app.detection.run --metric pm25            # filter by metric
    python -m app.detection.run --since 2026-05-01       # only recent
    python -m app.detection.run --dry-run                # summary only
"""

from __future__ import annotations

import argparse
import asyncio
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.geo import distance_km
from app.db.models import Anomaly, DataPoint

from .consensus import ConsensusAnomaly
from .engine import DetectionEngine
from .isolation_forest import IsolationForestDetector, IsolationForestInput
from .stl import STLDetector
from .zscore import ZScoreDetector


# Metrics the engine runs on directly. Auxiliary atmospheric / met fields
# (GFS u_10m/v_10m wind components, pbl_height) are joined into the
# IsolationForest aux_features instead of being treated as detection targets.
# These names must match what the collectors emit (openaq.PARAMETER_MAP,
# sentinel5p.PRODUCT_TYPE_MAP); a mismatch silently drops the whole series.
PRIMARY_METRICS: frozenset[str] = frozenset(
    {
        # OpenAQ surface pollutants
        "pm25",
        "pm10",
        "no2",
        "ozone",
        "co",
        "so2",
        # Sentinel-5P column densities (collector prefixes these with s5p_)
        "s5p_no2_column",
        "s5p_so2_column",
        "s5p_co_column",
        "s5p_hcho_column",
    }
)

AUX_METRIC_U = "u_10m"
AUX_METRIC_V = "v_10m"
AUX_METRIC_PBL = "pbl_height"

# GFS cycles are 6 hours apart; ±3h reaches the nearest cycle from any
# primary timestamp without crossing into "stale aux" territory.
AUX_TIME_TOLERANCE: timedelta = timedelta(hours=3)

# Minimum series length to attempt detection at all, set by IsolationForest's
# min_points=50. STL's own floor is cadence-dependent (2*period+1 — 49 for
# hourly, 193 for 15-min) and enforced separately in _engine_for, which drops
# STL and records the reason when a qualifying group is still too short for it.
MIN_GROUP_POINTS: int = 50

# STL models the diurnal cycle, so its period must be samples-per-day for the
# series' actual cadence — period=24 is only right for hourly data, and the
# collected series range from sub-hourly OpenAQ to ~daily Sentinel-5P
# granules. Below this many samples per day there is no diurnal structure to
# decompose; STL is skipped and Z-score (and IF) carry detection.
MIN_STL_SAMPLES_PER_DAY: int = 4


def _stl_period_for(series: list[tuple[datetime, float]]) -> int | None:
    """Samples per diurnal cycle from the observed cadence, or None to skip STL."""
    if len(series) < 2:
        return None
    deltas = sorted(
        (later - earlier).total_seconds()
        for (earlier, _), (later, _) in zip(series, series[1:], strict=False)
        if later > earlier
    )
    if not deltas:
        return None
    median = deltas[len(deltas) // 2]
    period = round(86400.0 / median)
    return period if period >= MIN_STL_SAMPLES_PER_DAY else None


def _engine_for(
    series: list[tuple[datetime, float]],
) -> tuple[DetectionEngine, str | None]:
    """The 3-detector engine plus a reason if STL was dropped (else None).

    STL needs ``2*period+1`` samples for its cadence-derived period, so a
    sub-hourly series resolves to a large period (15-min -> 96 -> floor 193)
    that a 50-192 point group cannot meet. Rather than attach an STL that
    silently returns ``[]``, drop it and surface the reason so the group is not
    reported "ok" while one of three detectors never fired.
    """
    period = _stl_period_for(series)
    if period is None:
        return (
            _three_detector_engine(stl=None),
            "stl skipped: cadence too coarse for diurnal decomposition",
        )
    floor = 2 * period + 1
    if len(series) < floor:
        reason = (
            f"stl skipped: {len(series)} points < {floor} required for "
            f"cadence period {period}"
        )
        return _three_detector_engine(stl=None), reason
    return _three_detector_engine(stl=STLDetector(period=period)), None


def _three_detector_engine(*, stl: STLDetector | None) -> DetectionEngine:
    return DetectionEngine(
        zscore=ZScoreDetector(),
        stl=stl,
        isolation_forest=IsolationForestDetector(),
    )


@dataclass(frozen=True)
class GroupKey:
    """Identifies one time-series: a single station / granule / grid cell."""

    source: str
    metric: str
    source_entity_id: str


@dataclass
class GroupSummary:
    key: GroupKey
    n_points: int
    n_anomalies: int = 0
    n_severe: int = 0
    n_moderate: int = 0
    n_minor: int = 0
    iso_forest_used: bool = False
    skipped_reason: str | None = None
    persist_error: str | None = None


@dataclass
class RunSummary:
    n_groups_examined: int = 0
    n_groups_run: int = 0
    n_anomalies_emitted: int = 0
    n_persisted: int = 0
    groups: list[GroupSummary] = field(default_factory=list)


async def load_points(
    session: AsyncSession,
    *,
    source: str | None = None,
    metric: str | None = None,
    since: datetime | None = None,
) -> list[DataPoint]:
    """Load DataPoints matching the filters, ordered by timestamp."""
    stmt = select(DataPoint).order_by(DataPoint.timestamp)
    if source is not None:
        stmt = stmt.where(DataPoint.source == source)
    if metric is not None:
        stmt = stmt.where(DataPoint.metric == metric)
    if since is not None:
        stmt = stmt.where(DataPoint.timestamp >= since)
    return list((await session.execute(stmt)).scalars().all())


def group_points_by_series(
    points: Sequence[DataPoint],
    *,
    primary_metrics: frozenset[str] = PRIMARY_METRICS,
) -> dict[GroupKey, list[DataPoint]]:
    """Group DataPoints into (source, metric, source_entity_id) time-series.

    Drops points whose metric isn't in ``primary_metrics`` — those are aux
    fields that get joined into IsolationForest inputs separately.
    """
    groups: dict[GroupKey, list[DataPoint]] = defaultdict(list)
    for p in points:
        if p.metric not in primary_metrics:
            continue
        key = GroupKey(
            source=p.source,
            metric=p.metric,
            source_entity_id=p.source_entity_id,
        )
        groups[key].append(p)
    return dict(groups)


@dataclass
class _AuxCache:
    """GFS aux for a whole run, loaded once over the union time window.

    Holds only the station-independent raw data — the per-cell u/v component
    dicts and the raw pbl rows. Each group derives its own distance-ranked,
    window-filtered view in memory (see ``_shape_wind_cells`` /
    ``_shape_aux_cells``) instead of re-querying the same overlapping window.
    """

    u_by_cell: dict[tuple[datetime, str], tuple[float, float, float]]
    v_by_cell: dict[tuple[datetime, str], tuple[float, float, float]]
    pbl_rows: list[tuple[datetime, float, float, float, str]]


async def build_aux_inputs(
    session: AsyncSession,
    primary_points: Sequence[DataPoint],
    *,
    time_tolerance: timedelta = AUX_TIME_TOLERANCE,
    aux_cache: _AuxCache | None = None,
) -> list[IsolationForestInput] | None:
    """Assemble IsolationForest inputs by joining wind speed + pbl_height aux.

    Wind speed is derived from the GFS ``u_10m``/``v_10m`` components;
    ``pbl_height`` comes straight from GFS. Each aux value comes from the
    grid cell nearest the primary series' station, falling back outward only
    when a nearer cell has no row within ``time_tolerance`` — pooling cells
    and matching on time alone would join wind from an arbitrary cell up to
    ~50 km away. Returns ``None`` when either aux is entirely missing — IF
    wants a consistent multivariate feature set per row, and partial aux
    degrades to a univariate IF that the Z-score detector already covers
    better.

    Timestamps with no aux row in range are dropped from the IF input but
    still flow through Z-score and STL via the primary series.

    Pass ``aux_cache`` (built once by ``_load_aux_cache`` over the run's union
    window) to shape this group's view from memory instead of re-querying;
    the cached path is byte-identical to the per-group DB query because each
    group's window is a subset of the cache's window.
    """
    if not primary_points:
        return None

    first_ts = min(p.timestamp for p in primary_points)
    last_ts = max(p.timestamp for p in primary_points)
    window_start = first_ts - time_tolerance
    window_end = last_ts + time_tolerance
    station_lat = primary_points[0].lat
    station_lon = primary_points[0].lon

    if aux_cache is None:
        wind_cells = await _load_gfs_wind_cells(
            session, window_start, window_end, station_lat, station_lon
        )
        pbl_cells = await _load_aux_cells(
            session, AUX_METRIC_PBL, window_start, window_end, station_lat, station_lon
        )
    else:
        wind_cells = _shape_wind_cells(
            aux_cache.u_by_cell,
            aux_cache.v_by_cell,
            window_start,
            window_end,
            station_lat,
            station_lon,
        )
        pbl_cells = _shape_aux_cells(
            aux_cache.pbl_rows, window_start, window_end, station_lat, station_lon
        )

    if not wind_cells or not pbl_cells:
        return None

    iso_inputs: list[IsolationForestInput] = []
    for p in primary_points:
        target = _ensure_utc(p.timestamp)
        wind = _nearest_cell_value(target, wind_cells, time_tolerance)
        pbl = _nearest_cell_value(target, pbl_cells, time_tolerance)
        if wind is None or pbl is None:
            continue
        iso_inputs.append(
            IsolationForestInput(
                timestamp=p.timestamp,
                value=p.value,
                aux_features={"wind_speed": wind, "pbl_height": pbl},
            )
        )
    return iso_inputs or None


async def _load_aux_cache(
    session: AsyncSession,
    primary_points: Sequence[DataPoint],
    *,
    time_tolerance: timedelta = AUX_TIME_TOLERANCE,
) -> _AuxCache | None:
    """Load every group's GFS aux in one pass over the run's union window.

    Each group's per-build window is a subset of ``[min ts - tol, max ts +
    tol]``, so one query per component over that union — filtered per group in
    memory — returns exactly the rows the O(groups) per-group queries would.
    """
    if not primary_points:
        return None
    first_ts = min(p.timestamp for p in primary_points)
    last_ts = max(p.timestamp for p in primary_points)
    window_start = first_ts - time_tolerance
    window_end = last_ts + time_tolerance
    u_by_cell = await _load_gfs_component(
        session, AUX_METRIC_U, window_start, window_end
    )
    v_by_cell = await _load_gfs_component(
        session, AUX_METRIC_V, window_start, window_end
    )
    pbl_rows = await _fetch_aux_rows(
        session, AUX_METRIC_PBL, window_start, window_end
    )
    return _AuxCache(u_by_cell=u_by_cell, v_by_cell=v_by_cell, pbl_rows=pbl_rows)


async def _fetch_aux_rows(
    session: AsyncSession,
    metric: str,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[datetime, float, float, float, str]]:
    """Time-sorted ``(ts, value, lat, lon, entity)`` rows for one aux metric."""
    stmt = (
        select(
            DataPoint.timestamp,
            DataPoint.value,
            DataPoint.lat,
            DataPoint.lon,
            DataPoint.source_entity_id,
        )
        .where(DataPoint.metric == metric)
        .where(DataPoint.timestamp >= window_start)
        .where(DataPoint.timestamp <= window_end)
        .order_by(DataPoint.timestamp)
    )
    return [
        (_ensure_utc(ts), float(value), lat, lon, entity)
        for ts, value, lat, lon, entity in (await session.execute(stmt)).all()
    ]


def _shape_aux_cells(
    rows: Sequence[tuple[datetime, float, float, float, str]],
    window_start: datetime,
    window_end: datetime,
    station_lat: float,
    station_lon: float,
) -> list[list[tuple[datetime, float]]]:
    """One time-sorted series per grid cell, nearest cell to the station first.

    Rows are filtered to ``[window_start, window_end]`` so a cache spanning the
    whole run yields the same per-cell series a per-group query would. Rows
    arrive time-sorted, so each cell's series stays sorted.
    """
    lower = _ensure_utc(window_start)
    upper = _ensure_utc(window_end)
    by_cell: dict[str, tuple[float, list[tuple[datetime, float]]]] = {}
    for ts, value, lat, lon, entity in rows:
        if ts < lower or ts > upper:
            continue
        if entity not in by_cell:
            by_cell[entity] = (distance_km(station_lat, station_lon, lat, lon), [])
        by_cell[entity][1].append((ts, value))
    ranked = sorted(by_cell.values(), key=lambda cell: cell[0])
    return [series for _distance, series in ranked]


async def _load_aux_cells(
    session: AsyncSession,
    metric: str,
    window_start: datetime,
    window_end: datetime,
    station_lat: float,
    station_lon: float,
) -> list[list[tuple[datetime, float]]]:
    """One time-sorted series per grid cell, nearest cell to the station first."""
    rows = await _fetch_aux_rows(session, metric, window_start, window_end)
    return _shape_aux_cells(rows, window_start, window_end, station_lat, station_lon)


async def _load_gfs_wind_cells(
    session: AsyncSession,
    window_start: datetime,
    window_end: datetime,
    station_lat: float,
    station_lon: float,
) -> list[list[tuple[datetime, float]]]:
    """Derive 10-m wind speed per grid cell, nearest cell to the station first.

    GFS stores ``u_10m`` and ``v_10m`` as separate DataPoint metrics; each
    grid cell and cycle carries one of each at a shared
    ``(timestamp, source_entity_id)``. Pair them per cell-cycle and return
    ``wind_speed = hypot(u, v)``. Cells missing a component are skipped.
    """
    u_by_cell = await _load_gfs_component(
        session, AUX_METRIC_U, window_start, window_end
    )
    v_by_cell = await _load_gfs_component(
        session, AUX_METRIC_V, window_start, window_end
    )
    return _shape_wind_cells(
        u_by_cell, v_by_cell, window_start, window_end, station_lat, station_lon
    )


def _shape_wind_cells(
    u_by_cell: dict[tuple[datetime, str], tuple[float, float, float]],
    v_by_cell: dict[tuple[datetime, str], tuple[float, float, float]],
    window_start: datetime,
    window_end: datetime,
    station_lat: float,
    station_lon: float,
) -> list[list[tuple[datetime, float]]]:
    """``wind_speed = hypot(u, v)`` per cell-cycle, nearest cell first.

    Filters the component dicts to ``[window_start, window_end]`` on the shared
    ``(timestamp, cell)`` key; a u row and its paired v row carry the same
    timestamp, so testing u's bound is enough. A cache spanning the whole run
    thus yields the same per-cell series a per-group query would.
    """
    lower = _ensure_utc(window_start)
    upper = _ensure_utc(window_end)
    by_cell: dict[str, tuple[float, list[tuple[datetime, float]]]] = {}
    for key, (u, lat, lon) in u_by_cell.items():
        timestamp, entity = key
        if timestamp < lower or timestamp > upper:
            continue
        v_entry = v_by_cell.get(key)
        if v_entry is None:
            continue
        if entity not in by_cell:
            by_cell[entity] = (distance_km(station_lat, station_lon, lat, lon), [])
        by_cell[entity][1].append((timestamp, math.hypot(u, v_entry[0])))
    for _distance, series in by_cell.values():
        series.sort(key=lambda row: row[0])
    ranked = sorted(by_cell.values(), key=lambda cell: cell[0])
    return [series for _distance, series in ranked]


async def _load_gfs_component(
    session: AsyncSession,
    metric: str,
    window_start: datetime,
    window_end: datetime,
) -> dict[tuple[datetime, str], tuple[float, float, float]]:
    """(value, lat, lon) keyed by (timestamp, grid-cell entity) for one component."""
    stmt = (
        select(
            DataPoint.timestamp,
            DataPoint.source_entity_id,
            DataPoint.value,
            DataPoint.lat,
            DataPoint.lon,
        )
        .where(DataPoint.metric == metric)
        .where(DataPoint.timestamp >= window_start)
        .where(DataPoint.timestamp <= window_end)
    )
    return {
        (_ensure_utc(ts), entity): (float(value), lat, lon)
        for ts, entity, value, lat, lon in (await session.execute(stmt)).all()
    }


def _nearest_cell_value(
    target: datetime,
    cells: list[list[tuple[datetime, float]]],
    tolerance: timedelta,
) -> float | None:
    """The time-nearest value from the spatially nearest cell that has one."""
    for series in cells:
        value = _nearest_value(target, series, tolerance)
        if value is not None:
            return value
    return None


def _ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes from the DB as UTC.

    Postgres ``DateTime(timezone=True)`` returns tz-aware; SQLite (used in
    CI) drops the tzinfo on round-trip. Data is stored in UTC either way,
    so coercing naive → UTC at the read boundary keeps downstream
    timestamp arithmetic consistent.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _nearest_value(
    target: datetime,
    rows: Sequence[tuple[datetime, float]],
    tolerance: timedelta,
) -> float | None:
    """Return the value at the row whose timestamp is closest to ``target``,
    provided that closest row is within ``tolerance``. ``None`` otherwise."""
    best_dt: timedelta | None = None
    best_v: float | None = None
    for ts, v in rows:
        dt = abs(ts - target)
        if dt > tolerance:
            continue
        if best_dt is None or dt < best_dt:
            best_dt = dt
            best_v = v
    return best_v


async def persist_anomalies(
    session: AsyncSession,
    anomalies: Iterable[ConsensusAnomaly],
) -> int:
    """Persist consensus anomalies as Anomaly rows; skip duplicates.

    Idempotent natural key is ``(source, metric, timestamp, lat, lon)``.
    A future migration may promote this to a DB-level UniqueConstraint;
    until then we enforce it at the application layer.
    """
    persisted = 0
    for a in anomalies:
        existing = (
            await session.execute(
                select(Anomaly.id)
                .where(Anomaly.source == a.source)
                .where(Anomaly.metric == a.metric)
                .where(Anomaly.timestamp == a.timestamp)
                .where(Anomaly.lat == a.lat)
                .where(Anomaly.lon == a.lon)
                .limit(1)
            )
        ).scalar_one_or_none()
        if existing is not None:
            continue
        session.add(
            Anomaly(
                timestamp=a.timestamp,
                lat=a.lat,
                lon=a.lon,
                metric=a.metric,
                source=a.source,
                value=a.value,
                expected_value=a.expected_value,
                z_score=a.z_score,
                methods_triggered=list(a.methods_triggered),
                severity=a.severity,
            )
        )
        persisted += 1
    if persisted > 0:
        await session.commit()
    return persisted


async def run_detection(
    session: AsyncSession,
    *,
    source: str | None = None,
    metric: str | None = None,
    since: datetime | None = None,
    dry_run: bool = False,
    min_points: int = MIN_GROUP_POINTS,
) -> RunSummary:
    """Run detection on every qualifying (source, metric, entity) series.

    Skips groups under ``min_points`` (the IF / STL floors). Joins aux
    features when both wind and PBL are present; falls back to a Z-score +
    STL-only engine when aux is missing.
    """
    points = await load_points(
        session, source=source, metric=metric, since=since
    )
    groups = group_points_by_series(points)

    summary = RunSummary(n_groups_examined=len(groups))

    # Load the GFS aux once over the union of all group windows; each group
    # shapes its own view from this cache rather than re-querying the same
    # overlapping window per group.
    aux_cache = await _load_aux_cache(
        session, [p for grp in groups.values() for p in grp]
    )

    for key in sorted(groups, key=lambda k: (k.source, k.metric, k.source_entity_id)):
        group_points = sorted(groups[key], key=lambda p: p.timestamp)

        if len(group_points) < min_points:
            summary.groups.append(
                GroupSummary(
                    key=key,
                    n_points=len(group_points),
                    skipped_reason=(
                        f"insufficient points ({len(group_points)} < {min_points})"
                    ),
                )
            )
            continue

        series = [(p.timestamp, p.value) for p in group_points]
        iso_inputs = await build_aux_inputs(
            session, group_points, aux_cache=aux_cache
        )
        engine, stl_skip_reason = _engine_for(series)
        anomalies = engine.run(
            series,
            metric=key.metric,
            source=key.source,
            lat=group_points[0].lat,
            lon=group_points[0].lon,
            isolation_forest_inputs=iso_inputs,
        )

        group_summary = GroupSummary(
            key=key,
            n_points=len(group_points),
            n_anomalies=len(anomalies),
            n_severe=sum(1 for a in anomalies if a.severity == "severe"),
            n_moderate=sum(1 for a in anomalies if a.severity == "moderate"),
            n_minor=sum(1 for a in anomalies if a.severity == "minor"),
            iso_forest_used=iso_inputs is not None,
            skipped_reason=stl_skip_reason,
        )
        summary.groups.append(group_summary)
        summary.n_groups_run += 1
        summary.n_anomalies_emitted += len(anomalies)
        if not dry_run:
            # Record a per-group persist failure and continue: one group's DB
            # error must not abort the run and discard every other group's
            # summary. rollback() clears the failed transaction so the next
            # group can still commit.
            try:
                summary.n_persisted += await persist_anomalies(session, anomalies)
            except Exception as exc:
                await session.rollback()
                group_summary.persist_error = f"{type(exc).__name__}: {exc}"

    return summary


# CLI --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.detection.run",
        description=(
            "Run anomaly detection on collected DataPoints. "
            "Persists Anomaly rows idempotently."
        ),
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Filter to one data source (openaq, sentinel5p, noaa_gfs, openweather)",
    )
    parser.add_argument(
        "--metric",
        default=None,
        help="Filter to one metric (pm25, ozone, s5p_no2_column, ...)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ISO-format date or datetime; only process points at/after this time",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run detection but don't persist anomalies",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=MIN_GROUP_POINTS,
        help=f"Minimum points in a series to attempt detection (default {MIN_GROUP_POINTS})",
    )
    return parser.parse_args(argv)


def _parse_since(raw: str | None) -> datetime | None:
    if raw is None:
        return None
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _format_summary(summary: RunSummary) -> str:
    lines = [
        f"Groups examined: {summary.n_groups_examined}",
        f"Groups run:      {summary.n_groups_run}",
        f"Anomalies:       {summary.n_anomalies_emitted}",
        f"Persisted:       {summary.n_persisted}",
        "",
        f"{'source':<14} {'metric':<14} {'entity':<28} {'n':>6} "
        f"{'sev':>4} {'mod':>4} {'min':>4} {'IF':>3}  status",
    ]
    for g in summary.groups:
        if g.persist_error:
            status = f"persist failed: {g.persist_error}"
        elif g.skipped_reason:
            status = g.skipped_reason
        else:
            status = "ok"
        if_used = "yes" if g.iso_forest_used else "no"
        lines.append(
            f"{g.key.source:<14} {g.key.metric:<14} "
            f"{g.key.source_entity_id[:28]:<28} {g.n_points:>6} "
            f"{g.n_severe:>4} {g.n_moderate:>4} {g.n_minor:>4} "
            f"{if_used:>3}  {status}"
        )
    return "\n".join(lines)


async def _amain(argv: list[str] | None = None) -> int:
    # Imported lazily so library-mode use (and tests using a SQLite test
    # engine) don't pay for spinning up the production asyncpg engine.
    from app.db.session import async_session, engine_lifecycle

    async with engine_lifecycle():
        args = _parse_args(argv)
        since = _parse_since(args.since)
        async with async_session() as session:
            summary = await run_detection(
                session,
                source=args.source,
                metric=args.metric,
                since=since,
                dry_run=args.dry_run,
                min_points=args.min_points,
            )
        print(_format_summary(summary))
        return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
