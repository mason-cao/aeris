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

from app.db.models import Anomaly, DataPoint

from .consensus import ConsensusAnomaly
from .engine import DetectionEngine
from .isolation_forest import IsolationForestInput


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

# Minimum series length to attempt detection. IsolationForest defaults to
# min_points=50 and STL needs 2*period+1=49 for hourly data with daily
# seasonality, so 50 is the binding floor.
MIN_GROUP_POINTS: int = 50


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


async def build_aux_inputs(
    session: AsyncSession,
    primary_points: Sequence[DataPoint],
    *,
    time_tolerance: timedelta = AUX_TIME_TOLERANCE,
) -> list[IsolationForestInput] | None:
    """Assemble IsolationForest inputs by joining wind speed + pbl_height aux.

    Wind speed is derived from the GFS ``u_10m``/``v_10m`` components;
    ``pbl_height`` comes straight from GFS. Returns ``None`` when either aux
    is entirely missing — IF wants a consistent multivariate feature set per
    row, and partial aux degrades to a univariate IF that the Z-score
    detector already covers better.

    Per-timestamp matching is nearest-neighbor within ``time_tolerance``.
    Timestamps with no aux row in range are dropped from the IF input but
    still flow through Z-score and STL via the primary series.
    """
    if not primary_points:
        return None

    first_ts = min(p.timestamp for p in primary_points)
    last_ts = max(p.timestamp for p in primary_points)
    window_start = first_ts - time_tolerance
    window_end = last_ts + time_tolerance

    wind_rows = await _load_gfs_wind_rows(session, window_start, window_end)
    pbl_rows = await _load_aux_rows(
        session, AUX_METRIC_PBL, window_start, window_end
    )

    if not wind_rows or not pbl_rows:
        return None

    iso_inputs: list[IsolationForestInput] = []
    for p in primary_points:
        wind = _nearest_value(p.timestamp, wind_rows, time_tolerance)
        pbl = _nearest_value(p.timestamp, pbl_rows, time_tolerance)
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


async def _load_aux_rows(
    session: AsyncSession,
    metric: str,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[datetime, float]]:
    stmt = (
        select(DataPoint.timestamp, DataPoint.value)
        .where(DataPoint.metric == metric)
        .where(DataPoint.timestamp >= window_start)
        .where(DataPoint.timestamp <= window_end)
        .order_by(DataPoint.timestamp)
    )
    return [
        (_ensure_utc(ts), float(v))
        for ts, v in (await session.execute(stmt)).all()
    ]


async def _load_gfs_wind_rows(
    session: AsyncSession,
    window_start: datetime,
    window_end: datetime,
) -> list[tuple[datetime, float]]:
    """Derive 10-m wind speed from the GFS u/v components.

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

    wind: list[tuple[datetime, float]] = []
    for cell, u in u_by_cell.items():
        v = v_by_cell.get(cell)
        if v is None:
            continue
        timestamp, _entity = cell
        wind.append((timestamp, math.hypot(u, v)))
    wind.sort(key=lambda row: row[0])
    return wind


async def _load_gfs_component(
    session: AsyncSession,
    metric: str,
    window_start: datetime,
    window_end: datetime,
) -> dict[tuple[datetime, str], float]:
    """Load one GFS wind component keyed by (timestamp, grid-cell entity)."""
    stmt = (
        select(DataPoint.timestamp, DataPoint.source_entity_id, DataPoint.value)
        .where(DataPoint.metric == metric)
        .where(DataPoint.timestamp >= window_start)
        .where(DataPoint.timestamp <= window_end)
    )
    return {
        (_ensure_utc(ts), entity): float(value)
        for ts, entity, value in (await session.execute(stmt)).all()
    }


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
        iso_inputs = await build_aux_inputs(session, group_points)
        engine = DetectionEngine.with_defaults()
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
        )
        summary.groups.append(group_summary)
        summary.n_groups_run += 1
        summary.n_anomalies_emitted += len(anomalies)
        if not dry_run:
            summary.n_persisted += await persist_anomalies(session, anomalies)

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
        status = g.skipped_reason if g.skipped_reason else "ok"
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
    from app.db.session import async_session

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
