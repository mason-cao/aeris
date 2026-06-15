"""Cross-source enrichment.

Pulls a configurable window of context (72 h centred on the anomaly by
default) from every collected source within a spatial radius, and packs it
into the ``cross_source_summary_json`` payload of an
:class:`~app.db.models.EnrichmentRecord`.

That payload is the contract the corroboration scorer consumes: for each
source it preserves the per-entity, per-metric time-series inside the
spatiotemporal window, plus convenience aggregates (value range, the
observation nearest the anomaly in time). The summary is *pure context* —
no scoring and no physics derivations — so the scorer owns every judgement
call and the enrichment stays cheap to re-run.

The four collected sources sit on four different spatial conventions:
point OpenAQ stations, the 5-point OpenWeather grid, the 0.25 deg NOAA GFS
grid, and the single-anchor Sentinel-5P granule mean. Enrichment treats
them uniformly — every DataPoint carries its own ``(lat, lon)`` and a
``source_entity_id``, so grouping by ``(source, metric, entity)`` and
tagging each entity with its great-circle distance from the anomaly works
the same for all four.
"""

from __future__ import annotations

import argparse
import asyncio
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.geo import BoundingBox, distance_km, offset_coordinate
from app.db.models import Anomaly, DataPoint, EnrichmentRecord


# Bumped when the cross_source_summary_json shape changes; the corroboration
# scorer keys its parsing off this.
SCHEMA_VERSION = 1

# The sources Month 1 brought live. Coverage is reported against this set so
# a silent collector shows up as an explicit gap rather than absent JSON.
# Rows from any other source are still enriched, just not coverage-scored.
EXPECTED_SOURCES: tuple[str, ...] = (
    "openaq",
    "openweather",
    "noaa_gfs",
    "sentinel5p",
)


@dataclass(frozen=True)
class EnrichmentConfig:
    """Window geometry for one enrichment pass.

    ``hours_before``/``hours_after`` are kept separate rather than a single
    span so the window can be centred (the default, 36 + 36 = 72 h) or
    skewed toward history without changing any call site.
    """

    hours_before: float = 36.0
    hours_after: float = 36.0
    spatial_radius_km: float = 50.0


DEFAULT_CONFIG = EnrichmentConfig()


def _ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC.

    SQLite (CI) drops tzinfo on round-trip while Postgres keeps it; data is
    stored in UTC either way, so coercing at the boundary keeps window
    arithmetic and ISO serialization consistent. Mirrors
    ``app.detection.run._ensure_utc``.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _iso(dt: datetime) -> str:
    """ISO-8601 string for a UTC-coerced datetime."""
    return _ensure_utc(dt).isoformat()


def context_window(
    timestamp: datetime,
    config: EnrichmentConfig = DEFAULT_CONFIG,
) -> tuple[datetime, datetime]:
    """``(start, end)`` of the context window centred on ``timestamp``."""
    centre = _ensure_utc(timestamp)
    return (
        centre - timedelta(hours=config.hours_before),
        centre + timedelta(hours=config.hours_after),
    )


def anomaly_bounding_box(lat: float, lon: float, radius_km: float) -> BoundingBox:
    """Lat/lon box enclosing ``radius_km`` around a point.

    A cheap pre-filter for the DB query; callers still apply an exact
    great-circle test, since the box corners sit ~1.4x ``radius_km`` away.
    """
    ne_lat, ne_lon = offset_coordinate(
        lat, lon, north_km=radius_km, east_km=radius_km
    )
    sw_lat, sw_lon = offset_coordinate(
        lat, lon, north_km=-radius_km, east_km=-radius_km
    )
    return BoundingBox(
        min_lat=sw_lat, max_lat=ne_lat, min_lon=sw_lon, max_lon=ne_lon
    )


@dataclass(frozen=True)
class _Entry:
    """One in-window DataPoint paired with the facts derived for it."""

    point: DataPoint
    ts: datetime  # UTC-coerced timestamp
    distance_km: float  # great-circle distance from the anomaly


def build_cross_source_summary(
    anomaly: Anomaly,
    points: Sequence[DataPoint],
    *,
    window_start: datetime,
    window_end: datetime,
    config: EnrichmentConfig = DEFAULT_CONFIG,
) -> dict:
    """Pack in-window cross-source context into the EnrichmentRecord payload.

    Pure: no DB, no IO. ``points`` may be over-fetched — anything outside
    the window or beyond ``config.spatial_radius_km`` is dropped here too,
    so the result is correct regardless of how tightly the caller queried.
    """
    anomaly_ts = _ensure_utc(anomaly.timestamp)
    start = _ensure_utc(window_start)
    end = _ensure_utc(window_end)

    by_source: dict[str, list[_Entry]] = defaultdict(list)
    for point in points:
        point_ts = _ensure_utc(point.timestamp)
        if point_ts < start or point_ts > end:
            continue
        dist = distance_km(anomaly.lat, anomaly.lon, point.lat, point.lon)
        if dist > config.spatial_radius_km:
            continue
        by_source[point.source].append(_Entry(point, point_ts, dist))

    sources = {
        source: _summarize_source(by_source[source], anomaly_ts)
        for source in sorted(by_source)
    }
    coverage = {source: source in sources for source in EXPECTED_SOURCES}

    return {
        "schema_version": SCHEMA_VERSION,
        "anomaly": {
            "id": str(anomaly.id) if anomaly.id is not None else None,
            "timestamp": _iso(anomaly_ts),
            "lat": anomaly.lat,
            "lon": anomaly.lon,
            "metric": anomaly.metric,
            "source": anomaly.source,
            "value": anomaly.value,
            "severity": anomaly.severity,
        },
        "window": {
            "start": _iso(start),
            "end": _iso(end),
            "hours_before": config.hours_before,
            "hours_after": config.hours_after,
            "spatial_radius_km": config.spatial_radius_km,
        },
        "coverage": coverage,
        "sources": sources,
    }


def _summarize_source(entries: list[_Entry], anomaly_ts: datetime) -> dict:
    """Group one source's in-window entries by metric."""
    by_metric: dict[str, list[_Entry]] = defaultdict(list)
    for entry in entries:
        by_metric[entry.point.metric].append(entry)
    return {
        "n_entities": len({e.point.source_entity_id for e in entries}),
        "n_points": len(entries),
        "metrics": {
            metric: _summarize_metric(by_metric[metric], anomaly_ts)
            for metric in sorted(by_metric)
        },
    }


def _summarize_metric(entries: list[_Entry], anomaly_ts: datetime) -> dict:
    """Group one (source, metric)'s entries into per-entity time-series."""
    by_entity: dict[str, list[_Entry]] = defaultdict(list)
    for entry in entries:
        by_entity[entry.point.source_entity_id].append(entry)

    entity_summaries: list[dict] = []
    for entity_id in sorted(by_entity):
        ent_entries = sorted(by_entity[entity_id], key=lambda e: e.ts)
        representative = ent_entries[0]
        entity_summaries.append(
            {
                "entity_id": entity_id,
                "lat": representative.point.lat,
                "lon": representative.point.lon,
                "distance_km": round(representative.distance_km, 3),
                "n_points": len(ent_entries),
                "series": [[_iso(e.ts), e.point.value] for e in ent_entries],
            }
        )
    entity_summaries.sort(key=lambda e: (e["distance_km"], e["entity_id"]))

    values = [e.point.value for e in entries]
    return {
        "unit": entries[0].point.unit,
        "n_points": len(entries),
        "n_entities": len(entity_summaries),
        "value_range": {
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 4),
        },
        "nearest_in_time": _nearest_in_time(entries, anomaly_ts),
        "entities": entity_summaries,
    }


def _nearest_in_time(entries: list[_Entry], anomaly_ts: datetime) -> dict:
    """The single observation closest to the anomaly in time (ties: distance)."""
    nearest = min(
        entries,
        key=lambda e: (abs(e.ts - anomaly_ts), e.distance_km),
    )
    return {
        "t": _iso(nearest.ts),
        "v": nearest.point.value,
        "entity_id": nearest.point.source_entity_id,
        "distance_km": round(nearest.distance_km, 3),
        "dt_minutes": round(
            abs(nearest.ts - anomaly_ts).total_seconds() / 60.0, 1
        ),
    }


async def _query_box_window(
    session: AsyncSession,
    box: BoundingBox,
    window_start: datetime,
    window_end: datetime,
) -> list[DataPoint]:
    """Every DataPoint inside a lat/lon box and time window, ts-ordered.

    The coarse SQL pre-filter, shared by the per-anomaly and batched paths;
    callers apply the exact great-circle radius themselves (SQLite, used in
    CI, has no spatial functions).
    """
    stmt = (
        select(DataPoint)
        .where(DataPoint.timestamp >= window_start)
        .where(DataPoint.timestamp <= window_end)
        .where(DataPoint.lat >= box.min_lat)
        .where(DataPoint.lat <= box.max_lat)
        .where(DataPoint.lon >= box.min_lon)
        .where(DataPoint.lon <= box.max_lon)
        .order_by(DataPoint.timestamp)
    )
    return list((await session.execute(stmt)).scalars().all())


async def load_context_points(
    session: AsyncSession,
    *,
    lat: float,
    lon: float,
    window_start: datetime,
    window_end: datetime,
    radius_km: float,
) -> list[DataPoint]:
    """Load every DataPoint inside the spatiotemporal context window.

    A lat/lon bounding box does the coarse filtering in SQL; the exact
    great-circle radius is applied in Python because SQLite (CI) has no
    spatial functions. Rows come back ordered by timestamp.
    """
    box = anomaly_bounding_box(lat, lon, radius_km)
    rows = await _query_box_window(session, box, window_start, window_end)
    return [
        row
        for row in rows
        if distance_km(lat, lon, row.lat, row.lon) <= radius_km
    ]


async def enrich_anomaly(
    session: AsyncSession,
    anomaly: Anomaly,
    *,
    config: EnrichmentConfig = DEFAULT_CONFIG,
    context_points: Sequence[DataPoint] | None = None,
) -> EnrichmentRecord:
    """Build the EnrichmentRecord for one anomaly. Does not persist it.

    Pass ``context_points`` (a possibly over-fetched set from
    ``_batch_load_context``) to skip this anomaly's own DB query;
    ``build_cross_source_summary`` re-applies the exact window and radius, so
    a wider batch set yields the same record as a per-anomaly query.
    """
    window_start, window_end = context_window(anomaly.timestamp, config)
    if context_points is None:
        context_points = await load_context_points(
            session,
            lat=anomaly.lat,
            lon=anomaly.lon,
            window_start=window_start,
            window_end=window_end,
            radius_km=config.spatial_radius_km,
        )
    summary = build_cross_source_summary(
        anomaly,
        context_points,
        window_start=window_start,
        window_end=window_end,
        config=config,
    )
    return EnrichmentRecord(
        anomaly_id=anomaly.id,
        context_window_start=window_start,
        context_window_end=window_end,
        cross_source_summary_json=summary,
    )


async def persist_enrichment(
    session: AsyncSession,
    record: EnrichmentRecord,
) -> bool:
    """Insert an EnrichmentRecord unless the anomaly already has one.

    Insert-only idempotency keyed on ``anomaly_id`` — mirrors
    ``app.detection.run.persist_anomalies``. Re-enriching a changed anomaly
    means deleting its existing ``enrichment_records`` row first. Returns
    ``True`` when a row was inserted.
    """
    existing = (
        await session.execute(
            select(EnrichmentRecord.id)
            .where(EnrichmentRecord.anomaly_id == record.anomaly_id)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return False
    session.add(record)
    await session.commit()
    return True


@dataclass
class EnrichmentLine:
    """One anomaly's outcome in an enrichment run."""

    anomaly_id: str
    timestamp: datetime
    metric: str
    source: str
    severity: str
    n_points: int
    sources_present: list[str]
    persisted: bool
    error: str | None = None


@dataclass
class EnrichmentSummary:
    """Outcome of an :func:`enrich_pending_anomalies` pass."""

    n_pending: int = 0
    n_persisted: int = 0
    lines: list[EnrichmentLine] = field(default_factory=list)


def _line_for(
    anomaly: Anomaly,
    record: EnrichmentRecord,
    persisted: bool,
) -> EnrichmentLine:
    summary = record.cross_source_summary_json
    return EnrichmentLine(
        anomaly_id=str(anomaly.id),
        timestamp=_ensure_utc(anomaly.timestamp),
        metric=anomaly.metric,
        source=anomaly.source,
        severity=anomaly.severity,
        n_points=sum(src["n_points"] for src in summary["sources"].values()),
        sources_present=sorted(summary["sources"]),
        persisted=persisted,
    )


def _error_line_for(anomaly: Anomaly, exc: Exception) -> EnrichmentLine:
    """A line for an anomaly whose enrichment or persist failed mid-pass."""
    return EnrichmentLine(
        anomaly_id=str(anomaly.id),
        timestamp=_ensure_utc(anomaly.timestamp),
        metric=anomaly.metric,
        source=anomaly.source,
        severity=anomaly.severity,
        n_points=0,
        sources_present=[],
        persisted=False,
        error=f"{type(exc).__name__}: {exc}",
    )


async def _batch_load_context(
    session: AsyncSession,
    anomalies: Sequence[Anomaly],
    *,
    config: EnrichmentConfig = DEFAULT_CONFIG,
) -> dict[int, list[DataPoint]]:
    """Context points per anomaly index, one DB query per spatiotemporal bucket.

    Anomalies are bucketed by a coarse grid sized to the context window and
    radius; one query over each bucket's union envelope replaces the per-
    anomaly bbox scan (the N+1). Correctness is grouping-independent —
    ``build_cross_source_summary`` re-filters each anomaly to its own window
    and radius — so the bucket only governs over-fetch, not the result. A
    failed bucket query is left unmapped so its anomalies fall back to their
    own per-anomaly query (preserving the per-anomaly read isolation).
    """
    window_span_h = config.hours_before + config.hours_after or 1.0
    bucket_seconds = window_span_h * 3600.0
    # ~2x radius in degrees so genuine near-neighbours co-bucket without
    # dragging distant anomalies into one oversized envelope.
    grid_deg = max(2.0 * config.spatial_radius_km / 111.0, 1e-6)

    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, anomaly in enumerate(anomalies):
        ts = _ensure_utc(anomaly.timestamp)
        t_key = int(ts.timestamp() // bucket_seconds)
        lat_key = math.floor(anomaly.lat / grid_deg)
        lon_key = math.floor(anomaly.lon / grid_deg)
        buckets[(t_key, lat_key, lon_key)].append(i)

    context: dict[int, list[DataPoint]] = {}
    for idxs in buckets.values():
        boxes: list[BoundingBox] = []
        starts: list[datetime] = []
        ends: list[datetime] = []
        for i in idxs:
            anomaly = anomalies[i]
            ws, we = context_window(anomaly.timestamp, config)
            starts.append(ws)
            ends.append(we)
            boxes.append(
                anomaly_bounding_box(
                    anomaly.lat, anomaly.lon, config.spatial_radius_km
                )
            )
        union_box = BoundingBox(
            min_lat=min(b.min_lat for b in boxes),
            max_lat=max(b.max_lat for b in boxes),
            min_lon=min(b.min_lon for b in boxes),
            max_lon=max(b.max_lon for b in boxes),
        )
        try:
            points = await _query_box_window(
                session, union_box, min(starts), max(ends)
            )
        except Exception:
            continue
        # Detach the loaded rows: a per-anomaly rollback in the enrich loop
        # would otherwise expire these shared instances, and reading an expired
        # column later triggers a lazy reload outside the async greenlet. Their
        # columns are fully populated at query time, so detached reads are safe.
        for point in points:
            session.expunge(point)
        for i in idxs:
            context[i] = points
    return context


async def enrich_pending_anomalies(
    session: AsyncSession,
    *,
    config: EnrichmentConfig = DEFAULT_CONFIG,
    severity: str | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> EnrichmentSummary:
    """Enrich every anomaly that has no EnrichmentRecord yet.

    Each record is committed individually, so an interrupted run leaves the
    completed records persisted and a re-run picks up only the remainder.
    """
    stmt = select(Anomaly).where(~Anomaly.enrichment_records.any())
    if severity is not None:
        stmt = stmt.where(Anomaly.severity == severity)
    stmt = stmt.order_by(Anomaly.timestamp)
    if limit is not None:
        stmt = stmt.limit(limit)

    anomalies = list((await session.execute(stmt)).scalars().all())
    # Detach the loaded anomalies so a mid-pass rollback (below) can't expire
    # the ones still queued: async SQLAlchemy can't lazily reload an expired
    # column outside the greenlet. We only ever read their columns here.
    for anomaly in anomalies:
        session.expunge(anomaly)
    summary = EnrichmentSummary(n_pending=len(anomalies))

    # One context query per spatiotemporal bucket instead of one per anomaly.
    context_by_index = await _batch_load_context(session, anomalies, config=config)

    for index, anomaly in enumerate(anomalies):
        # Record a per-anomaly failure and continue: one anomaly's read or
        # persist error must not abort the pass and lose every other line.
        # rollback() clears the failed transaction so the next anomaly commits.
        try:
            record = await enrich_anomaly(
                session,
                anomaly,
                config=config,
                context_points=context_by_index.get(index),
            )
            persisted = False
            if not dry_run:
                persisted = await persist_enrichment(session, record)
                if persisted:
                    summary.n_persisted += 1
            summary.lines.append(_line_for(anomaly, record, persisted))
        except Exception as exc:
            # Build the line before rollback() expires the anomaly's loaded
            # columns (which would trigger a lazy reload outside the greenlet).
            line = _error_line_for(anomaly, exc)
            await session.rollback()
            summary.lines.append(line)

    return summary


# CLI --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.detection.enrichment",
        description=(
            "Enrich detected anomalies with cross-source context. "
            "Inserts one EnrichmentRecord per anomaly, idempotently."
        ),
    )
    parser.add_argument(
        "--severity",
        default=None,
        help="Only enrich anomalies of this severity (minor, moderate, severe)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after enriching this many anomalies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build enrichment records but do not persist them",
    )
    return parser.parse_args(argv)


def _format_summary(summary: EnrichmentSummary) -> str:
    lines = [
        f"Pending anomalies: {summary.n_pending}",
        f"Persisted:         {summary.n_persisted}",
        "",
        f"{'anomaly':<10} {'severity':<10} {'metric':<14} {'pts':>6} "
        f"{'saved':>6}  sources",
    ]
    for line in summary.lines:
        saved = "ERROR" if line.error else ("yes" if line.persisted else "no")
        sources = line.error if line.error else (",".join(line.sources_present) or "(none)")
        lines.append(
            f"{line.anomaly_id[:8]:<10} {line.severity:<10} "
            f"{line.metric:<14} {line.n_points:>6} {saved:>6}  {sources}"
        )
    return "\n".join(lines)


async def _amain(argv: list[str] | None = None) -> int:
    # Imported lazily so library-mode use (and tests on a SQLite engine)
    # don't pay for spinning up the production asyncpg engine.
    from app.db.session import async_session

    args = _parse_args(argv)
    async with async_session() as session:
        summary = await enrich_pending_anomalies(
            session,
            severity=args.severity,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    print(_format_summary(summary))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
