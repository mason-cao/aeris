"""Freeze the eval anomaly set: dedup into events, rank, take the top N.

The Month 2 inclusion rule is "top-50 summer anomalies by composite
severity". Two things have to be pinned in code before the freeze date for
that rule to be defensible:

- **Event dedup.** One physical event (a regional ozone afternoon, a plume
  crossing stations) gets flagged once per station-hour; without merging,
  a single event could fill half the eval set. Same-metric anomalies within
  ``MERGE_WINDOW`` and ``MERGE_RADIUS_KM`` of an event member join that
  event (single-linkage), and each event contributes exactly one anomaly —
  its strongest member.
- **Composite severity.** Lexicographic: detector-consensus count first
  (the severity tier), then |z| within a tier. Anomalies STL/IF flagged
  without a Z-score rank after any z-scored peer in the same tier.

Output is the frozen fixture ``harness.load_anomaly_set`` consumes, with the
criteria and freeze time recorded alongside the ids.

CLI: ``python -m app.eval.freeze --start 2026-06-01 --end 2026-08-31
--top 50 --out fixtures/eval50.json [--dry-run]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.geo import distance_km
from app.db.models import Anomaly

MERGE_WINDOW = timedelta(minutes=30)
MERGE_RADIUS_KM = 10.0
DEFAULT_TOP_N = 50


def _ensure_utc(dt: datetime) -> datetime:
    """Treat naive datetimes from the DB as UTC (SQLite drops tzinfo)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _rank_key(anomaly: Anomaly) -> tuple[int, float]:
    """Composite severity: (consensus count, |z|), compared lexicographically."""
    z = abs(anomaly.z_score) if anomaly.z_score is not None else float("-inf")
    return (len(anomaly.methods_triggered or []), z)


def group_events(
    anomalies: Sequence[Anomaly],
    *,
    merge_window: timedelta = MERGE_WINDOW,
    merge_radius_km: float = MERGE_RADIUS_KM,
) -> list[list[Anomaly]]:
    """Merge same-metric anomalies into events by spatiotemporal proximity.

    Single-linkage on a time-sorted scan: an anomaly joins the first event
    with any member within the merge window and radius, so an event can
    stretch across hours (a moving plume) without double-counting it.
    """
    events_by_metric: dict[str, list[list[Anomaly]]] = defaultdict(list)
    ordered = sorted(anomalies, key=lambda a: _ensure_utc(a.timestamp))
    for anomaly in ordered:
        ts = _ensure_utc(anomaly.timestamp)
        joined: list[Anomaly] | None = None
        for event in events_by_metric[anomaly.metric]:
            for member in event:
                close_in_time = (
                    abs(ts - _ensure_utc(member.timestamp)) <= merge_window
                )
                if close_in_time and (
                    distance_km(anomaly.lat, anomaly.lon, member.lat, member.lon)
                    <= merge_radius_km
                ):
                    joined = event
                    break
            if joined is not None:
                break
        if joined is None:
            events_by_metric[anomaly.metric].append([anomaly])
        else:
            joined.append(anomaly)
    return [event for events in events_by_metric.values() for event in events]


@dataclass
class FreezeResult:
    """One freeze pass: what was considered, merged, and selected."""

    window_start: datetime
    window_end: datetime
    top_n: int
    n_anomalies: int
    n_events: int
    selected: list[Anomaly]
    event_sizes: dict[uuid.UUID, int]
    missing_enrichment: list[uuid.UUID]


async def load_window_anomalies(
    session: AsyncSession,
    window_start: datetime,
    window_end: datetime,
) -> list[Anomaly]:
    """All anomalies with ``window_start <= timestamp < window_end``."""
    stmt = (
        select(Anomaly)
        .where(Anomaly.timestamp >= window_start)
        .where(Anomaly.timestamp < window_end)
        .order_by(Anomaly.timestamp)
    )
    return list((await session.execute(stmt)).scalars().all())


async def _missing_enrichment_ids(
    session: AsyncSession, anomalies: Sequence[Anomaly]
) -> list[uuid.UUID]:
    if not anomalies:
        return []
    stmt = (
        select(Anomaly.id)
        .where(Anomaly.id.in_([a.id for a in anomalies]))
        .where(~Anomaly.enrichment_records.any())
    )
    return list((await session.execute(stmt)).scalars().all())


async def freeze_eval_set(
    session: AsyncSession,
    *,
    window_start: datetime,
    window_end: datetime,
    top_n: int = DEFAULT_TOP_N,
) -> FreezeResult:
    """Select the top-N event representatives inside the eval window."""
    anomalies = await load_window_anomalies(session, window_start, window_end)
    events = group_events(anomalies)

    representatives = [max(event, key=_rank_key) for event in events]
    representatives.sort(key=lambda a: _ensure_utc(a.timestamp))
    representatives.sort(key=_rank_key, reverse=True)
    selected = representatives[:top_n]

    event_sizes = {
        max(event, key=_rank_key).id: len(event) for event in events
    }
    return FreezeResult(
        window_start=window_start,
        window_end=window_end,
        top_n=top_n,
        n_anomalies=len(anomalies),
        n_events=len(events),
        selected=selected,
        event_sizes=event_sizes,
        missing_enrichment=await _missing_enrichment_ids(session, selected),
    )


def fixture_payload(result: FreezeResult) -> dict:
    """The frozen-set JSON, in the shape ``harness.load_anomaly_set`` reads."""
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start": _ensure_utc(result.window_start).isoformat(),
            "end": _ensure_utc(result.window_end).isoformat(),
        },
        "criteria": {
            "rank": "detector-consensus count, then |z| within a tier",
            "event_merge": {
                "window_minutes": MERGE_WINDOW.total_seconds() / 60.0,
                "radius_km": MERGE_RADIUS_KM,
                "linkage": "single, per metric",
            },
            "top": result.top_n,
        },
        "n_window_anomalies": result.n_anomalies,
        "n_events": result.n_events,
        "anomaly_ids": [str(a.id) for a in result.selected],
    }


def _format_result(result: FreezeResult) -> str:
    lines = [
        f"Window:            {result.window_start:%Y-%m-%d} to "
        f"{result.window_end:%Y-%m-%d} (end exclusive)",
        f"Anomalies in window: {result.n_anomalies}",
        f"Events after merge:  {result.n_events}",
        f"Selected:            {len(result.selected)} (top {result.top_n})",
    ]
    if result.missing_enrichment:
        lines.append(
            f"WARNING: {len(result.missing_enrichment)} selected anomalies "
            "have no EnrichmentRecord — run python -m app.detection.enrichment"
        )
    lines += [
        "",
        f"{'rank':<5} {'anomaly':<10} {'timestamp':<22} {'metric':<14} "
        f"{'sev':<9} {'meth':>4} {'|z|':>7} {'event_n':>7}",
    ]
    for rank, a in enumerate(result.selected, start=1):
        z = f"{abs(a.z_score):.2f}" if a.z_score is not None else "-"
        lines.append(
            f"{rank:<5} {str(a.id)[:8]:<10} "
            f"{_ensure_utc(a.timestamp):%Y-%m-%d %H:%M}    {a.metric:<14} "
            f"{a.severity:<9} {len(a.methods_triggered or []):>4} {z:>7} "
            f"{result.event_sizes.get(a.id, 1):>7}"
        )
    return "\n".join(lines)


def _parse_date(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.freeze",
        description=(
            "Freeze the eval set: merge same-metric anomalies into events, "
            "rank by consensus count then |z|, keep the top N."
        ),
    )
    parser.add_argument(
        "--start",
        required=True,
        help="window start, ISO date or datetime (inclusive, UTC)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help=(
            "window end, ISO date or datetime; a bare date includes that "
            "whole day"
        ),
    )
    parser.add_argument("--top", type=int, default=DEFAULT_TOP_N)
    parser.add_argument(
        "--out",
        default=None,
        help="path for the frozen fixture JSON (required unless --dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the selection without writing the fixture",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is set")
    return args


async def _amain(argv: list[str] | None = None) -> int:
    # Imported lazily so library-mode use (and tests on a SQLite engine)
    # don't pay for spinning up the production asyncpg engine.
    from app.db.session import async_session

    args = _parse_args(argv)
    window_start = _parse_date(args.start)
    window_end = _parse_date(args.end)
    if "T" not in args.end:
        window_end += timedelta(days=1)

    async with async_session() as session:
        result = await freeze_eval_set(
            session,
            window_start=window_start,
            window_end=window_end,
            top_n=args.top,
        )
    print(_format_result(result))
    if not args.dry_run:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(fixture_payload(result), indent=2) + "\n")
        print(f"\nFrozen set written to {out}")
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
