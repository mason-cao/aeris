"""Data-ingestion liveness check for the unattended collection host.

The production collectors run unattended on a home server that feeds the eval
freeze; a silent stall (rejected key, crashed scheduler, network outage) would
corrupt the frozen window with nothing failing loudly. Credential preflight only
catches startup, and a collector run can report success while ingesting zero rows
(a rejected key that still resolves cached locations, an empty grid cycle), so
"the process ran" is not the same as "data is landing".

This checks the wall-clock age of the newest stored row per source against a
per-source budget. It is a pure check (:func:`check_liveness`) plus a CLI
(:func:`main`) that exits non-zero when any monitored source is stale, so a
scheduler wrapper can turn a non-zero exit -- or a missed run, via a dead-man's
switch -- into a push/email alert.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.logsetup import configure_logging
from app.collectors.registry import source_choices
from app.db.models import DataPoint
from app.db.session import async_session, engine_lifecycle

logger = logging.getLogger(__name__)

# Max wall-clock age of the newest row per source before it counts as stalled.
# Tuned to each source's real cadence: OpenAQ/OpenWeather poll hourly; GFS
# publishes 6-hourly cycles with NOMADS lag; Sentinel-5P is orbital (~1 overpass
# per day over a fixed point, and cloud/quality filtering drops whole days), so
# it gets a multi-day budget to avoid false alarms.
DEFAULT_BUDGETS_MINUTES: dict[str, float] = {
    "openaq": 180.0,
    "openweather": 180.0,
    "noaa_gfs": 720.0,
    "sentinel5p": 4320.0,
}

# A registered source with no explicit budget above: assume hourly-ish, so a new
# collector is monitored by default rather than silently exempt.
FALLBACK_BUDGET_MINUTES = 180.0


@dataclass(frozen=True)
class SourceLiveness:
    source: str
    last_collected_at: datetime | None
    age_minutes: float | None
    budget_minutes: float

    @property
    def stale(self) -> bool:
        # No rows at all is stale (a source that should be live but never landed
        # data); otherwise stale once the newest row ages past its budget.
        if self.age_minutes is None:
            return True
        return self.age_minutes > self.budget_minutes


@dataclass(frozen=True)
class LivenessReport:
    sources: tuple[SourceLiveness, ...]

    @property
    def stale_sources(self) -> tuple[SourceLiveness, ...]:
        return tuple(source for source in self.sources if source.stale)

    @property
    def healthy(self) -> bool:
        return not self.stale_sources


def _as_utc(value: datetime) -> datetime:
    # SQLite hands back naive datetimes for DateTime(timezone=True); the
    # collectors store UTC (server_default func.now()), so treat naive as UTC.
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _budget_for(source: str, budgets: Mapping[str, float]) -> float:
    return budgets.get(source, FALLBACK_BUDGET_MINUTES)


async def _latest_collected_at(
    session: AsyncSession, sources: Iterable[str]
) -> dict[str, datetime | None]:
    result = await session.execute(
        select(DataPoint.source, func.max(DataPoint.collected_at))
        .where(DataPoint.source.in_(list(sources)))
        .group_by(DataPoint.source)
    )
    return {source: latest for source, latest in result.all()}


async def check_liveness(
    session: AsyncSession,
    *,
    now: datetime | None = None,
    sources: Sequence[str] | None = None,
    budgets: Mapping[str, float] = DEFAULT_BUDGETS_MINUTES,
) -> LivenessReport:
    """Report per-source ingestion freshness against staleness budgets.

    A source is stale when its newest stored row is older than its budget, or
    when it has no rows at all.
    """
    now = now or datetime.now(timezone.utc)
    monitored = tuple(sources) if sources is not None else tuple(source_choices())
    latest = await _latest_collected_at(session, monitored)

    results: list[SourceLiveness] = []
    for source in monitored:
        last = latest.get(source)
        if last is not None:
            last = _as_utc(last)
            age_minutes: float | None = (now - last).total_seconds() / 60.0
        else:
            age_minutes = None
        results.append(
            SourceLiveness(
                source=source,
                last_collected_at=last,
                age_minutes=age_minutes,
                budget_minutes=_budget_for(source, budgets),
            )
        )
    return LivenessReport(sources=tuple(results))


def format_source(result: SourceLiveness) -> str:
    status = "STALE" if result.stale else "ok"
    age = "no-data" if result.age_minutes is None else f"{result.age_minutes:.0f}m"
    last = (
        result.last_collected_at.isoformat()
        if result.last_collected_at is not None
        else "never"
    )
    return " | ".join(
        [
            result.source,
            status,
            f"age={age}",
            f"budget={result.budget_minutes:.0f}m",
            f"last={last}",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check per-source data-ingestion liveness for AERIS."
    )
    parser.add_argument(
        "--source",
        choices=source_choices(),
        help="Check one source instead of all registered sources.",
    )
    parser.add_argument(
        "--max-age-minutes",
        type=float,
        default=None,
        help="Override every checked source's staleness budget with one value.",
    )
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    async with engine_lifecycle():
        configure_logging()
        parser = build_parser()
        args = parser.parse_args(argv)

        sources = (args.source,) if args.source else None
        if args.max_age_minutes is not None:
            if args.max_age_minutes <= 0:
                parser.error("--max-age-minutes must be positive")
            override = tuple(sources) if sources is not None else source_choices()
            budgets: Mapping[str, float] = {
                source: args.max_age_minutes for source in override
            }
        else:
            budgets = DEFAULT_BUDGETS_MINUTES

        async with async_session() as session:
            report = await check_liveness(session, sources=sources, budgets=budgets)

        for result in report.sources:
            print(format_source(result))

        if not report.healthy:
            stale = ", ".join(result.source for result in report.stale_sources)
            logger.error("Liveness check failed; stale sources: %s", stale)
            print(f"LIVENESS ALARM: stale sources: {stale}")
            return 1

        return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
