"""Label-free cross-source wind-direction disagreement diagnostics.

Measures the statistic the guard in ``app.llm.corroboration`` actually fires
on: the maximum pairwise angular separation of the event-nearest 'from'
bearings that survive their B8 age gate and the B2 calm-wind guard. The
production helpers are imported rather than reimplemented, so the reported
distribution cannot drift away from the scorer.

This reads enrichment summaries, which live in the derived analysis database
rather than in the attested raw snapshot (the snapshot carries no enrichment
rows). The analysis database is hashed before and after the read, and the
frozen fixture's own snapshot hash is checked against the locked value, so the
artifact records the whole provenance chain.

CLI: ``python -m app.eval.wind_disagreement_empirics --database
<analysis.db> --expected-sha256 <sha256>``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sqlite3
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path
from typing import Any

from app.llm.corroboration import (
    DEFAULT_WIND_TOLERANCE,
    WIND_DISAGREEMENT_BEARING_MULTIPLE,
    WIND_DISAGREEMENT_DECLARED_DATE,
    WIND_DISAGREEMENT_STATUS,
    WindTolerance,
    _fresh_nearest_value,
    _gfs_wind_components,
    _metric_block,
    _wind_from_bearing,
    calm_wind_source_decisions,
    wind_disagreement_decision,
    wind_disagreement_manifest_payload,
)
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    STUDY_END_EXCLUSIVE_AT,
    STUDY_START_AT,
)

# The two direction consumers and the sources each one polls, mirroring
# score_transport_direction and score_point_source_attribution.
CONSUMER_SOURCES: dict[str, tuple[str, ...]] = {
    "transport_direction": ("noaa_gfs", "openweather", "asos"),
    "point_source_attribution": ("noaa_gfs", "openweather"),
}

# Declared sweep grid. 90 is the shipped value; 45 is the bearing band itself
# (below which "disagreement" would be incoherent, since two sources inside one
# band can both corroborate); 180 is the guard's upper limit.
SWEEP_THRESHOLDS_DEG: tuple[float, ...] = (45.0, 60.0, 90.0, 120.0, 180.0)


@dataclass(frozen=True)
class EventMeasurement:
    """One anomaly's direction picture for one consumer."""

    anomaly_id: str
    measured_sources: tuple[str, ...]
    votable_sources: tuple[str, ...]
    max_pairwise_deg: float | None
    worst_pair: tuple[str, str] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "measured_sources": list(self.measured_sources),
            "votable_sources": list(self.votable_sources),
            "max_pairwise_deg": self.max_pairwise_deg,
            "worst_pair": list(self.worst_pair) if self.worst_pair else None,
        }


@dataclass(frozen=True)
class ThresholdOutcome:
    """What the guard would do to this population at one threshold."""

    threshold_deg: float
    silenced_events: int
    comparable_events: int
    total_events: int
    share_of_comparable: float | None
    share_of_total: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold_deg": self.threshold_deg,
            "silenced_events": self.silenced_events,
            "comparable_events": self.comparable_events,
            "total_events": self.total_events,
            "share_of_comparable": self.share_of_comparable,
            "share_of_total": self.share_of_total,
        }


@dataclass(frozen=True)
class PopulationEmpirics:
    """Disagreement distribution for one (consumer, population) pair."""

    consumer: str
    population: str
    sources: tuple[str, ...]
    total_events: int
    events_missing_enrichment: int
    measured_source_counts: dict[str, int]
    votable_source_counts: dict[str, int]
    comparable_events: int
    minimum_deg: float | None
    median_deg: float | None
    p90_deg: float | None
    maximum_deg: float | None
    worst_pair_counts: dict[str, int]
    sweep: tuple[ThresholdOutcome, ...]
    worst_events: tuple[EventMeasurement, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "population": self.population,
            "sources": list(self.sources),
            "total_events": self.total_events,
            "events_missing_enrichment": self.events_missing_enrichment,
            "measured_source_counts": self.measured_source_counts,
            "votable_source_counts": self.votable_source_counts,
            "comparable_events": self.comparable_events,
            "minimum_deg": self.minimum_deg,
            "median_deg": self.median_deg,
            "p90_deg": self.p90_deg,
            "maximum_deg": self.maximum_deg,
            "worst_pair_counts": self.worst_pair_counts,
            "sweep": [outcome.to_dict() for outcome in self.sweep],
            "worst_events": [event.to_dict() for event in self.worst_events],
        }


@dataclass(frozen=True)
class WindDisagreementReport:
    analysis_db_sha256: str
    fixture_snapshot_sha256: str | None
    locked_snapshot_sha256: str
    study_start: str
    study_end_exclusive: str
    declared_threshold_deg: float | None
    guard_manifest: dict[str, Any]
    populations: tuple[PopulationEmpirics, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "analysis_db_sha256": self.analysis_db_sha256,
            "fixture_snapshot_sha256": self.fixture_snapshot_sha256,
            "locked_snapshot_sha256": self.locked_snapshot_sha256,
            "study_start": self.study_start,
            "study_end_exclusive": self.study_end_exclusive,
            "declared_threshold_deg": self.declared_threshold_deg,
            "declared_status": WIND_DISAGREEMENT_STATUS,
            "declared_date": WIND_DISAGREEMENT_DECLARED_DATE,
            "bearing_multiple": WIND_DISAGREEMENT_BEARING_MULTIPLE,
            "guard_manifest": self.guard_manifest,
            "sweep_thresholds_deg": list(SWEEP_THRESHOLDS_DEG),
            "populations": [pop.to_dict() for pop in self.populations],
        }


def measured_bearings(summary: dict, sources: tuple[str, ...]) -> dict[str, float]:
    """Event-nearest 'from' bearing per source, via the production helpers."""
    bearings: dict[str, float] = {}
    for source in sources:
        if source == "noaa_gfs":
            u, v, _ = _gfs_wind_components(summary)
            if u is not None and v is not None:
                bearings[source] = _wind_from_bearing(u, v)
            continue
        raw, _ = _fresh_nearest_value(
            source, "wind_direction", _metric_block(summary, source, "wind_direction")
        )
        if raw is not None:
            bearings[source] = float(raw) % 360.0
    return bearings


def measure_event(
    anomaly_id: str,
    summary: dict,
    sources: tuple[str, ...],
) -> EventMeasurement:
    """One event's disagreement, measured the way the scorer measures it."""
    bearings = measured_bearings(summary, sources)
    decisions, _ = calm_wind_source_decisions(
        summary, sources, tolerance=DEFAULT_WIND_TOLERANCE
    )
    votable = {
        source: bearing
        for source, bearing in bearings.items()
        if decisions[source].direction_votable
    }
    # Threshold-free: the sweep applies thresholds afterwards, so the measured
    # separation is recorded once and never re-derived per threshold.
    decision = wind_disagreement_decision(
        votable, tolerance=WindTolerance(max_disagreement_deg=180.0)
    )
    return EventMeasurement(
        anomaly_id=anomaly_id,
        measured_sources=tuple(sorted(bearings)),
        votable_sources=tuple(sorted(votable)),
        max_pairwise_deg=decision.max_pairwise_deg,
        worst_pair=decision.worst_pair,
    )


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


def summarize(
    consumer: str,
    population: str,
    sources: tuple[str, ...],
    measurements: list[EventMeasurement],
    missing_enrichment: int,
) -> PopulationEmpirics:
    """Collapse per-event measurements into the reported distribution."""
    total = len(measurements) + missing_enrichment
    separations = sorted(
        m.max_pairwise_deg for m in measurements if m.max_pairwise_deg is not None
    )
    comparable = len(separations)
    sweep = tuple(
        ThresholdOutcome(
            threshold_deg=threshold,
            silenced_events=sum(1 for value in separations if value > threshold),
            comparable_events=comparable,
            total_events=total,
            share_of_comparable=(
                sum(1 for value in separations if value > threshold) / comparable
                if comparable
                else None
            ),
            share_of_total=(
                sum(1 for value in separations if value > threshold) / total
                if total
                else None
            ),
        )
        for threshold in SWEEP_THRESHOLDS_DEG
    )
    return PopulationEmpirics(
        consumer=consumer,
        population=population,
        sources=sources,
        total_events=total,
        events_missing_enrichment=missing_enrichment,
        measured_source_counts=dict(
            sorted(Counter(len(m.measured_sources) for m in measurements).items())
        ),
        votable_source_counts=dict(
            sorted(Counter(len(m.votable_sources) for m in measurements).items())
        ),
        comparable_events=comparable,
        minimum_deg=separations[0] if separations else None,
        median_deg=statistics.median(separations) if separations else None,
        p90_deg=_percentile(separations, 0.90) if separations else None,
        maximum_deg=separations[-1] if separations else None,
        worst_pair_counts=dict(
            sorted(
                Counter(
                    "/".join(m.worst_pair)
                    for m in measurements
                    if m.worst_pair is not None
                ).items()
            )
        ),
        sweep=sweep,
        worst_events=tuple(
            sorted(
                (m for m in measurements if m.max_pairwise_deg is not None),
                key=lambda m: -(m.max_pairwise_deg or 0.0),
            )[:10]
        ),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _window_anomaly_ids(connection: sqlite3.Connection) -> list[str]:
    rows = connection.execute(
        """
        SELECT id
        FROM anomalies
        WHERE timestamp >= ? AND timestamp < ?
        ORDER BY id
        """,
        (
            STUDY_START_AT.strftime("%Y-%m-%d %H:%M:%S"),
            STUDY_END_EXCLUSIVE_AT.strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )
    return [str(row[0]) for row in rows]


def _collect(
    connection: sqlite3.Connection,
    anomaly_ids: list[str],
) -> tuple[dict[str, list[EventMeasurement]], int]:
    """Measure every consumer in one pass; the summaries average 674 KB."""
    measurements: dict[str, list[EventMeasurement]] = {
        consumer: [] for consumer in CONSUMER_SOURCES
    }
    missing = 0
    for anomaly_id in anomaly_ids:
        row = connection.execute(
            "SELECT cross_source_summary_json FROM enrichment_records "
            "WHERE anomaly_id = ?",
            (anomaly_id,),
        ).fetchone()
        if row is None:
            missing += 1
            continue
        summary = json.loads(row[0])
        for consumer, sources in CONSUMER_SOURCES.items():
            measurements[consumer].append(
                measure_event(anomaly_id, summary, sources)
            )
    return measurements, missing


def run_empirics(
    database_path: Path,
    *,
    expected_sha256: str,
    anomaly_set: Path | None,
    populations: tuple[str, ...],
) -> WindDisagreementReport:
    """Run the disagreement empirics with pre/post analysis-DB verification."""
    resolved = database_path.resolve()
    before_hash = _sha256(resolved)
    if before_hash != expected_sha256:
        raise ValueError(
            f"analysis DB SHA-256 mismatch before read: "
            f"{before_hash} != {expected_sha256}"
        )

    fixture_snapshot: str | None = None
    frozen_ids: list[str] = []
    if anomaly_set is not None:
        fixture = json.loads(anomaly_set.read_text())
        fixture_snapshot = fixture.get("snapshot_sha256")
        if fixture_snapshot != LOCKED_SNAPSHOT_SHA256:
            raise ValueError(
                f"fixture snapshot {fixture_snapshot} is not the locked snapshot "
                f"{LOCKED_SNAPSHOT_SHA256}"
            )
        frozen_ids = list(fixture["anomaly_ids"])

    uri = f"file:{resolved}?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        collected: list[PopulationEmpirics] = []
        for population in populations:
            if population == "frozen":
                if not frozen_ids:
                    raise ValueError(
                        "the frozen population needs --anomaly-set"
                    )
                ids = frozen_ids
            else:
                ids = _window_anomaly_ids(connection)
            measurements, missing = _collect(connection, ids)
            for consumer, sources in CONSUMER_SOURCES.items():
                collected.append(
                    summarize(
                        consumer,
                        population,
                        sources,
                        measurements[consumer],
                        missing,
                    )
                )
    finally:
        connection.close()

    after_hash = _sha256(resolved)
    if after_hash != expected_sha256:
        raise ValueError(
            f"analysis DB SHA-256 mismatch after read: "
            f"{after_hash} != {expected_sha256}"
        )

    return WindDisagreementReport(
        analysis_db_sha256=after_hash,
        fixture_snapshot_sha256=fixture_snapshot,
        locked_snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
        study_start=STUDY_START_AT.astimezone(UTC).isoformat(),
        study_end_exclusive=STUDY_END_EXCLUSIVE_AT.astimezone(UTC).isoformat(),
        declared_threshold_deg=DEFAULT_WIND_TOLERANCE.max_disagreement_deg,
        guard_manifest=dict(wind_disagreement_manifest_payload()),
        populations=tuple(collected),
    )


def _format_value(value: float | int | None, *, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}g}"


def _format_share(value: float | None) -> str:
    return "N/A" if value is None else f"{100.0 * value:.2f}%"


def render_markdown(report: WindDisagreementReport) -> str:
    lines = [
        f"Declared threshold: {report.declared_threshold_deg} deg "
        f"({WIND_DISAGREEMENT_BEARING_MULTIPLE} x bearing band), "
        f"{WIND_DISAGREEMENT_STATUS}, {WIND_DISAGREEMENT_DECLARED_DATE}",
        f"Analysis DB SHA-256: {report.analysis_db_sha256}",
        f"Locked snapshot SHA-256: {report.locked_snapshot_sha256}",
        "",
        "### Disagreement distribution",
        "",
        "| Consumer | Population | Events | Missing enrichment | Comparable "
        "| Min | Median | p90 | Max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pop in report.populations:
        lines.append(
            "| "
            + " | ".join(
                (
                    pop.consumer,
                    pop.population,
                    str(pop.total_events),
                    str(pop.events_missing_enrichment),
                    str(pop.comparable_events),
                    _format_value(pop.minimum_deg),
                    _format_value(pop.median_deg),
                    _format_value(pop.p90_deg),
                    _format_value(pop.maximum_deg),
                )
            )
            + " |"
        )

    lines += [
        "",
        "### Threshold sweep (events the guard would silence)",
        "",
        "| Consumer | Population | Threshold (deg) | Silenced | Of comparable "
        "| Of all events |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for pop in report.populations:
        for outcome in pop.sweep:
            lines.append(
                "| "
                + " | ".join(
                    (
                        pop.consumer,
                        pop.population,
                        _format_value(outcome.threshold_deg),
                        str(outcome.silenced_events),
                        _format_share(outcome.share_of_comparable),
                        _format_share(outcome.share_of_total),
                    )
                )
                + " |"
            )

    lines += [
        "",
        "### Votable source counts (after age gate and calm-wind guard)",
        "",
        "| Consumer | Population | Distribution | Worst pairs |",
        "|---|---|---|---|",
    ]
    for pop in report.populations:
        lines.append(
            "| "
            + " | ".join(
                (
                    pop.consumer,
                    pop.population,
                    json.dumps(pop.votable_source_counts, sort_keys=True),
                    json.dumps(pop.worst_pair_counts, sort_keys=True),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.wind_disagreement_empirics",
        description=(
            "Measure cross-source wind-direction disagreement without labels."
        ),
    )
    parser.add_argument(
        "--database",
        required=True,
        type=Path,
        help="read-only analysis SQLite database holding enrichment_records",
    )
    parser.add_argument(
        "--expected-sha256", required=True, help="recorded analysis DB SHA-256"
    )
    parser.add_argument(
        "--anomaly-set",
        type=Path,
        default=None,
        help="frozen fixture; required for the frozen population",
    )
    parser.add_argument(
        "--population",
        choices=("frozen", "window", "both"),
        default="both",
        help="which anomalies to measure (default: both)",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON artifact path")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    populations = (
        ("frozen", "window") if args.population == "both" else (args.population,)
    )
    report = run_empirics(
        args.database,
        expected_sha256=args.expected_sha256,
        anomaly_set=args.anomaly_set,
        populations=populations,
    )
    payload = report.to_dict()
    if args.output is not None:
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if args.format == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_markdown(report))


if __name__ == "__main__":
    main()
