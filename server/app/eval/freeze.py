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
--top 50 --snapshot-sha256 <locked-sha256> --out fixtures/eval50.json
[--dry-run]``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.collectors.geo import distance_km
from app.db.models import Anomaly
from app.eval.baseline_locality_empirics import baseline_locality_manifest_payload
from app.eval.censoring_sensitivity import censoring_manifest_payload
from app.eval.observation_age_empirics import observation_age_manifest_payload
from app.eval.pruning_screen import pruning_manifest_payload
from app.llm.corroboration import (
    DEFAULT_BACKGROUND_TOLERANCE,
    DEFAULT_CHEMISTRY_TOLERANCE,
    DEFAULT_CONCENTRATION_TOLERANCE,
    DEFAULT_SECONDARY_TOLERANCE,
    DEFAULT_SOURCE_TYPE_TOLERANCE,
    DEFAULT_TEMPORAL_TOLERANCE,
    DEFAULT_TRAP_TOLERANCE,
    DEFAULT_WIND_TOLERANCE,
    calm_wind_manifest_payload,
)
from app.provenance.purpleair_qc import (
    LOCKED_SNAPSHOT_SHA256,
    purpleair_qc_manifest_payload,
)
from app.provenance.nomination import nomination_manifest_payload

# 90 min, not 30: the ground sources report hourly, so consecutive-hour flags
# at one station are 60 min apart — a 30 min window could never chain them and
# a 4-hour ozone afternoon at one station would fill four top-N slots, exactly
# the duplication the dedup exists to prevent. 90 min chains consecutive
# hourly flags with slack for reporting jitter while staying well under the
# ~6 h gap that separates genuinely distinct events.
MERGE_WINDOW = timedelta(minutes=90)
MERGE_RADIUS_KM = 10.0
DEFAULT_TOP_N = 50
_B18_DETECTORS = ("isolation_forest", "stl", "zscore")
B18_ACCEPT_UNSTRATIFIED = "accept_unstratified"
B18_STRATIFY = "stratify"
B18_PENDING = "pending_freeze_day_review"
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_FULL_COMMIT_PATTERN = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def validate_snapshot_sha256(value: str) -> str:
    """Return the canonical snapshot hash or fail before freeze selection."""
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("snapshot SHA-256 must be 64 lowercase hexadecimal characters")
    if value != LOCKED_SNAPSHOT_SHA256:
        raise ValueError(
            "snapshot SHA-256 does not match the locked audit snapshot: "
            f"{value} != {LOCKED_SNAPSHOT_SHA256}"
        )
    return value


def validate_code_commit(value: str) -> str:
    """Return a full Git object ID in canonical lowercase hexadecimal form."""
    if _FULL_COMMIT_PATTERN.fullmatch(value) is None:
        raise ValueError(
            "code commit must be a full 40- or 64-character lowercase Git object ID"
        )
    return value


def _run_git(repository_root: Path, command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", None) or str(exc)
        raise RuntimeError(f"Git provenance command failed: {detail}") from exc
    return completed.stdout


def repository_code_commit(repository_root: Path = REPOSITORY_ROOT) -> str:
    """Resolve clean repository HEAD; dirty code cannot claim commit provenance."""
    status = _run_git(
        repository_root,
        ["git", "status", "--porcelain", "--untracked-files=all"],
    )
    dirty_paths = "; ".join(line for line in status.splitlines() if line)
    if dirty_paths:
        raise RuntimeError(f"dirty repository cannot be frozen: {dirty_paths}")
    commit = _run_git(
        repository_root,
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
    ).strip()
    return validate_code_commit(commit)


def threshold_manifest_payload() -> dict[str, dict[str, int | float | None]]:
    """Serialize every live scorer-tolerance owner without copied values."""
    return {
        "atmospheric_trap": asdict(DEFAULT_TRAP_TOLERANCE),
        "background_vs_event": asdict(DEFAULT_BACKGROUND_TOLERANCE),
        "chemistry": asdict(DEFAULT_CHEMISTRY_TOLERANCE),
        "concentration_elevation": asdict(DEFAULT_CONCENTRATION_TOLERANCE),
        "emissions_source_type": asdict(DEFAULT_SOURCE_TYPE_TOLERANCE),
        "secondary_formation": asdict(DEFAULT_SECONDARY_TOLERANCE),
        "temporal_pattern": asdict(DEFAULT_TEMPORAL_TOLERANCE),
        "wind": asdict(DEFAULT_WIND_TOLERANCE),
    }


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

    Single-linkage: an anomaly joins every event holding a member within the
    merge window and radius, and those events are unioned together — so an
    anomaly that sits between two previously separate events bridges them into
    one. This is incremental connected-components over the per-metric
    proximity graph, so the partition is independent of arrival order. An event
    can stretch across hours (a moving plume) without double-counting it.
    """
    events_by_metric: dict[str, list[list[Anomaly]]] = defaultdict(list)
    ordered = sorted(anomalies, key=lambda a: _ensure_utc(a.timestamp))
    for anomaly in ordered:
        ts = _ensure_utc(anomaly.timestamp)
        metric_events = events_by_metric[anomaly.metric]
        matching: list[list[Anomaly]] = []
        for event in metric_events:
            for member in event:
                close_in_time = (
                    abs(ts - _ensure_utc(member.timestamp)) <= merge_window
                )
                if close_in_time and (
                    distance_km(anomaly.lat, anomaly.lon, member.lat, member.lon)
                    <= merge_radius_km
                ):
                    matching.append(event)
                    break
        if not matching:
            metric_events.append([anomaly])
            continue
        primary = matching[0]
        primary.append(anomaly)
        for other in matching[1:]:
            primary.extend(other)
            metric_events.remove(other)
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
    if top_n < 1:
        raise ValueError(f"top_n must be >= 1, got {top_n}")
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


def selection_composition(selected: Sequence[Anomaly]) -> dict[str, int]:
    """Selected-anomaly counts per ``source/metric/severity``.

    Recorded with the fixture (and printed at freeze time) because the rank
    key's reach differs by source — attainable consensus depends on cadence
    (STL) and aux availability (IF), and raw PurpleAir series are spikier than
    regulatory ones — so a skewed composition must be visible on freeze day,
    not discovered post hoc.
    """
    composition: dict[str, int] = {}
    for anomaly in selected:
        key = f"{anomaly.source}/{anomaly.metric}/{anomaly.severity}"
        composition[key] = composition.get(key, 0) + 1
    return dict(sorted(composition.items()))


def _validated_detector_availability(
    anomaly: Anomaly,
) -> dict[str, dict[str, bool | str | None]] | None:
    raw = anomaly.detector_availability_json
    if not isinstance(raw, Mapping) or set(raw) != set(_B18_DETECTORS):
        return None
    normalized: dict[str, dict[str, bool | str | None]] = {}
    for detector in _B18_DETECTORS:
        entry = raw.get(detector)
        if not isinstance(entry, Mapping):
            return None
        ran = entry.get("ran")
        skip_code = entry.get("skip_code")
        detail = entry.get("detail")
        if not isinstance(ran, bool):
            return None
        if ran and skip_code is not None:
            return None
        if not ran and (not isinstance(skip_code, str) or not skip_code):
            return None
        if detail is not None and not isinstance(detail, str):
            return None
        normalized[detector] = {
            "ran": ran,
            "skip_code": skip_code,
            "detail": detail,
        }
    return normalized


def b18_review_payload(
    result: FreezeResult,
    *,
    decision_status: str = B18_PENDING,
    decision_rationale: str | None = None,
) -> dict[str, object]:
    """Deterministic freeze-day provenance report; no skew decision."""
    missing: list[str] = []
    selected_rows: list[dict[str, object]] = []
    series_members: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    method_patterns: dict[str, int] = defaultdict(int)
    detector_counts: dict[str, dict[str, object]] = {
        detector: {
            "ran": 0,
            "skipped": 0,
            "missing": 0,
            "skip_reasons": {},
        }
        for detector in _B18_DETECTORS
    }

    for rank, anomaly in enumerate(result.selected, start=1):
        anomaly_id = str(anomaly.id)
        raw_entity_id = anomaly.source_entity_id
        entity_id = (
            raw_entity_id
            if isinstance(raw_entity_id, str) and raw_entity_id.strip()
            else None
        )
        availability = _validated_detector_availability(anomaly)
        if entity_id is None or availability is None:
            missing.append(anomaly_id)
        if entity_id is not None:
            series_members[(anomaly.source, anomaly.metric, entity_id)].append(
                anomaly_id
            )

        methods = sorted(str(method) for method in (anomaly.methods_triggered or []))
        method_pattern = "+".join(methods) if methods else "(none)"
        method_patterns[method_pattern] += 1
        for detector in _B18_DETECTORS:
            counts = detector_counts[detector]
            if availability is None:
                counts["missing"] = int(counts["missing"]) + 1
                continue
            entry = availability[detector]
            if entry["ran"] is True:
                counts["ran"] = int(counts["ran"]) + 1
            else:
                counts["skipped"] = int(counts["skipped"]) + 1
                skip_reasons = counts["skip_reasons"]
                assert isinstance(skip_reasons, dict)
                skip_code = str(entry["skip_code"])
                skip_reasons[skip_code] = skip_reasons.get(skip_code, 0) + 1

        selected_rows.append(
            {
                "rank": rank,
                "anomaly_id": anomaly_id,
                "timestamp": _ensure_utc(anomaly.timestamp).isoformat(),
                "source": anomaly.source,
                "metric": anomaly.metric,
                "source_entity_id": entity_id,
                "severity": anomaly.severity,
                "methods_triggered": methods,
                "detector_availability": availability,
                "event_size": result.event_sizes.get(anomaly.id, 1),
            }
        )

    trigger_series_counts = {
        "/".join(key): len(anomaly_ids)
        for key, anomaly_ids in sorted(series_members.items())
    }
    repeated_trigger_series = [
        {
            "source": source,
            "metric": metric,
            "source_entity_id": entity_id,
            "selected_count": len(series_members[(source, metric, entity_id)]),
            "anomaly_ids": series_members[(source, metric, entity_id)],
        }
        for source, metric, entity_id in sorted(series_members)
        if len(series_members[(source, metric, entity_id)]) > 1
    ]
    for counts in detector_counts.values():
        skip_reasons = counts["skip_reasons"]
        assert isinstance(skip_reasons, dict)
        counts["skip_reasons"] = dict(sorted(skip_reasons.items()))
    return {
        "schema_version": 1,
        "decision_status": decision_status,
        "decision_rationale": decision_rationale,
        "selected_count": len(result.selected),
        "provenance_complete": not missing,
        "missing_provenance_anomaly_ids": missing,
        "composition": selection_composition(result.selected),
        "triggered_method_patterns": dict(sorted(method_patterns.items())),
        "detector_availability": detector_counts,
        "trigger_series_counts": trigger_series_counts,
        "repeated_trigger_series": repeated_trigger_series,
        "selected": selected_rows,
    }


def fixture_payload(
    result: FreezeResult,
    *,
    snapshot_sha256: str,
    code_commit: str,
    b18_decision: str | None = None,
    b18_rationale: str | None = None,
) -> dict:
    """The frozen-set JSON, in the shape ``harness.load_anomaly_set`` reads."""
    snapshot_sha256 = validate_snapshot_sha256(snapshot_sha256)
    code_commit = validate_code_commit(code_commit)
    pending_review = b18_review_payload(result)
    if pending_review["provenance_complete"] is not True:
        missing = pending_review["missing_provenance_anomaly_ids"]
        raise ValueError(
            "B18 provenance incomplete for selected anomalies: "
            f"{missing}"
        )
    if b18_decision is None:
        raise ValueError("B18 freeze-day decision is required for fixture creation")
    if b18_decision == B18_STRATIFY:
        raise ValueError(
            "B18 stratified selection must be declared and implemented before "
            "fixture creation"
        )
    if b18_decision != B18_ACCEPT_UNSTRATIFIED:
        raise ValueError(f"unknown B18 freeze-day decision: {b18_decision}")
    if b18_rationale is None or not b18_rationale.strip():
        raise ValueError(
            "B18 review rationale is required when accepting unstratified selection"
        )
    freeze_day_review = b18_review_payload(
        result,
        decision_status=b18_decision,
        decision_rationale=b18_rationale.strip(),
    )
    return {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha256": snapshot_sha256,
        "code_commit": code_commit,
        "thresholds": threshold_manifest_payload(),
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
        "composition": selection_composition(result.selected),
        "freeze_day_review": freeze_day_review,
        "data_quality": {
            "baseline_locality": baseline_locality_manifest_payload(),
            "calm_wind_guard": calm_wind_manifest_payload(),
            "censoring": censoring_manifest_payload(),
            "nomination_eligibility": nomination_manifest_payload(),
            "observation_age_gates": observation_age_manifest_payload(),
            "purpleair_time_aware_qc": purpleair_qc_manifest_payload(),
            "variable_pruning": pruning_manifest_payload(),
        },
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
    composition = selection_composition(result.selected)
    if composition:
        lines.append("")
        lines.append("Selected composition (source/metric/severity):")
        lines += [f"  {key:<40} {n:>4}" for key, n in composition.items()]
    review = b18_review_payload(result)
    if review["missing_provenance_anomaly_ids"]:
        lines.append("")
        lines.append(
            "WARNING: B18 provenance missing for: "
            + ", ".join(review["missing_provenance_anomaly_ids"])
        )
    lines.append("")
    lines.append("Detector availability (ran/skipped/missing):")
    detector_availability = review["detector_availability"]
    assert isinstance(detector_availability, dict)
    for detector in _B18_DETECTORS:
        counts = detector_availability[detector]
        lines.append(
            f"  {detector:<20} {counts['ran']:>4}/"
            f"{counts['skipped']:>4}/{counts['missing']:>4}"
        )
    lines.append("")
    lines.append("Trigger series counts:")
    trigger_series_counts = review["trigger_series_counts"]
    assert isinstance(trigger_series_counts, dict)
    if trigger_series_counts:
        lines += [
            f"  {series:<48} {count:>4}"
            for series, count in trigger_series_counts.items()
        ]
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("Repeated trigger series:")
    repeated = review["repeated_trigger_series"]
    assert isinstance(repeated, list)
    if repeated:
        lines += [
            f"  {row['source']}/{row['metric']}/{row['source_entity_id']} "
            f"{row['selected_count']}"
            for row in repeated
        ]
    else:
        lines.append("  (none)")
    lines += [
        "",
        f"{'rank':<5} {'anomaly':<10} {'timestamp':<22} {'metric':<14} "
        f"{'entity':<18} {'sev':<9} {'meth':>4} {'|z|':>7} {'event_n':>7}",
    ]
    for rank, a in enumerate(result.selected, start=1):
        z = f"{abs(a.z_score):.2f}" if a.z_score is not None else "-"
        lines.append(
            f"{rank:<5} {str(a.id)[:8]:<10} "
            f"{_ensure_utc(a.timestamp):%Y-%m-%d %H:%M}    {a.metric:<14} "
            f"{(a.source_entity_id or '-')[:18]:<18} "
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
        "--snapshot-sha256",
        required=True,
        help="locked raw-snapshot SHA-256 represented by the analysis database",
    )
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
    parser.add_argument(
        "--b18-decision",
        choices=(B18_ACCEPT_UNSTRATIFIED, B18_STRATIFY),
        help=(
            "freeze-day composition decision; required for a real fixture, "
            "omitted for the pending dry-run review"
        ),
    )
    parser.add_argument(
        "--b18-rationale",
        help="Mason's nonblank freeze-day composition review rationale",
    )
    args = parser.parse_args(argv)
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is set")
    if not args.dry_run and args.b18_decision is None:
        parser.error("--b18-decision is required unless --dry-run is set")
    if not args.dry_run and (
        args.b18_rationale is None or not args.b18_rationale.strip()
    ):
        parser.error("--b18-rationale is required unless --dry-run is set")
    if args.dry_run and (args.b18_decision is not None or args.b18_rationale):
        parser.error("B18 decision/rationale are recorded only after the dry run")
    return args


async def _amain(argv: list[str] | None = None) -> int:
    # Imported lazily so library-mode use (and tests on a SQLite engine)
    # don't pay for spinning up the production asyncpg engine.
    from app.db.session import async_session, engine_lifecycle

    args = _parse_args(argv)
    snapshot_sha256 = validate_snapshot_sha256(args.snapshot_sha256)
    code_commit = repository_code_commit(REPOSITORY_ROOT)
    async with engine_lifecycle():
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
            payload = fixture_payload(
                result,
                snapshot_sha256=snapshot_sha256,
                code_commit=code_commit,
                b18_decision=args.b18_decision,
                b18_rationale=args.b18_rationale,
            )
            out.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"\nFrozen set written to {out}")
        return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
