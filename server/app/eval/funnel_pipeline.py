"""Disposable database orchestration and extraction for the B19 funnel."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from alembic import command
from alembic.config import Config
from sqlalchemy import func, inspect, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.db.migrate import _to_utc
from app.db.models import (
    Anomaly,
    Base,
    Claim,
    EnrichmentRecord,
    ExpertLabel,
    Explanation,
)
from app.detection.enrichment import enrich_pending_anomalies
from app.detection.run import run_detection
from app.eval.freeze import FreezeResult, freeze_eval_set, repository_code_commit
from app.eval.funnel_dry_run import (
    build_atomicity_worksheet,
    build_funnel_report,
    canonical_json,
    select_funnel_anomalies,
    write_iteration_reports,
)
from app.eval.harness import (
    DEFAULT_MODELS,
    ModelSweepSummary,
    run_harness,
)
from app.llm.client_base import LLMClient
from app.llm.corroboration import (
    ClaimType,
    calm_wind_source_decisions,
    direction_data_sources,
)
from app.llm.explain import make_client
from app.llm.observation_age import DEFAULT_OBSERVATION_AGE_GATES
from app.provenance.purpleair_qc import LOCKED_SNAPSHOT_SHA256


WINDOW_START: Final = datetime(2026, 6, 1, tzinfo=timezone.utc)
WINDOW_END: Final = datetime(2026, 7, 13, tzinfo=timezone.utc)
SERVER_ROOT: Final = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT: Final = SERVER_ROOT.parent
_RUN_DATE_RE = re.compile(r"\d{8}\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_BASE_TABLES: Final = ("data_points", "data_sources")
_DERIVED_MODELS: Final = (
    Anomaly,
    EnrichmentRecord,
    Explanation,
    Claim,
    ExpertLabel,
)
_WIND_SOURCES: Final = ("noaa_gfs", "openweather", "asos")


class FunnelPipelineError(RuntimeError):
    """Raised when disposable isolation or extraction cannot be proven."""


@dataclass(frozen=True)
class IterationPaths:
    database: Path
    building_database: Path
    payload: Path
    worksheet: Path
    manual_template: Path
    parse_failures: Path


def iteration_paths(
    output_dir: Path,
    *,
    run_date: str,
    iteration: int,
) -> IterationPaths:
    """Return every deterministic, iteration-scoped B19 artifact path."""
    if _RUN_DATE_RE.fullmatch(run_date) is None:
        raise FunnelPipelineError("run_date must use YYYYMMDD")
    try:
        datetime.strptime(run_date, "%Y%m%d")
    except ValueError as exc:
        raise FunnelPipelineError("run_date must be a valid YYYYMMDD date") from exc
    if type(iteration) is not int or iteration < 1:
        raise FunnelPipelineError("iteration must be a positive integer")
    database = output_dir / (
        f"aeris-b19-funnel-{run_date}-iteration-{iteration:03d}.db"
    )
    stem = f"b19-funnel-iteration-{iteration:03d}"
    return IterationPaths(
        database=database,
        building_database=Path(f"{database}.building"),
        payload=output_dir / f"{stem}.payload.json",
        worksheet=output_dir / f"{stem}.atomicity.json",
        manual_template=output_dir / f"{stem}.manual-decisions.json",
        parse_failures=output_dir / f"{stem}.parse_failures.jsonl",
    )


def file_sha256(path: Path) -> str:
    """Return the SHA-256 of one existing file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readonly_sqlite_url(database_path: Path) -> str:
    """Async SQLAlchemy URL whose SQLite connection cannot write."""
    resolved = database_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database does not exist: {resolved}")
    return f"sqlite+aiosqlite:///file:{resolved}?mode=ro&uri=true"


def _writable_sqlite_url(database_path: Path) -> str:
    return f"sqlite+aiosqlite:///{database_path.resolve()}"


def _upgrade_sqlite_to_head(database_path: Path) -> str:
    config = Config(str(SERVER_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(SERVER_ROOT / "alembic"))
    config.set_main_option(
        "sqlalchemy.url", f"sqlite:///{database_path.resolve()}"
    )
    command.upgrade(config, "head")
    with sqlite3.connect(database_path) as connection:
        row = connection.execute("SELECT version_num FROM alembic_version").fetchone()
    if row is None or not isinstance(row[0], str) or not row[0]:
        raise FunnelPipelineError("target Alembic head could not be verified")
    return row[0]


async def _source_table_names(engine: AsyncEngine) -> set[str]:
    async with engine.connect() as connection:
        return set(
            await connection.run_sync(lambda sync: inspect(sync).get_table_names())
        )


async def _copy_base_tables_read_only(
    source_engine: AsyncEngine,
    target_engine: AsyncEngine,
    *,
    chunk_size: int = 1000,
) -> dict[str, int]:
    """Copy only base inputs while the source transaction is read-only."""
    if type(chunk_size) is not int or chunk_size < 1:
        raise FunnelPipelineError("chunk_size must be a positive integer")
    present = await _source_table_names(source_engine)
    if "data_points" not in present:
        raise FunnelPipelineError("source analysis DB has no data_points table")
    counts: dict[str, int] = {}
    async with source_engine.connect() as source_connection:
        async with source_connection.begin():
            if source_engine.dialect.name == "sqlite":
                await source_connection.execute(text("PRAGMA query_only = ON"))
            elif source_engine.dialect.name == "postgresql":
                await source_connection.execute(text("SET TRANSACTION READ ONLY"))
            else:
                raise FunnelPipelineError(
                    "source analysis DB must be SQLite or PostgreSQL"
                )
            for table_name in _BASE_TABLES:
                if table_name not in present:
                    counts[table_name] = 0
                    continue
                table = Base.metadata.tables[table_name]
                columns = list(table.columns)
                copied = 0
                result = await source_connection.stream(table.select())
                async for partition in result.partitions(chunk_size):
                    rows = [
                        {
                            column.name: _to_utc(row._mapping[column.name])
                            for column in columns
                        }
                        for row in partition
                    ]
                    if rows:
                        async with target_engine.begin() as target_connection:
                            await target_connection.execute(table.insert(), rows)
                        copied += len(rows)
                counts[table_name] = copied
    return counts


async def _derived_counts(engine: AsyncEngine) -> dict[str, int]:
    counts: dict[str, int] = {}
    async with AsyncSession(engine) as session:
        for model in _DERIVED_MODELS:
            count = (
                await session.execute(select(func.count()).select_from(model))
            ).scalar_one()
            counts[model.__tablename__] = int(count)
    return counts


async def initialize_disposable_database(
    *,
    source_url: str,
    snapshot_sha256: str,
    target_path: Path,
    source_file: Path | None = None,
    expected_source_sha256: str | None = None,
    chunk_size: int = 1000,
) -> dict[str, Any]:
    """Create one head-migrated target containing base tables and nothing else."""
    if snapshot_sha256 != LOCKED_SNAPSHOT_SHA256:
        raise FunnelPipelineError(
            "snapshot SHA-256 attestation does not match the locked snapshot"
        )
    try:
        source_backend = make_url(source_url).get_backend_name()
    except Exception as exc:
        raise FunnelPipelineError("source analysis URL is malformed") from exc
    if source_backend not in {"sqlite", "postgresql"}:
        raise FunnelPipelineError(
            "source analysis DB must be SQLite or PostgreSQL"
        )
    if source_backend == "sqlite" and source_file is None:
        raise FunnelPipelineError(
            "SQLite source requires an explicit source_file and pre/post hash"
        )
    building_path = Path(f"{target_path}.building")
    if target_path.exists() or building_path.exists():
        raise FunnelPipelineError(
            f"disposable target already exists: {target_path} or {building_path}"
        )
    if source_file is None and expected_source_sha256 is not None:
        raise FunnelPipelineError(
            "expected_source_sha256 requires an explicit source_file"
        )
    source_before: str | None = None
    if source_file is not None:
        resolved_source = source_file.resolve()
        if not resolved_source.is_file():
            raise FunnelPipelineError(f"source SQLite file does not exist: {resolved_source}")
        if (
            not isinstance(expected_source_sha256, str)
            or _SHA256_RE.fullmatch(expected_source_sha256) is None
        ):
            raise FunnelPipelineError(
                "source SQLite copy requires its measured 64-character SHA-256"
            )
        source_before = file_sha256(resolved_source)
        if source_before != expected_source_sha256:
            raise FunnelPipelineError(
                "source SHA-256 mismatch before read: "
                f"{source_before} != {expected_source_sha256}"
            )
        if source_backend == "sqlite" and source_url != readonly_sqlite_url(
            resolved_source
        ):
            raise FunnelPipelineError(
                "SQLite source URL does not identify the hash-checked source_file"
            )

    target_path.parent.mkdir(parents=True, exist_ok=True)
    alembic_head = await asyncio.to_thread(_upgrade_sqlite_to_head, building_path)
    source_engine = create_async_engine(source_url, echo=False)
    source_kind = source_engine.dialect.name
    target_engine = create_async_engine(_writable_sqlite_url(building_path), echo=False)
    copied_counts: dict[str, int]
    derived_counts: dict[str, int]
    source_after: str | None = None
    try:
        copied_counts = await _copy_base_tables_read_only(
            source_engine,
            target_engine,
            chunk_size=chunk_size,
        )
        derived_counts = await _derived_counts(target_engine)
        if any(derived_counts.values()):
            raise FunnelPipelineError(
                f"new disposable DB contains copied derived rows: {derived_counts}"
            )
    finally:
        await source_engine.dispose()
        await target_engine.dispose()
        if source_file is not None:
            source_after = file_sha256(source_file.resolve())
            if source_before is not None and source_after != source_before:
                raise FunnelPipelineError(
                    "source SHA-256 changed during read: "
                    f"{source_before} != {source_after}"
                )
    initial_hash = file_sha256(building_path)
    try:
        os.link(building_path, target_path)
    except FileExistsError as exc:
        raise FunnelPipelineError(
            f"disposable target appeared during promotion: {target_path}"
        ) from exc
    building_path.unlink()
    return {
        "source_kind": source_kind,
        "snapshot_sha256_attestation": snapshot_sha256,
        "source_sha256_before": source_before,
        "source_sha256_after": source_after,
        "copied_counts": copied_counts,
        "derived_counts_before_detection": derived_counts,
        "alembic_head": alembic_head,
        "target_initial_sha256": initial_hash,
    }


def _ranked_anomaly_rows(result: FreezeResult) -> list[dict[str, Any]]:
    missing = {str(anomaly_id) for anomaly_id in result.missing_enrichment}
    return [
        {
            "anomaly_id": str(anomaly.id),
            "source": anomaly.source,
            "metric": anomaly.metric,
            "source_entity_id": anomaly.source_entity_id,
            "detector_availability": anomaly.detector_availability_json,
            "enrichment_present": str(anomaly.id) not in missing,
        }
        for anomaly in result.selected
    ]


def _detection_summary_payload(summary: Any) -> dict[str, Any]:
    return {
        "groups_examined": int(summary.n_groups_examined),
        "groups_run": int(summary.n_groups_run),
        "raw_anomalies": int(summary.n_raw_anomalies_emitted),
        "eligible_anomalies": int(summary.n_anomalies_emitted),
        "direction_excluded": int(summary.n_direction_excluded),
        "missing_expected_excluded": int(summary.n_missing_expected_excluded),
        "persisted": int(summary.n_persisted),
        "expected_value_source_counts": dict(summary.expected_value_source_counts),
        "persist_errors": sorted(
            str(group.persist_error)
            for group in summary.groups
            if getattr(group, "persist_error", None)
        ),
    }


def _enrichment_summary_payload(summary: Any) -> dict[str, Any]:
    errors = sorted(
        str(line.error) for line in summary.lines if getattr(line, "error", None)
    )
    return {
        "pending": int(summary.n_pending),
        "persisted": int(summary.n_persisted),
        "errors": errors,
    }


def _harness_summary_payload(
    summaries: Mapping[str, ModelSweepSummary],
) -> dict[str, dict[str, Any]]:
    return {
        model: {
            "completed": summary.completed,
            "skipped": summary.skipped,
            "parse_failures": summary.parse_failures,
            "errors": [
                {"anomaly_id": str(anomaly_id), "message": message}
                for anomaly_id, message in summary.errors
            ],
            "latency_ms": summary.total_latency_ms,
            "prompt_tokens": summary.prompt_tokens,
            "completion_tokens": summary.completion_tokens,
        }
        for model, summary in sorted(summaries.items())
    }


async def run_pipeline_stages(
    database_path: Path,
    *,
    confirm_model_run: bool,
    client_factory: Callable[[str], LLMClient] = make_client,
    failure_log: Path | None = None,
) -> dict[str, Any]:
    """Run every mutable B19 stage on one explicitly bound disposable DB."""
    if confirm_model_run is not True:
        raise FunnelPipelineError(
            "model execution requires explicit confirmation of the disposable run"
        )
    if not database_path.is_file():
        raise FunnelPipelineError(f"disposable database does not exist: {database_path}")
    engine = create_async_engine(_writable_sqlite_url(database_path), echo=False)
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    try:
        counts = await _derived_counts(engine)
        if any(counts.values()):
            raise FunnelPipelineError(
                f"disposable DB is not clean before detection: {counts}"
            )
        async with session_factory() as session:
            detection = await run_detection(session, since=WINDOW_START)
            detection_payload = _detection_summary_payload(detection)
            if detection_payload["persist_errors"]:
                raise FunnelPipelineError(
                    "detection persistence failed: "
                    + "; ".join(detection_payload["persist_errors"])
                )
            enrichment = await enrich_pending_anomalies(session)
            enrichment_payload = _enrichment_summary_payload(enrichment)
            if enrichment_payload["errors"]:
                raise FunnelPipelineError(
                    "enrichment failed: " + "; ".join(enrichment_payload["errors"])
                )
            ranked = await freeze_eval_set(
                session,
                window_start=WINDOW_START,
                window_end=WINDOW_END,
                top_n=2**31 - 1,
            )
            ranked_rows = _ranked_anomaly_rows(ranked)
            selection = select_funnel_anomalies(ranked_rows)
            selected_ids = [
                uuid.UUID(anomaly_id)
                for anomaly_id in selection["selected_anomaly_ids"]
            ]
            harness = await run_harness(
                session,
                selected_ids,
                models=list(DEFAULT_MODELS),
                client_factory=client_factory,
                failure_log=failure_log,
            )
        return {
            "database_filename": database_path.name,
            "window": {
                "start": WINDOW_START.isoformat(),
                "end_exclusive": WINDOW_END.isoformat(),
            },
            "detection": detection_payload,
            "enrichment": enrichment_payload,
            "ranked_event_count": ranked.n_events,
            "selection": selection,
            "harness": _harness_summary_payload(harness),
        }
    finally:
        await engine.dispose()


def _b8_rows(
    anomaly_id: str,
    summary: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    absences: list[dict[str, Any]] = []
    raw_sources = summary.get("sources")
    sources = raw_sources if isinstance(raw_sources, Mapping) else {}
    for source in DEFAULT_OBSERVATION_AGE_GATES.to_dict():
        raw_source = sources.get(source)
        source_block = raw_source if isinstance(raw_source, Mapping) else None
        raw_metrics = source_block.get("metrics") if source_block else None
        metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
        if not metrics:
            absences.append(
                {
                    "anomaly_id": anomaly_id,
                    "source": source,
                    "metric": None,
                    "reason": "source-absent-from-window",
                }
            )
            continue
        for metric, raw_block in sorted(metrics.items()):
            block = raw_block if isinstance(raw_block, Mapping) else {}
            raw_nearest = block.get("nearest_in_time")
            nearest = raw_nearest if isinstance(raw_nearest, Mapping) else {}
            if nearest.get("v") is None:
                absences.append(
                    {
                        "anomaly_id": anomaly_id,
                        "source": source,
                        "metric": str(metric),
                        "reason": "nearest-event-value-absent",
                    }
                )
                continue
            observations.append(
                {
                    "anomaly_id": anomaly_id,
                    "source": source,
                    "metric": str(metric),
                    "dt_minutes": nearest.get("dt_minutes"),
                }
            )
    return observations, absences


def _calm_decision_rows(
    anomaly_id: str,
    summary: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions, _notes = calm_wind_source_decisions(summary, _WIND_SOURCES)
    return decisions, [
        {"anomaly_id": anomaly_id, **decisions[source].to_dict()}
        for source in _WIND_SOURCES
    ]


async def extract_report_inputs(
    database_path: Path,
    *,
    iteration: int,
    git_commit: str,
    snapshot_sha256: str,
) -> dict[str, Any]:
    """Read exact B19 report fields from a completed disposable DB."""
    if snapshot_sha256 != LOCKED_SNAPSHOT_SHA256:
        raise FunnelPipelineError("snapshot SHA-256 attestation is not canonical")
    before_hash = file_sha256(database_path)
    engine = create_async_engine(readonly_sqlite_url(database_path), echo=False)
    session_factory = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    try:
        async with session_factory() as session:
            ranked = await freeze_eval_set(
                session,
                window_start=WINDOW_START,
                window_end=WINDOW_END,
                top_n=2**31 - 1,
            )
            ranked_rows = _ranked_anomaly_rows(ranked)
            selection = select_funnel_anomalies(ranked_rows)
            selected_ids = [
                uuid.UUID(anomaly_id)
                for anomaly_id in selection["selected_anomaly_ids"]
            ]
            records = list(
                (
                    await session.execute(
                        select(EnrichmentRecord)
                        .where(EnrichmentRecord.anomaly_id.in_(selected_ids))
                        .order_by(EnrichmentRecord.anomaly_id)
                    )
                )
                .scalars()
                .all()
            )
            summary_by_anomaly: dict[uuid.UUID, Mapping[str, Any]] = {}
            for record in records:
                if record.anomaly_id in summary_by_anomaly:
                    raise FunnelPipelineError(
                        f"selected anomaly {record.anomaly_id} has duplicate enrichment"
                    )
                if not isinstance(record.cross_source_summary_json, Mapping):
                    raise FunnelPipelineError(
                        f"selected anomaly {record.anomaly_id} has malformed enrichment"
                    )
                summary_by_anomaly[record.anomaly_id] = (
                    record.cross_source_summary_json
                )
            missing = sorted(set(selected_ids) - set(summary_by_anomaly), key=str)
            if missing:
                raise FunnelPipelineError(
                    "selected anomalies missing enrichment: "
                    + ", ".join(str(item) for item in missing)
                )

            explanations = list(
                (
                    await session.execute(
                        select(Explanation)
                        .where(Explanation.anomaly_id.in_(selected_ids))
                        .where(Explanation.model_name.in_(DEFAULT_MODELS))
                        .order_by(Explanation.anomaly_id, Explanation.model_name)
                    )
                )
                .scalars()
                .all()
            )
            cells = [
                {
                    "anomaly_id": str(explanation.anomaly_id),
                    "model": explanation.model_name,
                    "steps": (
                        explanation.reasoning_steps_json.get("steps")
                        if isinstance(explanation.reasoning_steps_json, Mapping)
                        else None
                    ),
                }
                for explanation in explanations
            ]
            explanation_ids = [explanation.id for explanation in explanations]
            raw_claim_rows = list(
                (
                    await session.execute(
                        select(Claim, Explanation.anomaly_id, Explanation.model_name)
                        .join(Explanation, Claim.explanation_id == Explanation.id)
                        .where(Claim.explanation_id.in_(explanation_ids))
                        .order_by(
                            Explanation.anomaly_id,
                            Explanation.model_name,
                            Claim.step_index,
                            Claim.id,
                        )
                    )
                ).all()
            )

            calm_by_anomaly: dict[uuid.UUID, dict[str, Any]] = {}
            calm_rows: list[dict[str, Any]] = []
            b8_observations: list[dict[str, Any]] = []
            b8_absences: list[dict[str, Any]] = []
            for anomaly_id in selected_ids:
                summary = summary_by_anomaly[anomaly_id]
                decisions, decision_rows = _calm_decision_rows(
                    str(anomaly_id), summary
                )
                calm_by_anomaly[anomaly_id] = decisions
                calm_rows.extend(decision_rows)
                observations, absences = _b8_rows(str(anomaly_id), summary)
                b8_observations.extend(observations)
                b8_absences.extend(absences)

            claims: list[dict[str, Any]] = []
            for claim, anomaly_id, model_name in raw_claim_rows:
                summary = summary_by_anomaly[anomaly_id]
                try:
                    primary = ClaimType(claim.claim_type)
                except ValueError:
                    primary = ClaimType.UNCLASSIFIED
                direction_sources = direction_data_sources(
                    claim.claim_text,
                    summary,
                    primary,
                )
                if primary is ClaimType.TRANSPORT_DIRECTION:
                    calm_sources: Sequence[str] = _WIND_SOURCES
                elif primary is ClaimType.POINT_SOURCE_ATTRIBUTION:
                    calm_sources = ("noaa_gfs", "openweather")
                else:
                    calm_sources = ()
                calm_flagged = any(
                    calm_by_anomaly[anomaly_id][source].calm is True
                    for source in calm_sources
                )
                claims.append(
                    {
                        "claim_id": str(claim.id),
                        "anomaly_id": str(anomaly_id),
                        "model": str(model_name),
                        "claim_text": claim.claim_text,
                        "claim_type": claim.claim_type,
                        "matched_types": claim.matched_types,
                        "cited_sources": claim.cited_sources,
                        "citation_outcome": claim.citation_outcome,
                        "citation_failure_reasons": (
                            claim.citation_failure_reasons_json
                        ),
                        "grounding_verdict": claim.grounding_verdict,
                        "skipped_phase2": claim.skipped_phase2,
                        "corroboration_score": claim.corroboration_score,
                        "evidence_n": claim.evidence_n,
                        "corroboration_evidence_summary": (
                            claim.corroboration_evidence_summary
                        ),
                        "causal": claim.causal,
                        "calm_wind_flagged": calm_flagged,
                        "direction_data_present": bool(direction_sources),
                    }
                )
    finally:
        await engine.dispose()

    after_hash = file_sha256(database_path)
    if after_hash != before_hash:
        raise FunnelPipelineError(
            f"disposable DB changed during extraction: {before_hash} != {after_hash}"
        )
    report_inputs = {
        "ranked_anomalies": ranked_rows,
        "cells": cells,
        "claims": claims,
        "b8_observations": b8_observations,
        "b8_absences": b8_absences,
        "calm_wind_decisions": calm_rows,
        "provenance": {
            "disposable_b19_not_official": True,
            "git_commit": git_commit,
            "db_copy_sha256": before_hash,
            "selected_anomaly_ids": selection["selected_anomaly_ids"],
            "iteration": iteration,
        },
    }
    return {
        "schema_version": 1,
        "database_filename": database_path.name,
        "database_sha256": before_hash,
        "snapshot_sha256_attestation": snapshot_sha256,
        "report_inputs": report_inputs,
    }


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(payload))


def write_preparation_artifacts(
    paths: IterationPaths,
    payload: Mapping[str, Any],
    *,
    worksheet: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write the payload, blinded worksheet, and editable null template."""
    destinations = (paths.payload, paths.worksheet, paths.manual_template)
    if any(path.exists() for path in destinations):
        raise FileExistsError(
            "B19 preparation artifacts are preserved and cannot be overwritten"
        )
    paths.payload.parent.mkdir(parents=True, exist_ok=True)
    raw_inputs = payload.get("report_inputs")
    if not isinstance(raw_inputs, Mapping):
        raise FunnelPipelineError("preparation payload has no report_inputs object")
    if worksheet is None:
        worksheet = build_atomicity_worksheet(raw_inputs.get("claims"))
    items = worksheet.get("items")
    if not isinstance(items, list):
        raise FunnelPipelineError("atomicity worksheet has no items array")
    manual_template = {
        str(item["decision_hash"]): None
        for item in items
        if isinstance(item, Mapping) and isinstance(item.get("decision_hash"), str)
    }
    if len(manual_template) != len(items):
        raise FunnelPipelineError("atomicity worksheet contains malformed hashes")
    _write_exclusive_json(paths.payload, payload)
    _write_exclusive_json(paths.worksheet, worksheet)
    _write_exclusive_json(paths.manual_template, manual_template)
    return {
        "payload": paths.payload,
        "worksheet": paths.worksheet,
        "manual_template": paths.manual_template,
    }


def _load_json_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FunnelPipelineError(f"cannot read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FunnelPipelineError(f"{description} must contain a JSON object")
    return value


def finalize_iteration(
    *,
    payload_path: Path,
    manual_decisions_path: Path,
    database_path: Path,
    output_dir: Path,
    current_code_commit: str,
) -> dict[str, Any]:
    """Verify the preserved DB and build a report without re-running any stage."""
    payload = _load_json_object(payload_path, "preparation payload")
    expected_hash = payload.get("database_sha256")
    if not isinstance(expected_hash, str) or _SHA256_RE.fullmatch(expected_hash) is None:
        raise FunnelPipelineError("preparation payload has invalid DB SHA-256")
    actual_hash = file_sha256(database_path)
    if actual_hash != expected_hash:
        raise FunnelPipelineError(
            f"disposable DB SHA-256 mismatch: {actual_hash} != {expected_hash}"
        )
    if payload.get("database_filename") != database_path.name:
        raise FunnelPipelineError("disposable DB filename does not match payload")
    if payload.get("snapshot_sha256_attestation") != LOCKED_SNAPSHOT_SHA256:
        raise FunnelPipelineError(
            "preparation payload snapshot attestation is not canonical"
        )
    report_inputs = payload.get("report_inputs")
    if not isinstance(report_inputs, Mapping):
        raise FunnelPipelineError("preparation payload has no report_inputs object")
    raw_provenance = report_inputs.get("provenance")
    if (
        not isinstance(raw_provenance, Mapping)
        or raw_provenance.get("db_copy_sha256") != expected_hash
    ):
        raise FunnelPipelineError(
            "preparation payload provenance hash does not match its DB hash"
        )
    if raw_provenance.get("git_commit") != current_code_commit:
        raise FunnelPipelineError(
            "finalization code commit differs from the preparation commit; "
            "start a fresh numbered funnel iteration"
        )
    decisions = _load_json_object(manual_decisions_path, "manual decisions")
    report = build_funnel_report(
        **dict(report_inputs),
        manual_atomicity=decisions,
    )
    report_paths = write_iteration_reports(output_dir, report)
    return {"report": report, "paths": report_paths}


async def prepare_iteration(
    *,
    source_url: str,
    snapshot_sha256: str,
    output_dir: Path,
    run_date: str,
    iteration: int,
    confirm_model_run: bool,
    source_file: Path | None = None,
    expected_source_sha256: str | None = None,
    client_factory: Callable[[str], LLMClient] = make_client,
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Initialize, run, extract, and pause at the blinded manual step."""
    if confirm_model_run is not True:
        raise FunnelPipelineError(
            "model execution requires explicit confirmation of the disposable run"
        )
    paths = iteration_paths(output_dir, run_date=run_date, iteration=iteration)
    commit = git_commit or repository_code_commit(REPOSITORY_ROOT)
    isolation = await initialize_disposable_database(
        source_url=source_url,
        source_file=source_file,
        expected_source_sha256=expected_source_sha256,
        snapshot_sha256=snapshot_sha256,
        target_path=paths.database,
    )
    pipeline = await run_pipeline_stages(
        paths.database,
        confirm_model_run=confirm_model_run,
        client_factory=client_factory,
        failure_log=paths.parse_failures,
    )
    payload = await extract_report_inputs(
        paths.database,
        iteration=iteration,
        git_commit=commit,
        snapshot_sha256=snapshot_sha256,
    )
    extracted_ids = payload["report_inputs"]["provenance"][
        "selected_anomaly_ids"
    ]
    if extracted_ids != pipeline["selection"]["selected_anomaly_ids"]:
        raise FunnelPipelineError(
            "selection changed between pipeline execution and read-only extraction"
        )
    payload["isolation"] = isolation
    payload["pipeline"] = pipeline
    payload["target_final_sha256"] = payload["database_sha256"]
    artifacts = write_preparation_artifacts(paths, payload)
    return {
        "database": paths.database,
        "payload": payload,
        "artifacts": artifacts,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.funnel_pipeline",
        description="Prepare or finalize one preserved disposable B19 iteration.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    source = prepare.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-analysis-url")
    source.add_argument("--source-sqlite", type=Path)
    prepare.add_argument("--source-sqlite-sha256")
    prepare.add_argument("--snapshot-sha256", required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--run-date", required=True)
    prepare.add_argument("--iteration", type=int, required=True)
    prepare.add_argument(
        "--confirm-disposable-model-run",
        action="store_true",
        help="Required acknowledgement before any model client is constructed.",
    )

    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--payload", type=Path, required=True)
    finalize.add_argument("--manual-decisions", type=Path, required=True)
    finalize.add_argument("--database", type=Path, required=True)
    finalize.add_argument("--output-dir", type=Path, required=True)
    return parser


async def _amain(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "finalize":
        result = finalize_iteration(
            payload_path=args.payload,
            manual_decisions_path=args.manual_decisions,
            database_path=args.database,
            output_dir=args.output_dir,
            current_code_commit=repository_code_commit(REPOSITORY_ROOT),
        )
        print(result["paths"]["json"])
        print(result["paths"]["markdown"])
        return 0 if result["report"]["go_no_go"]["status"] == "go" else 1

    source_file: Path | None = args.source_sqlite
    if args.confirm_disposable_model_run is not True:
        _parser().error("--confirm-disposable-model-run is required for prepare")
    if source_file is not None:
        if args.source_sqlite_sha256 is None:
            _parser().error("--source-sqlite-sha256 is required with --source-sqlite")
        source_url = readonly_sqlite_url(source_file)
    else:
        if args.source_sqlite_sha256 is not None:
            _parser().error(
                "--source-sqlite-sha256 is valid only with --source-sqlite"
            )
        source_url = str(args.source_analysis_url)
    result = await prepare_iteration(
        source_url=source_url,
        source_file=source_file,
        expected_source_sha256=args.source_sqlite_sha256,
        snapshot_sha256=args.snapshot_sha256,
        output_dir=args.output_dir,
        run_date=args.run_date,
        iteration=args.iteration,
        confirm_model_run=args.confirm_disposable_model_run,
    )
    print(result["database"])
    for path in result["artifacts"].values():
        print(path)
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
