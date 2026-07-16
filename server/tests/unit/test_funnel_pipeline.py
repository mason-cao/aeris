import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from app.db.models import (
    Anomaly,
    Claim,
    DataPoint,
    EnrichmentRecord,
    Explanation,
)
from app.db.schema import create_tables
import app.eval.funnel_pipeline as pipeline_mod
from app.eval.freeze import FreezeResult
from app.eval.funnel_dry_run import (
    ATOMICITY_SELF_CONTAINED,
    build_atomicity_worksheet,
    build_funnel_report,
)
from app.eval.funnel_pipeline import (
    FunnelPipelineError,
    extract_report_inputs,
    finalize_iteration,
    initialize_disposable_database,
    iteration_paths,
    prepare_iteration,
    readonly_sqlite_url,
    run_pipeline_stages,
    write_preparation_artifacts,
)
from app.eval.harness import DEFAULT_MODELS, ModelSweepSummary
from app.provenance.purpleair_qc import LOCKED_SNAPSHOT_SHA256


UTC = timezone.utc
WINDOW_START = datetime(2026, 6, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 7, 13, tzinfo=UTC)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _availability() -> dict[str, dict[str, object]]:
    return {
        detector: {"ran": True, "skip_code": None, "detail": None}
        for detector in ("isolation_forest", "stl", "zscore")
    }


def _anomaly(index: int, metric: str) -> Anomaly:
    return Anomaly(
        id=uuid.UUID(f"00000000-0000-4000-8000-{index:012d}"),
        timestamp=datetime(2026, 6, index + 1, 12, tzinfo=UTC),
        lat=29.76,
        lon=-95.37,
        metric=metric,
        source="openaq" if metric in {"pm25", "ozone"} else "tceq",
        source_entity_id=f"station-{index}",
        detector_availability_json=_availability(),
        value=40.0 + index,
        expected_value=10.0,
        z_score=10.0 - index,
        methods_triggered=["isolation_forest", "stl", "zscore"],
        severity="severe",
    )


def _metric_block(
    value: float,
    *,
    metric: str,
    entity_id: str,
    timestamp: str = "2026-06-02T12:00:00+00:00",
) -> dict[str, object]:
    return {
        "unit": "m/s" if "wind" in metric or metric in {"u_10m", "v_10m"} else "ug/m3",
        "n_points": 3,
        "n_entities": 1,
        "value_range": {"min": value, "max": value, "mean": value},
        "nearest_in_time": {
            "t": timestamp,
            "v": value,
            "entity_id": entity_id,
            "distance_km": 1.0,
            "dt_minutes": 30.0,
        },
        "entities": [
            {
                "entity_id": entity_id,
                "series": [
                    ("2026-06-01T12:00:00+00:00", value),
                    ("2026-06-02T12:00:00+00:00", value),
                    ("2026-06-03T12:00:00+00:00", value),
                ],
            }
        ],
    }


def _summary(anomaly: Anomaly) -> dict[str, object]:
    timestamp = anomaly.timestamp.isoformat()
    return {
        "schema_version": 1,
        "anomaly": {
            "id": str(anomaly.id),
            "timestamp": timestamp,
            "lat": anomaly.lat,
            "lon": anomaly.lon,
            "metric": anomaly.metric,
            "source": anomaly.source,
        },
        "window": {
            "start": (anomaly.timestamp - timedelta(hours=36)).isoformat(),
            "end": (anomaly.timestamp + timedelta(hours=36)).isoformat(),
            "spatial_radius_km": 50.0,
        },
        "coverage": {
            "openaq": True,
            "openweather": True,
            "noaa_gfs": True,
            "sentinel5p": False,
            "asos": True,
            "tceq": False,
            "purpleair": False,
        },
        "sources": {
            "openaq": {
                "metrics": {
                    "pm25": _metric_block(
                        30.0,
                        metric="pm25",
                        entity_id="openaq-1",
                        timestamp=timestamp,
                    ),
                    "ozone": {
                        "unit": "ppm",
                        "n_points": 0,
                        "n_entities": 0,
                        "nearest_in_time": {"v": None},
                        "entities": [],
                    },
                }
            },
            "noaa_gfs": {
                "metrics": {
                    "u_10m": _metric_block(
                        0.3,
                        metric="u_10m",
                        entity_id="gfs-cell",
                        timestamp=timestamp,
                    ),
                    "v_10m": _metric_block(
                        0.4,
                        metric="v_10m",
                        entity_id="gfs-cell",
                        timestamp=timestamp,
                    ),
                }
            },
            "openweather": {
                "metrics": {
                    "wind_speed": _metric_block(
                        0.5,
                        metric="wind_speed",
                        entity_id="ow-grid",
                        timestamp=timestamp,
                    ),
                    "wind_direction": _metric_block(
                        180.0,
                        metric="wind_direction",
                        entity_id="ow-grid",
                        timestamp=timestamp,
                    ),
                }
            },
            "asos": {
                "metrics": {
                    "wind_speed": _metric_block(
                        0.5,
                        metric="wind_speed",
                        entity_id="asos-1",
                        timestamp=timestamp,
                    ),
                    "wind_direction": _metric_block(
                        180.0,
                        metric="wind_direction",
                        entity_id="asos-1",
                        timestamp=timestamp,
                    ),
                }
            },
        },
    }


async def _seed_source_with_derived_rows(path: Path) -> None:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await create_tables(engine)
    anomaly = _anomaly(1, "pm25")
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add(
            DataPoint(
                timestamp=datetime(2026, 6, 1, 12, tzinfo=UTC),
                lat=29.76,
                lon=-95.37,
                metric="pm25",
                value=12.0,
                unit="ug/m3",
                source="openaq",
                source_entity_id="source-monitor",
            )
        )
        session.add(anomaly)
        await session.flush()
        explanation = Explanation(
            anomaly_id=anomaly.id,
            model_name=DEFAULT_MODELS[0],
            reasoning_steps_json={"steps": []},
            final_narrative="source-only derived row",
        )
        session.add(explanation)
        await session.flush()
        session.add(
            Claim(
                explanation_id=explanation.id,
                step_index=1,
                claim_type="unclassified",
                matched_types=["unclassified"],
                claim_text="source-only claim",
                cited_sources=[],
                citation_outcome="uncited",
                citation_failure_reasons_json=[],
                grounding_verdict="unverified",
                grounding_evidence_ref=None,
                causal=False,
                skipped_phase2=True,
                corroboration_score=None,
                corroboration_evidence_summary=None,
                evidence_n=0,
                per_source_verdicts=None,
                per_channel_verdicts=None,
                partial_verifiability=False,
                low_corroboration_flag=False,
            )
        )
        await session.commit()
    await engine.dispose()


async def _seed_completed_funnel(path: Path) -> list[Anomaly]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await create_tables(engine)
    metrics = ["pm25", "ozone", "no2", "so2", "co"]
    anomalies = [_anomaly(index, metric) for index, metric in enumerate(metrics, 1)]
    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add_all(anomalies)
        await session.flush()
        for anomaly in anomalies:
            session.add(
                EnrichmentRecord(
                    anomaly_id=anomaly.id,
                    context_window_start=anomaly.timestamp - timedelta(hours=36),
                    context_window_end=anomaly.timestamp + timedelta(hours=36),
                    cross_source_summary_json=_summary(anomaly),
                )
            )
            for model_index, model in enumerate(DEFAULT_MODELS, start=1):
                explanation = Explanation(
                    anomaly_id=anomaly.id,
                    model_name=model,
                    model_version="synthetic-v1",
                    reasoning_steps_json={
                        "steps": [
                            {
                                "prompt_tokens": 1000,
                                "completion_tokens": 100,
                                "attempts": 1,
                            }
                            for _ in range(4)
                        ]
                    },
                    final_narrative="synthetic",
                    prompt_tokens=4000,
                    completion_tokens=400,
                )
                session.add(explanation)
                await session.flush()
                is_first = anomaly is anomalies[0]
                session.add(
                    Claim(
                        explanation_id=explanation.id,
                        step_index=1,
                        claim_type=(
                            "transport_direction"
                            if is_first
                            else "concentration_elevation"
                        ),
                        matched_types=[
                            "transport_direction"
                            if is_first
                            else "concentration_elevation"
                        ],
                        claim_text=(
                            "Southerly winds transported pollution northward."
                            if is_first
                            else f"PM2.5 was elevated at monitor {anomaly.id}."
                        ),
                        cited_sources=["openaq"],
                        citation_outcome="cited_right",
                        citation_failure_reasons_json=[],
                        grounding_verdict="grounded",
                        grounding_evidence_ref={"matched_terms": ["pollution"]},
                        causal=False,
                        skipped_phase2=False,
                        corroboration_score=1.0,
                        corroboration_evidence_summary=(
                            "noaa_gfs: calm-wind guard SILENT"
                            if is_first
                            else "openaq: pm25 nearest=30 vs pre-anomaly baseline=8"
                        ),
                        evidence_n=1,
                        per_source_verdicts={"openaq": 1},
                        per_channel_verdicts={"ground_insitu": 1},
                        partial_verifiability=False,
                        low_corroboration_flag=False,
                    )
                )
        await session.commit()
    await engine.dispose()
    return anomalies


def test_iteration_paths_are_explicit_stable_and_iteration_scoped(tmp_path: Path) -> None:
    paths = iteration_paths(tmp_path, run_date="20260716", iteration=2)

    assert paths.database.name == "aeris-b19-funnel-20260716-iteration-002.db"
    assert paths.payload.name == "b19-funnel-iteration-002.payload.json"
    assert paths.worksheet.name == "b19-funnel-iteration-002.atomicity.json"
    assert paths.manual_template.name == (
        "b19-funnel-iteration-002.manual-decisions.json"
    )
    with pytest.raises(FunnelPipelineError, match="YYYYMMDD"):
        iteration_paths(tmp_path, run_date="2026-07-16", iteration=2)
    with pytest.raises(FunnelPipelineError, match="valid YYYYMMDD"):
        iteration_paths(tmp_path, run_date="20260230", iteration=2)
    with pytest.raises(FunnelPipelineError, match="positive"):
        iteration_paths(tmp_path, run_date="20260716", iteration=0)
    with pytest.raises(FileNotFoundError):
        readonly_sqlite_url(tmp_path / "missing.db")


@pytest.mark.asyncio
async def test_initialization_rejects_unverifiable_source_identity_before_copy(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.db"
    with pytest.raises(FunnelPipelineError, match="locked snapshot"):
        await initialize_disposable_database(
            source_url="sqlite+aiosqlite:///unused.db",
            snapshot_sha256="0" * 64,
            target_path=target,
        )
    with pytest.raises(FunnelPipelineError, match="requires an explicit"):
        await initialize_disposable_database(
            source_url="postgresql+asyncpg://readonly@example.invalid/aeris",
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            expected_source_sha256="0" * 64,
            target_path=target,
        )
    with pytest.raises(FunnelPipelineError, match="pre/post hash"):
        await initialize_disposable_database(
            source_url="sqlite+aiosqlite:///unused.db",
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            target_path=target,
        )
    with pytest.raises(FunnelPipelineError, match="does not exist"):
        await initialize_disposable_database(
            source_url="sqlite+aiosqlite:///unused.db",
            source_file=tmp_path / "missing.db",
            expected_source_sha256="0" * 64,
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            target_path=target,
        )
    source = tmp_path / "identity.db"
    source.write_bytes(b"identity")
    with pytest.raises(FunnelPipelineError, match="64-character"):
        await initialize_disposable_database(
            source_url=readonly_sqlite_url(source),
            source_file=source,
            expected_source_sha256="bad",
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            target_path=target,
        )
    other = tmp_path / "other.db"
    other.write_bytes(b"other")
    with pytest.raises(FunnelPipelineError, match="does not identify"):
        await initialize_disposable_database(
            source_url=readonly_sqlite_url(other),
            source_file=source,
            expected_source_sha256=_sha256(source),
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            target_path=target,
        )


@pytest.mark.asyncio
async def test_source_hash_is_rechecked_even_when_copy_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "empty-source.db"
    sqlite3_connection = sqlite3.connect(source)
    sqlite3_connection.close()
    expected_hash = _sha256(source)
    seen: list[Path] = []
    original = pipeline_mod.file_sha256

    def recording_hash(path: Path) -> str:
        seen.append(path.resolve())
        return original(path)

    monkeypatch.setattr(pipeline_mod, "file_sha256", recording_hash)
    with pytest.raises(FunnelPipelineError, match="no data_points"):
        await initialize_disposable_database(
            source_url=readonly_sqlite_url(source),
            source_file=source,
            expected_source_sha256=expected_hash,
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            target_path=tmp_path / "failed-target.db",
        )

    assert seen.count(source.resolve()) == 2


@pytest.mark.asyncio
async def test_initialization_reads_source_only_and_copies_no_derived_rows(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.db"
    await _seed_source_with_derived_rows(source)
    source_hash = _sha256(source)
    paths = iteration_paths(tmp_path, run_date="20260716", iteration=1)

    manifest = await initialize_disposable_database(
        source_url=readonly_sqlite_url(source),
        source_file=source,
        expected_source_sha256=source_hash,
        snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
        target_path=paths.database,
    )

    assert _sha256(source) == source_hash
    assert manifest["source_sha256_before"] == source_hash
    assert manifest["source_sha256_after"] == source_hash
    assert manifest["copied_counts"]["data_points"] == 1
    assert manifest["alembic_head"] == "b19e8c4d2a61"
    assert manifest["derived_counts_before_detection"]["expert_labels"] == 0
    assert paths.database.exists()
    assert not paths.building_database.exists()

    engine = create_async_engine(f"sqlite+aiosqlite:///{paths.database}")
    async with AsyncSession(engine) as session:
        assert (
            await session.execute(select(func.count()).select_from(DataPoint))
        ).scalar_one() == 1
        for model in (Anomaly, EnrichmentRecord, Explanation, Claim):
            assert (
                await session.execute(select(func.count()).select_from(model))
            ).scalar_one() == 0
    await engine.dispose()

    readonly = create_async_engine(readonly_sqlite_url(source))
    with pytest.raises(OperationalError):
        async with readonly.begin() as connection:
            await connection.execute(DataPoint.__table__.delete())
    await readonly.dispose()

    with pytest.raises(FunnelPipelineError, match="already exists"):
        await initialize_disposable_database(
            source_url=readonly_sqlite_url(source),
            source_file=source,
            expected_source_sha256=source_hash,
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            target_path=paths.database,
        )


@pytest.mark.asyncio
async def test_prepare_refuses_missing_model_confirmation_before_creating_db(
    tmp_path: Path,
) -> None:
    paths = iteration_paths(tmp_path, run_date="20260716", iteration=1)

    with pytest.raises(FunnelPipelineError, match="explicit confirmation"):
        await prepare_iteration(
            source_url="sqlite+aiosqlite:///does-not-matter.db",
            snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
            output_dir=tmp_path,
            run_date="20260716",
            iteration=1,
            confirm_model_run=False,
            git_commit="a" * 40,
        )

    assert not paths.database.exists()
    assert not paths.building_database.exists()


@pytest.mark.asyncio
async def test_pipeline_requires_confirmation_and_binds_every_stage_to_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "target.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{database}")
    await create_tables(engine)
    await engine.dispose()
    anomalies = [_anomaly(index, metric) for index, metric in enumerate(
        ["pm25", "ozone", "no2", "so2", "co"], 1
    )]
    seen_databases: list[str] = []

    def _record_session(session: AsyncSession) -> None:
        assert session.bind is not None
        seen_databases.append(str(session.bind.url.database))

    async def fake_detection(session: AsyncSession, **_kwargs):
        _record_session(session)
        return SimpleNamespace(
            n_groups_examined=5,
            n_groups_run=5,
            n_raw_anomalies_emitted=5,
            n_anomalies_emitted=5,
            n_direction_excluded=0,
            n_missing_expected_excluded=0,
            n_persisted=5,
            expected_value_source_counts={"zscore_rolling_mean": 5},
            groups=[],
        )

    async def fake_enrichment(session: AsyncSession, **_kwargs):
        _record_session(session)
        return SimpleNamespace(n_pending=5, n_persisted=5, lines=[])

    async def fake_freeze(session: AsyncSession, **_kwargs):
        _record_session(session)
        return FreezeResult(
            window_start=WINDOW_START,
            window_end=WINDOW_END,
            top_n=2**31 - 1,
            n_anomalies=5,
            n_events=5,
            selected=anomalies,
            event_sizes={anomaly.id: 1 for anomaly in anomalies},
            missing_enrichment=[],
        )

    async def fake_harness(session: AsyncSession, anomaly_ids, models, **_kwargs):
        _record_session(session)
        assert anomaly_ids == [anomaly.id for anomaly in anomalies]
        assert tuple(models) == DEFAULT_MODELS
        return {
            model: ModelSweepSummary(completed=5)
            for model in DEFAULT_MODELS
        }

    monkeypatch.setattr("app.eval.funnel_pipeline.run_detection", fake_detection)
    monkeypatch.setattr(
        "app.eval.funnel_pipeline.enrich_pending_anomalies", fake_enrichment
    )
    monkeypatch.setattr("app.eval.funnel_pipeline.freeze_eval_set", fake_freeze)
    monkeypatch.setattr("app.eval.funnel_pipeline.run_harness", fake_harness)

    with pytest.raises(FunnelPipelineError, match="explicit confirmation"):
        await run_pipeline_stages(database, confirm_model_run=False)
    assert seen_databases == []
    with pytest.raises(FunnelPipelineError, match="does not exist"):
        await run_pipeline_stages(
            tmp_path / "missing-target.db", confirm_model_run=True
        )

    result = await run_pipeline_stages(database, confirm_model_run=True)

    assert len(seen_databases) == 4
    assert set(seen_databases) == {str(database)}
    assert result["selection"]["selected_anomaly_ids"] == [
        str(anomaly.id) for anomaly in anomalies
    ]
    assert result["harness"][DEFAULT_MODELS[0]]["completed"] == 5


@pytest.mark.asyncio
async def test_extraction_is_read_only_exact_and_finalization_hash_guarded(
    tmp_path: Path,
) -> None:
    paths = iteration_paths(tmp_path, run_date="20260716", iteration=1)
    await _seed_completed_funnel(paths.database)
    before = _sha256(paths.database)

    payload = await extract_report_inputs(
        paths.database,
        iteration=1,
        git_commit="a" * 40,
        snapshot_sha256=LOCKED_SNAPSHOT_SHA256,
    )

    assert _sha256(paths.database) == before
    report_inputs = payload["report_inputs"]
    assert len(report_inputs["cells"]) == 15
    assert len(report_inputs["claims"]) == 15
    assert report_inputs["claims"][0]["citation_failure_reasons"] == []
    assert report_inputs["claims"][0]["corroboration_evidence_summary"]
    assert report_inputs["b8_observations"]
    assert {
        absence["source"] for absence in report_inputs["b8_absences"]
    } >= {"sentinel5p", "tceq", "purpleair"}
    transport = next(
        claim
        for claim in report_inputs["claims"]
        if claim["claim_type"] == "transport_direction"
    )
    assert transport["direction_data_present"] is True
    assert transport["calm_wind_flagged"] is True

    artifacts = write_preparation_artifacts(paths, payload)
    assert artifacts["payload"] == paths.payload
    worksheet = json.loads(paths.worksheet.read_text())
    assert all(
        set(item) == {"review_index", "decision_hash", "claim_text"}
        for item in worksheet["items"]
    )
    template = json.loads(paths.manual_template.read_text())
    assert set(template.values()) == {None}
    decisions = {key: ATOMICITY_SELF_CONTAINED for key in template}
    paths.manual_template.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n")

    result = finalize_iteration(
        payload_path=paths.payload,
        manual_decisions_path=paths.manual_template,
        database_path=paths.database,
        output_dir=tmp_path,
        current_code_commit="a" * 40,
    )
    assert result["report"]["go_no_go"]["status"] == "go"
    assert result["paths"]["json"].exists()
    assert result["paths"]["markdown"].exists()

    tampered_payload = json.loads(paths.payload.read_text())
    tampered_payload["report_inputs"]["provenance"]["db_copy_sha256"] = "c" * 64
    tampered_path = tmp_path / "tampered-payload.json"
    tampered_path.write_text(
        json.dumps(tampered_payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(FunnelPipelineError, match="provenance hash"):
        finalize_iteration(
            payload_path=tampered_path,
            manual_decisions_path=paths.manual_template,
            database_path=paths.database,
            output_dir=tmp_path / "tampered",
            current_code_commit="a" * 40,
        )

    wrong_snapshot_payload = json.loads(paths.payload.read_text())
    wrong_snapshot_payload["snapshot_sha256_attestation"] = "0" * 64
    wrong_snapshot_path = tmp_path / "wrong-snapshot-payload.json"
    wrong_snapshot_path.write_text(
        json.dumps(wrong_snapshot_payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(FunnelPipelineError, match="snapshot attestation"):
        finalize_iteration(
            payload_path=wrong_snapshot_path,
            manual_decisions_path=paths.manual_template,
            database_path=paths.database,
            output_dir=tmp_path / "wrong-snapshot",
            current_code_commit="a" * 40,
        )

    invalid_hash_payload = json.loads(paths.payload.read_text())
    invalid_hash_payload["database_sha256"] = "bad"
    invalid_hash_path = tmp_path / "invalid-hash-payload.json"
    invalid_hash_path.write_text(
        json.dumps(invalid_hash_payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(FunnelPipelineError, match="invalid DB SHA-256"):
        finalize_iteration(
            payload_path=invalid_hash_path,
            manual_decisions_path=paths.manual_template,
            database_path=paths.database,
            output_dir=tmp_path / "invalid-hash",
            current_code_commit="a" * 40,
        )

    wrong_name_payload = json.loads(paths.payload.read_text())
    wrong_name_payload["database_filename"] = "wrong.db"
    wrong_name_path = tmp_path / "wrong-name-payload.json"
    wrong_name_path.write_text(
        json.dumps(wrong_name_payload, indent=2, sort_keys=True) + "\n"
    )
    with pytest.raises(FunnelPipelineError, match="filename"):
        finalize_iteration(
            payload_path=wrong_name_path,
            manual_decisions_path=paths.manual_template,
            database_path=paths.database,
            output_dir=tmp_path / "wrong-name",
            current_code_commit="a" * 40,
        )

    with pytest.raises(FunnelPipelineError, match="code commit"):
        finalize_iteration(
            payload_path=paths.payload,
            manual_decisions_path=paths.manual_template,
            database_path=paths.database,
            output_dir=tmp_path / "wrong-commit",
            current_code_commit="d" * 40,
        )

    second_dir = tmp_path / "mismatch"
    second_dir.mkdir()
    paths.database.write_bytes(paths.database.read_bytes() + b"changed")
    with pytest.raises(FunnelPipelineError, match="SHA-256 mismatch"):
        finalize_iteration(
            payload_path=paths.payload,
            manual_decisions_path=paths.manual_template,
            database_path=paths.database,
            output_dir=second_dir,
            current_code_commit="a" * 40,
        )


def test_preparation_artifacts_are_byte_identical_and_never_overwritten(
    tmp_path: Path,
) -> None:
    claims = [
        {
            "anomaly_id": "a1",
            "claim_text": "NO2 was elevated downtown.",
        }
    ]
    worksheet = build_atomicity_worksheet(claims)
    payload = {
        "schema_version": 1,
        "database_filename": "aeris-b19-funnel-20260716-iteration-001.db",
        "database_sha256": "b" * 64,
        "report_inputs": {"claims": claims},
    }
    first_paths = iteration_paths(tmp_path / "first", run_date="20260716", iteration=1)
    second_paths = iteration_paths(tmp_path / "second", run_date="20260716", iteration=1)

    write_preparation_artifacts(first_paths, payload, worksheet=worksheet)
    write_preparation_artifacts(second_paths, payload, worksheet=worksheet)

    assert first_paths.payload.read_bytes() == second_paths.payload.read_bytes()
    assert first_paths.worksheet.read_bytes() == second_paths.worksheet.read_bytes()
    assert first_paths.manual_template.read_bytes() == second_paths.manual_template.read_bytes()
    with pytest.raises(FileExistsError, match="preserved"):
        write_preparation_artifacts(first_paths, payload, worksheet=worksheet)
