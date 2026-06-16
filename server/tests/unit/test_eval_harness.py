"""Eval harness: sweep (anomaly x model) cells, idempotently and fault-isolated.

Each cell runs the full explain pipeline and persists one Explanation; cells
that already exist are skipped before any LLM call, parse failures are counted
per model (a first-class result), and one bad cell never kills the sweep.
"""

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.models import Anomaly, EnrichmentRecord, Explanation
from app.eval.harness import (
    DEFAULT_MODELS,
    _parse_args,
    load_anomaly_set,
    run_harness,
)
from app.llm.client_base import LLMClient, RawCompletion

ANOMALY_TS = datetime(2026, 6, 5, 18, 0, tzinfo=UTC)


class ScriptedClient(LLMClient):
    """LLMClient returning scripted texts, one per call, counting calls."""

    def __init__(self, model: str, texts: list[str]) -> None:
        super().__init__()
        self.model_name = model
        self.model_version = "v1"
        self._texts = texts
        self.calls = 0

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        self.calls += 1
        return RawCompletion(
            text=self._texts.pop(0), prompt_tokens=10, completion_tokens=5
        )


class ExplodingClient(LLMClient):
    """LLMClient whose every call fails with a non-parse error."""

    def __init__(self, model: str) -> None:
        super().__init__()
        self.model_name = model
        self.model_version = "v1"

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        raise RuntimeError("connection torn down")


def _chain_script() -> list[str]:
    step = json.dumps(
        {
            "summary": "NO2 elevated near the anomaly",
            "claims": [
                {
                    "statement": "NO2 exceeded 80 ppb, nearest reading 82.0 ppb",
                    "cited_sources": ["openaq"],
                }
            ],
        }
    )
    empty = json.dumps({"summary": "nothing further", "claims": []})
    synthesis = json.dumps(
        {
            "final_narrative": "Local NO2 buildup under weak dispersion.",
            "stated_confidence": 0.7,
            "claims": [],
        }
    )
    return [step, empty, empty, synthesis]


def _scripted_factory(registry: dict[str, ScriptedClient], n_anomalies: int):
    def factory(model: str) -> LLMClient:
        client = ScriptedClient(model, _chain_script() * n_anomalies)
        registry[model] = client
        return client

    return factory


def _summary() -> dict:
    return {
        "schema_version": 1,
        "window": {
            "start": "2026-06-05T06:00:00+00:00",
            "end": "2026-06-06T00:00:00+00:00",
            "spatial_radius_km": 25.0,
        },
        "coverage": {"openaq": True},
        "sources": {
            "openaq": {
                "n_entities": 5,
                "n_points": 40,
                "metrics": {
                    "no2": {
                        "unit": "ppb",
                        "n_points": 40,
                        "n_entities": 5,
                        "value_range": {"min": 60.0, "max": 85.0, "mean": 72.0},
                        "nearest_in_time": {
                            "t": "2026-06-05T17:42:00+00:00",
                            "v": 82.0,
                            "entity_id": "st-1",
                            "distance_km": 2.1,
                            "dt_minutes": 18.0,
                        },
                        "entities": [],
                    }
                },
            }
        },
    }


def _anomaly() -> Anomaly:
    return Anomaly(
        id=uuid.uuid4(),
        timestamp=ANOMALY_TS,
        lat=29.76,
        lon=-95.37,
        metric="no2",
        source="openaq",
        value=85.0,
        expected_value=58.0,
        z_score=4.2,
        methods_triggered=["zscore"],
        severity="severe",
    )


def _record(anomaly: Anomaly) -> EnrichmentRecord:
    return EnrichmentRecord(
        anomaly_id=anomaly.id,
        context_window_start=datetime(2026, 6, 5, 6, 0, tzinfo=UTC),
        context_window_end=datetime(2026, 6, 6, 0, 0, tzinfo=UTC),
        cross_source_summary_json=_summary(),
    )


async def _seed(db_session, n: int) -> list[uuid.UUID]:
    anomalies = [_anomaly() for _ in range(n)]
    db_session.add_all(anomalies)
    db_session.add_all(_record(a) for a in anomalies)
    await db_session.commit()
    return [a.id for a in anomalies]


async def _n_explanations(db_session, ids: list[uuid.UUID]) -> int:
    return (
        await db_session.execute(
            select(func.count(Explanation.id)).where(Explanation.anomaly_id.in_(ids))
        )
    ).scalar_one()


# --- anomaly-set loading ---


def test_load_anomaly_set_accepts_plain_array(tmp_path):
    ids = [str(uuid.uuid4()), str(uuid.uuid4())]
    path = tmp_path / "set.json"
    path.write_text(json.dumps(ids))
    assert load_anomaly_set(path) == [uuid.UUID(i) for i in ids]


def test_load_anomaly_set_accepts_frozen_fixture_object(tmp_path):
    ids = [str(uuid.uuid4())]
    path = tmp_path / "eval50.json"
    path.write_text(json.dumps({"frozen_at": "2026-07-13", "anomaly_ids": ids}))
    assert load_anomaly_set(path) == [uuid.UUID(ids[0])]


def test_load_anomaly_set_rejects_other_shapes(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"ids": []}))
    with pytest.raises(ValueError, match="anomaly_ids"):
        load_anomaly_set(path)


def test_load_anomaly_set_dedups_preserving_order(tmp_path):
    a, b = str(uuid.uuid4()), str(uuid.uuid4())
    path = tmp_path / "dups.json"
    path.write_text(json.dumps([a, b, a]))  # a duplicated would sweep twice
    assert load_anomaly_set(path) == [uuid.UUID(a), uuid.UUID(b)]


# --- sweep ---


@pytest.mark.asyncio
async def test_run_harness_fills_every_cell(db_session):
    ids = await _seed(db_session, 2)
    registry: dict[str, ScriptedClient] = {}

    summaries = await run_harness(
        db_session,
        ids,
        models=["model-a", "model-b"],
        client_factory=_scripted_factory(registry, n_anomalies=2),
    )

    assert await _n_explanations(db_session, ids) == 4
    for model in ("model-a", "model-b"):
        assert summaries[model].completed == 2
        assert summaries[model].skipped == 0
        assert summaries[model].parse_failures == 0
        assert summaries[model].errors == []


@pytest.mark.asyncio
async def test_persist_returning_false_counts_as_skip(db_session, monkeypatch):
    # A raced/duplicate persist returns False; it must not inflate `completed`.
    import app.eval.harness as harness_mod

    ids = await _seed(db_session, 1)

    async def _persist_false(session, explanation):
        return False

    monkeypatch.setattr(harness_mod, "persist_explanation", _persist_false)

    summaries = await run_harness(
        db_session, ids, models=["model-a"],
        client_factory=_scripted_factory({}, n_anomalies=1),
    )

    assert summaries["model-a"].completed == 0
    assert summaries["model-a"].skipped == 1


@pytest.mark.asyncio
async def test_rerun_skips_existing_cells_without_llm_calls(db_session):
    ids = await _seed(db_session, 2)
    first: dict[str, ScriptedClient] = {}
    await run_harness(
        db_session, ids, models=["model-a"],
        client_factory=_scripted_factory(first, n_anomalies=2),
    )

    second: dict[str, ScriptedClient] = {}
    summaries = await run_harness(
        db_session, ids, models=["model-a"],
        client_factory=_scripted_factory(second, n_anomalies=2),
    )

    assert summaries["model-a"].skipped == 2
    assert summaries["model-a"].completed == 0
    # Every cell is resumed-complete, so no client is constructed at all.
    assert "model-a" not in second
    assert await _n_explanations(db_session, ids) == 2


@pytest.mark.asyncio
async def test_all_skipped_model_builds_no_client(db_session):
    # A fully-resumed model must not pay to construct its (cloud) client when
    # there is no cell left to fill.
    ids = await _seed(db_session, 2)
    await run_harness(
        db_session, ids, models=["model-a"],
        client_factory=_scripted_factory({}, n_anomalies=2),
    )

    built: list[str] = []

    def counting_factory(model: str) -> LLMClient:
        built.append(model)
        return ScriptedClient(model, _chain_script() * 2)

    summaries = await run_harness(
        db_session, ids, models=["model-a"], client_factory=counting_factory
    )

    assert summaries["model-a"].skipped == 2
    assert built == []


@pytest.mark.asyncio
async def test_parse_failure_is_counted_and_sweep_continues(db_session):
    ids = await _seed(db_session, 2)
    registry: dict[str, ScriptedClient] = {}

    def factory(model: str) -> LLMClient:
        # First anomaly: two unparseable responses -> LLMParseError
        # (generate() retries once). Second anomaly: clean 4-step chain.
        client = ScriptedClient(model, ["not json", "still not json"] + _chain_script())
        registry[model] = client
        return client

    summaries = await run_harness(
        db_session, ids, models=["model-a"], client_factory=factory
    )

    assert summaries["model-a"].parse_failures == 1
    assert summaries["model-a"].completed == 1
    assert await _n_explanations(db_session, ids) == 1


@pytest.mark.asyncio
async def test_unexpected_error_is_recorded_and_isolated(db_session):
    ids = await _seed(db_session, 2)

    summaries = await run_harness(
        db_session, ids, models=["model-a"],
        client_factory=lambda model: ExplodingClient(model),
    )

    assert summaries["model-a"].completed == 0
    assert len(summaries["model-a"].errors) == 2
    assert "connection torn down" in summaries["model-a"].errors[0][1]


# --- CLI ---


def test_parse_args_takes_set_path_and_model_list():
    args = _parse_args(["--anomaly-set", "fixtures/eval50.json"])
    assert args.anomaly_set == "fixtures/eval50.json"
    assert args.models == list(DEFAULT_MODELS)

    args = _parse_args(
        ["--anomaly-set", "s.json", "--models", "llama3:8b,gpt-5.4"]
    )
    assert args.models == ["llama3:8b", "gpt-5.4"]


def test_default_models_are_the_three_baselines():
    assert DEFAULT_MODELS == ("llama3:8b", "gpt-5.4", "gemini-3-flash-preview")
