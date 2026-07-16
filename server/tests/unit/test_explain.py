"""Explanation orchestrator: chain -> claims -> Phase 1 gate -> Phase 2 -> rows.

Uses a scripted LLM client so the pipeline runs without Ollama; the grounded
claim cites numbers that exist in the enrichment context, the fabricated one
does not, and only the grounded claim may reach Phase 2.
"""

import json
import uuid
from datetime import UTC, datetime

import pytest
from pydantic import BaseModel
from sqlalchemy import func, select

from app.db.models import Anomaly, Claim, EnrichmentRecord, Explanation
from app.llm.client_base import LLMClient, RawCompletion
from app.llm.explain import (
    _format_explanation,
    _parse_args,
    _with_unit,
    generate_explanation,
    make_client,
    persist_explanation,
    render_anomaly_text,
    render_enrichment_text,
)


def test_with_unit_renders_na_for_missing_value():
    # A None value used to render the literal string "None" into the grounding
    # context; show "n/a" instead.
    assert _with_unit(None, "ppb") == "n/a"
    assert _with_unit(None, None) == "n/a"
    assert _with_unit(12.0, "ppb") == "12.0 ppb"
    assert _with_unit(12.0, None) == "12.0"

ANOMALY_TS = datetime(2026, 6, 5, 18, 0, tzinfo=UTC)


class ScriptedClient(LLMClient):
    """LLMClient that returns scripted raw JSON, one response per call."""

    model_name = "scripted"
    model_version = "v1"

    def __init__(self, texts: list[str]) -> None:
        super().__init__()
        self._texts = list(texts)

    async def _complete(self, prompt: str, schema: type[BaseModel]) -> RawCompletion:
        return RawCompletion(
            text=self._texts.pop(0), prompt_tokens=10, completion_tokens=5
        )


def _script() -> list[str]:
    step = {
        "summary": "NO2 well above the in-window mean near the anomaly",
        "claims": [
            {
                "statement": "NO2 exceeded 80 ppb, nearest reading 82.0 ppb",
                "cited_sources": ["openaq"],
            }
        ],
    }
    empty = {"summary": "nothing further", "claims": []}
    synthesis = {
        "final_narrative": "Local NO2 buildup under weak dispersion.",
        "stated_confidence": 0.72,
        "claims": [
            {
                "statement": "SO2 spiked to 500 ppb from the ship channel",
                "cited_sources": [],
            }
        ],
    }
    return [json.dumps(step), json.dumps(empty), json.dumps(empty), json.dumps(synthesis)]


def _summary() -> dict:
    return {
        "schema_version": 1,
        "anomaly": {
            "timestamp": ANOMALY_TS.isoformat(),
            "lat": 29.76,
            "lon": -95.37,
            "metric": "no2",
            "source": "openaq",
            "value": 85.0,
            "severity": "severe",
        },
        "window": {
            "start": "2026-06-05T06:00:00+00:00",
            "end": "2026-06-06T00:00:00+00:00",
            "hours_before": 12,
            "hours_after": 6,
            "spatial_radius_km": 25.0,
        },
        "coverage": {
            "openaq": True,
            "openweather": False,
            "noaa_gfs": False,
            "sentinel5p": False,
        },
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


def test_render_enrichment_text_applies_only_ratified_prompt_pruning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary = _summary()
    summary["sources"]["noaa_gfs"] = {
        "metrics": {
            "gh_500": {
                "unit": "m",
                "n_points": 1,
                "n_entities": 1,
                "value_range": {"min": 5840.0, "max": 5840.0, "mean": 5840.0},
                "nearest_in_time": {
                    "t": ANOMALY_TS.isoformat(),
                    "v": 5840.0,
                    "distance_km": 0.0,
                    "dt_minutes": 0.0,
                },
                "entities": [],
            },
            "pbl_height": {
                "unit": "m",
                "n_points": 1,
                "n_entities": 1,
                "value_range": {"min": 500.0, "max": 500.0, "mean": 500.0},
                "nearest_in_time": {
                    "t": ANOMALY_TS.isoformat(),
                    "v": 500.0,
                    "distance_km": 0.0,
                    "dt_minutes": 0.0,
                },
                "entities": [],
            },
        }
    }
    monkeypatch.setattr(
        "app.llm.explain.metric_is_retained",
        lambda source, metric: (source, metric) != ("noaa_gfs", "gh_500"),
    )

    rendered = render_enrichment_text(summary)

    assert "noaa_gfs gh_500" not in rendered
    assert "noaa_gfs pbl_height" in rendered


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
        methods_triggered=["zscore", "stl"],
        severity="severe",
    )


def _record(anomaly: Anomaly) -> EnrichmentRecord:
    return EnrichmentRecord(
        anomaly_id=anomaly.id,
        context_window_start=datetime(2026, 6, 5, 6, 0, tzinfo=UTC),
        context_window_end=datetime(2026, 6, 6, 0, 0, tzinfo=UTC),
        cross_source_summary_json=_summary(),
    )


# --- rendering ---


def test_render_anomaly_text_names_the_signal():
    text = render_anomaly_text(_anomaly())
    assert "no2" in text
    assert "openaq" in text
    assert "85.0" in text
    assert "severe" in text
    assert "4.2" in text


def test_render_enrichment_text_includes_per_metric_stats():
    text = render_enrichment_text(_summary())
    assert "openaq" in text
    assert "no2" in text
    assert "82.0 ppb" in text  # nearest-in-time, what claims get grounded against
    assert "72.0 ppb" in text  # in-window mean
    assert "5 sensors" in text


def test_render_enrichment_text_names_silent_sources():
    text = render_enrichment_text(_summary())
    assert "sentinel5p" in text
    assert "no data" in text.lower()


def test_render_enrichment_text_includes_series_and_sensor_means():
    # Temporal and spatial claim types are scored against the series; the
    # model (and the labeler) must see the same shape, not just aggregates.
    summary = _summary()
    summary["sources"]["openaq"]["metrics"]["no2"]["entities"] = [
        {
            "entity_id": "st-1",
            "distance_km": 2.1,
            "series": [
                [f"2026-06-05T{h:02d}:00:00+00:00", v]
                for h, v in [(8, 60.0), (11, 64.0), (14, 70.0), (17, 82.0)]
            ],
        },
        {
            "entity_id": "st-2",
            "distance_km": 6.4,
            "series": [["2026-06-05T12:00:00+00:00", 66.0]],
        },
    ]
    text = render_enrichment_text(summary)
    # 6 h buckets: hours 8+11 pool into 06Z (mean 62), 14+17 into 12Z. The
    # 3 h width made the bucket lines dominate a ~24k-char rendered context,
    # past what llama3 8B's 8192-token window holds.
    assert "6h means:" in text
    assert "06-05 06Z 62" in text
    assert "sensor means:" in text
    assert "st-2 (6.4 km) 66" in text


def test_render_enrichment_text_omits_series_lines_when_sparse():
    # Aggregate-only summaries (one point, one entity) omit detail lines.
    text = render_enrichment_text(_summary())
    assert "6h means:" not in text
    assert "sensor means:" not in text


def test_render_enrichment_text_uses_grid_cell_terms():
    summary = _summary()
    block = summary["sources"].pop("openaq")
    summary["sources"]["noaa_gfs"] = block

    text = render_enrichment_text(summary)

    assert "5 grid cells" in text
    assert "5 stations" not in text


def test_render_enrichment_text_suppresses_granule_means():
    summary = _summary()
    block = summary["sources"].pop("openaq")
    metric = block["metrics"]["no2"]
    metric["entities"] = [
        {
            "entity_id": f"granule-{index}",
            "distance_km": 0.0,
            "series": [[f"2026-06-0{index}T12:00:00+00:00", float(index)]],
        }
        for index in (1, 2)
    ]
    summary["sources"]["sentinel5p"] = block

    text = render_enrichment_text(summary)

    assert "5 granules" in text
    assert "granule means:" not in text
    assert "station means:" not in text


# --- orchestration ---


@pytest.mark.asyncio
async def test_generate_explanation_builds_explanation_and_claims(db_session):
    anomaly = _anomaly()
    db_session.add_all([anomaly, _record(anomaly)])
    await db_session.commit()

    explanation = await generate_explanation(
        db_session, anomaly.id, ScriptedClient(_script())
    )

    assert explanation.anomaly_id == anomaly.id
    assert explanation.model_name == "scripted"
    assert explanation.final_narrative == "Local NO2 buildup under weak dispersion."
    assert explanation.stated_confidence == 0.72
    assert len(explanation.reasoning_steps_json["steps"]) == 4
    assert [
        {
            "prompt_tokens": step["prompt_tokens"],
            "completion_tokens": step["completion_tokens"],
            "attempts": step["attempts"],
        }
        for step in explanation.reasoning_steps_json["steps"]
    ] == [
        {"prompt_tokens": 10, "completion_tokens": 5, "attempts": 1}
        for _ in range(4)
    ]
    assert explanation.prompt_tokens == 40
    assert explanation.completion_tokens == 20
    assert len(explanation.claims) == 2

    grounded, fabricated = explanation.claims
    assert grounded.step_index == 1
    assert grounded.grounding_verdict == "grounded"
    assert grounded.citation_outcome == "cited_right"
    assert grounded.citation_failure_reasons_json == []
    assert grounded.skipped_phase2 is False
    assert grounded.claim_type == "concentration_elevation"
    assert grounded.matched_types[0] == "concentration_elevation"
    assert grounded.causal is False
    # Phase 2 ran but the only supporting source is the anomaly's own trigger
    # channel (openaq no2), whose tautological support is demoted to silent —
    # no independent channel corroborates, so the score is honestly None.
    assert grounded.corroboration_score is None
    assert grounded.corroboration_evidence_summary
    assert "trigger" in grounded.corroboration_evidence_summary.lower()
    assert grounded.evidence_n == 0
    assert grounded.per_source_verdicts["openaq"] == 0
    assert grounded.per_channel_verdicts["ground_insitu"] == 0
    assert grounded.cited_sources == ["openaq"]
    assert grounded.quantitative_exclusion_reason is None

    assert fabricated.step_index == 4
    assert fabricated.grounding_verdict == "unverified"
    assert fabricated.citation_outcome == "uncited"
    assert fabricated.citation_failure_reasons_json == []
    assert fabricated.skipped_phase2 is True
    assert fabricated.causal is False
    assert fabricated.corroboration_score is None
    assert fabricated.corroboration_evidence_summary is None
    assert fabricated.evidence_n == 0
    # Typed even though never scored: which types get fabricated is a result.
    # "ship channel" is a geographic reference, not an attribution cue, so
    # this reads as a concentration claim about SO2.
    assert fabricated.claim_type == "concentration_elevation"
    assert fabricated.matched_types == ["concentration_elevation"]
    assert fabricated.partial_verifiability is False
    assert fabricated.quantitative_exclusion_reason == "so2_underpowered"

    formatted = _format_explanation(explanation, persisted=False)
    assert "cited_right" in formatted
    assert "uncited" in formatted


@pytest.mark.asyncio
async def test_generate_explanation_requires_enrichment(db_session):
    anomaly = _anomaly()
    db_session.add(anomaly)
    await db_session.commit()

    with pytest.raises(ValueError, match="enrichment"):
        await generate_explanation(db_session, anomaly.id, ScriptedClient(_script()))


@pytest.mark.asyncio
async def test_persist_explanation_is_insert_only_per_model(db_session):
    anomaly = _anomaly()
    db_session.add_all([anomaly, _record(anomaly)])
    await db_session.commit()

    first = await generate_explanation(
        db_session, anomaly.id, ScriptedClient(_script())
    )
    assert await persist_explanation(db_session, first) is True

    second = await generate_explanation(
        db_session, anomaly.id, ScriptedClient(_script())
    )
    assert await persist_explanation(db_session, second) is False

    n_explanations = (
        await db_session.execute(select(func.count(Explanation.id)))
    ).scalar_one()
    n_claims = (
        await db_session.execute(select(func.count(Claim.id)))
    ).scalar_one()
    assert n_explanations == 1
    assert n_claims == 2
    persisted = (
        await db_session.execute(select(Explanation))
    ).scalar_one()
    assert persisted.reasoning_steps_json["steps"][0]["prompt_tokens"] == 10


@pytest.mark.asyncio
async def test_generate_explanation_persists_parse_attempt_count(db_session):
    anomaly = _anomaly()
    db_session.add_all([anomaly, _record(anomaly)])
    await db_session.commit()

    explanation = await generate_explanation(
        db_session,
        anomaly.id,
        ScriptedClient(["not-json", *_script()]),
    )

    steps = explanation.reasoning_steps_json["steps"]
    assert [step["attempts"] for step in steps] == [2, 1, 1, 1]


# --- CLI ---


def test_parse_args_takes_anomaly_id_and_model():
    args = _parse_args(["--anomaly-id", "abc-123", "--model", "llama3:8b"])
    assert args.anomaly_id == "abc-123"
    assert args.model == "llama3:8b"


def test_make_client_routes_models_to_providers(monkeypatch):
    from app.config import settings
    from app.llm.gemini_client import GeminiClient
    from app.llm.gpt_client import GPTClient
    from app.llm.ollama_client import OllamaClient

    monkeypatch.setattr(settings, "openai_api_key", "sk-test")
    monkeypatch.setattr(settings, "google_api_key", "g-test")

    assert isinstance(make_client("gpt-5.4"), GPTClient)
    assert isinstance(make_client("gemini-3-thinking"), GeminiClient)
    local = make_client("llama3:8b")
    assert isinstance(local, OllamaClient)
    assert local.model_name == "llama3:8b"
