"""Labeling CLI: present claims blind, capture verdicts, persist ExpertLabel.

The labeler must never see the pipeline's own judgments (grounding verdict,
corroboration score) — those are exactly what the labels get correlated
against. Identical claim texts across models are asked once and the verdict
fans out to every matching claim id.
"""

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from app.db.models import Anomaly, Claim, EnrichmentRecord, ExpertLabel, Explanation
from app.eval.label_cli import (
    _parse_args,
    collect_claim_groups,
    run_label_session,
)

ANOMALY_TS = datetime(2026, 6, 5, 18, 0, tzinfo=UTC)


class ScriptedIO:
    """Canned answers in, transcript out."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.transcript: list[str] = []

    def input_fn(self, prompt: str) -> str:
        self.transcript.append(prompt)
        return self._answers.pop(0)

    def echo(self, text: str = "") -> None:
        self.transcript.append(text)

    @property
    def text(self) -> str:
        return "\n".join(self.transcript)


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
        cross_source_summary_json={
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
        },
    )


def _claim(text: str, step: int = 1) -> Claim:
    return Claim(
        step_index=step,
        claim_type="concentration_elevation",
        claim_text=text,
        cited_sources=["openaq"],
        grounding_verdict="grounded",
        grounding_evidence_ref=None,
        skipped_phase2=False,
        corroboration_score=1.0,
        evidence_n=1,
        per_source_verdicts={"openaq": 1},
        partial_verifiability=False,
        low_corroboration_flag=False,
    )


def _explanation(anomaly: Anomaly, model: str, claims: list[Claim]) -> Explanation:
    return Explanation(
        anomaly_id=anomaly.id,
        model_name=model,
        model_version="v1",
        reasoning_steps_json={"steps": []},
        final_narrative="narrative",
        stated_confidence=0.7,
        claims=claims,
    )


async def _seed(db_session) -> Anomaly:
    """One anomaly, two models; 'shared claim' appears in both explanations."""
    anomaly = _anomaly()
    db_session.add_all(
        [
            anomaly,
            _record(anomaly),
            _explanation(
                anomaly,
                "model-a",
                [_claim("shared claim", step=1), _claim("only in a", step=2)],
            ),
            _explanation(anomaly, "model-b", [_claim("shared claim", step=1)]),
        ]
    )
    await db_session.commit()
    return anomaly


# --- claim grouping ---


@pytest.mark.asyncio
async def test_collect_claim_groups_dedups_identical_text(db_session):
    anomaly = await _seed(db_session)
    groups = await collect_claim_groups(db_session, anomaly.id)

    assert [g.claim_text for g in groups] == ["shared claim", "only in a"]
    assert len(groups[0].claim_ids) == 2  # one per model
    assert len(groups[1].claim_ids) == 1


@pytest.mark.asyncio
async def test_collect_claim_groups_can_filter_by_model(db_session):
    anomaly = await _seed(db_session)
    groups = await collect_claim_groups(db_session, anomaly.id, model="model-b")
    assert [g.claim_text for g in groups] == ["shared claim"]
    assert len(groups[0].claim_ids) == 1


# --- labeling session ---


@pytest.mark.asyncio
async def test_session_persists_verdicts_for_every_claim_id(db_session):
    anomaly = await _seed(db_session)
    io = ScriptedIO(
        [
            "v", "matches the station data",  # shared claim + note
            "i", "",                          # only in a, no note
            "stagnation under weak winds",    # true cause
        ]
    )

    label = await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    assert label is not None
    assert label.labeler == "bracco"
    assert label.true_cause == "stagnation under weak winds"
    validations = {v["claim_id"]: v for v in label.claim_validations_json}
    assert len(validations) == 3  # every claim id, including both shared copies
    verdicts = sorted(v["verdict"] for v in validations.values())
    assert verdicts == ["invalid", "valid", "valid"]
    noted = [v for v in validations.values() if v["note"]]
    assert len(noted) == 2  # the shared-claim note fans out with the verdict

    stored = (
        await db_session.execute(
            select(ExpertLabel).where(ExpertLabel.anomaly_id == anomaly.id)
        )
    ).scalars().all()
    assert len(stored) == 1


@pytest.mark.asyncio
async def test_session_is_blind_to_pipeline_judgments(db_session):
    anomaly = await _seed(db_session)
    io = ScriptedIO(["v", "", "v", "", "cause"])

    await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    shown = io.text.lower()
    assert "corroboration" not in shown
    assert "grounded" not in shown
    assert "unverified" not in shown
    assert "model-a" not in shown  # model identity would bias too


@pytest.mark.asyncio
async def test_invalid_verdict_input_reprompts(db_session):
    anomaly = await _seed(db_session)
    io = ScriptedIO(["x", "v", "", "u", "", "cause"])

    label = await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    assert label is not None
    verdicts = sorted(v["verdict"] for v in label.claim_validations_json)
    assert verdicts == ["unsure", "valid", "valid"]


@pytest.mark.asyncio
async def test_quit_aborts_without_persisting(db_session):
    anomaly = await _seed(db_session)
    io = ScriptedIO(["v", "", "q"])

    label = await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    assert label is None
    stored = (
        await db_session.execute(
            select(ExpertLabel).where(ExpertLabel.anomaly_id == anomaly.id)
        )
    ).scalars().all()
    assert stored == []


@pytest.mark.asyncio
async def test_relabeling_same_labeler_is_refused(db_session):
    anomaly = await _seed(db_session)
    io = ScriptedIO(["v", "", "v", "", "cause"])
    await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    with pytest.raises(ValueError, match="already labeled"):
        await run_label_session(
            db_session, anomaly.id, labeler="bracco",
            input_fn=ScriptedIO([]).input_fn, echo=lambda *_: None,
        )

    # A different labeler on the same anomaly is the IRR overlap case.
    io2 = ScriptedIO(["v", "", "v", "", "cause"])
    second = await run_label_session(
        db_session, anomaly.id, labeler="mason",
        input_fn=io2.input_fn, echo=io2.echo,
    )
    assert second is not None


@pytest.mark.asyncio
async def test_anomaly_without_claims_is_an_error(db_session):
    anomaly = _anomaly()
    db_session.add_all([anomaly, _record(anomaly)])
    await db_session.commit()

    with pytest.raises(ValueError, match="no claims"):
        await run_label_session(
            db_session, anomaly.id, labeler="bracco",
            input_fn=ScriptedIO([]).input_fn, echo=lambda *_: None,
        )


# --- CLI ---


def test_parse_args_requires_anomaly_and_labeler():
    args = _parse_args(["--anomaly-id", "abc", "--labeler", "bracco"])
    assert args.anomaly_id == "abc"
    assert args.labeler == "bracco"
    assert args.model is None

    args = _parse_args(
        ["--anomaly-id", "abc", "--labeler", "mason", "--model", "llama3:8b"]
    )
    assert args.model == "llama3:8b"
