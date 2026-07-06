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
    ClaimGroup,
    _parse_args,
    collect_claim_groups,
    presentation_order,
    run_label_session,
)

ANOMALY_TS = datetime(2026, 6, 5, 18, 0, tzinfo=UTC)

# Pinned so the per-(anomaly, labeler) presentation shuffle is deterministic
# across test runs — the scripted answers below are positional.
ANOMALY_ID = uuid.UUID(int=1)


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
        id=ANOMALY_ID,
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


# --- presentation order ---


def _groups(n: int) -> list[ClaimGroup]:
    return [
        ClaimGroup(claim_text=f"claim {i}", claim_ids=(uuid.uuid4(),))
        for i in range(n)
    ]


def test_presentation_order_is_deterministic_per_anomaly_and_labeler():
    groups = _groups(8)
    first = presentation_order(groups, ANOMALY_ID, "bracco")
    second = presentation_order(groups, ANOMALY_ID, "bracco")
    assert [g.claim_text for g in first] == [g.claim_text for g in second]


def test_presentation_order_decorrelates_from_model_block_order():
    # The raw query order blocks claims by model name; the shuffle must not
    # preserve it (fatigue/anchoring would correlate with model identity).
    # With 8 groups a seed-preserved identity order would be a 1/40320 fluke;
    # both labelers seeing the input order would prove the shuffle inert.
    groups = _groups(8)
    input_order = [g.claim_text for g in groups]
    orders = {
        labeler: [g.claim_text for g in presentation_order(groups, ANOMALY_ID, labeler)]
        for labeler in ("bracco", "mason")
    }
    assert any(order != input_order for order in orders.values())


@pytest.mark.asyncio
async def test_presentation_index_records_shown_order(db_session):
    anomaly = await _seed(db_session)
    io = ScriptedIO(["v", "", "i", "", "cause"])

    label = await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    assert label is not None
    by_index: dict[int, set[str]] = {}
    for v in label.claim_validations_json:
        by_index.setdefault(v["presentation_index"], set()).add(v["verdict"])
    # Two groups shown as 1 and 2; each group's fanned-out ids share one
    # verdict, so the order effects stay analyzable per shown position.
    assert set(by_index) == {1, 2}
    assert all(len(verdicts) == 1 for verdicts in by_index.values())


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
async def test_labeler_text_never_leaks_model_or_pipeline_judgments(db_session):
    # Cross-module invariant: the labeler transcript is built from
    # explain.render_anomaly_text / render_enrichment_text (shared with the
    # LLM-prompt path) plus claim_text only. The three things the labels get
    # correlated against — model identity, grounding verdict, corroboration
    # score — live on Explanation/Claim and must never reach the labeler, even
    # as those shared render functions evolve.
    anomaly = _anomaly()
    claim = _claim("shared claim", step=1)
    claim.grounding_verdict = "grounded"
    claim.corroboration_score = 0.4242  # distinctive, can't render by accident
    expl = _explanation(anomaly, "llama3:8b", [claim])
    db_session.add_all([anomaly, _record(anomaly), expl])
    await db_session.commit()

    io = ScriptedIO(["v", "", "cause"])
    await run_label_session(
        db_session, anomaly.id, labeler="bracco",
        input_fn=io.input_fn, echo=io.echo,
    )

    shown = io.text
    assert "shared claim" in shown  # the claim text is actually presented...
    lowered = shown.lower()
    for leaked in ("llama", "grounded", "corroboration", "0.4242"):
        assert leaked not in lowered, f"labeler saw {leaked!r}"


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
