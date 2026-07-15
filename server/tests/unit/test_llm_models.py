import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import UniqueConstraint, select

from app.db.models import Anomaly, Claim, ExpertLabel, Explanation


HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698


def test_integrity_pairs_have_named_unique_constraints() -> None:
    explanation_constraints = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in Explanation.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }
    label_constraints = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in ExpertLabel.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    }

    assert explanation_constraints["uq_explanations_anomaly_model"] == (
        "anomaly_id",
        "model_name",
    )
    assert label_constraints["uq_expert_labels_anomaly_labeler"] == (
        "anomaly_id",
        "labeler",
    )


def _make_anomaly(**overrides) -> Anomaly:
    defaults = dict(
        timestamp=datetime(2026, 7, 16, 14, 30, tzinfo=timezone.utc),
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
        metric="o3",
        source="openaq",
        value=0.092,
        expected_value=0.045,
        z_score=3.9,
        methods_triggered=["zscore", "stl"],
        severity="moderate",
    )
    defaults.update(overrides)
    return Anomaly(**defaults)


def _make_explanation(anomaly_id, **overrides) -> Explanation:
    defaults = dict(
        anomaly_id=anomaly_id,
        model_name="llama3:8b",
        model_version="2024-04",
        reasoning_steps_json={
            "physical_signature": "O3 elevated to 92 ppb in the afternoon",
            "candidate_causes": ["photochemical formation", "regional transport"],
            "evidence_evaluation": "winds light, high insolation",
            "synthesis": "local photochemical event",
        },
        final_narrative="Afternoon ozone peak consistent with photochemical formation.",
        stated_confidence=0.7,
        latency_ms=18234.5,
        prompt_tokens=1200,
        completion_tokens=480,
    )
    defaults.update(overrides)
    return Explanation(**defaults)


def _make_claim(explanation_id, **overrides) -> Claim:
    defaults = dict(
        explanation_id=explanation_id,
        step_index=1,
        claim_type="concentration_elevation",
        matched_types=["concentration_elevation", "temporal_pattern"],
        claim_text="Ground-level O3 exceeded 90 ppb in the afternoon.",
        cited_sources=["openaq"],
        citation_outcome="cited_right",
        grounding_verdict="grounded",
        grounding_evidence_ref={"source": "openaq", "metric": "o3", "value": 0.092},
        causal=False,
        skipped_phase2=False,
        corroboration_score=0.5,
        evidence_n=2,
        per_source_verdicts={"openaq": 1, "sentinel5p": 0, "gfs": -1, "openweather": 1},
        per_channel_verdicts={"ground_insitu": 1, "nwp": -1},
        partial_verifiability=False,
        low_corroboration_flag=False,
    )
    defaults.update(overrides)
    return Claim(**defaults)


class TestExplanationModel:
    @pytest.mark.asyncio
    async def test_persists_with_all_fields(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        db_session.add(_make_explanation(anomaly.id))
        await db_session.commit()

        loaded = (await db_session.execute(select(Explanation))).scalar_one()
        assert isinstance(loaded.id, uuid.UUID)
        assert loaded.anomaly_id == anomaly.id
        assert loaded.model_name == "llama3:8b"
        assert loaded.model_version == "2024-04"
        assert loaded.final_narrative.startswith("Afternoon ozone")
        assert loaded.stated_confidence == pytest.approx(0.7)
        assert loaded.latency_ms == pytest.approx(18234.5)
        assert loaded.prompt_tokens == 1200
        assert loaded.completion_tokens == 480
        assert loaded.created_at is not None

    @pytest.mark.asyncio
    async def test_reasoning_steps_json_roundtrips(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        db_session.add(_make_explanation(anomaly.id))
        await db_session.commit()

        loaded = (await db_session.execute(select(Explanation))).scalar_one()
        assert loaded.reasoning_steps_json["candidate_causes"] == [
            "photochemical formation",
            "regional transport",
        ]

    @pytest.mark.asyncio
    async def test_token_and_latency_fields_nullable(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        db_session.add(
            _make_explanation(
                anomaly.id,
                stated_confidence=None,
                latency_ms=None,
                prompt_tokens=None,
                completion_tokens=None,
            )
        )
        await db_session.commit()

        loaded = (await db_session.execute(select(Explanation))).scalar_one()
        assert loaded.stated_confidence is None
        assert loaded.latency_ms is None
        assert loaded.prompt_tokens is None


class TestClaimModel:
    @pytest.mark.asyncio
    async def test_persists_phase1_and_phase2_fields(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        explanation = _make_explanation(anomaly.id)
        db_session.add(explanation)
        await db_session.flush()
        db_session.add(_make_claim(explanation.id))
        await db_session.commit()

        loaded = (await db_session.execute(select(Claim))).scalar_one()
        assert loaded.explanation_id == explanation.id
        assert loaded.step_index == 1
        assert loaded.claim_type == "concentration_elevation"
        assert loaded.grounding_verdict == "grounded"
        assert loaded.citation_outcome == "cited_right"
        assert loaded.grounding_evidence_ref["metric"] == "o3"
        assert loaded.skipped_phase2 is False
        assert loaded.corroboration_score == pytest.approx(0.5)
        assert loaded.evidence_n == 2
        assert loaded.per_source_verdicts["gfs"] == -1
        assert loaded.matched_types == ["concentration_elevation", "temporal_pattern"]
        assert loaded.causal is False
        assert loaded.per_channel_verdicts == {"ground_insitu": 1, "nwp": -1}
        assert loaded.partial_verifiability is False
        assert loaded.quantitative_exclusion_reason is None
        assert loaded.low_corroboration_flag is False

    @pytest.mark.asyncio
    async def test_quantitative_exclusion_reason_roundtrips(self, db_session) -> None:
        anomaly = _make_anomaly(metric="so2")
        db_session.add(anomaly)
        await db_session.flush()
        explanation = _make_explanation(anomaly.id)
        db_session.add(explanation)
        await db_session.flush()
        db_session.add(
            _make_claim(
                explanation.id,
                claim_text="SO2 exceeded 1 ppb.",
                quantitative_exclusion_reason="so2_underpowered",
            )
        )
        await db_session.commit()

        loaded = (await db_session.execute(select(Claim))).scalar_one()
        assert loaded.quantitative_exclusion_reason == "so2_underpowered"

    @pytest.mark.asyncio
    async def test_phase1_unverified_claim_skips_phase2(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        explanation = _make_explanation(anomaly.id)
        db_session.add(explanation)
        await db_session.flush()
        db_session.add(
            _make_claim(
                explanation.id,
                grounding_verdict="unverified",
                grounding_evidence_ref=None,
                skipped_phase2=True,
                corroboration_score=None,
                evidence_n=0,
                per_source_verdicts=None,
            )
        )
        await db_session.commit()

        loaded = (await db_session.execute(select(Claim))).scalar_one()
        assert loaded.grounding_verdict == "unverified"
        assert loaded.grounding_evidence_ref is None
        assert loaded.skipped_phase2 is True
        assert loaded.corroboration_score is None

    @pytest.mark.asyncio
    async def test_cascades_when_explanation_deleted(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        explanation = _make_explanation(anomaly.id)
        db_session.add(explanation)
        await db_session.flush()
        db_session.add(_make_claim(explanation.id))
        await db_session.commit()

        await db_session.delete(explanation)
        await db_session.commit()

        remaining = (await db_session.execute(select(Claim))).scalars().all()
        assert remaining == []


class TestExpertLabelModel:
    @pytest.mark.asyncio
    async def test_persists_linked_to_anomaly(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        label = ExpertLabel(
            anomaly_id=anomaly.id,
            labeler="bracco",
            true_cause="photochemical ozone formation",
            claim_validations_json=[
                {"claim_id": "c1", "verdict": "correct", "note": "matches met data"},
                {"claim_id": "c2", "verdict": "incorrect", "note": "wrong direction"},
            ],
        )
        db_session.add(label)
        await db_session.commit()

        loaded = (await db_session.execute(select(ExpertLabel))).scalar_one()
        assert loaded.anomaly_id == anomaly.id
        assert loaded.labeler == "bracco"
        assert loaded.claim_validations_json[0]["verdict"] == "correct"
        assert loaded.created_at is not None

    @pytest.mark.asyncio
    async def test_cascades_when_anomaly_deleted(self, db_session) -> None:
        anomaly = _make_anomaly()
        db_session.add(anomaly)
        await db_session.flush()
        explanation = _make_explanation(anomaly.id)
        db_session.add(explanation)
        await db_session.flush()
        db_session.add(_make_claim(explanation.id))
        db_session.add(
            ExpertLabel(anomaly_id=anomaly.id, labeler="mason", true_cause="x")
        )
        await db_session.commit()

        await db_session.delete(anomaly)
        await db_session.commit()

        assert (await db_session.execute(select(Explanation))).scalars().all() == []
        assert (await db_session.execute(select(Claim))).scalars().all() == []
        assert (await db_session.execute(select(ExpertLabel))).scalars().all() == []
