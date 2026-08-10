"""Export of the B16 analysis payload from the database.

The payload has to satisfy build_analysis_manifest exactly, and it has to keep
missing distinct from unsure — a blank mark reaches the analysis as a null
verdict, and an unlabeled claim simply has no label row.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.db.models import Anomaly, Claim, ExpertLabel, Explanation
from app.eval.analysis_export import (
    AnalysisExportError,
    _frozen_anomaly_ids,
    build_payload,
    write_payload,
)
from app.eval.phase_analysis import AnalysisThresholds, build_analysis_manifest

MODELS = ("gpt-5.4", "gemini-3.6-flash", "llama3:8b")
FAST = AnalysisThresholds(bootstrap_resamples=99, wilcoxon_monte_carlo_resamples=99)


def _anomaly(index: int) -> Anomaly:
    return Anomaly(
        id=uuid.UUID(int=index),
        timestamp=datetime(2026, 6, 5, 18, 0, tzinfo=UTC),
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


def _claim(text: str, step: int, score: float | None) -> Claim:
    return Claim(
        step_index=step,
        claim_type="concentration_elevation",
        claim_text=text,
        cited_sources=["openaq"],
        grounding_verdict="grounded" if score is not None else "unverified",
        grounding_evidence_ref=None,
        skipped_phase2=score is None,
        corroboration_score=score,
        evidence_n=1 if score is not None else 0,
        per_source_verdicts={"openaq": 1},
        partial_verifiability=False,
        low_corroboration_flag=False,
    )


async def _seed(db_session, anomaly_count: int = 5) -> list[uuid.UUID]:
    anomaly_ids: list[uuid.UUID] = []
    for index in range(1, anomaly_count + 1):
        anomaly = _anomaly(index)
        anomaly_ids.append(anomaly.id)
        db_session.add(anomaly)
        for model_index, model in enumerate(MODELS):
            db_session.add(
                Explanation(
                    anomaly_id=anomaly.id,
                    model_name=model,
                    model_version="v1",
                    reasoning_steps_json={"steps": []},
                    final_narrative="narrative",
                    stated_confidence=0.7,
                    claims=[
                        _claim(
                            f"claim {index} {model} 0",
                            1,
                            0.5 if (index + model_index) % 2 == 0 else -0.5,
                        ),
                        _claim(f"claim {index} {model} 1", 2, None),
                    ],
                )
            )
    await db_session.commit()
    return anomaly_ids


@pytest.mark.asyncio
async def test_payload_shape_matches_the_analysis_contract(db_session) -> None:
    anomaly_ids = await _seed(db_session)

    payload = await build_payload(db_session, [str(i) for i in anomaly_ids])

    assert payload["schema_version"] == 1
    assert sorted(payload["fixture"]["models"]) == sorted(MODELS)
    assert len(payload["claims"]) == 5 * 3 * 2
    assert len(payload["explanations"]) == 15
    assert payload["error_cells"] == []
    assert payload["labels"] == []


@pytest.mark.asyncio
async def test_exported_payload_is_accepted_by_build_analysis_manifest(
    db_session,
) -> None:
    anomaly_ids = await _seed(db_session)
    for anomaly_id in anomaly_ids:
        db_session.add(
            ExpertLabel(
                anomaly_id=anomaly_id,
                labeler="mason",
                true_cause=None,
                claim_validations_json=[],
            )
        )
    await db_session.commit()

    payload = await build_payload(db_session, [str(i) for i in anomaly_ids])
    # Give every claim a label so the overlap is non-empty.
    payload["labels"] = [
        {"labeler": labeler, "claim_id": claim["claim_id"], "verdict": "valid"}
        for claim in payload["claims"]
        for labeler in ("mason", "bracco")
    ]

    manifest = build_analysis_manifest(payload, [], thresholds=FAST)

    assert manifest["schema_version"] == 1
    assert manifest["negative_controls"]["majority_label"]["predicted_verdict"] == (
        "valid"
    )


@pytest.mark.asyncio
async def test_blank_marks_export_as_null_verdicts_not_unsure(db_session) -> None:
    anomaly_ids = await _seed(db_session, anomaly_count=1)
    claim_ids = [str(claim["claim_id"]) for claim in
                 (await build_payload(db_session, [str(anomaly_ids[0])]))["claims"]]
    db_session.add(
        ExpertLabel(
            anomaly_id=anomaly_ids[0],
            labeler="bracco",
            true_cause=None,
            claim_validations_json=[
                {
                    "claim_id": claim_ids[0],
                    "verdict": None,
                    "note": None,
                    "presentation_index": 1,
                }
            ],
        )
    )
    await db_session.commit()

    payload = await build_payload(db_session, [str(anomaly_ids[0])])

    assert payload["labels"] == [
        {"labeler": "bracco", "claim_id": claim_ids[0], "verdict": None}
    ]


@pytest.mark.asyncio
async def test_missing_explanation_cells_become_error_cells(db_session) -> None:
    anomaly_ids = await _seed(db_session, anomaly_count=2)
    extra = _anomaly(99)
    db_session.add(extra)
    await db_session.commit()

    payload = await build_payload(
        db_session, [str(anomaly_ids[0]), str(anomaly_ids[1]), str(extra.id)]
    )

    assert sorted(cell["model"] for cell in payload["error_cells"]) == sorted(MODELS)
    assert {cell["anomaly_id"] for cell in payload["error_cells"]} == {str(extra.id)}


@pytest.mark.asyncio
async def test_anomalies_outside_the_fixture_are_excluded(db_session) -> None:
    anomaly_ids = await _seed(db_session, anomaly_count=3)

    payload = await build_payload(db_session, [str(anomaly_ids[0])])

    assert {claim["anomaly_id"] for claim in payload["claims"]} == {
        str(anomaly_ids[0])
    }


@pytest.mark.asyncio
async def test_a_label_for_an_unknown_claim_is_an_error(db_session) -> None:
    anomaly_ids = await _seed(db_session, anomaly_count=1)
    db_session.add(
        ExpertLabel(
            anomaly_id=anomaly_ids[0],
            labeler="mason",
            true_cause=None,
            claim_validations_json=[
                {"claim_id": str(uuid.UUID(int=999)), "verdict": "valid"}
            ],
        )
    )
    await db_session.commit()

    with pytest.raises(AnalysisExportError, match="references unknown claim"):
        await build_payload(db_session, [str(anomaly_ids[0])])


@pytest.mark.asyncio
async def test_a_model_count_other_than_three_is_refused(db_session) -> None:
    anomaly = _anomaly(1)
    db_session.add(anomaly)
    db_session.add(
        Explanation(
            anomaly_id=anomaly.id,
            model_name="only-one",
            model_version="v1",
            reasoning_steps_json={"steps": []},
            final_narrative="narrative",
            stated_confidence=0.7,
            claims=[_claim("solo", 1, 0.5)],
        )
    )
    await db_session.commit()

    with pytest.raises(AnalysisExportError, match="exactly 3 planned models"):
        await build_payload(db_session, [str(anomaly.id)])


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({}, "nonempty array"),
        ({"anomaly_ids": []}, "nonempty array"),
        ({"anomaly_ids": [""]}, "nonempty string"),
        ({"anomaly_ids": ["a", "a"]}, "duplicate"),
    ],
)
def test_malformed_fixture_is_rejected(
    tmp_path: Path, body: dict[str, object], message: str
) -> None:
    fixture = tmp_path / "eval.json"
    fixture.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(AnalysisExportError, match=message):
        _frozen_anomaly_ids(fixture)


def test_payload_write_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    payload = {"schema_version": 1, "b": [2, 1], "a": "x"}

    write_payload(first, payload)
    write_payload(second, payload)

    assert first.read_bytes() == second.read_bytes()
