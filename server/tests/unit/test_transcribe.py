"""Double-entry transcription of a returned packet into ExpertLabel rows.

A blank mark must reach the database as missing, never as unsure, and two
passes that disagree must write nothing at all — those are the two promises the
labeling guide makes to the labeler.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.models import Anomaly, Claim, ExpertLabel, Explanation
from app.eval.label_cli import collect_claim_groups, presentation_order
from app.eval.transcribe import (
    TranscriptionError,
    build_validations,
    parse_pass,
    reconcile,
    sha256_file,
    transcribe_packet,
)

ANOMALY_ID = uuid.UUID(int=1)
ANOMALY_TS = datetime(2026, 6, 5, 18, 0, tzinfo=UTC)
PDF_SHA = "a" * 64
LABELER = "bracco"


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


def _explanation(model: str, claims: list[Claim]) -> Explanation:
    return Explanation(
        anomaly_id=ANOMALY_ID,
        model_name=model,
        model_version="v1",
        reasoning_steps_json={"steps": []},
        final_narrative="narrative",
        stated_confidence=0.7,
        claims=claims,
    )


async def _seed(db_session) -> None:
    db_session.add_all(
        [
            _anomaly(),
            _explanation(
                "model-a", [_claim("shared claim", 1), _claim("only in a", 2)]
            ),
            _explanation("model-b", [_claim("shared claim", 1)]),
        ]
    )
    await db_session.commit()


def _pass(
    marks: dict[str, str | None],
    *,
    pass_number: int = 1,
    notes: dict[str, str] | None = None,
    true_cause: str | None = "an industrial plume",
    labeler: str = LABELER,
    pdf_sha256: str = PDF_SHA,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "anomaly_id": str(ANOMALY_ID),
        "labeler": labeler,
        "pass_number": pass_number,
        "returned_pdf_sha256": pdf_sha256,
        "marks": marks,
        "notes": notes or {},
        "true_cause": true_cause,
    }


# --- parsing ---


def test_marks_are_normalised_and_blanks_become_missing() -> None:
    parsed = parse_pass(_pass({"1": "v", "2": " I ", "3": "U", "4": None, "5": ""}))

    assert parsed.marks == {
        1: "valid",
        2: "invalid",
        3: "unsure",
        4: None,
        5: None,
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"schema_version": 2}, "schema_version must equal 1"),
        (_pass({"1": "X"}), "must be V, I, U, or blank"),
        (_pass({"0": "V"}), "must be positive"),
        (_pass({"nope": "V"}), "not a claim number"),
        (_pass({}), "nonempty object"),
        (_pass({"1": "V"}, pass_number=3), "pass_number must be 1 or 2"),
        (_pass({"1": "V"}, labeler=" "), "labeler must be a nonempty string"),
        (
            _pass({"1": "V"}, notes={"2": "orphan"}),
            "notes reference claims with no mark",
        ),
    ],
)
def test_malformed_pass_is_rejected(payload: object, message: str) -> None:
    with pytest.raises(TranscriptionError, match=message):
        parse_pass(payload)


def test_non_uuid_anomaly_id_is_rejected() -> None:
    payload = _pass({"1": "V"})
    payload["anomaly_id"] = "not-a-uuid"

    with pytest.raises(TranscriptionError, match="is not a UUID"):
        parse_pass(payload)


# --- reconciliation ---


def test_identical_passes_agree() -> None:
    first = parse_pass(_pass({"1": "V", "2": None}, pass_number=1))
    second = parse_pass(_pass({"1": "V", "2": None}, pass_number=2))

    report = reconcile(first, second)

    assert report["agreed"] is True
    assert report["marked_claim_count"] == 2
    assert report["blank_claim_count"] == 1


def test_verdict_disagreement_is_reported_per_claim() -> None:
    first = parse_pass(_pass({"1": "V", "2": "I"}, pass_number=1))
    second = parse_pass(_pass({"1": "V", "2": "U"}, pass_number=2))

    report = reconcile(first, second)

    assert report["agreed"] is False
    assert report["verdict_mismatches"] == [
        {"claim_number": 2, "pass_one": "invalid", "pass_two": "unsure"}
    ]


def test_a_blank_in_one_pass_only_is_a_disagreement() -> None:
    first = parse_pass(_pass({"1": "V"}, pass_number=1))
    second = parse_pass(_pass({"1": None}, pass_number=2))

    report = reconcile(first, second)

    assert report["agreed"] is False
    assert report["verdict_mismatches"][0]["pass_two"] is None


def test_missing_and_extra_claim_numbers_are_reported() -> None:
    first = parse_pass(_pass({"1": "V", "2": "V"}, pass_number=1))
    second = parse_pass(_pass({"1": "V", "3": "V"}, pass_number=2))

    report = reconcile(first, second)

    assert report["claims_only_in_pass_one"] == [2]
    assert report["claims_only_in_pass_two"] == [3]


def test_differing_pdf_checksums_are_a_header_mismatch() -> None:
    first = parse_pass(_pass({"1": "V"}, pass_number=1))
    second = parse_pass(_pass({"1": "V"}, pass_number=2, pdf_sha256="b" * 64))

    report = reconcile(first, second)

    assert report["agreed"] is False
    assert any("returned_pdf_sha256" in item for item in report["header_mismatches"])


def test_note_and_true_cause_differences_block_agreement() -> None:
    first = parse_pass(_pass({"1": "V"}, pass_number=1, notes={"1": "because"}))
    second = parse_pass(_pass({"1": "V"}, pass_number=2))
    assert reconcile(first, second)["note_mismatches"]

    third = parse_pass(_pass({"1": "V"}, pass_number=2, true_cause="something else"))
    assert reconcile(first, third)["true_cause_mismatch"] is True


def test_two_passes_with_the_same_number_are_refused() -> None:
    first = parse_pass(_pass({"1": "V"}, pass_number=1))

    with pytest.raises(TranscriptionError, match="numbered 1 and 2"):
        reconcile(first, first)


# --- fan-out ---


class _Group:
    def __init__(self, claim_ids: tuple[str, ...]) -> None:
        self.claim_ids = claim_ids


def test_validations_fan_out_to_every_claim_id_in_a_group() -> None:
    groups = [_Group(("c1", "c2")), _Group(("c3",))]

    validations = build_validations(groups, {1: "valid", 2: None}, {1: "note"})

    assert validations == [
        {
            "claim_id": "c1",
            "verdict": "valid",
            "note": "note",
            "presentation_index": 1,
        },
        {
            "claim_id": "c2",
            "verdict": "valid",
            "note": "note",
            "presentation_index": 1,
        },
        {"claim_id": "c3", "verdict": None, "note": None, "presentation_index": 2},
    ]


def test_a_skipped_line_is_a_transcription_defect_not_a_blank() -> None:
    groups = [_Group(("c1",)), _Group(("c2",))]

    with pytest.raises(TranscriptionError, match="no mark recorded for claim numbers"):
        build_validations(groups, {1: "valid"}, {})


def test_marks_beyond_the_packet_are_refused() -> None:
    groups = [_Group(("c1",))]

    with pytest.raises(TranscriptionError, match="not in the packet"):
        build_validations(groups, {1: "valid", 2: "valid"}, {})


# --- persistence ---


@pytest.mark.asyncio
async def test_agreeing_passes_persist_missing_as_null_verdict(db_session) -> None:
    await _seed(db_session)
    groups = presentation_order(
        await collect_claim_groups(db_session, ANOMALY_ID), ANOMALY_ID, LABELER
    )
    marks = {str(index): None for index in range(1, len(groups) + 1)}
    marks["1"] = "V"

    sidecar = await transcribe_packet(
        db_session,
        parse_pass(_pass(marks, pass_number=1)),
        parse_pass(_pass(marks, pass_number=2)),
        returned_pdf_sha256=PDF_SHA,
    )

    stored = (await db_session.execute(select(ExpertLabel))).scalars().all()
    assert len(stored) == 1
    verdicts = [row["verdict"] for row in stored[0].claim_validations_json]
    assert None in verdicts
    assert "unsure" not in verdicts
    assert sidecar["returned_pdf_sha256"] == PDF_SHA
    assert sidecar["missing_claim_count"] == len(groups) - 1


@pytest.mark.asyncio
async def test_disagreeing_passes_write_nothing(db_session) -> None:
    await _seed(db_session)
    groups = presentation_order(
        await collect_claim_groups(db_session, ANOMALY_ID), ANOMALY_ID, LABELER
    )
    first_marks = {str(index): "V" for index in range(1, len(groups) + 1)}
    second_marks = dict(first_marks)
    second_marks["1"] = "I"

    with pytest.raises(TranscriptionError, match="disagree"):
        await transcribe_packet(
            db_session,
            parse_pass(_pass(first_marks, pass_number=1)),
            parse_pass(_pass(second_marks, pass_number=2)),
            returned_pdf_sha256=PDF_SHA,
        )

    assert (await db_session.execute(select(ExpertLabel))).scalars().all() == []


@pytest.mark.asyncio
async def test_pdf_checksum_must_match_the_declared_one(db_session) -> None:
    await _seed(db_session)
    groups = presentation_order(
        await collect_claim_groups(db_session, ANOMALY_ID), ANOMALY_ID, LABELER
    )
    marks = {str(index): "V" for index in range(1, len(groups) + 1)}

    with pytest.raises(TranscriptionError, match="returned PDF SHA-256 mismatch"):
        await transcribe_packet(
            db_session,
            parse_pass(_pass(marks, pass_number=1)),
            parse_pass(_pass(marks, pass_number=2)),
            returned_pdf_sha256="c" * 64,
        )

    assert (await db_session.execute(select(ExpertLabel))).scalars().all() == []


@pytest.mark.asyncio
async def test_relabeling_the_same_labeler_is_refused(db_session) -> None:
    await _seed(db_session)
    groups = presentation_order(
        await collect_claim_groups(db_session, ANOMALY_ID), ANOMALY_ID, LABELER
    )
    marks = {str(index): "V" for index in range(1, len(groups) + 1)}
    passes = (
        parse_pass(_pass(marks, pass_number=1)),
        parse_pass(_pass(marks, pass_number=2)),
    )
    await transcribe_packet(db_session, *passes, returned_pdf_sha256=PDF_SHA)

    with pytest.raises(TranscriptionError, match="already labeled"):
        await transcribe_packet(db_session, *passes, returned_pdf_sha256=PDF_SHA)


@pytest.mark.asyncio
async def test_unknown_anomaly_is_refused(db_session) -> None:
    payload = _pass({"1": "V"})
    payload["anomaly_id"] = str(uuid.UUID(int=99))
    first = parse_pass(payload)
    payload_two = dict(payload, pass_number=2)

    with pytest.raises(TranscriptionError, match="no anomaly with id"):
        await transcribe_packet(
            db_session,
            first,
            parse_pass(payload_two),
            returned_pdf_sha256=PDF_SHA,
        )


def test_sha256_file_matches_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "returned.pdf"
    target.write_bytes(b"%PDF-1.4 pretend")

    import hashlib

    assert sha256_file(target) == hashlib.sha256(target.read_bytes()).hexdigest()


def test_sidecar_round_trips_as_json(tmp_path: Path) -> None:
    from app.eval.transcribe import write_sidecar

    target = tmp_path / "sidecar.json"
    write_sidecar(target, {"anomaly_id": str(ANOMALY_ID), "agreed": True})

    assert json.loads(target.read_text(encoding="utf-8"))["agreed"] is True
