"""Lossless export, verification, and restore of expert labels.

Labels cannot be regenerated, so the export has to survive a round trip with
every field intact, and the restore has to refuse rather than overwrite.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from app.db.models import Anomaly, ExpertLabel
from app.eval.label_backup import (
    LabelBackupError,
    build_export,
    load_export,
    payload_sha256,
    restore_export,
    verify_export,
    write_export,
)

CREATED = datetime(2026, 8, 11, 14, 30, tzinfo=UTC)


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


def _validations() -> list[dict]:
    return [
        {
            "claim_id": str(uuid.UUID(int=90)),
            "verdict": "valid",
            "note": "matches the monitor",
            "presentation_index": 1,
        },
        {
            "claim_id": str(uuid.UUID(int=91)),
            "verdict": None,
            "note": None,
            "presentation_index": 2,
        },
        {
            "claim_id": str(uuid.UUID(int=92)),
            "verdict": "unsure",
            "note": None,
            "presentation_index": 3,
        },
    ]


async def _seed(db_session, labelers=("mason", "bracco")) -> None:
    anomaly = _anomaly(1)
    db_session.add(anomaly)
    for labeler in labelers:
        db_session.add(
            ExpertLabel(
                anomaly_id=anomaly.id,
                labeler=labeler,
                true_cause="industrial plume" if labeler == "mason" else None,
                claim_validations_json=_validations(),
                created_at=CREATED,
            )
        )
    await db_session.commit()


@pytest.mark.asyncio
async def test_export_keeps_every_field_and_counts_missing_separately(
    db_session,
) -> None:
    await _seed(db_session)

    payload = await build_export(db_session)

    assert payload["label_count"] == 2
    assert payload["labeler_counts"] == {"bracco": 1, "mason": 1}
    # A blank mark is counted as missing, never folded into unsure.
    assert payload["claim_verdict_counts"] == {"missing": 2, "unsure": 2, "valid": 2}
    first = payload["labels"][0]
    assert set(first) == {
        "id",
        "anomaly_id",
        "labeler",
        "true_cause",
        "claim_validations_json",
        "created_at",
    }
    assert first["claim_validations_json"] == _validations()


@pytest.mark.asyncio
async def test_export_is_deterministic_and_checksummed(db_session, tmp_path: Path) -> None:
    await _seed(db_session)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    payload = await build_export(db_session)
    write_export(first_path, payload)
    write_export(second_path, await build_export(db_session))

    assert first_path.read_bytes() == second_path.read_bytes()
    assert payload["payload_sha256"] == payload_sha256(payload["labels"])


@pytest.mark.asyncio
async def test_export_round_trips_through_restore(db_session, tmp_path: Path) -> None:
    await _seed(db_session)
    path = tmp_path / "labels.json"
    write_export(path, await build_export(db_session))
    before = await build_export(db_session)

    for label in (await db_session.execute(select(ExpertLabel))).scalars().all():
        await db_session.delete(label)
    await db_session.commit()
    assert (await build_export(db_session))["label_count"] == 0

    _, labels = load_export(path)
    restored = await restore_export(db_session, labels)
    after = await build_export(db_session)

    assert restored == 2
    assert after["payload_sha256"] == before["payload_sha256"]
    assert after["labels"] == before["labels"]


@pytest.mark.asyncio
async def test_restore_refuses_to_overwrite_existing_labels(
    db_session, tmp_path: Path
) -> None:
    await _seed(db_session)
    path = tmp_path / "labels.json"
    write_export(path, await build_export(db_session))
    _, labels = load_export(path)

    with pytest.raises(LabelBackupError, match="refusing to restore over existing"):
        await restore_export(db_session, labels)

    assert (await build_export(db_session))["label_count"] == 2


@pytest.mark.asyncio
async def test_verify_reports_a_clean_match(db_session, tmp_path: Path) -> None:
    await _seed(db_session)
    path = tmp_path / "labels.json"
    write_export(path, await build_export(db_session))
    payload, _ = load_export(path)

    report = await verify_export(db_session, payload)

    assert report["matches"] is True
    assert report["only_in_export"] == []
    assert report["only_in_database"] == []
    assert report["differing_rows"] == []


@pytest.mark.asyncio
async def test_verify_detects_a_label_added_after_the_export(
    db_session, tmp_path: Path
) -> None:
    await _seed(db_session, labelers=("mason",))
    path = tmp_path / "labels.json"
    write_export(path, await build_export(db_session))
    payload, _ = load_export(path)

    db_session.add(
        ExpertLabel(
            anomaly_id=uuid.UUID(int=1),
            labeler="bracco",
            true_cause=None,
            claim_validations_json=_validations(),
            created_at=CREATED,
        )
    )
    await db_session.commit()

    report = await verify_export(db_session, payload)

    assert report["matches"] is False
    assert report["only_in_database"] == ["bracco/00000000-0000-0000-0000-000000000001"]


@pytest.mark.asyncio
async def test_verify_detects_an_edited_verdict(db_session, tmp_path: Path) -> None:
    await _seed(db_session, labelers=("mason",))
    path = tmp_path / "labels.json"
    write_export(path, await build_export(db_session))
    payload, _ = load_export(path)

    label = (await db_session.execute(select(ExpertLabel))).scalars().one()
    edited = _validations()
    edited[0] = {**edited[0], "verdict": "invalid"}
    label.claim_validations_json = edited
    await db_session.commit()

    report = await verify_export(db_session, payload)

    assert report["matches"] is False
    assert report["differing_rows"] == ["mason/00000000-0000-0000-0000-000000000001"]


def test_a_tampered_export_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    labels = [
        {
            "id": str(uuid.UUID(int=5)),
            "anomaly_id": str(uuid.UUID(int=1)),
            "labeler": "mason",
            "true_cause": None,
            "claim_validations_json": [{"claim_id": "c1", "verdict": "valid"}],
            "created_at": CREATED.isoformat(),
        }
    ]
    payload = {
        "schema_version": 1,
        "label_count": 1,
        "labeler_counts": {"mason": 1},
        "claim_verdict_counts": {"valid": 1},
        "payload_sha256": payload_sha256(labels),
        "labels": labels,
    }
    tampered = json.loads(json.dumps(payload))
    tampered["labels"][0]["claim_validations_json"][0]["verdict"] = "invalid"
    path.write_text(json.dumps(tampered), encoding="utf-8")

    with pytest.raises(LabelBackupError, match="checksum mismatch"):
        load_export(path)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda p: p.update(schema_version=2), "schema_version must equal 1"),
        (lambda p: p.update(labels="nope"), "labels must be an array"),
        (lambda p: p.update(label_count=99), "label_count"),
        (
            lambda p: p["labels"].append(dict(p["labels"][0])),
            "duplicate label for mason",
        ),
        (
            lambda p: p["labels"][0].update(anomaly_id="not-a-uuid"),
            "is not a UUID",
        ),
        (lambda p: p["labels"][0].update(labeler=" "), "must be a nonempty string"),
    ],
)
def test_malformed_export_is_rejected(tmp_path: Path, mutate, message: str) -> None:
    labels = [
        {
            "id": str(uuid.UUID(int=5)),
            "anomaly_id": str(uuid.UUID(int=1)),
            "labeler": "mason",
            "true_cause": None,
            "claim_validations_json": [],
            "created_at": CREATED.isoformat(),
        }
    ]
    payload = {
        "schema_version": 1,
        "label_count": 1,
        "labeler_counts": {"mason": 1},
        "claim_verdict_counts": {},
        "payload_sha256": payload_sha256(labels),
        "labels": labels,
    }
    mutate(payload)
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LabelBackupError, match=message):
        load_export(path)


@pytest.mark.asyncio
async def test_empty_database_exports_cleanly(db_session) -> None:
    payload = await build_export(db_session)

    assert payload["label_count"] == 0
    assert payload["labels"] == []
    assert payload["payload_sha256"] == payload_sha256([])
