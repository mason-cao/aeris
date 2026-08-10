"""Double-entry transcription of a returned PDF packet into ExpertLabel rows.

R6 item 5 requires that marks on a returned packet reach the database through a
declared verification step rather than a single typing pass. The guide promises
the labeler exactly that: two independent passes on separate days, compared
against each other, with any disagreement settled against their PDF before
anything is stored.

This module is that step. It refuses to write when the two passes disagree, it
records a blank mark as *missing* rather than coercing it to ``unsure``, and it
writes a provenance sidecar carrying the returned PDF's checksum so the stored
labels can always be traced back to the artifact they came from.

CLI: ``python -m app.eval.transcribe --pass-one one.json --pass-two two.json
--pdf returned.pdf --sidecar out.json``
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Anomaly, ExpertLabel
from app.eval.label_cli import collect_claim_groups, presentation_order

CHUNK_BYTES = 1024 * 1024
MARK_VERDICTS: Mapping[str, str] = {
    "V": "valid",
    "I": "invalid",
    "U": "unsure",
}


class TranscriptionError(ValueError):
    """Raised when a transcription pass or a reconciliation is not admissible."""


@dataclass(frozen=True)
class TranscriptionPass:
    """One independent typing of a returned packet."""

    anomaly_id: str
    labeler: str
    pass_number: int
    returned_pdf_sha256: str
    marks: dict[int, str | None]
    notes: dict[int, str]
    true_cause: str | None


def sha256_file(path: Path) -> str:
    """Checksum the returned artifact so stored labels stay traceable."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TranscriptionError(f"{field_name} must be a nonempty string")
    return value


def _mark_number(raw_key: object, field_name: str) -> int:
    """Printed claim numbers arrive as JSON object keys, so always as text."""
    if isinstance(raw_key, bool) or not isinstance(raw_key, str | int):
        raise TranscriptionError(f"{field_name} key must be a claim number")
    try:
        number = int(raw_key)
    except ValueError as exc:
        raise TranscriptionError(
            f"{field_name} key {raw_key!r} is not a claim number"
        ) from exc
    if number < 1:
        raise TranscriptionError(f"{field_name} key {number} must be positive")
    return number


def _parse_marks(raw_marks: object) -> dict[int, str | None]:
    if not isinstance(raw_marks, dict) or not raw_marks:
        raise TranscriptionError("marks must be a nonempty object")
    marks: dict[int, str | None] = {}
    for raw_key, raw_value in raw_marks.items():
        number = _mark_number(raw_key, "marks")
        if number in marks:
            raise TranscriptionError(f"duplicate mark for claim {number}")
        if raw_value is None or raw_value == "":
            marks[number] = None
            continue
        if not isinstance(raw_value, str):
            raise TranscriptionError(
                f"mark for claim {number} must be V, I, U, or blank"
            )
        letter = raw_value.strip().upper()
        if letter not in MARK_VERDICTS:
            raise TranscriptionError(
                f"mark for claim {number} must be V, I, U, or blank, not {raw_value!r}"
            )
        marks[number] = MARK_VERDICTS[letter]
    return marks


def _parse_notes(raw_notes: object) -> dict[int, str]:
    if raw_notes is None:
        return {}
    if not isinstance(raw_notes, dict):
        raise TranscriptionError("notes must be an object")
    notes: dict[int, str] = {}
    for raw_key, raw_value in raw_notes.items():
        number = _mark_number(raw_key, "notes")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise TranscriptionError(f"note for claim {number} must be a nonempty string")
        notes[number] = raw_value
    return notes


def parse_pass(payload: object) -> TranscriptionPass:
    """Validate one transcription pass."""
    if not isinstance(payload, dict):
        raise TranscriptionError("transcription pass root must be an object")
    if payload.get("schema_version") != 1:
        raise TranscriptionError("transcription schema_version must equal 1")

    pass_number = payload.get("pass_number")
    if pass_number not in (1, 2) or isinstance(pass_number, bool):
        raise TranscriptionError("pass_number must be 1 or 2")

    anomaly_id = _required_string(payload.get("anomaly_id"), "anomaly_id")
    try:
        uuid.UUID(anomaly_id)
    except ValueError as exc:
        raise TranscriptionError(f"anomaly_id {anomaly_id!r} is not a UUID") from exc

    marks = _parse_marks(payload.get("marks"))
    notes = _parse_notes(payload.get("notes"))
    unknown_notes = sorted(set(notes) - set(marks))
    if unknown_notes:
        raise TranscriptionError(
            "notes reference claims with no mark: "
            + ", ".join(str(number) for number in unknown_notes)
        )

    true_cause = payload.get("true_cause")
    if true_cause is not None and not isinstance(true_cause, str):
        raise TranscriptionError("true_cause must be a string or null")

    return TranscriptionPass(
        anomaly_id=anomaly_id,
        labeler=_required_string(payload.get("labeler"), "labeler"),
        pass_number=int(pass_number),
        returned_pdf_sha256=_required_string(
            payload.get("returned_pdf_sha256"), "returned_pdf_sha256"
        ),
        marks=marks,
        notes=notes,
        true_cause=true_cause.strip() or None if true_cause is not None else None,
    )


def reconcile(first: TranscriptionPass, second: TranscriptionPass) -> dict[str, object]:
    """Compare two passes and describe every disagreement.

    Returns a report rather than raising, so a mismatch can be shown to the
    operator in full instead of one field at a time.
    """
    if first.pass_number == second.pass_number:
        raise TranscriptionError("the two passes must be numbered 1 and 2")

    header_mismatches: list[str] = []
    for field_name, left, right in (
        ("anomaly_id", first.anomaly_id, second.anomaly_id),
        ("labeler", first.labeler, second.labeler),
        (
            "returned_pdf_sha256",
            first.returned_pdf_sha256,
            second.returned_pdf_sha256,
        ),
    ):
        if left != right:
            header_mismatches.append(f"{field_name}: {left!r} vs {right!r}")

    only_first = sorted(set(first.marks) - set(second.marks))
    only_second = sorted(set(second.marks) - set(first.marks))
    verdict_mismatches = [
        {
            "claim_number": number,
            "pass_one": first.marks[number],
            "pass_two": second.marks[number],
        }
        for number in sorted(set(first.marks) & set(second.marks))
        if first.marks[number] != second.marks[number]
    ]
    note_mismatches = [
        {
            "claim_number": number,
            "pass_one": first.notes.get(number),
            "pass_two": second.notes.get(number),
        }
        for number in sorted(set(first.notes) | set(second.notes))
        if first.notes.get(number) != second.notes.get(number)
    ]
    true_cause_mismatch = first.true_cause != second.true_cause

    agreed = not (
        header_mismatches
        or only_first
        or only_second
        or verdict_mismatches
        or note_mismatches
        or true_cause_mismatch
    )
    return {
        "agreed": agreed,
        "header_mismatches": header_mismatches,
        "claims_only_in_pass_one": only_first,
        "claims_only_in_pass_two": only_second,
        "verdict_mismatches": verdict_mismatches,
        "note_mismatches": note_mismatches,
        "true_cause_mismatch": true_cause_mismatch,
        "marked_claim_count": len(first.marks),
        "blank_claim_count": sum(
            1 for verdict in first.marks.values() if verdict is None
        ),
    }


def build_validations(
    ordered_groups: Sequence[object],
    marks: Mapping[int, str | None],
    notes: Mapping[int, str],
) -> list[dict[str, object]]:
    """Fan the printed claim numbers back out onto claim IDs.

    Every printed number must carry an explicit mark, blank included. An absent
    number means a line was skipped while typing, which is a transcription
    defect and not a labeler decision.
    """
    expected = set(range(1, len(ordered_groups) + 1))
    supplied = set(marks)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing:
        raise TranscriptionError(
            "no mark recorded for claim numbers: "
            + ", ".join(str(number) for number in missing)
        )
    if extra:
        raise TranscriptionError(
            "marks reference claim numbers not in the packet: "
            + ", ".join(str(number) for number in extra)
        )

    validations: list[dict[str, object]] = []
    for index, group in enumerate(ordered_groups, start=1):
        verdict = marks[index]
        note = notes.get(index)
        validations.extend(
            {
                "claim_id": str(claim_id),
                "verdict": verdict,
                "note": note,
                "presentation_index": index,
            }
            for claim_id in group.claim_ids  # type: ignore[attr-defined]
        )
    return validations


async def transcribe_packet(
    session: AsyncSession,
    first: TranscriptionPass,
    second: TranscriptionPass,
    *,
    returned_pdf_sha256: str,
) -> dict[str, object]:
    """Reconcile two passes and persist the agreed labels, or refuse."""
    report = reconcile(first, second)
    if not report["agreed"]:
        raise TranscriptionError(
            "the two transcription passes disagree; settle every difference "
            "against the returned PDF and retype, nothing was written: "
            + json.dumps(report, sort_keys=True)
        )
    if first.returned_pdf_sha256 != returned_pdf_sha256:
        raise TranscriptionError(
            f"returned PDF SHA-256 mismatch: {returned_pdf_sha256} != "
            f"{first.returned_pdf_sha256}"
        )

    anomaly_id = uuid.UUID(first.anomaly_id)
    anomaly = await session.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise TranscriptionError(f"no anomaly with id {anomaly_id}")

    existing = (
        await session.execute(
            select(ExpertLabel.id)
            .where(ExpertLabel.anomaly_id == anomaly_id)
            .where(ExpertLabel.labeler == first.labeler)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise TranscriptionError(
            f"anomaly {anomaly_id} already labeled by {first.labeler!r}; "
            "delete the expert_labels row to re-ingest"
        )

    groups = presentation_order(
        await collect_claim_groups(session, anomaly_id),
        anomaly_id,
        first.labeler,
    )
    if not groups:
        raise TranscriptionError(f"anomaly {anomaly_id} has no claims to label")

    validations = build_validations(groups, first.marks, first.notes)
    label = ExpertLabel(
        anomaly_id=anomaly_id,
        labeler=first.labeler,
        true_cause=first.true_cause,
        claim_validations_json=validations,
    )
    session.add(label)
    await session.commit()

    return {
        "schema_version": 1,
        "anomaly_id": first.anomaly_id,
        "labeler": first.labeler,
        "returned_pdf_sha256": returned_pdf_sha256,
        "unique_claim_count": len(groups),
        "claim_verdict_count": len(validations),
        "missing_claim_count": report["blank_claim_count"],
        "reconciliation": report,
    }


def write_sidecar(path: Path, sidecar: Mapping[str, object]) -> None:
    """Write the provenance record atomically."""
    payload = json.dumps(sidecar, indent=2, sort_keys=True) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _load_pass(path: Path) -> TranscriptionPass:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise TranscriptionError(f"cannot read pass {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise TranscriptionError(f"invalid JSON in pass {path}: {exc}") from exc
    return parse_pass(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.transcribe",
        description=(
            "Ingest a returned packet through two independent transcription "
            "passes; refuses to write unless they agree."
        ),
    )
    parser.add_argument("--pass-one", required=True, type=Path)
    parser.add_argument("--pass-two", required=True, type=Path)
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="the annotated PDF exactly as the labeler returned it",
    )
    parser.add_argument(
        "--sidecar",
        required=True,
        type=Path,
        help="provenance record written on success",
    )
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


async def _amain(argv: Sequence[str] | None = None) -> int:
    from app.db.session import async_session, engine_lifecycle

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        first = _load_pass(args.pass_one)
        second = _load_pass(args.pass_two)
        if not args.pdf.exists():
            raise TranscriptionError(f"returned PDF does not exist: {args.pdf}")
        pdf_sha256 = sha256_file(args.pdf)
    except (OSError, TranscriptionError) as exc:
        _argument_error(parser, str(exc))

    async with engine_lifecycle():
        async with async_session() as session:
            try:
                sidecar = await transcribe_packet(
                    session, first, second, returned_pdf_sha256=pdf_sha256
                )
            except TranscriptionError as exc:
                _argument_error(parser, str(exc))
    write_sidecar(args.sidecar, sidecar)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the double-entry transcription CLI."""
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
