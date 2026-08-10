"""Export the B16 analysis payload from the database.

``phase_analysis`` is a pure function over a JSON payload, which is what makes
it auditable, but nothing built that payload — so the analysis plan was
executable and unreachable at the same time. This module closes that gap.

Missing and unsure stay strictly distinct: a claim nobody labeled is simply
absent from ``labels``, and a claim marked blank on a returned packet carries a
null verdict. Neither is ever coerced into ``unsure``.

CLI: ``python -m app.eval.analysis_export --anomaly-set fixtures/eval50.json
--output analysis-input.json``
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Claim, ExpertLabel, Explanation


class AnalysisExportError(ValueError):
    """Raised when the exported payload would not be admissible."""


def _frozen_anomaly_ids(anomaly_set: Path) -> list[str]:
    try:
        fixture = json.loads(anomaly_set.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AnalysisExportError(f"cannot read fixture {anomaly_set}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise AnalysisExportError(
            f"invalid JSON in fixture {anomaly_set}: {exc}"
        ) from exc
    if not isinstance(fixture, dict):
        raise AnalysisExportError("fixture root must be an object")
    anomaly_ids = fixture.get("anomaly_ids")
    if not isinstance(anomaly_ids, list) or not anomaly_ids:
        raise AnalysisExportError("fixture anomaly_ids must be a nonempty array")
    validated: list[str] = []
    for rank, anomaly_id in enumerate(anomaly_ids, start=1):
        if not isinstance(anomaly_id, str) or not anomaly_id.strip():
            raise AnalysisExportError(
                f"fixture anomaly ID at rank {rank} must be a nonempty string"
            )
        validated.append(anomaly_id)
    if len(set(validated)) != len(validated):
        raise AnalysisExportError("fixture anomaly_ids contain a duplicate")
    return validated


async def build_payload(
    session: AsyncSession,
    anomaly_ids: Sequence[str],
) -> dict[str, object]:
    """Assemble the payload ``build_analysis_manifest`` consumes."""
    frozen = list(anomaly_ids)
    frozen_set = set(frozen)

    rows = (
        await session.execute(
            select(
                Claim.id,
                Claim.claim_text,
                Claim.claim_type,
                Claim.grounding_verdict,
                Claim.corroboration_score,
                Claim.evidence_n,
                Explanation.anomaly_id,
                Explanation.model_name,
            ).join(Explanation, Claim.explanation_id == Explanation.id)
        )
    ).all()

    claims: list[dict[str, object]] = []
    claim_ids: set[str] = set()
    for row in rows:
        anomaly_id = str(row.anomaly_id)
        if anomaly_id not in frozen_set:
            continue
        claim_id = str(row.id)
        claim_ids.add(claim_id)
        claims.append(
            {
                "claim_id": claim_id,
                "anomaly_id": anomaly_id,
                "model": row.model_name,
                "claim_text": row.claim_text,
                "claim_type": row.claim_type,
                "grounding_verdict": row.grounding_verdict,
                "corroboration_score": row.corroboration_score,
                "evidence_n": row.evidence_n,
            }
        )

    explanation_rows = (
        await session.execute(select(Explanation.anomaly_id, Explanation.model_name))
    ).all()
    explanations = sorted(
        {
            (str(row.anomaly_id), str(row.model_name))
            for row in explanation_rows
            if str(row.anomaly_id) in frozen_set
        }
    )
    models = sorted({model for _, model in explanations})
    if len(models) != 3:
        raise AnalysisExportError(
            f"B16 requires exactly 3 planned models; found {len(models)}: "
            + ", ".join(models)
        )

    label_rows = (
        await session.execute(
            select(ExpertLabel.anomaly_id, ExpertLabel.labeler, ExpertLabel.claim_validations_json)
        )
    ).all()
    labels: list[dict[str, object]] = []
    seen_labels: set[tuple[str, str]] = set()
    for row in label_rows:
        if str(row.anomaly_id) not in frozen_set:
            continue
        for validation in row.claim_validations_json or []:
            claim_id = str(validation.get("claim_id"))
            if claim_id not in claim_ids:
                raise AnalysisExportError(
                    f"label from {row.labeler!r} references unknown claim {claim_id}"
                )
            key = (str(row.labeler), claim_id)
            if key in seen_labels:
                raise AnalysisExportError(
                    f"duplicate label record: {row.labeler}/{claim_id}"
                )
            seen_labels.add(key)
            labels.append(
                {
                    "labeler": str(row.labeler),
                    "claim_id": claim_id,
                    "verdict": validation.get("verdict"),
                }
            )

    present = set(explanations)
    error_cells = [
        {"anomaly_id": anomaly_id, "model": model}
        for anomaly_id in frozen
        for model in models
        if (anomaly_id, model) not in present
    ]

    return {
        "schema_version": 1,
        "fixture": {"anomaly_ids": frozen, "models": models},
        "claims": sorted(
            claims,
            key=lambda claim: (
                str(claim["anomaly_id"]),
                str(claim["model"]),
                str(claim["claim_id"]),
            ),
        ),
        "labels": sorted(
            labels,
            key=lambda label: (str(label["labeler"]), str(label["claim_id"])),
        ),
        "explanations": [
            {"anomaly_id": anomaly_id, "model": model}
            for anomaly_id, model in explanations
        ],
        "error_cells": error_cells,
    }


def write_payload(path: Path, payload: Mapping[str, object]) -> None:
    """Write the payload atomically and byte-deterministically."""
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.analysis_export",
        description="Export the B16 analysis payload for phase_analysis.",
    )
    parser.add_argument("--anomaly-set", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _argument_error(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


async def _amain(argv: Sequence[str] | None = None) -> int:
    from app.db.session import async_session, engine_lifecycle

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        anomaly_ids = _frozen_anomaly_ids(args.anomaly_set)
    except AnalysisExportError as exc:
        _argument_error(parser, str(exc))

    async with engine_lifecycle():
        async with async_session() as session:
            try:
                payload = await build_payload(session, anomaly_ids)
            except AnalysisExportError as exc:
                _argument_error(parser, str(exc))
    write_payload(args.output, payload)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the analysis-payload export CLI."""
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    raise SystemExit(main())
