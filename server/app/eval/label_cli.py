"""Interactive labeling tool for the expert evaluation set.

Loads one anomaly's context, walks the labeler through its claims one at a
time, and persists an ExpertLabel with per-claim verdicts. Two properties
the Phase 2 analysis depends on:

- **Blinding.** The labeler never sees the pipeline's own judgments
  (grounding verdict, corroboration score) or which model made a claim —
  those are exactly what the labels get correlated against. Presentation
  order is a deterministic per-(anomaly, labeler) shuffle: the raw query
  order blocks claims by model name, so labeler fatigue/anchoring would
  correlate with model identity even with the text blinded.
- **Text-level deduplication.** Identical claim texts from different models
  are asked once; the verdict fans out to every matching claim id, each
  carrying its ``presentation_index`` so order effects stay analyzable.

One ExpertLabel per (anomaly, labeler), insert-only; relabeling means
deleting the existing row first. The Mason/Bracco overlap for inter-rater
agreement comes from labeling the same anomaly under different ``--labeler``
names.

CLI: ``python -m app.eval.label_cli --anomaly-id=<uuid> --labeler=<name>``
"""

from __future__ import annotations

import argparse
import asyncio
import random
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Anomaly, Claim, EnrichmentRecord, ExpertLabel, Explanation
from app.llm.explain import render_anomaly_text, render_enrichment_text

VERDICTS = {"v": "valid", "i": "invalid", "u": "unsure"}
QUIT = "q"


@dataclass(frozen=True)
class ClaimGroup:
    """One unique claim text and every claim row that carries it."""

    claim_text: str
    claim_ids: tuple[uuid.UUID, ...]


async def collect_claim_groups(
    session: AsyncSession,
    anomaly_id: uuid.UUID,
    model: str | None = None,
) -> list[ClaimGroup]:
    """Unique claim texts for an anomaly, in (model, step) first-seen order."""
    query = (
        select(Claim.id, Claim.claim_text)
        .join(Explanation, Claim.explanation_id == Explanation.id)
        .where(Explanation.anomaly_id == anomaly_id)
        .order_by(Explanation.model_name, Claim.step_index)
    )
    if model is not None:
        query = query.where(Explanation.model_name == model)
    rows = (await session.execute(query)).all()

    by_text: dict[str, list[uuid.UUID]] = {}
    for claim_id, claim_text in rows:
        by_text.setdefault(claim_text, []).append(claim_id)
    return [
        ClaimGroup(claim_text=text, claim_ids=tuple(ids))
        for text, ids in by_text.items()
    ]


def presentation_order(
    groups: list[ClaimGroup], anomaly_id: uuid.UUID, labeler: str
) -> list[ClaimGroup]:
    """Deterministic per-(anomaly, labeler) shuffle of the claim groups.

    The query orders claims by model name, so unshuffled the alphabetically-
    first model's claims are always labeled freshest — a position/fatigue
    confound correlated with model identity even though the text is blinded.
    Seeding on (anomaly, labeler) keeps a re-run session identical for the
    same labeler while decorrelating presentation order from model.
    """
    ordered = list(groups)
    random.Random(f"{anomaly_id}:{labeler}").shuffle(ordered)
    return ordered


def _ask_verdict(
    input_fn: Callable[[str], str],
    echo: Callable[[str], None],
) -> str | None:
    """One verdict letter, re-prompting until valid; None means quit."""
    while True:
        answer = input_fn("  [v]alid / [i]nvalid / [u]nsure / [q]uit: ").strip().lower()
        if answer == QUIT:
            return None
        if answer in VERDICTS:
            return VERDICTS[answer]
        echo("  please answer v, i, u, or q")


async def run_label_session(
    session: AsyncSession,
    anomaly_id: uuid.UUID,
    labeler: str,
    *,
    model: str | None = None,
    input_fn: Callable[[str], str] = input,
    echo: Callable[[str], None] = print,
) -> ExpertLabel | None:
    """Walk the labeler through one anomaly; returns None if they quit."""
    anomaly = await session.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise ValueError(f"no anomaly with id {anomaly_id}")

    existing = (
        await session.execute(
            select(ExpertLabel.id)
            .where(ExpertLabel.anomaly_id == anomaly_id)
            .where(ExpertLabel.labeler == labeler)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ValueError(
            f"anomaly {anomaly_id} already labeled by {labeler!r}; "
            "delete the expert_labels row to relabel"
        )

    record = (
        await session.execute(
            select(EnrichmentRecord)
            .where(EnrichmentRecord.anomaly_id == anomaly_id)
            .order_by(EnrichmentRecord.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()

    groups = presentation_order(
        await collect_claim_groups(session, anomaly_id, model=model),
        anomaly_id,
        labeler,
    )
    if not groups:
        raise ValueError(
            f"anomaly {anomaly_id} has no claims to label; "
            "run python -m app.eval.harness first"
        )

    echo("=== ANOMALY ===")
    echo(render_anomaly_text(anomaly))
    if record is not None:
        echo("")
        echo("=== DATA CONTEXT ===")
        echo(render_enrichment_text(record.cross_source_summary_json))
    echo("")
    echo(f"{len(groups)} claims to label (labeler: {labeler})")

    validations: list[dict] = []
    for i, group in enumerate(groups, start=1):
        echo("")
        echo(f"[{i}/{len(groups)}] {group.claim_text}")
        verdict = _ask_verdict(input_fn, echo)
        if verdict is None:
            echo("aborted; nothing saved")
            return None
        note = input_fn("  note (enter to skip): ").strip() or None
        validations.extend(
            {
                "claim_id": str(claim_id),
                "verdict": verdict,
                "note": note,
                "presentation_index": i,
            }
            for claim_id in group.claim_ids
        )

    echo("")
    true_cause = input_fn("most likely true cause, in your words: ").strip() or None

    label = ExpertLabel(
        anomaly_id=anomaly_id,
        labeler=labeler,
        true_cause=true_cause,
        claim_validations_json=validations,
    )
    session.add(label)
    await session.commit()
    echo(f"saved: {len(validations)} claim verdicts across {len(groups)} unique claims")
    return label


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.label_cli",
        description=(
            "Label one anomaly's claims. Presents each unique claim text "
            "once, blind to model identity and pipeline scores."
        ),
    )
    parser.add_argument(
        "--anomaly-id",
        required=True,
        help="UUID of the anomaly to label",
    )
    parser.add_argument(
        "--labeler",
        required=True,
        help="who is labeling (e.g. mason, bracco) — required for IRR",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="only label claims from this model (default: all models)",
    )
    return parser.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    from app.db.session import async_session, engine_lifecycle

    async with engine_lifecycle():
        args = _parse_args(argv)
        async with async_session() as session:
            label = await run_label_session(
                session,
                uuid.UUID(args.anomaly_id),
                labeler=args.labeler,
                model=args.model,
            )
        return 0 if label is not None else 1


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
