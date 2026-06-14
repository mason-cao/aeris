"""Sweep the eval anomaly set across all comparison models.

One cell = one (anomaly, model) pair run through the full explain pipeline
and persisted as an Explanation + Claim rows. The sweep is resumable: cells
that already have an Explanation are skipped before any LLM call, so a
crashed or partial run is finished by re-running the same command. Parse
failures are counted per model — that rate is a reported result, not noise.

CLI: ``python -m app.eval.harness --anomaly-set fixtures/eval50.json``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Explanation
from app.llm.client_base import LLMClient, LLMParseError
from app.llm.explain import generate_explanation, make_client, persist_explanation
from app.llm.gemini_client import DEFAULT_MODEL as GEMINI_DEFAULT
from app.llm.gpt_client import DEFAULT_MODEL as GPT_DEFAULT

logger = logging.getLogger(__name__)

DEFAULT_MODELS: tuple[str, ...] = ("llama3:8b", GPT_DEFAULT, GEMINI_DEFAULT)


@dataclass
class ModelSweepSummary:
    """Outcome counts for one model's pass over the anomaly set."""

    completed: int = 0
    skipped: int = 0
    parse_failures: int = 0
    errors: list[tuple[uuid.UUID, str]] = field(default_factory=list)


def load_anomaly_set(path: Path | str) -> list[uuid.UUID]:
    """Read the frozen anomaly-id fixture.

    Accepts a plain JSON array of UUID strings, or an object with an
    ``anomaly_ids`` key (the frozen-set format, which carries metadata like
    ``frozen_at`` alongside).
    """
    import json

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        if "anomaly_ids" not in data:
            raise ValueError(f"{path} has no 'anomaly_ids' key")
        data = data["anomaly_ids"]
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON array or an 'anomaly_ids' object")
    return [uuid.UUID(item) for item in data]


async def _cell_exists(
    session: AsyncSession, anomaly_id: uuid.UUID, model: str
) -> bool:
    existing = (
        await session.execute(
            select(Explanation.id)
            .where(Explanation.anomaly_id == anomaly_id)
            .where(Explanation.model_name == model)
            .limit(1)
        )
    ).scalar_one_or_none()
    return existing is not None


async def run_harness(
    session: AsyncSession,
    anomaly_ids: list[uuid.UUID],
    models: list[str],
    client_factory: Callable[[str], LLMClient] = make_client,
) -> dict[str, ModelSweepSummary]:
    """Fill every missing (anomaly, model) cell; never let one cell kill the sweep."""
    summaries: dict[str, ModelSweepSummary] = {}
    for model in models:
        summary = summaries.setdefault(model, ModelSweepSummary())
        client = client_factory(model)
        try:
            for anomaly_id in anomaly_ids:
                if await _cell_exists(session, anomaly_id, model):
                    summary.skipped += 1
                    continue
                try:
                    explanation = await generate_explanation(
                        session, anomaly_id, client
                    )
                    await persist_explanation(session, explanation)
                    summary.completed += 1
                except LLMParseError:
                    await session.rollback()
                    summary.parse_failures += 1
                    logger.warning(
                        "parse failure",
                        extra={"model": model, "anomaly_id": str(anomaly_id)},
                    )
                except Exception as e:
                    await session.rollback()
                    summary.errors.append((anomaly_id, str(e)))
                    logger.exception(
                        "cell failed",
                        extra={"model": model, "anomaly_id": str(anomaly_id)},
                    )
        finally:
            await client.close()
    return summaries


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.harness",
        description=(
            "Run the explain pipeline for every (anomaly, model) cell of the "
            "eval set. Resumable: existing cells are skipped, so re-running "
            "finishes a partial sweep."
        ),
    )
    parser.add_argument(
        "--anomaly-set",
        required=True,
        help="JSON fixture: array of anomaly UUIDs, or {anomaly_ids: [...]}",
    )
    parser.add_argument(
        "--models",
        type=lambda s: s.split(","),
        default=list(DEFAULT_MODELS),
        help=f"comma-separated model names (default: {','.join(DEFAULT_MODELS)})",
    )
    return parser.parse_args(argv)


def _format_summaries(summaries: dict[str, ModelSweepSummary]) -> str:
    lines = [
        f"{'model':<26} {'done':>5} {'skip':>5} {'parse-fail':>10} {'error':>6}"
    ]
    for model, s in summaries.items():
        lines.append(
            f"{model:<26} {s.completed:>5} {s.skipped:>5} "
            f"{s.parse_failures:>10} {len(s.errors):>6}"
        )
    for model, s in summaries.items():
        for anomaly_id, message in s.errors:
            lines.append(f"  {model} {anomaly_id}: {message}")
    return "\n".join(lines)


async def _amain(argv: list[str] | None = None) -> int:
    from app.db.session import async_session, engine_lifecycle

    async with engine_lifecycle():
        args = _parse_args(argv)
        anomaly_ids = load_anomaly_set(args.anomaly_set)
        async with async_session() as session:
            summaries = await run_harness(session, anomaly_ids, models=args.models)
        print(_format_summaries(summaries))
        failed = sum(len(s.errors) + s.parse_failures for s in summaries.values())
        return 1 if failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
