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

# Published per-1M-token prices (USD): (prompt_rate, completion_rate). Fill from
# each provider's current pricing page before the freeze to turn on the $est
# column; empty by default so no fabricated number is ever printed. Local models
# (Ollama) are free and intentionally absent.
USD_PER_MTOK: dict[str, tuple[float, float]] = {}


@dataclass
class ModelSweepSummary:
    """Outcome counts plus cumulative spend for one model's pass."""

    completed: int = 0
    skipped: int = 0
    parse_failures: int = 0
    errors: list[tuple[uuid.UUID, str]] = field(default_factory=list)
    # Cumulative across cells this run actually paid for (a successful generate),
    # so the freeze sweep's cost/latency is watchable as it runs.
    total_latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


def _estimate_cost_usd(model: str, summary: ModelSweepSummary) -> float | None:
    """USD estimate from configured rates, or None when the model is unpriced."""
    rate = USD_PER_MTOK.get(model)
    if rate is None:
        return None
    prompt_rate, completion_rate = rate
    return (
        summary.prompt_tokens * prompt_rate + summary.completion_tokens * completion_rate
    ) / 1_000_000


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
    # Order-preserving dedup: a duplicate id would otherwise sweep the same cell
    # twice. An empty set is almost certainly a mistake worth surfacing.
    ids = list(dict.fromkeys(uuid.UUID(item) for item in data))
    if not ids:
        logger.warning("anomaly set %s is empty", path)
    return ids


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
        # Build the client lazily, on the first cell that actually needs it: a
        # fully-resumed model skips every cell and must not pay to construct a
        # (cloud) client it never calls.
        client: LLMClient | None = None
        try:
            for anomaly_id in anomaly_ids:
                if await _cell_exists(session, anomaly_id, model):
                    summary.skipped += 1
                    continue
                if client is None:
                    client = client_factory(model)
                try:
                    explanation = await generate_explanation(
                        session, anomaly_id, client
                    )
                    # Count spend as soon as the call returns: the tokens/time
                    # were paid even if a racing run wins the persist below.
                    summary.total_latency_ms += explanation.latency_ms or 0.0
                    summary.prompt_tokens += explanation.prompt_tokens or 0
                    summary.completion_tokens += explanation.completion_tokens or 0
                    # persist returns False when the cell already exists (a
                    # concurrent run raced us); that is a skip, not a completion.
                    if await persist_explanation(session, explanation):
                        summary.completed += 1
                    else:
                        summary.skipped += 1
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
            if client is not None:
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
        f"{'model':<26} {'done':>5} {'skip':>5} {'parse-fail':>10} {'error':>6} "
        f"{'latency':>9} {'prompt-tok':>11} {'compl-tok':>10} {'$est':>8}"
    ]
    for model, s in summaries.items():
        cost = _estimate_cost_usd(model, s)
        cost_str = f"${cost:,.2f}" if cost is not None else "n/a"
        lines.append(
            f"{model:<26} {s.completed:>5} {s.skipped:>5} "
            f"{s.parse_failures:>10} {len(s.errors):>6} "
            f"{s.total_latency_ms / 1000:>8.1f}s {s.prompt_tokens:>11,} "
            f"{s.completion_tokens:>10,} {cost_str:>8}"
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
