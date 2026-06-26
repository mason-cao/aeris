"""Leave-one-out corroboration ablation (2026-06-24 peer-review audit, rec #2).

The corroboration score is a label-free eval proxy; this module measures how
much each *error-independent channel* actually contributes to it. The claims a
model produced are held fixed — no LLM re-runs — and each grounded claim is
re-scored under source/channel exclusion. Comparing a drop condition against the
``full`` baseline gives that channel's marginal contribution.

Two granularities, both reported:
- ``drop-source:<source>`` — withhold one source. Within a redundant channel
  (OpenAQ/TCEQ/EPA-AQS, or GFS/OpenWeather) this typically moves nothing, which
  is the point: it quantifies the redundancy the channel grouping assumes.
- ``drop-channel:<channel>`` — withhold every source in a channel (only emitted
  for channels with >= 2 present sources, else it duplicates the source drop).
  This is the genuine independence test: removing a whole channel should be the
  thing that lowers the multi-channel corroboration rate.

The headline metric is ``n_multi_channel`` — claims that keep >= 2 independent
channels (``evidence_n >= 2``). The audit's failure mode was the proxy resolving
on one source (``evidence_n = 1``); this is the experiment that exposes it.

CLI: ``python -m app.eval.ablation --anomaly-set fixtures/eval50.json``
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Claim, EnrichmentRecord, Explanation
from app.eval.harness import load_anomaly_set
from app.llm.corroboration import ScoredClaim, channel_of, score_claim

logger = logging.getLogger(__name__)


# --- pure scoring core -----------------------------------------------------


def exclude_sources(summary: Mapping, excluded: Iterable[str]) -> dict:
    """Copy ``summary`` with ``excluded`` sources removed from ``["sources"]``.

    The scorers read each source via ``summary["sources"].get(source)``, so a
    removed source resolves to SILENT and drops out of its channel's net sign —
    exactly as if it had never reported. The original is never mutated; nested
    metric dicts are shared (they are read-only here).
    """
    drop = set(excluded)
    out = dict(summary)
    out["sources"] = {
        name: block
        for name, block in summary.get("sources", {}).items()
        if name not in drop
    }
    return out


def rescore(
    claim_text: str, summary: Mapping, excluded: Iterable[str] = ()
) -> ScoredClaim:
    """Re-run the corroboration scorer on one claim with sources withheld."""
    return score_claim(claim_text, exclude_sources(summary, excluded))


@dataclass(frozen=True)
class Condition:
    """One ablation condition: a label and the sources it withholds."""

    label: str
    excluded: frozenset[str]


def build_conditions(present_sources: Iterable[str]) -> list[Condition]:
    """The ``full`` baseline, one source drop each, and per multi-member channel.

    A channel with a single present source gets no ``drop-channel`` condition —
    it would be identical to that source's ``drop-source`` condition.
    """
    present = set(present_sources)
    conditions = [Condition(label="full", excluded=frozenset())]
    for source in sorted(present):
        conditions.append(
            Condition(label=f"drop-source:{source}", excluded=frozenset({source}))
        )
    channels: dict[str, set[str]] = {}
    for source in present:
        channels.setdefault(channel_of(source), set()).add(source)
    for channel in sorted(channels):
        members = channels[channel]
        if len(members) >= 2:
            conditions.append(
                Condition(
                    label=f"drop-channel:{channel}", excluded=frozenset(members)
                )
            )
    return conditions


@dataclass(frozen=True)
class Outcome:
    """Aggregate corroboration metrics for one condition over a set of claims."""

    label: str
    n_claims: int
    n_verified: int          # evidence_n >= 1 (some channel had a verdict)
    n_multi_channel: int     # evidence_n >= 2 (independently corroborated)
    mean_evidence_n: float   # over all scored claims
    mean_score: float | None  # over verified claims only


def summarize(scored: Iterable[ScoredClaim], label: str) -> Outcome:
    """Collapse re-scored claims into the reported metrics for one condition."""
    results = [s.result for s in scored]
    n_claims = len(results)
    verified = [r for r in results if not r.unverified]
    multi = [r for r in verified if r.evidence_n >= 2]
    mean_evidence_n = (
        sum(r.evidence_n for r in results) / n_claims if n_claims else 0.0
    )
    scores = [r.corroboration_score for r in verified if r.corroboration_score is not None]
    mean_score = sum(scores) / len(scores) if scores else None
    return Outcome(
        label=label,
        n_claims=n_claims,
        n_verified=len(verified),
        n_multi_channel=len(multi),
        mean_evidence_n=mean_evidence_n,
        mean_score=mean_score,
    )


@dataclass(frozen=True)
class ClaimContext:
    """One grounded claim plus the model and the evidence it is scored against."""

    model: str
    claim_text: str
    summary: Mapping = field(hash=False)


def run_conditions(
    contexts: Sequence[ClaimContext], conditions: Sequence[Condition]
) -> dict[str, dict[str, Outcome]]:
    """Score every claim under every condition, grouped ``{model: {label: Outcome}}``.

    Per-model because the interesting question is whether one model's
    corroboration leans on a channel that another's does not.
    """
    by_model: dict[str, list[ClaimContext]] = {}
    for ctx in contexts:
        by_model.setdefault(ctx.model, []).append(ctx)

    results: dict[str, dict[str, Outcome]] = {}
    for model, model_contexts in by_model.items():
        per_label: dict[str, Outcome] = {}
        for condition in conditions:
            scored = [
                rescore(c.claim_text, c.summary, condition.excluded)
                for c in model_contexts
            ]
            per_label[condition.label] = summarize(scored, condition.label)
        results[model] = per_label
    return results


# --- DB driver -------------------------------------------------------------


async def _latest_summary(
    session: AsyncSession, anomaly_id: uuid.UUID
) -> Mapping | None:
    """The most recent enrichment summary for an anomaly (matches explain.py)."""
    return (
        await session.execute(
            select(EnrichmentRecord.cross_source_summary_json)
            .where(EnrichmentRecord.anomaly_id == anomaly_id)
            .order_by(EnrichmentRecord.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def load_claim_contexts(
    session: AsyncSession, anomaly_ids: Sequence[uuid.UUID]
) -> list[ClaimContext]:
    """Load Phase-2-scored claims and the evidence they were scored against.

    Only grounded claims (``skipped_phase2 is False``) — fabricated claims never
    feed the corroboration signal, so they are out of scope for its ablation. The
    summary is the anomaly's latest EnrichmentRecord, the same evidence the
    original score saw.
    """
    if not anomaly_ids:
        return []
    rows = (
        await session.execute(
            select(
                Explanation.anomaly_id,
                Explanation.model_name,
                Claim.claim_text,
            )
            .join(Claim, Claim.explanation_id == Explanation.id)
            .where(Explanation.anomaly_id.in_(anomaly_ids))
            .where(Claim.skipped_phase2.is_(False))
        )
    ).all()

    summaries: dict[uuid.UUID, Mapping | None] = {}
    contexts: list[ClaimContext] = []
    for anomaly_id, model_name, claim_text in rows:
        if anomaly_id not in summaries:
            summaries[anomaly_id] = await _latest_summary(session, anomaly_id)
        summary = summaries[anomaly_id]
        if summary is None:
            logger.warning(
                "anomaly %s has scored claims but no enrichment record; skipping",
                anomaly_id,
            )
            continue
        contexts.append(
            ClaimContext(model=model_name, claim_text=claim_text, summary=summary)
        )
    return contexts


async def run_ablation(
    session: AsyncSession, anomaly_ids: Sequence[uuid.UUID]
) -> dict[str, dict[str, Outcome]]:
    """End-to-end: load grounded claims, build conditions, score them all."""
    contexts = await load_claim_contexts(session, anomaly_ids)
    if not contexts:
        return {}
    present: set[str] = set()
    for ctx in contexts:
        present |= set(ctx.summary.get("sources", {}))
    conditions = build_conditions(present)
    return run_conditions(contexts, conditions)


# --- CLI -------------------------------------------------------------------


def _format_ablation(results: Mapping[str, Mapping[str, Outcome]]) -> str:
    if not results:
        return "no grounded claims in the given anomaly set"
    lines: list[str] = []
    header = (
        f"{'condition':<28} {'claims':>6} {'verif':>6} {'multi':>6} "
        f"{'multi%':>7} {'mean_n':>7} {'mean_s':>7} {'Δmulti':>7} {'Δmean_n':>8}"
    )
    for model in sorted(results):
        per_label = results[model]
        full = per_label.get("full")
        lines.append(f"== {model} ==")
        lines.append(header)
        for label, o in per_label.items():
            multi_pct = 100 * o.n_multi_channel / o.n_claims if o.n_claims else 0.0
            d_multi = o.n_multi_channel - full.n_multi_channel if full else 0
            d_mean = o.mean_evidence_n - full.mean_evidence_n if full else 0.0
            score = f"{o.mean_score:.2f}" if o.mean_score is not None else "n/a"
            lines.append(
                f"{label:<28} {o.n_claims:>6} {o.n_verified:>6} "
                f"{o.n_multi_channel:>6} {multi_pct:>6.1f}% {o.mean_evidence_n:>7.2f} "
                f"{score:>7} {d_multi:>+7} {d_mean:>+8.2f}"
            )
        lines.append("")
    return "\n".join(lines).rstrip()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.ablation",
        description=(
            "Leave-one-out corroboration ablation over the frozen eval set: "
            "re-score each grounded claim with sources/channels withheld and "
            "report each channel's marginal contribution to the proxy."
        ),
    )
    parser.add_argument(
        "--anomaly-set",
        required=True,
        help="JSON fixture: array of anomaly UUIDs, or {anomaly_ids: [...]}",
    )
    return parser.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    from app.db.session import async_session, engine_lifecycle

    async with engine_lifecycle():
        args = _parse_args(argv)
        anomaly_ids = load_anomaly_set(args.anomaly_set)
        async with async_session() as session:
            results = await run_ablation(session, anomaly_ids)
    print(_format_ablation(results))
    return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
