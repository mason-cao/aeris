"""Generate and persist an LLM explanation for one anomaly.

Pipeline per (anomaly, model): persisted EnrichmentRecord -> rendered prompt
context -> 4-step reasoning chain -> claim extraction -> Phase 1 grounding
(validate.py, mandatory before storage) -> Phase 2 corroboration on grounded
claims only -> Explanation + Claim rows.

CLI: ``python -m app.llm.explain --anomaly-id=<uuid> [--model=llama3:8b]``
"""

from __future__ import annotations

import argparse
import asyncio
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from statistics import fmean

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Anomaly, Claim, EnrichmentRecord, Explanation
from app.llm.client_base import LLMClient
from app.llm.corroboration import (
    PARTIALLY_VERIFIABLE_TYPES,
    classify_claim,
    low_corroboration_flag,
    quantitative_exclusion_reason,
    score_claim,
)
from app.llm.gemini_client import GeminiClient
from app.llm.gpt_client import GPTClient
from app.llm.ollama_client import OllamaClient
from app.llm.parser import ClaimDraft, extract_claim_drafts
from app.llm.reasoning_chain import run_reasoning_chain
from app.llm.validate import GROUNDED, GroundingResult, ground_claim_drafts


def _with_unit(value: object, unit: str | None) -> str:
    if value is None:
        return "n/a"
    return f"{value} {unit}" if unit else f"{value}"


# Time-bucketed pooled means rendered per metric so the model can reason
# about temporal patterns, and per-entity means so it can reason about
# spatial uniformity. Without these lines the prompt carried only
# range/mean/nearest aggregates while the Phase 2 scorers judged trend and
# spatial-CV claim types against the full series — claims the model had no
# data to make.
# 6 h, not 3 h: at 3 h the bucket lines dominated a ~24k-char rendered context
# (measured on live June data), pushing the step prompts past what llama3 8B's
# 8192-token window holds with generation headroom. 6 h halves the dominant
# lines while keeping 12 in-window points per metric — enough resolution for
# the trend and diurnal claims the scorers verify.
_SERIES_BUCKET_H = 6
_MAX_ENTITY_MEANS = 8

_ENTITY_TERMS: dict[str, tuple[str, str | None]] = {
    "asos": ("stations", "station means"),
    "epa_aqs": ("monitors", "monitor means"),
    "noaa_gfs": ("grid cells", "grid-cell means"),
    "openaq": ("sensors", "sensor means"),
    "openweather": ("query points", "query-point means"),
    "purpleair": ("sensors", "sensor means"),
    "sentinel5p": ("granules", None),
    "tceq": ("monitors", "monitor means"),
}


def _metric_points(data: Mapping) -> list[tuple[datetime, float]]:
    """All entities' (timestamp, value) pairs, UTC-coerced and time-sorted."""
    points: list[tuple[datetime, float]] = []
    for entity in data.get("entities", []):
        for iso, value in entity.get("series", []):
            try:
                ts = datetime.fromisoformat(iso)
                v = float(value)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            points.append((ts, v))
    return sorted(points)


def _render_bucket_means(data: Mapping) -> str | None:
    """'6h means: 06-04 00Z 12.3, 06Z 14.1, ...' or None when too sparse."""
    points = _metric_points(data)
    if not points:
        return None
    buckets: dict[datetime, list[float]] = {}
    for ts, v in points:
        floored = ts.replace(
            hour=ts.hour - ts.hour % _SERIES_BUCKET_H,
            minute=0,
            second=0,
            microsecond=0,
        )
        buckets.setdefault(floored, []).append(v)
    if len(buckets) < 2:
        return None
    parts: list[str] = []
    day: object = None
    for start in sorted(buckets):
        label = f"{start:%H}Z"
        if start.date() != day:
            day = start.date()
            label = f"{start:%m-%d} {label}"
        parts.append(f"{label} {fmean(buckets[start]):.4g}")
    return f"{_SERIES_BUCKET_H}h means: " + ", ".join(parts)


def _entity_terms(source: str) -> tuple[str, str | None]:
    return _ENTITY_TERMS.get(source, ("entities", "entity means"))


def _render_entity_means(data: Mapping, label: str | None) -> str | None:
    """Render per-entity means when the source has meaningful spatial entities."""
    if label is None:
        return None
    entities = data.get("entities", [])
    if len(entities) < 2:
        return None
    rendered: list[str] = []
    for entity in entities[:_MAX_ENTITY_MEANS]:
        values = [float(v) for _, v in entity.get("series", [])]
        if not values:
            continue
        rendered.append(
            f"{entity.get('entity_id')} ({entity.get('distance_km')} km) "
            f"{fmean(values):.4g}"
        )
    if len(rendered) < 2:
        return None
    overflow = len(entities) - _MAX_ENTITY_MEANS
    if overflow > 0:
        rendered.append(f"+{overflow} more")
    return f"{label}: " + ", ".join(rendered)


def render_anomaly_text(anomaly: Anomaly) -> str:
    """The ANOMALY block of the prompt: what was detected, where, how unusual."""
    parts = [
        f"{anomaly.source} {anomaly.metric} anomaly: value {anomaly.value}",
    ]
    if anomaly.expected_value is not None:
        parts.append(f"expected {anomaly.expected_value}")
    if anomaly.z_score is not None:
        parts.append(f"z-score {anomaly.z_score}")
    parts.append(f"severity {anomaly.severity}")
    head = ", ".join(parts)
    return (
        f"{head}, at {anomaly.timestamp.isoformat()} "
        f"({anomaly.lat:.3f}, {anomaly.lon:.3f}), detected by "
        f"{', '.join(anomaly.methods_triggered)}."
    )


def render_enrichment_text(summary: Mapping) -> str:
    """The DATA CONTEXT block of the prompt, from the enrichment summary.

    Also the Phase 1 grounding context: ``check_grounding`` verifies claims
    against exactly this text, so every number a model may legitimately cite
    (range, mean, nearest-in-time, bucketed series means, per-entity means)
    has to be spelled out here. The labeling CLI shows this same text, so
    whatever the model can see, the labeler can audit.
    """
    window = summary.get("window", {})
    lines = [
        f"Context window {window.get('start')} to {window.get('end')}, "
        f"radius {window.get('spatial_radius_km')} km."
    ]
    for source in sorted(summary.get("sources", {})):
        block = summary["sources"][source]
        for metric in sorted(block.get("metrics", {})):
            data = block["metrics"][metric]
            entity_label, means_label = _entity_terms(source)
            unit = data.get("unit")
            value_range = data.get("value_range", {})
            nearest = data.get("nearest_in_time", {})
            lines.append(
                f"- {source} {metric}: {data.get('n_entities')} {entity_label}, "
                f"{data.get('n_points')} points, "
                f"range {_with_unit(value_range.get('min'), unit)} to "
                f"{_with_unit(value_range.get('max'), unit)}, "
                f"mean {_with_unit(value_range.get('mean'), unit)}, "
                f"nearest to anomaly {_with_unit(nearest.get('v'), unit)} "
                f"at {nearest.get('t')} ({nearest.get('dt_minutes')} min away, "
                f"{nearest.get('distance_km')} km from anomaly)"
            )
            for extra in (
                _render_bucket_means(data),
                _render_entity_means(data, means_label),
            ):
                if extra:
                    lines.append(f"  {extra}")
    missing = sorted(
        src for src, present in summary.get("coverage", {}).items() if not present
    )
    if missing:
        lines.append(f"No data in window from: {', '.join(missing)}.")
    return "\n".join(lines)


def _claim_row(
    draft: ClaimDraft,
    grounding: GroundingResult,
    summary: Mapping,
) -> Claim:
    """One Claim row: Phase 1 verdict always, Phase 2 only for grounded claims.

    Ungrounded claims still get classified — which claim types a model
    fabricates is itself a result — but never scored (memo: fabricated claims
    must not feed the corroboration signal).
    """
    if grounding.verdict == GROUNDED:
        scored = score_claim(draft.claim_text, summary)
        return Claim(
            step_index=draft.step_index,
            claim_type=scored.claim_type.value,
            matched_types=[t.value for t in scored.matched_types],
            claim_text=draft.claim_text,
            cited_sources=draft.cited_sources,
            citation_outcome=grounding.citation_outcome,
            grounding_verdict=grounding.verdict,
            grounding_evidence_ref=grounding.evidence_ref,
            causal=grounding.causal,
            skipped_phase2=False,
            corroboration_score=scored.result.corroboration_score,
            evidence_n=scored.result.evidence_n,
            per_source_verdicts=scored.result.per_source_verdicts,
            per_channel_verdicts=scored.result.per_channel_verdicts,
            partial_verifiability=scored.partial_verifiability,
            quantitative_exclusion_reason=scored.quantitative_exclusion_reason,
            low_corroboration_flag=low_corroboration_flag(
                scored.result.corroboration_score,
                evidence_n=scored.result.evidence_n,
            ),
        )
    matched = classify_claim(draft.claim_text)
    primary = matched[0]
    return Claim(
        step_index=draft.step_index,
        claim_type=primary.value,
        matched_types=[t.value for t in matched],
        claim_text=draft.claim_text,
        cited_sources=draft.cited_sources,
        citation_outcome=grounding.citation_outcome,
        grounding_verdict=grounding.verdict,
        grounding_evidence_ref=None,
        causal=grounding.causal,
        skipped_phase2=True,
        corroboration_score=None,
        evidence_n=0,
        per_source_verdicts=None,
        per_channel_verdicts=None,
        partial_verifiability=primary in PARTIALLY_VERIFIABLE_TYPES,
        quantitative_exclusion_reason=quantitative_exclusion_reason(
            draft.claim_text,
            primary,
        ),
        low_corroboration_flag=False,
    )


async def generate_explanation(
    session: AsyncSession,
    anomaly_id: uuid.UUID,
    client: LLMClient,
) -> Explanation:
    """Run the full pipeline for one anomaly. Does not persist the result.

    Requires a persisted EnrichmentRecord: the explanation, the Phase 1
    grounding context, and the Phase 2 scorers must all see the same frozen
    evidence, not a re-derived one.
    """
    anomaly = await session.get(Anomaly, anomaly_id)
    if anomaly is None:
        raise ValueError(f"no anomaly with id {anomaly_id}")
    record = (
        await session.execute(
            select(EnrichmentRecord)
            .where(EnrichmentRecord.anomaly_id == anomaly_id)
            .order_by(EnrichmentRecord.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if record is None:
        raise ValueError(
            f"anomaly {anomaly_id} has no enrichment record; "
            "run python -m app.detection.enrichment first"
        )

    summary = record.cross_source_summary_json
    anomaly_text = render_anomaly_text(anomaly)
    enrichment_text = render_enrichment_text(summary)
    chain = await run_reasoning_chain(
        client,
        anomaly_text=anomaly_text,
        enrichment_text=enrichment_text,
    )
    drafts = extract_claim_drafts([step.response for step in chain.steps])
    # Ground against everything the model saw: the ANOMALY block carries
    # numbers (value, expected, z-score) a faithful step-1 restatement cites
    # that appear nowhere in the enrichment lines — grounding against the
    # enrichment text alone scored faithfulness as hallucination.
    groundings = ground_claim_drafts(drafts, f"{anomaly_text}\n{enrichment_text}")

    return Explanation(
        anomaly_id=anomaly.id,
        model_name=client.model_name,
        model_version=client.model_version,
        reasoning_steps_json={
            "steps": [
                {
                    "step": step.step.value,
                    "response": step.response.model_dump(),
                    "latency_ms": step.generation.latency_ms,
                }
                for step in chain.steps
            ]
        },
        final_narrative=chain.final_narrative,
        stated_confidence=chain.stated_confidence,
        latency_ms=chain.total_latency_ms,
        prompt_tokens=chain.total_prompt_tokens,
        completion_tokens=chain.total_completion_tokens,
        claims=[
            _claim_row(draft, grounding, summary)
            for draft, grounding in zip(drafts, groundings, strict=True)
        ],
    )


async def persist_explanation(
    session: AsyncSession,
    explanation: Explanation,
) -> bool:
    """Insert an Explanation unless the (anomaly, model) pair already has one.

    Insert-only idempotency mirroring ``persist_enrichment``; re-explaining a
    changed anomaly means deleting its existing ``explanations`` row first.
    Returns ``True`` when a row was inserted.
    """
    existing = (
        await session.execute(
            select(Explanation.id)
            .where(Explanation.anomaly_id == explanation.anomaly_id)
            .where(Explanation.model_name == explanation.model_name)
            .limit(1)
        )
    ).scalar_one_or_none()
    if existing is not None:
        return False
    session.add(explanation)
    await session.commit()
    return True


def make_client(model: str) -> LLMClient:
    """Pick the provider from the model name; anything unrecognized is Ollama."""
    if model.startswith("gpt"):
        return GPTClient(model=model)
    if model.startswith("gemini"):
        return GeminiClient(model=model)
    return OllamaClient(model=model)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.llm.explain",
        description=(
            "Generate an LLM explanation for one anomaly and persist it with "
            "Phase 1 + Phase 2 claim scores. One explanation per (anomaly, "
            "model), idempotently."
        ),
    )
    parser.add_argument(
        "--anomaly-id",
        required=True,
        help="UUID of the anomaly to explain",
    )
    parser.add_argument(
        "--model",
        default="llama3:8b",
        help=(
            "model to explain with; gpt-* and gemini-* route to the cloud "
            "clients, anything else to local Ollama (default: llama3:8b)"
        ),
    )
    return parser.parse_args(argv)


def _format_explanation(explanation: Explanation, persisted: bool) -> str:
    lines = [
        f"Model:      {explanation.model_name}",
        f"Confidence: {explanation.stated_confidence}",
        f"Latency:    {explanation.latency_ms} ms",
        f"Persisted:  {'yes' if persisted else 'no (anomaly+model already explained)'}",
        "",
        explanation.final_narrative,
        "",
        f"{'step':<6} {'type':<26} {'citation':<12} {'grounding':<12} {'phase2':<8} "
        f"{'score':>6} {'n':>3}",
    ]
    for claim in explanation.claims:
        score = "-" if claim.corroboration_score is None else f"{claim.corroboration_score:.2f}"
        phase2 = "skip" if claim.skipped_phase2 else "run"
        lines.append(
            f"{claim.step_index:<6} {claim.claim_type:<26} "
            f"{(claim.citation_outcome or 'legacy-null'):<12} "
            f"{claim.grounding_verdict:<12} {phase2:<8} {score:>6} "
            f"{claim.evidence_n:>3}"
        )
    return "\n".join(lines)


async def _amain(argv: list[str] | None = None) -> int:
    # Imported lazily so library-mode use (and tests on a SQLite engine)
    # don't pay for spinning up the production asyncpg engine.
    from app.db.session import async_session, engine_lifecycle

    async with engine_lifecycle():
        args = _parse_args(argv)
        client = make_client(args.model)
        try:
            async with async_session() as session:
                explanation = await generate_explanation(
                    session, uuid.UUID(args.anomaly_id), client
                )
                persisted = await persist_explanation(session, explanation)
        finally:
            await client.close()
        print(_format_explanation(explanation, persisted))
        return 0


def main() -> None:
    raise SystemExit(asyncio.run(_amain()))


if __name__ == "__main__":
    main()
