"""Leave-one-out corroboration ablation (2026-06-24 audit rec #2).

Holds each model's claims fixed and re-scores them under source/channel
exclusion — no LLM re-runs — to measure each measurement-process channel's
marginal contribution to the corroboration proxy. The headline metric is the
fraction of scored claims that keep >= 2 process groups (evidence_n>=2):
dropping a load-bearing channel must lower it, redundant within-channel drops
must not.
"""

import uuid
from datetime import UTC, datetime

import pytest

from app.db.models import Anomaly, Claim, EnrichmentRecord, Explanation
from app.eval.ablation import (
    TRIGGER_CONDITION_LABEL,
    ClaimContext,
    Condition,
    Outcome,
    build_conditions,
    exclude_sources,
    load_claim_contexts,
    rescore,
    resolve_excluded,
    run_ablation,
    run_conditions,
    summarize,
)
from app.llm.validate import GROUNDED

ANOMALY_TS = "2026-06-15T12:00:00+00:00"


def _elevated_metric(baseline: list[float], nearest: float) -> dict:
    """A metric block whose nearest value sits far above its pre-anomaly baseline."""
    series = [
        [f"2026-06-15T{h:02d}:00:00+00:00", v] for h, v in enumerate(baseline)
    ]
    return {
        "nearest_in_time": {"v": nearest},
        "entities": [{"entity_id": "e", "series": series}],
    }


def _summary(sources_metrics: dict, anomaly_ts: str = ANOMALY_TS) -> dict:
    """Minimal enrichment summary: {source: {metric: metric_block}}."""
    return {
        "anomaly": {"timestamp": anomaly_ts, "lat": 29.76, "lon": -95.37},
        "sources": {
            source: {"metrics": dict(metrics)}
            for source, metrics in sources_metrics.items()
        },
    }


# Ground (tceq) + satellite (s5p) = two process groups agreeing on a
# qualitative elevation, each against its own baseline in its own units — a
# surface-ppb threshold can never legitimately engage a mol/m^2 column.
GROUND_PLUS_SAT = _summary(
    {
        "tceq": {"no2": _elevated_metric([10.0, 11.0, 10.0, 10.5], 60.0)},
        "sentinel5p": {
            "s5p_no2_column": _elevated_metric(
                [5.0e-5, 5.5e-5, 5.2e-5, 5.1e-5], 2.1e-4
            )
        },
    }
)
# Two redundant ground sources + satellite: dropping one ground source must not
# change evidence (the channel survives); dropping both must.
TWO_GROUND_PLUS_SAT = _summary(
    {
        "openaq": {"no2": _elevated_metric([9.0, 10.0, 9.5, 10.0], 58.0)},
        "tceq": {"no2": _elevated_metric([10.0, 11.0, 10.0, 10.5], 61.0)},
        "sentinel5p": {
            "s5p_no2_column": _elevated_metric(
                [5.0e-5, 5.5e-5, 5.2e-5, 5.1e-5], 2.1e-4
            )
        },
    }
)
NO2_CLAIM = "NO2 was elevated"


class TestExcludeSources:
    def test_removes_named_source(self) -> None:
        out = exclude_sources(GROUND_PLUS_SAT, {"sentinel5p"})
        assert set(out["sources"]) == {"tceq"}

    def test_does_not_mutate_original(self) -> None:
        before = set(GROUND_PLUS_SAT["sources"])
        exclude_sources(GROUND_PLUS_SAT, {"tceq"})
        assert set(GROUND_PLUS_SAT["sources"]) == before

    def test_excluding_absent_source_is_noop(self) -> None:
        out = exclude_sources(GROUND_PLUS_SAT, {"purpleair"})
        assert set(out["sources"]) == {"tceq", "sentinel5p"}

    def test_excluding_nothing_preserves_sources(self) -> None:
        out = exclude_sources(GROUND_PLUS_SAT, set())
        assert set(out["sources"]) == {"tceq", "sentinel5p"}

    def test_preserves_non_source_keys(self) -> None:
        out = exclude_sources(GROUND_PLUS_SAT, {"tceq"})
        assert out["anomaly"] == GROUND_PLUS_SAT["anomaly"]


class TestRescore:
    def test_full_reproduces_two_channel_score(self) -> None:
        scored = rescore(NO2_CLAIM, GROUND_PLUS_SAT)
        assert scored.result.evidence_n == 2
        assert scored.result.corroboration_score == 1.0

    def test_leave_one_channel_out_drops_evidence(self) -> None:
        # Excluding the whole satellite channel leaves only ground -> n=1.
        scored = rescore(NO2_CLAIM, GROUND_PLUS_SAT, {"sentinel5p"})
        assert scored.result.evidence_n == 1

    def test_dropping_sole_ground_source_silences_its_channel(self) -> None:
        scored = rescore(NO2_CLAIM, GROUND_PLUS_SAT, {"tceq"})
        assert scored.result.per_channel_verdicts.get("ground_insitu") in (None, 0)
        assert scored.result.evidence_n == 1  # only satellite remains

    def test_within_channel_redundancy_keeps_evidence(self) -> None:
        # openaq + tceq + s5p: dropping ONE ground source must not lower n,
        # because the ground channel still carries a vote.
        full = rescore(NO2_CLAIM, TWO_GROUND_PLUS_SAT)
        drop_one = rescore(NO2_CLAIM, TWO_GROUND_PLUS_SAT, {"openaq"})
        assert full.result.evidence_n == 2
        assert drop_one.result.evidence_n == 2

    def test_dropping_whole_ground_channel_lowers_evidence(self) -> None:
        drop_channel = rescore(NO2_CLAIM, TWO_GROUND_PLUS_SAT, {"openaq", "tceq"})
        assert drop_channel.result.evidence_n == 1


class TestBuildConditions:
    def test_includes_full_baseline_first(self) -> None:
        conds = build_conditions({"tceq", "sentinel5p"})
        assert conds[0] == Condition(label="full", excluded=frozenset())

    def test_one_drop_per_present_source(self) -> None:
        conds = build_conditions({"tceq", "sentinel5p"})
        labels = {c.label for c in conds}
        assert "drop-source:tceq" in labels
        assert "drop-source:sentinel5p" in labels

    def test_multimember_channel_gets_a_channel_drop(self) -> None:
        # openaq + tceq share ground_insitu -> a drop-channel condition that
        # removes both.
        conds = build_conditions({"openaq", "tceq", "sentinel5p"})
        by_label = {c.label: c for c in conds}
        assert "drop-channel:ground_insitu" in by_label
        assert by_label["drop-channel:ground_insitu"].excluded == frozenset(
            {"openaq", "tceq"}
        )

    def test_single_member_channel_has_no_redundant_channel_drop(self) -> None:
        # satellite has one source; drop-channel would duplicate drop-source.
        conds = build_conditions({"openaq", "tceq", "sentinel5p"})
        labels = {c.label for c in conds}
        assert "drop-channel:satellite_column" not in labels

    def test_empty_sources_yields_only_full(self) -> None:
        assert build_conditions(set()) == [Condition(label="full", excluded=frozenset())]

    def test_trigger_condition_included_when_sources_present(self) -> None:
        labels = {c.label for c in build_conditions({"tceq", "sentinel5p"})}
        assert TRIGGER_CONDITION_LABEL in labels


class TestTriggerCondition:
    """The per-claim circularity check: drop the channel detection selected on."""

    def _triggered(self) -> dict:
        summary = _summary(
            {
                "tceq": {"no2": _elevated_metric([10.0, 11.0, 10.0, 10.5], 52.0)},
                "openaq": {"no2": _elevated_metric([9.0, 10.0, 9.5, 10.0], 50.0)},
                "sentinel5p": {
                    "s5p_no2_column": _elevated_metric(
                        [5.0e-5, 5.5e-5, 5.2e-5, 5.1e-5], 2.1e-4
                    )
                },
            }
        )
        summary["anomaly"].update({"source": "tceq", "metric": "no2"})
        return summary

    def test_resolves_to_every_trigger_channel_member(self) -> None:
        condition = Condition(label=TRIGGER_CONDITION_LABEL, excluded=frozenset())
        excluded = resolve_excluded(condition, self._triggered())
        assert excluded == frozenset({"tceq", "openaq"})

    def test_resolves_to_nothing_without_anomaly_source(self) -> None:
        condition = Condition(label=TRIGGER_CONDITION_LABEL, excluded=frozenset())
        assert resolve_excluded(condition, GROUND_PLUS_SAT) == frozenset()

    def test_static_conditions_resolve_to_their_own_exclusions(self) -> None:
        condition = Condition(label="drop-source:tceq", excluded=frozenset({"tceq"}))
        assert resolve_excluded(condition, self._triggered()) == frozenset({"tceq"})

    def test_trigger_drop_removes_the_kept_contradiction(self) -> None:
        # In the full condition the trigger channel's contradiction of a
        # misstated threshold is kept (the asymmetric demotion silences only
        # its tautological support). Dropping the trigger channel removes that
        # contradiction too — the difference is exactly what this condition
        # measures.
        summary = self._triggered()
        claim = "NO2 exceeded 500 ppb"
        full = rescore(claim, summary)
        assert full.result.corroboration_score == -1.0
        condition = Condition(label=TRIGGER_CONDITION_LABEL, excluded=frozenset())
        dropped = rescore(claim, summary, resolve_excluded(condition, summary))
        assert dropped.result.unverified is True


def _scored(claim_text: str, summary: dict, excluded=()):
    return rescore(claim_text, summary, excluded)


class TestSummarize:
    def test_counts_verified_and_multi_channel(self) -> None:
        scored = [
            _scored(NO2_CLAIM, GROUND_PLUS_SAT),         # n=2 multi
            _scored(NO2_CLAIM, GROUND_PLUS_SAT, {"sentinel5p"}),  # n=1 verified
        ]
        out = summarize(scored, label="x")
        assert out.n_claims == 2
        assert out.n_verified == 2
        assert out.n_multi_channel == 1

    def test_mean_evidence_n_over_all_claims(self) -> None:
        scored = [
            _scored(NO2_CLAIM, GROUND_PLUS_SAT),                  # 2
            _scored(NO2_CLAIM, GROUND_PLUS_SAT, {"sentinel5p"}),  # 1
        ]
        out = summarize(scored, label="x")
        assert out.mean_evidence_n == pytest.approx(1.5)

    def test_mean_score_over_verified_only(self) -> None:
        out = summarize([_scored(NO2_CLAIM, GROUND_PLUS_SAT)], label="x")
        assert out.mean_score == pytest.approx(1.0)

    def test_mean_score_none_when_nothing_verified(self) -> None:
        # An unclassified claim aggregates all-silent -> unverified.
        out = summarize([_scored("the weather was nice", GROUND_PLUS_SAT)], label="x")
        assert out.n_verified == 0
        assert out.mean_score is None

    def test_empty_is_zeroed(self) -> None:
        out = summarize([], label="x")
        assert out == Outcome(
            label="x",
            n_claims=0,
            n_verified=0,
            n_multi_channel=0,
            mean_evidence_n=0.0,
            mean_score=None,
        )


class TestRunConditions:
    def test_groups_by_model_then_label(self) -> None:
        contexts = [
            ClaimContext(model="llama3:8b", claim_text=NO2_CLAIM, summary=GROUND_PLUS_SAT),
            ClaimContext(model="gpt-x", claim_text=NO2_CLAIM, summary=GROUND_PLUS_SAT),
        ]
        conds = build_conditions({"tceq", "sentinel5p"})
        results = run_conditions(contexts, conds)
        assert set(results) == {"llama3:8b", "gpt-x"}
        assert "full" in results["llama3:8b"]

    def test_dropping_load_bearing_channel_lowers_multi_channel_rate(self) -> None:
        contexts = [
            ClaimContext(model="m", claim_text=NO2_CLAIM, summary=GROUND_PLUS_SAT)
        ]
        conds = build_conditions({"tceq", "sentinel5p"})
        results = run_conditions(contexts, conds)["m"]
        assert results["full"].n_multi_channel == 1
        # Removing satellite collapses the only second channel.
        assert results["drop-source:sentinel5p"].n_multi_channel == 0

    def test_redundant_source_drop_does_not_lower_evidence(self) -> None:
        contexts = [
            ClaimContext(model="m", claim_text=NO2_CLAIM, summary=TWO_GROUND_PLUS_SAT)
        ]
        conds = build_conditions({"openaq", "tceq", "sentinel5p"})
        results = run_conditions(contexts, conds)["m"]
        assert results["full"].n_multi_channel == 1
        assert results["drop-source:openaq"].n_multi_channel == 1  # tceq covers it
        assert results["drop-channel:ground_insitu"].n_multi_channel == 0


# --- DB driver -------------------------------------------------------------


def _anomaly() -> Anomaly:
    return Anomaly(
        id=uuid.uuid4(),
        timestamp=datetime(2026, 6, 15, 12, 0, tzinfo=UTC),
        lat=29.76,
        lon=-95.37,
        metric="no2",
        source="tceq",
        value=85.0,
        expected_value=58.0,
        z_score=4.2,
        methods_triggered=["zscore"],
        severity="severe",
    )


def _claim(text: str, *, grounded: bool) -> Claim:
    return Claim(
        step_index=0,
        claim_type="concentration_elevation",
        claim_text=text,
        cited_sources=["tceq"],
        grounding_verdict=GROUNDED if grounded else "ungrounded",
        skipped_phase2=not grounded,
        corroboration_score=None,
        evidence_n=0,
        per_source_verdicts=None,
    )


async def _seed_explained(db_session, summary: dict, model: str) -> uuid.UUID:
    anomaly = _anomaly()
    db_session.add(anomaly)
    db_session.add(
        EnrichmentRecord(
            anomaly_id=anomaly.id,
            context_window_start=datetime(2026, 6, 15, 0, 0, tzinfo=UTC),
            context_window_end=datetime(2026, 6, 16, 0, 0, tzinfo=UTC),
            cross_source_summary_json=summary,
        )
    )
    db_session.add(
        Explanation(
            anomaly_id=anomaly.id,
            model_name=model,
            model_version="v1",
            reasoning_steps_json={"steps": []},
            final_narrative="n",
            claims=[
                _claim(NO2_CLAIM, grounded=True),
                _claim("a fabricated aside", grounded=False),
            ],
        )
    )
    await db_session.commit()
    return anomaly.id


@pytest.mark.asyncio
async def test_load_claim_contexts_returns_only_grounded(db_session) -> None:
    anomaly_id = await _seed_explained(db_session, GROUND_PLUS_SAT, "llama3:8b")
    contexts = await load_claim_contexts(db_session, [anomaly_id])
    assert len(contexts) == 1  # the ungrounded claim is excluded
    ctx = contexts[0]
    assert ctx.model == "llama3:8b"
    assert ctx.claim_text == NO2_CLAIM
    assert set(ctx.summary["sources"]) == {"tceq", "sentinel5p"}


@pytest.mark.asyncio
async def test_load_claim_contexts_ignores_other_anomalies(db_session) -> None:
    kept = await _seed_explained(db_session, GROUND_PLUS_SAT, "llama3:8b")
    await _seed_explained(db_session, GROUND_PLUS_SAT, "gpt-x")
    contexts = await load_claim_contexts(db_session, [kept])
    assert {c.model for c in contexts} == {"llama3:8b"}


@pytest.mark.asyncio
async def test_run_ablation_end_to_end(db_session) -> None:
    anomaly_id = await _seed_explained(db_session, GROUND_PLUS_SAT, "llama3:8b")
    results = await run_ablation(db_session, [anomaly_id])
    model = results["llama3:8b"]
    assert model["full"].n_multi_channel == 1
    # Dropping either independent channel removes the multi-channel corroboration.
    assert model["drop-source:sentinel5p"].n_multi_channel == 0
    assert model["drop-source:tceq"].n_multi_channel == 0


@pytest.mark.asyncio
async def test_run_ablation_uses_latest_enrichment_record(db_session) -> None:
    anomaly_id = await _seed_explained(db_session, GROUND_PLUS_SAT, "llama3:8b")
    # A newer record with no satellite channel must be the one scored against.
    # created_at pinned to the future so ordering is independent of the 1-second
    # CURRENT_TIMESTAMP resolution that would otherwise tie the two rows.
    db_session.add(
        EnrichmentRecord(
            anomaly_id=anomaly_id,
            context_window_start=datetime(2026, 6, 15, 1, 0, tzinfo=UTC),
            context_window_end=datetime(2026, 6, 16, 1, 0, tzinfo=UTC),
            cross_source_summary_json=_summary(
                {"tceq": {"no2": _elevated_metric([10.0, 11.0, 10.0, 10.5], 60.0)}}
            ),
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
        )
    )
    await db_session.commit()
    results = await run_ablation(db_session, [anomaly_id])
    assert results["llama3:8b"]["full"].n_multi_channel == 0  # single channel now


@pytest.mark.asyncio
async def test_run_ablation_empty_set_is_empty(db_session) -> None:
    assert await run_ablation(db_session, []) == {}
