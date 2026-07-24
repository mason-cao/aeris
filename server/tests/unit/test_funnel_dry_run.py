import copy
import json
from pathlib import Path

import pytest

from app.eval.funnel_dry_run import (
    ATOMICITY_COMPOUND,
    ATOMICITY_EXTERNAL_ANTECEDENT,
    ATOMICITY_MISSING_SUBJECT,
    ATOMICITY_OTHER_CONTEXT,
    ATOMICITY_SELF_CONTAINED,
    FunnelAuditError,
    FunnelSelectionError,
    build_atomicity_worksheet,
    build_funnel_report,
    canonical_json,
    render_markdown,
    screen_atomicity,
    select_funnel_anomalies,
    summarize_b17_silence,
    summarize_b8_rate,
    summarize_citations,
    write_iteration_reports,
)
from app.eval.harness import DEFAULT_MODELS
from app.llm.corroboration import concentration_claim_shape


ANOMALY_IDS = [f"00000000-0000-4000-8000-{index:012d}" for index in range(1, 9)]


def _detector_availability() -> dict[str, dict[str, object]]:
    return {
        detector: {"ran": True, "skip_code": None, "detail": None}
        for detector in ("isolation_forest", "stl", "zscore")
    }


def _ranked(metrics: list[str] | None = None) -> list[dict[str, object]]:
    metric_order = metrics or ["pm25", "pm25", "ozone", "no2", "ozone", "so2", "co"]
    return [
        {
            "anomaly_id": ANOMALY_IDS[index],
            "source": "openaq" if metric in {"pm25", "ozone"} else "tceq",
            "metric": metric,
            "source_entity_id": f"station-{index + 1}",
            "detector_availability": _detector_availability(),
            "enrichment_present": True,
        }
        for index, metric in enumerate(metric_order)
    ]


def _step(prompt_tokens: int = 1000) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 100,
        "attempts": 1,
    }


def _cells(selected_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "anomaly_id": anomaly_id,
            "model": model,
            "steps": [_step() for _ in range(4)],
        }
        for anomaly_id in selected_ids
        for model in DEFAULT_MODELS
    ]


def _claims(selected_ids: list[str]) -> list[dict[str, object]]:
    claims: list[dict[str, object]] = []
    for anomaly_index, anomaly_id in enumerate(selected_ids, start=1):
        for model_index, model in enumerate(DEFAULT_MODELS, start=1):
            claims.append(
                {
                    "claim_id": f"claim-{anomaly_index}-{model_index}",
                    "anomaly_id": anomaly_id,
                    "model": model,
                    "claim_text": f"PM2.5 was elevated at monitor {anomaly_index}.",
                    "claim_type": "concentration_elevation",
                    "matched_types": ["concentration_elevation"],
                    "cited_sources": ["openaq"],
                    "citation_outcome": "cited_right",
                    "citation_failure_reasons": [],
                    "grounding_verdict": "grounded",
                    "skipped_phase2": False,
                    "corroboration_score": 1.0,
                    "evidence_n": 1,
                    "corroboration_evidence_summary": (
                        "openaq: pm25 nearest=30.0 vs pre-anomaly baseline=8.0 "
                        "(+2.0 sigma=2.0; entity_id=station; baseline_n=4)"
                    ),
                    "causal": False,
                    "calm_wind_flagged": False,
                    "direction_data_present": False,
                }
            )
    return claims


def _b8_observations(selected_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "anomaly_id": anomaly_id,
            "source": "openaq",
            "metric": "pm25",
            "dt_minutes": 30.0,
        }
        for anomaly_id in selected_ids
    ]


def _calm_decisions(selected_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "anomaly_id": anomaly_id,
            "source": source,
            "window_n": 3,
            "event_speed_ms": 2.0,
            "raw_cutoff_ms": 0.5,
            "effective_cutoff_ms": 1.5,
            "guard_enabled": True,
            "calm": False,
            "direction_votable": True,
            "reason": "at_or_above_cutoff",
            "floor_status": "bracco_confirmed",
        }
        for anomaly_id in selected_ids
        for source in ("noaa_gfs", "openweather", "asos")
    ]


def _manual_decisions(claims: list[dict[str, object]]) -> dict[str, str]:
    worksheet = build_atomicity_worksheet(claims)
    return {
        str(item["decision_hash"]): ATOMICITY_SELF_CONTAINED
        for item in worksheet["items"]
    }


def _valid_inputs() -> dict[str, object]:
    ranked = _ranked()
    selection = select_funnel_anomalies(ranked)
    selected_ids = list(selection["selected_anomaly_ids"])
    claims = _claims(selected_ids)
    return {
        "ranked_anomalies": ranked,
        "cells": _cells(selected_ids),
        "claims": claims,
        "b8_observations": _b8_observations(selected_ids),
        "b8_absences": [],
        "calm_wind_decisions": _calm_decisions(selected_ids),
        "manual_atomicity": _manual_decisions(claims),
        "provenance": {
            "disposable_b19_not_official": True,
            "git_commit": "a" * 40,
            "db_copy_sha256": "b" * 64,
            "selected_anomaly_ids": selected_ids,
            "iteration": 1,
        },
    }


def test_selector_uses_first_distinct_metric_and_reports_plain_top_five() -> None:
    result = select_funnel_anomalies(_ranked())

    assert result["selected_anomaly_ids"] == [
        ANOMALY_IDS[0],
        ANOMALY_IDS[2],
        ANOMALY_IDS[3],
        ANOMALY_IDS[5],
        ANOMALY_IDS[6],
    ]
    assert [row["selection_reason"] for row in result["selected"]] == [
        "first_distinct_metric"
    ] * 5
    assert result["plain_top_five_anomaly_ids"] == ANOMALY_IDS[:5]


def test_selector_fills_by_global_rank_only_when_fewer_than_five_metrics() -> None:
    result = select_funnel_anomalies(
        _ranked(["pm25", "pm25", "ozone", "no2", "ozone", "no2"])
    )

    assert result["selected_anomaly_ids"] == [
        ANOMALY_IDS[0],
        ANOMALY_IDS[2],
        ANOMALY_IDS[3],
        ANOMALY_IDS[1],
        ANOMALY_IDS[4],
    ]
    assert [row["selection_reason"] for row in result["selected"]] == [
        "first_distinct_metric",
        "first_distinct_metric",
        "first_distinct_metric",
        "global_rank_fill",
        "global_rank_fill",
    ]


def test_selector_missing_enrichment_hard_stops_without_substitution() -> None:
    ranked = _ranked()
    ranked[2]["enrichment_present"] = False

    with pytest.raises(FunnelSelectionError, match=ANOMALY_IDS[2]):
        select_funnel_anomalies(ranked)


@pytest.mark.parametrize(
    ("text", "category"),
    [
        ("This indicates elevated ozone.", ATOMICITY_EXTERNAL_ANTECEDENT),
        ("Was elevated during the afternoon.", ATOMICITY_MISSING_SUBJECT),
        (
            "NO2 was elevated; wind speeds were low.",
            ATOMICITY_COMPOUND,
        ),
        ("As noted, NO2 was elevated.", ATOMICITY_OTHER_CONTEXT),
        ("NO2 was elevated at the downtown monitor.", ATOMICITY_SELF_CONTAINED),
    ],
)
def test_atomicity_lexical_categories_fire(text: str, category: str) -> None:
    assert screen_atomicity(text) == category


def test_atomicity_screen_uses_declared_first_match_order() -> None:
    assert (
        screen_atomicity("This indicates NO2 rose; winds were low.")
        == ATOMICITY_EXTERNAL_ANTECEDENT
    )


def test_atomicity_worksheet_is_seeded_deduplicated_and_model_blind() -> None:
    inputs = _valid_inputs()
    claims = inputs["claims"]
    assert isinstance(claims, list)

    first = build_atomicity_worksheet(claims)
    second = build_atomicity_worksheet(list(reversed(claims)))

    assert first == second
    assert len(first["items"]) == 5
    for item in first["items"]:
        assert set(item) == {"review_index", "decision_hash", "claim_text"}
        assert all("model" not in key and "anomaly" not in key for key in item)


def test_complete_manual_atomicity_is_required() -> None:
    inputs = _valid_inputs()
    manual = inputs["manual_atomicity"]
    assert isinstance(manual, dict)
    manual.pop(next(iter(manual)))

    with pytest.raises(FunnelAuditError, match="manual atomicity"):
        build_funnel_report(**inputs)


def test_valid_report_has_exact_cells_dual_units_and_go_status() -> None:
    report = build_funnel_report(**_valid_inputs())

    assert report["go_no_go"]["status"] == "go"
    assert report["counting_units"] == {
        "claim_rows": 15,
        "unique_anomaly_exact_text": 5,
    }
    assert report["cell_audit"]["completed_cells"] == 15
    assert report["selection"]["plain_top_five_anomaly_ids"] == ANOMALY_IDS[:5]
    assert report["costs"]["status"] == "available"
    assert report["costs"]["per_model"]["gpt-5.4"]["estimated_cost_usd"] == (
        pytest.approx(0.08)
    )
    assert report["costs"]["per_model"]["gemini-3.6-flash"][
        "estimated_cost_usd"
    ] == pytest.approx(0.045)
    assert report["costs"]["per_model"]["llama3:8b"][
        "estimated_cost_usd"
    ] is None
    assert report["costs"]["pricing_provenance"]["gemini-3.6-flash"][
        "billing_basis"
    ].endswith("thinking tokens")


def test_fourteen_cells_is_a_structural_hard_stop() -> None:
    inputs = _valid_inputs()
    cells = inputs["cells"]
    assert isinstance(cells, list)
    cells.pop()

    report = build_funnel_report(**inputs)

    assert report["go_no_go"]["status"] == "hard_stop"
    assert "completed cells 14 != 15" in report["go_no_go"]["hard_stops"]


def test_missing_step_usage_is_a_hard_stop() -> None:
    inputs = _valid_inputs()
    cells = inputs["cells"]
    assert isinstance(cells, list)
    cells[0]["steps"][0]["prompt_tokens"] = None

    report = build_funnel_report(**inputs)

    assert report["go_no_go"]["status"] == "hard_stop"
    assert any(
        "missing token metadata" in reason
        for reason in report["go_no_go"]["hard_stops"]
    )


def test_local_prompt_token_hard_and_review_boundaries_are_exact() -> None:
    hard_inputs = _valid_inputs()
    hard_cells = hard_inputs["cells"]
    assert isinstance(hard_cells, list)
    hard_cells[0]["steps"][0]["prompt_tokens"] = 8192
    hard = build_funnel_report(**hard_inputs)
    assert hard["go_no_go"]["status"] == "hard_stop"
    assert any(">= 8192" in reason for reason in hard["go_no_go"]["hard_stops"])

    at_review_inputs = _valid_inputs()
    at_review_cells = at_review_inputs["cells"]
    assert isinstance(at_review_cells, list)
    at_review_cells[0]["steps"][0]["prompt_tokens"] = 7680
    at_review = build_funnel_report(**at_review_inputs)
    assert not any(
        "> 7680" in reason for reason in at_review["go_no_go"]["review_items"]
    )

    past_review_inputs = _valid_inputs()
    past_review_cells = past_review_inputs["cells"]
    assert isinstance(past_review_cells, list)
    past_review_cells[0]["steps"][0]["prompt_tokens"] = 7681
    past_review = build_funnel_report(**past_review_inputs)
    assert any(
        "> 7680" in reason for reason in past_review["go_no_go"]["review_items"]
    )


def test_citation_variants_count_three_reasons_multiple_and_omission() -> None:
    base = {
        "model": DEFAULT_MODELS[0],
        "cited_sources": ["openaq"],
        "citation_outcome": "cited_right",
        "citation_failure_reasons": [],
    }
    claims = [
        {**base, "cited_sources": [], "citation_outcome": "uncited"},
        {
            **base,
            "cited_sources": [""],
            "citation_outcome": "cited_wrong",
            "citation_failure_reasons": [
                {"index": 0, "citation": "", "reason": "blank-only"}
            ],
        },
        {
            **base,
            "cited_sources": ["mystery"],
            "citation_outcome": "cited_wrong",
            "citation_failure_reasons": [
                {
                    "index": 0,
                    "citation": "mystery",
                    "reason": "unrecognized-source",
                }
            ],
        },
        {
            **base,
            "cited_sources": ["sentinel5p"],
            "citation_outcome": "cited_wrong",
            "citation_failure_reasons": [
                {
                    "index": 0,
                    "citation": "sentinel5p",
                    "reason": "recognized-but-absent-from-context",
                }
            ],
        },
        {
            **base,
            "cited_sources": ["openaq", "", "sentinel5p"],
            "citation_outcome": "cited_wrong",
            "citation_failure_reasons": [
                {"index": 1, "citation": "", "reason": "blank-only"},
                {
                    "index": 2,
                    "citation": "sentinel5p",
                    "reason": "recognized-but-absent-from-context",
                },
            ],
        },
    ]

    summary = summarize_citations(claims, DEFAULT_MODELS)
    local = summary[DEFAULT_MODELS[0]]
    assert local["all_claim_rows"] == 5
    assert local["claims_with_nonblank_citation"] == 3
    assert local["all_blank_citation_claims"] == 1
    assert local["omission_count"] == 1
    assert local["reason_counts_all_claims"] == {
        "blank-only": 2,
        "recognized-but-absent-from-context": 2,
        "unrecognized-source": 1,
    }
    assert local["reason_counts_rate_denominator"] == {
        "blank-only": 1,
        "recognized-but-absent-from-context": 2,
        "unrecognized-source": 1,
    }
    assert local["multiple_reasons_claim_count"] == 1
    assert [record["reason"] for record in local["failure_records"]] == [
        "blank-only",
        "unrecognized-source",
        "recognized-but-absent-from-context",
        "blank-only",
        "recognized-but-absent-from-context",
    ]


def test_malformed_citation_outcome_fails_loudly() -> None:
    inputs = _valid_inputs()
    claims = inputs["claims"]
    assert isinstance(claims, list)
    claims[0]["citation_failure_reasons"] = [
        {"index": 0, "citation": "wrong", "reason": "unrecognized-source"}
    ]

    with pytest.raises(FunnelAuditError, match="malformed outcome"):
        build_funnel_report(**inputs)


@pytest.mark.parametrize(
    ("sources", "outcome", "reasons"),
    [
        (
            [""],
            "cited_right",
            [{"index": 0, "citation": "", "reason": "blank-only"}],
        ),
        (
            ["openaq", ""],
            "cited_wrong",
            [{"index": 1, "citation": "", "reason": "blank-only"}],
        ),
    ],
)
def test_citation_outcome_must_match_nonblank_failure_state(
    sources: list[str],
    outcome: str,
    reasons: list[dict[str, object]],
) -> None:
    with pytest.raises(FunnelAuditError, match="malformed outcome"):
        summarize_citations(
            [
                {
                    "model": DEFAULT_MODELS[0],
                    "cited_sources": sources,
                    "citation_outcome": outcome,
                    "citation_failure_reasons": reasons,
                }
            ],
            DEFAULT_MODELS,
        )


def test_b8_exact_twenty_percent_passes_and_above_stops() -> None:
    exact = summarize_b8_rate("openaq", stale_count=20, denominator=100)
    above = summarize_b8_rate("openaq", stale_count=2001, denominator=10_000)

    assert exact["fraction_silenced"] == 0.2
    assert exact["hourly_hard_stop"] is False
    assert above["fraction_silenced"] == 0.2001
    assert above["hourly_hard_stop"] is True


def test_planted_b8_real_location_fixture_stops_only_above_twenty_percent() -> None:
    exact_inputs = _valid_inputs()
    selected_ids = exact_inputs["provenance"]["selected_anomaly_ids"]
    exact_inputs["b8_observations"] = [
        {
            "anomaly_id": selected_ids[index % 5],
            "source": "openaq",
            "metric": f"metric-{index}",
            "dt_minutes": 91.0 if index < 20 else 90.0,
        }
        for index in range(100)
    ]
    exact = build_funnel_report(**exact_inputs)
    assert exact["go_no_go"]["status"] == "go"

    above_inputs = _valid_inputs()
    selected_ids = above_inputs["provenance"]["selected_anomaly_ids"]
    above_inputs["b8_observations"] = [
        {
            "anomaly_id": selected_ids[index % 5],
            "source": "openaq",
            "metric": f"metric-{index}",
            "dt_minutes": 91.0 if index < 2001 else 90.0,
        }
        for index in range(10_000)
    ]
    above = build_funnel_report(**above_inputs)
    assert above["go_no_go"]["status"] == "hard_stop"
    assert (
        "hourly B8 real-location silence rate > 20% for openaq"
        in above["go_no_go"]["hard_stops"]
    )


def test_no_headline_evidence_zero_grounded_and_all_unclassified_stop() -> None:
    no_evidence_inputs = _valid_inputs()
    claims = no_evidence_inputs["claims"]
    assert isinstance(claims, list)
    for claim in claims:
        claim["evidence_n"] = 0
        claim["corroboration_score"] = None
    no_evidence = build_funnel_report(**no_evidence_inputs)
    assert any(
        "headline claims have evidence_n=0" in reason
        for reason in no_evidence["go_no_go"]["hard_stops"]
    )

    unclassified_inputs = _valid_inputs()
    unclassified_claims = unclassified_inputs["claims"]
    assert isinstance(unclassified_claims, list)
    for claim in unclassified_claims:
        if claim["model"] == DEFAULT_MODELS[0]:
            claim["claim_type"] = "unclassified"
            claim["matched_types"] = ["unclassified"]
    unclassified = build_funnel_report(**unclassified_inputs)
    assert any(
        f"100% unclassified claims for model {DEFAULT_MODELS[0]}" == reason
        for reason in unclassified["go_no_go"]["hard_stops"]
    )

    zero_grounded_inputs = _valid_inputs()
    zero_grounded_claims = zero_grounded_inputs["claims"]
    assert isinstance(zero_grounded_claims, list)
    for claim in zero_grounded_claims:
        if claim["model"] == DEFAULT_MODELS[1]:
            claim["grounding_verdict"] = "unverified"
            claim["skipped_phase2"] = True
            claim["corroboration_score"] = None
            claim["evidence_n"] = 0
            claim["corroboration_evidence_summary"] = None
    zero_grounded = build_funnel_report(**zero_grounded_inputs)
    assert any(
        f"zero grounded claims for model {DEFAULT_MODELS[1]}" == reason
        for reason in zero_grounded["go_no_go"]["hard_stops"]
    )

    no_headline_inputs = _valid_inputs()
    no_headline_claims = no_headline_inputs["claims"]
    assert isinstance(no_headline_claims, list)
    for claim in no_headline_claims:
        claim["claim_type"] = "temporal_pattern"
        claim["matched_types"] = ["temporal_pattern"]
    no_headline = build_funnel_report(**no_headline_inputs)
    assert "all headline claims have evidence_n=0" in no_headline["go_no_go"][
        "hard_stops"
    ]


def test_concentration_shape_and_b17_silence_split_exclude_numeric_claims() -> None:
    assert concentration_claim_shape("NO2 was elevated downtown") == "qualitative"
    assert concentration_claim_shape("NO2 exceeded 20 ppb downtown") == "threshold"
    assert concentration_claim_shape("NO2 was 20 ppb downtown") == "point"

    common = {
        "anomaly_id": ANOMALY_IDS[0],
        "model": DEFAULT_MODELS[0],
        "claim_type": "concentration_elevation",
    }
    claims = [
        {
            **common,
            "claim_text": "NO2 was elevated downtown",
            "corroboration_evidence_summary": (
                "tceq: no station-matched pre-anomaly baseline "
                "(baseline_n=2; reason=matched baseline n < 3)"
            ),
        },
        {
            **common,
            "claim_text": "Ozone was elevated downtown",
            "corroboration_evidence_summary": "openaq: no ozone in window",
        },
        {
            **common,
            "claim_text": "PM2.5 was elevated downtown",
            "corroboration_evidence_summary": (
                "openaq: no pm25 in window; purpleair: no station-matched "
                "pre-anomaly baseline (baseline_n=1; "
                "reason=matched baseline n < 3)"
            ),
        },
        {
            **common,
            "claim_text": "NO2 exceeded 20 ppb downtown",
            "corroboration_evidence_summary": "tceq: no no2 in window",
        },
    ]

    summary = summarize_b17_silence(claims)
    assert summary["claim_rows"] == {
        "qualitative_concentration": 3,
        "matched_baseline_n_lt_3": 2,
        "no_data_in_window": 2,
        "both": 1,
    }
    assert summary["unique_anomaly_exact_text"] == summary["claim_rows"]


def test_calm_wind_rates_use_claim_rows_and_exact_text_units() -> None:
    inputs = _valid_inputs()
    claims = inputs["claims"]
    assert isinstance(claims, list)
    for index, claim in enumerate(claims[:3]):
        claim["claim_type"] = "transport_direction"
        claim["matched_types"] = ["transport_direction"]
        claim["claim_text"] = "Southerly winds transported pollution northward."
        claim["direction_data_present"] = index < 2
        claim["calm_wind_flagged"] = index == 0
    inputs["manual_atomicity"] = _manual_decisions(claims)

    report = build_funnel_report(**inputs)
    calm = report["tables"]["calm_wind"]
    assert calm["claim_rows"]["all"]["numerator"] == 1
    assert calm["claim_rows"]["all"]["denominator"] == 15
    assert calm["claim_rows"]["eligible_direction"]["numerator"] == 1
    assert calm["claim_rows"]["eligible_direction"]["denominator"] == 2
    assert calm["unique_anomaly_exact_text"]["all"]["numerator"] == 1
    assert calm["unique_anomaly_exact_text"]["all"]["denominator"] == 5
    assert calm["unique_anomaly_exact_text"]["eligible_direction"] == {
        "numerator": 1,
        "denominator": 1,
        "fraction": 1.0,
    }


def test_speed_based_calm_flag_remains_visible_without_direction_data() -> None:
    inputs = _valid_inputs()
    claims = inputs["claims"]
    assert isinstance(claims, list)
    claim = claims[0]
    claim["claim_type"] = "transport_direction"
    claim["matched_types"] = ["transport_direction"]
    claim["claim_text"] = "Southerly winds transported pollution northward."
    claim["direction_data_present"] = False
    claim["calm_wind_flagged"] = True
    inputs["manual_atomicity"] = _manual_decisions(claims)

    report = build_funnel_report(**inputs)

    calm = report["tables"]["calm_wind"]
    assert calm["claim_rows"]["all"]["numerator"] == 1
    assert calm["claim_rows"]["eligible_direction"] == {
        "numerator": 0,
        "denominator": 0,
        "fraction": None,
    }


def test_b8_structural_absences_are_reported_outside_stale_denominator() -> None:
    inputs = _valid_inputs()
    inputs["b8_absences"] = [
        {
            "anomaly_id": ANOMALY_IDS[0],
            "source": "asos",
            "metric": None,
            "reason": "source-absent-from-window",
        },
        {
            "anomaly_id": ANOMALY_IDS[0],
            "source": "openaq",
            "metric": "ozone",
            "reason": "nearest-event-value-absent",
        },
    ]

    report = build_funnel_report(**inputs)

    b8 = report["tables"]["b8_real_location"]
    assert b8["sources"]["asos"]["denominator"] == 0
    assert b8["structural_absences"] == inputs["b8_absences"]


def test_json_and_markdown_are_byte_identical_and_zero_denominator_is_n0() -> None:
    inputs = _valid_inputs()
    inputs["b8_observations"] = []
    first = build_funnel_report(**copy.deepcopy(inputs))
    second = build_funnel_report(**copy.deepcopy(inputs))

    assert canonical_json(first) == canonical_json(second)
    assert render_markdown(first) == render_markdown(second)
    markdown = render_markdown(first)
    assert "n=0" in markdown
    assert "recognized-but-absent-from-context" in markdown
    assert "Citation failure strings" in markdown
    assert "Claim type distribution" in markdown
    assert "Local/cloud prompt ratios" in markdown
    assert "Effective cutoff" in markdown
    assert "Estimated USD" in markdown
    for heading in (
        line for line in markdown.splitlines() if line.startswith("### ")
    ):
        assert "claim rows=15; unique decisions=5" in heading


def test_manual_atomicity_above_ten_percent_is_a_review_not_hard_stop() -> None:
    inputs = _valid_inputs()
    manual = inputs["manual_atomicity"]
    assert isinstance(manual, dict)
    manual[next(iter(manual))] = ATOMICITY_COMPOUND

    report = build_funnel_report(**inputs)

    assert report["go_no_go"]["status"] == "review_required"
    assert any(
        "manual non-self-contained rate > 10%" in item
        for item in report["go_no_go"]["review_items"]
    )


def test_missing_report_provenance_fails_before_a_report_can_claim_identity() -> None:
    inputs = _valid_inputs()
    provenance = inputs["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("db_copy_sha256")

    with pytest.raises(FunnelAuditError, match="missing provenance fields"):
        build_funnel_report(**inputs)


def test_iteration_writer_refuses_to_overwrite_prior_report(tmp_path: Path) -> None:
    report = build_funnel_report(**_valid_inputs())

    paths = write_iteration_reports(tmp_path, report)
    assert json.loads(paths["json"].read_text())["provenance"]["iteration"] == 1
    assert paths["markdown"].exists()
    with pytest.raises(FileExistsError, match="preserved"):
        write_iteration_reports(tmp_path, report)
