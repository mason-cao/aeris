"""Deterministic B19 anomaly selection and disposable-funnel audit.

This module is intentionally pure: it accepts an explicit extraction payload
and never opens a database, snapshot, network connection, or model client. The
disposable-DB orchestration layer is responsible for producing the payload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from app.eval.harness import DEFAULT_MODELS, USD_PER_MTOK
from app.llm.corroboration import (
    CALM_WIND_FLOOR_STATUS,
    ClaimType,
    HEADLINE_TYPES,
    calm_wind_manifest_payload,
    concentration_claim_shape,
)
from app.llm.observation_age import (
    DEFAULT_OBSERVATION_AGE_GATES,
    assess_observation_age,
)
from app.llm.validate import (
    CITATION_REASON_ABSENT,
    CITATION_REASON_BLANK,
    CITATION_REASON_UNRECOGNIZED,
    CITED_RIGHT,
    CITED_WRONG,
    GROUNDED,
    UNCITED,
    UNVERIFIED,
)


MASTER_SEED: Final = 20_260_716
SELECTED_ANOMALY_COUNT: Final = 5
EXPECTED_STEP_COUNT: Final = 4
LOCAL_PROMPT_REVIEW_LIMIT: Final = 7_680
LOCAL_PROMPT_HARD_LIMIT: Final = 8_192

ATOMICITY_SELF_CONTAINED: Final = "self-contained"
ATOMICITY_EXTERNAL_ANTECEDENT: Final = "external-antecedent"
ATOMICITY_MISSING_SUBJECT: Final = "missing-subject-or-metric"
ATOMICITY_COMPOUND: Final = "compound"
ATOMICITY_OTHER_CONTEXT: Final = "other-context-dependence"
ATOMICITY_CATEGORIES: Final = (
    ATOMICITY_SELF_CONTAINED,
    ATOMICITY_EXTERNAL_ANTECEDENT,
    ATOMICITY_MISSING_SUBJECT,
    ATOMICITY_COMPOUND,
    ATOMICITY_OTHER_CONTEXT,
)

_CITATION_REASONS: Final = (
    CITATION_REASON_BLANK,
    CITATION_REASON_UNRECOGNIZED,
    CITATION_REASON_ABSENT,
)
_DETECTORS: Final = ("isolation_forest", "stl", "zscore")
_WIND_SOURCES: Final = ("noaa_gfs", "openweather", "asos")
_DIRECTION_TYPES: Final = frozenset(
    {
        ClaimType.TRANSPORT_DIRECTION.value,
        ClaimType.POINT_SOURCE_ATTRIBUTION.value,
    }
)
_HOURLY_SOURCES: Final = frozenset(
    {"openaq", "tceq", "purpleair", "asos", "openweather", "epa_aqs"}
)
_CLAIM_TYPES: Final = frozenset(item.value for item in ClaimType)
_HEADLINE_TYPE_VALUES: Final = frozenset(item.value for item in HEADLINE_TYPES)
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")

_REPORTING_VERBS = (
    r"is|are|was|were|has|had|can|could|may|might|would|"
    r"shows?|suggests?|indicates?|implies?|supports?|means?"
)
_EXTERNAL_ANTECEDENT_RE = re.compile(
    rf"^\s*(?:this|that|it|they|these|those|he|she)\s+"
    rf"(?:{_REPORTING_VERBS})\b",
    re.IGNORECASE,
)
_MISSING_SUBJECT_RE = re.compile(
    r"^\s*(?:is|are|was|were|remained|increased|decreased|rose|fell|"
    r"peaked|exceeded|elevated|stagnant|transported|consistent\s+with|"
    r"likely\s+due\s+to)\b",
    re.IGNORECASE,
)
_FINITE_VERBS = (
    r"is|are|was|were|has|have|had|rose|fell|peaked|increased|decreased|"
    r"remained|exceeded|shows?|suggests?|indicates?|supports?|transported"
)
_COMPOUND_RE = re.compile(
    rf"(?:[.;]\s+|\b(?:and|but|while|whereas)\s+)"
    rf"(?:the\s+)?(?:[a-z][a-z0-9_-]*)(?:\s+[a-z][a-z0-9_-]*){{0,2}}?"
    rf"\s+(?:{_FINITE_VERBS})\b",
    re.IGNORECASE,
)
_OTHER_CONTEXT_RE = re.compile(
    r"(?:\bas\s+(?:noted|described)\b|\bthe\s+(?:former|latter)\b|"
    r"\bsame\s+(?:pattern|source|period)\b|"
    r"^\s*(?:also|therefore|however|moreover|additionally|again)\b)",
    re.IGNORECASE,
)
_NO_DATA_IN_WINDOW_RE = re.compile(
    r"(?:no verified-monitor \S+ observation in window|no \S+ in window)",
    re.IGNORECASE,
)


class FunnelSelectionError(ValueError):
    """Raised when the ratified B19 anomaly selection cannot be honored."""


class FunnelAuditError(ValueError):
    """Raised when an input is malformed rather than a reportable finding."""


def _required_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FunnelAuditError(f"{field} must be an object")
    return value


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FunnelAuditError(f"{field} must be a nonempty string")
    return value


def _strict_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise FunnelAuditError(f"{field} must be boolean")
    return value


def _nonnegative_int(value: object, field: str) -> int | None:
    if type(value) is not int or value < 0:
        return None
    return value


def _finite_or_none(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FunnelAuditError(f"malformed outcome: {field} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise FunnelAuditError(f"malformed outcome: {field} must be finite")
    return result


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise FunnelAuditError(
            f"invalid rate counts: numerator={numerator}, denominator={denominator}"
        )
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def _validated_detector_availability(
    value: object,
    *,
    anomaly_id: str,
) -> dict[str, dict[str, bool | str | None]]:
    if not isinstance(value, Mapping) or set(value) != set(_DETECTORS):
        raise FunnelSelectionError(
            f"selected anomaly {anomaly_id} has missing detector provenance"
        )
    normalized: dict[str, dict[str, bool | str | None]] = {}
    for detector in _DETECTORS:
        raw_entry = value.get(detector)
        if not isinstance(raw_entry, Mapping):
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} has malformed {detector} provenance"
            )
        ran = raw_entry.get("ran")
        skip_code = raw_entry.get("skip_code")
        detail = raw_entry.get("detail")
        if not isinstance(ran, bool):
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} has malformed {detector}.ran"
            )
        if ran and skip_code is not None:
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} ran {detector} with a skip code"
            )
        if not ran and (not isinstance(skip_code, str) or not skip_code):
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} skipped {detector} without a code"
            )
        if detail is not None and not isinstance(detail, str):
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} has malformed {detector}.detail"
            )
        normalized[detector] = {
            "ran": ran,
            "skip_code": skip_code,
            "detail": detail,
        }
    return normalized


def _ranked_rows(ranked_anomalies: object) -> list[dict[str, Any]]:
    if not isinstance(ranked_anomalies, list):
        raise FunnelSelectionError("ranked anomalies must be an array")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, raw in enumerate(ranked_anomalies, start=1):
        if not isinstance(raw, Mapping):
            raise FunnelSelectionError(f"ranked anomaly {rank} must be an object")
        try:
            anomaly_id = _required_string(raw.get("anomaly_id"), "anomaly_id")
            source = _required_string(raw.get("source"), "source")
            metric = _required_string(raw.get("metric"), "metric")
        except FunnelAuditError as exc:
            raise FunnelSelectionError(str(exc)) from exc
        if anomaly_id in seen:
            raise FunnelSelectionError(f"duplicate ranked anomaly ID: {anomaly_id}")
        seen.add(anomaly_id)
        rows.append(
            {
                "freeze_rank": rank,
                "anomaly_id": anomaly_id,
                "source": source,
                "metric": metric,
                "source_entity_id": raw.get("source_entity_id"),
                "detector_availability": raw.get("detector_availability"),
                "enrichment_present": raw.get("enrichment_present"),
            }
        )
    return rows


def select_funnel_anomalies(
    ranked_anomalies: object,
    *,
    count: int = SELECTED_ANOMALY_COUNT,
) -> dict[str, Any]:
    """Apply the ratified distinct-metric-first B19 selection rule."""
    if type(count) is not int or count < 1:
        raise FunnelSelectionError("selection count must be a positive integer")
    ranked = _ranked_rows(ranked_anomalies)
    if len(ranked) < count:
        raise FunnelSelectionError(
            f"ranked population has {len(ranked)} rows; cannot select {count}"
        )

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    seen_metrics: set[str] = set()
    for row in ranked:
        metric = str(row["metric"])
        if metric in seen_metrics:
            continue
        selected.append({**row, "selection_reason": "first_distinct_metric"})
        selected_ids.add(str(row["anomaly_id"]))
        seen_metrics.add(metric)
        if len(selected) == count:
            break

    if len(selected) < count:
        for row in ranked:
            anomaly_id = str(row["anomaly_id"])
            if anomaly_id in selected_ids:
                continue
            selected.append({**row, "selection_reason": "global_rank_fill"})
            selected_ids.add(anomaly_id)
            if len(selected) == count:
                break

    for row in selected:
        anomaly_id = str(row["anomaly_id"])
        if row["enrichment_present"] is not True:
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} has no enrichment; no substitution"
            )
        entity_id = row["source_entity_id"]
        if not isinstance(entity_id, str) or not entity_id:
            raise FunnelSelectionError(
                f"selected anomaly {anomaly_id} has missing trigger-entity provenance"
            )
        row["detector_availability"] = _validated_detector_availability(
            row["detector_availability"], anomaly_id=anomaly_id
        )

    plain_top_five = ranked[:count]
    return {
        "strategy": (
            "first freeze-ranked representative per distinct metric; if the "
            "full rank has fewer than five metrics, fill unused slots by "
            "global freeze rank"
        ),
        "ranked_population_count": len(ranked),
        "selected_anomaly_ids": [str(row["anomaly_id"]) for row in selected],
        "selected": selected,
        "plain_top_five_anomaly_ids": [
            str(row["anomaly_id"]) for row in plain_top_five
        ],
        "plain_top_five": plain_top_five,
    }


def screen_atomicity(claim_text: str) -> str:
    """Apply the declared diagnostic lexical screen, first match only."""
    text = _required_string(claim_text, "claim_text")
    if _EXTERNAL_ANTECEDENT_RE.search(text):
        return ATOMICITY_EXTERNAL_ANTECEDENT
    if _MISSING_SUBJECT_RE.search(text):
        return ATOMICITY_MISSING_SUBJECT
    if _COMPOUND_RE.search(text):
        return ATOMICITY_COMPOUND
    if _OTHER_CONTEXT_RE.search(text):
        return ATOMICITY_OTHER_CONTEXT
    return ATOMICITY_SELF_CONTAINED


def _decision_hash(anomaly_id: str, claim_text: str) -> str:
    payload = f"{anomaly_id}\0{claim_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _decision_units(claims: object) -> list[dict[str, str]]:
    if not isinstance(claims, list):
        raise FunnelAuditError("claims must be an array")
    units: dict[tuple[str, str], dict[str, str]] = {}
    hashes: dict[str, tuple[str, str]] = {}
    for position, raw in enumerate(claims, start=1):
        claim = _required_mapping(raw, f"claim {position}")
        anomaly_id = _required_string(
            claim.get("anomaly_id"), f"claim {position} anomaly_id"
        )
        claim_text = _required_string(
            claim.get("claim_text"), f"claim {position} claim_text"
        )
        key = (anomaly_id, claim_text)
        decision_hash = _decision_hash(*key)
        prior = hashes.get(decision_hash)
        if prior is not None and prior != key:
            raise FunnelAuditError("atomicity decision-hash collision")
        hashes[decision_hash] = key
        units[key] = {
            "anomaly_id": anomaly_id,
            "claim_text": claim_text,
            "decision_hash": decision_hash,
        }
    return [units[key] for key in sorted(units)]


def build_atomicity_worksheet(claims: object) -> dict[str, Any]:
    """Return the seeded, model-blind manual atomicity worksheet."""
    units = _decision_units(claims)
    seed = np.random.SeedSequence(MASTER_SEED).spawn(1)[0]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(units)).tolist()
    items = [
        {
            "review_index": review_index,
            "decision_hash": units[position]["decision_hash"],
            "claim_text": units[position]["claim_text"],
        }
        for review_index, position in enumerate(order, start=1)
    ]
    return {
        "schema_version": 1,
        "master_seed": MASTER_SEED,
        "unit": "unique (anomaly_id, exact claim_text)",
        "blinding": (
            "model, anomaly context, citations, lexical result, and scorer "
            "outcomes omitted"
        ),
        "categories": list(ATOMICITY_CATEGORIES),
        "items": items,
    }


def _validate_manual_atomicity(
    units: Sequence[Mapping[str, str]],
    manual_atomicity: object,
) -> dict[str, str]:
    if not isinstance(manual_atomicity, Mapping):
        raise FunnelAuditError("manual atomicity decisions must be an object")
    expected = {str(unit["decision_hash"]) for unit in units}
    supplied = set(manual_atomicity)
    missing = sorted(expected - supplied)
    extra = sorted(supplied - expected)
    if missing or extra:
        raise FunnelAuditError(
            "manual atomicity decisions must be complete and exact: "
            f"missing={missing}, extra={extra}"
        )
    normalized: dict[str, str] = {}
    for decision_hash in sorted(expected):
        category = manual_atomicity[decision_hash]
        if category not in ATOMICITY_CATEGORIES:
            raise FunnelAuditError(
                f"manual atomicity category for {decision_hash} is invalid: "
                f"{category!r}"
            )
        normalized[decision_hash] = str(category)
    return normalized


def _atomicity_summary(
    claims: Sequence[Mapping[str, Any]],
    manual_atomicity: object,
) -> dict[str, Any]:
    units = _decision_units(list(claims))
    manual = _validate_manual_atomicity(units, manual_atomicity)
    rows: list[dict[str, str]] = []
    manual_failures = 0
    disagreements = 0
    screen_counts: Counter[str] = Counter()
    manual_counts: Counter[str] = Counter()
    for unit in units:
        decision_hash = unit["decision_hash"]
        screen = screen_atomicity(unit["claim_text"])
        adjudication = manual[decision_hash]
        screen_counts[screen] += 1
        manual_counts[adjudication] += 1
        if adjudication != ATOMICITY_SELF_CONTAINED:
            manual_failures += 1
        if screen != adjudication:
            disagreements += 1
        rows.append(
            {
                "decision_hash": decision_hash,
                "claim_text": unit["claim_text"],
                "lexical_screen": screen,
                "manual_category": adjudication,
            }
        )
    denominator = len(units)
    return {
        "manual_non_self_contained": _rate(manual_failures, denominator),
        "lexical_screen_counts": {
            category: screen_counts.get(category, 0)
            for category in ATOMICITY_CATEGORIES
        },
        "manual_counts": {
            category: manual_counts.get(category, 0)
            for category in ATOMICITY_CATEGORIES
        },
        "screen_manual_disagreement": _rate(disagreements, denominator),
        "manual_review_trigger": 10 * manual_failures > denominator,
        "rows": sorted(rows, key=lambda row: row["decision_hash"]),
    }


def _normalize_citation_fields(
    claim: Mapping[str, Any],
    *,
    description: str,
) -> dict[str, Any]:
    raw_sources = claim.get("cited_sources")
    if raw_sources is None:
        sources: list[str] = []
    elif isinstance(raw_sources, list) and all(
        isinstance(source, str) for source in raw_sources
    ):
        sources = list(raw_sources)
    else:
        raise FunnelAuditError(
            f"malformed outcome: {description} cited_sources must be strings"
        )

    outcome = claim.get("citation_outcome")
    if outcome not in {CITED_RIGHT, CITED_WRONG, UNCITED}:
        raise FunnelAuditError(
            f"malformed outcome: {description} citation_outcome={outcome!r}"
        )
    raw_reasons = claim.get("citation_failure_reasons")
    if not isinstance(raw_reasons, list):
        raise FunnelAuditError(
            f"malformed outcome: {description} citation reasons must be an array"
        )

    reasons: list[dict[str, int | str]] = []
    seen_indexes: set[int] = set()
    for position, raw_reason in enumerate(raw_reasons, start=1):
        if not isinstance(raw_reason, Mapping) or set(raw_reason) != {
            "index",
            "citation",
            "reason",
        }:
            raise FunnelAuditError(
                f"malformed outcome: {description} reason {position} shape"
            )
        index = raw_reason.get("index")
        citation = raw_reason.get("citation")
        reason = raw_reason.get("reason")
        if type(index) is not int or index < 0 or index >= len(sources):
            raise FunnelAuditError(
                f"malformed outcome: {description} reason index {index!r}"
            )
        if index in seen_indexes:
            raise FunnelAuditError(
                f"malformed outcome: {description} duplicate reason index {index}"
            )
        seen_indexes.add(index)
        if citation != sources[index]:
            raise FunnelAuditError(
                f"malformed outcome: {description} reason citation mismatch"
            )
        if reason not in _CITATION_REASONS:
            raise FunnelAuditError(
                f"malformed outcome: {description} unknown reason {reason!r}"
            )
        if (reason == CITATION_REASON_BLANK) != (not str(citation).strip()):
            raise FunnelAuditError(
                f"malformed outcome: {description} blank-only mismatch"
            )
        reasons.append(
            {"index": index, "citation": str(citation), "reason": str(reason)}
        )

    reason_by_index = {int(reason["index"]): reason for reason in reasons}
    for index, source in enumerate(sources):
        if not source.strip() and (
            index not in reason_by_index
            or reason_by_index[index]["reason"] != CITATION_REASON_BLANK
        ):
            raise FunnelAuditError(
                f"malformed outcome: {description} unrecorded blank citation"
            )
    if not sources:
        expected_outcome = UNCITED
    else:
        has_nonblank = any(source.strip() for source in sources)
        has_nonblank_failure = any(
            reason["reason"] != CITATION_REASON_BLANK for reason in reasons
        )
        expected_outcome = (
            CITED_WRONG
            if not has_nonblank or has_nonblank_failure
            else CITED_RIGHT
        )
    if outcome != expected_outcome:
        raise FunnelAuditError(
            f"malformed outcome: {description} citation_outcome={outcome!r}; "
            f"expected {expected_outcome!r} from persisted reasons"
        )
    return {
        "cited_sources": sources,
        "citation_outcome": str(outcome),
        "citation_failure_reasons": reasons,
    }


def summarize_citations(
    claims: object,
    models: Sequence[str] = DEFAULT_MODELS,
) -> dict[str, dict[str, Any]]:
    """Summarize B11 omission and B19 citation reasons per model."""
    if not isinstance(claims, list):
        raise FunnelAuditError("claims must be an array")
    model_order = tuple(models)
    if len(model_order) != len(set(model_order)):
        raise FunnelAuditError("models must be unique")
    prepared: list[dict[str, Any]] = []
    for position, raw_claim in enumerate(claims, start=1):
        claim = _required_mapping(raw_claim, f"claim {position}")
        model = _required_string(claim.get("model"), f"claim {position} model")
        if model not in model_order:
            raise FunnelAuditError(f"claim {position} has unexpected model {model}")
        prepared.append(
            {
                "model": model,
                "claim_ref": (
                    claim.get("claim_id")
                    if isinstance(claim.get("claim_id"), str)
                    else f"input-{position}"
                ),
                "anomaly_id": (
                    claim.get("anomaly_id")
                    if isinstance(claim.get("anomaly_id"), str)
                    else None
                ),
                **_normalize_citation_fields(
                    claim,
                    description=f"claim {position}",
                ),
            }
        )

    result: dict[str, dict[str, Any]] = {}
    for model in model_order:
        rows = [claim for claim in prepared if claim["model"] == model]
        rate_denominator_rows = [
            claim
            for claim in rows
            if any(source.strip() for source in claim["cited_sources"])
        ]
        claim_reason_counts_all: Counter[str] = Counter()
        claim_reason_counts_denominator: Counter[str] = Counter()
        string_reason_counts: Counter[str] = Counter()
        multiple = 0
        for claim in rows:
            categories = {
                str(reason["reason"])
                for reason in claim["citation_failure_reasons"]
            }
            claim_reason_counts_all.update(categories)
            string_reason_counts.update(
                str(reason["reason"])
                for reason in claim["citation_failure_reasons"]
            )
            if any(source.strip() for source in claim["cited_sources"]):
                claim_reason_counts_denominator.update(categories)
            if len(categories) >= 2:
                multiple += 1
        result[model] = {
            "all_claim_rows": len(rows),
            "claims_with_nonblank_citation": len(rate_denominator_rows),
            "all_blank_citation_claims": sum(
                1
                for claim in rows
                if claim["cited_sources"]
                and not any(source.strip() for source in claim["cited_sources"])
            ),
            "omission_count": sum(
                claim["citation_outcome"] == UNCITED for claim in rows
            ),
            "omission_rate": _rate(
                sum(claim["citation_outcome"] == UNCITED for claim in rows),
                len(rows),
            ),
            "reason_counts_all_claims": {
                reason: claim_reason_counts_all.get(reason, 0)
                for reason in _CITATION_REASONS
            },
            "reason_string_counts_all_claims": {
                reason: string_reason_counts.get(reason, 0)
                for reason in _CITATION_REASONS
            },
            "reason_counts_rate_denominator": {
                reason: claim_reason_counts_denominator.get(reason, 0)
                for reason in _CITATION_REASONS
            },
            "reason_rates": {
                reason: _rate(
                    claim_reason_counts_denominator.get(reason, 0),
                    len(rate_denominator_rows),
                )
                for reason in _CITATION_REASONS
            },
            "multiple_reasons_claim_count": multiple,
            "failure_records": [
                {
                    "claim_ref": claim["claim_ref"],
                    "anomaly_id": claim["anomaly_id"],
                    "citation_index": reason["index"],
                    "citation": reason["citation"],
                    "reason": reason["reason"],
                }
                for claim in rows
                for reason in claim["citation_failure_reasons"]
            ],
        }
    return result


def summarize_b8_rate(
    source: str,
    *,
    stale_count: int,
    denominator: int,
) -> dict[str, Any]:
    """Return one B8 rate using the exact integer 20% hard-stop boundary."""
    if source not in DEFAULT_OBSERVATION_AGE_GATES.to_dict():
        raise FunnelAuditError(f"unknown B8 source: {source}")
    if type(stale_count) is not int or type(denominator) is not int:
        raise FunnelAuditError("B8 counts must be integers")
    rate = _rate(stale_count, denominator)
    return {
        "source": source,
        "gate_minutes": DEFAULT_OBSERVATION_AGE_GATES.for_source(source),
        "stale_count": stale_count,
        "denominator": denominator,
        "fraction_silenced": rate["fraction"],
        "hourly": source in _HOURLY_SOURCES,
        "hourly_hard_stop": (
            source in _HOURLY_SOURCES
            and denominator > 0
            and 5 * stale_count > denominator
        ),
    }


def _summarize_b8_observations(
    observations: object,
    absences: object,
    selected_ids: set[str],
) -> tuple[dict[str, Any], list[str]]:
    if not isinstance(observations, list):
        raise FunnelAuditError("B8 observations must be an array")
    counts: dict[str, list[int]] = {
        source: [0, 0]
        for source in DEFAULT_OBSERVATION_AGE_GATES.to_dict()
    }
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for position, raw in enumerate(observations, start=1):
        row = _required_mapping(raw, f"B8 observation {position}")
        anomaly_id = _required_string(
            row.get("anomaly_id"), f"B8 observation {position} anomaly_id"
        )
        source = _required_string(
            row.get("source"), f"B8 observation {position} source"
        )
        metric = _required_string(
            row.get("metric"), f"B8 observation {position} metric"
        )
        if anomaly_id not in selected_ids:
            raise FunnelAuditError(
                f"B8 observation references unselected anomaly {anomaly_id}"
            )
        if source not in counts:
            raise FunnelAuditError(f"B8 observation has unknown source {source}")
        key = (anomaly_id, source, metric)
        if key in seen:
            raise FunnelAuditError(f"duplicate B8 observation: {key}")
        seen.add(key)
        raw_dt = row.get("dt_minutes")
        if (
            isinstance(raw_dt, bool)
            or not isinstance(raw_dt, (int, float))
            or not math.isfinite(float(raw_dt))
            or float(raw_dt) < 0.0
        ):
            raise FunnelAuditError(
                f"malformed outcome: B8 observation {key} has invalid dt_minutes"
            )
        decision = assess_observation_age(source, raw_dt)
        counts[source][1] += 1
        if not decision.votes:
            counts[source][0] += 1
        rows.append(
            {
                "anomaly_id": anomaly_id,
                "source": source,
                "metric": metric,
                "dt_minutes": float(raw_dt),
                "gate_minutes": decision.gate_minutes,
                "silenced": not decision.votes,
            }
        )

    sources = {
        source: summarize_b8_rate(
            source,
            stale_count=counts[source][0],
            denominator=counts[source][1],
        )
        for source in sorted(counts)
    }
    hard_stops = [
        f"hourly B8 real-location silence rate > 20% for {source}"
        for source, summary in sources.items()
        if summary["hourly_hard_stop"] is True
    ]
    if not isinstance(absences, list):
        raise FunnelAuditError("B8 structural absences must be an array")
    normalized_absences: list[dict[str, str | None]] = []
    seen_absences: set[tuple[str, str, str | None, str]] = set()
    allowed_absence_reasons = {
        "source-absent-from-window",
        "nearest-event-value-absent",
    }
    for position, raw in enumerate(absences, start=1):
        absence = _required_mapping(raw, f"B8 structural absence {position}")
        anomaly_id = _required_string(
            absence.get("anomaly_id"),
            f"B8 structural absence {position} anomaly_id",
        )
        source = _required_string(
            absence.get("source"), f"B8 structural absence {position} source"
        )
        raw_metric = absence.get("metric")
        metric = (
            None
            if raw_metric is None
            else _required_string(
                raw_metric, f"B8 structural absence {position} metric"
            )
        )
        reason = _required_string(
            absence.get("reason"), f"B8 structural absence {position} reason"
        )
        if anomaly_id not in selected_ids:
            raise FunnelAuditError(
                f"B8 structural absence references unselected anomaly {anomaly_id}"
            )
        if source not in counts:
            raise FunnelAuditError(
                f"B8 structural absence has unknown source {source}"
            )
        if reason not in allowed_absence_reasons:
            raise FunnelAuditError(
                f"B8 structural absence has unknown reason {reason}"
            )
        if (reason == "source-absent-from-window") != (metric is None):
            raise FunnelAuditError(
                "B8 source absence must use metric=null and metric absence must "
                "name its metric"
            )
        key = (anomaly_id, source, metric, reason)
        if key in seen_absences:
            raise FunnelAuditError(f"duplicate B8 structural absence: {key}")
        seen_absences.add(key)
        normalized_absences.append(
            {
                "anomaly_id": anomaly_id,
                "source": source,
                "metric": metric,
                "reason": reason,
            }
        )
    return {
        "population": (
            "selected-anomaly source/metric blocks with a stored nearest "
            "event value"
        ),
        "sources": sources,
        "rows": sorted(
            rows,
            key=lambda row: (
                row["source"], row["metric"], row["anomaly_id"]
            ),
        ),
        "structural_absences": sorted(
            normalized_absences,
            key=lambda row: (
                str(row["source"]),
                str(row["metric"] or ""),
                str(row["anomaly_id"]),
            ),
        ),
    }, hard_stops


def summarize_b17_silence(claims: object) -> dict[str, dict[str, int]]:
    """Split qualitative concentration abstentions using persisted notes."""
    if not isinstance(claims, list):
        raise FunnelAuditError("claims must be an array")
    rows: list[tuple[tuple[str, str], bool, bool]] = []
    for position, raw in enumerate(claims, start=1):
        claim = _required_mapping(raw, f"claim {position}")
        if claim.get("claim_type") != ClaimType.CONCENTRATION_ELEVATION.value:
            continue
        claim_text = _required_string(
            claim.get("claim_text"), f"claim {position} claim_text"
        )
        if concentration_claim_shape(claim_text) != "qualitative":
            continue
        anomaly_id = _required_string(
            claim.get("anomaly_id"), f"claim {position} anomaly_id"
        )
        note = claim.get("corroboration_evidence_summary")
        note_text = note if isinstance(note, str) else ""
        matched = "reason=matched baseline n < 3" in note_text
        no_data = _NO_DATA_IN_WINDOW_RE.search(note_text) is not None
        rows.append(((anomaly_id, claim_text), matched, no_data))

    claim_counts = {
        "qualitative_concentration": len(rows),
        "matched_baseline_n_lt_3": sum(matched for _key, matched, _no in rows),
        "no_data_in_window": sum(no_data for _key, _matched, no_data in rows),
        "both": sum(matched and no_data for _key, matched, no_data in rows),
    }
    unique: dict[tuple[str, str], tuple[bool, bool]] = {}
    for key, matched, no_data in rows:
        prior = unique.get(key, (False, False))
        unique[key] = (prior[0] or matched, prior[1] or no_data)
    unique_counts = {
        "qualitative_concentration": len(unique),
        "matched_baseline_n_lt_3": sum(value[0] for value in unique.values()),
        "no_data_in_window": sum(value[1] for value in unique.values()),
        "both": sum(value[0] and value[1] for value in unique.values()),
    }
    return {
        "claim_rows": claim_counts,
        "unique_anomaly_exact_text": unique_counts,
    }


def _normalize_claims(
    claims: object,
    *,
    selected_ids: set[str],
    models: tuple[str, ...],
) -> list[dict[str, Any]]:
    if not isinstance(claims, list):
        raise FunnelAuditError("claims must be an array")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, raw in enumerate(claims, start=1):
        claim = _required_mapping(raw, f"claim {position}")
        claim_id = _required_string(claim.get("claim_id"), f"claim {position} ID")
        if claim_id in seen_ids:
            raise FunnelAuditError(f"malformed outcome: duplicate claim ID {claim_id}")
        seen_ids.add(claim_id)
        anomaly_id = _required_string(
            claim.get("anomaly_id"), f"claim {claim_id} anomaly_id"
        )
        model = _required_string(claim.get("model"), f"claim {claim_id} model")
        if anomaly_id not in selected_ids or model not in models:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} references an unexpected cell"
            )
        claim_text = _required_string(
            claim.get("claim_text"), f"claim {claim_id} text"
        )
        claim_type = claim.get("claim_type")
        if claim_type not in _CLAIM_TYPES:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} type={claim_type!r}"
            )
        matched_types = claim.get("matched_types")
        if (
            not isinstance(matched_types, list)
            or not matched_types
            or matched_types[0] != claim_type
            or any(item not in _CLAIM_TYPES for item in matched_types)
        ):
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} matched_types"
            )
        citation = _normalize_citation_fields(
            claim,
            description=f"claim {claim_id}",
        )
        grounding = claim.get("grounding_verdict")
        if grounding not in {GROUNDED, UNVERIFIED}:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} grounding={grounding!r}"
            )
        skipped = claim.get("skipped_phase2")
        if not isinstance(skipped, bool):
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} skipped_phase2"
            )
        evidence_n = claim.get("evidence_n")
        if type(evidence_n) is not int or evidence_n < 0:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} evidence_n"
            )
        score = _finite_or_none(
            claim.get("corroboration_score"),
            f"claim {claim_id} corroboration_score",
        )
        if score is not None and not -1.0 <= score <= 1.0:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} score outside [-1, 1]"
            )
        evidence_summary = claim.get("corroboration_evidence_summary")
        if grounding == GROUNDED:
            if skipped or not isinstance(evidence_summary, str) or not evidence_summary:
                raise FunnelAuditError(
                    f"malformed outcome: grounded claim {claim_id} Phase-2 fields"
                )
        elif (
            not skipped
            or score is not None
            or evidence_n != 0
            or evidence_summary is not None
        ):
            raise FunnelAuditError(
                f"malformed outcome: unverified claim {claim_id} Phase-2 fields"
            )
        if (evidence_n == 0) != (score is None):
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} score/evidence_n mismatch"
            )
        causal = claim.get("causal")
        calm_flagged = claim.get("calm_wind_flagged")
        direction_present = claim.get("direction_data_present")
        if not all(
            isinstance(value, bool)
            for value in (causal, calm_flagged, direction_present)
        ):
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} boolean audit fields"
            )
        if calm_flagged and claim_type not in _DIRECTION_TYPES:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} calm flag is ineligible"
            )
        if direction_present and claim_type not in _DIRECTION_TYPES:
            raise FunnelAuditError(
                f"malformed outcome: claim {claim_id} direction-data annotation"
            )
        normalized.append(
            {
                "claim_id": claim_id,
                "anomaly_id": anomaly_id,
                "model": model,
                "claim_text": claim_text,
                "claim_type": str(claim_type),
                "matched_types": list(matched_types),
                **citation,
                "grounding_verdict": str(grounding),
                "skipped_phase2": skipped,
                "corroboration_score": score,
                "evidence_n": evidence_n,
                "corroboration_evidence_summary": evidence_summary,
                "causal": causal,
                "calm_wind_flagged": calm_flagged,
                "direction_data_present": direction_present,
            }
        )
    return sorted(
        normalized,
        key=lambda row: (row["anomaly_id"], row["model"], row["claim_id"]),
    )


def _audit_cells(
    cells: object,
    *,
    selected_ids: Sequence[str],
    models: tuple[str, ...],
) -> tuple[dict[str, Any], list[str], list[str], dict[str, Any]]:
    if not isinstance(cells, list):
        raise FunnelAuditError("cells must be an array")
    expected = {(anomaly_id, model) for anomaly_id in selected_ids for model in models}
    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    hard_stops: list[str] = []
    review_items: list[str] = []
    step_rows: list[dict[str, Any]] = []
    totals: dict[tuple[str, str], tuple[int | None, int | None]] = {}
    token_totals: dict[str, list[int]] = {model: [0, 0] for model in models}

    for position, raw in enumerate(cells, start=1):
        cell = _required_mapping(raw, f"cell {position}")
        anomaly_id = _required_string(
            cell.get("anomaly_id"), f"cell {position} anomaly_id"
        )
        model = _required_string(cell.get("model"), f"cell {position} model")
        key = (anomaly_id, model)
        if key in seen:
            hard_stops.append(f"duplicate completed cell {anomaly_id}/{model}")
            continue
        seen[key] = cell
        raw_steps = cell.get("steps")
        if not isinstance(raw_steps, list) or len(raw_steps) != EXPECTED_STEP_COUNT:
            hard_stops.append(
                f"cell {anomaly_id}/{model} has {len(raw_steps) if isinstance(raw_steps, list) else 'malformed'} steps != 4"
            )
            totals[key] = (None, None)
            continue
        prompt_total = 0
        completion_total = 0
        complete_tokens = True
        for step_index, raw_step in enumerate(raw_steps, start=1):
            if not isinstance(raw_step, Mapping):
                hard_stops.append(
                    f"missing token metadata for {anomaly_id}/{model}/step {step_index}"
                )
                complete_tokens = False
                continue
            prompt_tokens = _nonnegative_int(
                raw_step.get("prompt_tokens"), "prompt_tokens"
            )
            completion_tokens = _nonnegative_int(
                raw_step.get("completion_tokens"), "completion_tokens"
            )
            attempts = raw_step.get("attempts")
            if prompt_tokens is None or completion_tokens is None:
                hard_stops.append(
                    f"missing token metadata for {anomaly_id}/{model}/step {step_index}"
                )
                complete_tokens = False
            if type(attempts) is not int or attempts < 1:
                hard_stops.append(
                    f"missing usage provenance for {anomaly_id}/{model}/step {step_index}"
                )
            if model == models[0] and prompt_tokens is not None:
                if prompt_tokens >= LOCAL_PROMPT_HARD_LIMIT:
                    hard_stops.append(
                        f"local prompt_eval_count >= 8192 for {anomaly_id}/{model}/step {step_index}"
                    )
                elif prompt_tokens > LOCAL_PROMPT_REVIEW_LIMIT:
                    review_items.append(
                        f"local prompt_eval_count > 7680 for {anomaly_id}/{model}/step {step_index}"
                    )
            step_rows.append(
                {
                    "anomaly_id": anomaly_id,
                    "model": model,
                    "step": step_index,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "attempts": attempts if type(attempts) is int else None,
                }
            )
            if prompt_tokens is not None:
                prompt_total += prompt_tokens
            if completion_tokens is not None:
                completion_total += completion_tokens
        totals[key] = (
            prompt_total if complete_tokens else None,
            completion_total if complete_tokens else None,
        )
        if complete_tokens:
            token_totals[model][0] += prompt_total
            token_totals[model][1] += completion_total

    if len(seen) != len(expected):
        hard_stops.append(f"completed cells {len(seen)} != {len(expected)}")
    missing = sorted(expected - set(seen))
    extra = sorted(set(seen) - expected)
    if missing:
        hard_stops.append(
            "missing cells: " + ", ".join(f"{a}/{m}" for a, m in missing)
        )
    if extra:
        hard_stops.append(
            "unexpected cells: " + ", ".join(f"{a}/{m}" for a, m in extra)
        )

    ratios: list[dict[str, Any]] = []
    local_model = models[0]
    for anomaly_id in selected_ids:
        local_prompt = totals.get((anomaly_id, local_model), (None, None))[0]
        for cloud_model in models[1:]:
            cloud_prompt = totals.get((anomaly_id, cloud_model), (None, None))[0]
            ratio = (
                local_prompt / cloud_prompt
                if local_prompt is not None
                and cloud_prompt is not None
                and cloud_prompt > 0
                else None
            )
            ratios.append(
                {
                    "anomaly_id": anomaly_id,
                    "local_model": local_model,
                    "cloud_model": cloud_model,
                    "local_prompt_tokens": local_prompt,
                    "cloud_prompt_tokens": cloud_prompt,
                    "local_to_cloud_ratio": ratio,
                }
            )

    per_model_cost: dict[str, dict[str, Any]] = {}
    pricing_complete = True
    for model in models:
        prompt_tokens, completion_tokens = token_totals[model]
        rates = USD_PER_MTOK.get(model)
        if rates is None:
            estimate = None
            if model != local_model:
                pricing_complete = False
        else:
            estimate = (
                prompt_tokens * rates[0] + completion_tokens * rates[1]
            ) / 1_000_000
        per_model_cost[model] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "usd_per_mtok": (
                {"prompt": rates[0], "completion": rates[1]}
                if rates is not None
                else None
            ),
            "estimated_cost_usd": estimate,
        }
    costs = {
        "status": "available" if pricing_complete else "n/a",
        "per_model": per_model_cost,
    }
    return (
        {
            "expected_cells": len(expected),
            "completed_cells": len(seen),
            "missing_cells": [list(item) for item in missing],
            "unexpected_cells": [list(item) for item in extra],
            "steps": sorted(
                step_rows,
                key=lambda row: (row["model"], row["anomaly_id"], row["step"]),
            ),
            "local_cloud_prompt_ratios": ratios,
        },
        hard_stops,
        review_items,
        costs,
    )


def _validate_calm_decisions(
    decisions: object,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(decisions, list):
        raise FunnelAuditError("calm-wind decisions must be an array")
    expected = {(anomaly_id, source) for anomaly_id in selected_ids for source in _WIND_SOURCES}
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for position, raw in enumerate(decisions, start=1):
        decision = _required_mapping(raw, f"calm-wind decision {position}")
        anomaly_id = _required_string(
            decision.get("anomaly_id"), f"calm-wind decision {position} anomaly_id"
        )
        source = _required_string(
            decision.get("source"), f"calm-wind decision {position} source"
        )
        key = (anomaly_id, source)
        if key not in expected or key in seen:
            raise FunnelAuditError(f"malformed outcome: calm-wind decision {key}")
        seen.add(key)
        window_n = decision.get("window_n")
        if type(window_n) is not int or window_n < 0:
            raise FunnelAuditError(
                f"malformed outcome: calm-wind decision {key} window_n"
            )
        guard_enabled = decision.get("guard_enabled")
        direction_votable = decision.get("direction_votable")
        if not isinstance(guard_enabled, bool) or not isinstance(
            direction_votable, bool
        ):
            raise FunnelAuditError(
                f"malformed outcome: calm-wind decision {key} booleans"
            )
        calm = decision.get("calm")
        if calm is not None and not isinstance(calm, bool):
            raise FunnelAuditError(
                f"malformed outcome: calm-wind decision {key} calm"
            )
        floor_status = decision.get("floor_status")
        if floor_status != CALM_WIND_FLOOR_STATUS:
            raise FunnelAuditError(
                f"malformed outcome: calm-wind decision {key} floor status"
            )
        normalized.append(
            {
                "anomaly_id": anomaly_id,
                "source": source,
                "window_n": window_n,
                "event_speed_ms": _finite_or_none(
                    decision.get("event_speed_ms"), f"calm-wind {key} event speed"
                ),
                "raw_cutoff_ms": _finite_or_none(
                    decision.get("raw_cutoff_ms"), f"calm-wind {key} raw cutoff"
                ),
                "effective_cutoff_ms": _finite_or_none(
                    decision.get("effective_cutoff_ms"),
                    f"calm-wind {key} effective cutoff",
                ),
                "guard_enabled": guard_enabled,
                "calm": calm,
                "direction_votable": direction_votable,
                "reason": _required_string(
                    decision.get("reason"), f"calm-wind decision {key} reason"
                ),
                "floor_status": str(floor_status),
            }
        )
    missing = sorted(expected - seen)
    if missing:
        raise FunnelAuditError(
            "malformed outcome: missing calm-wind decisions "
            + ", ".join(f"{a}/{s}" for a, s in missing)
        )
    return {
        "guard_manifest": calm_wind_manifest_payload(),
        "decisions": sorted(
            normalized, key=lambda row: (row["anomaly_id"], row["source"])
        ),
    }


def _summarize_calm_wind(claims: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    claim_rows = list(claims)
    eligible_rows = [
        claim
        for claim in claim_rows
        if claim["claim_type"] in _DIRECTION_TYPES
        and claim["direction_data_present"] is True
    ]
    unique: dict[tuple[str, str], dict[str, bool]] = {}
    for claim in claim_rows:
        key = (str(claim["anomaly_id"]), str(claim["claim_text"]))
        state = unique.setdefault(key, {"flagged": False, "eligible": False})
        state["flagged"] = state["flagged"] or bool(claim["calm_wind_flagged"])
        state["eligible"] = state["eligible"] or (
            claim["claim_type"] in _DIRECTION_TYPES
            and claim["direction_data_present"] is True
        )
    unique_eligible = [state for state in unique.values() if state["eligible"]]
    return {
        "claim_rows": {
            "all": _rate(
                sum(bool(claim["calm_wind_flagged"]) for claim in claim_rows),
                len(claim_rows),
            ),
            "eligible_direction": _rate(
                sum(bool(claim["calm_wind_flagged"]) for claim in eligible_rows),
                len(eligible_rows),
            ),
        },
        "unique_anomaly_exact_text": {
            "all": _rate(
                sum(state["flagged"] for state in unique.values()), len(unique)
            ),
            "eligible_direction": _rate(
                sum(state["flagged"] for state in unique_eligible),
                len(unique_eligible),
            ),
        },
    }


def _model_tables(
    claims: Sequence[Mapping[str, Any]],
    models: tuple[str, ...],
) -> dict[str, Any]:
    phase1: dict[str, dict[str, Any]] = {}
    type_distribution: dict[str, dict[str, int]] = {}
    causal: dict[str, dict[str, int | float | None]] = {}
    skipped: dict[str, dict[str, int | float | None]] = {}
    unclassified: dict[str, dict[str, int | float | None]] = {}
    evidence: dict[str, dict[str, int]] = {}
    for model in models:
        rows = [claim for claim in claims if claim["model"] == model]
        by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for claim in rows:
            by_type[str(claim["claim_type"])].append(claim)
        phase1[model] = {
            claim_type: _rate(
                sum(row["grounding_verdict"] == GROUNDED for row in typed),
                len(typed),
            )
            for claim_type, typed in sorted(by_type.items())
        }
        type_distribution[model] = dict(
            sorted(Counter(str(row["claim_type"]) for row in rows).items())
        )
        causal[model] = _rate(sum(bool(row["causal"]) for row in rows), len(rows))
        skipped[model] = _rate(
            sum(bool(row["skipped_phase2"]) for row in rows), len(rows)
        )
        unclassified[model] = _rate(
            sum(row["claim_type"] == ClaimType.UNCLASSIFIED.value for row in rows),
            len(rows),
        )
        evidence[model] = {
            str(value): count
            for value, count in sorted(
                Counter(int(row["evidence_n"]) for row in rows).items()
            )
        }
    return {
        "phase1_pass_by_type": phase1,
        "claim_type_distribution": type_distribution,
        "causal_share": causal,
        "skipped_phase2": skipped,
        "unclassified": unclassified,
        "evidence_n_distribution": evidence,
    }


def _validate_provenance(
    provenance: object,
    selected_ids: Sequence[str],
) -> dict[str, Any]:
    raw = _required_mapping(provenance, "provenance")
    required = {
        "disposable_b19_not_official",
        "git_commit",
        "db_copy_sha256",
        "selected_anomaly_ids",
        "iteration",
    }
    if set(raw) != required:
        raise FunnelAuditError(
            "missing provenance fields: "
            f"required={sorted(required)}, supplied={sorted(raw)}"
        )
    if raw.get("disposable_b19_not_official") is not True:
        raise FunnelAuditError("B19 provenance must mark the DB disposable")
    commit = raw.get("git_commit")
    db_hash = raw.get("db_copy_sha256")
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise FunnelAuditError("missing provenance fields: invalid git commit")
    if not isinstance(db_hash, str) or _SHA256_RE.fullmatch(db_hash) is None:
        raise FunnelAuditError("missing provenance fields: invalid DB copy SHA-256")
    supplied_ids = raw.get("selected_anomaly_ids")
    if supplied_ids != list(selected_ids):
        raise FunnelAuditError(
            "missing provenance fields: selected anomaly IDs do not match selector"
        )
    iteration = raw.get("iteration")
    if type(iteration) is not int or iteration < 1:
        raise FunnelAuditError("missing provenance fields: iteration must be positive")
    return {
        "disposable_b19_not_official": True,
        "git_commit": commit,
        "db_copy_sha256": db_hash,
        "selected_anomaly_ids": list(selected_ids),
        "iteration": iteration,
    }


def build_funnel_report(
    *,
    ranked_anomalies: object,
    cells: object,
    claims: object,
    b8_observations: object,
    b8_absences: object,
    calm_wind_decisions: object,
    manual_atomicity: object,
    provenance: object,
    models: Sequence[str] = DEFAULT_MODELS,
) -> dict[str, Any]:
    """Build the canonical B19 report from one disposable extraction payload."""
    model_order = tuple(models)
    if model_order != tuple(DEFAULT_MODELS):
        raise FunnelAuditError(
            f"models must equal live harness.DEFAULT_MODELS: {DEFAULT_MODELS!r}"
        )
    selection = select_funnel_anomalies(ranked_anomalies)
    selected_ids = list(selection["selected_anomaly_ids"])
    selected_set = set(selected_ids)
    normalized_provenance = _validate_provenance(provenance, selected_ids)
    normalized_claims = _normalize_claims(
        claims,
        selected_ids=selected_set,
        models=model_order,
    )
    decisions = _decision_units(normalized_claims)
    counting_units = {
        "claim_rows": len(normalized_claims),
        "unique_anomaly_exact_text": len(decisions),
    }
    atomicity = _atomicity_summary(normalized_claims, manual_atomicity)
    cell_audit, cell_stops, token_reviews, costs = _audit_cells(
        cells,
        selected_ids=selected_ids,
        models=model_order,
    )
    b8, b8_stops = _summarize_b8_observations(
        b8_observations,
        b8_absences,
        selected_set,
    )
    calm_context = _validate_calm_decisions(calm_wind_decisions, selected_ids)
    model_tables = _model_tables(normalized_claims, model_order)
    citation_table = summarize_citations(normalized_claims, model_order)
    b17 = summarize_b17_silence(normalized_claims)
    calm = _summarize_calm_wind(normalized_claims)

    hard_stops = [*cell_stops, *b8_stops]
    headline_claims = [
        claim
        for claim in normalized_claims
        if claim["claim_type"] in _HEADLINE_TYPE_VALUES
    ]
    if not headline_claims or all(
        int(claim["evidence_n"]) == 0 for claim in headline_claims
    ):
        hard_stops.append("all headline claims have evidence_n=0")
    for model in model_order:
        model_claims = [
            claim for claim in normalized_claims if claim["model"] == model
        ]
        if not any(claim["grounding_verdict"] == GROUNDED for claim in model_claims):
            hard_stops.append(f"zero grounded claims for model {model}")
        if model_claims and all(
            claim["claim_type"] == ClaimType.UNCLASSIFIED.value
            for claim in model_claims
        ):
            hard_stops.append(f"100% unclassified claims for model {model}")

    review_items = list(token_reviews)
    if atomicity["manual_review_trigger"] is True:
        review_items.append(
            "manual non-self-contained rate > 10%; prompt iteration and a "
            "fresh numbered funnel run are required"
        )
    report = {
        "schema_version": 1,
        "provenance": normalized_provenance,
        "models": list(model_order),
        "selection": selection,
        "counting_units": counting_units,
        "cell_audit": cell_audit,
        "tables": {
            **model_tables,
            "citations": citation_table,
            "atomicity": atomicity,
            "b8_real_location": b8,
            "b17_qualitative_silence": b17,
            "calm_wind": calm,
            "calm_wind_context": calm_context,
        },
        "costs": costs,
        "go_no_go": {
            "status": (
                "hard_stop"
                if hard_stops
                else "review_required"
                if review_items
                else "go"
            ),
            "hard_stops": sorted(set(hard_stops)),
            "review_items": sorted(set(review_items)),
            "review_required_sections": [
                "phase1 asymmetry",
                "causal share",
                "claim-type mix",
                "calm-wind rates",
                "Sentinel-5P silence",
                "atomicity",
            ],
        },
    }
    return report


def canonical_json(report: Mapping[str, Any]) -> str:
    """Return deterministic canonical report JSON with one trailing newline."""
    return json.dumps(
        report,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"


def _format_rate(value: Mapping[str, Any]) -> str:
    denominator = int(value["denominator"])
    if denominator == 0:
        return "n=0"
    numerator = int(value["numerator"])
    fraction = float(value["fraction"])
    return f"{numerator}/{denominator} ({fraction:.2%})"


def _heading(title: str, counts: Mapping[str, int]) -> str:
    return (
        f"### {title} (claim rows={counts['claim_rows']}; "
        f"unique decisions={counts['unique_anomaly_exact_text']})"
    )


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render the deterministic human-review companion to canonical JSON."""
    counts = report["counting_units"]
    selection = report["selection"]
    tables = report["tables"]
    lines = ["# B19 disposable funnel audit", ""]

    lines += [
        _heading("Selection", counts),
        "",
        "| Selection order | Freeze rank | Anomaly | Source | Metric | Reason |",
        "|---:|---:|---|---|---|---|",
    ]
    for index, row in enumerate(selection["selected"], start=1):
        lines.append(
            f"| {index} | {row['freeze_rank']} | {row['anomaly_id']} | "
            f"{row['source']} | {row['metric']} | {row['selection_reason']} |"
        )
    lines += [
        "",
        "Plain top five: " + ", ".join(selection["plain_top_five_anomaly_ids"]),
        "",
        _heading("Cells and truncation", counts),
        "",
        f"Completed cells: {report['cell_audit']['completed_cells']}/"
        f"{report['cell_audit']['expected_cells']}",
        "",
        "| Model | Anomaly | Step | Prompt tokens | Completion tokens | Attempts |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in report["cell_audit"]["steps"]:
        lines.append(
            f"| {row['model']} | {row['anomaly_id']} | {row['step']} | "
            f"{row['prompt_tokens']} | {row['completion_tokens']} | "
            f"{row['attempts']} |"
        )

    lines += [
        "",
        _heading("Local/cloud prompt ratios", counts),
        "",
        "| Anomaly | Local model | Cloud model | Local prompt | Cloud prompt | Local/cloud ratio |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in report["cell_audit"]["local_cloud_prompt_ratios"]:
        ratio = (
            f"{row['local_to_cloud_ratio']:.4f}"
            if row["local_to_cloud_ratio"] is not None
            else "n/a"
        )
        lines.append(
            f"| {row['anomaly_id']} | {row['local_model']} | "
            f"{row['cloud_model']} | {row['local_prompt_tokens']} | "
            f"{row['cloud_prompt_tokens']} | {ratio} |"
        )

    lines += [
        "",
        _heading("Phase-1 pass by model and claim type", counts),
        "",
        "| Model | Claim type | Grounded |",
        "|---|---|---:|",
    ]
    for model in report["models"]:
        for claim_type, rate in tables["phase1_pass_by_type"][model].items():
            lines.append(f"| {model} | {claim_type} | {_format_rate(rate)} |")

    lines += [
        "",
        _heading("Citation outcomes", counts),
        "",
        "| Model | Omission | Nonblank citation claims | All-blank claims | blank-only | unrecognized-source | recognized-but-absent-from-context | Multiple reasons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in report["models"]:
        row = tables["citations"][model]
        lines.append(
            f"| {model} | {_format_rate(row['omission_rate'])} | "
            f"{row['claims_with_nonblank_citation']} | "
            f"{row['all_blank_citation_claims']} | "
            f"{_format_rate(row['reason_rates']['blank-only'])} | "
            f"{_format_rate(row['reason_rates']['unrecognized-source'])} | "
            f"{_format_rate(row['reason_rates']['recognized-but-absent-from-context'])} | "
            f"{row['multiple_reasons_claim_count']} |"
        )

    lines += [
        "",
        _heading("Citation failure strings", counts),
        "",
        "| Model | Anomaly | Claim | Citation index | Citation | Reason |",
        "|---|---|---|---:|---|---|",
    ]
    citation_failure_n = 0
    for model in report["models"]:
        for row in tables["citations"][model]["failure_records"]:
            citation_failure_n += 1
            lines.append(
                f"| {_markdown_cell(model)} | "
                f"{_markdown_cell(row['anomaly_id'])} | "
                f"{_markdown_cell(row['claim_ref'])} | "
                f"{row['citation_index']} | "
                f"{_markdown_cell(row['citation'])} | "
                f"{_markdown_cell(row['reason'])} |"
            )
    if citation_failure_n == 0:
        lines.append("| n=0 | n=0 | n=0 | n=0 | n=0 | n=0 |")

    lines += [
        "",
        _heading("Claim types and Phase-2 routing", counts),
        "",
        "| Model | Unclassified | Skipped Phase 2 | Causal |",
        "|---|---:|---:|---:|",
    ]
    for model in report["models"]:
        lines.append(
            f"| {model} | {_format_rate(tables['unclassified'][model])} | "
            f"{_format_rate(tables['skipped_phase2'][model])} | "
            f"{_format_rate(tables['causal_share'][model])} |"
        )

    lines += [
        "",
        _heading("Claim type distribution", counts),
        "",
        "| Model | Claim type | Claim rows |",
        "|---|---|---:|",
    ]
    for model in report["models"]:
        distribution = tables["claim_type_distribution"][model]
        if not distribution:
            lines.append(f"| {model} | n=0 | n=0 |")
        for claim_type, count in distribution.items():
            lines.append(f"| {model} | {claim_type} | {count} |")

    lines += [
        "",
        _heading("Evidence count distribution", counts),
        "",
        "| Model | evidence_n | Claim rows |",
        "|---|---:|---:|",
    ]
    for model in report["models"]:
        distribution = tables["evidence_n_distribution"][model]
        if not distribution:
            lines.append(f"| {model} | n=0 | n=0 |")
        for evidence_n, count in distribution.items():
            lines.append(f"| {model} | {evidence_n} | {count} |")

    atomicity = tables["atomicity"]
    lines += [
        "",
        _heading("Atomicity", counts),
        "",
        f"Manual non-self-contained: "
        f"{_format_rate(atomicity['manual_non_self_contained'])}",
        "",
        f"Screen/manual disagreement: "
        f"{_format_rate(atomicity['screen_manual_disagreement'])}",
    ]

    b17 = tables["b17_qualitative_silence"]
    lines += [
        "",
        _heading("B17 qualitative silence", counts),
        "",
        "| Unit | Qualitative | Matched baseline n<3 | No data in window | Both |",
        "|---|---:|---:|---:|---:|",
    ]
    for unit in ("claim_rows", "unique_anomaly_exact_text"):
        row = b17[unit]
        lines.append(
            f"| {unit} | {row['qualitative_concentration']} | "
            f"{row['matched_baseline_n_lt_3']} | {row['no_data_in_window']} | "
            f"{row['both']} |"
        )

    calm = tables["calm_wind"]
    lines += [
        "",
        _heading("Calm-wind flags", counts),
        "",
        "| Unit | All claims | Eligible direction claims |",
        "|---|---:|---:|",
    ]
    for unit in ("claim_rows", "unique_anomaly_exact_text"):
        lines.append(
            f"| {unit} | {_format_rate(calm[unit]['all'])} | "
            f"{_format_rate(calm[unit]['eligible_direction'])} |"
        )

    lines += [
        "",
        _heading("Calm-wind cutoffs and guard state", counts),
        "",
        "| Anomaly | Source | Raw cutoff | Effective cutoff | Event speed | Enabled | Calm | Floor status |",
        "|---|---|---:|---:|---:|---|---|---|",
    ]
    for row in tables["calm_wind_context"]["decisions"]:
        lines.append(
            f"| {row['anomaly_id']} | {row['source']} | "
            f"{row['raw_cutoff_ms']} | {row['effective_cutoff_ms']} | "
            f"{row['event_speed_ms']} | "
            f"{'yes' if row['guard_enabled'] else 'no'} | {row['calm']} | "
            f"{row['floor_status']} |"
        )

    lines += [
        "",
        _heading("B8 real-location age gates", counts),
        "",
        "| Source | Gate minutes | Silenced | Hourly hard stop |",
        "|---|---:|---:|---|",
    ]
    for source, row in tables["b8_real_location"]["sources"].items():
        rate = {
            "numerator": row["stale_count"],
            "denominator": row["denominator"],
            "fraction": row["fraction_silenced"],
        }
        lines.append(
            f"| {source} | {row['gate_minutes']} | {_format_rate(rate)} | "
            f"{'yes' if row['hourly_hard_stop'] else 'no'} |"
        )

    lines += [
        "",
        _heading("B8 structural absences", counts),
        "",
        "| Anomaly | Source | Metric | Reason |",
        "|---|---|---|---|",
    ]
    absences = tables["b8_real_location"]["structural_absences"]
    if not absences:
        lines.append("| n=0 | n=0 | n=0 | n=0 |")
    for row in absences:
        lines.append(
            f"| {row['anomaly_id']} | {row['source']} | "
            f"{row['metric']} | {row['reason']} |"
        )

    lines += [
        "",
        _heading("Costs", counts),
        "",
        f"Pricing status: {report['costs']['status']}",
        "",
        "| Model | Prompt tokens | Completion tokens | Estimated USD |",
        "|---|---:|---:|---:|",
    ]
    for model in report["models"]:
        row = report["costs"]["per_model"][model]
        estimate = (
            f"${row['estimated_cost_usd']:.6f}"
            if row["estimated_cost_usd"] is not None
            else "n/a"
        )
        lines.append(
            f"| {model} | {row['prompt_tokens']} | "
            f"{row['completion_tokens']} | {estimate} |"
        )
    lines += [
        "",
        _heading("Go/no-go", counts),
        "",
        f"Status: {report['go_no_go']['status']}",
    ]
    hard_stops = report["go_no_go"]["hard_stops"]
    lines.append("Hard stops: " + ("; ".join(hard_stops) if hard_stops else "none"))
    review_items = report["go_no_go"]["review_items"]
    lines.append(
        "Triggered review items: "
        + ("; ".join(review_items) if review_items else "none")
    )
    return "\n".join(lines) + "\n"


def write_iteration_reports(
    output_dir: Path,
    report: Mapping[str, Any],
) -> dict[str, Path]:
    """Write one numbered JSON/Markdown pair without replacing prior runs."""
    provenance = _required_mapping(report.get("provenance"), "report provenance")
    iteration = provenance.get("iteration")
    if type(iteration) is not int or iteration < 1:
        raise FunnelAuditError("report iteration must be a positive integer")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"b19-funnel-iteration-{iteration:03d}"
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError(
            f"B19 iteration {iteration} is preserved and cannot be overwritten"
        )
    with json_path.open("x", encoding="utf-8") as handle:
        handle.write(canonical_json(report))
    with markdown_path.open("x", encoding="utf-8") as handle:
        handle.write(render_markdown(report))
    return {"json": json_path, "markdown": markdown_path}
