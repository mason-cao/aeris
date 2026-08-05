"""B18 freeze-day detector and trigger-series provenance review."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from app.db.models import Anomaly
from app.provenance.purpleair_qc import LOCKED_SNAPSHOT_SHA256
from app.eval.freeze import (
    FreezeResult,
    _format_result,
    b18_review_payload,
    fixture_payload,
)


T0 = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
SNAPSHOT_SHA256 = LOCKED_SNAPSHOT_SHA256
CODE_COMMIT = "a" * 40


def _availability(
    *,
    stl_ran: bool = True,
    isolation_forest_ran: bool = True,
) -> dict[str, dict[str, bool | str | None]]:
    return {
        "zscore": {"ran": True, "skip_code": None, "detail": None},
        "stl": {
            "ran": stl_ran,
            "skip_code": None if stl_ran else "cadence_too_coarse",
            "detail": (
                None
                if stl_ran
                else "stl skipped: cadence too coarse for diurnal decomposition"
            ),
        },
        "isolation_forest": {
            "ran": isolation_forest_ran,
            "skip_code": (
                None if isolation_forest_ran else "missing_complete_gfs_aux"
            ),
            "detail": None,
        },
    }


def _anomaly(
    *,
    source: str = "openaq",
    metric: str = "ozone",
    entity_id: str | None = "station-1",
    methods: list[str] | None = None,
    availability: dict[str, dict[str, bool | str | None]] | None = None,
    timestamp: datetime = T0,
) -> Anomaly:
    return Anomaly(
        id=uuid.uuid4(),
        timestamp=timestamp,
        lat=29.76,
        lon=-95.37,
        metric=metric,
        source=source,
        source_entity_id=entity_id,
        detector_availability_json=(
            _availability() if availability is None else availability
        ),
        value=100.0,
        expected_value=20.0,
        z_score=4.0,
        methods_triggered=methods or ["zscore"],
        severity="moderate",
    )


def _result(selected: list[Anomaly]) -> FreezeResult:
    return FreezeResult(
        window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
        window_end=datetime(2026, 7, 13, tzinfo=timezone.utc),
        top_n=50,
        n_anomalies=len(selected),
        n_events=len(selected),
        selected=selected,
        event_sizes={anomaly.id: index + 1 for index, anomaly in enumerate(selected)},
        missing_enrichment=[],
    )


def test_review_reports_exact_series_detector_states_and_ran_without_trigger() -> None:
    first = _anomaly(
        methods=["zscore"],
        availability=_availability(stl_ran=True, isolation_forest_ran=False),
    )
    second = _anomaly(
        methods=["zscore", "isolation_forest"],
        availability=_availability(stl_ran=False, isolation_forest_ran=True),
        timestamp=T0.replace(hour=15),
    )
    third = _anomaly(
        source="tceq",
        metric="no2",
        entity_id="cams-7",
        methods=["zscore", "stl"],
        availability=_availability(stl_ran=True, isolation_forest_ran=False),
    )

    review = b18_review_payload(_result([first, second, third]))

    assert review["provenance_complete"] is True
    assert review["missing_provenance_anomaly_ids"] == []
    assert review["trigger_series_counts"] == {
        "openaq/ozone/station-1": 2,
        "tceq/no2/cams-7": 1,
    }
    assert review["repeated_trigger_series"] == [
        {
            "source": "openaq",
            "metric": "ozone",
            "source_entity_id": "station-1",
            "selected_count": 2,
            "anomaly_ids": [str(first.id), str(second.id)],
        }
    ]
    assert review["detector_availability"]["stl"] == {
        "ran": 2,
        "skipped": 1,
        "missing": 0,
        "skip_reasons": {"cadence_too_coarse": 1},
    }
    assert review["detector_availability"]["isolation_forest"] == {
        "ran": 1,
        "skipped": 2,
        "missing": 0,
        "skip_reasons": {"missing_complete_gfs_aux": 2},
    }
    assert review["triggered_method_patterns"] == {
        "isolation_forest+zscore": 1,
        "stl+zscore": 1,
        "zscore": 1,
    }
    assert review["selected"][0]["source_entity_id"] == "station-1"
    assert review["selected"][0]["event_size"] == 1
    assert review["selected"][1]["event_size"] == 2
    assert review["selected"][0]["detector_availability"]["stl"]["ran"] is True
    assert review["selected"][0]["methods_triggered"] == ["zscore"]


def test_review_is_explicit_for_empty_selection() -> None:
    review = b18_review_payload(_result([]))

    assert review["provenance_complete"] is True
    assert review["decision_status"] == "pending_freeze_day_review"
    assert review["decision_rationale"] is None
    assert review["selected"] == []
    assert review["trigger_series_counts"] == {}
    assert review["repeated_trigger_series"] == []
    assert review["detector_availability"] == {
        detector: {
            "ran": 0,
            "skipped": 0,
            "missing": 0,
            "skip_reasons": {},
        }
        for detector in ("isolation_forest", "stl", "zscore")
    }


def test_missing_provenance_is_reported_and_real_fixture_refuses() -> None:
    anomaly = _anomaly()
    anomaly.source_entity_id = None
    anomaly.detector_availability_json = None
    result = _result([anomaly])

    review = b18_review_payload(result)

    assert review["provenance_complete"] is False
    assert review["missing_provenance_anomaly_ids"] == [str(anomaly.id)]
    with pytest.raises(ValueError, match="B18.*provenance"):
        fixture_payload(
            result,
            snapshot_sha256=SNAPSHOT_SHA256,
            code_commit=CODE_COMMIT,
        )


@pytest.mark.parametrize(
    "malformed",
    [
        {},
        {
            "zscore": {"ran": True, "skip_code": None, "detail": None},
            "stl": {"ran": "yes", "skip_code": None, "detail": None},
            "isolation_forest": {
                "ran": False,
                "skip_code": None,
                "detail": None,
            },
        },
    ],
)
def test_malformed_detector_provenance_is_incomplete(
    malformed: dict[str, object],
) -> None:
    anomaly = _anomaly()
    anomaly.detector_availability_json = malformed

    review = b18_review_payload(_result([anomaly]))

    assert review["provenance_complete"] is False
    assert review["missing_provenance_anomaly_ids"] == [str(anomaly.id)]
    assert all(
        counts["missing"] == 1
        for counts in review["detector_availability"].values()
    )


def test_review_normalizes_naive_and_aware_timestamp_identically() -> None:
    anomaly = _anomaly(timestamp=T0)
    result = _result([anomaly])
    aware = b18_review_payload(result)["selected"][0]["timestamp"]

    anomaly.timestamp = T0.replace(tzinfo=None)
    naive = b18_review_payload(result)["selected"][0]["timestamp"]

    assert naive == aware == "2026-07-02T12:00:00+00:00"


def test_fixture_and_text_include_complete_b18_review() -> None:
    anomaly = _anomaly()
    result = _result([anomaly])

    payload = fixture_payload(
        result,
        snapshot_sha256=SNAPSHOT_SHA256,
        code_commit=CODE_COMMIT,
        b18_decision="accept_unstratified",
        b18_rationale="No source, method, or exact-series pathology observed.",
    )
    rendered = _format_result(result)

    assert payload["freeze_day_review"] == b18_review_payload(
        result,
        decision_status="accept_unstratified",
        decision_rationale=(
            "No source, method, or exact-series pathology observed."
        ),
    )
    assert "Detector availability" in rendered
    assert "Repeated trigger series" in rendered
    assert "openaq/ozone/station-1" in rendered
    assert payload["freeze_day_review"]["decision_status"] == (
        "accept_unstratified"
    )
    assert payload["freeze_day_review"]["decision_rationale"] == (
        "No source, method, or exact-series pathology observed."
    )


def test_real_fixture_requires_reviewed_decision_and_rationale() -> None:
    result = _result([_anomaly()])

    with pytest.raises(ValueError, match="B18.*decision"):
        fixture_payload(
            result,
            snapshot_sha256=SNAPSHOT_SHA256,
            code_commit=CODE_COMMIT,
        )
    with pytest.raises(ValueError, match="B18.*rationale"):
        fixture_payload(
            result,
            snapshot_sha256=SNAPSHOT_SHA256,
            code_commit=CODE_COMMIT,
            b18_decision="accept_unstratified",
            b18_rationale="   ",
        )


def test_stratify_decision_requires_a_stratified_selection() -> None:
    # The rule is implemented now, so this no longer refuses the decision
    # outright. What it must still refuse is a fixture whose declared decision
    # and executed selection disagree.
    with pytest.raises(ValueError, match="does not match the selection"):
        fixture_payload(
            _result([_anomaly()]),
            snapshot_sha256=SNAPSHOT_SHA256,
            code_commit=CODE_COMMIT,
            b18_decision="stratify",
            b18_rationale="The dry run shows source skew.",
        )
