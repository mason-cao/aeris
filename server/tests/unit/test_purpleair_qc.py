"""B7 PurpleAir time-aware eligibility and saturation QC."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

import app.detection.enrichment as enrichment_module
import app.detection.run as detection_module
import app.llm.corroboration as corroboration_module
from app.db.models import Anomaly, DataPoint
from app.provenance.purpleair_qc import (
    DEFAULT_QC_PARAMETERS,
    PurpleAirReading,
    evaluate_purpleair_qc,
)


START = datetime(2026, 6, 1, tzinfo=timezone.utc)
END = START + timedelta(days=4)
PEERS = tuple(f"peer-{index:02d}" for index in range(12))


def _hourly_values(
    entity_id: str,
    values: list[float],
    *,
    start: datetime = START,
) -> list[PurpleAirReading]:
    return [
        PurpleAirReading(
            entity_id=entity_id,
            timestamp=start + timedelta(hours=index),
            value=value,
        )
        for index, value in enumerate(values)
    ]


def _network(
    candidate_values: list[float],
    *,
    peer_values: list[float] | None = None,
    peer_ids: tuple[str, ...] = PEERS,
) -> list[PurpleAirReading]:
    peers = peer_values or [10.0] * len(candidate_values)
    readings = _hourly_values("candidate", candidate_values)
    for peer_id in peer_ids:
        readings.extend(_hourly_values(peer_id, peers))
    return readings


def test_multi_day_tenfold_fault_is_excluded_but_recovery_is_retained() -> None:
    candidate = [10.0] * 24 + [100.0] * 48 + [10.0] * 24
    result = evaluate_purpleair_qc(_network(candidate), START, END)

    assert not result.is_eligible("candidate", START + timedelta(hours=48))
    assert result.is_eligible("candidate", START + timedelta(hours=95))
    assert any(segment.entity_id == "candidate" for segment in result.segments)


def test_uniform_network_wide_fivefold_event_is_not_excluded() -> None:
    values = [10.0] * 24 + [50.0] * 48 + [10.0] * 24
    result = evaluate_purpleair_qc(
        _network(values, peer_values=values), START, END
    )

    assert all(
        result.is_eligible("candidate", START + timedelta(hours=hour))
        for hour in range(24, 72)
    )


def test_three_hour_fourfold_single_sensor_plume_is_not_excluded() -> None:
    values = [10.0] * 96
    values[40:43] = [40.0, 40.0, 40.0]
    result = evaluate_purpleair_qc(_network(values), START, END)

    assert all(
        result.is_eligible("candidate", START + timedelta(hours=hour))
        for hour in range(40, 43)
    )


def test_primary_ratio_and_absolute_boundaries_are_inclusive() -> None:
    ratio_boundary = evaluate_purpleair_qc(
        _network([50.0] * 96, peer_values=[10.0] * 96), START, END
    )
    absolute_boundary = evaluate_purpleair_qc(
        _network([20.0] * 96, peer_values=[4.0] * 96), START, END
    )
    below_absolute = evaluate_purpleair_qc(
        _network([19.999] * 96, peer_values=[3.0] * 96), START, END
    )

    midpoint = START + timedelta(hours=48)
    assert not ratio_boundary.is_eligible("candidate", midpoint)
    assert not absolute_boundary.is_eligible("candidate", midpoint)
    assert below_absolute.is_eligible("candidate", midpoint)


def test_zero_network_median_satisfies_ratio_condition() -> None:
    result = evaluate_purpleair_qc(
        _network([20.0] * 96, peer_values=[0.0] * 96), START, END
    )

    assert not result.is_eligible("candidate", START + timedelta(hours=48))


def test_saturation_boundary_excludes_below_network_exception() -> None:
    values = [10.0] * 96
    values[48] = DEFAULT_QC_PARAMETERS.saturation_ug_m3
    result = evaluate_purpleair_qc(_network(values), START, END)
    key = ("candidate", START + timedelta(hours=48))

    assert not result.is_eligible(*key)
    assert "saturation" in result.exclusion_reasons[key]


def test_network_extreme_boundary_retains_saturated_reading() -> None:
    values = [10.0] * 96
    values[48] = 500.0
    peers = [10.0] * 96
    peers[48] = 100.0
    result = evaluate_purpleair_qc(
        _network(values, peer_values=peers), START, END
    )

    assert result.is_eligible("candidate", START + timedelta(hours=48))


def test_sparse_network_is_kept_and_marked_unevaluated() -> None:
    result = evaluate_purpleair_qc(
        _network([100.0] * 96, peer_ids=PEERS[:9]), START, END
    )
    center = START + timedelta(hours=48)

    assert result.is_eligible("candidate", center)
    assert result.unevaluated_windows[("candidate", center)] == "peer_sensor_floor"
    assert result.unevaluated_window_count > 0


def _point(*, entity_id: str, timestamp: datetime, value: float) -> DataPoint:
    return DataPoint(
        timestamp=timestamp,
        lat=29.7604,
        lon=-95.3698,
        metric="pm25",
        value=value,
        unit="ug/m3",
        source="purpleair",
        source_entity_id=entity_id,
        raw_json=None,
    )


def _scoring_summary(timestamp: datetime) -> dict[str, Any]:
    def _entity(entity_id: str, values: list[float]) -> dict[str, Any]:
        return {
            "entity_id": entity_id,
            "lat": 29.7604,
            "lon": -95.3698,
            "distance_km": 1.0,
            "n_points": len(values),
            "series": [
                [
                    (timestamp - timedelta(hours=len(values) - 1 - index)).isoformat(),
                    value,
                ]
                for index, value in enumerate(values)
            ],
        }

    bad = _entity("bad", [10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    good = _entity("good", [60.0, 50.0, 40.0, 30.0, 20.0, 10.0])
    return {
        "anomaly": {"timestamp": timestamp.isoformat()},
        "sources": {
            "purpleair": {
                "metrics": {
                    "pm25": {
                        "unit": "ug/m3",
                        "nearest_in_time": {
                            "t": timestamp.isoformat(),
                            "v": 60.0,
                            "entity_id": "bad",
                        },
                        "entities": [bad, good],
                    }
                }
            }
        },
    }


def test_same_predicate_filters_detection_enrichment_and_scoring(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = START + timedelta(hours=48)

    def eligible(entity_id: str, observed_at: datetime) -> bool:
        return entity_id != "bad"

    monkeypatch.setattr(
        detection_module, "purpleair_reading_is_eligible", eligible
    )
    monkeypatch.setattr(
        enrichment_module, "purpleair_reading_is_eligible", eligible
    )
    monkeypatch.setattr(
        corroboration_module, "purpleair_reading_is_eligible", eligible
    )

    points = [
        _point(entity_id="bad", timestamp=timestamp, value=60.0),
        _point(entity_id="good", timestamp=timestamp, value=10.0),
    ]
    groups = detection_module.group_points_by_series(points)
    assert {key.source_entity_id for key in groups} == {"good"}

    anomaly = Anomaly(
        timestamp=timestamp,
        lat=29.7604,
        lon=-95.3698,
        metric="pm25",
        source="openaq",
        value=30.0,
        expected_value=10.0,
        z_score=4.0,
        methods_triggered=["zscore"],
        severity="moderate",
    )
    summary = enrichment_module.build_cross_source_summary(
        anomaly,
        points,
        window_start=timestamp - timedelta(hours=1),
        window_end=timestamp + timedelta(hours=1),
    )
    entities = summary["sources"]["purpleair"]["metrics"]["pm25"]["entities"]
    assert [entity["entity_id"] for entity in entities] == ["good"]

    verdicts, note = corroboration_module.score_temporal_pattern(
        "PM2.5 concentrations rose through the morning",
        _scoring_summary(timestamp),
    )
    assert verdicts["purpleair"] == corroboration_module.CONTRADICTING
    assert "observed=down claimed=up" in note

