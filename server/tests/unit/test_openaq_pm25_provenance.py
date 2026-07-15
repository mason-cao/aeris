"""B6 OpenAQ PM2.5 provider/instrument provenance enforcement."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest

from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    score_concentration_elevation,
    score_temporal_pattern,
)
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    load_openaq_pm25_fixture,
    verified_monitor_entity_ids,
)


def _entities() -> list[dict[str, Any]]:
    fixture = load_openaq_pm25_fixture()
    return cast(list[dict[str, Any]], fixture["entities"])


def _entity_id(*, provider: str, classification: str) -> str:
    return min(
        str(row["entity_id"])
        for row in _entities()
        if row["provider"] == provider and row["classification"] == classification
    )


def _pm25_summary(entity_values: Mapping[str, float]) -> dict[str, Any]:
    anomaly_timestamp = "2026-06-15T12:00:00+00:00"
    entities = [
        {
            "entity_id": entity_id,
            "lat": 29.76,
            "lon": -95.37,
            "distance_km": index + 0.1,
            "n_points": 1,
            "series": [[anomaly_timestamp, value]],
        }
        for index, (entity_id, value) in enumerate(entity_values.items())
    ]
    pooled_nearest_id, pooled_nearest_value = next(iter(entity_values.items()))
    return {
        "anomaly": {
            "timestamp": anomaly_timestamp,
            "lat": 29.76,
            "lon": -95.37,
        },
        "sources": {
            "openaq": {
                "metrics": {
                    "pm25": {
                        "unit": "ug/m3",
                        "n_points": len(entities),
                        "n_entities": len(entities),
                        "nearest_in_time": {
                            "t": anomaly_timestamp,
                            "v": pooled_nearest_value,
                            "entity_id": pooled_nearest_id,
                            "distance_km": 0.1,
                            "dt_minutes": 0.0,
                        },
                        "entities": entities,
                    }
                }
            }
        },
    }


def _pm25_series_summary(
    entity_values: Mapping[str, list[float]],
) -> dict[str, Any]:
    timestamps = [
        f"2026-06-15T{hour:02d}:00:00+00:00" for hour in range(7, 13)
    ]
    entities = [
        {
            "entity_id": entity_id,
            "lat": 29.76,
            "lon": -95.37,
            "distance_km": index + 0.1,
            "n_points": len(values),
            "series": [list(pair) for pair in zip(timestamps, values, strict=True)],
        }
        for index, (entity_id, values) in enumerate(entity_values.items())
    ]
    first_id, first_values = next(iter(entity_values.items()))
    return {
        "anomaly": {"timestamp": timestamps[-1], "lat": 29.76, "lon": -95.37},
        "sources": {
            "openaq": {
                "metrics": {
                    "pm25": {
                        "unit": "ug/m3",
                        "nearest_in_time": {
                            "t": timestamps[-1],
                            "v": first_values[-1],
                            "entity_id": first_id,
                        },
                        "entities": entities,
                    }
                }
            }
        },
    }


def test_fixture_matches_snapshot_audit_counts() -> None:
    fixture = load_openaq_pm25_fixture()

    assert fixture["snapshot_sha256"] == LOCKED_SNAPSHOT_SHA256
    assert fixture["study_window"] == {
        "start": "2026-06-01T00:00:00Z",
        "end_exclusive": "2026-07-13T00:00:00Z",
    }
    assert fixture["audit_counts"] == {
        "inline_by_provider": {
            "AirGradient": {"rows": 17627, "entities": 26},
            "AirNow": {"rows": 3227, "entities": 12},
            "Clarity": {"rows": 16969, "entities": 55},
        },
        "archive_without_inline_metadata": {"rows": 21861, "entities": 92},
        "archive_mapped_by_provider": {
            "AirGradient": {"rows": 3310, "entities": 25},
            "AirNow": {"rows": 1786, "entities": 12},
            "Clarity": {"rows": 16765, "entities": 55},
        },
        "resolved_by_class": {
            "verified_monitor": {"rows": 5013, "entities": 12},
            "non_monitor_sensor": {"rows": 54671, "entities": 81},
            "unmappable_archive": {"rows": 0, "entities": 0},
        },
    }
    assert fixture["unmappable_archive_entities"] == []
    assert fixture["instrument_metadata_revisions"] == [
        {
            "entity_id": "16648860",
            "provider": "AirGradient",
            "is_monitor": False,
            "observed_instrument_name_sets": [
                {"instrument_names": ["N/A"], "rows": 1},
                {
                    "instrument_names": ["Unknown AirGradient Sensor"],
                    "rows": 553,
                },
            ],
        }
    ]
    assert len(_entities()) == 93
    assert len(verified_monitor_entity_ids()) == 12


@pytest.mark.parametrize("provider", ["Clarity", "AirGradient"])
def test_non_monitor_entity_alone_makes_openaq_pm_leg_silent(provider: str) -> None:
    non_monitor_id = _entity_id(
        provider=provider, classification="non_monitor_sensor"
    )
    summary = _pm25_summary({non_monitor_id: 100.0})

    verdicts, note = score_concentration_elevation(
        "PM2.5 exceeded 50 ug/m3", summary
    )

    assert verdicts["openaq"] == SILENT
    assert "no verified-monitor pm25 observation" in note


@pytest.mark.parametrize("provider", ["Clarity", "AirGradient"])
def test_non_monitor_cannot_override_verified_monitor_vote(provider: str) -> None:
    non_monitor_id = _entity_id(
        provider=provider, classification="non_monitor_sensor"
    )
    monitor_id = _entity_id(provider="AirNow", classification="verified_monitor")
    summary = _pm25_summary({non_monitor_id: 100.0, monitor_id: 20.0})

    verdicts, note = score_concentration_elevation(
        "PM2.5 exceeded 50 ug/m3", summary
    )

    assert verdicts["openaq"] == CONTRADICTING
    assert f"nearest=20.0" in note


@pytest.mark.parametrize("provider", ["Clarity", "AirGradient"])
def test_non_monitor_series_cannot_enter_descriptive_pm_leg(provider: str) -> None:
    non_monitor_id = _entity_id(
        provider=provider, classification="non_monitor_sensor"
    )
    monitor_id = _entity_id(provider="AirNow", classification="verified_monitor")
    summary = _pm25_series_summary(
        {
            non_monitor_id: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            monitor_id: [60.0, 50.0, 40.0, 30.0, 20.0, 10.0],
        }
    )

    verdicts, note = score_temporal_pattern(
        "PM2.5 concentrations rose through the morning", summary
    )

    assert verdicts["openaq"] == CONTRADICTING
    assert "observed=down claimed=up" in note
