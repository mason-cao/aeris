"""B6 OpenAQ PM2.5 provider/instrument provenance enforcement."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from pathlib import Path
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
    STUDY_END_EXCLUSIVE,
    STUDY_START,
    derive_openaq_pm25_fixture,
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
        "start": STUDY_START,
        "end_exclusive": STUDY_END_EXCLUSIVE,
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


def _synthetic_database(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE data_points (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                source_entity_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        rows: list[tuple[str, str, str, str, str, str]] = []
        row_id = 0

        def add(
            metric: str,
            entity_id: str,
            payload: dict[str, object],
        ) -> None:
            nonlocal row_id
            row_id += 1
            rows.append(
                (
                    str(row_id),
                    "openaq",
                    metric,
                    entity_id,
                    f"2026-06-15T{row_id:02d}:00:00Z",
                    json.dumps(payload),
                )
            )

        def location(
            provider: str,
            is_monitor: bool,
            instrument: str,
        ) -> dict[str, object]:
            return {
                "location": {
                    "provider": {"name": provider},
                    "isMonitor": is_monitor,
                    "instruments": [{"name": instrument}],
                }
            }

        for metric in ("ozone", "pm10", "pm25"):
            entity_id = f"{metric}-monitor"
            add(metric, entity_id, location("AirNow", True, "Government Monitor"))
            add(metric, entity_id, {"archive": {"path": "historical"}})
        add("pm25", "clarity", location("Clarity", False, "Clarity Sensor"))
        add(
            "pm25",
            "airgradient",
            location("AirGradient", False, "N/A"),
        )
        add(
            "pm25",
            "airgradient",
            location("AirGradient", False, "Unknown AirGradient Sensor"),
        )
        add("pm25", "archive-only", {"archive": {"path": "unmapped"}})
        connection.executemany(
            "INSERT INTO data_points VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_multimetric_derivation_is_read_only_deterministic_and_exact(
    tmp_path: Path,
) -> None:
    database = tmp_path / "synthetic.db"
    expected_hash = _synthetic_database(database)

    first = derive_openaq_pm25_fixture(database, expected_sha256=expected_hash)
    second = derive_openaq_pm25_fixture(database, expected_sha256=expected_hash)

    assert first == second
    assert hashlib.sha256(database.read_bytes()).hexdigest() == expected_hash
    assert first["schema_version"] == 2
    assert first["eligible_entity_counts"] == {
        "ozone": 1,
        "pm10": 1,
        "pm25": 1,
    }
    assert first["eligible_entity_ids_by_metric"] == {
        "ozone": ["ozone-monitor"],
        "pm10": ["pm10-monitor"],
        "pm25": ["pm25-monitor"],
    }
    assert first["unmappable_archive_entities_by_metric"] == [
        {"metric": "pm25", "entity_id": "archive-only", "archive_rows": 1}
    ]
    assert first["instrument_metadata_revisions_by_metric"] == [
        {
            "metric": "pm25",
            "entity_id": "airgradient",
            "provider": "AirGradient",
            "is_monitor": False,
            "observed_instrument_name_sets": [
                {"instrument_names": ["N/A"], "rows": 1},
                {
                    "instrument_names": ["Unknown AirGradient Sensor"],
                    "rows": 1,
                },
            ],
        }
    ]
    assert first["audit_counts_by_metric"]["pm25"]["resolved_by_class"] == {
        "verified_monitor": {"rows": 2, "entities": 1},
        "non_monitor_sensor": {"rows": 3, "entities": 2},
        "unmappable_archive": {"rows": 1, "entities": 1},
    }


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
    assert "nearest=20.0" in note


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
