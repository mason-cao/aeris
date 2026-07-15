"""B8 observation-age gates and label-free cadence empirics."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval.observation_age_empirics import (
    AgeObservation,
    STUDY_END_EXCLUSIVE,
    STUDY_START,
    empirical_anchor_centers,
    main,
    metric_age_empirics,
    nearest_observation_age,
    render_markdown,
    run_empirics,
    write_report,
)
from app.llm.observation_age import (
    DEFAULT_OBSERVATION_AGE_GATES,
    assess_observation_age,
)


def test_declared_source_gates_and_inclusive_boundary() -> None:
    assert DEFAULT_OBSERVATION_AGE_GATES.to_dict() == {
        "asos": 90.0,
        "epa_aqs": 90.0,
        "noaa_gfs": 360.0,
        "openaq": 90.0,
        "openweather": 90.0,
        "purpleair": 90.0,
        "sentinel5p": 720.0,
        "tceq": 90.0,
    }

    at_gate = assess_observation_age("tceq", 90.0)
    past_gate = assess_observation_age("tceq", 91.0)

    assert at_gate.votes is True
    assert at_gate.reason is None
    assert past_gate.votes is False
    assert past_gate.reason == "stale"


@pytest.mark.parametrize("value", [None, "", -1.0, math.nan, math.inf])
def test_missing_or_invalid_age_is_explicitly_silent(value: object) -> None:
    decision = assess_observation_age("openaq", value)

    assert decision.votes is False
    assert decision.reason == "missing_or_invalid"
    assert decision.gate_minutes == 90.0


def test_anchor_population_reuses_b2_timezone_stable_full_windows() -> None:
    aware = empirical_anchor_centers(STUDY_START, STUDY_END_EXCLUSIVE)
    naive = empirical_anchor_centers(
        STUDY_START.replace(tzinfo=None),
        STUDY_END_EXCLUSIVE.replace(tzinfo=None),
    )

    assert aware == naive
    assert len(aware) == 936
    assert aware[0] == datetime(2026, 6, 2, 12, tzinfo=UTC)
    assert aware[-1] == datetime(2026, 7, 11, 11, tzinfo=UTC)


def test_nearest_age_includes_endpoints_and_breaks_time_ties_by_distance() -> None:
    anchor = datetime(2026, 6, 5, 12, tzinfo=UTC)
    observations = [
        AgeObservation(
            entity_id="farther",
            timestamp=anchor - timedelta(hours=36),
            lat=29.80,
            lon=-95.37,
        ),
        AgeObservation(
            entity_id="nearer",
            timestamp=anchor + timedelta(hours=36),
            lat=29.761,
            lon=-95.37,
        ),
        AgeObservation(
            entity_id="outside-radius",
            timestamp=anchor,
            lat=31.0,
            lon=-95.37,
        ),
    ]

    nearest = nearest_observation_age(
        observations,
        anchor,
        anchor_lat=29.7604,
        anchor_lon=-95.3698,
        radius_km=50.0,
    )

    assert nearest is not None
    assert nearest.entity_id == "nearer"
    assert nearest.dt_minutes == 2160.0


def test_metric_empirics_counts_absence_and_strictly_silences_past_gate() -> None:
    first = datetime(2026, 6, 3, 12, tzinfo=UTC)
    second = datetime(2026, 6, 7, 12, tzinfo=UTC)
    third = datetime(2026, 6, 11, 12, tzinfo=UTC)
    observations = [
        AgeObservation("one", first - timedelta(minutes=90), 29.76, -95.37),
        AgeObservation("two", second - timedelta(minutes=91), 29.76, -95.37),
    ]

    result = metric_age_empirics(
        source="tceq",
        metric="no2",
        observations=observations,
        anchors=(first, second, third),
        gate_minutes=90.0,
        anchor_lat=29.7604,
        anchor_lon=-95.3698,
        radius_km=50.0,
    )

    assert result.anchor_count == 3
    assert result.anchors_with_data == 2
    assert result.anchors_without_data == 1
    assert result.minimum == 90.0
    assert result.p50 == 90.5
    assert result.p90 == pytest.approx(90.9)
    assert result.p95 == pytest.approx(90.95)
    assert result.p99 == pytest.approx(90.99)
    assert result.maximum == 91.0
    assert result.silenced_anchor_count == 1
    assert result.silenced_fraction == 0.5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_snapshot(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "snapshot.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE data_points (
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                source_entity_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO data_points VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "tceq",
                    "no2",
                    "near",
                    "2026-06-05 12:00:00",
                    10.0,
                    29.7604,
                    -95.3698,
                ),
                (
                    "tceq",
                    "no2",
                    "outside",
                    "2026-06-05 12:00:00",
                    10.0,
                    31.0,
                    -95.3698,
                ),
                (
                    "noaa_gfs",
                    "pbl_height",
                    "cell",
                    "2026-06-05 12:00:00",
                    500.0,
                    29.7604,
                    -95.3698,
                ),
                (
                    "openaq",
                    "pm25",
                    "not-a-verified-monitor",
                    "2026-06-05 12:00:00",
                    20.0,
                    29.7604,
                    -95.3698,
                ),
                (
                    "purpleair",
                    "pm25",
                    "288282",
                    "2026-06-09 08:00:00",
                    876.0,
                    29.7604,
                    -95.3698,
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()
    return path, _sha256(path)


def test_snapshot_run_is_hash_guarded_filtered_serialized_and_stops_loudly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database, snapshot_hash = _synthetic_snapshot(tmp_path)

    report = run_empirics(database, expected_sha256=snapshot_hash)

    assert report.snapshot_sha256 == snapshot_hash
    assert report.anchor_count == 936
    assert report.input_rows == 5
    assert report.quality_excluded_rows == 2
    assert report.eligible_in_radius_rows == 2
    assert {(row.source, row.metric) for row in report.metrics} == {
        ("noaa_gfs", "pbl_height"),
        ("tceq", "no2"),
    }
    assert report.stop_rule_violations
    assert "tceq/no2" in report.stop_rule_violations[0]
    assert "epa_aqs" in report.structurally_absent_sources
    assert "Hourly >20% stop-rule violations" in render_markdown(report)

    output = tmp_path / "age-report.json"
    write_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["snapshot_sha256"] == snapshot_hash
    assert payload["gates_minutes"]["sentinel5p"] == 720.0

    cli_output = tmp_path / "age-report-cli.json"
    exit_code = main(
        [
            "--database",
            str(database),
            "--expected-sha256",
            snapshot_hash,
            "--output",
            str(cli_output),
            "--format",
            "markdown",
        ]
    )
    assert exit_code == 2
    assert cli_output.exists()
    assert "tceq/no2" in capsys.readouterr().out
    assert _sha256(database) == snapshot_hash

    with pytest.raises(ValueError, match="mismatch before read"):
        run_empirics(database, expected_sha256="0" * 64)
