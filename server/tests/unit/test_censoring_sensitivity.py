"""B15 label-free censoring sensitivity and snapshot safeguards."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval.censoring_sensitivity import (
    CensoringObservation,
    metric_censoring_sensitivity,
    render_markdown,
    run_sensitivity,
    write_report,
)


ANCHOR = datetime(2026, 6, 5, 12, tzinfo=UTC)


def _observation(
    hours: float,
    value: float,
    *,
    entity_id: str = "monitor",
) -> CensoringObservation:
    return CensoringObservation(
        entity_id=entity_id,
        timestamp=ANCHOR + timedelta(hours=hours),
        value=value,
        distance_km=1.0,
    )


def test_metric_sensitivity_uses_scorer_window_and_reports_verdict_change() -> None:
    result = metric_censoring_sensitivity(
        source="tceq",
        metric="so2",
        unit="ppb",
        observations=(
            _observation(-6.0, -0.2),
            _observation(-5.0, 0.2),
            _observation(-3.0, 0.5),
            _observation(0.0, 1.0),
        ),
        anchors=(ANCHOR,),
    )

    assert result.censor_limit == 0.5
    assert result.replacement_value == 0.25
    assert result.baseline_observation_instances == 3
    assert result.censored_observation_instances == 2
    assert result.censored_fraction == pytest.approx(2 / 3)
    assert result.primary_evaluable_windows == 1
    assert result.deletion_evaluable_windows == 0
    assert result.paired_evaluable_windows == 0
    assert result.primary_supporting == 1
    assert result.primary_contradicting == 0
    assert result.primary_silent == 0
    assert result.deletion_silent == 1
    assert result.changed_verdict_count == 1


def test_metric_sensitivity_quantifies_deletion_mean_shift_when_paired() -> None:
    result = metric_censoring_sensitivity(
        source="tceq",
        metric="so2",
        unit="ppb",
        observations=(
            _observation(-8.0, -0.2),
            _observation(-7.0, 0.2),
            _observation(-6.0, 0.5),
            _observation(-5.0, 0.7),
            _observation(-3.0, 0.9),
            _observation(0.0, 1.5),
        ),
        anchors=(ANCHOR,),
    )

    assert result.primary_evaluable_windows == 1
    assert result.deletion_evaluable_windows == 1
    assert result.paired_evaluable_windows == 1
    assert result.mean_shift_minimum is not None
    assert result.mean_shift_minimum > 0.0
    assert result.mean_shift_p50 == result.mean_shift_minimum
    assert result.mean_shift_mean == result.mean_shift_minimum
    assert result.mean_shift_p95 == result.mean_shift_minimum
    assert result.mean_shift_maximum == result.mean_shift_minimum
    assert result.deletion_mean_higher_fraction == 1.0
    assert result.deletion_mean_equal_fraction == 0.0
    assert result.deletion_mean_lower_fraction == 0.0


def test_metric_sensitivity_normalizes_naive_and_aware_timestamps() -> None:
    aware = metric_censoring_sensitivity(
        source="tceq",
        metric="so2",
        unit="ppb",
        observations=(
            _observation(-6.0, 0.2),
            _observation(-5.0, 0.5),
            _observation(-3.0, 0.8),
            _observation(0.0, 1.0),
        ),
        anchors=(ANCHOR,),
    )
    naive = metric_censoring_sensitivity(
        source="tceq",
        metric="so2",
        unit="ppb",
        observations=tuple(
            CensoringObservation(
                entity_id=row.entity_id,
                timestamp=row.timestamp.replace(tzinfo=None),
                value=row.value,
                distance_km=row.distance_km,
            )
            for row in (
                _observation(-6.0, 0.2),
                _observation(-5.0, 0.5),
                _observation(-3.0, 0.8),
                _observation(0.0, 1.0),
            )
        ),
        anchors=(ANCHOR.replace(tzinfo=None),),
    )

    assert naive == aware


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot(tmp_path: Path, rows: list[tuple[object, ...]]) -> tuple[Path, str]:
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
                unit TEXT NOT NULL,
                lat REAL NOT NULL,
                lon REAL NOT NULL
            )
            """
        )
        connection.executemany(
            "INSERT INTO data_points VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return path, _sha256(path)


def test_snapshot_run_is_hash_guarded_serialized_and_unit_checked(
    tmp_path: Path,
) -> None:
    rows = [
        (
            "tceq",
            "so2",
            "monitor",
            f"2026-06-05 {hour:02d}:00:00",
            value,
            "ppb",
            29.7604,
            -95.3698,
        )
        for hour, value in enumerate((-0.2, 0.2, 0.5, 0.7, 0.9, 1.5))
    ]
    database, snapshot_hash = _snapshot(tmp_path, rows)

    report = run_sensitivity(database, expected_sha256=snapshot_hash)

    assert report.snapshot_sha256 == snapshot_hash
    assert report.anchor_count == 1488
    assert report.unit_assertion_passed is True
    assert len(report.metrics) == 1
    assert report.metrics[0].source == "tceq"
    assert report.metrics[0].unit == "ppb"
    assert "Deletion - primary" in render_markdown(report)

    output = tmp_path / "sensitivity.json"
    write_report(report, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["snapshot_sha256"] == snapshot_hash
    assert payload["rules"]["ground_so2_limit_ppb"] == 0.5
    assert payload["rules"]["sentinel_so2_limit_mol_m2"] == 4.46e-4
    assert _sha256(database) == snapshot_hash


def test_snapshot_run_stops_on_mixed_units_without_mutating_input(
    tmp_path: Path,
) -> None:
    rows = [
        (
            "tceq",
            "no2",
            "monitor",
            "2026-06-05 00:00:00",
            1.0,
            unit,
            29.7604,
            -95.3698,
        )
        for unit in ("ppb", "ppm")
    ]
    database, snapshot_hash = _snapshot(tmp_path, rows)

    with pytest.raises(ValueError, match="multiple units.*tceq/no2"):
        run_sensitivity(database, expected_sha256=snapshot_hash)

    assert _sha256(database) == snapshot_hash


def test_unit_interlock_precedes_b6_eligibility_filter(tmp_path: Path) -> None:
    rows = [
        (
            "openaq",
            "pm25",
            f"unverified-{index}",
            "2026-06-05 00:00:00",
            1.0,
            unit,
            29.7604,
            -95.3698,
        )
        for index, unit in enumerate(("ug/m3", "ppm"))
    ]
    database, snapshot_hash = _snapshot(tmp_path, rows)

    with pytest.raises(ValueError, match="multiple units.*openaq/pm25"):
        run_sensitivity(database, expected_sha256=snapshot_hash)

    assert _sha256(database) == snapshot_hash
