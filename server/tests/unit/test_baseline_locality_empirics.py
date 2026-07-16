"""B17 label-free pooled-versus-station baseline comparison."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.eval.baseline_locality_empirics import (
    BaselineObservation,
    baseline_locality_manifest_payload,
    build_report,
    load_baseline_locality_fixture,
    main,
    metric_baseline_locality,
    render_markdown,
    run_comparison,
    write_report,
)

ANCHOR = datetime(2026, 6, 5, 12, 0, tzinfo=UTC)


def _observation(
    entity_id: str,
    hour: int,
    value: float,
    *,
    distance_km: float,
    aware: bool = True,
) -> BaselineObservation:
    timestamp = datetime(2026, 6, 5, hour, 0, tzinfo=UTC)
    if not aware:
        timestamp = timestamp.replace(tzinfo=None)
    return BaselineObservation(entity_id, timestamp, value, distance_km)


def _spatial_confounding_observations(
    *,
    aware: bool = True,
) -> tuple[BaselineObservation, ...]:
    return tuple(
        [
            _observation("near", hour, 10.0, distance_km=1.0, aware=aware)
            for hour in (6, 7, 9)
        ]
        + [
            _observation("far", hour, 100.0, distance_km=10.0, aware=aware)
            for hour in (6, 7, 9)
        ]
        + [
            _observation("near", 12, 20.0, distance_km=1.0, aware=aware),
            _observation("far", 12, 100.0, distance_km=10.0, aware=aware),
        ]
    )


def test_station_matching_removes_pooled_spatial_confounding() -> None:
    result = metric_baseline_locality(
        source="tceq",
        metric="ozone",
        unit="ppb",
        observations=_spatial_confounding_observations(),
        anchors=(ANCHOR,),
    )

    assert result.event_eligible_count == 1
    assert result.pooled_evaluable_count == 1
    assert result.matched_evaluable_count == 1
    assert result.pooled_supporting == 0
    assert result.pooled_contradicting == 1
    assert result.matched_supporting == 1
    assert result.matched_contradicting == 0
    assert result.pooled_support_rate == 0.0
    assert result.matched_support_rate == 1.0
    assert result.changed_verdict_count == 1
    assert result.matched_baseline_n_minimum == 3
    assert result.matched_baseline_n_p50 == 3
    assert result.matched_baseline_n_p95 == 3
    assert result.matched_baseline_n_maximum == 3


def test_sparse_matched_history_is_silent_without_pooled_rescue() -> None:
    observations = (
        _observation("near", 7, 10.0, distance_km=1.0),
        _observation("near", 9, 10.0, distance_km=1.0),
        _observation("far", 6, 10.0, distance_km=10.0),
        _observation("far", 7, 10.0, distance_km=10.0),
        _observation("far", 9, 10.0, distance_km=10.0),
        _observation("near", 12, 20.0, distance_km=1.0),
    )

    result = metric_baseline_locality(
        source="tceq",
        metric="ozone",
        unit="ppb",
        observations=observations,
        anchors=(ANCHOR,),
    )

    assert result.pooled_supporting == 1
    assert result.matched_silent == 1
    assert result.matched_evaluable_count == 0
    assert result.matched_baseline_n_minimum == 2
    assert result.changed_verdict_count == 1


def test_empty_metric_is_all_silent_with_zero_baseline_counts() -> None:
    result = metric_baseline_locality(
        source="tceq",
        metric="ozone",
        unit="ppb",
        observations=(),
        anchors=(ANCHOR,),
    )

    assert result.event_eligible_count == 0
    assert result.pooled_silent == 1
    assert result.matched_silent == 1
    assert result.matched_baseline_n_minimum == 0
    assert result.matched_baseline_n_p50 == 0
    assert result.matched_baseline_n_p95 == 0
    assert result.matched_baseline_n_maximum == 0


def test_naive_and_aware_inputs_produce_identical_metric_results() -> None:
    aware = metric_baseline_locality(
        source="tceq",
        metric="ozone",
        unit="ppb",
        observations=_spatial_confounding_observations(),
        anchors=(ANCHOR,),
    )
    naive = metric_baseline_locality(
        source="tceq",
        metric="ozone",
        unit="ppb",
        observations=_spatial_confounding_observations(aware=False),
        anchors=(ANCHOR.replace(tzinfo=None),),
    )

    assert aware == naive


def test_exact_support_boundary_remains_silent() -> None:
    observations = (
        _observation("near", 6, 9.0, distance_km=1.0),
        _observation("near", 7, 10.0, distance_km=1.0),
        _observation("near", 9, 11.0, distance_km=1.0),
        _observation("near", 12, 10.816496580927726, distance_km=1.0),
    )

    result = metric_baseline_locality(
        source="tceq",
        metric="ozone",
        unit="ppb",
        observations=observations,
        anchors=(ANCHOR,),
    )

    assert result.matched_supporting == 0
    assert result.matched_contradicting == 0
    assert result.matched_silent == 1


def test_report_write_load_and_manifest_are_deterministic(tmp_path: Path) -> None:
    report = build_report(
        {("tceq", "ozone", "ppb"): _spatial_confounding_observations()},
        snapshot_sha256="a" * 64,
        anchors=(ANCHOR,),
        input_rows=8,
        eligible_in_radius_rows=8,
        quality_excluded_rows=0,
        anchor_lat=29.7604,
        anchor_lon=-95.3698,
        radius_km=50.0,
    )
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_report(report, first)
    write_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    loaded = load_baseline_locality_fixture(
        first,
        expected_snapshot_sha256="a" * 64,
    )
    assert loaded["rules"]["baseline_locality"] == "nearest_event_entity"
    manifest = baseline_locality_manifest_payload(
        first,
        expected_snapshot_sha256="a" * 64,
    )
    assert manifest["artifact"] == "first.json"
    assert len(manifest["artifact_sha256"]) == 64


def test_fixture_loader_rejects_unshipped_locality(tmp_path: Path) -> None:
    path = tmp_path / "wrong.json"
    path.write_text(
        '{"schema_version": 1, "snapshot_sha256": "'
        + ("a" * 64)
        + '", "rules": {"baseline_locality": "network_pooled"}, '
        '"metrics": []}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline locality"):
        load_baseline_locality_fixture(
            path,
            expected_snapshot_sha256="a" * 64,
        )


def test_versioned_fixture_records_executed_snapshot_comparison() -> None:
    payload = load_baseline_locality_fixture()
    metrics = {
        (row["source"], row["metric"]): row for row in payload["metrics"]
    }

    assert payload["anchor_count"] == 936
    assert payload["input_rows"] == 165265
    assert payload["eligible_in_radius_rows"] == 109720
    assert payload["quality_excluded_rows"] == 55545
    assert payload["structurally_absent_sources"] == ["epa_aqs"]
    assert metrics[("openaq", "pm25")]["pooled_supporting"] == 326
    assert metrics[("openaq", "pm25")]["matched_supporting"] == 179
    assert metrics[("sentinel5p", "s5p_co_column")][
        "matched_evaluable_count"
    ] == 0


def _temporary_snapshot(path: Path) -> str:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE data_points (
                source TEXT NOT NULL,
                metric TEXT NOT NULL,
                source_entity_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value,
                unit TEXT,
                lat,
                lon
            )
            """
        )
        rows = [
            ("tceq", "ozone", "near", f"2026-06-05 {hour:02d}:00:00", value, "ppb", 29.7604, -95.3698)
            for hour, value in ((6, 10.0), (7, 10.0), (9, 10.0), (12, 20.0))
        ]
        rows.extend(
            [
                ("tceq", "temperature", "irrelevant", "2026-06-05 12:00:00", 30.0, "C", 29.7604, -95.3698),
                ("tceq", "ozone", "outside", "2026-06-05 12:00:00", 20.0, "ppb", 35.0, -95.3698),
                ("openaq", "pm25", "not-a-monitor", "2026-06-05 12:00:00", 12.0, "ug/m3", 29.7604, -95.3698),
                ("purpleair", "pm25", "synthetic", "2026-06-05 12:00:00", 12.0, "ug/m3", 29.7604, -95.3698),
                ("sentinel5p", "s5p_no2_column", "granule", "2026-06-05 12:00:00", 0.0001, "mol/m^2", 29.7604, -95.3698),
            ]
        )
        connection.executemany(
            """
            INSERT INTO data_points (
                source, metric, source_entity_id, timestamp,
                value, unit, lat, lon
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_read_only_snapshot_run_and_cli_report_real_loader_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "snapshot.db"
    expected_hash = _temporary_snapshot(database)

    report = run_comparison(
        database,
        expected_sha256=expected_hash,
        anchor_lat=29.7604,
        anchor_lon=-95.3698,
        radius_km=50.0,
    )

    assert report.snapshot_sha256 == expected_hash
    assert report.input_rows == 8
    assert report.eligible_in_radius_rows == 6
    assert report.quality_excluded_rows == 1
    assert {metric.source for metric in report.metrics} == {
        "purpleair",
        "sentinel5p",
        "tceq",
    }
    assert "nearest-event-entity" in render_markdown(report)

    output = tmp_path / "report.json"
    exit_code = main(
        [
            "--database",
            str(database),
            "--expected-sha256",
            expected_hash,
            "--anchor-lat",
            "29.7604",
            "--anchor-lon",
            "-95.3698",
            "--radius-km",
            "50",
            "--output",
            str(output),
            "--format",
            "json",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert output.read_bytes().endswith(b"\n")
    assert printed["input_rows"] == 8
    assert hashlib.sha256(database.read_bytes()).hexdigest() == expected_hash


def test_snapshot_hash_mismatch_stops_before_read(tmp_path: Path) -> None:
    database = tmp_path / "snapshot.db"
    _temporary_snapshot(database)

    with pytest.raises(ValueError, match="mismatch before read"):
        run_comparison(database, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    "observation",
    [
        BaselineObservation("", ANCHOR, 1.0, 0.0),
        BaselineObservation("e", ANCHOR, float("nan"), 0.0),
        BaselineObservation("e", ANCHOR, 1.0, -1.0),
    ],
)
def test_invalid_synthetic_observations_fail_loudly(
    observation: BaselineObservation,
) -> None:
    with pytest.raises(ValueError):
        metric_baseline_locality(
            source="tceq",
            metric="ozone",
            unit="ppb",
            observations=(observation,),
            anchors=(ANCHOR,),
        )


def test_duplicate_entity_timestamp_fails_loudly() -> None:
    duplicate = _observation("e", 12, 1.0, distance_km=1.0)

    with pytest.raises(ValueError, match="duplicate baseline observation"):
        metric_baseline_locality(
            source="tceq",
            metric="ozone",
            unit="ppb",
            observations=(duplicate, duplicate),
            anchors=(ANCHOR,),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"schema_version": 2}, "unsupported"),
        ({"snapshot_sha256": "b" * 64}, "snapshot mismatch"),
        ({"unit_assertion_passed": False}, "unit assertion"),
        ({"metrics": "not-a-list"}, "metrics"),
    ],
)
def test_fixture_loader_rejects_invalid_evidence_fields(
    tmp_path: Path,
    mutation: dict[str, object],
    message: str,
) -> None:
    report = build_report(
        {},
        snapshot_sha256="a" * 64,
        anchors=(ANCHOR,),
        input_rows=0,
        eligible_in_radius_rows=0,
        quality_excluded_rows=0,
        anchor_lat=29.7604,
        anchor_lon=-95.3698,
        radius_km=50.0,
    ).to_dict()
    report.update(mutation)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_baseline_locality_fixture(
            path,
            expected_snapshot_sha256="a" * 64,
        )
