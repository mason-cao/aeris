"""A-9 label-free GFS nearest-component alignment empirics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval import gfs_component_alignment_empirics
from app.eval.gfs_component_alignment_empirics import (
    GfsComponentObservation,
    GfsComponentAlignmentReport,
    assess_anchor,
    build_report,
    render_markdown,
)
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256


EVENT = datetime(2026, 6, 5, 12, tzinfo=UTC)
LAT = 29.7604
LON = -95.3698


def _observation(
    metric: str,
    timestamp: datetime,
    value: float,
    *,
    entity_id: str = "cell-a",
    lat: float = LAT,
    lon: float = LON,
) -> GfsComponentObservation:
    return GfsComponentObservation(
        metric=metric,
        entity_id=entity_id,
        timestamp=timestamp,
        value=value,
        unit="m/s",
        lat=lat,
        lon=lon,
    )


def test_anchor_exact_timestamp_pair_is_old_and_new_eligible() -> None:
    result = assess_anchor(
        [
            _observation("u_10m", EVENT, 1.0),
            _observation("v_10m", EVENT, 2.0),
        ],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.both_values_present is True
    assert result.both_fresh is True
    assert result.timestamp_exact is True
    assert result.old_pair_eligible is True
    assert result.new_pair_eligible is True
    assert result.mismatch_minutes is None


def test_anchor_mismatch_was_old_eligible_and_is_new_silent() -> None:
    result = assess_anchor(
        [
            _observation("u_10m", EVENT, 1.0),
            _observation("v_10m", EVENT + timedelta(hours=6), 2.0),
        ],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.both_fresh is True
    assert result.timestamp_mismatch is True
    assert result.old_pair_eligible is True
    assert result.new_pair_eligible is False
    assert result.mismatch_minutes == 360.0


def test_anchor_b8_exact_boundary_votes_and_one_minute_past_silences() -> None:
    at_gate = assess_anchor(
        [
            _observation("u_10m", EVENT + timedelta(hours=6), 1.0),
            _observation("v_10m", EVENT + timedelta(hours=6), 2.0),
        ],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )
    past_gate = assess_anchor(
        [
            _observation("u_10m", EVENT + timedelta(minutes=361), 1.0),
            _observation("v_10m", EVENT + timedelta(minutes=361), 2.0),
        ],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert at_gate.both_fresh is True
    assert at_gate.new_pair_eligible is True
    assert past_gate.both_fresh is False
    assert past_gate.new_pair_eligible is False


def test_anchor_records_equal_timestamp_entity_disagreement_without_silencing() -> None:
    result = assess_anchor(
        [
            _observation("u_10m", EVENT, 1.0, entity_id="cell-a"),
            _observation("v_10m", EVENT, 2.0, entity_id="cell-b"),
        ],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.timestamp_exact is True
    assert result.equal_timestamp_entity_mismatch is True
    assert result.new_pair_eligible is True


def test_anchor_normalizes_naive_and_offset_aware_timestamps() -> None:
    result = assess_anchor(
        [
            _observation("u_10m", EVENT.replace(tzinfo=None), 1.0),
            _observation(
                "v_10m",
                datetime.fromisoformat("2026-06-05T07:00:00-05:00"),
                2.0,
            ),
        ],
        EVENT.replace(tzinfo=None),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.timestamp_exact is True
    assert result.new_pair_eligible is True


def test_report_records_alignment_counts_and_mismatch_distribution() -> None:
    report = build_report(
        [
            _observation("u_10m", EVENT, 1.0),
            _observation("v_10m", EVENT + timedelta(hours=6), 2.0),
        ],
        snapshot_sha256="abc",
        anchors=(EVENT,),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert report.anchor_count == 1
    assert report.u_observation_count == 1
    assert report.v_observation_count == 1
    assert report.old_pair_eligible_count == 1
    assert report.new_pair_eligible_count == 0
    assert report.changed_to_silent_count == 1
    assert report.timestamp_mismatch_count == 1
    assert report.mismatch_minutes.minimum == 360.0
    assert report.mismatch_minutes.maximum == 360.0


def test_report_rejects_duplicates_nonfinite_rows_and_duplicate_anchors() -> None:
    row = _observation("u_10m", EVENT, 1.0)
    with pytest.raises(ValueError, match="duplicate GFS component"):
        build_report(
            [row, row],
            snapshot_sha256="abc",
            anchors=(EVENT,),
            anchor_lat=LAT,
            anchor_lon=LON,
        )

    invalid = GfsComponentObservation(
        **{**row.__dict__, "value": float("nan")}
    )
    with pytest.raises(ValueError, match="finite"):
        build_report(
            [invalid],
            snapshot_sha256="abc",
            anchors=(EVENT,),
            anchor_lat=LAT,
            anchor_lon=LON,
        )

    with pytest.raises(ValueError, match="anchors must be unique"):
        build_report(
            [row],
            snapshot_sha256="abc",
            anchors=(EVENT, EVENT),
            anchor_lat=LAT,
            anchor_lon=LON,
        )


def test_markdown_and_json_report_pair_eligibility() -> None:
    report = build_report(
        [
            _observation("u_10m", EVENT, 1.0),
            _observation("v_10m", EVENT + timedelta(hours=6), 2.0),
        ],
        snapshot_sha256="abc",
        anchors=(EVENT,),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    rendered = render_markdown(report)
    payload = report.to_dict()

    assert "| Old pair eligible | 1 | 100.00% |" in rendered
    assert "| New pair eligible | 0 | 0.00% |" in rendered
    assert "| Timestamp mismatch | 1 | 100.00% |" in rendered
    assert "| Mismatch minutes | 360 | 360 | 360 | 360 |" in rendered
    assert payload["changed_to_silent_count"] == 1
    assert json.dumps(payload, sort_keys=True) == json.dumps(
        report.to_dict(), sort_keys=True
    )


def _sqlite_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE data_points (
                source_entity_id TEXT,
                timestamp TEXT,
                value REAL,
                unit TEXT,
                source TEXT,
                metric TEXT,
                lat REAL,
                lon REAL
            )
            """
        )
        rows = [
            ("cell-a", EVENT.isoformat(), 1.0, "m/s", "noaa_gfs", "u_10m", LAT, LON),
            (
                "cell-a",
                (EVENT + timedelta(hours=6)).isoformat(),
                2.0,
                "m/s",
                "noaa_gfs",
                "v_10m",
                LAT,
                LON,
            ),
        ]
        connection.executemany(
            """
            INSERT INTO data_points
                (source_entity_id, timestamp, value, unit, source, metric, lat, lon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def test_run_empirics_requires_canonical_hash_and_checks_both_sides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "snapshot.db"
    _sqlite_snapshot(database)

    with pytest.raises(ValueError, match="canonical locked"):
        gfs_component_alignment_empirics.run_empirics(
            database,
            expected_sha256="wrong",
        )

    monkeypatch.setattr(
        gfs_component_alignment_empirics,
        "_snapshot_sha256",
        lambda _path: "mismatch",
    )
    with pytest.raises(ValueError, match="before read"):
        gfs_component_alignment_empirics.run_empirics(
            database,
            expected_sha256=LOCKED_SNAPSHOT_SHA256,
        )

    hashes = iter((LOCKED_SNAPSHOT_SHA256, "changed"))
    monkeypatch.setattr(
        gfs_component_alignment_empirics,
        "_snapshot_sha256",
        lambda _path: next(hashes),
    )
    with pytest.raises(RuntimeError, match="after read"):
        gfs_component_alignment_empirics.run_empirics(
            database,
            expected_sha256=LOCKED_SNAPSHOT_SHA256,
        )


def test_run_empirics_reads_immutable_sqlite_and_cli_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "snapshot.db"
    output = tmp_path / "report.json"
    _sqlite_snapshot(database)
    monkeypatch.setattr(
        gfs_component_alignment_empirics,
        "_snapshot_sha256",
        lambda _path: LOCKED_SNAPSHOT_SHA256,
    )
    monkeypatch.setattr(
        gfs_component_alignment_empirics,
        "candidate_centers",
        lambda _start, _end: (EVENT,),
    )

    report = gfs_component_alignment_empirics.run_empirics(
        database,
        expected_sha256=LOCKED_SNAPSHOT_SHA256,
    )

    assert isinstance(report, GfsComponentAlignmentReport)
    assert report.input_row_count == 2
    assert report.changed_to_silent_count == 1

    monkeypatch.setattr(
        gfs_component_alignment_empirics,
        "run_empirics",
        lambda _path, **_kwargs: report,
    )
    exit_code = gfs_component_alignment_empirics.main(
        [
            "--database",
            str(database),
            "--expected-sha256",
            LOCKED_SNAPSHOT_SHA256,
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert stdout_payload == file_payload == report.to_dict()
