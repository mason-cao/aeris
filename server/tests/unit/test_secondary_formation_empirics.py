"""A-3 label-free secondary-formation coupling empirics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval import secondary_formation_empirics
from app.eval.secondary_formation_empirics import (
    FormationObservation,
    SecondaryFormationEmpiricalReport,
    assess_anchor,
    build_report,
    render_markdown,
)
from app.llm.corroboration import CONTRADICTING, SILENT, SUPPORTING
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256


EVENT = datetime(2026, 6, 5, 15, tzinfo=UTC)
LAT = 29.7604
LON = -95.3698


def _observation(
    source: str,
    metric: str,
    hour: int,
    value: float,
    *,
    entity_id: str | None = None,
    timestamp: datetime | None = None,
    lat: float = LAT,
    lon: float = LON,
) -> FormationObservation:
    return FormationObservation(
        source=source,
        metric=metric,
        entity_id=entity_id or f"{source}-{metric}",
        timestamp=timestamp or datetime(2026, 6, 5, hour, tzinfo=UTC),
        value=value,
        unit="pct" if metric == "cloud_cover" else "ppb",
        lat=lat,
        lon=lon,
    )


def _lag_observations(*, ozone_points: int = 3) -> list[FormationObservation]:
    no2 = [
        _observation("tceq", "no2", 14, 20.0),
        _observation("tceq", "no2", 15, 50.0),
        _observation("tceq", "no2", 16, 20.0),
    ]
    ozone = [
        _observation("openaq", "ozone", 17, 20.0),
        _observation("openaq", "ozone", 18, 60.0),
        _observation("openaq", "ozone", 19, 20.0),
    ][:ozone_points]
    return no2 + ozone


def test_anchor_conditions_exact_cloud_boundary_on_a_voted_lag() -> None:
    observations = _lag_observations() + [
        _observation("openweather", "cloud_cover", 15, 50.0)
    ]

    result = assess_anchor(observations, EVENT, anchor_lat=LAT, anchor_lon=LON)

    assert result.ozone_point_count == 3
    assert result.no2_point_count == 3
    assert result.lag_verdict == SUPPORTING
    assert result.cloud_mean == 50.0
    assert result.former_insolation_verdict == SUPPORTING
    assert result.conditional_insolation_verdict == SUPPORTING


def test_anchor_silences_clear_and_overcast_cloud_when_lag_is_low_n() -> None:
    low_n = _lag_observations(ozone_points=2)
    clear = assess_anchor(
        low_n + [_observation("openweather", "cloud_cover", 15, 49.9999)],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )
    overcast = assess_anchor(
        low_n + [_observation("openweather", "cloud_cover", 15, 50.0001)],
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert clear.lag_verdict == SILENT
    assert clear.former_insolation_verdict == SUPPORTING
    assert clear.conditional_insolation_verdict == SILENT
    assert overcast.lag_verdict == SILENT
    assert overcast.former_insolation_verdict == CONTRADICTING
    assert overcast.conditional_insolation_verdict == SILENT


def test_anchor_missing_cloud_stays_silent_after_lag_vote() -> None:
    result = assess_anchor(
        _lag_observations(),
        EVENT,
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.lag_verdict == SUPPORTING
    assert result.cloud_mean is None
    assert result.former_insolation_verdict == SILENT
    assert result.conditional_insolation_verdict == SILENT


def test_anchor_normalizes_naive_and_offset_aware_timestamps() -> None:
    aware = _lag_observations() + [
        _observation("openweather", "cloud_cover", 15, 25.0)
    ]
    naive = [
        FormationObservation(
            source=row.source,
            metric=row.metric,
            entity_id=row.entity_id,
            timestamp=row.timestamp.replace(tzinfo=None),
            value=row.value,
            unit=row.unit,
            lat=row.lat,
            lon=row.lon,
        )
        for row in aware
    ]

    aware_result = assess_anchor(aware, EVENT, anchor_lat=LAT, anchor_lon=LON)
    naive_result = assess_anchor(
        naive,
        EVENT.replace(tzinfo=None),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert naive_result == aware_result


def test_report_records_point_distributions_and_removed_cloud_only_vote() -> None:
    observations = _lag_observations(ozone_points=2) + [
        _observation("openweather", "cloud_cover", 15, 25.0)
    ]

    report = build_report(
        observations,
        snapshot_sha256="abc",
        anchors=(EVENT,),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert report.anchor_count == 1
    assert report.ozone_point_counts.frequency == ((2, 1),)
    assert report.ozone_point_counts.below_floor_count == 1
    assert report.no2_point_counts.frequency == ((3, 1),)
    assert report.lag_outcomes.silent_count == 1
    assert report.former_insolation_outcomes.supporting_count == 1
    assert report.conditional_insolation_outcomes.silent_count == 1
    assert report.former_cloud_only_votes_silenced == 1
    assert report.former_supporting_votes_silenced == 1
    assert report.former_contradicting_votes_silenced == 0


def test_report_rejects_duplicate_and_nonfinite_relevant_rows() -> None:
    row = _observation("openaq", "ozone", 17, 20.0)
    with pytest.raises(ValueError, match="duplicate relevant observation"):
        build_report(
            [row, row],
            snapshot_sha256="abc",
            anchors=(EVENT,),
            anchor_lat=LAT,
            anchor_lon=LON,
        )

    invalid = _observation("openaq", "ozone", 17, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        build_report(
            [invalid],
            snapshot_sha256="abc",
            anchors=(EVENT,),
            anchor_lat=LAT,
            anchor_lon=LON,
        )


def test_markdown_and_json_report_old_and_new_outcomes() -> None:
    report = build_report(
        _lag_observations(ozone_points=2)
        + [_observation("openweather", "cloud_cover", 15, 25.0)],
        snapshot_sha256="abc",
        anchors=(EVENT,),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    rendered = render_markdown(report)
    payload = report.to_dict()

    assert "| Former unconditional insolation | 1 (100.00%) | 0 (0.00%) | 0 (0.00%) |" in rendered
    assert "| Conditional insolation | 0 (0.00%) | 0 (0.00%) | 1 (100.00%) |" in rendered
    assert "| Former cloud-only votes silenced | 1 |" in rendered
    assert payload["former_cloud_only_votes_silenced"] == 1
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
            (
                row.entity_id,
                row.timestamp.isoformat(),
                row.value,
                row.unit,
                row.source,
                row.metric,
                row.lat,
                row.lon,
            )
            for row in _lag_observations()
            + [_observation("openweather", "cloud_cover", 15, 25.0)]
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
        secondary_formation_empirics.run_empirics(
            database,
            expected_sha256="wrong",
        )

    monkeypatch.setattr(
        secondary_formation_empirics,
        "_snapshot_sha256",
        lambda _path: "mismatch",
    )
    with pytest.raises(ValueError, match="before read"):
        secondary_formation_empirics.run_empirics(
            database,
            expected_sha256=LOCKED_SNAPSHOT_SHA256,
        )

    hashes = iter((LOCKED_SNAPSHOT_SHA256, "changed"))
    monkeypatch.setattr(
        secondary_formation_empirics,
        "_snapshot_sha256",
        lambda _path: next(hashes),
    )
    with pytest.raises(RuntimeError, match="after read"):
        secondary_formation_empirics.run_empirics(
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
        secondary_formation_empirics,
        "_snapshot_sha256",
        lambda _path: LOCKED_SNAPSHOT_SHA256,
    )
    monkeypatch.setattr(
        secondary_formation_empirics,
        "candidate_centers",
        lambda _start, _end: (EVENT,),
    )

    report = secondary_formation_empirics.run_empirics(
        database,
        expected_sha256=LOCKED_SNAPSHOT_SHA256,
    )

    assert isinstance(report, SecondaryFormationEmpiricalReport)
    assert report.input_row_count == 7
    assert report.anchor_count == 1
    assert report.lag_outcomes.supporting_count == 1

    monkeypatch.setattr(
        secondary_formation_empirics,
        "run_empirics",
        lambda _path, **_kwargs: report,
    )
    exit_code = secondary_formation_empirics.main(
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


def test_anchor_excludes_endpoint_outside_window_and_radius() -> None:
    observations = _lag_observations() + [
        _observation("openweather", "cloud_cover", 15, 25.0),
        _observation(
            "openaq",
            "ozone",
            0,
            999.0,
            entity_id="late",
            timestamp=EVENT + timedelta(hours=36, microseconds=1),
        ),
        _observation(
            "openaq",
            "ozone",
            18,
            999.0,
            entity_id="far",
            lat=31.0,
        ),
    ]

    result = assess_anchor(observations, EVENT, anchor_lat=LAT, anchor_lon=LON)

    assert result.ozone_point_count == 3
    assert result.lag_verdict == SUPPORTING
