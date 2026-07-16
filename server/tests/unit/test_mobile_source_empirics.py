"""A-4 label-free mobile-source local-day empirics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval import mobile_source_empirics
from app.eval.mobile_source_empirics import (
    MobileObservation,
    MobileSourceEmpiricalReport,
    assess_pair,
    build_report,
    render_markdown,
)
from app.llm.corroboration import CONTRADICTING, SILENT, SUPPORTING
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    verified_monitor_entity_ids,
)


EVENT = datetime(2026, 6, 5, 15, tzinfo=UTC)
LAT = 29.7604
LON = -95.3698


def _observation(
    source: str,
    metric: str,
    timestamp: datetime,
    value: float,
    *,
    entity_id: str | None = None,
    lat: float = LAT,
    lon: float = LON,
) -> MobileObservation:
    return MobileObservation(
        source=source,
        metric=metric,
        entity_id=entity_id or f"{source}-{metric}",
        timestamp=timestamp,
        value=value,
        unit="ug/m3",
        lat=lat,
        lon=lon,
    )


def _neighbor_peak_rows(
    *,
    anomaly_day_points: int = 4,
    source: str = "openaq",
    metric: str = "no2",
    entity_id: str | None = None,
) -> list[MobileObservation]:
    rows = [
        _observation(
            source,
            metric,
            datetime(2026, 6, 4, 13, tzinfo=UTC),
            100.0,
            entity_id=entity_id,
        )
    ]
    day_values = ((14, 20.0), (15, 50.0), (16, 30.0), (17, 25.0))
    rows.extend(
        _observation(
            source,
            metric,
            datetime(2026, 6, 5, hour, tzinfo=UTC),
            value,
            entity_id=entity_id,
        )
        for hour, value in day_values[:anomaly_day_points]
    )
    return rows


def test_pair_ignores_neighbor_peak_and_uses_anomaly_day_points() -> None:
    result = assess_pair(
        _neighbor_peak_rows(),
        EVENT,
        source="openaq",
        metric="no2",
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.anomaly_day_point_count == 4
    assert result.whole_window_verdict == SUPPORTING
    assert result.anomaly_day_verdict == CONTRADICTING


def test_pair_neighbor_rows_cannot_rescue_below_floor_day() -> None:
    result = assess_pair(
        _neighbor_peak_rows(anomaly_day_points=3),
        EVENT,
        source="openaq",
        metric="no2",
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.anomaly_day_point_count == 3
    assert result.whole_window_verdict == SUPPORTING
    assert result.anomaly_day_verdict == SILENT


def test_pair_uses_only_verified_openaq_pm25_monitors() -> None:
    monitor_id = min(verified_monitor_entity_ids(), key=int)
    rows = _neighbor_peak_rows(
        source="openaq",
        metric="pm25",
        entity_id=monitor_id,
    )
    rows.append(
        _observation(
            "openaq",
            "pm25",
            datetime(2026, 6, 5, 13, tzinfo=UTC),
            999.0,
            entity_id="not-a-verified-monitor",
        )
    )

    result = assess_pair(
        rows,
        EVENT,
        source="openaq",
        metric="pm25",
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.anomaly_day_point_count == 4
    assert result.anomaly_day_verdict == CONTRADICTING


def test_pair_normalizes_naive_and_offset_aware_timestamps() -> None:
    aware = _neighbor_peak_rows()
    naive = [
        MobileObservation(
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

    aware_result = assess_pair(
        aware,
        EVENT,
        source="openaq",
        metric="no2",
        anchor_lat=LAT,
        anchor_lon=LON,
    )
    naive_result = assess_pair(
        naive,
        EVENT.replace(tzinfo=None),
        source="openaq",
        metric="no2",
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert naive_result == aware_result


def test_report_records_n_distribution_outcomes_and_transition() -> None:
    report = build_report(
        _neighbor_peak_rows(),
        snapshot_sha256="abc",
        anchors=(EVENT,),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert report.anchor_count == 1
    assert len(report.pairs) == 1
    pair = report.pairs[0]
    assert (pair.source, pair.metric) == ("openaq", "no2")
    assert pair.anomaly_day_point_counts.frequency == ((4, 1),)
    assert pair.anomaly_day_point_counts.below_floor_count == 0
    assert pair.whole_window_outcomes.supporting_count == 1
    assert pair.anomaly_day_outcomes.contradicting_count == 1
    assert pair.changed_verdict_count == 1
    assert pair.transition_counts == ((SUPPORTING, CONTRADICTING, 1),)


def test_report_rejects_duplicates_nonfinite_values_and_duplicate_anchors() -> None:
    row = _neighbor_peak_rows()[0]
    with pytest.raises(ValueError, match="duplicate relevant observation"):
        build_report(
            [row, row],
            snapshot_sha256="abc",
            anchors=(EVENT,),
            anchor_lat=LAT,
            anchor_lon=LON,
        )

    invalid = MobileObservation(
        **{**row.__dict__, "value": float("nan")},
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


def test_markdown_and_json_report_whole_window_and_local_day() -> None:
    report = build_report(
        _neighbor_peak_rows(),
        snapshot_sha256="abc",
        anchors=(EVENT,),
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    rendered = render_markdown(report)
    payload = report.to_dict()

    assert "| openaq | no2 | 4 | 4 | 4 | 4 | 0 (0.00%) |" in rendered
    assert "| Whole window | 1 (100.00%) | 0 (0.00%) | 0 (0.00%) |" in rendered
    assert "| Anomaly local day | 0 (0.00%) | 1 (100.00%) | 0 (0.00%) |" in rendered
    assert payload["pairs"][0]["changed_verdict_count"] == 1
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
            for row in _neighbor_peak_rows()
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
        mobile_source_empirics.run_empirics(database, expected_sha256="wrong")

    monkeypatch.setattr(
        mobile_source_empirics,
        "_snapshot_sha256",
        lambda _path: "mismatch",
    )
    with pytest.raises(ValueError, match="before read"):
        mobile_source_empirics.run_empirics(
            database,
            expected_sha256=LOCKED_SNAPSHOT_SHA256,
        )

    hashes = iter((LOCKED_SNAPSHOT_SHA256, "changed"))
    monkeypatch.setattr(
        mobile_source_empirics,
        "_snapshot_sha256",
        lambda _path: next(hashes),
    )
    with pytest.raises(RuntimeError, match="after read"):
        mobile_source_empirics.run_empirics(
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
        mobile_source_empirics,
        "_snapshot_sha256",
        lambda _path: LOCKED_SNAPSHOT_SHA256,
    )
    monkeypatch.setattr(
        mobile_source_empirics,
        "candidate_centers",
        lambda _start, _end: (EVENT,),
    )

    report = mobile_source_empirics.run_empirics(
        database,
        expected_sha256=LOCKED_SNAPSHOT_SHA256,
    )

    assert isinstance(report, MobileSourceEmpiricalReport)
    assert report.input_row_count == 5
    assert report.anchor_count == 1
    assert report.pairs[0].changed_verdict_count == 1

    monkeypatch.setattr(
        mobile_source_empirics,
        "run_empirics",
        lambda _path, **_kwargs: report,
    )
    exit_code = mobile_source_empirics.main(
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


def test_pair_excludes_rows_outside_window_and_radius() -> None:
    rows = _neighbor_peak_rows()
    rows.extend(
        (
            _observation(
                "openaq",
                "no2",
                EVENT + timedelta(hours=36, microseconds=1),
                999.0,
                entity_id="late",
            ),
            _observation(
                "openaq",
                "no2",
                datetime(2026, 6, 5, 13, tzinfo=UTC),
                999.0,
                entity_id="far",
                lat=31.0,
            ),
        )
    )

    result = assess_pair(
        rows,
        EVENT,
        source="openaq",
        metric="no2",
        anchor_lat=LAT,
        anchor_lon=LON,
    )

    assert result.anomaly_day_point_count == 4
    assert result.anomaly_day_verdict == CONTRADICTING
