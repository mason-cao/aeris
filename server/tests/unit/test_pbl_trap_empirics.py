"""B1 label-free same-cell PBL-reference empirics."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from app.eval import pbl_trap_empirics
from app.eval.pbl_trap_empirics import (
    PblObservation,
    PblTrapEmpiricalReport,
    assess_anchor,
    build_report,
    render_markdown,
)
from app.llm.corroboration import CONTRADICTING, SILENT, SUPPORTING
from app.provenance.openaq_pm25 import LOCKED_SNAPSHOT_SHA256


EVENT = datetime(2026, 6, 5, 18, tzinfo=UTC)


def _observation(
    entity_id: str,
    timestamp: datetime,
    value: float,
) -> PblObservation:
    return PblObservation(entity_id, timestamp, value)


def test_anchor_uses_same_cell_other_utc_dates_and_population_sd() -> None:
    observations = [
        _observation("cell-a", EVENT - timedelta(days=1), 100.0),
        _observation("cell-a", EVENT, 0.0),
        _observation("cell-a", EVENT + timedelta(days=1), 300.0),
        _observation("cell-b", EVENT - timedelta(days=1), 999.0),
        _observation("cell-a", EVENT + timedelta(hours=12), 999.0),
    ]

    result = assess_anchor(observations[1], observations)

    assert result.reference_n == 2
    assert result.reference_value_n == 2
    assert result.reference_mean == 200.0
    assert result.reference_pstdev == 100.0
    assert result.support_threshold == 0.0
    assert result.verdict == SUPPORTING
    assert result.silence_reason is None


def test_anchor_exact_mean_contradicts_and_open_band_is_silent() -> None:
    references = [
        _observation("cell-a", EVENT - timedelta(days=1), 100.0),
        _observation("cell-a", EVENT + timedelta(days=1), 300.0),
    ]

    at_mean = assess_anchor(
        _observation("cell-a", EVENT, 200.0),
        references,
    )
    inside_band = assess_anchor(
        _observation("cell-a", EVENT, 0.1),
        references,
    )

    assert at_mean.verdict == CONTRADICTING
    assert inside_band.verdict == SILENT
    assert inside_band.silence_reason == "between_threshold_and_mean"


def test_anchor_zero_spread_and_insufficient_days_are_silent() -> None:
    zero_spread = assess_anchor(
        _observation("cell-a", EVENT, 250.0),
        [
            _observation("cell-a", EVENT - timedelta(days=1), 300.0),
            _observation("cell-a", EVENT + timedelta(days=1), 300.0),
        ],
    )
    insufficient = assess_anchor(
        _observation("cell-a", EVENT, 0.0),
        [
            _observation("cell-a", EVENT - timedelta(days=1), 100.0),
            _observation("cell-b", EVENT + timedelta(days=1), 300.0),
        ],
    )

    assert zero_spread.verdict == SILENT
    assert zero_spread.silence_reason == "zero_spread"
    assert zero_spread.reference_n == 2
    assert insufficient.verdict == SILENT
    assert insufficient.silence_reason == "insufficient_distinct_days"
    assert insufficient.reference_n == 1


def test_anchor_normalizes_naive_and_offset_aware_timestamps_to_utc() -> None:
    anchor = _observation(
        "cell-a",
        datetime(2026, 6, 5, 18),
        0.0,
    )
    observations = [
        _observation(
            "cell-a",
            datetime(2026, 6, 4, 18, tzinfo=UTC),
            100.0,
        ),
        _observation(
            "cell-a",
            datetime.fromisoformat("2026-06-06T14:00:00-04:00"),
            300.0,
        ),
    ]

    result = assess_anchor(anchor, observations)

    assert result.reference_n == 2
    assert result.verdict == SUPPORTING


def test_report_uses_only_complete_window_anchors_and_rejects_duplicates() -> None:
    start = datetime(2026, 6, 1, tzinfo=UTC)
    end = datetime(2026, 6, 5, tzinfo=UTC)
    observations = [
        _observation("cell-a", start + timedelta(hours=35), 100.0),
        _observation("cell-a", start + timedelta(hours=36), 100.0),
        _observation("cell-a", end - timedelta(hours=37), 100.0),
        _observation("cell-a", end - timedelta(hours=36), 100.0),
    ]

    report = build_report(
        observations,
        snapshot_sha256="abc",
        study_start=start,
        study_end_exclusive=end,
    )

    assert report.observation_count == 4
    assert report.candidate_anchor_count == 2
    assert report.reference_n_counts == ((1, 2),)
    assert report.silent_count == 2
    assert report.silent_rate == 1.0

    duplicate = observations + [observations[1]]
    with pytest.raises(ValueError, match="duplicate PBL observation"):
        build_report(
            duplicate,
            snapshot_sha256="abc",
            study_start=start,
            study_end_exclusive=end,
        )


def test_markdown_reports_reference_distribution_and_outcome_rates() -> None:
    report = PblTrapEmpiricalReport(
        snapshot_sha256="abc",
        study_start="start",
        study_end_exclusive="end",
        observation_count=12,
        candidate_anchor_count=10,
        reference_n_counts=((1, 2), (2, 8)),
        insufficient_distinct_day_count=2,
        zero_spread_count=1,
        evaluable_count=7,
        supporting_count=1,
        contradicting_count=4,
        silent_count=5,
        supporting_rate=0.1,
        contradicting_rate=0.4,
        silent_rate=0.5,
    )

    rendered = render_markdown(report)

    assert "| 1 | 2 | 20.00% |" in rendered
    assert "| 2 | 8 | 80.00% |" in rendered
    assert "| Support | 1 | 10.00% |" in rendered
    assert "| Silent | 5 | 50.00% |" in rendered

    payload = report.to_dict()
    assert payload["reference_n_distribution"] == [
        {"reference_n": 1, "anchor_count": 2, "fraction": 0.2},
        {"reference_n": 2, "anchor_count": 8, "fraction": 0.8},
    ]
    assert payload["tolerance"] == {
        "suppression_sigma": 2.0,
        "min_same_hour_points": 2,
        "sd_estimator": "population",
    }


def test_invalid_observations_and_interval_fail_loudly() -> None:
    with pytest.raises(ValueError, match="entity_id"):
        assess_anchor(_observation("", EVENT, 1.0), [])
    with pytest.raises(ValueError, match="finite"):
        assess_anchor(_observation("cell-a", EVENT, float("nan")), [])
    with pytest.raises(ValueError, match="study end"):
        build_report(
            [],
            snapshot_sha256="abc",
            study_start=EVENT,
            study_end_exclusive=EVENT,
        )


def _sqlite_snapshot(path: str) -> None:
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
                metric TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO data_points
                (source_entity_id, timestamp, value, unit, source, metric)
            VALUES
                ('cell-a', '2026-06-05T18:00:00Z', 500.0, 'm',
                 'noaa_gfs', 'pbl_height')
            """
        )
        connection.commit()
    finally:
        connection.close()


def test_run_empirics_requires_canonical_hash_and_checks_both_sides(
    tmp_path,
    monkeypatch,
) -> None:
    database = tmp_path / "snapshot.db"
    _sqlite_snapshot(str(database))

    with pytest.raises(ValueError, match="canonical locked"):
        pbl_trap_empirics.run_empirics(database, expected_sha256="wrong")

    monkeypatch.setattr(
        pbl_trap_empirics,
        "_snapshot_sha256",
        lambda _path: "mismatch",
    )
    with pytest.raises(ValueError, match="before read"):
        pbl_trap_empirics.run_empirics(
            database,
            expected_sha256=LOCKED_SNAPSHOT_SHA256,
        )

    hashes = iter((LOCKED_SNAPSHOT_SHA256, "changed"))
    monkeypatch.setattr(
        pbl_trap_empirics,
        "_snapshot_sha256",
        lambda _path: next(hashes),
    )
    with pytest.raises(ValueError, match="after read"):
        pbl_trap_empirics.run_empirics(
            database,
            expected_sha256=LOCKED_SNAPSHOT_SHA256,
        )


def test_run_empirics_reads_query_only_sqlite_and_main_renders_both_formats(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    database = tmp_path / "snapshot.db"
    _sqlite_snapshot(str(database))
    monkeypatch.setattr(
        pbl_trap_empirics,
        "_snapshot_sha256",
        lambda _path: LOCKED_SNAPSHOT_SHA256,
    )

    report = pbl_trap_empirics.run_empirics(
        database,
        expected_sha256=LOCKED_SNAPSHOT_SHA256,
    )

    assert report.observation_count == 1
    assert report.candidate_anchor_count == 1
    assert report.silent_count == 1

    monkeypatch.setattr(
        pbl_trap_empirics,
        "run_empirics",
        lambda _path, *, expected_sha256: report,
    )
    pbl_trap_empirics.main(
        [
            "--database",
            str(database),
            "--expected-sha256",
            LOCKED_SNAPSHOT_SHA256,
            "--format",
            "json",
        ]
    )
    assert json.loads(capsys.readouterr().out)["observation_count"] == 1

    pbl_trap_empirics.main(
        [
            "--database",
            str(database),
            "--expected-sha256",
            LOCKED_SNAPSHOT_SHA256,
            "--format",
            "markdown",
        ]
    )
    assert "| Stored PBL rows | 1 |" in capsys.readouterr().out
