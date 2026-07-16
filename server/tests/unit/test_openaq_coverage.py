"""C4′ OpenAQ frozen-window coverage empirics."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.eval import openaq_coverage
from app.eval.openaq_coverage import (
    NON_MONITOR_SENSOR,
    UNMAPPABLE_ARCHIVE,
    VERIFIED_MONITOR,
    OpenAQObservation,
    build_report,
    derive_report,
    main,
    render_markdown,
    write_report,
)


START = datetime(2026, 6, 1, tzinfo=UTC)
END = datetime(2026, 6, 2, tzinfo=UTC)


def _inline(
    provider: str = "AirNow",
    *,
    is_monitor: bool = True,
    instruments: tuple[str, ...] = ("Government Monitor",),
) -> dict[str, object]:
    return {
        "location": {
            "provider": {"name": provider},
            "isMonitor": is_monitor,
            "instruments": [{"name": name} for name in instruments],
        },
        "sensor": {"id": 1},
    }


def _archive() -> dict[str, object]:
    return {"archive": {"sensors_id": 1}}


def _observation(
    entity_id: str,
    hour: float,
    *,
    metric: str = "ozone",
    unit: str = "ppm",
    raw_json: object | None = None,
    collected_hour: float | None = None,
) -> OpenAQObservation:
    timestamp = START + timedelta(hours=hour)
    return OpenAQObservation(
        entity_id=entity_id,
        timestamp=timestamp,
        collected_at=START
        + timedelta(hours=collected_hour if collected_hour is not None else hour + 1),
        metric=metric,
        unit=unit,
        raw_json=_inline() if raw_json is None else raw_json,
    )


def _build(
    observations: list[OpenAQObservation],
    *,
    cutoff_hour: float = 4.5,
    expected_pm25: dict[str, str] | None = None,
) -> dict[str, object]:
    return build_report(
        observations,
        snapshot_sha256="abc",
        snapshot_max_collected_at=START + timedelta(hours=cutoff_hour),
        expected_pm25_classifications=(
            {} if expected_pm25 is None else expected_pm25
        ),
        study_start=START,
        study_end_exclusive=END,
    )


def test_report_separates_roster_active_span_and_ingest_paths() -> None:
    report = _build(
        [
            _observation("airnow", 0.1),
            _observation("airnow", 0.8, raw_json=_archive()),
            _observation("airnow", 2.0),
            _observation(
                "clarity",
                3.0,
                raw_json=_inline(
                    "Clarity",
                    is_monitor=False,
                    instruments=("Clarity Node",),
                ),
            ),
        ]
    )

    overall = report["overall_summary"]
    assert overall["row_count"] == 4
    assert overall["observed_entity_hours"] == 3
    assert overall["roster_expected_entity_hours"] == 8
    assert overall["roster_coverage_fraction"] == 3 / 8
    assert overall["active_expected_entity_hours"] == 4
    assert overall["active_coverage_fraction"] == 3 / 4
    assert overall["missing_active_entity_hours"] == 1
    assert overall["longest_internal_gap_hours"] == 1
    assert overall["inline_contributed_entity_hours"] == 3
    assert overall["archive_contributed_entity_hours"] == 1
    assert overall["archive_only_entity_hours"] == 0
    assert overall["both_path_entity_hours"] == 1
    assert overall["overlap_row_count"] == 1

    summaries = {
        (row["entity_class"], row["provider"]): row
        for row in report["group_summaries"]
    }
    monitor = summaries[(VERIFIED_MONITOR, "AirNow")]
    assert monitor["observed_entity_hours"] == 2
    assert monitor["active_expected_entity_hours"] == 3
    assert monitor["roster_expected_entity_hours"] == 4
    sensor = summaries[(NON_MONITOR_SENSOR, "Clarity")]
    assert sensor["observed_entity_hours"] == 1
    assert sensor["active_expected_entity_hours"] == 1


def test_archive_only_entity_is_unmappable_and_archive_only() -> None:
    report = _build([_observation("archive-only", 1.0, raw_json=_archive())])

    summary = report["group_summaries"][0]
    assert summary["entity_class"] == UNMAPPABLE_ARCHIVE
    assert summary["provider"] is None
    assert summary["archive_only_entity_hours"] == 1
    assert summary["inline_contributed_entity_hours"] == 0
    assert summary["active_coverage_fraction"] == 1.0


def test_hour_boundary_dedup_and_partial_final_hour_are_exact() -> None:
    report = _build(
        [
            _observation("a", 0.25),
            _observation("a", 0.99, raw_json=_archive()),
            _observation("a", 1.0),
            _observation("a", 3.0),
        ],
        cutoff_hour=3.75,
    )

    assert report["coverage_window"]["end_exclusive"] == "2026-06-01T03:00:00Z"
    assert report["input_row_count"] == 4
    assert report["analyzed_row_count"] == 3
    assert report["incomplete_trailing_hour_row_count"] == 1
    daily = report["daily_coverage"][0]
    assert daily["completed_clock_hours"] == 3
    assert daily["row_count"] == 3
    assert daily["observed_entity_hours"] == 2
    assert daily["overlap_row_count"] == 1
    assert daily["both_path_entity_hours"] == 1


def test_archive_rows_map_to_inline_entity_metadata() -> None:
    report = _build(
        [
            _observation(
                "gradient",
                0.0,
                raw_json=_inline(
                    "AirGradient", is_monitor=False, instruments=("N/A",)
                ),
            ),
            _observation("gradient", 1.0, raw_json=_archive()),
            _observation(
                "gradient",
                2.0,
                raw_json=_inline(
                    "AirGradient",
                    is_monitor=False,
                    instruments=("Unknown AirGradient Sensor",),
                ),
            ),
        ]
    )

    assert report["group_summaries"][0]["entity_class"] == NON_MONITOR_SENSOR
    assert report["group_summaries"][0]["provider"] == "AirGradient"
    assert report["group_summaries"][0]["archive_only_entity_hours"] == 1


@pytest.mark.parametrize(
    ("observations", "message"),
    [
        (
            [
                _observation(
                    "other",
                    0.0,
                    raw_json=_inline("Other", is_monitor=True),
                )
            ],
            "undeclared provider/monitor/instrument combination",
        ),
        (
            [
                _observation("conflict", 0.0),
                _observation(
                    "conflict",
                    1.0,
                    raw_json=_inline("Clarity", is_monitor=False),
                ),
            ],
            "conflicting provider metadata",
        ),
        (
            [_observation("a", 0.0, raw_json={"location": {}, "archive": {}})],
            "exactly one ingest path",
        ),
        ([_observation("", 0.0)], "nonempty entity ID"),
        ([_observation("a", 0.0, raw_json="not-json")], "raw_json must be an object"),
    ],
)
def test_invalid_entity_metadata_or_rows_fail_loudly(
    observations: list[OpenAQObservation], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _build(observations)


def test_a8_single_unit_assertion_is_exact_per_metric() -> None:
    report = _build(
        [
            _observation("ozone-a", 0.0, metric="ozone", unit="ppm"),
            _observation("pm-a", 0.0, metric="pm10", unit="ug/m3"),
        ]
    )
    assert report["unit_by_metric"] == {"ozone": "ppm", "pm10": "ug/m3"}

    with pytest.raises(ValueError, match="ozone has multiple units"):
        _build(
            [
                _observation("a", 0.0, unit="ppm"),
                _observation("b", 1.0, unit="ppb"),
            ]
        )


def test_pm25_entities_and_classes_must_exactly_match_b6_fixture() -> None:
    pm25 = _observation("pm", 0.0, metric="pm25", unit="ug/m3")
    report = _build([pm25], expected_pm25={"pm": VERIFIED_MONITOR})
    assert report["b6_pm25_fixture_match"] is True

    with pytest.raises(ValueError, match="PM2.5 class drift"):
        _build([pm25], expected_pm25={"pm": NON_MONITOR_SENSOR})
    with pytest.raises(ValueError, match="PM2.5 entity-set drift"):
        _build([pm25], expected_pm25={"other": VERIFIED_MONITOR})


def test_naive_and_offset_aware_timestamps_normalize_to_utc() -> None:
    naive = _observation("a", 0.0)
    aware = _observation("a", 1.0)
    naive = OpenAQObservation(
        **{
            **naive.__dict__,
            "timestamp": START.replace(tzinfo=None),
            "collected_at": (START + timedelta(hours=1)).replace(tzinfo=None),
        }
    )
    aware = OpenAQObservation(
        **{
            **aware.__dict__,
            "timestamp": datetime.fromisoformat("2026-05-31T20:00:00-05:00"),
        }
    )

    report = _build([naive, aware])
    assert report["overall_summary"]["observed_entity_hours"] == 2


def test_empty_rows_and_out_of_window_timestamp_fail() -> None:
    with pytest.raises(ValueError, match="at least one OpenAQ row"):
        _build([])

    outside = _observation("a", 0.0)
    outside = OpenAQObservation(
        **{**outside.__dict__, "timestamp": END}
    )
    with pytest.raises(ValueError, match="outside declared study window"):
        _build([outside])


def test_report_json_and_markdown_are_deterministic(tmp_path: Path) -> None:
    report = _build([_observation("a", 0.0), _observation("a", 2.0)])
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"

    write_report(report, first)
    write_report(report, second)

    assert first.read_bytes() == second.read_bytes()
    assert json.loads(first.read_text(encoding="utf-8")) == report
    assert render_markdown(report) == render_markdown(report)
    assert (
        "| 2026-06-01 | ozone | ppm | verified_monitor | AirNow |"
        in render_markdown(report)
    )


def _sqlite_snapshot(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE data_points (
                source_entity_id TEXT,
                timestamp TEXT,
                collected_at TEXT,
                metric TEXT,
                unit TEXT,
                source TEXT,
                raw_json TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO data_points
                (source_entity_id, timestamp, collected_at, metric, unit,
                 source, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "a",
                    "2026-06-01T00:15:00Z",
                    "2026-06-01T01:05:00Z",
                    "ozone",
                    "ppm",
                    "openaq",
                    json.dumps(_inline()),
                ),
                (
                    "met",
                    "2026-06-01T04:00:00Z",
                    "2026-06-01T04:15:00Z",
                    "wind_speed",
                    "m/s",
                    "asos",
                    "{}",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sqlite_report_is_hash_guarded_and_uses_snapshot_collection_cutoff(
    tmp_path: Path,
) -> None:
    database = tmp_path / "snapshot.db"
    _sqlite_snapshot(database)
    expected_hash = _sha256(database)

    report = derive_report(
        database,
        expected_sha256=expected_hash,
        expected_pm25_classifications={},
        study_start=START,
        study_end_exclusive=END,
    )

    assert _sha256(database) == expected_hash
    assert report["snapshot_sha256"] == expected_hash
    assert report["coverage_window"]["end_exclusive"] == "2026-06-01T04:00:00Z"
    assert report["overall_summary"]["roster_expected_entity_hours"] == 4

    with pytest.raises(ValueError, match="mismatch before read"):
        derive_report(
            database,
            expected_sha256="0" * 64,
            expected_pm25_classifications={},
            study_start=START,
            study_end_exclusive=END,
        )


def test_post_read_hash_mismatch_stops_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "snapshot.db"
    _sqlite_snapshot(database)
    expected_hash = _sha256(database)
    hashes = iter((expected_hash, "f" * 64))
    monkeypatch.setattr(openaq_coverage, "_sha256", lambda _: next(hashes))

    with pytest.raises(RuntimeError, match="mismatch after read"):
        derive_report(
            database,
            expected_sha256=expected_hash,
            expected_pm25_classifications={},
            study_start=START,
            study_end_exclusive=END,
        )


def test_cli_failure_does_not_create_output(tmp_path: Path) -> None:
    output = tmp_path / "coverage.json"

    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--database",
                str(tmp_path / "missing.db"),
                "--expected-sha256",
                "0" * 64,
                "--output",
                str(output),
            ]
        )

    assert not output.exists()
