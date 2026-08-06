"""Label-free cross-source wind-direction disagreement empirics."""

from __future__ import annotations

import json

import pytest

from app.eval.wind_disagreement_empirics import (
    CONSUMER_SOURCES,
    SWEEP_THRESHOLDS_DEG,
    EventMeasurement,
    measure_event,
    measured_bearings,
    render_markdown,
    run_empirics,
    summarize,
)
from app.llm.corroboration import DEFAULT_WIND_TOLERANCE

_T0 = "2026-06-15T00:00:00+00:00"
_T1 = "2026-06-15T01:00:00+00:00"


def _metric(nearest_value: float, series: list[tuple[str, float]]) -> dict:
    return {
        "nearest_in_time": {
            "t": _T0,
            "v": nearest_value,
            "entity_id": "station-a",
            "dt_minutes": 0.0,
        },
        "entities": [{"entity_id": "station-a", "series": [list(p) for p in series]}],
    }


_SPEEDS = [(_T0, 6.0), (_T1, 10.0)]


def _summary(directions: dict[str, float]) -> dict:
    sources = {
        source: {
            "metrics": {
                "wind_direction": _metric(bearing, [(_T0, bearing), (_T1, bearing)]),
                "wind_speed": _metric(8.0, _SPEEDS),
            }
        }
        for source, bearing in directions.items()
    }
    return {"anomaly": {"lat": 29.76, "lon": -95.37}, "sources": sources}


def test_bearings_come_from_the_production_helpers() -> None:
    summary = _summary({"openweather": 190.0, "asos": 10.0})

    bearings = measured_bearings(summary, ("openweather", "asos"))

    assert bearings == {"openweather": 190.0, "asos": 10.0}


def test_measure_event_reports_separation_without_applying_a_threshold() -> None:
    # The sweep applies thresholds afterwards, so the measurement itself must
    # stay threshold-free or every sweep row would collapse to the shipped one.
    summary = _summary({"openweather": 190.0, "asos": 10.0})

    measurement = measure_event("anomaly-1", summary, ("openweather", "asos"))

    assert measurement.max_pairwise_deg == pytest.approx(180.0)
    assert measurement.votable_sources == ("asos", "openweather")
    assert measurement.worst_pair == ("asos", "openweather")


def test_calm_sources_drop_out_of_the_votable_set() -> None:
    summary = _summary({"openweather": 190.0, "asos": 10.0})
    summary["sources"]["asos"]["metrics"]["wind_speed"] = _metric(
        0.0, [(_T0, 0.0), (_T1, 0.0)]
    )

    measurement = measure_event("anomaly-1", summary, ("openweather", "asos"))

    assert measurement.measured_sources == ("asos", "openweather")
    assert measurement.votable_sources == ("openweather",)
    assert measurement.max_pairwise_deg is None


def test_sweep_counts_are_strictly_above_each_threshold() -> None:
    measurements = [
        EventMeasurement("a", ("x", "y"), ("x", "y"), 45.0, ("x", "y")),
        EventMeasurement("b", ("x", "y"), ("x", "y"), 90.0, ("x", "y")),
        EventMeasurement("c", ("x", "y"), ("x", "y"), 91.0, ("x", "y")),
        EventMeasurement("d", ("x",), ("x",), None, None),
    ]

    empirics = summarize("transport_direction", "frozen", ("x", "y"), measurements, 0)
    by_threshold = {row.threshold_deg: row for row in empirics.sweep}

    assert empirics.comparable_events == 3
    assert empirics.total_events == 4
    # Exactly at the threshold is not silenced; the guard is inclusive there.
    assert by_threshold[90.0].silenced_events == 1
    assert by_threshold[45.0].silenced_events == 2
    assert by_threshold[180.0].silenced_events == 0
    assert by_threshold[90.0].share_of_comparable == pytest.approx(1 / 3)
    assert by_threshold[90.0].share_of_total == pytest.approx(1 / 4)


def test_missing_enrichment_counts_toward_totals_not_comparables() -> None:
    empirics = summarize("transport_direction", "frozen", ("x", "y"), [], 7)

    assert empirics.total_events == 7
    assert empirics.events_missing_enrichment == 7
    assert empirics.comparable_events == 0
    assert empirics.median_deg is None
    assert all(row.share_of_comparable is None for row in empirics.sweep)


def test_sweep_grid_brackets_the_shipped_threshold() -> None:
    assert DEFAULT_WIND_TOLERANCE.max_disagreement_deg in SWEEP_THRESHOLDS_DEG
    assert min(SWEEP_THRESHOLDS_DEG) == DEFAULT_WIND_TOLERANCE.bearing_deg
    assert max(SWEEP_THRESHOLDS_DEG) == 180.0


def test_consumers_match_the_scorers_they_stand_in_for() -> None:
    assert CONSUMER_SOURCES["transport_direction"] == (
        "noaa_gfs",
        "openweather",
        "asos",
    )
    assert CONSUMER_SOURCES["point_source_attribution"] == ("noaa_gfs", "openweather")


def test_markdown_renders_every_population_and_sweep_row() -> None:
    from app.eval.wind_disagreement_empirics import WindDisagreementReport

    empirics = summarize(
        "transport_direction",
        "frozen",
        ("x", "y"),
        [EventMeasurement("a", ("x", "y"), ("x", "y"), 120.0, ("x", "y"))],
        0,
    )
    report = WindDisagreementReport(
        analysis_db_sha256="a" * 64,
        fixture_snapshot_sha256="b" * 64,
        locked_snapshot_sha256="b" * 64,
        study_start="2026-06-01T00:00:00+00:00",
        study_end_exclusive="2026-08-05T00:00:00+00:00",
        declared_threshold_deg=90.0,
        guard_manifest={},
        populations=(empirics,),
    )

    rendered = render_markdown(report)
    sweep_section = rendered.split("### Threshold sweep")[1].split("###")[0]

    # A 120-degree separation is silenced at 90 and survives at 120, so the
    # rendered rows carry the inclusive boundary through to the report.
    assert "| transport_direction | frozen | 90 | 1 |" in sweep_section
    assert "| transport_direction | frozen | 120 | 0 |" in sweep_section
    assert sweep_section.count("| transport_direction | frozen |") == len(
        SWEEP_THRESHOLDS_DEG
    )
    assert json.dumps(report.to_dict())


def test_hash_mismatch_refuses_to_read(tmp_path) -> None:
    database = tmp_path / "analysis.db"
    database.write_bytes(b"not a database")

    with pytest.raises(ValueError, match="before read"):
        run_empirics(
            database,
            expected_sha256="0" * 64,
            anomaly_set=None,
            populations=("window",),
        )


def test_fixture_bound_to_a_foreign_snapshot_is_rejected(tmp_path) -> None:
    import hashlib

    database = tmp_path / "analysis.db"
    database.write_bytes(b"not a database")
    digest = hashlib.sha256(b"not a database").hexdigest()
    fixture = tmp_path / "eval.json"
    fixture.write_text(json.dumps({"snapshot_sha256": "f" * 64, "anomaly_ids": []}))

    with pytest.raises(ValueError, match="not the locked snapshot"):
        run_empirics(
            database,
            expected_sha256=digest,
            anomaly_set=fixture,
            populations=("frozen",),
        )
