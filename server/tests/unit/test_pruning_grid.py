"""Ratified B3/D4 prompt-only grid screen."""

from __future__ import annotations

import json
import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from app.eval.pruning_screen import (
    D4_STATISTICAL_CAVEATS,
    LAGS_HOURS,
    POLLUTANT_GRID,
    HourlyObservation,
    MechanismAssessment,
    PruningThresholds,
    SnapshotPruningInput,
    build_hourly_series,
    build_metric_scope,
    build_snapshot_manifest,
    candidate_statistics,
    deseasonalize_hourly,
    evaluate_pruning_grid,
    finalize_grid_decisions,
    load_mechanism_assessments,
    load_pruning_fixture,
    load_snapshot_inputs,
    main,
    metric_is_retained,
    pair_leading_series,
    pruning_manifest_payload,
    write_numbered_manifest,
    _spearman_statistic,
)
from app.llm.corroboration import phase2_metric_owners
from app.provenance.openaq_pm25 import verified_monitor_entity_ids
from app.provenance.purpleair_qc import LOCKED_SNAPSHOT_SHA256


T0 = datetime(2026, 6, 1, tzinfo=UTC)
FAST_THRESHOLDS = PruningThresholds(
    bootstrap_resamples=49,
    bootstrap_seed=20260716,
    min_pairs=3,
)


def _observation(
    source: str,
    metric: str,
    entity: str,
    hour: int,
    value: float,
    *,
    unit: str = "unit",
) -> HourlyObservation:
    return HourlyObservation(
        source=source,
        metric=metric,
        entity_id=entity,
        timestamp=T0 + timedelta(hours=hour),
        value=value,
        unit=unit,
    )


def _passing_cell(
    candidate_source: str,
    candidate_metric: str,
    pollutant_source: str,
    pollutant_metric: str,
    lag_hours: int,
) -> dict[str, object]:
    return {
        "candidate_source": candidate_source,
        "candidate_metric": candidate_metric,
        "pollutant_source": pollutant_source,
        "pollutant_metric": pollutant_metric,
        "lag_hours": lag_hours,
        "eligible_pair_count": 100,
        "rho": 0.049,
        "p_value": 0.20,
        "ci_low": -0.1,
        "ci_high": 0.0,
        "evaluable": True,
        "unevaluable_reason": None,
        "nonsignificant": True,
        "negligible": True,
        "ci_covers_zero": True,
        "statistical_drop_condition": True,
        "bootstrap_substream": {
            "master_seed": 20260716,
            "spawn_index": 0,
            "spawn_key": [0],
        },
    }


def _all_passing_cells(
    source: str = "noaa_gfs",
    metric: str = "gh_500",
) -> list[dict[str, object]]:
    return [
        _passing_cell(source, metric, pollutant_source, pollutant_metric, lag)
        for pollutant_source, pollutant_metric in POLLUTANT_GRID
        for lag in LAGS_HOURS
    ]


def test_scorer_owned_inventory_exempts_live_phase2_metrics() -> None:
    owners = phase2_metric_owners()

    assert "score_atmospheric_trap" in owners[("noaa_gfs", "pbl_height")]
    assert "score_transport_direction" in owners[("noaa_gfs", "u_10m")]
    assert "score_secondary_formation" in owners[("openweather", "cloud_cover")]
    assert "score_concentration_elevation" in owners[("openaq", "pm25")]
    assert ("noaa_gfs", "gh_500") not in owners
    assert ("noaa_gfs", "precipitable_water") not in owners


def test_metric_scope_is_exact_rendered_minus_code_owned_inventory() -> None:
    rendered = [
        {"source": "noaa_gfs", "metric": "gh_500", "unit": "m"},
        {"source": "noaa_gfs", "metric": "pbl_height", "unit": "m"},
        {"source": "openweather", "metric": "humidity", "unit": "percent"},
        {"source": "openweather", "metric": "wind_speed", "unit": "m/s"},
    ]

    scope = build_metric_scope(rendered, scorer_owners=phase2_metric_owners())

    assert [(row["source"], row["metric"]) for row in scope["candidates"]] == [
        ("noaa_gfs", "gh_500"),
        ("openweather", "humidity"),
    ]
    assert [(row["source"], row["metric"]) for row in scope["exempt_rendered"]] == [
        ("noaa_gfs", "pbl_height"),
        ("openweather", "wind_speed"),
    ]
    assert scope["exempt_rendered"][0]["owners"]


def test_metric_scope_rejects_multiple_units_for_one_rendered_metric() -> None:
    rendered = [
        {"source": "noaa_gfs", "metric": "gh_500", "unit": "m"},
        {"source": "noaa_gfs", "metric": "gh_500", "unit": "dam"},
    ]

    with pytest.raises(ValueError, match="multiple rendered units"):
        build_metric_scope(rendered, scorer_owners={})


def test_hourly_series_collapse_entity_duplicates_before_cross_entity_aggregate() -> None:
    observations = [
        _observation("openaq", "pm25", "monitor-a", 0, 10.0),
        _observation("openaq", "pm25", "monitor-a", 0, 20.0),
        _observation("openaq", "pm25", "monitor-b", 0, 30.0),
        _observation("noaa_gfs", "gh_500", "cell-a", 0, 1.0),
        _observation("noaa_gfs", "gh_500", "cell-a", 0, 9.0),
        _observation("noaa_gfs", "gh_500", "cell-b", 0, 3.0),
    ]

    pollutants, candidates = build_hourly_series(
        observations,
        pollutant_keys={("openaq", "pm25")},
        candidate_keys={("noaa_gfs", "gh_500")},
    )

    assert pollutants[("openaq", "pm25")][T0] == pytest.approx(22.5)
    assert candidates[("noaa_gfs", "gh_500")][T0] == pytest.approx(4.0)


def test_deseasonalization_removes_window_wide_utc_hour_means() -> None:
    series = {
        T0 + timedelta(hours=hour): float(hour % 24 + 2 * (hour // 24))
        for hour in range(48)
    }

    residuals = deseasonalize_hourly(series)

    assert residuals[T0] == pytest.approx(-1.0)
    assert residuals[T0 + timedelta(hours=24)] == pytest.approx(1.0)
    for utc_hour in range(24):
        values = [
            value for timestamp, value in residuals.items() if timestamp.hour == utc_hour
        ]
        assert sum(values) / len(values) == pytest.approx(0.0)


def test_lag_means_candidate_leads_pollutant() -> None:
    pollutant = {T0 + timedelta(hours=6): 10.0}
    candidate = {T0: 5.0}

    x, y = pair_leading_series(pollutant, candidate, lag_hours=6)

    assert x.tolist() == [5.0]
    assert y.tolist() == [10.0]


def test_exact_100_pair_floor_is_evaluable_and_99_is_not() -> None:
    rng = np.random.default_rng(17)
    x100 = rng.normal(size=100)
    y100 = rng.normal(size=100)
    thresholds = PruningThresholds(bootstrap_resamples=49)

    at_floor = candidate_statistics(
        y100,
        x100,
        rng=np.random.default_rng(3),
        thresholds=thresholds,
    )
    below_floor = candidate_statistics(
        y100[:-1],
        x100[:-1],
        rng=np.random.default_rng(3),
        thresholds=thresholds,
    )

    assert at_floor.evaluable is True
    assert at_floor.eligible_pair_count == 100
    assert below_floor.evaluable is False
    assert below_floor.unevaluable_reason == "eligible n < 100"


def test_vectorized_bootstrap_statistic_matches_scipy_with_ties() -> None:
    first = np.asarray([[1.0, 1.0, 2.0, 3.0], [4.0, 3.0, 2.0, 1.0]])
    second = np.asarray([[1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0]])

    actual = _spearman_statistic(first, second, axis=-1)

    from scipy import stats

    expected = np.asarray(
        [stats.spearmanr(x, y).statistic for x, y in zip(first, second, strict=True)]
    )
    assert actual == pytest.approx(expected)


def test_grid_has_six_pollutants_by_four_lags_and_sorted_seed_substreams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def planted_statistics(*args: object, **kwargs: object) -> object:
        from app.eval.pruning_screen import CandidateStatistics

        first = args[0]
        return CandidateStatistics(
            eligible_pair_count=len(first),
            rho=0.01,
            p_value=0.5,
            ci_low=-0.1,
            ci_high=0.1,
            evaluable=True,
            unevaluable_reason=None,
        )

    monkeypatch.setattr(
        "app.eval.pruning_screen.candidate_statistics", planted_statistics
    )
    hours = [T0 + timedelta(hours=index) for index in range(130)]
    pollutant_series = {
        key: {timestamp: float(index % 17) for index, timestamp in enumerate(hours)}
        for key in POLLUTANT_GRID
    }
    candidate_series = {
        ("noaa_gfs", "gh_500"): {
            timestamp: float((index * 7) % 19)
            for index, timestamp in enumerate(hours)
        }
    }

    cells = evaluate_pruning_grid(
        pollutant_series,
        candidate_series,
        thresholds=FAST_THRESHOLDS,
    )

    assert len(cells) == 24
    assert [cell["bootstrap_substream"]["spawn_index"] for cell in cells] == list(
        range(24)
    )
    assert {
        (cell["pollutant_source"], cell["pollutant_metric"], cell["lag_hours"])
        for cell in cells
    } == {
        (source, metric, lag)
        for source, metric in POLLUTANT_GRID
        for lag in LAGS_HOURS
    }


def test_one_relevant_or_inevaluable_cell_keeps_whole_variable() -> None:
    relevant_cells = _all_passing_cells()
    relevant_cells[-1]["statistical_drop_condition"] = False
    relevant_cells[-1]["negligible"] = False
    relevant_cells[-1]["rho"] = 0.05
    inevaluable_cells = _all_passing_cells(metric="precipitable_water")
    inevaluable_cells[0].update(
        evaluable=False,
        unevaluable_reason="eligible n < 100",
        nonsignificant=None,
        negligible=None,
        ci_covers_zero=None,
        statistical_drop_condition=False,
    )

    result = finalize_grid_decisions(
        relevant_cells + inevaluable_cells,
        assessments={},
    )
    rows = {(row["source"], row["metric"]): row for row in result["variables"]}

    assert rows[("noaa_gfs", "gh_500")]["decision"] == "keep"
    assert rows[("noaa_gfs", "gh_500")]["statistical_cells_passing"] == 23
    assert rows[("noaa_gfs", "precipitable_water")]["decision"] == "keep"
    assert rows[("noaa_gfs", "precipitable_water")]["inevaluable_cell_count"] == 1
    assert result["mechanism_review_complete"] is True


def test_all_cells_pass_still_requires_unambiguous_mechanism_veto() -> None:
    cells = _all_passing_cells()
    key = "noaa_gfs/gh_500"

    relevant = finalize_grid_decisions(
        cells,
        assessments={
            key: MechanismAssessment(
                relevant=True,
                ambiguous=False,
                assessment="Synoptic ridging and subsidence can promote stagnation.",
            )
        },
    )
    ambiguous = finalize_grid_decisions(
        cells,
        assessments={
            key: MechanismAssessment(
                relevant=False,
                ambiguous=True,
                assessment="The mechanism assessment remains ambiguous.",
            )
        },
    )
    no_mechanism = finalize_grid_decisions(
        cells,
        assessments={
            key: MechanismAssessment(
                relevant=False,
                ambiguous=False,
                assessment="No standard atmospheric mechanism applies.",
            )
        },
    )
    missing = finalize_grid_decisions(cells, assessments={})

    assert relevant["variables"][0]["decision"] == "keep"
    assert ambiguous["variables"][0]["decision"] == "keep"
    assert no_mechanism["variables"][0]["decision"] == "drop"
    assert missing["variables"][0]["decision"] == "keep"
    assert missing["mechanism_review_complete"] is False


def test_statistical_caveats_are_exact_and_keep_biased() -> None:
    assert D4_STATISTICAL_CAVEATS == (
        "uncorrected multiplicity is keep-biased under the all-cells rule",
        "autocorrelation can inflate significance and is keep-biased here",
        "iid-bootstrap confidence intervals may be too narrow and are keep-biased here",
    )


def test_numbered_manifest_is_exclusive_and_canonical(tmp_path: Path) -> None:
    path = tmp_path / "pruning-screen-run-001.json"
    manifest = {"schema_version": 2, "run_number": 1, "variables": []}

    write_numbered_manifest(manifest, path, run_number=1)
    first = path.read_bytes()

    assert first == (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        write_numbered_manifest(manifest, path, run_number=1)
    with pytest.raises(ValueError, match="run number"):
        write_numbered_manifest(manifest, tmp_path / "bad.json", run_number=2)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_snapshot_loader_is_hash_guarded_and_marks_exact_b9_entity_eligibility(
    tmp_path: Path,
) -> None:
    database = tmp_path / "snapshot.db"
    eligible = sorted(verified_monitor_entity_ids("pm25"))[0]
    with sqlite3.connect(database) as connection:
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
            [
                (
                    "openaq",
                    "pm25",
                    eligible,
                    T0.isoformat(),
                    10.0,
                    "ug/m3",
                    29.7604,
                    -95.3698,
                ),
                (
                    "openaq",
                    "pm25",
                    "clarity-not-eligible",
                    T0.isoformat(),
                    999.0,
                    "ug/m3",
                    29.7604,
                    -95.3698,
                ),
                (
                    "noaa_gfs",
                    "gh_500",
                    "grid-cell",
                    T0.isoformat(),
                    5840.0,
                    "m",
                    29.7604,
                    -95.3698,
                ),
            ],
        )
    expected_hash = _sha256(database)

    loaded = load_snapshot_inputs(database, expected_sha256=expected_hash)

    assert _sha256(database) == expected_hash
    assert [(row["source"], row["metric"]) for row in loaded.rendered_metrics] == [
        ("noaa_gfs", "gh_500"),
        ("openaq", "pm25"),
    ]
    openaq = [row for row in loaded.observations if row.source == "openaq"]
    assert [row.nomination_eligible for row in openaq] == [True, False]


def test_snapshot_manifest_contains_scope_inputs_cells_and_mechanism_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def planted_statistics(*args: object, **kwargs: object) -> object:
        from app.eval.pruning_screen import CandidateStatistics

        first = args[0]
        return CandidateStatistics(
            eligible_pair_count=len(first),
            rho=0.01,
            p_value=0.5,
            ci_low=-0.1,
            ci_high=0.1,
            evaluable=True,
            unevaluable_reason=None,
        )

    monkeypatch.setattr(
        "app.eval.pruning_screen.candidate_statistics", planted_statistics
    )
    observations: list[HourlyObservation] = []
    for hour in range(130):
        for source, metric in POLLUTANT_GRID:
            observations.append(
                HourlyObservation(
                    source=source,
                    metric=metric,
                    entity_id=f"{source}-{metric}",
                    timestamp=T0 + timedelta(hours=hour),
                    value=float((hour * 5 + len(metric)) % 23),
                    unit="pollutant-unit",
                    nomination_eligible=True,
                )
            )
        observations.append(
            HourlyObservation(
                source="noaa_gfs",
                metric="gh_500",
                entity_id="grid-cell",
                timestamp=T0 + timedelta(hours=hour),
                value=float(5800 + (hour * 7) % 31),
                unit="m",
                nomination_eligible=False,
            )
        )
        observations.append(
            HourlyObservation(
                source="noaa_gfs",
                metric="pbl_height",
                entity_id="grid-cell",
                timestamp=T0 + timedelta(hours=hour),
                value=float(500 + hour % 10),
                unit="m",
                nomination_eligible=False,
            )
        )
    loaded = SnapshotPruningInput(
        snapshot_sha256="a" * 64,
        observations=tuple(observations),
        rendered_metrics=tuple(
            [
                {"source": source, "metric": metric, "unit": "pollutant-unit"}
                for source, metric in POLLUTANT_GRID
            ]
            + [
                {"source": "noaa_gfs", "metric": "gh_500", "unit": "m"},
                {"source": "noaa_gfs", "metric": "pbl_height", "unit": "m"},
            ]
        ),
        input_row_count=len(observations),
        finite_in_radius_row_count=len(observations),
        quality_excluded_row_count=0,
    )

    manifest = build_snapshot_manifest(
        loaded,
        run_number=1,
        assessments={
            "noaa_gfs/gh_500": MechanismAssessment(
                relevant=True,
                ambiguous=False,
                assessment="gh_500 (synoptic ridging/subsidence -> stagnation)",
            )
        },
        thresholds=FAST_THRESHOLDS,
    )

    assert manifest["schema_version"] == 2
    assert manifest["real_screen_executed"] is True
    assert manifest["run_number"] == 1
    assert manifest["statistical_caveats"] == list(D4_STATISTICAL_CAVEATS)
    assert len(manifest["cells"]) == 24
    assert manifest["mechanism_review_complete"] is True
    assert manifest["drop_metric_keys"] == []
    assert manifest["variables"][0]["decision"] == "keep"
    assert manifest["inventories"]["candidates"][0]["metric"] == "gh_500"
    assert any(
        row["metric"] == "pbl_height"
        for row in manifest["inventories"]["exempt_rendered"]
    )
    assert len(manifest["hourly_inputs"]["pollutants"]) == 6
    assert len(manifest["hourly_inputs"]["candidates"]) == 1


def test_renderer_policy_refuses_exempt_drop_and_applies_candidate_drop() -> None:
    manifest = {
        "schema_version": 2,
        "real_screen_executed": True,
        "mechanism_review_complete": True,
        "inventories": {
            "exempt_rendered": [
                {
                    "source": "noaa_gfs",
                    "metric": "pbl_height",
                    "unit": "m",
                    "owners": ["score_atmospheric_trap"],
                }
            ]
        },
        "drop_metric_keys": ["noaa_gfs/gh_500"],
    }

    assert metric_is_retained("noaa_gfs", "gh_500", manifest=manifest) is False
    assert metric_is_retained("noaa_gfs", "pbl_height", manifest=manifest) is True

    manifest["drop_metric_keys"] = ["noaa_gfs/pbl_height"]
    with pytest.raises(ValueError, match="scorer-exempt metric"):
        metric_is_retained("noaa_gfs", "pbl_height", manifest=manifest)


def test_mechanism_assessment_input_is_numbered_verbatim_and_unique(
    tmp_path: Path,
) -> None:
    path = tmp_path / "mechanisms.json"
    payload = {
        "schema_version": 1,
        "run_number": 1,
        "assessments": [
            {
                "source": "noaa_gfs",
                "metric": "gh_500",
                "relevant": True,
                "ambiguous": False,
                "assessment": "gh_500 (synoptic ridging/subsidence -> stagnation)",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assessments = load_mechanism_assessments(path, run_number=1)

    assert assessments["noaa_gfs/gh_500"].assessment == payload["assessments"][0][
        "assessment"
    ]
    payload["assessments"].append(dict(payload["assessments"][0]))
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate mechanism assessment"):
        load_mechanism_assessments(path, run_number=1)


def test_real_cli_writes_one_exclusive_numbered_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database = tmp_path / "snapshot.db"
    database.write_bytes(b"synthetic")
    assessments_path = tmp_path / "mechanisms.json"
    assessments_path.write_text(
        json.dumps({"schema_version": 1, "run_number": 1, "assessments": []}),
        encoding="utf-8",
    )
    output = tmp_path / "pruning-screen-run-001.json"
    loaded = SnapshotPruningInput(
        snapshot_sha256="a" * 64,
        observations=(),
        rendered_metrics=(),
        input_row_count=0,
        finite_in_radius_row_count=0,
        quality_excluded_row_count=0,
    )
    manifest = {
        "schema_version": 2,
        "run_number": 1,
        "real_screen_executed": True,
        "mechanism_review_complete": True,
        "variables": [],
    }
    monkeypatch.setattr("app.eval.pruning_screen.load_snapshot_inputs", lambda *args, **kwargs: loaded)
    monkeypatch.setattr("app.eval.pruning_screen.build_snapshot_manifest", lambda *args, **kwargs: manifest)

    assert main(
        [
            "--database",
            str(database),
            "--expected-sha256",
            "a" * 64,
            "--run-number",
            "1",
            "--mechanism-assessments",
            str(assessments_path),
            "--output",
            str(output),
        ]
    ) == 0
    assert json.loads(output.read_text(encoding="utf-8")) == manifest
    with pytest.raises(FileExistsError, match="cannot be overwritten"):
        main(
            [
                "--database",
                str(database),
                "--expected-sha256",
                "a" * 64,
                "--run-number",
                "1",
                "--mechanism-assessments",
                str(assessments_path),
                "--output",
                str(output),
            ]
        )


def test_active_screen_and_verbatim_mechanism_input_are_freeze_linked() -> None:
    fixture = load_pruning_fixture()
    payload = pruning_manifest_payload()

    assert fixture["snapshot_sha256"] == LOCKED_SNAPSHOT_SHA256
    assert payload["artifact"] == "pruning_screen.run-001.json"
    assert len(payload["screen"]["cells"]) == 360
    assessments = payload["mechanism_assessment_input"]["assessments"]
    assert [row["assessment"] for row in assessments] == [
        "gh_500 (synoptic ridging/subsidence -> stagnation)",
        "precipitable_water (moisture/washout)",
    ]
