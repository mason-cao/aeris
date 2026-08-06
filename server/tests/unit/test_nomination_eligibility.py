"""B9 regulatory-monitor, entity, and elevation nomination lock."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from app.db.models import Anomaly, DataPoint
from app.detection.consensus import ConsensusAnomaly
from app.detection.isolation_forest import IsolationForestAnomaly
from app.detection.run import (
    filter_elevation_nominees,
    group_points_by_series,
    run_detection,
)
from app.detection.stl import STLAnomaly
from app.detection.zscore import ZScoreAnomaly
from app.llm.corroboration import CONTRADICTING, SILENT, score_concentration_elevation
from app.provenance.nomination import (
    NOMINATING_METRICS_BY_SOURCE,
    nomination_manifest_payload,
    series_is_nomination_eligible,
    validate_nominating_metric_disjointness,
)
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    load_openaq_pm25_fixture,
    verified_monitor_entity_ids,
)


T0 = datetime(2026, 6, 15, 12, 0, tzinfo=timezone.utc)


def _point(
    *,
    source: str,
    metric: str,
    entity_id: str,
    value: float = 10.0,
) -> DataPoint:
    return DataPoint(
        timestamp=T0,
        lat=29.76,
        lon=-95.37,
        metric=metric,
        value=value,
        unit="ug/m3",
        source=source,
        source_entity_id=entity_id,
        raw_json=None,
    )


def _consensus(
    *,
    value: float,
    expected: float | None,
    zscore: bool = False,
    stl: bool = False,
    isolation_forest: bool = False,
) -> ConsensusAnomaly:
    z_detail = (
        ZScoreAnomaly(
            timestamp=T0,
            value=value,
            expected_value=float(expected),
            z_score=4.0,
            window_n=20,
        )
        if zscore and expected is not None
        else None
    )
    stl_detail = (
        STLAnomaly(
            timestamp=T0,
            value=value,
            expected_value=float(expected),
            residual=value - float(expected),
            residual_score=4.0,
            period=24,
        )
        if stl and expected is not None
        else None
    )
    if_detail = (
        IsolationForestAnomaly(
            timestamp=T0,
            value=value,
            decision_score=-0.2,
            anomaly_score=-0.7,
            features_used=["value"],
        )
        if isolation_forest
        else None
    )
    methods = [
        name
        for name, included in (
            ("zscore", zscore),
            ("stl", stl),
            ("isolation_forest", isolation_forest),
        )
        if included
    ]
    return ConsensusAnomaly(
        timestamp=T0,
        lat=29.76,
        lon=-95.37,
        metric="pm25",
        source="openaq",
        value=value,
        expected_value=expected,
        z_score=z_detail.z_score if z_detail else None,
        methods_triggered=methods or ["zscore"],
        severity="moderate",
        zscore_detail=z_detail,
        stl_detail=stl_detail,
        isolation_forest_detail=if_detail,
    )


def _entity_id(metric: str, provider: str) -> str:
    fixture = load_openaq_pm25_fixture()
    return min(
        str(row["entity_id"])
        for row in fixture["metric_entities"]
        if row["metric"] == metric and row["provider"] == provider
    )


@pytest.mark.parametrize("metric", ["pm25", "pm10", "ozone"])
def test_openaq_exact_entity_allowlist_by_metric(metric: str) -> None:
    eligible_id = min(verified_monitor_entity_ids(metric))

    assert series_is_nomination_eligible("openaq", metric, eligible_id) is True
    assert series_is_nomination_eligible("openaq", metric, "unknown") is False


@pytest.mark.parametrize("provider", ["Clarity", "AirGradient"])
def test_non_monitor_openaq_entity_never_eligible(provider: str) -> None:
    entity_id = _entity_id("pm25", provider)

    assert series_is_nomination_eligible("openaq", "pm25", entity_id) is False


@pytest.mark.parametrize("metric", ["no2", "so2", "co"])
def test_tceq_declared_metrics_are_eligible(metric: str) -> None:
    assert series_is_nomination_eligible("tceq", metric, "monitor-1") is True


@pytest.mark.parametrize(
    ("source", "metric"),
    [
        ("purpleair", "pm25"),
        ("sentinel5p", "s5p_no2_column"),
        ("noaa_gfs", "pbl_height"),
        ("openweather", "wind_speed"),
        ("asos", "wind_speed"),
        ("epa_aqs", "pm25"),
        ("unknown", "pm25"),
        ("tceq", "pm25"),
        ("openaq", "no2"),
    ],
)
def test_every_excluded_source_or_metric_is_ineligible(
    source: str, metric: str
) -> None:
    assert series_is_nomination_eligible(source, metric, "anything") is False


def test_nominating_metric_sets_are_disjoint() -> None:
    validate_nominating_metric_disjointness()

    assert NOMINATING_METRICS_BY_SOURCE == {
        "openaq": frozenset({"pm25", "pm10", "ozone"}),
        "tceq": frozenset({"no2", "so2", "co"}),
    }


def test_grouping_enforces_entity_policy_without_cli_help() -> None:
    monitor_id = min(verified_monitor_entity_ids("pm25"))
    clarity_id = _entity_id("pm25", "Clarity")
    points = [
        _point(source="openaq", metric="pm25", entity_id=monitor_id),
        _point(source="openaq", metric="pm25", entity_id=clarity_id),
        _point(source="openaq", metric="pm25", entity_id="unmappable"),
        _point(source="purpleair", metric="pm25", entity_id="pa-1"),
        _point(source="tceq", metric="no2", entity_id="tceq-1"),
        _point(source="tceq", metric="pm25", entity_id="tceq-1"),
    ]

    groups = group_points_by_series(points)

    assert {(key.source, key.metric, key.source_entity_id) for key in groups} == {
        ("openaq", "pm25", monitor_id),
        ("tceq", "no2", "tceq-1"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("source", "entity_id"),
    [
        ("openaq", _entity_id("pm25", "Clarity")),
        ("openaq", _entity_id("pm25", "AirGradient")),
        ("openaq", "unmappable"),
        ("purpleair", "pa-1"),
    ],
)
async def test_excluded_entity_can_never_persist_an_anomaly(
    db_session: object,
    source: str,
    entity_id: str,
) -> None:
    for index in range(60):
        point = _point(
            source=source,
            metric="pm25",
            entity_id=entity_id,
            value=500.0 if index == 40 else 10.0,
        )
        point.timestamp = T0 + timedelta(hours=index)
        db_session.add(point)
        await db_session.commit()

    summary = await run_detection(db_session, source=source)
    persisted = (await db_session.execute(select(Anomaly))).scalars().all()

    assert summary.n_groups_examined == 0
    assert summary.n_anomalies_emitted == 0
    assert persisted == []


@pytest.mark.asyncio
async def test_run_persists_only_strict_elevations_and_reports_exclusions(
    db_session: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for index in range(50):
        point = _point(
            source="tceq",
            metric="no2",
            entity_id="tceq-1",
            value=float(index),
        )
        point.timestamp = T0 + timedelta(hours=index)
        db_session.add(point)
        await db_session.commit()

    raw = [
        _consensus(value=10.0, expected=9.0, zscore=True),
        _consensus(value=10.0, expected=10.0, zscore=True),
        _consensus(value=9.0, expected=10.0, stl=True),
        _consensus(value=10.0, expected=None, isolation_forest=True),
    ]
    raw = [
        anomaly.model_copy(update={"source": "tceq", "metric": "no2"})
        for anomaly in raw
    ]

    class FakeEngine:
        def run(self, *_args: object, **_kwargs: object) -> list[ConsensusAnomaly]:
            return raw

    async def no_aux(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "app.detection.run._engine_for",
        lambda _series: (FakeEngine(), None),
    )
    monkeypatch.setattr("app.detection.run.build_aux_inputs", no_aux)

    summary = await run_detection(db_session, source="tceq")
    persisted = (await db_session.execute(select(Anomaly))).scalars().all()

    assert summary.n_raw_anomalies_emitted == 4
    assert summary.n_anomalies_emitted == 1
    assert summary.n_direction_excluded == 2
    assert summary.n_missing_expected_excluded == 1
    assert summary.n_persisted == 1
    assert len(persisted) == 1
    assert persisted[0].value == 10.0
    assert persisted[0].expected_value == 9.0


def test_strict_elevation_boundary_and_missing_expected_are_excluded() -> None:
    result = filter_elevation_nominees(
        [
            _consensus(value=10.0, expected=9.0, zscore=True),
            _consensus(value=10.0, expected=10.0, zscore=True),
            _consensus(value=9.0, expected=10.0, stl=True),
            _consensus(value=10.0, expected=None, isolation_forest=True),
        ]
    )

    assert [anomaly.value for anomaly in result.eligible] == [10.0]
    assert result.raw_count == 4
    assert result.eligible_count == 1
    assert result.direction_excluded_count == 2
    assert result.missing_expected_excluded_count == 1


def test_expected_value_source_accounting_prefers_zscore_then_stl() -> None:
    result = filter_elevation_nominees(
        [
            _consensus(value=10.0, expected=9.0, zscore=True, stl=True),
            _consensus(value=10.0, expected=9.0, stl=True),
            _consensus(value=10.0, expected=None, isolation_forest=True),
        ]
    )

    assert result.expected_value_source_counts == {
        "isolation_forest_only_undefined": 1,
        "stl_trend_plus_seasonal": 1,
        "zscore_rolling_mean": 1,
    }


def test_nonfinite_expected_is_treated_as_undefined() -> None:
    anomaly = _consensus(value=10.0, expected=None, isolation_forest=True)
    object.__setattr__(anomaly, "expected_value", math.nan)

    result = filter_elevation_nominees([anomaly])

    assert result.eligible == []
    assert result.missing_expected_excluded_count == 1


@pytest.mark.parametrize("metric", ["pm10", "ozone"])
def test_scoring_uses_same_v2_metric_allowlist(metric: str) -> None:
    monitor_id = min(verified_monitor_entity_ids(metric))
    timestamp = T0.isoformat()
    summary = {
        "anomaly": {
            "timestamp": timestamp,
            "source": "tceq",
            "metric": "no2",
        },
        "sources": {
            "openaq": {
                "metrics": {
                    metric: {
                        "unit": "ppm" if metric == "ozone" else "ug/m3",
                        "nearest_in_time": {
                            "t": timestamp,
                            "v": 100.0,
                            "entity_id": "unknown",
                            "dt_minutes": 0.0,
                        },
                        "entities": [
                            {
                                "entity_id": "unknown",
                                "distance_km": 0.1,
                                "series": [[timestamp, 100.0]],
                            },
                            {
                                "entity_id": monitor_id,
                                "distance_km": 0.2,
                                "series": [[timestamp, 1.0]],
                            },
                        ],
                    }
                }
            }
        },
    }
    claim = (
        "Ozone exceeded 50 ppm" if metric == "ozone" else "PM10 exceeded 50 ug/m3"
    )

    verdicts, _note = score_concentration_elevation(claim, summary)

    assert verdicts["openaq"] == CONTRADICTING


def test_unknown_only_openaq_metric_block_is_silent() -> None:
    timestamp = T0.isoformat()
    summary = {
        "anomaly": {"timestamp": timestamp, "source": "tceq", "metric": "no2"},
        "sources": {
            "openaq": {
                "metrics": {
                    "ozone": {
                        "unit": "ppm",
                        "nearest_in_time": {
                            "t": timestamp,
                            "v": 100.0,
                            "entity_id": "unknown",
                            "dt_minutes": 0.0,
                        },
                        "entities": [
                            {
                                "entity_id": "unknown",
                                "distance_km": 0.1,
                                "series": [[timestamp, 100.0]],
                            }
                        ],
                    }
                }
            }
        },
    }

    verdicts, note = score_concentration_elevation("Ozone exceeded 50 ppm", summary)

    assert verdicts["openaq"] == SILENT
    assert "no verified-monitor ozone observation" in note


def test_nonfixture_openaq_metric_keeps_generic_silence_reason() -> None:
    summary = {
        "anomaly": {
            "timestamp": T0.isoformat(),
            "source": "tceq",
            "metric": "no2",
        },
        "sources": {
            "openaq": {
                "metrics": {
                    "no2": {
                        "nearest_in_time": {"v": None},
                        "entities": [],
                    }
                }
            }
        },
    }

    verdicts, note = score_concentration_elevation(
        "NO2 concentrations were elevated",
        summary,
    )

    assert verdicts["openaq"] == SILENT
    assert "openaq: no no2 in window" in note
    assert "verified-monitor no2" not in note


def test_active_fixture_is_single_multimetric_v2_artifact() -> None:
    fixture = load_openaq_pm25_fixture()
    manifest = nomination_manifest_payload()

    assert fixture["schema_version"] == 2
    assert fixture["fixture_id"] == "openaq-regulatory-entity-provenance-v2"
    assert fixture["nominating_metrics"] == ["ozone", "pm10", "pm25"]
    assert fixture["eligible_entity_counts"] == {
        "ozone": 18,
        "pm10": 3,
        "pm25": 12,
    }
    assert fixture["snapshot_sha256"] == LOCKED_SNAPSHOT_SHA256
    assert manifest["artifact"] == "openaq_regulatory_entity_provenance.v2.json"
    assert manifest["fixture_id"] == fixture["fixture_id"]
    assert manifest["schema_version"] == 2
    assert manifest["snapshot_sha256"] == LOCKED_SNAPSHOT_SHA256
    assert manifest["covered_metrics"] == ["ozone", "pm10", "pm25"]
    assert manifest["artifact_sha256"] == (
        "78171a8e9312706ea10559ce8a36dbd482e97ecd4e174a96b5f41d0f32102cea"
    )
