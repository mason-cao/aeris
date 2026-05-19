from datetime import datetime, timedelta, timezone

import pytest

from app.detection.consensus import (
    ALL_METHODS,
    ConsensusAnomaly,
    Severity,
    merge,
    severity_for,
)
from app.detection.isolation_forest import IsolationForestAnomaly
from app.detection.stl import STLAnomaly
from app.detection.zscore import ZScoreAnomaly


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698


def _z(ts: datetime, value: float = 100.0, expected: float = 10.0, z: float = 5.0) -> ZScoreAnomaly:
    return ZScoreAnomaly(
        timestamp=ts,
        value=value,
        expected_value=expected,
        z_score=z,
        window_n=24,
    )


def _stl(ts: datetime, value: float = 100.0, expected: float = 12.0) -> STLAnomaly:
    return STLAnomaly(
        timestamp=ts,
        value=value,
        expected_value=expected,
        residual=value - expected,
        residual_score=(value - expected) / 2.0,
        period=24,
    )


def _if(ts: datetime, value: float = 100.0) -> IsolationForestAnomaly:
    return IsolationForestAnomaly(
        timestamp=ts,
        value=value,
        decision_score=-0.3,
        anomaly_score=-0.7,
        features_used=["value", "hour_of_day", "day_of_week"],
    )


def _merge_kwargs(**overrides):
    base = dict(
        metric="pm25",
        source="openaq",
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
    )
    base.update(overrides)
    return base


class TestSeverityFor:
    def test_one_method_is_minor(self) -> None:
        assert severity_for(1) == "minor"

    def test_two_methods_is_moderate(self) -> None:
        assert severity_for(2) == "moderate"

    def test_three_methods_is_severe(self) -> None:
        assert severity_for(3) == "severe"

    def test_zero_methods_raises(self) -> None:
        with pytest.raises(ValueError):
            severity_for(0)

    def test_more_than_three_methods_raises(self) -> None:
        with pytest.raises(ValueError):
            severity_for(4)

    def test_negative_methods_raises(self) -> None:
        with pytest.raises(ValueError):
            severity_for(-1)


class TestMergeEmptyInputs:
    def test_no_detector_results_returns_empty(self) -> None:
        assert merge(**_merge_kwargs()) == []

    def test_all_empty_sequences_returns_empty(self) -> None:
        result = merge(
            **_merge_kwargs(),
            zscore_results=[],
            stl_results=[],
            isolation_forest_results=[],
        )
        assert result == []


class TestMergeSingleMethod:
    def test_only_zscore_produces_minor_severity(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, value=87.4, expected=10.0, z=5.0)],
        )
        assert len(result) == 1
        anomaly = result[0]
        assert isinstance(anomaly, ConsensusAnomaly)
        assert anomaly.timestamp == ts
        assert anomaly.value == pytest.approx(87.4)
        assert anomaly.expected_value == pytest.approx(10.0)
        assert anomaly.z_score == pytest.approx(5.0)
        assert anomaly.methods_triggered == ["zscore"]
        assert anomaly.severity == "minor"
        assert anomaly.metric == "pm25"
        assert anomaly.source == "openaq"
        assert anomaly.lat == pytest.approx(HOUSTON_LAT)
        assert anomaly.lon == pytest.approx(HOUSTON_LON)

    def test_only_stl_produces_minor_severity(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            stl_results=[_stl(ts, value=87.4, expected=12.0)],
        )
        assert len(result) == 1
        assert result[0].methods_triggered == ["stl"]
        assert result[0].severity == "minor"
        # STL's expected propagates when z-score is absent
        assert result[0].expected_value == pytest.approx(12.0)
        # No z-score data → field is None
        assert result[0].z_score is None

    def test_only_isolation_forest_produces_minor_severity(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            isolation_forest_results=[_if(ts, value=87.4)],
        )
        assert len(result) == 1
        assert result[0].methods_triggered == ["isolation_forest"]
        assert result[0].severity == "minor"
        # IF doesn't provide expected_value or z_score
        assert result[0].expected_value is None
        assert result[0].z_score is None
        assert result[0].value == pytest.approx(87.4)


class TestMergeTwoMethods:
    def test_zscore_plus_stl_at_same_ts_is_moderate(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, value=87.4, expected=10.0, z=5.0)],
            stl_results=[_stl(ts, value=87.4, expected=12.0)],
        )
        assert len(result) == 1
        anomaly = result[0]
        assert sorted(anomaly.methods_triggered) == ["stl", "zscore"]
        assert anomaly.severity == "moderate"
        # Z-score's expected_value takes priority over STL's
        assert anomaly.expected_value == pytest.approx(10.0)
        assert anomaly.z_score == pytest.approx(5.0)

    def test_zscore_plus_isolation_forest_is_moderate(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, z=5.0)],
            isolation_forest_results=[_if(ts)],
        )
        assert len(result) == 1
        assert sorted(result[0].methods_triggered) == ["isolation_forest", "zscore"]
        assert result[0].severity == "moderate"
        assert result[0].z_score == pytest.approx(5.0)

    def test_stl_plus_isolation_forest_is_moderate(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            stl_results=[_stl(ts, expected=12.0)],
            isolation_forest_results=[_if(ts)],
        )
        assert len(result) == 1
        assert sorted(result[0].methods_triggered) == ["isolation_forest", "stl"]
        assert result[0].severity == "moderate"
        assert result[0].expected_value == pytest.approx(12.0)
        assert result[0].z_score is None


class TestMergeThreeMethods:
    def test_all_three_at_same_ts_is_severe(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, value=87.4, expected=10.0, z=5.0)],
            stl_results=[_stl(ts, value=87.4, expected=12.0)],
            isolation_forest_results=[_if(ts, value=87.4)],
        )
        assert len(result) == 1
        anomaly = result[0]
        assert sorted(anomaly.methods_triggered) == sorted(ALL_METHODS)
        assert anomaly.severity == "severe"
        assert anomaly.value == pytest.approx(87.4)
        assert anomaly.expected_value == pytest.approx(10.0)
        assert anomaly.z_score == pytest.approx(5.0)


class TestMergeDistinctTimestamps:
    def test_disjoint_timestamps_each_produce_separate_anomalies(self) -> None:
        t1 = T0 + timedelta(hours=10)
        t2 = T0 + timedelta(hours=20)
        t3 = T0 + timedelta(hours=30)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(t1, z=5.0)],
            stl_results=[_stl(t2)],
            isolation_forest_results=[_if(t3)],
        )
        assert len(result) == 3
        by_ts = {a.timestamp: a for a in result}
        assert by_ts[t1].methods_triggered == ["zscore"]
        assert by_ts[t1].severity == "minor"
        assert by_ts[t2].methods_triggered == ["stl"]
        assert by_ts[t2].severity == "minor"
        assert by_ts[t3].methods_triggered == ["isolation_forest"]
        assert by_ts[t3].severity == "minor"

    def test_partial_overlap_assigns_per_timestamp_severity(self) -> None:
        t_both = T0 + timedelta(hours=10)
        t_z_only = T0 + timedelta(hours=20)
        t_if_only = T0 + timedelta(hours=30)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(t_both, z=5.0), _z(t_z_only, z=4.0)],
            stl_results=[_stl(t_both)],
            isolation_forest_results=[_if(t_if_only)],
        )
        assert len(result) == 3
        by_ts = {a.timestamp: a for a in result}
        assert by_ts[t_both].severity == "moderate"
        assert sorted(by_ts[t_both].methods_triggered) == ["stl", "zscore"]
        assert by_ts[t_z_only].severity == "minor"
        assert by_ts[t_z_only].methods_triggered == ["zscore"]
        assert by_ts[t_if_only].severity == "minor"
        assert by_ts[t_if_only].methods_triggered == ["isolation_forest"]


class TestMergeOrdering:
    def test_output_is_sorted_by_timestamp_ascending(self) -> None:
        t_late = T0 + timedelta(hours=30)
        t_mid = T0 + timedelta(hours=20)
        t_early = T0 + timedelta(hours=10)
        # Feed in reverse chronological order; merge should sort.
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(t_late, z=5.0), _z(t_mid, z=4.0), _z(t_early, z=3.5)],
        )
        assert [a.timestamp for a in result] == [t_early, t_mid, t_late]

    def test_methods_triggered_is_deterministic_order(self) -> None:
        # Methods are reported in canonical order regardless of input order,
        # so downstream consumers (DB writers, UI) get stable strings.
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            isolation_forest_results=[_if(ts)],
            stl_results=[_stl(ts)],
            zscore_results=[_z(ts, z=5.0)],
        )
        assert len(result) == 1
        # Canonical order matches the iteration order of ALL_METHODS
        assert result[0].methods_triggered == list(ALL_METHODS)


class TestMergeFieldPriority:
    def test_zscore_expected_wins_over_stl_when_both_present(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, expected=10.0)],
            stl_results=[_stl(ts, expected=12.0)],
        )
        assert result[0].expected_value == pytest.approx(10.0)

    def test_stl_expected_used_when_only_stl_present(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            stl_results=[_stl(ts, expected=12.0)],
            isolation_forest_results=[_if(ts)],
        )
        # Both fire (moderate), but only STL provides expected_value
        assert result[0].severity == "moderate"
        assert result[0].expected_value == pytest.approx(12.0)

    def test_value_comes_from_any_flagging_detector(self) -> None:
        # All detectors observe the same underlying value at a timestamp.
        ts = T0 + timedelta(hours=10)
        for results in [
            dict(zscore_results=[_z(ts, value=87.4, z=5.0)]),
            dict(stl_results=[_stl(ts, value=87.4)]),
            dict(isolation_forest_results=[_if(ts, value=87.4)]),
        ]:
            r = merge(**_merge_kwargs(), **results)
            assert r[0].value == pytest.approx(87.4)


class TestMergePreservesDetails:
    def test_per_detector_raw_outputs_preserved_for_research(self) -> None:
        ts = T0 + timedelta(hours=10)
        z_a = _z(ts, value=87.4, expected=10.0, z=5.0)
        stl_a = _stl(ts, value=87.4, expected=12.0)
        if_a = _if(ts, value=87.4)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[z_a],
            stl_results=[stl_a],
            isolation_forest_results=[if_a],
        )
        assert result[0].zscore_detail == z_a
        assert result[0].stl_detail == stl_a
        assert result[0].isolation_forest_detail == if_a

    def test_missing_detector_details_are_none(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, z=5.0)],
        )
        assert result[0].zscore_detail is not None
        assert result[0].stl_detail is None
        assert result[0].isolation_forest_detail is None


class TestMergeValueConsistency:
    def test_inconsistent_values_across_detectors_at_same_ts_raises(self) -> None:
        # Same timestamp must mean the same observed value. If the caller
        # passes mismatched values, that's a data-integrity bug we surface
        # rather than silently picking one.
        ts = T0 + timedelta(hours=10)
        with pytest.raises(ValueError, match="inconsistent"):
            merge(
                **_merge_kwargs(),
                zscore_results=[_z(ts, value=87.4, z=5.0)],
                stl_results=[_stl(ts, value=42.0)],  # different value!
            )


class TestMergeTypes:
    def test_severity_is_one_of_the_three_literals(self) -> None:
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, z=5.0)],
        )
        assert result[0].severity in ("minor", "moderate", "severe")

    def test_all_methods_constant_is_complete(self) -> None:
        assert set(ALL_METHODS) == {"zscore", "stl", "isolation_forest"}

    def test_severity_type_alias_is_str_literal(self) -> None:
        # Sanity: Severity values must be plain strings (for JSON / DB persistence)
        ts = T0 + timedelta(hours=10)
        result = merge(
            **_merge_kwargs(),
            zscore_results=[_z(ts, z=5.0)],
        )
        sev: Severity = result[0].severity
        assert isinstance(sev, str)
