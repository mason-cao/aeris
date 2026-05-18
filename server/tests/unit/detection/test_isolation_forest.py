import math
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest

from app.detection.isolation_forest import (
    IsolationForestAnomaly,
    IsolationForestDetector,
    IsolationForestInput,
)


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


def _hourly_inputs(
    n: int,
    value_fn: Callable[[int], float] = lambda _i: 50.0,
    aux_fn: Callable[[int], dict[str, float]] | None = None,
) -> list[IsolationForestInput]:
    aux = aux_fn or (lambda _i: {})
    return [
        IsolationForestInput(
            timestamp=T0 + timedelta(hours=i),
            value=value_fn(i),
            aux_features=aux(i),
        )
        for i in range(n)
    ]


class TestIFDetectorBasics:
    def test_empty_input_returns_no_anomalies(self) -> None:
        detector = IsolationForestDetector()
        assert detector.detect([]) == []

    def test_insufficient_data_returns_no_anomalies(self) -> None:
        detector = IsolationForestDetector(min_points=50)
        inputs = _hourly_inputs(20)
        assert detector.detect(inputs) == []


class TestIFDetectorUnivariate:
    def test_extreme_value_outlier_is_flagged(self) -> None:
        n = 200

        def value_fn(i: int) -> float:
            return 50.0 + 5.0 * math.sin(2.0 * math.pi * i / 24)

        inputs = _hourly_inputs(n, value_fn=value_fn)
        spike_idx = 100
        spike_ts = inputs[spike_idx].timestamp
        inputs[spike_idx] = IsolationForestInput(
            timestamp=spike_ts,
            value=500.0,
        )

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        anomalies = detector.detect(inputs)

        flagged = {a.timestamp for a in anomalies}
        assert spike_ts in flagged
        # contamination=0.05 caps at ~5% of 200 = ~10; allow some slack
        assert 1 <= len(anomalies) <= int(n * 0.12) + 1


class TestIFDetectorMultivariate:
    def test_outlier_in_aux_feature_alone_is_flagged(self) -> None:
        n = 200

        def value_fn(i: int) -> float:
            return 50.0 + 5.0 * math.sin(2.0 * math.pi * i / 24)

        inputs = _hourly_inputs(
            n,
            value_fn=value_fn,
            aux_fn=lambda _i: {"wind_speed": 3.0, "pbl_height": 1000.0},
        )
        anomaly_idx = 120
        anomaly_ts = inputs[anomaly_idx].timestamp
        # Value typical for this hour, but met fields are wildly atypical:
        # very high wind speed AND a collapsed boundary layer.
        normal_value = inputs[anomaly_idx].value
        inputs[anomaly_idx] = IsolationForestInput(
            timestamp=anomaly_ts,
            value=normal_value,
            aux_features={"wind_speed": 50.0, "pbl_height": 50.0},
        )

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        anomalies = detector.detect(inputs)

        flagged = {a.timestamp for a in anomalies}
        assert anomaly_ts in flagged

    def test_features_used_lists_intrinsic_and_aux_columns(self) -> None:
        n = 100
        inputs = _hourly_inputs(
            n,
            value_fn=lambda i: 50.0 + (i % 7),
            aux_fn=lambda _i: {"wind_speed": 3.0, "pbl_height": 1000.0},
        )
        inputs[50] = IsolationForestInput(
            timestamp=inputs[50].timestamp,
            value=999.0,
            aux_features={"wind_speed": 99.0, "pbl_height": 10.0},
        )

        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        anomalies = detector.detect(inputs)

        assert anomalies
        for a in anomalies:
            assert set(a.features_used) == {
                "value",
                "hour_of_day",
                "day_of_week",
                "wind_speed",
                "pbl_height",
            }

    def test_inconsistent_aux_feature_keys_raises(self) -> None:
        n = 60
        inputs = _hourly_inputs(n, aux_fn=lambda _i: {"wind_speed": 1.0})
        mismatched_ts = inputs[10].timestamp
        inputs[10] = IsolationForestInput(
            timestamp=mismatched_ts,
            value=inputs[10].value,
            aux_features={"pbl_height": 500.0},
        )
        detector = IsolationForestDetector(random_state=42)
        with pytest.raises(ValueError, match="aux_features"):
            detector.detect(inputs)

    def test_non_finite_aux_feature_raises(self) -> None:
        n = 60
        inputs = _hourly_inputs(n, aux_fn=lambda _i: {"wind_speed": 1.0})
        inputs[5] = IsolationForestInput(
            timestamp=inputs[5].timestamp,
            value=inputs[5].value,
            aux_features={"wind_speed": float("nan")},
        )
        detector = IsolationForestDetector(random_state=42)
        with pytest.raises(ValueError, match="finite"):
            detector.detect(inputs)


class TestIFDetectorConfiguration:
    def test_random_state_makes_results_reproducible(self) -> None:
        n = 100
        inputs = _hourly_inputs(n, value_fn=lambda i: 50.0 + (i % 5))
        inputs[80] = IsolationForestInput(
            timestamp=inputs[80].timestamp,
            value=500.0,
        )

        first = IsolationForestDetector(random_state=42).detect(inputs)
        second = IsolationForestDetector(random_state=42).detect(inputs)
        assert [a.timestamp for a in first] == [a.timestamp for a in second]
        for a, b in zip(first, second, strict=True):
            assert a.decision_score == b.decision_score

    def test_higher_contamination_flags_more(self) -> None:
        n = 100
        inputs = _hourly_inputs(n, value_fn=lambda i: 50.0 + (i % 5))
        for idx in (20, 40, 60, 80, 95):
            inputs[idx] = IsolationForestInput(
                timestamp=inputs[idx].timestamp,
                value=200.0 + idx,
            )
        few = IsolationForestDetector(
            contamination=0.05, random_state=42
        ).detect(inputs)
        many = IsolationForestDetector(
            contamination=0.20, random_state=42
        ).detect(inputs)
        assert len(many) > len(few)

    def test_invalid_contamination_raises(self) -> None:
        with pytest.raises(ValueError):
            IsolationForestDetector(contamination=0)
        with pytest.raises(ValueError):
            IsolationForestDetector(contamination=0.6)
        with pytest.raises(ValueError):
            IsolationForestDetector(contamination=-0.1)

    def test_invalid_n_estimators_raises(self) -> None:
        with pytest.raises(ValueError):
            IsolationForestDetector(n_estimators=0)
        with pytest.raises(ValueError):
            IsolationForestDetector(n_estimators=-5)

    def test_invalid_min_points_raises(self) -> None:
        with pytest.raises(ValueError):
            IsolationForestDetector(min_points=1)


class TestIFDetectorContract:
    def test_returns_isolation_forest_anomaly_models(self) -> None:
        n = 100
        inputs = _hourly_inputs(n, value_fn=lambda i: 50.0 + (i % 5))
        inputs[50] = IsolationForestInput(
            timestamp=inputs[50].timestamp,
            value=500.0,
        )
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        anomalies = detector.detect(inputs)

        assert anomalies
        assert all(isinstance(a, IsolationForestAnomaly) for a in anomalies)
        assert anomalies == sorted(anomalies, key=lambda a: a.timestamp)

    def test_decision_score_is_negative_for_flagged_points(self) -> None:
        # sklearn's decision_function convention: negative = anomaly.
        n = 100
        inputs = _hourly_inputs(n, value_fn=lambda i: 50.0 + (i % 5))
        inputs[50] = IsolationForestInput(
            timestamp=inputs[50].timestamp,
            value=500.0,
        )
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        anomalies = detector.detect(inputs)
        assert all(a.decision_score < 0 for a in anomalies)

    def test_unsorted_input_produces_sorted_output(self) -> None:
        n = 100
        inputs = _hourly_inputs(n, value_fn=lambda i: 50.0 + (i % 5))
        inputs[50] = IsolationForestInput(
            timestamp=inputs[50].timestamp,
            value=500.0,
        )
        shuffled = list(reversed(inputs))
        detector = IsolationForestDetector(contamination=0.05, random_state=42)
        anomalies = detector.detect(shuffled)
        assert anomalies == sorted(anomalies, key=lambda a: a.timestamp)
