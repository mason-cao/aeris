import math
from datetime import datetime, timedelta, timezone

import pytest

from app.detection.stl import STLAnomaly, STLDetector


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
PERIOD = 24


def _seasonal_series(
    n_periods: int = 3,
    period: int = PERIOD,
    base: float = 50.0,
    amplitude: float = 10.0,
) -> list[tuple[datetime, float]]:
    """Pure sinusoidal series, evenly-spaced hourly, no noise."""
    n = n_periods * period
    return [
        (
            T0 + timedelta(hours=i),
            base + amplitude * math.sin(2.0 * math.pi * i / period),
        )
        for i in range(n)
    ]


class TestSTLDetectorBasics:
    def test_empty_input_returns_no_anomalies(self) -> None:
        detector = STLDetector()
        assert detector.detect([]) == []

    def test_insufficient_history_returns_no_anomalies(self) -> None:
        # period=24 -> min_points defaults to 2*24+1 = 49
        detector = STLDetector(period=24)
        short_series = _seasonal_series(n_periods=1)[:30]
        assert detector.detect(short_series) == []

    def test_pure_seasonal_series_returns_no_anomalies(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = _seasonal_series(n_periods=3)
        assert detector.detect(series) == []

    def test_constant_series_does_not_raise_and_returns_no_anomalies(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = [(T0 + timedelta(hours=i), 5.0) for i in range(72)]
        assert detector.detect(series) == []


class TestSTLDetectorOutliers:
    def test_single_positive_spike_detected(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = list(_seasonal_series(n_periods=3))
        spike_idx = 36
        spike_ts, original_value = series[spike_idx]
        series[spike_idx] = (spike_ts, original_value + 100.0)

        anomalies = detector.detect(series)

        spike = next((a for a in anomalies if a.timestamp == spike_ts), None)
        assert spike is not None
        assert spike.value == original_value + 100.0
        assert spike.residual_score > 2.5
        assert spike.residual > 0

    def test_single_negative_spike_detected_with_negative_score(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = list(_seasonal_series(n_periods=3))
        spike_idx = 36
        spike_ts, original_value = series[spike_idx]
        series[spike_idx] = (spike_ts, original_value - 100.0)

        anomalies = detector.detect(series)

        spike = next((a for a in anomalies if a.timestamp == spike_ts), None)
        assert spike is not None
        assert spike.residual_score < -2.5
        assert spike.residual < 0

    def test_expected_value_approximates_clean_seasonal(self) -> None:
        # For a small spike on a clean seasonal baseline, the detector's
        # expected_value (trend+seasonal) should be close to the underlying
        # seasonal value at that timestamp.
        detector = STLDetector(period=24, threshold=2.5)
        clean = _seasonal_series(n_periods=4)
        series = list(clean)
        spike_idx = 50
        spike_ts, original_value = series[spike_idx]
        series[spike_idx] = (spike_ts, original_value + 50.0)

        anomalies = detector.detect(series)
        spike = next((a for a in anomalies if a.timestamp == spike_ts), None)
        assert spike is not None
        # The model's expected value should land near the clean baseline,
        # not near the spiked value (within a few units of leakage).
        assert abs(spike.expected_value - original_value) < 5.0

    def test_multiple_spikes_all_detected(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = list(_seasonal_series(n_periods=5))
        spikes = {30: +80.0, 60: -80.0, 90: +120.0}
        for idx, delta in spikes.items():
            ts, original = series[idx]
            series[idx] = (ts, original + delta)

        anomalies = detector.detect(series)
        flagged_timestamps = {a.timestamp for a in anomalies}
        for idx in spikes:
            assert series[idx][0] in flagged_timestamps


class TestSTLDetectorConfiguration:
    def test_higher_threshold_suppresses_marginal_anomalies(self) -> None:
        # The same series scored under threshold=2.5 vs threshold=20.0:
        # a moderate spike should clear 2.5 but not 20.0.
        series = list(_seasonal_series(n_periods=3))
        ts, original = series[36]
        series[36] = (ts, original + 20.0)

        default = STLDetector(period=24, threshold=2.5).detect(series)
        strict = STLDetector(period=24, threshold=20.0).detect(series)

        assert len(default) >= 1
        # Every strictly-flagged anomaly must clear the strict threshold
        for a in strict:
            assert abs(a.residual_score) > 20.0
        assert len(strict) <= len(default)

    def test_period_can_be_customized(self) -> None:
        period = 12
        n = 4 * period
        clean = [
            (
                T0 + timedelta(hours=i),
                50.0 + 10.0 * math.sin(2.0 * math.pi * i / period),
            )
            for i in range(n)
        ]
        series = list(clean)
        spike_idx = 20
        ts, original = series[spike_idx]
        series[spike_idx] = (ts, original + 100.0)

        detector = STLDetector(period=period, threshold=2.5)
        anomalies = detector.detect(series)
        flagged_timestamps = {a.timestamp for a in anomalies}
        assert ts in flagged_timestamps
        assert all(a.period == period for a in anomalies)

    def test_invalid_period_raises(self) -> None:
        with pytest.raises(ValueError):
            STLDetector(period=1)
        with pytest.raises(ValueError):
            STLDetector(period=0)
        with pytest.raises(ValueError):
            STLDetector(period=-1)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            STLDetector(threshold=0)
        with pytest.raises(ValueError):
            STLDetector(threshold=-1.0)

    def test_invalid_min_points_raises(self) -> None:
        with pytest.raises(ValueError):
            STLDetector(period=24, min_points=10)  # below 2*period+1 floor


class TestSTLDetectorContract:
    def test_returns_stl_anomaly_pydantic_models(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = list(_seasonal_series(n_periods=3))
        series[36] = (series[36][0], series[36][1] + 100.0)

        anomalies = detector.detect(series)

        assert anomalies, "expected at least one anomaly"
        assert all(isinstance(a, STLAnomaly) for a in anomalies)
        assert anomalies == sorted(anomalies, key=lambda a: a.timestamp)

    def test_unsorted_input_is_sorted_internally(self) -> None:
        detector = STLDetector(period=24, threshold=2.5)
        series = list(_seasonal_series(n_periods=3))
        spike_ts = series[36][0]
        series[36] = (spike_ts, series[36][1] + 100.0)

        shuffled = list(reversed(series))
        anomalies = detector.detect(shuffled)

        assert any(a.timestamp == spike_ts for a in anomalies)
        assert anomalies == sorted(anomalies, key=lambda a: a.timestamp)
