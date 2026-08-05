import math
from datetime import datetime, timedelta, timezone

import pytest

from app.detection.zscore import ZScoreAnomaly, ZScoreDetector


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)


def _series(
    values: list[float], step: timedelta = timedelta(hours=1)
) -> list[tuple[datetime, float]]:
    """Build an evenly-spaced (timestamp, value) series anchored at T0."""
    return [(T0 + i * step, v) for i, v in enumerate(values)]


class TestZScoreDetectorBasics:
    def test_empty_input_returns_no_anomalies(self) -> None:
        detector = ZScoreDetector()
        assert detector.detect([]) == []

    def test_insufficient_history_returns_no_anomalies(self) -> None:
        # Only 5 points but min_points defaults to 10
        detector = ZScoreDetector()
        series = _series([10.0, 11.0, 9.0, 10.5, 9.5])
        assert detector.detect(series) == []

    def test_stable_series_returns_no_anomalies(self) -> None:
        # Tight noise band, no value exceeds +/-3 sigma of the rolling mean
        detector = ZScoreDetector()
        values = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.3, 9.7, 10.0,
                  10.1, 9.9, 10.2, 9.8, 10.0]
        assert detector.detect(_series(values)) == []

    def test_constant_series_does_not_raise(self) -> None:
        # std == 0 must not cause division by zero
        detector = ZScoreDetector()
        series = _series([5.0] * 20)
        assert detector.detect(series) == []

    def test_quantized_flat_baseline_does_not_manufacture_a_giant_z(self) -> None:
        # 5.0 is binary-exact so its std is exactly 0, but 0.1 is not: numpy
        # returns ~1.4e-17 for a constant window of 0.1, which an `std == 0`
        # test lets through and which turns a one-tick rise into z ~1e15. TCEQ
        # reports CO at 0.1 ppm, so this window shape is routine, and the
        # freeze ranks on |z| — a single escapee would take the top slot.
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        series = _series([0.1] * 15 + [0.2])
        assert detector.detect(series) == []

    def test_flat_baseline_is_rejected_at_any_scale(self) -> None:
        # The floor is relative, so it must behave the same for satellite
        # column densities as for ppm.
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        for level in (0.1, 3.7, 1.0e15, 2.5e-6):
            series = _series([level] * 15 + [level * 2])
            assert detector.detect(series) == [], f"leaked at level {level}"

    def test_a_genuinely_narrow_baseline_still_produces_anomalies(self) -> None:
        # The floor must not silently swallow real low-variance series. This
        # baseline varies by 1% and stays well above the 1e-9 relative floor.
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [0.1, 0.101, 0.099, 0.1, 0.101, 0.099, 0.1, 0.101, 0.099, 0.1]
        anomalies = detector.detect(_series(baseline + [0.2]))
        assert len(anomalies) == 1
        assert anomalies[0].value == 0.2
        assert math.isfinite(anomalies[0].z_score)


class TestZScoreDetectorOutliers:
    def test_single_positive_outlier_detected(self) -> None:
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        # 10 quiet points, then a spike that is far above the rolling mean
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        series = _series(baseline + [100.0])
        anomalies = detector.detect(series)
        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.timestamp == T0 + timedelta(hours=10)
        assert anomaly.value == 100.0
        assert anomaly.z_score > 3.0

    def test_single_negative_outlier_detected_with_negative_z(self) -> None:
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [50.0, 51.0, 49.0, 50.0, 51.0, 49.0, 50.0, 51.0, 49.0, 50.0]
        series = _series(baseline + [-50.0])
        anomalies = detector.detect(series)
        assert len(anomalies) == 1
        assert anomalies[0].z_score < -3.0

    def test_borderline_outlier_above_threshold_is_flagged(self) -> None:
        # Construct: baseline has mean=10, std=sqrt(0.6)~0.7746.
        # Test value at z=3.5 -> value = 10 + 3.5*0.7746 = 12.711
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        series = _series(baseline + [12.711])
        anomalies = detector.detect(series)
        assert len(anomalies) == 1
        assert anomalies[0].z_score == pytest.approx(3.5, abs=0.01)

    def test_value_under_threshold_is_not_flagged(self) -> None:
        # Same baseline, value at z=2.5 -> value = 10 + 2.5*0.7746 = 11.937
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        series = _series(baseline + [11.937])
        assert detector.detect(series) == []


class TestZScoreArithmetic:
    def test_exact_z_score_value_matches_expected_arithmetic(self) -> None:
        # Baseline mean=10, population std=sqrt(0.6)~=0.7745966692
        # Spike at 20 -> expected z = (20-10)/sqrt(0.6) ~= 12.9099
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        series = _series(baseline + [20.0])
        anomalies = detector.detect(series)
        assert len(anomalies) == 1
        expected_z = (20.0 - 10.0) / math.sqrt(0.6)
        assert anomalies[0].expected_value == pytest.approx(10.0)
        assert anomalies[0].z_score == pytest.approx(expected_z, rel=1e-9)
        assert anomalies[0].window_n == 10


class TestZScoreDetectorConfiguration:
    def test_threshold_can_be_loosened_to_catch_smaller_outliers(self) -> None:
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        # z = 2.5 -> value = 11.937
        series = _series(baseline + [11.937])
        assert ZScoreDetector(threshold=3.0).detect(series) == []
        loose = ZScoreDetector(threshold=2.0).detect(series)
        assert len(loose) == 1

    def test_window_excludes_points_older_than_window_duration(self) -> None:
        detector = ZScoreDetector(
            window=timedelta(hours=5), min_points=4, threshold=3.0
        )
        # Old quiet block (way outside the 5h window), then a high-baseline block,
        # then a spike. With the 5h window, the spike is judged against the high
        # baseline only - so it should NOT be flagged at the same z-score as if
        # the older quiet points were included.
        old_block = [(T0 + timedelta(hours=i), 5.0) for i in range(10)]
        # 6 high values at hours 100..105
        high_baseline = [
            (T0 + timedelta(hours=100 + i), 50.0 + (i % 2)) for i in range(6)
        ]
        # Spike at hour 106 is only +1 above the rolling-window mean ~= 50.5
        spike = [(T0 + timedelta(hours=106), 51.0)]
        anomalies = detector.detect(old_block + high_baseline + spike)
        assert anomalies == []

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError):
            ZScoreDetector(threshold=0)
        with pytest.raises(ValueError):
            ZScoreDetector(threshold=-1)

    def test_invalid_min_points_raises(self) -> None:
        with pytest.raises(ValueError):
            ZScoreDetector(min_points=1)


class TestZScoreWindowBounds:
    def test_duplicate_timestamp_siblings_excluded_from_each_others_window(
        self,
    ) -> None:
        # The window is half-open by *timestamp*: cutoff <= ts < ts_i. Two points
        # sharing one timestamp must each see only the strictly-earlier points,
        # never each other - so both spikes are scored against the 10-point
        # baseline alone (window_n == 10), not 10-plus-the-sibling.
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        dup_ts = T0 + timedelta(hours=10)
        series = _series(baseline) + [(dup_ts, 100.0), (dup_ts, 100.0)]

        anomalies = detector.detect(series)

        at_dup = [a for a in anomalies if a.timestamp == dup_ts]
        assert len(at_dup) == 2
        assert all(a.window_n == 10 for a in at_dup)

    def test_lower_window_bound_is_inclusive_of_the_cutoff_point(self) -> None:
        # cutoff = ts_i - window is inclusive: the point sitting exactly on the
        # cutoff belongs to the window. window=10h means the hour-0 point is the
        # cutoff for the hour-10 point, so its window holds all 10 earlier points.
        detector = ZScoreDetector(
            window=timedelta(hours=10), min_points=2, threshold=3.0
        )
        earlier = [9.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        series = _series(earlier) + [(T0 + timedelta(hours=10), 100.0)]

        anomalies = detector.detect(series)

        spike = next(
            (a for a in anomalies if a.timestamp == T0 + timedelta(hours=10)), None
        )
        assert spike is not None
        assert spike.window_n == 10


class TestZScoreDetectorLookbackOnly:
    def test_anomaly_classification_uses_only_past_points(self) -> None:
        # Build a series where a single early spike would not be detected
        # against the small window before it, but a later point sees the spike
        # in its lookback window. The early spike must NOT be flagged solely
        # because of future points; the later point may or may not be flagged
        # depending on stats. Here we assert specifically: the spike-time point
        # is not flagged because its lookback has insufficient points (min_points=10).
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        early_spike_series = [
            (T0, 5.0),
            (T0 + timedelta(hours=1), 100.0),  # spike at hour 1, only 1 prior point
        ] + [
            (T0 + timedelta(hours=2 + i), 5.0) for i in range(50)
        ]
        anomalies = detector.detect(early_spike_series)
        # The spike itself (hour 1) is NOT flagged: only 1 prior point.
        assert all(a.timestamp != T0 + timedelta(hours=1) for a in anomalies)

    def test_returns_zscore_anomaly_pydantic_models(self) -> None:
        detector = ZScoreDetector(min_points=10, threshold=3.0)
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        series = _series(baseline + [100.0])
        anomalies = detector.detect(series)
        assert all(isinstance(a, ZScoreAnomaly) for a in anomalies)
        # Ordered by timestamp ascending
        assert anomalies == sorted(anomalies, key=lambda a: a.timestamp)
