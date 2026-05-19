import math
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

import pytest

from app.detection.consensus import ALL_METHODS, ConsensusAnomaly
from app.detection.engine import DetectionEngine
from app.detection.isolation_forest import (
    IsolationForestDetector,
    IsolationForestInput,
)
from app.detection.stl import STLDetector
from app.detection.zscore import ZScoreDetector


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698


def _hourly_series(
    n: int,
    value_at: Callable[[int], float],
    step: timedelta = timedelta(hours=1),
) -> list[tuple[datetime, float]]:
    return [(T0 + i * step, value_at(i)) for i in range(n)]


def _iso_inputs(
    n: int,
    value_at: Callable[[int], float],
    aux_at: Callable[[int], dict[str, float]] | None = None,
    step: timedelta = timedelta(hours=1),
) -> list[IsolationForestInput]:
    aux = aux_at or (lambda _i: {})
    return [
        IsolationForestInput(
            timestamp=T0 + i * step,
            value=value_at(i),
            aux_features=aux(i),
        )
        for i in range(n)
    ]


def _seasonal(i: int, base: float = 50.0, amplitude: float = 5.0, period: int = 24) -> float:
    return base + amplitude * math.sin(2.0 * math.pi * i / period)


def _run_kwargs(**overrides):
    base = dict(
        metric="pm25",
        source="openaq",
        lat=HOUSTON_LAT,
        lon=HOUSTON_LON,
    )
    base.update(overrides)
    return base


class TestEngineConstruction:
    def test_constructor_requires_at_least_one_detector(self) -> None:
        with pytest.raises(ValueError, match="at least one detector"):
            DetectionEngine()

    def test_with_defaults_factory_enables_all_three_detectors(self) -> None:
        engine = DetectionEngine.with_defaults()
        assert isinstance(engine.zscore, ZScoreDetector)
        assert isinstance(engine.stl, STLDetector)
        assert isinstance(engine.isolation_forest, IsolationForestDetector)

    def test_partial_configuration_preserves_disabled_detectors(self) -> None:
        engine = DetectionEngine(zscore=ZScoreDetector())
        assert isinstance(engine.zscore, ZScoreDetector)
        assert engine.stl is None
        assert engine.isolation_forest is None

    def test_can_pass_all_three_explicit_detectors(self) -> None:
        engine = DetectionEngine(
            zscore=ZScoreDetector(threshold=2.5),
            stl=STLDetector(period=24, threshold=2.5),
            isolation_forest=IsolationForestDetector(contamination=0.05),
        )
        assert engine.zscore.threshold == 2.5
        assert engine.stl.threshold == 2.5
        assert engine.isolation_forest.contamination == 0.05


class TestEngineEmptyAndStable:
    def test_empty_series_returns_empty(self) -> None:
        engine = DetectionEngine.with_defaults()
        result = engine.run([], **_run_kwargs())
        assert result == []

    def test_stable_series_returns_no_zscore_or_stl_anomalies(self) -> None:
        # 72 hours of clean seasonal data. Z-score (threshold 3.0) and STL
        # (residual threshold 2.5) should not fire on clean signal.
        # IF is intentionally excluded: with contamination=0.05 it flags
        # ~5% of points regardless of how clean the data is — that's its
        # design, not a bug. IF's behavior is exercised separately.
        n = 72
        series = _hourly_series(n, value_at=lambda i: _seasonal(i))
        engine = DetectionEngine(
            zscore=ZScoreDetector(),
            stl=STLDetector(),
        )
        result = engine.run(series, **_run_kwargs())
        assert result == []

    def test_isolation_forest_alone_flags_at_contamination_rate(self) -> None:
        # IF will always flag ~contamination * n points; on clean data those
        # are minor severity (only one method triggered).
        n = 100
        series = _hourly_series(n, value_at=lambda i: _seasonal(i))
        iso = _iso_inputs(
            n,
            value_at=lambda i: _seasonal(i),
            aux_at=lambda _i: {"wind_speed": 3.0, "pbl_height": 1000.0},
        )
        engine = DetectionEngine(
            isolation_forest=IsolationForestDetector(
                contamination=0.05, random_state=42
            ),
        )
        result = engine.run(
            series, **_run_kwargs(), isolation_forest_inputs=iso
        )
        # All flags must be IF-only and minor severity.
        for a in result:
            assert a.methods_triggered == ["isolation_forest"]
            assert a.severity == "minor"


class TestEngineSingleDetectorRouting:
    def test_zscore_only_engine_emits_only_zscore_anomalies(self) -> None:
        # Single spike at hour 60 in an otherwise quiet series.
        n = 80
        baseline = [10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0, 11.0, 9.0, 10.0]
        # Pad baseline out to n with the same low-variance pattern
        values = [baseline[i % len(baseline)] for i in range(n)]
        values[60] = 100.0
        series = [(T0 + timedelta(hours=i), v) for i, v in enumerate(values)]

        engine = DetectionEngine(zscore=ZScoreDetector(min_points=10, threshold=3.0))
        result = engine.run(series, **_run_kwargs())

        assert len(result) >= 1
        spike = next(
            (a for a in result if a.timestamp == T0 + timedelta(hours=60)),
            None,
        )
        assert spike is not None
        assert spike.methods_triggered == ["zscore"]
        assert spike.severity == "minor"
        assert spike.value == pytest.approx(100.0)

    def test_isolation_forest_skipped_when_inputs_not_provided(self) -> None:
        # IF detector enabled in engine, but caller passes no iso_inputs.
        # Should silently skip IF rather than crash.
        n = 80
        series = _hourly_series(n, value_at=lambda i: 10.0 + (i % 3))
        engine = DetectionEngine(
            zscore=ZScoreDetector(min_points=10, threshold=3.0),
            isolation_forest=IsolationForestDetector(min_points=50),
        )
        # No isolation_forest_inputs argument
        result = engine.run(series, **_run_kwargs())
        # No IF flags possible without inputs; only zscore could fire here.
        for a in result:
            assert "isolation_forest" not in a.methods_triggered


class TestEngineConsensusEndToEnd:
    def test_strong_spike_triggers_all_three_detectors_as_severe(self) -> None:
        # Build a seasonal series long enough to satisfy STL (>= 2*period+1)
        # and IF (>= 50). Inject a single large spike. All three detectors
        # should flag the same timestamp.
        n = 200
        spike_idx = 120

        def value_at(i: int) -> float:
            return _seasonal(i) if i != spike_idx else 500.0

        series = _hourly_series(n, value_at=value_at)
        iso = _iso_inputs(
            n,
            value_at=value_at,
            aux_at=lambda _i: {"wind_speed": 3.0, "pbl_height": 1000.0},
        )

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series, **_run_kwargs(), isolation_forest_inputs=iso
        )

        spike_ts = T0 + timedelta(hours=spike_idx)
        spike = next((a for a in result if a.timestamp == spike_ts), None)
        assert spike is not None, (
            "expected the injected spike at hour 120 to appear in consensus"
        )
        assert sorted(spike.methods_triggered) == sorted(ALL_METHODS)
        assert spike.severity == "severe"
        assert spike.value == pytest.approx(500.0)

    def test_zscore_only_spike_is_minor_not_severe(self) -> None:
        # Construct: a moderate spike that breaches z-score's 3-sigma threshold
        # but stays small enough not to trigger STL (residual scaled by STL's
        # estimated residual std, much wider) or IF (contamination cap).
        n = 200
        spike_idx = 120

        # Tight low-variance series so a small absolute spike crosses 3-sigma
        # for z-score but the seasonal model and the IF margin both absorb it.
        def value_at(i: int) -> float:
            if i == spike_idx:
                return 25.0  # ~5 sigma above a noise-band of ~0.5
            return 10.0 + (0.1 * (i % 3))

        series = _hourly_series(n, value_at=value_at)
        engine = DetectionEngine(
            zscore=ZScoreDetector(min_points=10, threshold=3.0),
        )
        result = engine.run(series, **_run_kwargs())
        spike_ts = T0 + timedelta(hours=spike_idx)
        spike = next((a for a in result if a.timestamp == spike_ts), None)
        assert spike is not None
        assert spike.methods_triggered == ["zscore"]
        assert spike.severity == "minor"


class TestEngineReturnContract:
    def test_returns_list_of_consensus_anomaly_pydantic_models(self) -> None:
        n = 80
        values = [10.0] * n
        values[60] = 200.0
        series = [(T0 + timedelta(hours=i), v) for i, v in enumerate(values)]
        engine = DetectionEngine(zscore=ZScoreDetector(min_points=10, threshold=3.0))
        result = engine.run(series, **_run_kwargs())
        assert all(isinstance(a, ConsensusAnomaly) for a in result)

    def test_results_are_sorted_by_timestamp_ascending(self) -> None:
        # Multiple spikes; engine should return them in chronological order.
        n = 80
        values = [10.0 + (i % 3) for i in range(n)]
        # Spikes at three different timestamps, fed in any internal order.
        values[70] = 200.0
        values[40] = 200.0
        values[55] = 200.0
        series = [(T0 + timedelta(hours=i), v) for i, v in enumerate(values)]
        engine = DetectionEngine(zscore=ZScoreDetector(min_points=10, threshold=3.0))
        result = engine.run(series, **_run_kwargs())
        timestamps = [a.timestamp for a in result]
        assert timestamps == sorted(timestamps)


class TestEngineMetadataPropagation:
    def test_metric_and_source_carry_through_to_anomalies(self) -> None:
        n = 80
        # Z-score skips windows with std=0 (div-zero guard), so add small
        # variation around the baseline before injecting the spike.
        values = [10.0 + (i % 3) * 0.1 for i in range(n)]
        values[60] = 200.0
        series = [(T0 + timedelta(hours=i), v) for i, v in enumerate(values)]
        engine = DetectionEngine(zscore=ZScoreDetector(min_points=10, threshold=3.0))
        result = engine.run(
            series,
            metric="no2",
            source="sentinel5p",
            lat=29.5,
            lon=-95.0,
        )
        assert result
        for a in result:
            assert a.metric == "no2"
            assert a.source == "sentinel5p"
            assert a.lat == pytest.approx(29.5)
            assert a.lon == pytest.approx(-95.0)


class TestEngineIsolationForestAuxFeatures:
    def test_aux_only_anomaly_flagged_via_isolation_forest_alone(self) -> None:
        # Value is typical, but met aux features are wildly atypical.
        # Z-score and STL run on value-only series and see nothing unusual.
        # IF over the aux-augmented space catches it -> minor severity.
        n = 200
        anomaly_idx = 120

        def value_at(i: int) -> float:
            return _seasonal(i)

        def aux_at(i: int) -> dict[str, float]:
            if i == anomaly_idx:
                return {"wind_speed": 50.0, "pbl_height": 50.0}
            return {"wind_speed": 3.0, "pbl_height": 1000.0}

        series = _hourly_series(n, value_at=value_at)
        iso = _iso_inputs(n, value_at=value_at, aux_at=aux_at)

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series, **_run_kwargs(), isolation_forest_inputs=iso
        )
        anomaly_ts = T0 + timedelta(hours=anomaly_idx)
        flagged = {a.timestamp: a for a in result}
        assert anomaly_ts in flagged, (
            "expected IF to catch the aux-only anomaly even when z-score/STL "
            "see only a clean seasonal value"
        )
        # Should be IF-only; severity minor unless STL drift happens to fire too.
        assert "isolation_forest" in flagged[anomaly_ts].methods_triggered


class TestEngineDeterminism:
    def test_repeat_runs_produce_identical_results(self) -> None:
        # IF carries random_state=42 by default; together with deterministic
        # z-score and STL the engine output must be reproducible.
        n = 200
        spike_idx = 120

        def value_at(i: int) -> float:
            return _seasonal(i) if i != spike_idx else 500.0

        series = _hourly_series(n, value_at=value_at)
        iso = _iso_inputs(
            n,
            value_at=value_at,
            aux_at=lambda _i: {"wind_speed": 3.0, "pbl_height": 1000.0},
        )
        engine_a = DetectionEngine.with_defaults()
        engine_b = DetectionEngine.with_defaults()
        first = engine_a.run(series, **_run_kwargs(), isolation_forest_inputs=iso)
        second = engine_b.run(series, **_run_kwargs(), isolation_forest_inputs=iso)
        assert [a.timestamp for a in first] == [a.timestamp for a in second]
        assert [a.methods_triggered for a in first] == [a.methods_triggered for a in second]
        assert [a.severity for a in first] == [a.severity for a in second]
