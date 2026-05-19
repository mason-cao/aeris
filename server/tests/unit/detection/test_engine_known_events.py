"""Multi-event integration tests for the detection engine.

Builds synthetic Houston-like time-series (hourly PM2.5 plus contemporaneous
wind/PBL aux features) with hand-injected anomaly events of known character
and asserts the engine catches each at the expected severity tier.

These tests stand in for the real ``known_anomalies.json`` fixture set the
Month 2 plan calls for; once we have hand-labeled real anomalies, they'll
move into ``tests/detection/test_known_anomalies.py`` with the same harness.
"""

import math
from datetime import datetime, timedelta, timezone

import pytest

from app.detection.consensus import ALL_METHODS, ConsensusAnomaly
from app.detection.engine import DetectionEngine
from app.detection.isolation_forest import IsolationForestInput


T0 = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698


def _houston_pm25_baseline(i: int) -> float:
    """Synthetic Houston PM2.5 baseline.

    Diurnal cycle (peak around morning rush hour, ~07:00) layered with a
    weekly cycle. Mean ~22 ug/m^3, swing ~6 ug/m^3 (rough match to OpenAQ
    Houston-area station summaries).
    """
    diurnal = 6.0 * math.sin(2.0 * math.pi * (i - 1) / 24)
    weekly = 2.0 * math.sin(2.0 * math.pi * i / (24 * 7))
    return 22.0 + diurnal + weekly


def _typical_aux(i: int) -> dict[str, float]:
    """Typical wind / PBL for a Houston spring day."""
    wind = 3.5 + 1.0 * math.sin(2.0 * math.pi * i / 24)
    pbl = 900.0 + 300.0 * math.sin(2.0 * math.pi * (i - 6) / 24)
    return {"wind_speed": wind, "pbl_height": pbl}


def _build_series_with_events(
    n_hours: int,
    events: dict[int, tuple[float, dict[str, float] | None]],
) -> tuple[list[tuple[datetime, float]], list[IsolationForestInput]]:
    """Build (univariate series, IF inputs) with injected events.

    ``events`` maps hour-index -> (override_value, override_aux_or_None).
    A None aux override means "keep typical aux at that hour".
    """
    series: list[tuple[datetime, float]] = []
    iso: list[IsolationForestInput] = []
    for i in range(n_hours):
        ts = T0 + timedelta(hours=i)
        if i in events:
            value, aux_override = events[i]
            aux = aux_override if aux_override is not None else _typical_aux(i)
        else:
            value = _houston_pm25_baseline(i)
            aux = _typical_aux(i)
        series.append((ts, value))
        iso.append(IsolationForestInput(timestamp=ts, value=value, aux_features=aux))
    return series, iso


class TestKnownHoustonEvents:
    def test_refinery_upset_severity_severe(self) -> None:
        # Refinery upset: PM2.5 spikes ~10x typical with stagnant aux
        # (low wind, collapsed PBL). All three detectors should fire.
        n = 200
        spike_idx = 120
        # 10x typical: ~220 ug/m^3 (well into "hazardous" AQI band)
        events = {
            spike_idx: (
                220.0,
                {"wind_speed": 0.5, "pbl_height": 200.0},  # stagnation
            ),
        }
        series, iso = _build_series_with_events(n, events)

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series,
            metric="pm25",
            source="openaq",
            lat=HOUSTON_LAT,
            lon=HOUSTON_LON,
            isolation_forest_inputs=iso,
        )
        spike_ts = T0 + timedelta(hours=spike_idx)
        spike = next((a for a in result if a.timestamp == spike_ts), None)
        assert spike is not None
        assert sorted(spike.methods_triggered) == sorted(ALL_METHODS)
        assert spike.severity == "severe"
        assert spike.value == pytest.approx(220.0)
        assert spike.expected_value is not None  # z-score / STL fill this
        assert spike.z_score is not None

    def test_moderate_event_severity_moderate_or_better(self) -> None:
        # A moderate elevation: ~3-4 sigma over baseline but not extreme.
        # Z-score and STL should agree; IF may or may not.
        # We accept "moderate" (2 methods) or "severe" (3) — both are valid.
        n = 200
        spike_idx = 100
        events = {spike_idx: (60.0, None)}  # ~3x typical, typical aux
        series, iso = _build_series_with_events(n, events)

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series,
            metric="pm25",
            source="openaq",
            lat=HOUSTON_LAT,
            lon=HOUSTON_LON,
            isolation_forest_inputs=iso,
        )
        spike_ts = T0 + timedelta(hours=spike_idx)
        spike = next((a for a in result if a.timestamp == spike_ts), None)
        assert spike is not None
        assert spike.severity in ("moderate", "severe")
        assert "zscore" in spike.methods_triggered

    def test_meteorological_only_event_via_isolation_forest(self) -> None:
        # The value stays typical but the meteorology is wildly atypical:
        # very high wind plus deep PBL. Z-score and STL see nothing.
        # IF over the augmented space should catch it.
        n = 200
        event_idx = 110
        baseline_value = _houston_pm25_baseline(event_idx)
        events = {
            event_idx: (
                baseline_value,
                {"wind_speed": 25.0, "pbl_height": 3000.0},
            ),
        }
        series, iso = _build_series_with_events(n, events)

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series,
            metric="pm25",
            source="openaq",
            lat=HOUSTON_LAT,
            lon=HOUSTON_LON,
            isolation_forest_inputs=iso,
        )
        event_ts = T0 + timedelta(hours=event_idx)
        flagged = next((a for a in result if a.timestamp == event_ts), None)
        assert flagged is not None
        # IF must be among the triggering methods; Z-score should not be.
        assert "isolation_forest" in flagged.methods_triggered
        assert "zscore" not in flagged.methods_triggered

    def test_multiple_distinct_events_are_all_caught(self) -> None:
        # Three distinct events at different timestamps; engine should
        # catch each as its own ConsensusAnomaly.
        n = 240
        events = {
            80: (180.0, {"wind_speed": 0.5, "pbl_height": 200.0}),  # severe
            140: (90.0, None),                                        # moderate
            200: (150.0, {"wind_speed": 0.3, "pbl_height": 150.0}),  # severe
        }
        series, iso = _build_series_with_events(n, events)

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series,
            metric="pm25",
            source="openaq",
            lat=HOUSTON_LAT,
            lon=HOUSTON_LON,
            isolation_forest_inputs=iso,
        )
        flagged_timestamps = {a.timestamp for a in result}
        for idx in events:
            ts = T0 + timedelta(hours=idx)
            assert ts in flagged_timestamps, (
                f"engine missed injected event at hour {idx}"
            )

    def test_no_phantom_anomalies_on_baseline_hours(self) -> None:
        # Inject one extreme spike; verify the engine doesn't also flag
        # the immediately-adjacent baseline hours as severe. (IF will
        # likely emit a few minor flags scattered through the series due
        # to its contamination prior — those are acceptable as long as
        # they're not at the same timestamps as the injected event.)
        n = 200
        spike_idx = 120
        events = {spike_idx: (220.0, {"wind_speed": 0.5, "pbl_height": 200.0})}
        series, iso = _build_series_with_events(n, events)

        engine = DetectionEngine.with_defaults()
        result = engine.run(
            series,
            metric="pm25",
            source="openaq",
            lat=HOUSTON_LAT,
            lon=HOUSTON_LON,
            isolation_forest_inputs=iso,
        )
        spike_ts = T0 + timedelta(hours=spike_idx)
        severe_anomalies = [a for a in result if a.severity == "severe"]
        # Only the injected spike should reach severe consensus.
        assert len(severe_anomalies) == 1
        assert severe_anomalies[0].timestamp == spike_ts


class TestKnownEventsReproducibility:
    def test_known_event_results_are_deterministic_across_runs(self) -> None:
        # The full pipeline (z-score + STL + IF + consensus) must produce
        # byte-identical results on repeat runs given the same input.
        # IF's random_state=42 default and the deterministic statistical
        # detectors combine to give a stable result.
        n = 200
        events = {120: (220.0, {"wind_speed": 0.5, "pbl_height": 200.0})}
        series, iso = _build_series_with_events(n, events)

        def _run() -> list[ConsensusAnomaly]:
            return DetectionEngine.with_defaults().run(
                series,
                metric="pm25",
                source="openaq",
                lat=HOUSTON_LAT,
                lon=HOUSTON_LON,
                isolation_forest_inputs=iso,
            )

        first = _run()
        second = _run()
        assert [a.timestamp for a in first] == [a.timestamp for a in second]
        assert [a.severity for a in first] == [a.severity for a in second]
        assert [a.methods_triggered for a in first] == [
            a.methods_triggered for a in second
        ]
