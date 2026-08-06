"""Cross-source wind-direction disagreement guard (declared 2026-08-06)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.llm.corroboration import (
    CONTRADICTING,
    DEFAULT_WIND_TOLERANCE,
    SILENT,
    SUPPORTING,
    WIND_DISAGREEMENT_BEARING_MULTIPLE,
    WIND_DISAGREEMENT_STATUS,
    aggregate_verdicts,
    score_point_source_attribution,
    score_transport_direction,
    wind_disagreement_decision,
    wind_disagreement_manifest_payload,
)


def _metric(
    nearest_value: float,
    values: list[tuple[str, float]],
    *,
    entity_id: str = "station-a",
    dt_minutes: float = 0.0,
) -> dict:
    return {
        "nearest_in_time": {
            "t": values[0][0],
            "v": nearest_value,
            "entity_id": entity_id,
            "dt_minutes": dt_minutes,
        },
        "entities": [
            {
                "entity_id": entity_id,
                "series": [[timestamp, value] for timestamp, value in values],
            }
        ],
    }


def _summary(sources: dict) -> dict:
    return {
        "anomaly": {"lat": 29.76, "lon": -95.37},
        "sources": {
            source: {"metrics": metrics} for source, metrics in sources.items()
        },
    }


# Brisk winds that vary, so mean - 2*pstdev stays under the event speed and the
# B2 guard passes; these tests are about disagreement, not calmness.
_T0 = "2026-06-15T00:00:00+00:00"
_T1 = "2026-06-15T01:00:00+00:00"
_TIMES = [(_T0, 6.0), (_T1, 10.0)]
_WINDY_SPEED = _metric(8.0, _TIMES)
_EVENT_SPEED_MS = 8.0


def _gfs_metrics(bearing: float, speed: float = _EVENT_SPEED_MS) -> dict:
    """GFS reports u/v components, so a bearing has to be projected back."""
    import math

    def components(deg: float, magnitude: float) -> tuple[float, float]:
        radians = math.radians(deg)
        return -magnitude * math.sin(radians), -magnitude * math.cos(radians)

    u_event, v_event = components(bearing, speed)
    # A second window sample so pstdev > 0 and the raw cutoff falls below the
    # event speed; same bearing, slower, at the other timestamp.
    u_other, v_other = components(bearing, speed * 0.5)

    def block(event: float, other: float) -> dict:
        return {
            "nearest_in_time": {
                "t": _T0,
                "v": event,
                "entity_id": "gfs:29.75,-95.25",
                "dt_minutes": 0.0,
            },
            "entities": [
                {
                    "entity_id": "gfs:29.75,-95.25",
                    "series": [[_T0, event], [_T1, other]],
                }
            ],
        }

    return {"u_10m": block(u_event, u_other), "v_10m": block(v_event, v_other)}


def _direction_summary(directions: dict[str, float]) -> dict:
    """A summary where every named source reports a fresh, non-calm direction."""
    sources: dict[str, dict] = {}
    for source, bearing in directions.items():
        if source == "noaa_gfs":
            sources[source] = _gfs_metrics(bearing)
        else:
            sources[source] = {
                "wind_direction": _metric(bearing, _TIMES),
                "wind_speed": _WINDY_SPEED,
            }
    return _summary(sources)


# --- the threshold is derived, not tuned -----------------------------------


def test_default_threshold_is_exactly_twice_the_bearing_band() -> None:
    # Two sources further apart than 2*bearing_deg cannot both fall within
    # bearing_deg of one claimed bearing, so this factor is the point at which
    # disagreement becomes logically incompatible with joint corroboration.
    # If bearing_deg ever moves, this must move with it.
    assert DEFAULT_WIND_TOLERANCE.max_disagreement_deg == (
        WIND_DISAGREEMENT_BEARING_MULTIPLE * DEFAULT_WIND_TOLERANCE.bearing_deg
    )


def test_at_the_threshold_both_sources_can_still_support_one_bearing() -> None:
    threshold = DEFAULT_WIND_TOLERANCE.max_disagreement_deg
    assert threshold == 90.0
    # North and east are exactly the threshold apart, and their midpoint,
    # northeast, sits one bearing band from each. Both must still support it,
    # which is why the guard cannot fire at or below the threshold.
    summary = _direction_summary({"noaa_gfs": 0.0, "asos": 90.0})

    verdicts, _ = score_transport_direction(
        "Winds from the northeast carried the plume.", summary
    )

    assert verdicts["noaa_gfs"] == SUPPORTING
    assert verdicts["asos"] == SUPPORTING


# --- decision function ------------------------------------------------------


@pytest.mark.parametrize(
    ("separation", "resolvable"),
    [(89.0, True), (90.0, True), (90.5, False), (91.0, False), (180.0, False)],
)
def test_boundary_is_inclusive_at_exactly_the_threshold(
    separation: float, resolvable: bool
) -> None:
    decision = wind_disagreement_decision({"a": 0.0, "b": separation})

    assert decision.resolvable is resolvable
    assert decision.max_pairwise_deg == pytest.approx(separation)


def test_separation_wraps_across_north() -> None:
    decision = wind_disagreement_decision({"a": 350.0, "b": 10.0})

    assert decision.max_pairwise_deg == pytest.approx(20.0)
    assert decision.resolvable is True


def test_maximum_pairwise_not_spread_about_a_mean() -> None:
    # Two sources agree tightly and one is antipodal. A circular mean would be
    # degenerate here; the max-pairwise statistic is well defined.
    decision = wind_disagreement_decision(
        {"asos": 180.0, "noaa_gfs": 185.0, "openweather": 0.0}
    )

    assert decision.resolvable is False
    assert decision.max_pairwise_deg == pytest.approx(180.0)
    assert decision.worst_pair == ("asos", "openweather")


@pytest.mark.parametrize("directions", [{}, {"a": 12.0}])
def test_fewer_than_two_sources_cannot_disagree(directions: dict) -> None:
    decision = wind_disagreement_decision(directions)

    assert decision.resolvable is True
    assert decision.reason == "insufficient_sources"
    assert decision.max_pairwise_deg is None


def test_guard_can_be_disabled_for_the_sensitivity_sweep() -> None:
    tolerance = replace(DEFAULT_WIND_TOLERANCE, max_disagreement_deg=None)

    decision = wind_disagreement_decision({"a": 0.0, "b": 179.0}, tolerance=tolerance)

    assert decision.resolvable is True
    assert decision.reason == "guard_disabled"
    assert decision.status == "not_configured"


@pytest.mark.parametrize("threshold", [0.0, -1.0, 180.5, float("nan")])
def test_out_of_range_threshold_is_rejected(threshold: float) -> None:
    tolerance = replace(DEFAULT_WIND_TOLERANCE, max_disagreement_deg=threshold)

    with pytest.raises(ValueError, match="max_disagreement_deg"):
        wind_disagreement_decision({"a": 0.0, "b": 10.0}, tolerance=tolerance)


def test_non_finite_bearing_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        wind_disagreement_decision({"a": 0.0, "b": float("inf")})


def test_decision_is_serializable_and_order_independent() -> None:
    first = wind_disagreement_decision({"asos": 10.0, "noaa_gfs": 200.0})
    second = wind_disagreement_decision({"noaa_gfs": 200.0, "asos": 10.0})

    assert first == second
    payload = first.to_dict()
    assert payload["directions"] == [["asos", 10.0], ["noaa_gfs", 200.0]]
    assert payload["resolvable"] is False
    assert payload["status"] == WIND_DISAGREEMENT_STATUS


# --- wiring into both direction consumers -----------------------------------


def test_transport_leg_goes_fully_silent_when_sources_cannot_agree() -> None:
    # The regression this exists for: without the guard, the NWP channel nets
    # its own split to silent and the surviving ASOS vote emits a saturated
    # verdict off a wind field two sources disagree about by 173 degrees.
    summary = _direction_summary(
        {"noaa_gfs": 187.0, "openweather": 0.0, "asos": 200.0}
    )

    verdicts, note = score_transport_direction(
        "Pollution was transported from the south.", summary
    )
    result = aggregate_verdicts(verdicts)

    assert set(verdicts.values()) == {SILENT}
    assert result.corroboration_score is None
    assert result.evidence_n == 0
    assert result.unverified is True
    assert "wind-disagreement guard SILENT" in note
    assert "max_pairwise=173" in note


def test_point_source_leg_goes_silent_when_sources_cannot_agree() -> None:
    summary = _direction_summary({"noaa_gfs": 10.0, "openweather": 190.0})
    summary["anomaly"] = {"lat": 29.76, "lon": -95.37}

    verdicts, note = score_point_source_attribution(
        "A refinery near 29.73 N, 95.22 W drove the plume.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert verdicts["openweather"] == SILENT
    assert "wind-disagreement guard SILENT" in note


def test_agreeing_sources_still_score_normally() -> None:
    summary = _direction_summary(
        {"noaa_gfs": 180.0, "openweather": 190.0, "asos": 175.0}
    )

    verdicts, note = score_transport_direction(
        "Pollution was transported from the south.", summary
    )
    result = aggregate_verdicts(verdicts)

    assert verdicts["noaa_gfs"] == SUPPORTING
    assert verdicts["asos"] == SUPPORTING
    assert result.corroboration_score == pytest.approx(1.0)
    assert result.evidence_n == 2
    assert "wind-disagreement guard passed" in note


def test_agreeing_sources_can_still_contradict_together() -> None:
    # The guard silences unresolvable fields; it must not silence a resolved
    # field that happens to contradict the claim.
    summary = _direction_summary({"noaa_gfs": 0.0, "openweather": 10.0, "asos": 5.0})

    verdicts, _ = score_transport_direction(
        "Pollution was transported from the south.", summary
    )
    result = aggregate_verdicts(verdicts)

    assert verdicts["noaa_gfs"] == CONTRADICTING
    assert verdicts["asos"] == CONTRADICTING
    assert result.corroboration_score == pytest.approx(-1.0)


def test_calm_sources_are_excluded_before_the_disagreement_is_measured() -> None:
    # A calm source's direction is already untrusted, so counting it as a
    # disagreeing party would silence events the B2 guard has handled. This is
    # the real 07356251 shape: ASOS reports the METAR calm sentinel 0/0 while
    # GFS and OpenWeather agree closely.
    summary = _direction_summary({"noaa_gfs": 193.0, "openweather": 205.0})
    summary["sources"]["asos"] = {
        "wind_direction": _metric(0.0, _TIMES),
        "wind_speed": _metric(0.0, [("2026-06-15T00:00:00+00:00", 0.0),
                                    ("2026-06-15T01:00:00+00:00", 0.0)]),
    }

    verdicts, note = score_transport_direction(
        "NO2 was carried in from the south.", summary
    )
    result = aggregate_verdicts(verdicts)

    assert verdicts["asos"] == SILENT
    assert verdicts["noaa_gfs"] == SUPPORTING
    assert verdicts["openweather"] == SUPPORTING
    assert result.evidence_n == 1
    assert "wind-disagreement guard passed" in note


def test_guard_does_not_fire_when_the_claim_has_no_bearing() -> None:
    summary = _direction_summary({"noaa_gfs": 0.0, "asos": 180.0})

    verdicts, note = score_transport_direction("PM2.5 was elevated.", summary)

    assert set(verdicts.values()) == {SILENT}
    assert "wind-disagreement guard" not in note


# --- manifest ---------------------------------------------------------------


def test_manifest_records_the_derivation_and_that_bracco_has_not_confirmed() -> None:
    payload = wind_disagreement_manifest_payload()

    assert payload["threshold_deg"] == 90.0
    assert payload["bearing_deg"] == 45.0
    assert payload["threshold_is_derived"] is True
    assert payload["status"] == WIND_DISAGREEMENT_STATUS
    assert payload["bracco_confirmed"] is False


def test_manifest_reports_a_hand_set_threshold_as_underived() -> None:
    tolerance = replace(DEFAULT_WIND_TOLERANCE, max_disagreement_deg=60.0)

    payload = wind_disagreement_manifest_payload(tolerance)

    assert payload["threshold_deg"] == 60.0
    assert payload["threshold_is_derived"] is False
