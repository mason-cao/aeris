"""Phase 2 corroboration scorer — shared aggregator.

The aggregator turns per-source verdicts (+1 supporting / -1 contradicting /
0 silent) into the scalar ``corroboration_score`` and ``evidence_n`` that the
research analysis correlates against expert labels. Spec:
docs/specs/2026-05-21-corroboration-scorer-design.md.
"""

import math

from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    SUPPORTING,
    aggregate_verdicts,
    low_corroboration_flag,
    score_concentration_elevation,
    score_meteorological_state,
    score_transport_direction,
)


def test_all_silent_returns_null_and_unverified():
    result = aggregate_verdicts({"openaq": SILENT, "noaa_gfs": SILENT})
    assert result.corroboration_score is None
    assert result.evidence_n == 0
    assert result.unverified is True


def test_empty_verdicts_is_unverified():
    result = aggregate_verdicts({})
    assert result.corroboration_score is None
    assert result.evidence_n == 0
    assert result.unverified is True


def test_all_supporting_scores_plus_one():
    result = aggregate_verdicts(
        {"openaq": SUPPORTING, "sentinel5p": SUPPORTING, "noaa_gfs": SUPPORTING}
    )
    assert result.corroboration_score == 1.0
    assert result.evidence_n == 3
    assert result.supporting == 3
    assert result.contradicting == 0
    assert result.unverified is False


def test_all_contradicting_scores_minus_one():
    result = aggregate_verdicts(
        {"openaq": CONTRADICTING, "openweather": CONTRADICTING}
    )
    assert result.corroboration_score == -1.0
    assert result.evidence_n == 2
    assert result.contradicting == 2


def test_mixed_verdicts_average_over_evidence_n():
    result = aggregate_verdicts(
        {
            "openaq": SUPPORTING,
            "sentinel5p": SILENT,
            "noaa_gfs": CONTRADICTING,
            "openweather": SUPPORTING,
        }
    )
    assert result.supporting == 2
    assert result.contradicting == 1
    assert result.evidence_n == 3
    assert math.isclose(result.corroboration_score, 1 / 3, rel_tol=1e-9)
    assert result.unverified is False


def test_silent_sources_excluded_from_evidence_n():
    result = aggregate_verdicts(
        {"openaq": SUPPORTING, "sentinel5p": SILENT, "noaa_gfs": SILENT}
    )
    assert result.evidence_n == 1
    assert result.corroboration_score == 1.0


def test_result_preserves_per_source_verdicts():
    verdicts = {"openaq": SUPPORTING, "noaa_gfs": CONTRADICTING}
    result = aggregate_verdicts(verdicts)
    assert result.per_source_verdicts == verdicts


def test_low_corroboration_flag_requires_strong_negative_and_two_sources():
    # Strongly contradicted across >= 2 independent sources -> flagged.
    assert low_corroboration_flag(-0.6, evidence_n=2) is True
    assert low_corroboration_flag(-1.0, evidence_n=3) is True
    # A lone contradicting source is too little evidence to flag.
    assert low_corroboration_flag(-1.0, evidence_n=1) is False
    # Weak disagreement is not a flag.
    assert low_corroboration_flag(-0.4, evidence_n=3) is False
    # No evidence at all -> never flagged.
    assert low_corroboration_flag(None, evidence_n=0) is False


# --- concentration_elevation (headline type 1: OpenAQ + Sentinel-5P) ---


def _summary_with(metrics_by_source: dict) -> dict:
    """A minimal enrichment summary carrying {source: {metric: {...}}}."""
    return {
        "schema_version": 1,
        "sources": {
            src: {"metrics": metrics}
            for src, metrics in metrics_by_source.items()
        },
    }


def test_concentration_threshold_claim_supported_by_openaq():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 60.0, "max": 85.0, "mean": 72.0},
                    "nearest_in_time": {"v": 82.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "Ground-level NO2 exceeded 80 ppb in the afternoon.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_concentration_threshold_claim_contradicted_by_openaq():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 8.0, "max": 15.0, "mean": 11.0},
                    "nearest_in_time": {"v": 12.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation("NO2 exceeded 80 ppb.", summary)
    assert verdicts["openaq"] == CONTRADICTING


def test_concentration_unmatched_pollutant_is_silent():
    summary = _summary_with(
        {
            "openaq": {
                "ozone": {
                    "unit": "ppb",
                    "value_range": {"min": 20.0, "max": 40.0, "mean": 30.0},
                    "nearest_in_time": {"v": 35.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation("NO2 exceeded 80 ppb.", summary)
    assert verdicts.get("openaq", SILENT) == SILENT


def _with_anomaly(summary: dict, timestamp: str) -> dict:
    return {**summary, "anomaly": {"timestamp": timestamp}}


def _hourly_series(start_hour: int, values: list[float]) -> list[list]:
    return [
        [f"2026-06-05T{start_hour + i:02d}:00:00+00:00", v]
        for i, v in enumerate(values)
    ]


def test_concentration_qualitative_elevated_uses_pre_anomaly_baseline():
    # Steady ~12 before the event, 52 at the anomaly. The baseline must come
    # from points ending before the anomaly — the in-window mean would carry
    # the spike itself and corroborate any restatement of the detection.
    summary = _with_anomaly(
        _summary_with(
            {
                "openaq": {
                    "pm25": {
                        "unit": "ug/m3",
                        "value_range": {"min": 10.0, "max": 55.0, "mean": 20.0},
                        "nearest_in_time": {"v": 52.0},
                        "entities": [
                            {
                                "entity_id": "s1",
                                "series": _hourly_series(
                                    0, [12.0, 11.0, 13.0, 12.0, 12.0, 52.0]
                                ),
                            }
                        ],
                    }
                }
            }
        ),
        "2026-06-05T05:00:00+00:00",
    )
    verdicts, note = score_concentration_elevation(
        "PM2.5 was elevated across the area.", summary
    )
    assert verdicts["openaq"] == SUPPORTING
    assert "pre-anomaly baseline" in note


def test_concentration_qualitative_without_baseline_is_silent():
    # Two pre-anomaly points are not a baseline; no verdict either way.
    summary = _with_anomaly(
        _summary_with(
            {
                "openaq": {
                    "pm25": {
                        "unit": "ug/m3",
                        "value_range": {"min": 10.0, "max": 55.0, "mean": 20.0},
                        "nearest_in_time": {"v": 52.0},
                        "entities": [
                            {
                                "entity_id": "s1",
                                "series": _hourly_series(3, [12.0, 13.0, 52.0]),
                            }
                        ],
                    }
                }
            }
        ),
        "2026-06-05T05:00:00+00:00",
    )
    verdicts, _ = score_concentration_elevation(
        "PM2.5 was elevated across the area.", summary
    )
    assert verdicts["openaq"] == SILENT


def test_concentration_point_value_within_tolerance_supports():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 60.0, "max": 90.0, "mean": 72.0},
                    "nearest_in_time": {"v": 82.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "NO2 was around 75 ppb at the nearest monitor.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_concentration_point_value_far_off_contradicts():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 25.0, "max": 35.0, "mean": 30.0},
                    "nearest_in_time": {"v": 30.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "NO2 was around 80 ppb at the nearest monitor.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_concentration_threshold_ignores_clock_times():
    # "exceeded typical levels at 14:00" carries no threshold number; 14 from
    # the clock time must not become one. With no quantity and no baseline
    # series the verdict stays silent.
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 8.0, "max": 15.0, "mean": 11.0},
                    "nearest_in_time": {"v": 12.0},
                }
            }
        }
    )
    verdicts, note = score_concentration_elevation(
        "NO2 exceeded typical levels at 14:00.", summary
    )
    assert verdicts["openaq"] == SILENT
    assert "no pre-anomaly baseline" in note


def test_concentration_sentinel_column_supports_no2_claim():
    summary = _with_anomaly(
        _summary_with(
            {
                "sentinel5p": {
                    "s5p_no2_column": {
                        "unit": "mol/m^2",
                        "value_range": {"min": 4.0e-5, "max": 9.0e-5, "mean": 6.0e-5},
                        "nearest_in_time": {"v": 8.5e-5},
                        "entities": [
                            {
                                "entity_id": "granules",
                                "series": _hourly_series(
                                    0, [4.0e-5, 4.5e-5, 5.0e-5, 4.2e-5, 8.5e-5]
                                ),
                            }
                        ],
                    }
                }
            }
        ),
        "2026-06-05T06:00:00+00:00",
    )
    verdicts, _ = score_concentration_elevation(
        "Tropospheric NO2 was elevated.", summary
    )
    assert verdicts["sentinel5p"] == SUPPORTING


# --- transport_direction (headline type 2: NOAA GFS 10m wind + OpenWeather) ---


def test_transport_southerly_wind_matches_gfs():
    # GFS u=0, v=4 -> wind FROM the south (180). "Southerly winds" -> from south.
    summary = _summary_with(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": 0.0}},
                "v_10m": {"nearest_in_time": {"v": 4.0}},
            }
        }
    )
    verdicts, _ = score_transport_direction(
        "Southerly winds pushed the plume inland.", summary
    )
    assert verdicts["noaa_gfs"] == SUPPORTING


def test_transport_northward_transport_matches_gfs():
    # Wind from the south (180) carries pollution toward the north.
    summary = _summary_with(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": 0.0}},
                "v_10m": {"nearest_in_time": {"v": 4.0}},
            }
        }
    )
    verdicts, _ = score_transport_direction(
        "Emissions were carried northward over the bay.", summary
    )
    assert verdicts["noaa_gfs"] == SUPPORTING


def test_transport_opposite_direction_contradicts():
    summary = _summary_with(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": 0.0}},
                "v_10m": {"nearest_in_time": {"v": 4.0}},
            }
        }
    )
    verdicts, _ = score_transport_direction(
        "Northerly winds kept the air clean.", summary
    )
    assert verdicts["noaa_gfs"] == CONTRADICTING


def test_transport_uses_openweather_wind_direction():
    # OpenWeather reports wind FROM 175 deg (~south); "southerly" -> from 180.
    summary = _summary_with(
        {"openweather": {"wind_direction": {"nearest_in_time": {"v": 175.0}}}}
    )
    verdicts, _ = score_transport_direction("Southerly flow dominated.", summary)
    assert verdicts["openweather"] == SUPPORTING


def test_transport_no_wind_data_is_silent():
    summary = _summary_with({"openaq": {"no2": {"nearest_in_time": {"v": 50.0}}}})
    verdicts, _ = score_transport_direction("Southerly winds.", summary)
    assert verdicts.get("noaa_gfs", SILENT) == SILENT
    assert verdicts.get("openweather", SILENT) == SILENT


# --- meteorological_state (headline type 3: OpenWeather + NOAA GFS) ---


def test_met_stagnant_supported_by_low_winds():
    # GFS speed ~1.2 m/s, OpenWeather 1.5 m/s -> both below the 2 m/s stagnant cut.
    summary = _summary_with(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": 0.8}},
                "v_10m": {"nearest_in_time": {"v": 0.9}},
            },
            "openweather": {"wind_speed": {"nearest_in_time": {"v": 1.5}}},
        }
    )
    verdicts, _ = score_meteorological_state(
        "Conditions were stagnant with barely any air movement.", summary
    )
    assert verdicts["noaa_gfs"] == SUPPORTING
    assert verdicts["openweather"] == SUPPORTING


def test_met_stagnant_contradicted_by_strong_winds():
    # GFS speed ~6.4 m/s -> not stagnant.
    summary = _summary_with(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": 5.0}},
                "v_10m": {"nearest_in_time": {"v": 4.0}},
            }
        }
    )
    verdicts, _ = score_meteorological_state("It was stagnant all afternoon.", summary)
    assert verdicts["noaa_gfs"] == CONTRADICTING


def test_met_numeric_temperature_supported_by_openweather():
    summary = _summary_with(
        {"openweather": {"temperature": {"nearest_in_time": {"v": 35.0}}}}
    )
    verdicts, _ = score_meteorological_state(
        "Surface temperatures were around 34 C.", summary
    )
    assert verdicts["openweather"] == SUPPORTING


def test_met_no_relevant_data_is_silent():
    summary = _summary_with({"openaq": {"no2": {"nearest_in_time": {"v": 50.0}}}})
    verdicts, _ = score_meteorological_state("Conditions were stagnant.", summary)
    assert verdicts.get("noaa_gfs", SILENT) == SILENT
    assert verdicts.get("openweather", SILENT) == SILENT


# --- the seven descriptive claim types (4-10) ---------------------------

from app.llm.corroboration import (  # noqa: E402
    score_atmospheric_trap,
    score_background_vs_event,
    score_chemistry,
    score_emissions_source_type,
    score_point_source_attribution,
    score_secondary_formation,
    score_temporal_pattern,
)


def _series(start_hour: int, values: list[float]) -> list[list]:
    """Hourly [iso, value] pairs starting at start_hour UTC on 2026-06-05."""
    return [
        [f"2026-06-05T{(start_hour + i) % 24:02d}:00:00+00:00", v]
        for i, v in enumerate(values)
    ]


def _entity(entity_id: str, series: list[list], *, lat: float = 29.76, lon: float = -95.37) -> dict:
    return {
        "entity_id": entity_id,
        "lat": lat,
        "lon": lon,
        "distance_km": 5.0,
        "n_points": len(series),
        "series": series,
    }


def _metric_from_entities(entities: list[dict]) -> dict:
    values = [v for e in entities for _, v in e["series"]]
    return {
        "unit": "ug/m3",
        "n_points": len(values),
        "n_entities": len(entities),
        "value_range": {
            "min": min(values),
            "max": max(values),
            "mean": sum(values) / len(values),
        },
        "nearest_in_time": {"v": values[-1]},
        "entities": entities,
    }


# --- atmospheric_trap (type 4) ---


def test_trap_inversion_supported_when_aloft_warmer_than_surface():
    summary = _summary_with(
        {
            "noaa_gfs": {
                "t_850": {"nearest_in_time": {"v": 24.0}},
                "pbl_height": {
                    "nearest_in_time": {"v": 300.0},
                    "value_range": {"mean": 700.0},
                },
            },
            "openweather": {"temperature": {"nearest_in_time": {"v": 20.0}}},
        }
    )
    verdicts, _ = score_atmospheric_trap(
        "A low boundary layer and a thermal inversion trapped emissions near the surface.",
        summary,
    )
    assert verdicts["noaa_gfs"] == SUPPORTING
    assert verdicts["openweather"] == SUPPORTING


def test_trap_inversion_contradicted_when_surface_warmer():
    summary = _summary_with(
        {
            "noaa_gfs": {"t_850": {"nearest_in_time": {"v": 15.0}}},
            "openweather": {"temperature": {"nearest_in_time": {"v": 30.0}}},
        }
    )
    verdicts, _ = score_atmospheric_trap(
        "An inversion kept the pollution near the ground.", summary
    )
    assert verdicts["noaa_gfs"] == CONTRADICTING
    assert verdicts["openweather"] == CONTRADICTING


def test_trap_numeric_pbl_checked_within_tolerance():
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": {
                    "nearest_in_time": {"v": 480.0},
                    "value_range": {"mean": 900.0},
                }
            }
        }
    )
    verdicts, _ = score_atmospheric_trap(
        "The boundary layer height was around 400 m, trapping emissions.", summary
    )
    assert verdicts["noaa_gfs"] == SUPPORTING


def test_trap_no_data_is_silent():
    verdicts, _ = score_atmospheric_trap(
        "An inversion trapped pollution.", _summary_with({})
    )
    assert verdicts["noaa_gfs"] == SILENT
    assert verdicts["openweather"] == SILENT


# --- temporal_pattern (type 5) ---


def test_temporal_rising_trend_supported():
    entities = [_entity("a", _series(10, [10, 12, 15, 18, 22, 26]))]
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern(
        "PM2.5 concentrations climbed steadily through the afternoon.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_temporal_rising_claim_contradicted_by_falling_series():
    entities = [_entity("a", _series(10, [30, 26, 22, 15, 12, 8]))]
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern(
        "PM2.5 levels rose all afternoon.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_temporal_too_few_points_is_silent():
    entities = [_entity("a", _series(10, [10, 12]))]
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern("PM2.5 rose sharply.", summary)
    assert verdicts["openaq"] == SILENT


def test_temporal_rising_claim_reads_the_hours_into_the_anomaly():
    # Rising into a 12:00 anomaly, falling after. Pooled over the whole
    # window the trend reads "down"; the claim is about the build-up.
    rise_then_fall = _entity(
        "s1", _series(6, [5, 10, 15, 20, 25, 30, 28, 24, 20, 16, 12, 8, 4, 2])
    )
    summary = {
        **_summary_with(
            {"openaq": {"pm25": _metric_from_entities([rise_then_fall])}}
        ),
        "anomaly": {"timestamp": "2026-06-05T12:00:00+00:00"},
    }
    verdicts, _ = score_temporal_pattern(
        "PM2.5 climbed steadily through the morning.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_temporal_without_direction_or_pollutant_returns_no_verdicts():
    verdicts, note = score_temporal_pattern(
        "The situation evolved overnight.", _summary_with({})
    )
    assert verdicts == {}
    assert note


# --- chemistry (type 6, qualitative-only) ---


def test_chemistry_elevated_hcho_and_depressed_ozone_supported():
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_hcho_column": {
                    "nearest_in_time": {"v": 8e-5},
                    "value_range": {"mean": 4e-5},
                }
            },
            "openaq": {
                "ozone": {
                    "nearest_in_time": {"v": 20.0},
                    "value_range": {"mean": 45.0},
                }
            },
        }
    )
    verdicts, _ = score_chemistry(
        "Elevated HCHO with depressed ozone suggests fresh VOC emissions.", summary
    )
    assert verdicts["sentinel5p"] == SUPPORTING
    assert verdicts["openaq"] == SUPPORTING


def test_chemistry_hcho_silent_granule_is_silent_not_contradicting():
    summary = _summary_with({"sentinel5p": {}})
    verdicts, _ = score_chemistry("Elevated HCHO points to fresh VOCs.", summary)
    assert verdicts["sentinel5p"] == SILENT


def test_chemistry_noisy_zone_is_silent():
    # Nearest below the mean but within the 50% noise buffer: no verdict.
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_hcho_column": {
                    "nearest_in_time": {"v": 3.6e-5},
                    "value_range": {"mean": 4e-5},
                }
            }
        }
    )
    verdicts, _ = score_chemistry("Elevated HCHO points to fresh VOCs.", summary)
    assert verdicts["sentinel5p"] == SILENT


def test_chemistry_clear_contradiction():
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_hcho_column": {
                    "nearest_in_time": {"v": 1e-5},
                    "value_range": {"mean": 4e-5},
                }
            }
        }
    )
    verdicts, _ = score_chemistry("Elevated HCHO points to fresh VOCs.", summary)
    assert verdicts["sentinel5p"] == CONTRADICTING


# --- point_source_attribution (type 7, qualitative-only) ---


def _summary_with_anomaly(metrics_by_source: dict) -> dict:
    summary = _summary_with(metrics_by_source)
    summary["anomaly"] = {"lat": 29.76, "lon": -95.37}
    return summary


def test_point_source_wind_from_source_direction_supports():
    # Claimed source ESE of the anomaly; wind from ~101 deg (ESE).
    summary = _summary_with_anomaly(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": -5.0}},
                "v_10m": {"nearest_in_time": {"v": 1.0}},
            }
        }
    )
    verdicts, _ = score_point_source_attribution(
        "Plume signature consistent with a refinery upset near 29.73N, -95.22W.",
        summary,
    )
    assert verdicts["noaa_gfs"] == SUPPORTING


def test_point_source_wind_from_opposite_direction_contradicts():
    summary = _summary_with_anomaly(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": 5.0}},
                "v_10m": {"nearest_in_time": {"v": -1.0}},
            }
        }
    )
    verdicts, _ = score_point_source_attribution(
        "Plume signature consistent with a refinery upset near 29.73N, -95.22W.",
        summary,
    )
    assert verdicts["noaa_gfs"] == CONTRADICTING


def test_point_source_uses_openweather_direction():
    summary = _summary_with_anomaly(
        {"openweather": {"wind_direction": {"nearest_in_time": {"v": 100.0}}}}
    )
    verdicts, _ = score_point_source_attribution(
        "Plume consistent with an upset near 29.73N, -95.22W.", summary
    )
    assert verdicts["openweather"] == SUPPORTING


def test_point_source_without_coordinates_is_silent():
    summary = _summary_with_anomaly(
        {
            "noaa_gfs": {
                "u_10m": {"nearest_in_time": {"v": -5.0}},
                "v_10m": {"nearest_in_time": {"v": 1.0}},
            }
        }
    )
    verdicts, _ = score_point_source_attribution(
        "This came from the refinery by the Ship Channel.", summary
    )
    assert verdicts["noaa_gfs"] == SILENT


# --- emissions_source_type (type 8) ---


def test_source_type_mobile_supported_by_morning_peak():
    # Peak at 13:00 UTC = 08:00 CDT, inside the morning-rush window.
    entities = [_entity("a", _series(10, [5, 8, 12, 30, 14, 9, 6]))]
    summary = _summary_with({"openaq": {"no2": _metric_from_entities(entities)}})
    verdicts, _ = score_emissions_source_type(
        "NO2 pattern consistent with rush-hour mobile traffic emissions.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_source_type_point_supported_by_localized_concentration():
    entities = [
        _entity("a", _series(10, [80, 85, 82, 88])),
        _entity("b", _series(10, [10, 11, 9, 12])),
        _entity("c", _series(10, [11, 10, 12, 9])),
    ]
    summary = _summary_with({"openaq": {"no2": _metric_from_entities(entities)}})
    verdicts, _ = score_emissions_source_type(
        "Persistent NO2 consistent with a Ship Channel point source.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_source_type_point_contradicted_by_uniform_field():
    entities = [
        _entity("a", _series(10, [20, 21, 19, 22])),
        _entity("b", _series(10, [21, 20, 22, 19])),
        _entity("c", _series(10, [19, 22, 20, 21])),
    ]
    summary = _summary_with({"openaq": {"no2": _metric_from_entities(entities)}})
    verdicts, _ = score_emissions_source_type(
        "Persistent NO2 from a single industrial point source.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_source_type_unknown_intent_returns_no_verdicts():
    verdicts, note = score_emissions_source_type(
        "The pollution had some origin.", _summary_with({})
    )
    assert verdicts == {}
    assert note


# --- secondary_formation (type 9) ---


def test_secondary_formation_supported_by_lag_and_clear_sky():
    no2 = _entity("n", _series(10, [40, 55, 30, 20, 15, 12]))  # peak 11:00
    o3 = _entity("o", _series(10, [20, 25, 30, 45, 60, 72]))  # peak 15:00
    summary = _summary_with(
        {
            "openaq": {
                "no2": _metric_from_entities([no2]),
                "ozone": _metric_from_entities([o3]),
            },
            "openweather": {"cloud_cover": {"value_range": {"mean": 15.0}}},
        }
    )
    verdicts, _ = score_secondary_formation(
        "Afternoon ozone peak consistent with photochemical formation from morning NOx.",
        summary,
    )
    assert verdicts["openaq"] == SUPPORTING
    assert verdicts["openweather"] == SUPPORTING


def test_secondary_formation_contradicted_when_ozone_peaks_first():
    no2 = _entity("n", _series(10, [10, 12, 15, 30, 45, 50]))  # peak 15:00
    o3 = _entity("o", _series(10, [60, 72, 45, 30, 22, 18]))  # peak 11:00
    summary = _summary_with(
        {
            "openaq": {
                "no2": _metric_from_entities([no2]),
                "ozone": _metric_from_entities([o3]),
            }
        }
    )
    verdicts, _ = score_secondary_formation(
        "The afternoon ozone formed from the morning NO2 emissions.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_secondary_formation_overcast_contradicts_insolation():
    summary = _summary_with(
        {"openweather": {"cloud_cover": {"value_range": {"mean": 85.0}}}}
    )
    verdicts, _ = score_secondary_formation(
        "Photochemical ozone formation from morning emissions.", summary
    )
    assert verdicts["openweather"] == CONTRADICTING
    assert verdicts["openaq"] == SILENT


def test_secondary_formation_only_reads_the_anomaly_day():
    # Day-2 (June 4) series would invert the lag if pooled into the test; the
    # anomaly is on June 5, so only that day's peaks count. With June 4's o3
    # peak included, the global o3 peak would precede the no2 peak.
    day2_o3 = [
        [f"2026-06-04T{h:02d}:00:00+00:00", v]
        for h, v in [(14, 90.0), (15, 95.0), (16, 88.0)]
    ]
    no2 = _entity("n", _series(10, [40, 55, 30, 20, 15, 12]))  # peak 11:00 Jun 5
    o3 = _entity("o", day2_o3 + _series(10, [20, 25, 30, 45, 60, 72]))
    summary = {
        **_summary_with(
            {
                "openaq": {
                    "no2": _metric_from_entities([no2]),
                    "ozone": _metric_from_entities([o3]),
                }
            }
        ),
        "anomaly": {"timestamp": "2026-06-05T15:00:00+00:00"},
    }
    verdicts, note = score_secondary_formation(
        "The afternoon ozone formed from the morning NO2 emissions.", summary
    )
    assert verdicts["openaq"] == SUPPORTING
    assert "anomaly day" in note


# --- background_vs_event (type 10) ---


def _stations(means: list[float], obs_per_station: int = 6) -> list[dict]:
    return [
        _entity(f"s{i}", _series(10, [m] * obs_per_station))
        for i, m in enumerate(means)
    ]


def test_background_regional_claim_supported_by_uniform_stations():
    entities = _stations([20, 21, 19, 22, 20])
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "Elevated PM2.5 across all monitors suggests regional transport.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_background_regional_claim_contradicted_by_localized_spike():
    entities = _stations([95, 12, 10, 11, 13])
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "The haze was regional, not a local source.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_background_local_claim_supported_by_localized_spike():
    entities = _stations([95, 12, 10, 11, 13])
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "An isolated spike at one monitor points to a local source.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_background_precondition_unmet_is_silent():
    # Bracco data-quality precondition: needs >= 5 qualifying stations.
    entities = _stations([20, 21, 19])
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, note = score_background_vs_event(
        "Uniform PM2.5 suggests a regional event.", summary
    )
    assert verdicts["openaq"] == SILENT
    assert "precondition" in note.lower()


def test_background_sparse_stations_do_not_count_toward_precondition():
    # 5 stations but two have fewer than 6 observations each.
    entities = _stations([20, 21, 19], obs_per_station=8) + _stations(
        [22, 18], obs_per_station=3
    )
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, note = score_background_vs_event(
        "Uniform PM2.5 suggests a regional event.", summary
    )
    assert verdicts["openaq"] == SILENT
    assert "precondition" in note.lower()
