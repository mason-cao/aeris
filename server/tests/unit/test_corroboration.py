"""Phase 2 corroboration scorer — shared aggregator.

The aggregator turns per-source verdicts (+1 supporting / -1 contradicting /
0 silent) into the scalar ``corroboration_score`` and ``evidence_n`` that the
research analysis correlates against expert labels. Spec:
docs/specs/2026-05-21-corroboration-scorer-design.md.
"""


from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    SUPPORTING,
    ClaimType,
    aggregate_verdicts,
    classify_claim,
    low_corroboration_flag,
    score_claim,
    score_concentration_elevation,
    score_meteorological_state,
    score_transport_direction,
)
from app.provenance.openaq_pm25 import verified_monitor_entity_ids


_OPENAQ_MONITOR_IDS = tuple(
    sorted(verified_monitor_entity_ids(), key=int)
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


def test_mixed_verdicts_aggregate_by_independent_channel():
    # Channel-aware (2026-06-24 audit rec #3): openaq (ground) supports;
    # sentinel5p (satellite) is silent; noaa_gfs + openweather are ONE NWP
    # channel and disagree, netting to silent. Only the ground channel carries a
    # verdict, so evidence_n is 1 — not the old per-source count of 3.
    result = aggregate_verdicts(
        {
            "openaq": SUPPORTING,
            "sentinel5p": SILENT,
            "noaa_gfs": CONTRADICTING,
            "openweather": SUPPORTING,
        }
    )
    assert result.supporting == 1
    assert result.contradicting == 0
    assert result.evidence_n == 1
    assert result.corroboration_score == 1.0
    assert result.unverified is False
    assert result.per_channel_verdicts["nwp"] == SILENT


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
    for metrics in metrics_by_source.values():
        for block in metrics.values():
            nearest = block.get("nearest_in_time")
            if isinstance(nearest, dict) and nearest.get("v") is not None:
                nearest.setdefault("dt_minutes", 0.0)
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


def test_concentration_under_threshold_claim_supported_by_openaq():
    # "stayed below 50" is the mirror of "exceeded 80": a low nearest value
    # supports it. The qualitative-elevation branch would invert this.
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 8.0, "max": 22.0, "mean": 14.0},
                    "nearest_in_time": {"v": 18.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "NO2 stayed below 50 ppb at the nearest monitor.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_concentration_under_threshold_claim_contradicted_by_openaq():
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 70.0, "max": 95.0, "mean": 82.0},
                    "nearest_in_time": {"v": 88.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "NO2 stayed below 50 ppb at the nearest monitor.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_concentration_elevated_but_below_threshold_is_under_claim():
    # "elevated but stayed below 80" is an under-claim despite the word
    # "elevated": nearest 60 sits below 80, so it is supported.
    summary = _summary_with(
        {
            "openaq": {
                "no2": {
                    "unit": "ppb",
                    "value_range": {"min": 40.0, "max": 65.0, "mean": 55.0},
                    "nearest_in_time": {"v": 60.0},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "NO2 was elevated but stayed below 80 ppb.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


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


def test_concentration_so2_negative_column_is_silent_not_contradicting():
    # S5P SO2 columns scatter symmetrically about zero at Houston background; a
    # negative retrieval is sub-detection noise, not a real low concentration,
    # so it must abstain rather than contradict an elevation claim.
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_so2_column": {
                    "unit": "mol/m^2",
                    "nearest_in_time": {"v": -9.79e-05},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "SO2 column exceeded 0.0006 mol/m2.", summary
    )
    assert verdicts["sentinel5p"] == SILENT


def test_concentration_so2_subdetection_positive_is_silent():
    # The in-window maximum (~0.23 DU) is equally below the detection limit; a
    # small positive column must not spuriously support a claim either.
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_so2_column": {
                    "unit": "mol/m^2",
                    "nearest_in_time": {"v": 1.0e-04},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "SO2 column exceeded 0.00005 mol/m2.", summary
    )
    assert verdicts["sentinel5p"] == SILENT


def test_concentration_so2_above_detection_limit_is_scored():
    # A genuine column above the detection limit is scored normally — the gate
    # silences noise, not real SO2 signal.
    summary = _summary_with(
        {
            "sentinel5p": {
                "s5p_so2_column": {
                    "unit": "mol/m^2",
                    "nearest_in_time": {"v": 6.0e-04},
                }
            }
        }
    )
    verdicts, _ = score_concentration_elevation(
        "SO2 column exceeded 0.0005 mol/m2.", summary
    )
    assert verdicts["sentinel5p"] == SUPPORTING


def test_concentration_ground_so2_negative_is_silent():
    # Ground SO2 (TCEQ/EPA AQS, ppb) scatters about zero below the monitor's
    # detection limit — 54-62% of in-window ground SO2 reads negative. A negative
    # reading is non-physical noise and must abstain, not contradict, mirroring
    # the satellite-column gate.
    summary = _summary_with(
        {"tceq": {"so2": {"unit": "ppb", "nearest_in_time": {"v": -1.6}}}}
    )
    verdicts, _ = score_concentration_elevation("SO2 exceeded 1 ppb.", summary)
    assert verdicts["tceq"] == SILENT


def test_concentration_ground_so2_below_ppb_floor_is_silent():
    # A small positive reading under the ~0.5 ppb floor is equally sub-detection
    # noise and must not spuriously support an elevation claim.
    summary = _summary_with(
        {"tceq": {"so2": {"unit": "ppb", "nearest_in_time": {"v": 0.3}}}}
    )
    verdicts, _ = score_concentration_elevation("SO2 exceeded 0.2 ppb.", summary)
    assert verdicts["tceq"] == SILENT


def test_concentration_ground_so2_above_floor_is_scored():
    # Real ground SO2 above the detection floor is scored normally — the gate
    # silences noise, not signal.
    summary = _summary_with(
        {"tceq": {"so2": {"unit": "ppb", "nearest_in_time": {"v": 5.0}}}}
    )
    verdicts, _ = score_concentration_elevation("SO2 exceeded 1 ppb.", summary)
    assert verdicts["tceq"] == SUPPORTING


def test_concentration_negative_ground_no2_is_silent():
    # The non-physical floor is species-agnostic: a negative ground NO2 reading
    # is instrument noise about zero and abstains rather than contradicting.
    summary = _summary_with(
        {"tceq": {"no2": {"unit": "ppb", "nearest_in_time": {"v": -2.7}}}}
    )
    verdicts, _ = score_concentration_elevation("NO2 exceeded 10 ppb.", summary)
    assert verdicts["tceq"] == SILENT


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
                                "entity_id": _OPENAQ_MONITOR_IDS[0],
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
                                "entity_id": _OPENAQ_MONITOR_IDS[0],
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


def test_concentration_qualitative_within_sigma_band_is_silent():
    # Baseline 12 +/- ~0.82 (sigma); nearest 12.5 sits above the mean but
    # inside the sigma band — too close to call either way. The old bare
    # mean-exceedance rule scored this a coin-flip "support".
    summary = _with_anomaly(
        _summary_with(
            {
                "openaq": {
                    "pm25": {
                        "unit": "ug/m3",
                        "value_range": {"min": 10.0, "max": 55.0, "mean": 20.0},
                        "nearest_in_time": {"v": 12.5},
                        "entities": [
                            {
                                "entity_id": _OPENAQ_MONITOR_IDS[0],
                                "series": _hourly_series(
                                    0, [12.0, 11.0, 13.0, 12.0, 12.0, 12.5]
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
    assert verdicts["openaq"] == SILENT
    assert "noise band" in note


def test_concentration_qualitative_below_baseline_contradicts():
    summary = _with_anomaly(
        _summary_with(
            {
                "openaq": {
                    "pm25": {
                        "unit": "ug/m3",
                        "value_range": {"min": 8.0, "max": 14.0, "mean": 11.0},
                        "nearest_in_time": {"v": 9.0},
                        "entities": [
                            {
                                "entity_id": _OPENAQ_MONITOR_IDS[0],
                                "series": _hourly_series(
                                    0, [12.0, 11.0, 13.0, 12.0, 12.0, 9.0]
                                ),
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
    assert verdicts["openaq"] == CONTRADICTING


def _trigger_summary() -> dict:
    """An openaq pm25 anomaly whose own channel would otherwise vote on it."""
    return {
        "schema_version": 1,
        "anomaly": {
            "timestamp": "2026-06-05T05:00:00+00:00",
            "source": "openaq",
            "metric": "pm25",
        },
        "sources": {
            "openaq": {
                "metrics": {
                    "pm25": {
                        "unit": "ug/m3",
                        "nearest_in_time": {"v": 52.0},
                        "entities": [
                            {
                                "entity_id": _OPENAQ_MONITOR_IDS[0],
                                "series": _hourly_series(
                                    0, [12.0, 11.0, 13.0, 12.0, 12.0, 52.0]
                                ),
                            }
                        ],
                    }
                }
            },
            "purpleair": {
                "metrics": {
                    "pm25": {
                        "unit": "ug/m3",
                        "nearest_in_time": {"v": 180.0},
                        "entities": [
                            {
                                "entity_id": "sensor-1",
                                "series": _hourly_series(
                                    0, [40.0, 42.0, 41.0, 40.0, 41.0, 180.0]
                                ),
                            }
                        ],
                    }
                }
            },
        },
    }


def test_concentration_trigger_channel_support_is_demoted_to_silent():
    # Detection selected on openaq pm25 being elevated, so openaq's support of
    # "pm25 was elevated" is tautological — the independent optical channel
    # (PurpleAir vs its own baseline) carries the real corroboration.
    verdicts, note = score_concentration_elevation(
        "PM2.5 was elevated near the anomaly.", _trigger_summary()
    )
    assert verdicts["openaq"] == SILENT
    assert verdicts["purpleair"] == SUPPORTING
    assert "trigger-channel support demoted" in note


def test_concentration_trigger_channel_contradiction_is_kept():
    # The asymmetry: the trigger source misreading is tautology-free evidence.
    # A claim misstating the very data that triggered detection (threshold 500
    # vs nearest 52) must keep the contradiction.
    verdicts, _ = score_concentration_elevation(
        "PM2.5 exceeded 500 ug/m3.", _trigger_summary()
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_concentration_no_anomaly_metadata_skips_trigger_demotion():
    # Summaries without anomaly source/metric (synthetic fixtures) behave as
    # before — the demotion needs to know what triggered detection.
    summary = _trigger_summary()
    summary["anomaly"] = {"timestamp": "2026-06-05T05:00:00+00:00"}
    verdicts, _ = score_concentration_elevation(
        "PM2.5 was elevated near the anomaly.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_concentration_surface_threshold_not_judged_by_column():
    # "exceeded 80 ppb" vs a column density in mol/m^2 (~1e-4) is a guaranteed
    # spurious contradiction by unit accident; the satellite must abstain from
    # surface-worded absolute claims.
    summary = _summary_with(
        {
            "openaq": {
                "no2": {"unit": "ppb", "nearest_in_time": {"v": 82.0}}
            },
            "sentinel5p": {
                "s5p_no2_column": {
                    "unit": "mol/m^2",
                    "nearest_in_time": {"v": 8.5e-5},
                }
            },
        }
    )
    verdicts, note = score_concentration_elevation("NO2 exceeded 80 ppb.", summary)
    assert verdicts["openaq"] == SUPPORTING
    assert verdicts["sentinel5p"] == SILENT
    assert "not comparable" in note


def test_concentration_column_threshold_not_judged_by_surface():
    # The mirror: a column-worded threshold must not be "supported" by a ppb
    # surface reading that dwarfs it numerically.
    summary = _summary_with(
        {
            "openaq": {
                "no2": {"unit": "ppb", "nearest_in_time": {"v": 82.0}}
            },
            "sentinel5p": {
                "s5p_no2_column": {
                    "unit": "mol/m^2",
                    "nearest_in_time": {"v": 2.1e-4},
                }
            },
        }
    )
    verdicts, _ = score_concentration_elevation(
        "The NO2 column exceeded 0.0002 mol/m2.", summary
    )
    assert verdicts["sentinel5p"] == SUPPORTING
    assert verdicts["openaq"] == SILENT


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
        "nearest_in_time": {
            "t": entities[-1]["series"][-1][0],
            "v": values[-1],
            "entity_id": entities[-1]["entity_id"],
        },
        "entities": entities,
    }


# --- atmospheric_trap (type 4) ---


def _pbl_block(
    nearest_v: float,
    nearest_iso: str,
    series: list[list],
    *,
    entity_id: str = "cell-a",
    other_entities: list[dict] | None = None,
) -> dict:
    """A pbl_height metric block with an explicit nearest and pooled series."""
    entities = [{"entity_id": entity_id, "series": series}]
    entities.extend(other_entities or [])
    return {
        "nearest_in_time": {
            "v": nearest_v,
            "t": nearest_iso,
            "entity_id": entity_id,
        },
        "value_range": {
            "mean": sum(
                v
                for entity in entities
                for _, v in entity["series"]
            )
            / sum(len(entity["series"]) for entity in entities)
        },
        "entities": entities,
    }


def _pbl_cycles(by_day_hour: dict[tuple[int, int], float]) -> list[list]:
    """[iso, value] rows for {(june_day, utc_hour): pbl_m} GFS cycles."""
    return [
        [f"2026-06-{day:02d}T{hour:02d}:00:00+00:00", v]
        for (day, hour), v in sorted(by_day_hour.items())
    ]


def test_trap_suppressed_pbl_supports_inversion_claim():
    # Nearest 18Z PBL (300 m) sits far below the other days' 18Z values
    # (~1400 m): genuinely suppressed mixing, whatever the wording (inversion,
    # capping, shallow PBL) — the T850-vs-surface criterion this replaces was
    # unreachable in a Houston summer (0/100 June hours) and auto-contradicted
    # every inversion claim.
    series = _pbl_cycles(
        {
            (3, 18): 1350.0,
            (4, 18): 1450.0,
            (5, 18): 300.0,
            (4, 6): 250.0,
            (5, 6): 260.0,
        }
    )
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(300.0, "2026-06-05T18:00:00+00:00", series)
            }
        }
    )
    verdicts, note = score_atmospheric_trap(
        "A thermal inversion trapped emissions near the surface.", summary
    )
    assert verdicts["noaa_gfs"] == SUPPORTING
    assert "same-hour" in note


def test_trap_normal_pbl_contradicts_inversion_claim():
    series = _pbl_cycles(
        {
            (3, 18): 1350.0,
            (4, 18): 1450.0,
            (5, 18): 1500.0,
            (4, 6): 250.0,
            (5, 6): 260.0,
        }
    )
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(1500.0, "2026-06-05T18:00:00+00:00", series)
            }
        }
    )
    verdicts, _ = score_atmospheric_trap(
        "An inversion kept the pollution near the ground.", summary
    )
    assert verdicts["noaa_gfs"] == CONTRADICTING


def test_trap_nocturnal_low_pbl_is_not_auto_support():
    # A 250 m PBL at 06Z is a normal night, not suppression: the same-hour
    # comparison must read it against other nights (~250 m), not against the
    # 72 h all-hours mean the old check used (which any nocturnal claim beat).
    series = _pbl_cycles(
        {
            (3, 6): 240.0,
            (4, 6): 260.0,
            (5, 6): 250.0,
            (4, 18): 1400.0,
            (5, 18): 1500.0,
        }
    )
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(250.0, "2026-06-05T06:00:00+00:00", series)
            }
        }
    )
    verdicts, _ = score_atmospheric_trap(
        "A shallow boundary layer trapped emissions overnight.", summary
    )
    assert verdicts["noaa_gfs"] == CONTRADICTING


def test_trap_insufficient_same_hour_history_is_silent():
    # One same-hour peer is not a diurnal baseline; abstain rather than guess.
    series = _pbl_cycles({(4, 18): 1400.0, (5, 18): 300.0})
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(300.0, "2026-06-05T18:00:00+00:00", series)
            }
        }
    )
    verdicts, note = score_atmospheric_trap(
        "An inversion trapped the pollution.", summary
    )
    assert verdicts["noaa_gfs"] == SILENT
    assert "insufficient same-hour" in note


def test_trap_same_cell_filter_rejects_spatial_replicates() -> None:
    event_series = _pbl_cycles({(4, 18): 100.0, (5, 18): 0.0})
    other_cell = {
        "entity_id": "cell-b",
        "series": _pbl_cycles({(6, 18): 300.0}),
    }
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    0.0,
                    "2026-06-05T18:00:00+00:00",
                    event_series,
                    other_entities=[other_cell],
                )
            }
        }
    )

    verdicts, note = score_atmospheric_trap(
        "A shallow boundary layer trapped pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "distinct-day" in note
    assert "n=1" in note


def test_trap_same_day_replicates_do_not_satisfy_distinct_day_floor() -> None:
    series = [
        ["2026-06-04T18:00:00+00:00", 100.0],
        ["2026-06-04T18:30:00+00:00", 300.0],
        ["2026-06-05T18:00:00+00:00", 0.0],
    ]
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    0.0, "2026-06-05T18:00:00+00:00", series
                )
            }
        }
    )

    verdicts, note = score_atmospheric_trap(
        "An inversion trapped the pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "distinct-day" in note
    assert "n=1" in note


def test_trap_population_sd_worked_example_and_exact_boundary() -> None:
    # peers {100, 300}: mean=200, pstdev=100, and
    # mean - 2*pstdev = 0 = min(a, b) - |a-b|/2.
    assert 200.0 - 2.0 * 100.0 == 100.0 - abs(100.0 - 300.0) / 2.0 == 0.0
    series = _pbl_cycles({(4, 18): 100.0, (5, 18): 0.0, (6, 18): 300.0})
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    0.0, "2026-06-05T18:00:00+00:00", series
                )
            }
        }
    )

    verdicts, note = score_atmospheric_trap(
        "A shallow boundary layer trapped pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SUPPORTING
    assert "threshold=0.0" in note
    assert "n=2" in note


def test_trap_value_just_above_two_sigma_floor_is_silent() -> None:
    series = _pbl_cycles({(4, 18): 100.0, (5, 18): 0.1, (6, 18): 300.0})
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    0.1, "2026-06-05T18:00:00+00:00", series
                )
            }
        }
    )

    verdicts, _ = score_atmospheric_trap(
        "A shallow boundary layer trapped pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT


def test_trap_value_exactly_at_reference_mean_contradicts() -> None:
    series = _pbl_cycles({(4, 18): 100.0, (5, 18): 200.0, (6, 18): 300.0})
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    200.0, "2026-06-05T18:00:00+00:00", series
                )
            }
        }
    )

    verdicts, _ = score_atmospheric_trap(
        "An inversion trapped the pollution.", summary
    )

    assert verdicts["noaa_gfs"] == CONTRADICTING


def test_trap_zero_spread_reference_is_silent() -> None:
    series = _pbl_cycles({(4, 18): 300.0, (5, 18): 250.0, (6, 18): 300.0})
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    250.0, "2026-06-05T18:00:00+00:00", series
                )
            }
        }
    )

    verdicts, note = score_atmospheric_trap(
        "A shallow boundary layer trapped pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "zero-spread" in note
    assert "n=2" in note


def test_trap_naive_and_aware_timestamps_match_in_utc() -> None:
    series = [
        ["2026-06-04T18:00:00+00:00", 100.0],
        ["2026-06-05T18:00:00", 0.0],
        ["2026-06-06T14:00:00-04:00", 300.0],
    ]
    summary = _summary_with(
        {
            "noaa_gfs": {
                "pbl_height": _pbl_block(
                    0.0, "2026-06-05T18:00:00", series
                )
            }
        }
    )

    verdicts, note = score_atmospheric_trap(
        "A shallow boundary layer trapped pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SUPPORTING
    assert "18Z" in note


def test_trap_missing_event_cell_is_silent() -> None:
    series = _pbl_cycles({(4, 18): 100.0, (5, 18): 0.0, (6, 18): 300.0})
    block = _pbl_block(0.0, "2026-06-05T18:00:00+00:00", series)
    del block["nearest_in_time"]["entity_id"]
    summary = _summary_with({"noaa_gfs": {"pbl_height": block}})

    verdicts, note = score_atmospheric_trap(
        "A shallow boundary layer trapped pollution.", summary
    )

    assert verdicts["noaa_gfs"] == SILENT
    assert "missing event grid cell" in note


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
    assert verdicts == {"noaa_gfs": SILENT}


# --- temporal_pattern (type 5) ---


def test_temporal_rising_trend_supported():
    entities = [
        _entity(_OPENAQ_MONITOR_IDS[0], _series(10, [10, 12, 15, 18, 22, 26]))
    ]
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern(
        "PM2.5 concentrations climbed steadily through the afternoon.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_temporal_rising_claim_contradicted_by_falling_series():
    entities = [
        _entity(_OPENAQ_MONITOR_IDS[0], _series(10, [30, 26, 22, 15, 12, 8]))
    ]
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern(
        "PM2.5 levels rose all afternoon.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_temporal_too_few_points_is_silent():
    entities = [_entity(_OPENAQ_MONITOR_IDS[0], _series(10, [10, 12]))]
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern("PM2.5 rose sharply.", summary)
    assert verdicts["openaq"] == SILENT


def test_temporal_rising_claim_reads_the_hours_into_the_anomaly():
    # Rising into a 12:00 anomaly, falling after. Pooled over the whole
    # window the trend reads "down"; the claim is about the build-up.
    rise_then_fall = _entity(
        _OPENAQ_MONITOR_IDS[0],
        _series(6, [5, 10, 15, 20, 25, 30, 28, 24, 20, 16, 12, 8, 4, 2]),
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


def test_source_type_point_silent_when_stations_lack_coverage():
    # Each station reports too few in-window readings for its mean to anchor the
    # spatial-CV verdict. Like the background_vs_event scorer, the point/area
    # check must return silent on the data-quality precondition rather than read
    # a verdict off single-reading "means".
    entities = [
        _entity("a", _series(10, [80, 85, 82])),
        _entity("b", _series(10, [10, 11, 9])),
        _entity("c", _series(10, [11, 10, 12])),
    ]
    summary = _summary_with({"openaq": {"no2": _metric_from_entities(entities)}})
    verdicts, note = score_emissions_source_type(
        "Persistent NO2 consistent with a Ship Channel point source.", summary
    )
    assert verdicts["openaq"] == SILENT
    assert "obs" in note


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


def _stations(
    means: list[float],
    obs_per_station: int = 6,
    *,
    entity_ids: tuple[str, ...] | None = None,
    entity_offset: int = 0,
) -> list[dict]:
    return [
        _entity(
            entity_ids[entity_offset + i] if entity_ids else f"s{i}",
            _series(10, [m] * obs_per_station),
        )
        for i, m in enumerate(means)
    ]


def test_background_regional_claim_supported_by_uniform_stations():
    entities = _stations(
        [20, 21, 19, 22, 20], entity_ids=_OPENAQ_MONITOR_IDS
    )
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "Elevated PM2.5 across all monitors suggests regional transport.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_background_regional_claim_contradicted_by_localized_spike():
    entities = _stations(
        [95, 12, 10, 11, 13], entity_ids=_OPENAQ_MONITOR_IDS
    )
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "The haze was regional, not a local source.", summary
    )
    assert verdicts["openaq"] == CONTRADICTING


def test_background_local_claim_supported_by_localized_spike():
    entities = _stations(
        [95, 12, 10, 11, 13], entity_ids=_OPENAQ_MONITOR_IDS
    )
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "An isolated spike at one monitor points to a local source.", summary
    )
    assert verdicts["openaq"] == SUPPORTING


def test_background_precondition_unmet_is_silent():
    # Bracco data-quality precondition: needs >= 5 qualifying stations.
    entities = _stations([20, 21, 19], entity_ids=_OPENAQ_MONITOR_IDS)
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, note = score_background_vs_event(
        "Uniform PM2.5 suggests a regional event.", summary
    )
    assert verdicts["openaq"] == SILENT
    assert "precondition" in note.lower()


def test_background_sparse_stations_do_not_count_toward_precondition():
    # 5 stations but two have fewer than 6 observations each.
    entities = _stations(
        [20, 21, 19], obs_per_station=8, entity_ids=_OPENAQ_MONITOR_IDS
    ) + _stations(
        [22, 18],
        obs_per_station=3,
        entity_ids=_OPENAQ_MONITOR_IDS,
        entity_offset=3,
    )
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})
    verdicts, note = score_background_vs_event(
        "Uniform PM2.5 suggests a regional event.", summary
    )
    assert verdicts["openaq"] == SILENT
    assert "precondition" in note.lower()


# --- new-channel integration for the descriptive types ---
# The May-June coverage audit found OpenAQ carries zero in-window NO2/SO2/CO
# for Houston; scorers hardcoded to OpenAQ were structurally silent for the
# petrochemical species TCEQ supplies, and ignored PurpleAir's 70-sensor
# PM2.5 field for the spatial checks.


def test_temporal_trend_scored_from_tceq_no2():
    entities = [_entity("cams-1", _series(10, [10, 12, 15, 18, 22, 26]))]
    summary = _summary_with({"tceq": {"no2": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern(
        "NO2 concentrations climbed steadily through the afternoon.", summary
    )
    assert verdicts["tceq"] == SUPPORTING


def test_temporal_trend_scored_from_purpleair_pm25():
    # Trend direction is invariant under PurpleAir's multiplicative bias.
    entities = [_entity("pa-1", _series(10, [30, 26, 22, 15, 12, 8]))]
    summary = _summary_with({"purpleair": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_temporal_pattern("PM2.5 dropped through the day.", summary)
    assert verdicts["purpleair"] == SUPPORTING


def test_chemistry_no2_leg_scored_from_tceq():
    summary = _summary_with(
        {
            "tceq": {
                "no2": {
                    "nearest_in_time": {"v": 60.0},
                    "value_range": {"mean": 20.0},
                }
            }
        }
    )
    verdicts, _ = score_chemistry(
        "Elevated NO2 alongside elevated HCHO points to fresh emissions.", summary
    )
    assert verdicts["tceq"] == SUPPORTING


def test_secondary_formation_uses_tceq_no2_with_openaq_ozone():
    # The realistic Houston pairing: ozone from OpenAQ, NO2 from TCEQ. The
    # pooled ground check assigns the lag verdict to both contributors, and
    # the aggregator counts their shared channel once.
    no2 = _entity("cams", _series(10, [40, 55, 30, 20, 15, 12]))  # peak 11:00
    o3 = _entity("aq", _series(10, [20, 25, 30, 45, 60, 72]))  # peak 15:00
    summary = _summary_with(
        {
            "tceq": {"no2": _metric_from_entities([no2])},
            "openaq": {"ozone": _metric_from_entities([o3])},
        }
    )
    verdicts, _ = score_secondary_formation(
        "Afternoon ozone peak consistent with photochemical formation.", summary
    )
    assert verdicts["openaq"] == SUPPORTING
    assert verdicts["tceq"] == SUPPORTING
    assert aggregate_verdicts(verdicts).evidence_n == 1  # one shared channel


def test_background_spatial_cv_scored_from_purpleair():
    entities = _stations([20, 21, 19, 22, 20])
    summary = _summary_with({"purpleair": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_background_vs_event(
        "Elevated PM2.5 across all monitors suggests regional transport.", summary
    )
    assert verdicts["purpleair"] == SUPPORTING
    assert verdicts["openaq"] == SILENT


def test_source_type_spatial_cv_scored_from_purpleair():
    entities = [
        _entity("pa-a", _series(10, [80, 85, 82, 88])),
        _entity("pa-b", _series(10, [10, 11, 9, 12])),
        _entity("pa-c", _series(10, [11, 10, 12, 9])),
    ]
    summary = _summary_with({"purpleair": {"pm25": _metric_from_entities(entities)}})
    verdicts, _ = score_emissions_source_type(
        "PM2.5 pattern consistent with a point source facility.", summary
    )
    assert verdicts["purpleair"] == SUPPORTING


# --- dispatch: under-cued primary masks a contradicted secondary ---


def test_under_cued_primary_type_drops_a_contradicted_secondary():
    # A compound claim routes to two types: rank-1 background_vs_event
    # (regional uniformity) and rank-2 concentration_elevation ("exceeding
    # 80 ppb"). score_claim scores only the primary. Here the data leaves the
    # background scorer under-cued -- too few stations for its precondition,
    # so it returns silent -- and the claim ships unverified even though its
    # concentration sub-claim is flatly contradicted (nearest ~11 ppb, not
    # >80). The rank-2 contradiction never reaches the score: a known limit of
    # single-primary scoring, pinned here so a dispatch change can't silently
    # alter it. Fixing it (fall through to rank-2) is a scoring-design change.
    claim = "PM2.5 was elevated across all stations in the region, exceeding 80 ppb."
    entities = _stations(
        [12.0, 11.0], entity_ids=_OPENAQ_MONITOR_IDS
    )  # only 2 stations -> precondition unmet
    summary = _summary_with({"openaq": {"pm25": _metric_from_entities(entities)}})

    matched = classify_claim(claim)
    assert matched[0] is ClaimType.BACKGROUND_VS_EVENT
    assert ClaimType.CONCENTRATION_ELEVATION in matched

    # If the secondary were scored, it would contradict the "exceeding 80 ppb".
    rank2_verdicts, _ = score_concentration_elevation(claim, summary)
    assert rank2_verdicts["openaq"] == CONTRADICTING

    # But only the under-cued primary is scored, so nothing is corroborated or
    # contradicted -- the claim ships unverified ("green").
    scored = score_claim(claim, summary)
    assert scored.claim_type is ClaimType.BACKGROUND_VS_EVENT
    assert scored.result.unverified is True
    assert scored.result.corroboration_score is None
