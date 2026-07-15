"""Channel-aware aggregation: corroboration counts measurement-process groups,
not raw sources (2026-06-24 audit rec #3, channel-grouping foundation)."""

from app.llm.corroboration import (
    CONTRADICTING,
    SILENT,
    SOURCE_CHANNELS,
    SUPPORTING,
    aggregate_verdicts,
    channel_of,
    score_concentration_elevation,
    score_meteorological_state,
    score_transport_direction,
)
from app.provenance.openaq_pm25 import verified_monitor_entity_ids


_OPENAQ_MONITOR_ID = min(verified_monitor_entity_ids(), key=int)


def _summary(sources_metrics: dict, anomaly_ts: str = "2026-06-15T12:00:00+00:00") -> dict:
    """Minimal enrichment summary: {source: {metric: nearest_value}}."""
    summary = {
        "anomaly": {"timestamp": anomaly_ts, "lat": 29.76, "lon": -95.37},
        "sources": {
            source: {
                "metrics": {
                    metric: {
                        "nearest_in_time": {"v": value, "dt_minutes": 0.0}
                    }
                    for metric, value in metrics.items()
                }
            }
            for source, metrics in sources_metrics.items()
        },
    }
    openaq_pm25 = (
        summary.get("sources", {})
        .get("openaq", {})
        .get("metrics", {})
        .get("pm25")
    )
    if openaq_pm25:
        value = openaq_pm25["nearest_in_time"]["v"]
        openaq_pm25["nearest_in_time"].update(
            {"t": anomaly_ts, "entity_id": _OPENAQ_MONITOR_ID}
        )
        openaq_pm25["entities"] = [
            {
                "entity_id": _OPENAQ_MONITOR_ID,
                "lat": 29.76,
                "lon": -95.37,
                "distance_km": 0.0,
                "n_points": 1,
                "series": [[anomaly_ts, value]],
            }
        ]
    purpleair_pm25 = (
        summary.get("sources", {})
        .get("purpleair", {})
        .get("metrics", {})
        .get("pm25")
    )
    if purpleair_pm25:
        value = purpleair_pm25["nearest_in_time"]["v"]
        purpleair_pm25["nearest_in_time"].update(
            {"t": anomaly_ts, "entity_id": "synthetic-purpleair"}
        )
        purpleair_pm25["entities"] = [
            {
                "entity_id": "synthetic-purpleair",
                "lat": 29.76,
                "lon": -95.37,
                "distance_km": 0.0,
                "n_points": 1,
                "series": [[anomaly_ts, value]],
            }
        ]
    return summary


class TestChannelMap:
    def test_redundant_ground_sources_share_a_channel(self) -> None:
        # TCEQ and EPA AQS are the same regulatory monitors as OpenAQ.
        assert channel_of("openaq") == channel_of("tceq") == channel_of("epa_aqs")

    def test_nwp_sources_share_a_channel(self) -> None:
        assert channel_of("noaa_gfs") == channel_of("openweather")

    def test_independent_channels_are_distinct(self) -> None:
        channels = {
            channel_of(s)
            for s in ("openaq", "purpleair", "sentinel5p", "noaa_gfs", "asos")
        }
        assert len(channels) == 5  # ground / optical / satellite / nwp / met-insitu

    def test_unlisted_source_gets_its_own_channel(self) -> None:
        assert channel_of("some_future_source") == "some_future_source"


class TestAggregateChannels:
    def test_distinct_channels_count_separately(self) -> None:
        # Ground + satellite agreeing = 2 process groups (unchanged from the
        # old per-source behavior, since they were already distinct).
        r = aggregate_verdicts({"openaq": SUPPORTING, "sentinel5p": SUPPORTING})
        assert r.evidence_n == 2
        assert r.corroboration_score == 1.0
        assert r.supporting == 2

    def test_redundant_ground_sources_count_once(self) -> None:
        # The double-counting fix: three same-channel ground sources = 1 channel.
        r = aggregate_verdicts(
            {"openaq": SUPPORTING, "tceq": SUPPORTING, "epa_aqs": SUPPORTING}
        )
        assert r.evidence_n == 1
        assert r.corroboration_score == 1.0
        assert r.per_channel_verdicts == {"ground_insitu": SUPPORTING}

    def test_nwp_pair_collapses_to_one_channel(self) -> None:
        # GFS + OpenWeather are common-mode NWP — one channel, not two.
        r = aggregate_verdicts({"noaa_gfs": SUPPORTING, "openweather": SUPPORTING})
        assert r.evidence_n == 1
        assert r.corroboration_score == 1.0

    def test_genuine_two_channel_pair(self) -> None:
        # Optical PurpleAir vs regulatory OpenAQ = two channels; ground vs sat too.
        r = aggregate_verdicts({"openaq": SUPPORTING, "purpleair": SUPPORTING})
        assert r.evidence_n == 2
        r2 = aggregate_verdicts({"tceq": SUPPORTING, "sentinel5p": SUPPORTING})
        assert r2.evidence_n == 2

    def test_within_channel_disagreement_nets_to_silent(self) -> None:
        # Two sources in one channel disagreeing = no channel consensus.
        r = aggregate_verdicts({"noaa_gfs": SUPPORTING, "openweather": CONTRADICTING})
        assert r.per_channel_verdicts == {"nwp": SILENT}
        assert r.evidence_n == 0
        assert r.corroboration_score is None
        assert r.unverified is True

    def test_silent_members_do_not_sink_their_channel(self) -> None:
        r = aggregate_verdicts({"openaq": SUPPORTING, "tceq": SILENT})
        assert r.evidence_n == 1
        assert r.per_channel_verdicts == {"ground_insitu": SUPPORTING}

    def test_majority_within_channel(self) -> None:
        # Two support, one contradict in the ground channel -> net support.
        r = aggregate_verdicts(
            {"openaq": SUPPORTING, "tceq": SUPPORTING, "epa_aqs": CONTRADICTING}
        )
        assert r.per_channel_verdicts == {"ground_insitu": SUPPORTING}
        assert r.evidence_n == 1

    def test_cross_channel_conflict_scores_zero(self) -> None:
        r = aggregate_verdicts({"openaq": SUPPORTING, "sentinel5p": CONTRADICTING})
        assert r.evidence_n == 2
        assert r.corroboration_score == 0.0

    def test_all_silent_is_unverified(self) -> None:
        r = aggregate_verdicts({"openaq": SILENT, "noaa_gfs": SILENT})
        assert r.unverified is True
        assert r.corroboration_score is None
        assert r.evidence_n == 0

    def test_empty_is_unverified(self) -> None:
        r = aggregate_verdicts({})
        assert r.unverified is True
        assert r.evidence_n == 0

    def test_per_source_verdicts_preserved(self) -> None:
        verdicts = {"openaq": SUPPORTING, "tceq": SUPPORTING}
        r = aggregate_verdicts(verdicts)
        assert r.per_source_verdicts == verdicts

    def test_source_channels_map_is_complete(self) -> None:
        # Every collector source has an explicit channel assignment.
        assert set(SOURCE_CHANNELS) == {
            "openaq",
            "tceq",
            "epa_aqs",
            "purpleair",
            "sentinel5p",
            "noaa_gfs",
            "openweather",
            "asos",
        }


class TestScorerChannelInclusion:
    """The new sources actually vote: type-1 ground/optical + ASOS met channels."""

    def test_pm25_gets_regulatory_plus_optical_channels(self) -> None:
        # OpenAQ (regulatory) + PurpleAir (optical) = a genuine two-channel
        # PM2.5 pair — different instrument physics.
        summary = _summary({"openaq": {"pm25": 40.0}, "purpleair": {"pm25": 38.0}})
        verdicts, _ = score_concentration_elevation("PM2.5 exceeded 30 ug/m3", summary)
        result = aggregate_verdicts(verdicts)
        assert result.evidence_n == 2
        assert result.per_channel_verdicts == {
            "ground_insitu": SUPPORTING,
            "ground_optical": SUPPORTING,
        }

    def test_no2_ground_gap_filled_by_tceq(self) -> None:
        # The audit's exact finding: in-window ground NO2 is zero from OpenAQ.
        # TCEQ now supplies it, so the ground channel votes where it couldn't.
        summary = _summary({"tceq": {"no2": 60.0}})
        verdicts, _ = score_concentration_elevation("NO2 exceeded 50 ppb", summary)
        assert verdicts.get("tceq") == SUPPORTING
        result = aggregate_verdicts(verdicts)
        assert result.per_channel_verdicts.get("ground_insitu") == SUPPORTING

    def test_no2_ground_vs_satellite_is_two_channels(self) -> None:
        # TCEQ ground + S5P column = the genuine satellite-vs-ground pair. The
        # claim is qualitative ("elevated") so each source votes against its
        # own pre-anomaly baseline in its own units — a surface-ppb threshold
        # can never be judged by a mol/m^2 column.
        def _elevated(baseline: list[float], nearest: float) -> dict:
            series = [
                [f"2026-06-15T{h:02d}:00:00+00:00", v]
                for h, v in enumerate(baseline)
            ]
            return {
                "nearest_in_time": {"v": nearest, "dt_minutes": 0.0},
                "entities": [{"entity_id": "e", "series": series}],
            }

        summary = _summary({})
        summary["sources"] = {
            "tceq": {"metrics": {"no2": _elevated([10.0, 11.0, 10.0, 10.5], 60.0)}},
            "openaq": {"metrics": {"no2": _elevated([9.0, 10.0, 9.5, 10.0], 58.0)}},
            "sentinel5p": {
                "metrics": {
                    "s5p_no2_column": _elevated(
                        [5.0e-5, 5.5e-5, 5.2e-5, 5.1e-5], 2.1e-4
                    )
                }
            },
        }
        verdicts, _ = score_concentration_elevation("NO2 was elevated", summary)
        result = aggregate_verdicts(verdicts)
        # openaq+tceq collapse to one ground channel; s5p is the second channel.
        assert result.evidence_n == 2
        assert set(result.per_channel_verdicts) >= {"ground_insitu", "satellite_column"}

    def test_redundant_ground_sources_do_not_inflate_evidence(self) -> None:
        # openaq + tceq + epa_aqs all agreeing must count as ONE ground channel.
        summary = _summary(
            {"openaq": {"no2": 60.0}, "tceq": {"no2": 61.0}, "epa_aqs": {"no2": 59.0}}
        )
        verdicts, _ = score_concentration_elevation("NO2 exceeded 50 ppb", summary)
        result = aggregate_verdicts(verdicts)
        assert result.evidence_n == 1

    def test_asos_votes_on_transport_direction(self) -> None:
        summary = _summary({"asos": {"wind_direction": 180.0}})
        verdicts, _ = score_transport_direction("winds from the south", summary)
        assert verdicts.get("asos") == SUPPORTING
        result = aggregate_verdicts(verdicts)
        assert result.per_channel_verdicts.get("met_insitu") == SUPPORTING

    def test_asos_is_independent_channel_from_nwp(self) -> None:
        # GFS+OpenWeather (NWP) agree, ASOS (in-situ) agrees -> two channels.
        summary = _summary(
            {
                "noaa_gfs": {"u_10m": 0.0, "v_10m": 5.0},  # wind from the south
                "openweather": {"wind_direction": 180.0},
                "asos": {"wind_direction": 175.0},
            }
        )
        verdicts, _ = score_transport_direction("winds from the south", summary)
        result = aggregate_verdicts(verdicts)
        assert result.evidence_n == 2
        assert set(result.per_channel_verdicts) >= {"nwp", "met_insitu"}

    def test_asos_votes_on_meteorological_state(self) -> None:
        summary = _summary({"asos": {"wind_speed": 1.0}})
        verdicts, _ = score_meteorological_state("conditions were stagnant", summary)
        assert verdicts.get("asos") == SUPPORTING
