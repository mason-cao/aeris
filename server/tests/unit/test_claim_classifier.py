"""Claim-type classification + Phase 2 dispatch.

Step 1 of the corroboration flow: rule-based routing of each claim to one of
the 10 taxonomy types, then dispatch to that type's scorer. Spec:
docs/specs/2026-05-21-corroboration-scorer-design.md.
"""

from app.llm.corroboration import (
    HEADLINE_TYPES,
    QUALITATIVE_ONLY_TYPES,
    ClaimType,
    classify_claim,
    score_claim,
)


# --- taxonomy metadata ---


def test_claim_type_values_are_the_taxonomy_names():
    assert ClaimType.CONCENTRATION_ELEVATION.value == "concentration_elevation"
    assert ClaimType.BACKGROUND_VS_EVENT.value == "background_vs_event"
    assert ClaimType.UNCLASSIFIED.value == "unclassified"


def test_headline_types_are_the_three_from_the_memo():
    assert HEADLINE_TYPES == frozenset(
        {
            ClaimType.CONCENTRATION_ELEVATION,
            ClaimType.TRANSPORT_DIRECTION,
            ClaimType.METEOROLOGICAL_STATE,
        }
    )


def test_qualitative_only_types_per_bracco_addendum():
    assert QUALITATIVE_ONLY_TYPES == frozenset(
        {ClaimType.CHEMISTRY, ClaimType.POINT_SOURCE_ATTRIBUTION}
    )


# --- classification, one claim per type ---


def test_threshold_pollutant_claim_is_concentration_elevation():
    types = classify_claim("Ground-level NO2 exceeded 80 ppb in the afternoon.")
    assert types[0] is ClaimType.CONCENTRATION_ELEVATION


def test_directional_transport_claim_is_transport_direction():
    types = classify_claim("Pollution was carried northward by southerly winds.")
    assert types[0] is ClaimType.TRANSPORT_DIRECTION


def test_stagnation_claim_is_meteorological_state():
    types = classify_claim("Conditions were stagnant with winds below 2 m/s.")
    assert types[0] is ClaimType.METEOROLOGICAL_STATE


def test_inversion_claim_is_atmospheric_trap():
    types = classify_claim(
        "A temperature inversion trapped pollutants near the surface."
    )
    assert types[0] is ClaimType.ATMOSPHERIC_TRAP


def test_trend_claim_is_temporal_pattern():
    types = classify_claim("PM2.5 concentrations rose steadily overnight.")
    assert types[0] is ClaimType.TEMPORAL_PATTERN


def test_species_relationship_claim_is_chemistry():
    types = classify_claim(
        "Elevated HCHO and depressed NO2 point to a VOC-limited regime."
    )
    assert types[0] is ClaimType.CHEMISTRY


def test_named_facility_with_coordinates_is_point_source_attribution():
    types = classify_claim(
        "A plume from the refinery at 29.72 N, 95.21 W reached the monitor."
    )
    assert types[0] is ClaimType.POINT_SOURCE_ATTRIBUTION


def test_rush_hour_claim_is_emissions_source_type():
    types = classify_claim("The pattern is consistent with rush-hour traffic.")
    assert types[0] is ClaimType.EMISSIONS_SOURCE_TYPE


def test_photochemical_claim_is_secondary_formation():
    types = classify_claim("Ozone formed photochemically from precursors.")
    assert types[0] is ClaimType.SECONDARY_FORMATION


def test_regional_uniformity_claim_is_background_vs_event():
    types = classify_claim("Readings were uniform across all stations.")
    assert types[0] is ClaimType.BACKGROUND_VS_EVENT


def test_unmatched_claim_is_unclassified():
    assert classify_claim("The cause of this anomaly is uncertain.") == [
        ClaimType.UNCLASSIFIED
    ]


# --- classification, overlapping claims ---


def test_regional_elevation_claim_ranks_background_over_concentration():
    types = classify_claim("PM2.5 was elevated across all stations in the region.")
    assert types[0] is ClaimType.BACKGROUND_VS_EVENT
    assert ClaimType.CONCENTRATION_ELEVATION in types


def test_facility_claim_without_coordinates_ranks_source_type_first():
    # Without coordinates the point-source scorer has nothing to check, so a
    # facility claim routes to emissions_source_type, whose point/area/mobile
    # pattern check actually applies. Attribution stays in the match list.
    types = classify_claim("A plume from the refinery reached the monitor.")
    assert types[0] is ClaimType.EMISSIONS_SOURCE_TYPE
    assert ClaimType.POINT_SOURCE_ATTRIBUTION in types


def test_transport_claim_mentioning_ship_channel_stays_transport():
    # The memo's own headline example for transport_direction. "Ship Channel"
    # is a geographic reference, not an attribution cue — routing on it sent
    # headline transport claims into a qualitative-only type.
    types = classify_claim(
        "Southerly winds advected pollutants from the Ship Channel northward"
    )
    assert types[0] is ClaimType.TRANSPORT_DIRECTION


def test_memo_examples_route_to_their_taxonomy_rows():
    # The design memo's example column, one per claim type. If routing drifts,
    # the taxonomy table and the dispatcher have diverged.
    memo_examples = {
        ClaimType.CONCENTRATION_ELEVATION: (
            "Ground-level NO2 exceeded 80 ppb between 14:00-18:00 CT"
        ),
        ClaimType.TRANSPORT_DIRECTION: (
            "Southerly winds advected pollutants from the Ship Channel northward"
        ),
        ClaimType.METEOROLOGICAL_STATE: "Stagnant conditions: wind speed <2 m/s",
        ClaimType.ATMOSPHERIC_TRAP: (
            "A low PBL combined with thermal inversion trapped emissions "
            "near the surface"
        ),
        ClaimType.TEMPORAL_PATTERN: (
            "Concentrations rose monotonically over a 4-hour window"
        ),
        ClaimType.CHEMISTRY: (
            "Elevated HCHO with depressed O3 suggests fresh VOC emissions"
        ),
        ClaimType.POINT_SOURCE_ATTRIBUTION: (
            "Plume signature consistent with a refinery upset near "
            "29.73 N, -95.22 W"
        ),
        ClaimType.EMISSIONS_SOURCE_TYPE: (
            "Pattern consistent with a Ship Channel point source rather than "
            "mobile-source rush-hour"
        ),
        ClaimType.SECONDARY_FORMATION: (
            "Afternoon O3 peak consistent with photochemical formation "
            "downwind of morning NOx emissions"
        ),
        ClaimType.BACKGROUND_VS_EVENT: (
            "Elevated PM2.5 across all monitors with similar magnitude "
            "suggests regional transport, not a local source"
        ),
    }
    for expected, example in memo_examples.items():
        assert classify_claim(example)[0] is expected, example


# --- dispatch ---


def _summary_with(metrics_by_source: dict) -> dict:
    return {
        "schema_version": 1,
        "sources": {
            src: {"metrics": metrics}
            for src, metrics in metrics_by_source.items()
        },
    }


def test_score_claim_dispatches_to_concentration_scorer():
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
    scored = score_claim(
        "Ground-level NO2 exceeded 80 ppb in the afternoon.", summary
    )
    assert scored.claim_type is ClaimType.CONCENTRATION_ELEVATION
    assert scored.result.corroboration_score == 1.0
    assert scored.result.evidence_n == 1
    assert scored.qualitative_only is False
    assert scored.partial_verifiability is False


def test_score_claim_unclassified_is_all_silent():
    scored = score_claim("The cause of this anomaly is uncertain.", _summary_with({}))
    assert scored.claim_type is ClaimType.UNCLASSIFIED
    assert scored.result.unverified is True
    assert scored.result.evidence_n == 0
    assert scored.result.corroboration_score is None


def test_score_claim_marks_chemistry_qualitative_only():
    scored = score_claim(
        "Elevated HCHO and depressed NO2 point to a VOC-limited regime.",
        _summary_with({}),
    )
    assert scored.claim_type is ClaimType.CHEMISTRY
    assert scored.qualitative_only is True
    assert scored.partial_verifiability is True


def test_score_claim_keeps_full_match_list():
    scored = score_claim(
        "PM2.5 was elevated across all stations in the region.", _summary_with({})
    )
    assert scored.claim_type is ClaimType.BACKGROUND_VS_EVENT
    assert ClaimType.CONCENTRATION_ELEVATION in scored.matched_types
