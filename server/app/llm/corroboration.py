"""Phase 2 — cross-source corroboration scorer.

For each Phase-1-grounded claim about an atmospheric anomaly, score it against
the agreement of the data sources, grouped into measurement-process channels
(ground in-situ, ground optical, satellite column, NWP, met in-situ). Sources
that share a measurement process collapse to one group, so raw source count
does not inflate the score. The groups are not assumed statistically
independent: NWP analyses can assimilate observations reported by the direct
meteorology channel. Design + claim taxonomy:
docs/specs/2026-05-21-corroboration-scorer-design.md; channel grouping:
docs/specs/2026-06-24-channel-independence-collectors.md.

The module provides the shared aggregator that collapses per-source verdicts
into the scalar ``corroboration_score`` + ``evidence_n``, plus one scorer per
claim type: 3 headline types (1-3) and 7 descriptive types (4-10, of which
``chemistry`` and ``point_source_attribution`` are qualitative-only per the
memo's 2026-06-10 addendum).
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from statistics import fmean, pstdev
from typing import cast

from app.llm.observation_age import assess_observation_age
from app.llm.validate import strip_locators, threshold_cues, within_tolerance
from app.provenance.openaq_pm25 import (
    NOMINATING_METRICS as OPENAQ_REGULATORY_METRICS,
    verified_monitor_entity_ids,
)
from app.provenance.purpleair_qc import purpleair_reading_is_eligible

logger = logging.getLogger(__name__)

# Per-source verdict on a single claim. A source either supports the claim,
# contradicts it, or is silent (no data bearing on it within the window).
SUPPORTING = 1
CONTRADICTING = -1
SILENT = 0


def _fresh_nearest_value(
    source: str,
    aspect: str,
    block: Mapping | None,
) -> tuple[float | None, str | None]:
    """Return a nearest value only when its declared B8 age gate passes."""
    if not block:
        return None, None
    nearest = block.get("nearest_in_time")
    if not isinstance(nearest, Mapping) or nearest.get("v") is None:
        return None, None
    raw_dt_minutes = nearest.get("dt_minutes")
    decision = assess_observation_age(source, raw_dt_minutes)
    if decision.votes:
        return cast(float, nearest["v"]), None
    note = (
        f"{source}: {aspect} age-gated SILENT "
        f"(dt_minutes={raw_dt_minutes!r}, "
        f"gate_minutes={decision.gate_minutes}, reason={decision.reason})"
    )
    return None, note

# Each source's measurement channel. Sources that share a measurement process
# collapse to one channel, so corroboration counts measurement-process groups,
# not raw sources. TCEQ and EPA AQS share regulatory monitor sites;
# GFS/OpenWeather are both NWP-derived. OpenAQ PM2.5 enters the ground channel
# only after the B6 entity-provenance filter retains verified AirNow government
# monitors; Clarity/AirGradient and unmappable archive entities are excluded.
# The groups are not statistically independent: GFS analyses assimilate
# ASOS/METAR observations, and any weighting claim needs residual-error
# measurements that have not yet been completed. An unlisted source gets its
# own channel.
SOURCE_CHANNELS: dict[str, str] = {
    "openaq": "ground_insitu",       # PM2.5 is entity-filtered before voting
    "tceq": "ground_insitu",         # same regulatory monitors (preliminary feed)
    "epa_aqs": "ground_insitu",      # historical AQS monitor samples
    "purpleair": "ground_optical",   # low-cost optical PM — different instrument physics
    "sentinel5p": "satellite_column",
    "noaa_gfs": "nwp",               # numerical weather prediction
    "openweather": "nwp",            # blended NWP product (shares model heritage)
    "asos": "met_insitu",            # raw anemometer / thermometer
}


def channel_of(source: str) -> str:
    """The measurement-process channel for ``source`` (own name if unlisted)."""
    return SOURCE_CHANNELS.get(source, source)

# low_corroboration_flag threshold (memo: metadata signal, not a scoring gate).
_LOW_CORROBORATION_SCORE = -0.5
_LOW_CORROBORATION_MIN_EVIDENCE = 2


@dataclass(frozen=True)
class CorroborationResult:
    """Aggregated cross-source verdict for one claim.

    ``corroboration_score`` is ``None`` (never 0) when every source is silent,
    so "no evidence" is not conflated with "balanced evidence" downstream.
    ``evidence_n`` travels with the score because (score=+1, n=1) and
    (score=+1, n=3) carry different evidential weight.
    """

    corroboration_score: float | None
    evidence_n: int
    supporting: int
    contradicting: int
    unverified: bool
    per_source_verdicts: dict[str, int]
    per_channel_verdicts: dict[str, int] = field(default_factory=dict)


def aggregate_verdicts(per_source_verdicts: Mapping[str, int]) -> CorroborationResult:
    """Collapse per-source verdicts into a channel-aware score and evidence count.

    Sources are first grouped into measurement-process channels
    (:data:`SOURCE_CHANNELS`) and each channel takes the net sign of its members'
    verdicts, so redundant sources (TCEQ+AQS, GFS+OpenWeather) count once and a
    within-channel disagreement nets to silent. ``evidence_n`` is then the number
    of channels carrying a verdict, not raw source
    count — and ``score = (supporting - contradicting) / evidence_n`` in [-1, +1]
    (``None`` when every channel is silent). Single-source and distinct-channel
    claims are unchanged from the old per-source behaviour.
    """
    channel_sums: dict[str, int] = {}
    for source, verdict in per_source_verdicts.items():
        channel = channel_of(source)
        channel_sums[channel] = channel_sums.get(channel, 0) + verdict
    per_channel = {
        channel: (
            SUPPORTING if total > 0 else CONTRADICTING if total < 0 else SILENT
        )
        for channel, total in channel_sums.items()
    }

    supporting = sum(1 for v in per_channel.values() if v == SUPPORTING)
    contradicting = sum(1 for v in per_channel.values() if v == CONTRADICTING)
    evidence_n = supporting + contradicting
    verdicts = dict(per_source_verdicts)

    if evidence_n == 0:
        return CorroborationResult(
            corroboration_score=None,
            evidence_n=0,
            supporting=0,
            contradicting=0,
            unverified=True,
            per_source_verdicts=verdicts,
            per_channel_verdicts=per_channel,
        )

    return CorroborationResult(
        corroboration_score=(supporting - contradicting) / evidence_n,
        evidence_n=evidence_n,
        supporting=supporting,
        contradicting=contradicting,
        unverified=False,
        per_source_verdicts=verdicts,
        per_channel_verdicts=per_channel,
    )


def low_corroboration_flag(score: float | None, *, evidence_n: int) -> bool:
    """Phase 2 metadata flag: strongly contradicted across >= 2 channels.

    Not a gate — the raw ``corroboration_score`` is what the research analysis
    correlates against expert labels; this is a convenience signal for
    downstream product code.
    """
    if score is None:
        return False
    return (
        score <= _LOW_CORROBORATION_SCORE
        and evidence_n >= _LOW_CORROBORATION_MIN_EVIDENCE
    )


# ---------------------------------------------------------------------------
# Headline claim type 1 — concentration_elevation (OpenAQ + Sentinel-5P)
# ---------------------------------------------------------------------------

# Claim pollutant aliases -> OpenAQ metric name. Word-boundary matched so short
# codes ("co", "bc") don't fire inside other words ("could", "across").
_POLLUTANT_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\bnitrogen dioxide\b", "no2"),
    (r"\bno2\b", "no2"),
    (r"\bozone\b", "ozone"),
    (r"\bo3\b", "ozone"),
    (r"\bpm2\.?5\b", "pm25"),
    (r"\bpm10\b", "pm10"),
    (r"\bsulfur dioxide\b", "so2"),
    (r"\bsulphur dioxide\b", "so2"),
    (r"\bso2\b", "so2"),
    (r"\bcarbon monoxide\b", "co"),
    (r"\bco\b", "co"),
    (r"\bblack carbon\b", "bc"),
    (r"\bbc\b", "bc"),
)

# OpenAQ species that also have a TROPOMI column product to cross-check against.
_SENTINEL_COLUMN: dict[str, str] = {
    "no2": "s5p_no2_column",
    "so2": "s5p_so2_column",
    "co": "s5p_co_column",
}

# Threshold cue detection ("exceeded 80" vs "was 80") is shared with the
# Phase 1 grounding check (validate.threshold_cues) so both phases read
# threshold claims with the same >= semantics.


def _blank_match(match: re.Match) -> str:
    return " " * len(match.group())


@dataclass(frozen=True)
class ConcentrationTolerance:
    """Draft tolerances for concentration_elevation (pending Dr. Bracco)."""

    # Numeric point claim ("was 80 ppb"): within this fraction of the
    # measured nearest-in-time value (memo: ±25% of measured value).
    value_pct: float = 0.25
    # Qualitative "elevated": the value nearest the anomaly must exceed the
    # pre-anomaly baseline mean by this many baseline standard deviations.
    # A bare mean-exceedance criterion (the old ratio 1.0) is a coin flip
    # under noise — any value a hair above the mean "supported". Requiring
    # mean + k*sigma makes support an exceedance call scaled to the series'
    # own variability; values between the mean and the sigma band are SILENT
    # (can't call it either way), mirroring the chemistry noise buffer.
    # Draft, pending Dr. Bracco.
    elevated_sigma: float = 1.0
    # The baseline needs this many points, all ending this many hours before
    # the anomaly. The spike must not sit inside its own baseline — against
    # the in-window mean, any restatement of the detection event would be
    # near-automatically corroborated by the source that triggered it.
    min_baseline_points: int = 3
    baseline_gap_h: float = 3.0
    # S5P SO2 columns below the TROPOMI detection limit (~1 DU) are retrieval
    # noise that scatters about zero, not a real concentration. A column below
    # this is scored SILENT, not as a verdict, so noise cannot support or
    # contradict an SO2 claim. 1 DU ~= 4.46e-4 mol/m^2. Draft, like every other
    # tolerance here (pending Dr. Bracco).
    so2_detection_limit_mol_m2: float = 4.46e-4
    # Ground SO2 (TCEQ/EPA AQS, ppb) sits in an even wider noise band than the
    # satellite column: in-window, 54-62% of ground SO2 reads negative — the
    # UV-fluorescence monitors scatter about zero below their ~0.5 ppb hourly
    # method detection limit. A ground SO2 reading below this floor (and any
    # non-physical negative ground concentration, for every species) is scored
    # SILENT, the ground analogue of the satellite gate above, so detection-limit
    # noise neither votes nor poisons the pre-anomaly baseline. Draft, pending
    # Dr. Bracco.
    so2_ground_detection_limit_ppb: float = 0.5


DEFAULT_CONCENTRATION_TOLERANCE = ConcentrationTolerance()


class BaselineCensoringStrategy(str, Enum):
    """B15 primary and sensitivity treatments for censored baselines."""

    LIMIT_HALF = "limit_half"
    DELETE = "delete"


def baseline_censor_limit(
    source: str,
    metric: str,
    *,
    tolerance: ConcentrationTolerance = DEFAULT_CONCENTRATION_TOLERANCE,
) -> float | None:
    """Declared censoring limit, or None when B15 declares no treatment."""
    if source == "sentinel5p" and metric == "s5p_so2_column":
        return tolerance.so2_detection_limit_mol_m2
    if channel_of(source) == "ground_insitu" or source == "purpleair":
        return tolerance.so2_ground_detection_limit_ppb if metric == "so2" else 0.0
    return None


def censor_baseline_values(
    values: list[float],
    *,
    limit: float | None,
    strategy: BaselineCensoringStrategy,
) -> list[float]:
    """Apply B15 to baseline values without mutating stored observations."""
    if limit is None:
        return list(values)
    if not math.isfinite(limit) or limit < 0.0:
        raise ValueError("baseline censoring limit must be finite and nonnegative")
    if strategy is BaselineCensoringStrategy.LIMIT_HALF:
        replacement = limit / 2.0
        return [value if value >= limit else replacement for value in values]
    if strategy is BaselineCensoringStrategy.DELETE:
        return [value for value in values if value >= limit]
    raise ValueError(f"unsupported baseline censoring strategy: {strategy!r}")


def qualitative_elevation_verdict(
    nearest: float,
    baseline_values: list[float],
    *,
    tolerance: ConcentrationTolerance = DEFAULT_CONCENTRATION_TOLERANCE,
) -> int | None:
    """Score the qualitative concentration band shared by B15 sensitivity."""
    if not baseline_values:
        return None
    baseline = fmean(baseline_values)
    spread = pstdev(baseline_values)
    support_floor = baseline + tolerance.elevated_sigma * spread
    if nearest > support_floor:
        return SUPPORTING
    if nearest <= baseline:
        return CONTRADICTING
    return SILENT


def _anomaly_ts(summary: Mapping) -> datetime | None:
    """The anomaly timestamp from the enrichment summary, UTC-coerced."""
    raw = (summary.get("anomaly") or {}).get("timestamp")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _station_pre_anomaly_values(
    block: Mapping | None,
    anomaly_ts: datetime | None,
    *,
    entity_id: str | None,
    gap_h: float,
    min_points: int,
    censor_limit: float | None = None,
    censoring_strategy: BaselineCensoringStrategy = (
        BaselineCensoringStrategy.LIMIT_HALF
    ),
) -> tuple[list[float], int, str | None]:
    """Return the B17 nearest-event-entity baseline and its eligibility state.

    Censored values receive the declared B15 treatment before the observation
    floor is checked. No other entity may rescue a missing or insufficient
    matched series.
    """
    if anomaly_ts is None:
        return [], 0, "missing_anomaly_timestamp"
    if not isinstance(entity_id, str) or not entity_id:
        return [], 0, "missing_nearest_entity_id"
    if not block:
        return [], 0, "missing_metric_block"
    raw_entities = block.get("entities")
    if not isinstance(raw_entities, (list, tuple)):
        return [], 0, "malformed_entity_collection"
    matching = [
        entity
        for entity in raw_entities
        if isinstance(entity, Mapping) and entity.get("entity_id") == entity_id
    ]
    if not matching:
        return [], 0, "nearest_entity_not_found"
    if len(matching) != 1:
        return [], 0, "duplicate_nearest_entity"

    series = matching[0].get("series")
    if not isinstance(series, (list, tuple)):
        return [], 0, "malformed_station_series"
    cutoff = anomaly_ts.astimezone(timezone.utc) - timedelta(hours=gap_h)
    raw_values: list[float] = []
    for row in series:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            return [], 0, "malformed_station_series"
        try:
            timestamp = datetime.fromisoformat(
                str(row[0]).replace("Z", "+00:00")
            )
            value = float(row[1])
        except (TypeError, ValueError):
            return [], 0, "malformed_station_series"
        if not math.isfinite(value):
            return [], 0, "malformed_station_series"
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        if timestamp <= cutoff:
            raw_values.append(value)

    values = censor_baseline_values(
        raw_values,
        limit=censor_limit,
        strategy=censoring_strategy,
    )
    baseline_n = len(values)
    if baseline_n < min_points:
        return [], baseline_n, f"matched baseline n < {min_points}"
    return values, baseline_n, None


def _resolve_pollutant(claim_text: str) -> tuple[str | None, str | None]:
    """(OpenAQ metric, Sentinel column metric) named in the claim, if any."""
    lowered = claim_text.lower()
    for pattern, metric in _POLLUTANT_PATTERNS:
        if re.search(pattern, lowered):
            return metric, _SENTINEL_COLUMN.get(metric)
    return None, None


def _threshold_value(claim_text: str) -> tuple[float, str] | None:
    """``(value, "over" | "under")`` for a threshold claim, else None.

    "exceeded 80" yields ``(80.0, "over")`` and "stayed below 5" yields
    ``(5.0, "under")`` — the mirror Phase 1 grounding already handles, so the
    two phases read threshold claims the same way. Clock times and dates are
    blanked first (the Phase 1 locator rule), and pollutant tokens are blanked
    position-preserving, so "exceeded typical values at 14:00" yields no
    threshold while "NO2 exceeded 80 ppb between 14:00-18:00" yields 80. The
    number must sit at/after a cue word — numbers before it describe something
    else — and the nearest cue preceding the number sets the direction (mirrors
    ``validate._threshold_relation``).
    """
    lowered = strip_locators(claim_text.lower())
    cues = threshold_cues(lowered)
    if not cues:
        return None
    cleaned = lowered
    for pattern, _metric in _POLLUTANT_PATTERNS:
        cleaned = re.sub(pattern, _blank_match, cleaned)
    match = re.search(r"\d+(?:\.\d+)?", cleaned[cues[0][0]:])
    if not match:
        return None
    position = cues[0][0] + match.start()
    relation = cues[0][1]
    for offset, kind in cues:
        if offset > position:
            break
        relation = kind
    return float(match.group()), relation


def _point_value(claim_text: str) -> float | None:
    """The numeric value in a 'was N' style point claim, else None.

    Threshold-worded claims are handled by ``_threshold_value``; this covers
    the memo's "within 25% of the measured value" shape.
    """
    lowered = strip_locators(claim_text.lower())
    if threshold_cues(lowered):
        return None
    cleaned = lowered
    for pattern, _metric in _POLLUTANT_PATTERNS:
        cleaned = re.sub(pattern, _blank_match, cleaned)
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def concentration_claim_shape(claim_text: str) -> str:
    """Return the scorer's deterministic concentration-claim shape.

    B19 uses this public classification helper to separate B17 qualitative
    baseline abstentions from numeric/threshold claims without re-scoring a
    claim or inferring its shape from a verdict integer.
    """
    if _threshold_value(claim_text) is not None:
        return "threshold"
    if _point_value(claim_text) is not None:
        return "point"
    return "qualitative"


# A claim about a satellite column density, not a surface concentration.
# Absolute (threshold / point) claims are only comparable to a source whose
# stored quantity matches the claim's: "exceeded 80 ppb" against a column in
# mol/m^2 (~1e-4) is a guaranteed spurious contradiction, and "column exceeded
# 5e-4 mol/m2" against a ppb surface reading is a guaranteed spurious support.
_COLUMN_CLAIM_RE = re.compile(r"\bcolumn\b|mol/m")


def _is_column_claim(claim_text: str) -> bool:
    return _COLUMN_CLAIM_RE.search(claim_text.lower()) is not None


def score_concentration_elevation(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: ConcentrationTolerance = DEFAULT_CONCENTRATION_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a 'pollutant was elevated' claim against the ground + satellite sources.

    Three claim shapes (memo type 1):
    - threshold ("exceeded 80"): the value nearest the anomaly meets the
      threshold;
    - point value ("was 80 ppb"): nearest value within ``value_pct``;
    - qualitative ("elevated"): nearest value above the pre-anomaly baseline
      by ``elevated_sigma`` baseline standard deviations, silent when too few
      pre-anomaly points exist to call one.

    Absolute (threshold/point) claims are unit-scoped: surface-worded claims
    are judged by surface sources only and column-worded claims by the
    satellite column only (``_is_column_claim``) — cross-quantity comparisons
    produce verdicts by unit accident, not measurement. The qualitative shape
    is direction-only against each source's own baseline, so every source
    participates regardless of units.

    The anomaly's own triggering channel is handled asymmetrically: its
    SUPPORTING vote on a claim about the anomaly metric is demoted to SILENT
    (detection selected on that source being elevated, so support is
    tautological), while a CONTRADICTING vote is kept (the model misstating
    the very data that triggered detection is the most informative negative
    signal Phase 2 has).

    v1 otherwise assumes the claim's unit matches the stored metric's unit (no
    ppb<->ug/m^3 conversion). Returns ``(per_source_verdicts, evidence_summary)``.
    """
    openaq_metric, sentinel_metric = _resolve_pollutant(claim_text)
    shape = concentration_claim_shape(claim_text)
    threshold = _threshold_value(claim_text) if shape == "threshold" else None
    point = _point_value(claim_text) if shape == "point" else None
    anomaly_ts = _anomaly_ts(summary)
    column_claim = _is_column_claim(claim_text)
    anomaly_info = summary.get("anomaly") or {}
    trigger_metric = anomaly_info.get("metric")
    trigger_source = anomaly_info.get("source")
    trigger_channel = channel_of(trigger_source) if trigger_source else None
    # Ground in-situ sources (OpenAQ regulatory + TCEQ/EPA AQS — one channel) all
    # report the claimed species; PurpleAir adds an optical PM2.5 channel; S5P the
    # satellite column. Sources without the metric in window resolve to SILENT
    # below, and the channel-aware aggregator collapses the redundant ground ones,
    # so this is where TCEQ fills the in-window NO2/SO2/CO gap the audit found.
    relevant: dict[str, str] = {}
    if openaq_metric is not None:
        for ground_source in ("openaq", "tceq", "epa_aqs"):
            relevant[ground_source] = openaq_metric
        if openaq_metric == "pm25":
            # PurpleAir PM2.5 is stored raw (uncorrected): the optical Plantower
            # reads ~4-5x regulatory mass in humid Houston (hygroscopic growth).
            # For qualitative elevation the multiplicative bias largely cancels
            # against its own pre-anomaly baseline; for absolute threshold/point
            # claims it can over-support — a known limitation pending the
            # RH-resolved EPA Barkjohn correction (0.524*PA - 0.0862*RH + 5.75),
            # which needs co-located humidity not yet collected and lands
            # post-freeze as the optical-channel upgrade (audit 2026-06-24).
            relevant["purpleair"] = "pm25"
    if sentinel_metric is not None:
        relevant["sentinel5p"] = sentinel_metric

    verdicts: dict[str, int] = {}
    notes: list[str] = []

    for source, metric in relevant.items():
        if metric is None:
            continue
        raw_data = (
            summary.get("sources", {})
            .get(source, {})
            .get("metrics", {})
            .get(metric)
        )
        data = _metric_block(summary, source, metric)
        if not data or data.get("nearest_in_time", {}).get("v") is None:
            verdicts[source] = SILENT
            if (
                source == "openaq"
                and metric in OPENAQ_REGULATORY_METRICS
                and raw_data
            ):
                notes.append(
                    f"openaq: no verified-monitor {metric} observation in window"
                )
            else:
                notes.append(f"{source}: no {metric} in window")
            continue

        nearest, age_note = _fresh_nearest_value(source, metric, data)
        if age_note is not None:
            verdicts[source] = SILENT
            notes.append(age_note)
            continue
        if nearest is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: no {metric} event value")
            continue
        censor_limit = baseline_censor_limit(
            source,
            metric,
            tolerance=tolerance,
        )
        if (
            metric == "s5p_so2_column"
            and nearest < tolerance.so2_detection_limit_mol_m2
        ):
            verdicts[source] = SILENT
            notes.append(
                f"{source}: {metric} nearest={nearest} below detection "
                f"limit {tolerance.so2_detection_limit_mol_m2}"
            )
            continue
        # Ground in-situ concentrations that are non-physical (<0) or, for SO2,
        # below the monitor's ~0.5 ppb detection floor are instrument noise that
        # scatters about zero, not a measurement. Gate them SILENT — the ground
        # analogue of the satellite SO2 gate above — so they neither cast a
        # verdict nor poison the pre-anomaly baseline below.
        ground_floor: float | None = None
        if channel_of(source) == "ground_insitu":
            ground_floor = (
                tolerance.so2_ground_detection_limit_ppb if metric == "so2" else 0.0
            )
            if nearest < ground_floor:
                verdicts[source] = SILENT
                notes.append(
                    f"{source}: {metric} nearest={nearest} below ground "
                    f"detection floor {ground_floor}"
                )
                continue
        if threshold is not None or point is not None:
            # Unit scoping: absolute claims only compare like quantities.
            if column_claim and source != "sentinel5p":
                verdicts[source] = SILENT
                notes.append(
                    f"{source}: column-worded absolute claim not comparable "
                    f"to surface {metric}"
                )
                continue
            if not column_claim and source == "sentinel5p":
                verdicts[source] = SILENT
                notes.append(
                    f"{source}: surface-worded absolute claim not comparable "
                    f"to {metric} column"
                )
                continue
        if threshold is not None:
            limit, kind = threshold
            if kind == "under":
                verdict = SUPPORTING if nearest <= limit else CONTRADICTING
            else:
                verdict = SUPPORTING if nearest >= limit else CONTRADICTING
            notes.append(
                f"{source}: {metric} nearest={nearest} vs {kind}-threshold={limit}"
            )
        elif point is not None:
            within = within_tolerance(point, nearest, tolerance.value_pct)
            verdict = SUPPORTING if within else CONTRADICTING
            notes.append(
                f"{source}: {metric} nearest={nearest} vs claimed={point}"
            )
        else:
            nearest_entity_id = data.get("nearest_in_time", {}).get("entity_id")
            baseline_values, baseline_n, baseline_reason = (
                _station_pre_anomaly_values(
                    data,
                    anomaly_ts,
                    entity_id=nearest_entity_id,
                    gap_h=tolerance.baseline_gap_h,
                    min_points=tolerance.min_baseline_points,
                    censor_limit=censor_limit,
                )
            )
            if not baseline_values:
                verdicts[source] = SILENT
                notes.append(
                    f"{source}: {metric} no station-matched pre-anomaly "
                    f"baseline (entity_id={nearest_entity_id}; "
                    f"baseline_n={baseline_n}; reason={baseline_reason})"
                )
                continue
            baseline = fmean(baseline_values)
            spread = pstdev(baseline_values)
            verdict = qualitative_elevation_verdict(
                nearest,
                baseline_values,
                tolerance=tolerance,
            )
            if verdict == SILENT:
                # Above the mean but inside the sigma band: too close to call.
                verdicts[source] = SILENT
                notes.append(
                    f"{source}: {metric} nearest={nearest} within noise band "
                    f"of pre-anomaly baseline={round(baseline, 4)} "
                    f"(+{tolerance.elevated_sigma} sigma={round(spread, 4)}; "
                    f"entity_id={nearest_entity_id}; baseline_n={baseline_n})"
                )
                continue
            notes.append(
                f"{source}: {metric} nearest={nearest} "
                f"vs pre-anomaly baseline={round(baseline, 4)} "
                f"(+{tolerance.elevated_sigma} sigma={round(spread, 4)}; "
                f"entity_id={nearest_entity_id}; baseline_n={baseline_n})"
            )
        # Trigger-channel asymmetry: detection already selected on the trigger
        # channel reading elevated, so its support is tautological (demoted to
        # SILENT); its contradiction is not (kept).
        if (
            verdict == SUPPORTING
            and trigger_channel is not None
            and metric == trigger_metric
            and channel_of(source) == trigger_channel
        ):
            verdicts[source] = SILENT
            notes.append(
                f"{source}: trigger-channel support demoted to silent "
                "(circular with detection)"
            )
            continue
        verdicts[source] = verdict

    if openaq_metric is None and sentinel_metric is None:
        notes.append("no recognized pollutant in claim")
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Headline claim types 2 & 3 — transport_direction + meteorological_state
# (NOAA GFS 10 m wind, OpenWeather wind / temperature)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindTolerance:
    """Tolerances for the wind / met headline types.

    The calm-wind floor (``calm_floor_ms``) is Bracco-confirmed (2026-07-24).
    The bearing/speed/stagnant/temperature tolerances remain Mason-owned
    drafts that were never reviewed on the July 15 call.
    """

    bearing_deg: float = 45.0  # transport direction within this of measured wind
    speed_ms: float = 1.5      # numeric wind-speed claim within this of measured
    stagnant_ms: float = 2.0   # qualitative "stagnant" means wind below this
    temp_c: float = 2.0        # temperature claim within this of measured
    calm_sigma: float = 2.0
    calm_min_points: int = 2
    calm_floor_ms: float | None = 1.5


DEFAULT_WIND_TOLERANCE = WindTolerance()

# Bracco confirmed the 1.5 m/s floor in writing on 2026-07-24 ("yes for 3");
# see docs/bracco/2026-07-24-packet-reply.md and the protocol-lock addendum.
CALM_WIND_FLOOR_STATUS = "bracco_confirmed"
CALM_WIND_SHIP_STATUS = "shipped_bracco_confirmed"
CALM_WIND_CONFIRMATION_DATE = "2026-07-24"


@dataclass(frozen=True)
class CalmWindDecision:
    """One source-local B2 cutoff and event-speed decision."""

    source: str
    window_n: int
    event_speed_ms: float | None
    raw_cutoff_ms: float | None
    effective_cutoff_ms: float | None
    guard_enabled: bool
    calm: bool | None
    direction_votable: bool
    reason: str
    floor_status: str

    def to_dict(self) -> dict[str, str | int | float | bool | None]:
        return {
            "source": self.source,
            "window_n": self.window_n,
            "event_speed_ms": self.event_speed_ms,
            "raw_cutoff_ms": self.raw_cutoff_ms,
            "effective_cutoff_ms": self.effective_cutoff_ms,
            "guard_enabled": self.guard_enabled,
            "calm": self.calm,
            "direction_votable": self.direction_votable,
            "reason": self.reason,
            "floor_status": self.floor_status,
        }

    def evidence_note(self) -> str:
        fields = (
            f"event_speed={self.event_speed_ms} raw_cutoff={self.raw_cutoff_ms} "
            f"effective_cutoff={self.effective_cutoff_ms} n={self.window_n} "
            f"floor_status={self.floor_status}"
        )
        if self.calm:
            return (
                f"{self.source}: calm-wind guard SILENT ({fields}); "
                "wind direction unstable under calm conditions"
            )
        if self.reason == "raw_cutoff_nonpositive_guard_disabled":
            return f"{self.source}: calm-wind guard disabled LOUDLY ({fields})"
        if not self.direction_votable:
            return (
                f"{self.source}: calm-wind guard unevaluable SILENT "
                f"(reason={self.reason}; {fields})"
            )
        return f"{self.source}: calm-wind guard passed ({fields})"


def calm_wind_decision(
    source: str,
    window_speeds: Sequence[float],
    event_speed: float | None,
    *,
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> CalmWindDecision:
    """Evaluate the declared source-local B2 guard."""
    floor = tolerance.calm_floor_ms
    if floor is not None and (not math.isfinite(floor) or floor < 0.0):
        raise ValueError("calm-wind floor must be finite and non-negative")
    floor_status = CALM_WIND_FLOOR_STATUS if floor is not None else "not_configured"
    try:
        speeds = [float(value) for value in window_speeds]
    except (TypeError, ValueError):
        speeds = [math.nan]
    if any(not math.isfinite(value) or value < 0.0 for value in speeds):
        return CalmWindDecision(
            source, len(speeds), None, None, None, False, None, False,
            "invalid_window_speed", floor_status,
        )
    if len(speeds) < tolerance.calm_min_points:
        return CalmWindDecision(
            source, len(speeds), None, None, None, False, None, False,
            "insufficient_window", floor_status,
        )

    raw_cutoff = fmean(speeds) - tolerance.calm_sigma * pstdev(speeds)
    if floor is None and raw_cutoff <= 0.0:
        logger.warning(
            "%s calm-wind guard disabled: raw cutoff %.6g m/s is nonpositive "
            "and no floor is configured",
            source,
            raw_cutoff,
        )
        return CalmWindDecision(
            source, len(speeds), None, raw_cutoff, None, False, None, True,
            "raw_cutoff_nonpositive_guard_disabled", floor_status,
        )
    effective_cutoff = max(raw_cutoff, floor) if floor is not None else raw_cutoff
    try:
        speed = float(event_speed) if event_speed is not None else None
    except (TypeError, ValueError):
        speed = None
    if speed is None or not math.isfinite(speed) or speed < 0.0:
        return CalmWindDecision(
            source, len(speeds), None, raw_cutoff, effective_cutoff, True,
            None, False, "missing_event_speed", floor_status,
        )
    calm = speed < effective_cutoff
    return CalmWindDecision(
        source=source,
        window_n=len(speeds),
        event_speed_ms=speed,
        raw_cutoff_ms=raw_cutoff,
        effective_cutoff_ms=effective_cutoff,
        guard_enabled=True,
        calm=calm,
        direction_votable=not calm,
        reason="calm" if calm else "at_or_above_cutoff",
        floor_status=floor_status,
    )


def calm_wind_manifest_payload(
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> dict[str, str | int | float | bool | None]:
    """Freeze-manifest status for the proposed B2 amendment."""
    return {
        "formula": "mean(speed) - calm_sigma*pstdev(speed)",
        "calm_sigma": tolerance.calm_sigma,
        "minimum_window_points": tolerance.calm_min_points,
        "floor_ms": tolerance.calm_floor_ms,
        "floor_status": (
            CALM_WIND_FLOOR_STATUS
            if tolerance.calm_floor_ms is not None
            else "not_configured"
        ),
        "bracco_amendment_confirmed": True,
        "bracco_confirmation_date": CALM_WIND_CONFIRMATION_DATE,
        "ship_status": CALM_WIND_SHIP_STATUS,
        "raw_nonpositive_without_floor": "disabled_loudly",
        "event_comparison": "strictly_below_effective_cutoff_is_calm",
    }

_CARDINALS: dict[str, float] = {
    "north": 0.0,
    "northeast": 45.0,
    "east": 90.0,
    "southeast": 135.0,
    "south": 180.0,
    "southwest": 225.0,
    "west": 270.0,
    "northwest": 315.0,
}
# Compound directions first so "north" doesn't shadow "northeast".
_DIR_ALT = r"(northeast|northwest|southeast|southwest|north|south|east|west)"
_TRANSPORT_VERBS = (
    "carr", "advect", "push", "transport", "blew", "blown", "drift", "moved", "moving",
)
_STAGNANT_WORDS = (
    "stagnant", "calm", "still air", "barely any air", "light wind", "weak wind",
)


def _wind_from_bearing(u: float, v: float) -> float:
    """Meteorological 'from' bearing (deg, 0=N, 90=E) for wind components.

    ``u`` is the eastward component, ``v`` the northward (GFS 10 m winds). Wind
    blowing toward the north (v>0) comes *from* the south (180).
    """
    return math.degrees(math.atan2(-u, -v)) % 360.0


def _wind_speed(u: float, v: float) -> float:
    return math.hypot(u, v)


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, in [0, 180]."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _claimed_from_bearing(claim_text: str) -> float | None:
    """The wind 'from' bearing a transport claim implies, or None.

    Handles wind-source framing ("southerly winds" -> from the south) and
    transport-target framing ("carried northward" -> wind from the south).
    """
    text = claim_text.lower()
    m = re.search(r"\b" + _DIR_ALT + r"erly\b", text)
    if m:
        return _CARDINALS[m.group(1)]
    m = re.search(r"\b" + _DIR_ALT + r"ward", text)
    if m:
        return (_CARDINALS[m.group(1)] + 180.0) % 360.0
    m = re.search(r"(?:from|out of)\s+the\s+" + _DIR_ALT + r"\b", text)
    if m:
        return _CARDINALS[m.group(1)]
    m = re.search(r"(?:toward|towards|to the)\s+(?:the\s+)?" + _DIR_ALT + r"\b", text)
    if m:
        return (_CARDINALS[m.group(1)] + 180.0) % 360.0
    if any(verb in text for verb in _TRANSPORT_VERBS):
        m = re.search(r"\b" + _DIR_ALT + r"\b", text)
        if m:
            return (_CARDINALS[m.group(1)] + 180.0) % 360.0
    m = re.search(r"\b" + _DIR_ALT + r"\s+winds?\b", text)
    if m:
        return _CARDINALS[m.group(1)]
    return None


def _gfs_wind_components(
    summary: Mapping,
) -> tuple[float | None, float | None, tuple[str, ...]]:
    gfs = summary.get("sources", {}).get("noaa_gfs", {}).get("metrics", {})
    u_block = gfs.get("u_10m")
    v_block = gfs.get("v_10m")
    raw_u, u_note = _fresh_nearest_value("noaa_gfs", "u_10m", u_block)
    raw_v, v_note = _fresh_nearest_value("noaa_gfs", "v_10m", v_block)
    notes = [note for note in (u_note, v_note) if note is not None]
    if raw_u is None or raw_v is None:
        return None, None, tuple(notes)

    def nearest_timestamp(
        block: Mapping | None,
    ) -> tuple[datetime | None, str]:
        nearest = block.get("nearest_in_time") if block else None
        raw_timestamp = nearest.get("t") if isinstance(nearest, Mapping) else None
        if raw_timestamp is None:
            return None, "missing"
        try:
            timestamp = datetime.fromisoformat(
                str(raw_timestamp).replace("Z", "+00:00")
            )
        except ValueError:
            return None, "malformed"
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp = timestamp.astimezone(timezone.utc)
        return timestamp, "valid"

    u_timestamp, u_status = nearest_timestamp(u_block)
    v_timestamp, v_status = nearest_timestamp(v_block)
    if u_timestamp is None or v_timestamp is None:
        reason = "missing_timestamp" if "missing" in {u_status, v_status} else "malformed_timestamp"
        notes.append(
            "noaa_gfs: u/v timestamp pairing SILENT "
            f"(reason={reason}; u_status={u_status}; v_status={v_status})"
        )
        return None, None, tuple(notes)
    if u_timestamp != v_timestamp:
        notes.append(
            "noaa_gfs: u/v timestamp pairing SILENT "
            f"(reason=mismatch; u_t={u_timestamp.isoformat()}; "
            f"v_t={v_timestamp.isoformat()})"
        )
        return None, None, tuple(notes)

    u = float(raw_u) if raw_u is not None else None
    v = float(raw_v) if raw_v is not None else None
    return u, v, tuple(notes)


def _window_speed_values(block: Mapping | None) -> list[float]:
    if not block:
        return []
    values: list[float] = []
    for entity in block.get("entities", []):
        for row in entity.get("series", []):
            if not isinstance(row, Sequence) or len(row) != 2:
                values.append(math.nan)
                continue
            try:
                values.append(float(row[1]))
            except (TypeError, ValueError):
                values.append(math.nan)
    return values


def _component_index(
    block: Mapping | None,
) -> tuple[dict[tuple[str, datetime], float], bool]:
    indexed: dict[tuple[str, datetime], float] = {}
    invalid = False
    if not block:
        return indexed, invalid
    for entity in block.get("entities", []):
        raw_entity_id = entity.get("entity_id")
        if raw_entity_id is None:
            invalid = True
            continue
        entity_id = str(raw_entity_id)
        for row in entity.get("series", []):
            if not isinstance(row, Sequence) or len(row) != 2:
                invalid = True
                continue
            try:
                timestamp = datetime.fromisoformat(str(row[0]).replace("Z", "+00:00"))
                value = float(row[1])
            except (TypeError, ValueError):
                invalid = True
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = timestamp.astimezone(timezone.utc)
            if not math.isfinite(value):
                invalid = True
                continue
            key = (entity_id, timestamp)
            if key in indexed:
                invalid = True
                continue
            indexed[key] = value
    return indexed, invalid


def _gfs_window_speeds(summary: Mapping) -> list[float]:
    metrics = summary.get("sources", {}).get("noaa_gfs", {}).get("metrics", {})
    u_values, u_invalid = _component_index(metrics.get("u_10m"))
    v_values, v_invalid = _component_index(metrics.get("v_10m"))
    if u_invalid or v_invalid:
        return [math.nan]
    return [
        math.hypot(u_values[key], v_values[key])
        for key in sorted(set(u_values) & set(v_values))
    ]


def calm_wind_source_decisions(
    summary: Mapping,
    sources: Sequence[str],
    *,
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> tuple[dict[str, CalmWindDecision], tuple[str, ...]]:
    """Build source-local B2 decisions from one stored enrichment summary."""
    decisions: dict[str, CalmWindDecision] = {}
    notes: list[str] = []
    for source in sources:
        if source == "noaa_gfs":
            u, v, age_notes = _gfs_wind_components(summary)
            notes.extend(age_notes)
            event_speed = math.hypot(u, v) if u is not None and v is not None else None
            window_speeds = _gfs_window_speeds(summary)
        elif source in {"asos", "openweather"}:
            speed_block = (
                summary.get("sources", {})
                .get(source, {})
                .get("metrics", {})
                .get("wind_speed")
            )
            raw_speed, age_note = _fresh_nearest_value(
                source, "wind_speed", speed_block
            )
            if age_note is not None:
                notes.append(age_note)
            try:
                event_speed = float(raw_speed) if raw_speed is not None else None
            except (TypeError, ValueError):
                event_speed = None
            window_speeds = _window_speed_values(speed_block)
        else:
            raise ValueError(f"unsupported calm-wind source {source!r}")
        decision = calm_wind_decision(
            source,
            window_speeds,
            event_speed,
            tolerance=tolerance,
        )
        decisions[source] = decision
        notes.append(decision.evidence_note())
    return decisions, tuple(notes)


def score_transport_direction(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a wind-transport claim against GFS 10 m wind + OpenWeather direction."""
    claimed_from = _claimed_from_bearing(claim_text)

    measured: dict[str, float] = {}
    age_notes: list[str] = []
    if claimed_from is not None:
        u, v, gfs_age_notes = _gfs_wind_components(summary)
        age_notes.extend(gfs_age_notes)
        if u is not None and v is not None:
            measured["noaa_gfs"] = _wind_from_bearing(u, v)
    ow = summary.get("sources", {}).get("openweather", {}).get("metrics", {})
    if claimed_from is not None:
        ow_dir, ow_age_note = _fresh_nearest_value(
            "openweather", "wind_direction", ow.get("wind_direction")
        )
        if ow_age_note is not None:
            age_notes.append(ow_age_note)
        if ow_dir is not None:
            measured["openweather"] = float(ow_dir) % 360.0
    # ASOS anemometers are a direct-observation channel, distinct from the NWP
    # products but correlated through data assimilation and blended inputs.
    asos = summary.get("sources", {}).get("asos", {}).get("metrics", {})
    if claimed_from is not None:
        asos_dir, asos_age_note = _fresh_nearest_value(
            "asos", "wind_direction", asos.get("wind_direction")
        )
        if asos_age_note is not None:
            age_notes.append(asos_age_note)
        if asos_dir is not None:
            measured["asos"] = float(asos_dir) % 360.0

    guard_decisions: dict[str, CalmWindDecision] = {}
    guard_notes: tuple[str, ...] = ()
    if claimed_from is not None:
        guard_decisions, guard_notes = calm_wind_source_decisions(
            summary,
            ("noaa_gfs", "openweather", "asos"),
            tolerance=tolerance,
        )

    verdicts: dict[str, int] = {}
    notes: list[str] = [*age_notes, *guard_notes]
    for source in ("noaa_gfs", "openweather", "asos"):
        if source not in measured or claimed_from is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: no comparable wind direction")
            continue
        if not guard_decisions[source].direction_votable:
            verdicts[source] = SILENT
            continue
        diff = _angular_diff(claimed_from, measured[source])
        verdicts[source] = (
            SUPPORTING if diff <= tolerance.bearing_deg else CONTRADICTING
        )
        notes.append(
            f"{source}: claimed_from={claimed_from:.0f} "
            f"measured_from={measured[source]:.0f} diff={diff:.0f}"
        )
    return verdicts, "; ".join(notes)


def _wind_intent(claim_text: str) -> tuple[str, float | None] | None:
    """('value', x) for a numeric speed, ('low', None) for stagnation, else None."""
    text = claim_text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*m/?s", text)
    if m:
        return ("value", float(m.group(1)))
    if any(word in text for word in _STAGNANT_WORDS):
        return ("low", None)
    return None


def _claimed_temperature(claim_text: str) -> float | None:
    """A surface-temperature value in C named in the claim, else None."""
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:°\s*c|degrees?\s*c|deg\s*c|c\b)", claim_text.lower()
    )
    return float(m.group(1)) if m else None


def _combine(*verdicts: int | None) -> int:
    """Per-source roll-up across aspects: contradiction dominates, else support."""
    present = [x for x in verdicts if x is not None]
    if CONTRADICTING in present:
        return CONTRADICTING
    if SUPPORTING in present:
        return SUPPORTING
    return SILENT


def score_meteorological_state(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a 'stagnant / hot' state claim against GFS wind + OpenWeather.

    Wind speed is checked against both GFS (|u, v|) and OpenWeather; temperature
    against OpenWeather only (GFS carries 850 hPa temp, not surface). Qualitative
    "hot"/"cold" without a number is deferred to a later version.
    """
    wind = _wind_intent(claim_text)
    temp = _claimed_temperature(claim_text)

    def wind_verdict(measured: float | None) -> int | None:
        if measured is None or wind is None:
            return None
        if wind[0] == "low":
            return SUPPORTING if measured < tolerance.stagnant_ms else CONTRADICTING
        return (
            SUPPORTING
            if abs(measured - wind[1]) <= tolerance.speed_ms
            else CONTRADICTING
        )

    def temp_verdict(measured: float | None) -> int | None:
        if measured is None or temp is None:
            return None
        return SUPPORTING if abs(measured - temp) <= tolerance.temp_c else CONTRADICTING

    age_notes: list[str] = []
    gfs_speed: float | None = None
    if wind is not None:
        u, v, gfs_age_notes = _gfs_wind_components(summary)
        age_notes.extend(gfs_age_notes)
        if u is not None and v is not None:
            gfs_speed = _wind_speed(u, v)
    ow = summary.get("sources", {}).get("openweather", {}).get("metrics", {})
    ow_speed: float | None = None
    if wind is not None:
        ow_speed, ow_speed_note = _fresh_nearest_value(
            "openweather", "wind_speed", ow.get("wind_speed")
        )
        if ow_speed_note is not None:
            age_notes.append(ow_speed_note)
    ow_temp: float | None = None
    if temp is not None:
        ow_temp, ow_temp_note = _fresh_nearest_value(
            "openweather", "temperature", ow.get("temperature")
        )
        if ow_temp_note is not None:
            age_notes.append(ow_temp_note)
    asos = summary.get("sources", {}).get("asos", {}).get("metrics", {})
    asos_speed: float | None = None
    if wind is not None:
        asos_speed, asos_speed_note = _fresh_nearest_value(
            "asos", "wind_speed", asos.get("wind_speed")
        )
        if asos_speed_note is not None:
            age_notes.append(asos_speed_note)
    asos_temp: float | None = None
    if temp is not None:
        asos_temp, asos_temp_note = _fresh_nearest_value(
            "asos", "temperature", asos.get("temperature")
        )
        if asos_temp_note is not None:
            age_notes.append(asos_temp_note)

    verdicts = {
        "noaa_gfs": _combine(wind_verdict(gfs_speed)),
        "openweather": _combine(wind_verdict(ow_speed), temp_verdict(ow_temp)),
        # ASOS in-situ wind/temp — independent of the NWP channel.
        "asos": _combine(wind_verdict(asos_speed), temp_verdict(asos_temp)),
    }
    notes = [f"wind_intent={wind} temp={temp}", *age_notes]
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Shared helpers for the descriptive claim types (4-10)
# ---------------------------------------------------------------------------


def _metric_block(summary: Mapping, source: str, metric: str) -> Mapping | None:
    block = summary.get("sources", {}).get(source, {}).get("metrics", {}).get(metric)
    if not block:
        return None
    if source == "openaq" and metric in OPENAQ_REGULATORY_METRICS:
        return _verified_openaq_block(summary, block, metric=metric)
    if source == "purpleair" and metric == "pm25":
        return _eligible_purpleair_pm25_block(summary, block)
    return block


def _verified_openaq_block(
    summary: Mapping,
    block: Mapping,
    *,
    metric: str,
) -> Mapping | None:
    """Rebuild an OpenAQ metric block from its exact B6 v2 allowlist."""
    eligible_ids = verified_monitor_entity_ids(metric)
    raw_entities = block.get("entities")
    if not isinstance(raw_entities, list):
        return None

    anomaly_ts = _anomaly_ts(summary)
    entities: list[dict] = []
    nearest_candidates: list[tuple[datetime, float, str, float]] = []
    values: list[float] = []

    for raw_entity in raw_entities:
        if not isinstance(raw_entity, Mapping):
            continue
        entity_id = str(raw_entity.get("entity_id", ""))
        if entity_id not in eligible_ids:
            continue
        try:
            distance = float(raw_entity.get("distance_km", math.inf))
        except (TypeError, ValueError):
            distance = math.inf

        series: list[list[object]] = []
        for raw_pair in raw_entity.get("series", []):
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                continue
            raw_timestamp, raw_value = raw_pair
            try:
                timestamp = datetime.fromisoformat(str(raw_timestamp))
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if not math.isfinite(value):
                continue
            normalized_timestamp = timestamp.isoformat()
            series.append([normalized_timestamp, value])
            values.append(value)
            nearest_candidates.append((timestamp, distance, entity_id, value))

        if series:
            entity = dict(raw_entity)
            entity["entity_id"] = entity_id
            entity["n_points"] = len(series)
            entity["series"] = series
            entities.append(entity)

    if not entities or not nearest_candidates:
        return None

    if anomaly_ts is None:
        original_nearest = block.get("nearest_in_time", {})
        if str(original_nearest.get("entity_id", "")) not in eligible_ids:
            return None
        nearest = dict(original_nearest)
    else:
        timestamp, distance, entity_id, value = min(
            nearest_candidates,
            key=lambda item: (
                abs(item[0] - anomaly_ts),
                item[1],
                item[2],
                item[0],
            ),
        )
        nearest = {
            "t": timestamp.isoformat(),
            "v": value,
            "entity_id": entity_id,
            "distance_km": round(distance, 3),
            "dt_minutes": round(
                abs(timestamp - anomaly_ts).total_seconds() / 60.0, 1
            ),
        }

    filtered = dict(block)
    filtered["n_points"] = len(values)
    filtered["n_entities"] = len(entities)
    filtered["value_range"] = {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }
    filtered["nearest_in_time"] = nearest
    filtered["entities"] = entities
    return filtered


def _eligible_purpleair_pm25_block(
    summary: Mapping, block: Mapping
) -> Mapping | None:
    """Rebuild a PurpleAir PM2.5 block from B7-eligible rows only."""
    raw_entities = block.get("entities")
    if not isinstance(raw_entities, list):
        return None

    anomaly_ts = _anomaly_ts(summary)
    entities: list[dict] = []
    nearest_candidates: list[tuple[datetime, float, str, float]] = []
    values: list[float] = []

    for raw_entity in raw_entities:
        if not isinstance(raw_entity, Mapping):
            continue
        entity_id = str(raw_entity.get("entity_id", ""))
        try:
            distance = float(raw_entity.get("distance_km", math.inf))
        except (TypeError, ValueError):
            distance = math.inf

        series: list[list[object]] = []
        for raw_pair in raw_entity.get("series", []):
            if not isinstance(raw_pair, (list, tuple)) or len(raw_pair) != 2:
                continue
            raw_timestamp, raw_value = raw_pair
            try:
                timestamp = datetime.fromisoformat(str(raw_timestamp))
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if (
                not math.isfinite(value)
                or not purpleair_reading_is_eligible(entity_id, timestamp)
            ):
                continue
            normalized_timestamp = timestamp.isoformat()
            series.append([normalized_timestamp, value])
            values.append(value)
            nearest_candidates.append((timestamp, distance, entity_id, value))

        if series:
            entity = dict(raw_entity)
            entity["entity_id"] = entity_id
            entity["n_points"] = len(series)
            entity["series"] = series
            entities.append(entity)

    if not entities or not nearest_candidates:
        return None

    nearest: dict[str, object] = {}
    if anomaly_ts is None:
        original_nearest = block.get("nearest_in_time", {})
        raw_timestamp = original_nearest.get("t")
        entity_id = str(original_nearest.get("entity_id", ""))
        try:
            timestamp = datetime.fromisoformat(str(raw_timestamp))
        except (TypeError, ValueError):
            timestamp = None
        if timestamp is not None:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if purpleair_reading_is_eligible(entity_id, timestamp):
                nearest = dict(original_nearest)
    else:
        timestamp, distance, entity_id, value = min(
            nearest_candidates,
            key=lambda item: (
                abs(item[0] - anomaly_ts),
                item[1],
                item[2],
                item[0],
            ),
        )
        nearest = {
            "t": timestamp.isoformat(),
            "v": value,
            "entity_id": entity_id,
            "distance_km": round(distance, 3),
            "dt_minutes": round(
                abs(timestamp - anomaly_ts).total_seconds() / 60.0, 1
            ),
        }

    filtered = dict(block)
    filtered["n_points"] = len(values)
    filtered["n_entities"] = len(entities)
    filtered["value_range"] = {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 4),
    }
    filtered["nearest_in_time"] = nearest
    filtered["entities"] = entities
    return filtered


# The redundant regulatory ground sources (one channel; the aggregator
# collapses their agreement) and the per-metric candidates the descriptive
# scorers consult. The May-June coverage audit found OpenAQ carries zero
# in-window NO2/SO2/CO for Houston, so scorers hardcoded to OpenAQ were
# structurally silent for exactly the petrochemical species TCEQ was added
# to supply.
_GROUND_INSITU_SOURCES: tuple[str, ...] = ("openaq", "tceq", "epa_aqs")


def _ground_sources_for(metric: str | None) -> tuple[str, ...]:
    """Ground-station sources that can carry ``metric``, optical included."""
    if metric == "pm25":
        return _GROUND_INSITU_SOURCES + ("purpleair",)
    return _GROUND_INSITU_SOURCES


def phase2_metric_owners() -> dict[tuple[str, str], tuple[str, ...]]:
    """Every source/metric pair a live Phase-2 path or locked guard can read.

    B3 derives its exempt inventory from this scorer-owned registry instead of
    duplicating an allowlist in the pruning tool. Pollutant entries are built
    from the same alias and satellite-column mappings used by claim routing;
    fixed meteorology entries name the exact consuming scorer or locked guard.
    """
    owners: dict[tuple[str, str], set[str]] = {}

    def add(source: str, metric: str, *functions: str) -> None:
        owners.setdefault((source, metric), set()).update(functions)

    pollutant_metrics = sorted({metric for _pattern, metric in _POLLUTANT_PATTERNS})
    general_ground_owners = (
        "score_concentration_elevation",
        "score_temporal_pattern",
        "score_emissions_source_type",
        "score_background_vs_event",
    )
    for source in _GROUND_INSITU_SOURCES:
        for metric in pollutant_metrics:
            add(source, metric, *general_ground_owners)
        for metric in ("no2", "ozone"):
            add(source, metric, "score_chemistry", "score_secondary_formation")
    add("purpleair", "pm25", *general_ground_owners)
    for metric in sorted(set(_SENTINEL_COLUMN.values())):
        add(
            "sentinel5p",
            metric,
            "score_concentration_elevation",
            "score_temporal_pattern",
        )
    add("sentinel5p", "s5p_hcho_column", "score_chemistry")

    for metric in ("u_10m", "v_10m"):
        add(
            "noaa_gfs",
            metric,
            "score_transport_direction",
            "score_meteorological_state",
            "score_point_source_attribution",
            "calm_wind_source_decisions",
            "locked_rule_B2",
        )
    add(
        "noaa_gfs",
        "pbl_height",
        "score_atmospheric_trap",
        "locked_rule_R1",
    )

    add(
        "openweather",
        "wind_direction",
        "score_transport_direction",
        "score_point_source_attribution",
    )
    add(
        "openweather",
        "wind_speed",
        "score_meteorological_state",
        "calm_wind_source_decisions",
        "locked_rule_B2",
    )
    add("openweather", "temperature", "score_meteorological_state")
    add("openweather", "cloud_cover", "score_secondary_formation")

    add("asos", "wind_direction", "score_transport_direction")
    add(
        "asos",
        "wind_speed",
        "score_meteorological_state",
        "calm_wind_source_decisions",
        "locked_rule_B2",
    )
    add("asos", "temperature", "score_meteorological_state")
    return {
        key: tuple(sorted(functions))
        for key, functions in sorted(owners.items())
    }


def _window_mean(block: Mapping | None) -> float | None:
    if not block:
        return None
    return block.get("value_range", {}).get("mean")


def _pooled_series(block: Mapping | None) -> list[tuple[datetime, float]]:
    """All entities' (timestamp, value) pairs merged, UTC-coerced, time-sorted."""
    if not block:
        return []
    pairs: list[tuple[datetime, float]] = []
    for entity in block.get("entities", []):
        for iso, value in entity.get("series", []):
            try:
                ts = datetime.fromisoformat(iso)
                v = float(value)
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            pairs.append((ts, v))
    return sorted(pairs)


def _station_means(block: Mapping | None, *, min_obs: int = 1) -> list[float]:
    """Per-station mean of in-window values, for stations with enough coverage."""
    if not block:
        return []
    means: list[float] = []
    for entity in block.get("entities", []):
        values = [v for _, v in entity.get("series", [])]
        if len(values) >= min_obs:
            means.append(fmean(values))
    return means


def _spatial_cv(means: list[float]) -> float | None:
    """Coefficient of variation across station means; None when undefined."""
    if len(means) < 2:
        return None
    mean = fmean(means)
    if mean == 0:
        return None
    return pstdev(means) / abs(mean)


def _earliest_keyword(text: str, groups: Mapping[str, tuple[str, ...]]) -> str | None:
    """The group whose keyword appears first in the text; None if none match.

    First-mention wins so "a point source rather than rush-hour mobile" reads
    as a point-source assertion, not a mobile one.
    """
    best: tuple[int, str] | None = None
    for group, keywords in groups.items():
        for keyword in keywords:
            index = text.find(keyword)
            if index >= 0 and (best is None or index < best[0]):
                best = (index, group)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# Claim type 4 — atmospheric_trap (GFS PBL height + T@850, OpenWeather sfc T)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrapTolerance:
    """Atmospheric-trap tolerances; only the R1 fields are Bracco-locked."""

    pbl_m: float = 200.0    # numeric PBL-height claim within this of GFS
    # R1: qualitative suppression requires the event PBL to be at least two
    # population standard deviations below its same-cell/same-hour reference.
    suppression_sigma: float = 2.0
    # R1 requires this many distinct reference days on the matched cell.
    min_same_hour_points: int = 2


DEFAULT_TRAP_TOLERANCE = TrapTolerance()

_PBL_WORDS = ("pbl", "boundary layer")
_LOW_PBL_WORDS = ("low", "shallow")
# Qualitative trap vocabulary. Inversion/capping claims are verified via PBL
# suppression too: the T850-vs-2m criterion this replaces is physically
# unreachable in a Houston summer (0 of 100 paired June hours had t_850 >
# surface; mean gap +9.3 C) because the relevant inversions sit *below*
# 850 hPa — every inversion claim was auto-contradicted regardless of truth.
# A trapping inversion expresses itself in the GFS fields as a suppressed
# mixing height, which pbl_height measures directly. Criterion pending
# Dr. Bracco.
_TRAP_QUAL_WORDS = _LOW_PBL_WORDS + (
    "inversion", "trapped", "trapping", "capping", "capped",
)


def _claimed_pbl_height(text: str) -> float | None:
    if not any(word in text for word in _PBL_WORDS):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", text)
    return float(m.group(1)) if m else None


def _same_hour_values(
    block: Mapping | None, nearest_iso: str | None
) -> tuple[int | None, str | None, list[float], int]:
    """Return the R1 same-cell/same-hour reference and distinct-day count.

    The event-nearest entity fixes the GFS grid cell. Reference readings must
    be on other UTC calendar dates at the same UTC hour; neither spatial
    replicates nor extra readings on the event date can satisfy the floor.
    """
    if not nearest_iso or not block:
        return None, None, [], 0
    try:
        nearest_ts = datetime.fromisoformat(nearest_iso)
    except (TypeError, ValueError):
        return None, None, [], 0
    if nearest_ts.tzinfo is None:
        nearest_ts = nearest_ts.replace(tzinfo=timezone.utc)
    else:
        nearest_ts = nearest_ts.astimezone(timezone.utc)

    nearest = block.get("nearest_in_time")
    if not isinstance(nearest, Mapping) or nearest.get("entity_id") is None:
        return nearest_ts.hour, None, [], 0
    entity_id = str(nearest["entity_id"])

    values: list[float] = []
    reference_dates: set = set()
    for entity in block.get("entities", []):
        if str(entity.get("entity_id")) != entity_id:
            continue
        for raw_iso, raw_value in entity.get("series", []):
            try:
                ts = datetime.fromisoformat(raw_iso)
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.astimezone(timezone.utc)
            if ts.date() == nearest_ts.date() or ts.hour != nearest_ts.hour:
                continue
            values.append(value)
            reference_dates.add(ts.date())
    return nearest_ts.hour, entity_id, values, len(reference_dates)


def score_atmospheric_trap(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: TrapTolerance = DEFAULT_TRAP_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a PBL / inversion / trapping claim against GFS mixing height.

    A numeric PBL claim is checked within ``pbl_m`` of the GFS value nearest
    the anomaly. Qualitative trap wording (shallow/low PBL, inversion,
    capping, trapped) is checked as *suppression*: the nearest PBL must sit
    ``suppression_sigma`` standard deviations below the mean of same-hour-of-
    day readings elsewhere in the window, so the diurnal PBL cycle cannot
    masquerade as a verdict (a nocturnal claim is compared against other
    nights, not against afternoon mixing heights).
    """
    text = claim_text.lower()
    pbl_block = _metric_block(summary, "noaa_gfs", "pbl_height")
    claimed = _claimed_pbl_height(text)
    has_qualitative_intent = any(w in text for w in _TRAP_QUAL_WORDS)
    pbl_nearest: float | None = None
    age_note: str | None = None
    if claimed is not None or has_qualitative_intent:
        pbl_nearest, age_note = _fresh_nearest_value(
            "noaa_gfs", "pbl_height", pbl_block
        )

    pbl_verdict: int | None = None
    note = age_note or f"pbl_nearest={pbl_nearest}"
    if age_note is not None:
        return {"noaa_gfs": SILENT}, note
    if pbl_nearest is not None and claimed is not None:
        pbl_verdict = (
            SUPPORTING
            if abs(pbl_nearest - claimed) <= tolerance.pbl_m
            else CONTRADICTING
        )
        note += f" claimed={claimed}"
    elif pbl_nearest is not None and has_qualitative_intent:
        nearest_iso = (pbl_block or {}).get("nearest_in_time", {}).get("t")
        hour, entity_id, peers, distinct_days = _same_hour_values(
            pbl_block, nearest_iso
        )
        if entity_id is None:
            note += " missing event grid cell; same-cell reference unavailable"
        elif distinct_days >= tolerance.min_same_hour_points:
            peer_mean = fmean(peers)
            peer_spread = pstdev(peers)
            suppression_floor = (
                peer_mean - tolerance.suppression_sigma * peer_spread
            )
            if peer_spread == 0.0:
                note += (
                    f" same-hour({hour:02d}Z) cell={entity_id} "
                    f"mean={round(peer_mean, 1)} sigma=0.0 "
                    f"n={distinct_days} zero-spread reference"
                )
            elif pbl_nearest <= suppression_floor:
                pbl_verdict = SUPPORTING
            elif pbl_nearest >= peer_mean:
                pbl_verdict = CONTRADICTING
            if peer_spread != 0.0:
                note += (
                    f" same-hour({hour:02d}Z) cell={entity_id} "
                    f"mean={round(peer_mean, 1)} "
                    f"sigma={round(peer_spread, 1)} "
                    f"threshold={round(suppression_floor, 1)} "
                    f"n={distinct_days}"
                )
        else:
            note += (
                " insufficient same-hour same-cell distinct-day history "
                f"(n={distinct_days})"
            )

    return {"noaa_gfs": _combine(pbl_verdict)}, note


# ---------------------------------------------------------------------------
# Claim type 5 — temporal_pattern (trend direction on any source's series)
# ---------------------------------------------------------------------------

_RISE_WORDS = ("rose", "rising", "climb", "increas", "grew", "build", "accumulat", "ramp")
_FALL_WORDS = ("fell", "falling", "declin", "decreas", "drop", "subsid", "dissipat", "eased")

# Below this many in-window points a half-vs-half trend test is meaningless.
_MIN_TREND_POINTS = 4


def _trend_intent(text: str) -> str | None:
    if any(word in text for word in _RISE_WORDS):
        return "up"
    if any(word in text for word in _FALL_WORDS):
        return "down"
    return None


@dataclass(frozen=True)
class TemporalTolerance:
    """Draft windowing for temporal_pattern (pending Dr. Bracco).

    Trend claims describe the hours around the event, not the full 72 h
    context window — peaks on other days would otherwise vote on a claim
    about this one. Rising claims are tested on the lookback ending at the
    anomaly; falling claims on the hours after it.
    """

    window_h: float = 12.0


DEFAULT_TEMPORAL_TOLERANCE = TemporalTolerance()


def _claim_trend_window(
    series: list[tuple[datetime, float]],
    intent: str,
    anomaly_ts: datetime | None,
    window_h: float,
) -> list[tuple[datetime, float]]:
    """The slice of the series a trend claim is actually about.

    Without an anomaly timestamp (synthetic summaries) the whole series is
    used, matching the enrichment window.
    """
    if anomaly_ts is None:
        return series
    if intent == "up":
        start, end = anomaly_ts - timedelta(hours=window_h), anomaly_ts
    else:
        start, end = anomaly_ts, anomaly_ts + timedelta(hours=window_h)
    return [(ts, v) for ts, v in series if start <= ts <= end]


def score_temporal_pattern(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: TemporalTolerance = DEFAULT_TEMPORAL_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a 'levels rose / fell' claim by trend direction, no fixed tolerance.

    Trend = second-half mean vs first-half mean of the pooled series inside
    the claim's window (memo: "just whether the trend actually moved that
    way") — rising claims read the hours leading into the anomaly, falling
    claims the hours after it.
    """
    text = claim_text.lower()
    openaq_metric, sentinel_metric = _resolve_pollutant(text)
    intent = _trend_intent(text)
    if openaq_metric is None or intent is None:
        return {}, "no recognized pollutant + trend direction in claim"

    anomaly_ts = _anomaly_ts(summary)
    verdicts: dict[str, int] = {}
    notes: list[str] = []
    # Every ground source that can carry the species trends independently
    # (trend direction is invariant under PurpleAir's multiplicative bias);
    # the channel aggregator collapses the redundant regulatory ones.
    candidates = [
        (source, openaq_metric) for source in _ground_sources_for(openaq_metric)
    ]
    candidates.append(("sentinel5p", sentinel_metric))
    for source, metric in candidates:
        if metric is None:
            continue
        series = _claim_trend_window(
            _pooled_series(_metric_block(summary, source, metric)),
            intent,
            anomaly_ts,
            tolerance.window_h,
        )
        if len(series) < _MIN_TREND_POINTS:
            verdicts[source] = SILENT
            notes.append(f"{source}: {len(series)} points < {_MIN_TREND_POINTS}")
            continue
        values = [v for _, v in series]
        half = len(values) // 2
        first, second = fmean(values[:half]), fmean(values[half:])
        if second == first:
            verdicts[source] = SILENT
            notes.append(f"{source}: flat series")
            continue
        observed = "up" if second > first else "down"
        verdicts[source] = SUPPORTING if observed == intent else CONTRADICTING
        notes.append(
            f"{source}: {metric} first_half={first:.2f} second_half={second:.2f} "
            f"observed={observed} claimed={intent}"
        )
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Claim type 6 — chemistry (Sentinel-5P HCHO, OpenAQ O3/NO2) — qualitative-only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChemistryTolerance:
    """Draft tolerance for chemistry (pending Dr. Bracco).

    TROPOMI HCHO is noisy at single-orbit resolution, so directional checks
    get a +/-50% buffer zone around the window mean in which the granule is
    treated as silent rather than contradicting (memo rule).
    """

    noise_buffer: float = 0.5


DEFAULT_CHEMISTRY_TOLERANCE = ChemistryTolerance()

_UP_ADJ = r"elevated|enhanced|high|increased|excess"
_DOWN_ADJ = r"depressed|low|suppressed|reduced|depleted"
_CHEM_SPECIES: tuple[tuple[str, str], ...] = (
    (r"hcho|formaldehyde", "hcho"),
    (r"ozone|o3", "ozone"),
    (r"no2|nitrogen dioxide", "no2"),
)


def _species_directions(text: str) -> dict[str, str]:
    """{'hcho': 'up', 'ozone': 'down', ...} for adjective-species mentions."""
    directions: dict[str, str] = {}
    for pattern, species in _CHEM_SPECIES:
        m = re.search(
            rf"({_UP_ADJ}|{_DOWN_ADJ})\s+(?:\w+\s+)?(?:{pattern})\b", text
        )
        if m:
            directions[species] = "up" if re.fullmatch(_UP_ADJ, m.group(1)) else "down"
    return directions


def _direction_verdict(
    nearest: float | None,
    mean: float | None,
    direction: str,
    *,
    noise_buffer: float = 0.0,
) -> int | None:
    if nearest is None or mean is None:
        return None
    if direction == "up":
        if nearest > mean:
            return SUPPORTING
        return CONTRADICTING if nearest < mean * (1 - noise_buffer) else SILENT
    if nearest < mean:
        return SUPPORTING
    return CONTRADICTING if nearest > mean * (1 + noise_buffer) else SILENT


def score_chemistry(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: ChemistryTolerance = DEFAULT_CHEMISTRY_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a chemical-signature claim. Qualitative-only per Bracco 2026-06-10:
    scored and stored, but excluded from all quantitative reporting."""
    directions = _species_directions(claim_text.lower())
    verdicts: dict[str, int] = {}
    notes: list[str] = []

    if "hcho" in directions:
        block = _metric_block(summary, "sentinel5p", "s5p_hcho_column")
        nearest, age_note = _fresh_nearest_value(
            "sentinel5p", "s5p_hcho_column", block
        )
        verdict = _direction_verdict(
            nearest,
            _window_mean(block),
            directions["hcho"],
            noise_buffer=tolerance.noise_buffer,
        )
        verdicts["sentinel5p"] = SILENT if verdict is None else verdict
        if age_note is not None:
            notes.append(age_note)
        else:
            notes.append(
                "sentinel5p: hcho granule silent"
                if verdict is None
                else f"sentinel5p: hcho {directions['hcho']} vs window mean"
            )

    ground_species = [s for s in ("ozone", "no2") if s in directions]
    if ground_species:
        # Each ground in-situ source checks its own readings (OpenAQ carries
        # no in-window NO2 for Houston; TCEQ does), one verdict per source —
        # the channel aggregator collapses their redundancy.
        for source in _GROUND_INSITU_SOURCES:
            legs = []
            for species in ground_species:
                block = _metric_block(summary, source, species)
                nearest, age_note = _fresh_nearest_value(source, species, block)
                legs.append(
                    _direction_verdict(
                        nearest, _window_mean(block), directions[species]
                    )
                )
                if age_note is not None:
                    notes.append(age_note)
                elif block is not None:
                    notes.append(f"{source}: {species} {directions[species]} checked")
            verdicts[source] = _combine(*legs)

    if not directions:
        notes.append("no species direction recognized in claim")
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Claim type 7 — point_source_attribution (wind direction only) — qualitative-only
# ---------------------------------------------------------------------------

_COORD_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*°?\s*([ns])\b[^a-z0-9-]*(-?\d+(?:\.\d+)?)\s*°?\s*([ew])\b"
)


def _claimed_coordinates(text: str) -> tuple[float, float] | None:
    m = _COORD_RE.search(text)
    if not m:
        return None
    lat = abs(float(m.group(1))) * (1.0 if m.group(2) == "n" else -1.0)
    lon = abs(float(m.group(3))) * (1.0 if m.group(4) == "e" else -1.0)
    return lat, lon


def _bearing_deg(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> float:
    """Initial bearing from one point to another (equirectangular, fine at 50 km)."""
    mid_lat = math.radians((from_lat + to_lat) / 2.0)
    dx = (to_lon - from_lon) * math.cos(mid_lat)
    dy = to_lat - from_lat
    return math.degrees(math.atan2(dx, dy)) % 360.0


def score_point_source_attribution(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Direction-only check: does the wind blow from the claimed source toward
    the anomaly? Qualitative-only per Bracco 2026-06-10. Sentinel-5P is silent
    in v1 (granule-mean carries no spatial gradient; per-pixel is Month 4)."""
    anomaly = summary.get("anomaly") or {}
    coords = _claimed_coordinates(claim_text.lower())
    expected_from: float | None = None
    if coords and anomaly.get("lat") is not None and anomaly.get("lon") is not None:
        expected_from = _bearing_deg(
            anomaly["lat"], anomaly["lon"], coords[0], coords[1]
        )

    measured: dict[str, float] = {}
    age_notes: list[str] = []
    if expected_from is not None:
        u, v, gfs_age_notes = _gfs_wind_components(summary)
        age_notes.extend(gfs_age_notes)
        if u is not None and v is not None:
            measured["noaa_gfs"] = _wind_from_bearing(u, v)
        ow_dir, ow_age_note = _fresh_nearest_value(
            "openweather",
            "wind_direction",
            _metric_block(summary, "openweather", "wind_direction"),
        )
        if ow_age_note is not None:
            age_notes.append(ow_age_note)
        if ow_dir is not None:
            measured["openweather"] = float(ow_dir) % 360.0

    guard_decisions: dict[str, CalmWindDecision] = {}
    guard_notes: tuple[str, ...] = ()
    if expected_from is not None:
        guard_decisions, guard_notes = calm_wind_source_decisions(
            summary,
            ("noaa_gfs", "openweather"),
            tolerance=tolerance,
        )

    verdicts: dict[str, int] = {}
    notes: list[str] = [*age_notes, *guard_notes]
    for source in ("noaa_gfs", "openweather"):
        if expected_from is None or source not in measured:
            verdicts[source] = SILENT
            notes.append(f"{source}: no claimed coordinates or wind data")
            continue
        if not guard_decisions[source].direction_votable:
            verdicts[source] = SILENT
            continue
        diff = _angular_diff(expected_from, measured[source])
        verdicts[source] = (
            SUPPORTING if diff <= tolerance.bearing_deg else CONTRADICTING
        )
        notes.append(
            f"{source}: source_bearing={expected_from:.0f} "
            f"wind_from={measured[source]:.0f} diff={diff:.0f}"
        )
    notes.append("sentinel5p: granule-mean only, no spatial check in v1")
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Claim type 8 — emissions_source_type (OpenAQ temporal + spatial pattern)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SourceTypeTolerance:
    """Draft thresholds for emissions_source_type (pending Dr. Bracco)."""

    cv_localized: float = 0.5   # spatial CV at/above this reads as a point source
    cv_uniform: float = 0.3     # spatial CV at/below this reads as area-wide
    morning_start_h: int = 6    # local rush-hour window for the mobile check
    morning_end_h: int = 10
    min_stations: int = 3
    min_points: int = 4
    # Per-station in-window obs floor for the spatial-CV path: a station with
    # fewer readings has too thin a mean to anchor the point/area verdict. Set
    # to match this type's own data-sufficiency floor (min_points), looser than
    # background_vs_event's 6 — consistent with this type's looser min_stations.
    min_obs_per_station: int = 4


DEFAULT_SOURCE_TYPE_TOLERANCE = SourceTypeTolerance()

# Houston CDT. Fixed offset is correct for the summer-only eval window (v1).
_LOCAL_UTC_OFFSET_H = -5

_SOURCE_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "mobile": ("mobile", "traffic", "rush-hour", "rush hour", "vehicle", "highway", "cars"),
    "point": ("point source", "refinery", "industrial", "plant", "facility", "upset"),
    "area": ("area source", "area-wide", "areawide", "city-wide", "distributed"),
}


def score_emissions_source_type(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: SourceTypeTolerance = DEFAULT_SOURCE_TYPE_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a mobile / point / area source-type claim against station patterns.

    mobile: pooled series peaks in the local morning-rush window.
    point: high spatial CV across stations (one dominates).
    area: low spatial CV (uniform field).
    Each ground source is scored on its own station field — CV is never mixed
    across instruments with different response scales (PurpleAir's
    multiplicative bias cancels within its own CV, not against regulatory
    mass). Sentinel-5P spatial pattern and wind legs are deferred
    (granule-mean v1).
    """
    text = claim_text.lower()
    intent = _earliest_keyword(text, _SOURCE_TYPE_KEYWORDS)
    if intent is None:
        return {}, "no source-type intent recognized in claim"

    # Default to pm25, not no2: the May–June 2026 coverage audit found zero
    # active NO2/SO2/CO ground sensors in OpenAQ's Houston 50 km radius, so a
    # no2 default would leave every unnamed-pollutant claim silent.
    metric, _sentinel = _resolve_pollutant(text)
    metric = metric or "pm25"

    verdicts: dict[str, int] = {}
    notes: list[str] = []
    anomaly_ts = _anomaly_ts(summary) if intent == "mobile" else None
    for source in _ground_sources_for(metric):
        block = _metric_block(summary, source, metric)

        if intent == "mobile":
            series = _local_day_slice(_pooled_series(block), anomaly_ts)
            scope = "anomaly day" if anomaly_ts is not None else "window fallback"
            if len(series) < tolerance.min_points:
                verdicts[source] = SILENT
                notes.append(
                    f"{source}: {len(series)} points on {scope}, "
                    f"need {tolerance.min_points}"
                )
                continue
            peak_ts, _peak_v = max(series, key=lambda pair: pair[1])
            local_hour = (peak_ts.hour + _LOCAL_UTC_OFFSET_H) % 24
            in_rush = (
                tolerance.morning_start_h <= local_hour < tolerance.morning_end_h
            )
            verdicts[source] = SUPPORTING if in_rush else CONTRADICTING
            notes.append(
                f"{source}: {metric} peak at {local_hour:02d}:00 local "
                f"({scope})"
            )
            continue

        means = _station_means(block, min_obs=tolerance.min_obs_per_station)
        if len(means) < tolerance.min_stations:
            verdicts[source] = SILENT
            notes.append(
                f"{source}: {len(means)} stations with >= "
                f"{tolerance.min_obs_per_station} obs, need {tolerance.min_stations}"
            )
            continue
        cv = _spatial_cv(means)
        if cv is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: spatial CV undefined")
            continue

        if intent == "point":
            if cv >= tolerance.cv_localized:
                verdict = SUPPORTING
            elif cv <= tolerance.cv_uniform:
                verdict = CONTRADICTING
            else:
                verdict = SILENT
        else:  # area
            if cv <= tolerance.cv_uniform:
                verdict = SUPPORTING
            elif cv >= tolerance.cv_localized:
                verdict = CONTRADICTING
            else:
                verdict = SILENT
        verdicts[source] = verdict
        notes.append(f"{source}: {metric} spatial_cv={cv:.2f} intent={intent}")
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Claim type 9 — secondary_formation (O3 lags NO2 + insolation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondaryTolerance:
    """Draft thresholds for secondary_formation (pending Dr. Bracco)."""

    min_lag_h: float = 2.0           # O3 peak at least this many hours after NO2
    clear_sky_max_cloud_pct: float = 50.0
    min_points: int = 3


DEFAULT_SECONDARY_TOLERANCE = SecondaryTolerance()


def _local_day_slice(
    series: list[tuple[datetime, float]],
    anomaly_ts: datetime | None,
) -> list[tuple[datetime, float]]:
    """Points falling on the anomaly's local (CDT) calendar day.

    Without an anomaly timestamp (synthetic summaries) the whole series is
    returned. Fixed offset matches the summer-only eval window (v1).
    """
    if anomaly_ts is None:
        return series
    offset = timedelta(hours=_LOCAL_UTC_OFFSET_H)
    day = (anomaly_ts + offset).date()
    return [(ts, v) for ts, v in series if (ts + offset).date() == day]


def score_secondary_formation(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: SecondaryTolerance = DEFAULT_SECONDARY_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a photochemical-formation claim: O3 peak lags NO2 peak by >= 2 h
    (pooled ground in-situ) on a day with sufficient insolation (OpenWeather
    cloud cover). The insolation leg is conditional: it may vote only after
    the chemical lag leg produced a verdict.

    NO2 and O3 are pooled across the ground in-situ sources (OpenAQ carries
    ozone but no in-window NO2 for Houston; TCEQ carries the NO2) and the lag
    verdict lands on every source that contributed points — they share one
    channel, so the aggregator counts the pooled check once. Both legs are
    restricted to the anomaly's local calendar day — the 72 h window spans
    three days, and an O3 peak on day 2 vs an NO2 peak on day 3 says nothing
    about this day's photochemistry.
    """
    anomaly_ts = _anomaly_ts(summary)

    def _pooled_ground(metric: str) -> tuple[list[tuple[datetime, float]], set[str]]:
        pairs: list[tuple[datetime, float]] = []
        contributing: set[str] = set()
        for source in _GROUND_INSITU_SOURCES:
            series = _pooled_series(_metric_block(summary, source, metric))
            if series:
                contributing.add(source)
                pairs.extend(series)
        return _local_day_slice(sorted(pairs), anomaly_ts), contributing

    ozone, o3_sources = _pooled_ground("ozone")
    no2, no2_sources = _pooled_ground("no2")

    verdicts: dict[str, int] = {source: SILENT for source in _GROUND_INSITU_SOURCES}
    notes: list[str] = []
    lag_verdict: int | None = None

    if len(ozone) >= tolerance.min_points and len(no2) >= tolerance.min_points:
        o3_peak = max(ozone, key=lambda pair: pair[1])[0]
        no2_peak = max(no2, key=lambda pair: pair[1])[0]
        lag_h = (o3_peak - no2_peak).total_seconds() / 3600.0
        lag_verdict = SUPPORTING if lag_h >= tolerance.min_lag_h else CONTRADICTING
        for source in o3_sources | no2_sources:
            verdicts[source] = lag_verdict
        notes.append(
            f"{'/'.join(sorted(o3_sources | no2_sources))}: o3 peak lags no2 "
            f"by {lag_h:.1f} h (anomaly day)"
        )
    else:
        notes.append("ground in-situ: insufficient o3/no2 series on anomaly day")

    if lag_verdict is None:
        verdicts["openweather"] = SILENT
        notes.append("openweather: insolation conditioned SILENT; lag leg unavailable")
        return verdicts, "; ".join(notes)

    cloud_block = _metric_block(summary, "openweather", "cloud_cover")
    cloud_day = _local_day_slice(_pooled_series(cloud_block), anomaly_ts)
    if cloud_day:
        cloud_mean = fmean([v for _, v in cloud_day])
    else:
        # Summaries without per-entity series fall back to the window mean —
        # a coarser insolation proxy, but better than refusing a verdict.
        cloud_mean = _window_mean(cloud_block)
    if cloud_mean is None:
        verdicts["openweather"] = SILENT
        notes.append("openweather: no cloud cover in window")
    else:
        verdicts["openweather"] = (
            SUPPORTING
            if cloud_mean <= tolerance.clear_sky_max_cloud_pct
            else CONTRADICTING
        )
        notes.append(f"openweather: mean cloud {cloud_mean:.0f}%")
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Claim type 10 — background_vs_event (OpenAQ spatial uniformity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BackgroundTolerance:
    """Draft thresholds for background_vs_event (pending Dr. Bracco).

    min_stations / min_obs_per_station implement the data-quality precondition
    from her 2026-06-10 email (memo addendum): without enough reporting
    stations the spatial-CV check returns silent, never a verdict.
    """

    cv_regional: float = 0.3   # CV at/below this reads as a regional regime
    cv_local: float = 0.6      # CV at/above this reads as a localized event
    min_stations: int = 5
    min_obs_per_station: int = 6


DEFAULT_BACKGROUND_TOLERANCE = BackgroundTolerance()

_BACKGROUND_KEYWORDS: dict[str, tuple[str, ...]] = {
    "regional": (
        "regional", "widespread", "across all", "across every", "background",
        "whole area", "saharan", "area-wide",
    ),
    "local": (
        "local source", "localized", "isolated", "single monitor", "one monitor",
        "one spot", "local event",
    ),
}


def score_background_vs_event(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: BackgroundTolerance = DEFAULT_BACKGROUND_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a regional-vs-local claim by spatial CV across each source's stations.

    Per-source CV, never mixed across instruments: PurpleAir's dense sensor
    field (70+ Houston units) answers the uniformity question for PM2.5 where
    OpenAQ's sparser regulatory network cannot, and its multiplicative bias
    cancels within its own CV.
    """
    text = claim_text.lower()
    intent = _earliest_keyword(text, _BACKGROUND_KEYWORDS)
    if intent is None:
        return {}, "no regional/local intent recognized in claim"

    metric, _sentinel = _resolve_pollutant(text)
    metric = metric or "pm25"

    verdicts: dict[str, int] = {}
    notes: list[str] = []
    for source in _ground_sources_for(metric):
        block = _metric_block(summary, source, metric)
        means = _station_means(block, min_obs=tolerance.min_obs_per_station)

        if len(means) < tolerance.min_stations:
            verdicts[source] = SILENT
            notes.append(
                f"{source}: data-quality precondition unmet — {len(means)} "
                f"stations with >= {tolerance.min_obs_per_station} obs, need "
                f"{tolerance.min_stations}"
            )
            continue

        cv = _spatial_cv(means)
        if cv is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: spatial CV undefined")
            continue

        if intent == "regional":
            if cv <= tolerance.cv_regional:
                verdict = SUPPORTING
            elif cv >= tolerance.cv_local:
                verdict = CONTRADICTING
            else:
                verdict = SILENT
        else:  # local
            if cv >= tolerance.cv_local:
                verdict = SUPPORTING
            elif cv <= tolerance.cv_regional:
                verdict = CONTRADICTING
            else:
                verdict = SILENT
        verdicts[source] = verdict
        notes.append(
            f"{source}: {metric} spatial_cv={cv:.2f} over {len(means)} "
            f"stations intent={intent}"
        )
    return verdicts, "; ".join(notes)


# ---------------------------------------------------------------------------
# Claim-type classification + dispatch (memo flow, step 1)
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    """The 10-type claim taxonomy, plus a routing fallback for claims the
    rules don't recognize (still stored, scored all-silent)."""

    CONCENTRATION_ELEVATION = "concentration_elevation"
    TRANSPORT_DIRECTION = "transport_direction"
    METEOROLOGICAL_STATE = "meteorological_state"
    ATMOSPHERIC_TRAP = "atmospheric_trap"
    TEMPORAL_PATTERN = "temporal_pattern"
    CHEMISTRY = "chemistry"
    POINT_SOURCE_ATTRIBUTION = "point_source_attribution"
    EMISSIONS_SOURCE_TYPE = "emissions_source_type"
    SECONDARY_FORMATION = "secondary_formation"
    BACKGROUND_VS_EVENT = "background_vs_event"
    UNCLASSIFIED = "unclassified"


def direction_data_sources(
    claim_text: str,
    summary: Mapping,
    claim_type: ClaimType | str,
) -> tuple[str, ...]:
    """Sources with a fresh comparable direction value for B19 eligibility.

    This reports only data presence. It deliberately does not apply the B2
    calm-wind guard: a direction observation remains present even when calm
    conditions make its vote abstain.
    """
    primary = ClaimType(claim_type)
    present: list[str] = []
    if primary is ClaimType.TRANSPORT_DIRECTION:
        if _claimed_from_bearing(claim_text) is None:
            return ()
        u, v, _notes = _gfs_wind_components(summary)
        if u is not None and v is not None:
            present.append("noaa_gfs")
        for source in ("openweather", "asos"):
            direction, _note = _fresh_nearest_value(
                source,
                "wind_direction",
                _metric_block(summary, source, "wind_direction"),
            )
            if direction is not None:
                present.append(source)
        return tuple(present)
    if primary is ClaimType.POINT_SOURCE_ATTRIBUTION:
        anomaly = summary.get("anomaly") or {}
        coordinates = _claimed_coordinates(claim_text.lower())
        if (
            coordinates is None
            or anomaly.get("lat") is None
            or anomaly.get("lon") is None
        ):
            return ()
        u, v, _notes = _gfs_wind_components(summary)
        if u is not None and v is not None:
            present.append("noaa_gfs")
        direction, _note = _fresh_nearest_value(
            "openweather",
            "wind_direction",
            _metric_block(summary, "openweather", "wind_direction"),
        )
        if direction is not None:
            present.append("openweather")
    return tuple(present)


# Headline types get inferential statistics (N>=20 target); the rest are
# descriptive only.
HEADLINE_TYPES = frozenset(
    {
        ClaimType.CONCENTRATION_ELEVATION,
        ClaimType.TRANSPORT_DIRECTION,
        ClaimType.METEOROLOGICAL_STATE,
    }
)

# Bracco 2026-06-10: scored and stored but excluded from all quantitative
# reporting. Coincides with the taxonomy's partial-verifiability column today;
# kept separate because the addendum can move independently of the taxonomy.
QUALITATIVE_ONLY_TYPES = frozenset(
    {ClaimType.CHEMISTRY, ClaimType.POINT_SOURCE_ATTRIBUTION}
)

PARTIALLY_VERIFIABLE_TYPES = frozenset(
    {ClaimType.CHEMISTRY, ClaimType.POINT_SOURCE_ATTRIBUTION}
)

SO2_QUANTITATIVE_EXCLUSION_REASON = "so2_underpowered"


def quantitative_exclusion_reason(
    claim_text: str,
    primary: ClaimType,
) -> str | None:
    """B15 quantitative exclusion attached without changing claim routing."""
    pollutant, _sentinel = _resolve_pollutant(claim_text)
    if primary is ClaimType.CONCENTRATION_ELEVATION and pollutant == "so2":
        return SO2_QUANTITATIVE_EXCLUSION_REASON
    return None

# Classifier-only keyword groups. Routing cues, not verdict logic — the
# scorers re-derive what they need from the claim text themselves.
# "Ship Channel" is deliberately not an attribution cue: it is a geographic
# reference that appears in transport and source-type claims constantly in
# Houston, and routing on it drained the headline transport_direction type
# into a qualitative-only one.
_TRAP_WORDS = _PBL_WORDS + ("inversion", "trapped", "trapping", "capping", "mixing height")
_ATTRIBUTION_WORDS = ("plume", "smokestack", "stack emissions")
_SECONDARY_WORDS = ("secondary", "photochem", "precursor", "ozone formation", "ozone production")
# Word-boundary matched so "ratio" doesn't fire inside "concentrations".
_CHEM_RE = re.compile(
    r"\bhcho\b|\bformaldehyde\b|\bratios?\b|\bvocs?\b|\bnox-limited\b|\btitration\b"
)
_ELEVATION_RE = re.compile(rf"{_UP_ADJ}|spike|exceed|surpass|topped")
_HEAT_RE = re.compile(r"\bhot\b|\bheat\b|\bhumid")


def classify_claim(claim_text: str) -> list[ClaimType]:
    """Rule-based routing of a claim into taxonomy types, most verifiable first.

    A claim can match several types ("PM2.5 was elevated across all stations"
    is both background_vs_event and concentration_elevation); the first entry
    is the primary type the dispatcher scores. No match -> [UNCLASSIFIED].

    Ordering: explicit coordinates are the strongest point-source signal and
    the one shape that scorer can actually check, so they rank early; the
    looser attribution vocabulary ("plume") ranks after transport, because
    "southerly winds advected the plume northward" is a transport claim
    (headline type) that merely mentions a plume.
    """
    text = claim_text.lower()
    openaq_metric, sentinel_metric = _resolve_pollutant(text)
    has_pollutant = openaq_metric is not None or sentinel_metric is not None

    checks: tuple[tuple[ClaimType, bool], ...] = (
        (ClaimType.ATMOSPHERIC_TRAP, any(w in text for w in _TRAP_WORDS)),
        (ClaimType.POINT_SOURCE_ATTRIBUTION, _claimed_coordinates(text) is not None),
        (
            ClaimType.EMISSIONS_SOURCE_TYPE,
            _earliest_keyword(text, _SOURCE_TYPE_KEYWORDS) is not None,
        ),
        (ClaimType.SECONDARY_FORMATION, any(w in text for w in _SECONDARY_WORDS)),
        (
            ClaimType.CHEMISTRY,
            len(_species_directions(text)) >= 2
            or _CHEM_RE.search(text) is not None,
        ),
        (
            ClaimType.BACKGROUND_VS_EVENT,
            _earliest_keyword(text, _BACKGROUND_KEYWORDS) is not None,
        ),
        (ClaimType.TRANSPORT_DIRECTION, _claimed_from_bearing(text) is not None),
        (
            ClaimType.POINT_SOURCE_ATTRIBUTION,
            any(w in text for w in _ATTRIBUTION_WORDS),
        ),
        (ClaimType.TEMPORAL_PATTERN, _trend_intent(text) is not None),
        (
            ClaimType.METEOROLOGICAL_STATE,
            _wind_intent(text) is not None
            or _claimed_temperature(text) is not None
            or _HEAT_RE.search(text) is not None,
        ),
        (
            ClaimType.CONCENTRATION_ELEVATION,
            has_pollutant
            and (
                _threshold_value(text) is not None
                or _ELEVATION_RE.search(text) is not None
            ),
        ),
    )
    matches: list[ClaimType] = []
    for claim_type, hit in checks:
        if hit and claim_type not in matches:
            matches.append(claim_type)
    return matches or [ClaimType.UNCLASSIFIED]


_SCORERS: dict[ClaimType, Callable[[str, Mapping], tuple[dict[str, int], str]]] = {
    ClaimType.CONCENTRATION_ELEVATION: score_concentration_elevation,
    ClaimType.TRANSPORT_DIRECTION: score_transport_direction,
    ClaimType.METEOROLOGICAL_STATE: score_meteorological_state,
    ClaimType.ATMOSPHERIC_TRAP: score_atmospheric_trap,
    ClaimType.TEMPORAL_PATTERN: score_temporal_pattern,
    ClaimType.CHEMISTRY: score_chemistry,
    ClaimType.POINT_SOURCE_ATTRIBUTION: score_point_source_attribution,
    ClaimType.EMISSIONS_SOURCE_TYPE: score_emissions_source_type,
    ClaimType.SECONDARY_FORMATION: score_secondary_formation,
    ClaimType.BACKGROUND_VS_EVENT: score_background_vs_event,
}


@dataclass(frozen=True)
class ScoredClaim:
    """One claim routed and scored: everything Phase 2 contributes to a Claim row."""

    claim_type: ClaimType
    matched_types: tuple[ClaimType, ...]
    result: CorroborationResult
    evidence_summary: str
    partial_verifiability: bool
    qualitative_only: bool
    quantitative_exclusion_reason: str | None


def score_claim(claim_text: str, summary: Mapping) -> ScoredClaim:
    """Classify a claim and run its primary type's scorer.

    Unclassified claims aggregate an empty verdict set (all-silent,
    ``unverified=True``) rather than being dropped, so the eval can report how
    often the model asserts things outside the taxonomy.
    """
    matched = classify_claim(claim_text)
    primary = matched[0]
    verdicts: dict[str, int]
    if primary is ClaimType.UNCLASSIFIED:
        verdicts, note = {}, "no claim type recognized"
    else:
        verdicts, note = _SCORERS[primary](claim_text, summary)
    return ScoredClaim(
        claim_type=primary,
        matched_types=tuple(matched),
        result=aggregate_verdicts(verdicts),
        evidence_summary=note,
        partial_verifiability=primary in PARTIALLY_VERIFIABLE_TYPES,
        qualitative_only=primary in QUALITATIVE_ONLY_TYPES,
        quantitative_exclusion_reason=quantitative_exclusion_reason(
            claim_text,
            primary,
        ),
    )
