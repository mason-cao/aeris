"""Phase 2 — cross-source corroboration scorer.

For each Phase-1-grounded claim about an atmospheric anomaly, score it against
the agreement of the four data sources (OpenAQ, Sentinel-5P, NOAA GFS,
OpenWeather), which sense different facets of one shared physical state through
largely independent measurement processes. Design + claim taxonomy:
docs/specs/2026-05-21-corroboration-scorer-design.md.

This module currently provides the shared aggregator that collapses per-source
verdicts into the scalar ``corroboration_score`` + ``evidence_n``. The ten
per-claim-type scorers build on top of it.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

# Per-source verdict on a single claim. A source either supports the claim,
# contradicts it, or is silent (no data bearing on it within the window).
SUPPORTING = 1
CONTRADICTING = -1
SILENT = 0

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


def aggregate_verdicts(per_source_verdicts: Mapping[str, int]) -> CorroborationResult:
    """Collapse per-source verdicts into a scalar score and evidence count.

    ``score = (supporting - contradicting) / evidence_n`` in [-1, +1];
    ``None`` when ``evidence_n == 0`` (every source silent).
    """
    supporting = sum(1 for v in per_source_verdicts.values() if v == SUPPORTING)
    contradicting = sum(1 for v in per_source_verdicts.values() if v == CONTRADICTING)
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
        )

    return CorroborationResult(
        corroboration_score=(supporting - contradicting) / evidence_n,
        evidence_n=evidence_n,
        supporting=supporting,
        contradicting=contradicting,
        unverified=False,
        per_source_verdicts=verdicts,
    )


def low_corroboration_flag(score: float | None, *, evidence_n: int) -> bool:
    """Phase 2 metadata flag: strongly contradicted across >= 2 sources.

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

# Words that make a numeric claim a threshold ("exceeded 80") rather than a
# point value ("was 80"). Threshold claims are met by measured >= claimed.
_THRESHOLD_WORDS: tuple[str, ...] = (
    "exceed",
    "above",
    "over ",
    "surpass",
    "topped",
    "reached",
    "greater than",
    "more than",
    ">",
)


@dataclass(frozen=True)
class ConcentrationTolerance:
    """Draft tolerances for concentration_elevation (pending Dr. Bracco)."""

    # Qualitative "elevated": the value nearest the anomaly must exceed the
    # in-window baseline (mean) by at least this ratio.
    elevated_ratio: float = 1.0


DEFAULT_CONCENTRATION_TOLERANCE = ConcentrationTolerance()


def _resolve_pollutant(claim_text: str) -> tuple[str | None, str | None]:
    """(OpenAQ metric, Sentinel column metric) named in the claim, if any."""
    lowered = claim_text.lower()
    for pattern, metric in _POLLUTANT_PATTERNS:
        if re.search(pattern, lowered):
            return metric, _SENTINEL_COLUMN.get(metric)
    return None, None


def _threshold_value(claim_text: str) -> float | None:
    """The numeric threshold in an 'exceeded N' style claim, else None."""
    lowered = claim_text.lower()
    if not any(word in lowered for word in _THRESHOLD_WORDS):
        return None
    # Strip pollutant tokens first so digits inside names (NO2, PM2.5, O3) are
    # not mistaken for the threshold value.
    cleaned = lowered
    for pattern, _metric in _POLLUTANT_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned)
    match = re.search(r"\d+(?:\.\d+)?", cleaned)
    return float(match.group()) if match else None


def score_concentration_elevation(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: ConcentrationTolerance = DEFAULT_CONCENTRATION_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a 'pollutant was elevated' claim against OpenAQ + Sentinel-5P.

    v1 assumes the claim's unit matches the stored metric's unit (no
    ppb<->ug/m^3 conversion). On that assumption a threshold claim is supported
    when the value nearest the anomaly meets the threshold, and a qualitative
    "elevated" is supported when the nearest value exceeds the in-window mean.
    Returns ``(per_source_verdicts, evidence_summary)``.
    """
    openaq_metric, sentinel_metric = _resolve_pollutant(claim_text)
    threshold = _threshold_value(claim_text)
    relevant = {"openaq": openaq_metric, "sentinel5p": sentinel_metric}

    sources = summary.get("sources", {})
    verdicts: dict[str, int] = {}
    notes: list[str] = []

    for source, metric in relevant.items():
        if metric is None:
            continue
        data = sources.get(source, {}).get("metrics", {}).get(metric)
        if not data or data.get("nearest_in_time", {}).get("v") is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: no {metric} in window")
            continue

        nearest = data["nearest_in_time"]["v"]
        if threshold is not None:
            verdict = SUPPORTING if nearest >= threshold else CONTRADICTING
            notes.append(
                f"{source}: {metric} nearest={nearest} vs threshold={threshold}"
            )
        else:
            mean = data.get("value_range", {}).get("mean", nearest)
            baseline = mean * tolerance.elevated_ratio
            verdict = SUPPORTING if nearest > baseline else CONTRADICTING
            notes.append(f"{source}: {metric} nearest={nearest} vs baseline={mean}")
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
    """Draft tolerances for the wind / met headline types (pending Dr. Bracco)."""

    bearing_deg: float = 45.0  # transport direction within this of measured wind
    speed_ms: float = 1.5      # numeric wind-speed claim within this of measured
    stagnant_ms: float = 2.0   # qualitative "stagnant" means wind below this
    temp_c: float = 2.0        # temperature claim within this of measured


DEFAULT_WIND_TOLERANCE = WindTolerance()

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


def _gfs_wind_components(summary: Mapping) -> tuple[float | None, float | None]:
    gfs = summary.get("sources", {}).get("noaa_gfs", {}).get("metrics", {})
    u = gfs.get("u_10m", {}).get("nearest_in_time", {}).get("v")
    v = gfs.get("v_10m", {}).get("nearest_in_time", {}).get("v")
    return u, v


def score_transport_direction(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: WindTolerance = DEFAULT_WIND_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a wind-transport claim against GFS 10 m wind + OpenWeather direction."""
    claimed_from = _claimed_from_bearing(claim_text)

    measured: dict[str, float] = {}
    u, v = _gfs_wind_components(summary)
    if u is not None and v is not None:
        measured["noaa_gfs"] = _wind_from_bearing(u, v)
    ow = summary.get("sources", {}).get("openweather", {}).get("metrics", {})
    ow_dir = ow.get("wind_direction", {}).get("nearest_in_time", {}).get("v")
    if ow_dir is not None:
        measured["openweather"] = float(ow_dir) % 360.0

    verdicts: dict[str, int] = {}
    notes: list[str] = []
    for source in ("noaa_gfs", "openweather"):
        if source not in measured or claimed_from is None:
            verdicts[source] = SILENT
            notes.append(f"{source}: no comparable wind direction")
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

    u, v = _gfs_wind_components(summary)
    gfs_speed = _wind_speed(u, v) if (u is not None and v is not None) else None
    ow = summary.get("sources", {}).get("openweather", {}).get("metrics", {})
    ow_speed = ow.get("wind_speed", {}).get("nearest_in_time", {}).get("v")
    ow_temp = ow.get("temperature", {}).get("nearest_in_time", {}).get("v")

    verdicts = {
        "noaa_gfs": _combine(wind_verdict(gfs_speed)),
        "openweather": _combine(wind_verdict(ow_speed), temp_verdict(ow_temp)),
    }
    return verdicts, f"wind_intent={wind} temp={temp}"
