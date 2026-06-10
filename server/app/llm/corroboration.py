"""Phase 2 — cross-source corroboration scorer.

For each Phase-1-grounded claim about an atmospheric anomaly, score it against
the agreement of the four data sources (OpenAQ, Sentinel-5P, NOAA GFS,
OpenWeather), which sense different facets of one shared physical state through
largely independent measurement processes. Design + claim taxonomy:
docs/specs/2026-05-21-corroboration-scorer-design.md.

The module provides the shared aggregator that collapses per-source verdicts
into the scalar ``corroboration_score`` + ``evidence_n``, plus one scorer per
claim type: 3 headline types (1-3) and 7 descriptive types (4-10, of which
``chemistry`` and ``point_source_attribution`` are qualitative-only per the
memo's 2026-06-10 addendum).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from statistics import fmean, pstdev

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


# ---------------------------------------------------------------------------
# Shared helpers for the descriptive claim types (4-10)
# ---------------------------------------------------------------------------


def _metric_block(summary: Mapping, source: str, metric: str) -> Mapping | None:
    block = summary.get("sources", {}).get(source, {}).get("metrics", {}).get(metric)
    return block or None


def _nearest(block: Mapping | None) -> float | None:
    if not block:
        return None
    return block.get("nearest_in_time", {}).get("v")


def _window_mean(block: Mapping | None) -> float | None:
    if not block:
        return None
    return block.get("value_range", {}).get("mean")


def _pooled_series(block: Mapping | None) -> list[tuple[datetime, float]]:
    """All entities' (timestamp, value) pairs merged and time-sorted."""
    if not block:
        return []
    pairs: list[tuple[datetime, float]] = []
    for entity in block.get("entities", []):
        for iso, value in entity.get("series", []):
            try:
                pairs.append((datetime.fromisoformat(iso), float(value)))
            except (TypeError, ValueError):
                continue
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
    """Draft tolerances for atmospheric_trap (pending Dr. Bracco)."""

    pbl_m: float = 200.0  # numeric PBL-height claim within this of GFS


DEFAULT_TRAP_TOLERANCE = TrapTolerance()

_PBL_WORDS = ("pbl", "boundary layer")
_LOW_PBL_WORDS = ("low", "shallow")


def _claimed_pbl_height(text: str) -> float | None:
    if not any(word in text for word in _PBL_WORDS):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", text)
    return float(m.group(1)) if m else None


def score_atmospheric_trap(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: TrapTolerance = DEFAULT_TRAP_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a PBL / inversion trapping claim.

    Inversion = air aloft (GFS 850 hPa) warmer than the surface (OpenWeather);
    the verdict lands on both sources because each contributes one leg.
    A numeric PBL claim is checked within ``pbl_m`` of GFS; a qualitative
    "low/shallow" PBL is checked against the in-window mean.
    """
    text = claim_text.lower()
    wants_inversion = "inversion" in text

    pbl_block = _metric_block(summary, "noaa_gfs", "pbl_height")
    t_850 = _nearest(_metric_block(summary, "noaa_gfs", "t_850"))
    surface_t = _nearest(_metric_block(summary, "openweather", "temperature"))

    pbl_verdict: int | None = None
    pbl_nearest = _nearest(pbl_block)
    if pbl_nearest is not None and any(word in text for word in _PBL_WORDS):
        claimed = _claimed_pbl_height(text)
        if claimed is not None:
            pbl_verdict = (
                SUPPORTING
                if abs(pbl_nearest - claimed) <= tolerance.pbl_m
                else CONTRADICTING
            )
        elif any(word in text for word in _LOW_PBL_WORDS):
            mean = _window_mean(pbl_block)
            if mean is not None:
                pbl_verdict = SUPPORTING if pbl_nearest < mean else CONTRADICTING

    inversion_verdict: int | None = None
    if wants_inversion and t_850 is not None and surface_t is not None:
        inversion_verdict = SUPPORTING if t_850 > surface_t else CONTRADICTING

    verdicts = {
        "noaa_gfs": _combine(pbl_verdict, inversion_verdict),
        "openweather": _combine(inversion_verdict),
    }
    note = (
        f"pbl_nearest={pbl_nearest} t_850={t_850} surface_t={surface_t} "
        f"inversion_claimed={wants_inversion}"
    )
    return verdicts, note


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


def score_temporal_pattern(
    claim_text: str,
    summary: Mapping,
) -> tuple[dict[str, int], str]:
    """Score a 'levels rose / fell' claim by trend direction, no fixed tolerance.

    Trend = second-half mean vs first-half mean of the pooled in-window series
    (memo: "just whether the trend actually moved that way").
    """
    text = claim_text.lower()
    openaq_metric, sentinel_metric = _resolve_pollutant(text)
    intent = _trend_intent(text)
    if openaq_metric is None or intent is None:
        return {}, "no recognized pollutant + trend direction in claim"

    verdicts: dict[str, int] = {}
    notes: list[str] = []
    for source, metric in (("openaq", openaq_metric), ("sentinel5p", sentinel_metric)):
        if metric is None:
            continue
        series = _pooled_series(_metric_block(summary, source, metric))
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
        verdict = _direction_verdict(
            _nearest(block),
            _window_mean(block),
            directions["hcho"],
            noise_buffer=tolerance.noise_buffer,
        )
        verdicts["sentinel5p"] = SILENT if verdict is None else verdict
        notes.append(
            "sentinel5p: hcho granule silent"
            if verdict is None
            else f"sentinel5p: hcho {directions['hcho']} vs window mean"
        )

    openaq_species = [s for s in ("ozone", "no2") if s in directions]
    if openaq_species:
        legs = []
        for species in openaq_species:
            block = _metric_block(summary, "openaq", species)
            legs.append(
                _direction_verdict(_nearest(block), _window_mean(block), directions[species])
            )
            notes.append(f"openaq: {species} {directions[species]} checked")
        verdicts["openaq"] = _combine(*legs)

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
    u, v = _gfs_wind_components(summary)
    if u is not None and v is not None:
        measured["noaa_gfs"] = _wind_from_bearing(u, v)
    ow_dir = _nearest(_metric_block(summary, "openweather", "wind_direction"))
    if ow_dir is not None:
        measured["openweather"] = float(ow_dir) % 360.0

    verdicts: dict[str, int] = {}
    notes: list[str] = []
    for source in ("noaa_gfs", "openweather"):
        if expected_from is None or source not in measured:
            verdicts[source] = SILENT
            notes.append(f"{source}: no claimed coordinates or wind data")
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
    """Score a mobile / point / area source-type claim against OpenAQ patterns.

    mobile: pooled series peaks in the local morning-rush window.
    point: high spatial CV across stations (one dominates).
    area: low spatial CV (uniform field).
    Sentinel-5P spatial pattern and wind legs are deferred (granule-mean v1).
    """
    text = claim_text.lower()
    intent = _earliest_keyword(text, _SOURCE_TYPE_KEYWORDS)
    if intent is None:
        return {}, "no source-type intent recognized in claim"

    metric, _sentinel = _resolve_pollutant(text)
    metric = metric or "no2"
    block = _metric_block(summary, "openaq", metric)

    if intent == "mobile":
        series = _pooled_series(block)
        if len(series) < tolerance.min_points:
            return {"openaq": SILENT}, f"openaq: {len(series)} points, need {tolerance.min_points}"
        peak_ts, _peak_v = max(series, key=lambda pair: pair[1])
        local_hour = (peak_ts.hour + _LOCAL_UTC_OFFSET_H) % 24
        in_rush = tolerance.morning_start_h <= local_hour < tolerance.morning_end_h
        return (
            {"openaq": SUPPORTING if in_rush else CONTRADICTING},
            f"openaq: {metric} peak at {local_hour:02d}:00 local",
        )

    means = _station_means(block)
    if len(means) < tolerance.min_stations:
        return (
            {"openaq": SILENT},
            f"openaq: {len(means)} stations, need {tolerance.min_stations}",
        )
    cv = _spatial_cv(means)
    if cv is None:
        return {"openaq": SILENT}, "openaq: spatial CV undefined"

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
    return {"openaq": verdict}, f"openaq: {metric} spatial_cv={cv:.2f} intent={intent}"


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


def score_secondary_formation(
    claim_text: str,
    summary: Mapping,
    *,
    tolerance: SecondaryTolerance = DEFAULT_SECONDARY_TOLERANCE,
) -> tuple[dict[str, int], str]:
    """Score a photochemical-formation claim: O3 peak lags NO2 peak by >= 2 h
    (OpenAQ) on a day with sufficient insolation (OpenWeather cloud cover)."""
    ozone = _pooled_series(_metric_block(summary, "openaq", "ozone"))
    no2 = _pooled_series(_metric_block(summary, "openaq", "no2"))

    verdicts: dict[str, int] = {}
    notes: list[str] = []

    if len(ozone) >= tolerance.min_points and len(no2) >= tolerance.min_points:
        o3_peak = max(ozone, key=lambda pair: pair[1])[0]
        no2_peak = max(no2, key=lambda pair: pair[1])[0]
        lag_h = (o3_peak - no2_peak).total_seconds() / 3600.0
        verdicts["openaq"] = (
            SUPPORTING if lag_h >= tolerance.min_lag_h else CONTRADICTING
        )
        notes.append(f"openaq: o3 peak lags no2 by {lag_h:.1f} h")
    else:
        verdicts["openaq"] = SILENT
        notes.append("openaq: insufficient o3/no2 series")

    cloud_mean = _window_mean(_metric_block(summary, "openweather", "cloud_cover"))
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
    """Score a regional-vs-local claim by spatial CV across OpenAQ stations."""
    text = claim_text.lower()
    intent = _earliest_keyword(text, _BACKGROUND_KEYWORDS)
    if intent is None:
        return {}, "no regional/local intent recognized in claim"

    metric, _sentinel = _resolve_pollutant(text)
    metric = metric or "pm25"
    block = _metric_block(summary, "openaq", metric)
    means = _station_means(block, min_obs=tolerance.min_obs_per_station)

    if len(means) < tolerance.min_stations:
        return (
            {"openaq": SILENT},
            (
                f"openaq: data-quality precondition unmet — {len(means)} stations "
                f"with >= {tolerance.min_obs_per_station} obs, need "
                f"{tolerance.min_stations}"
            ),
        )

    cv = _spatial_cv(means)
    if cv is None:
        return {"openaq": SILENT}, "openaq: spatial CV undefined"

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
    return (
        {"openaq": verdict},
        f"openaq: {metric} spatial_cv={cv:.2f} over {len(means)} stations intent={intent}",
    )


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

# Classifier-only keyword groups. Routing cues, not verdict logic — the
# scorers re-derive what they need from the claim text themselves.
_TRAP_WORDS = _PBL_WORDS + ("inversion", "trapped", "trapping", "capping", "mixing height")
_ATTRIBUTION_WORDS = ("plume", "ship channel", "smokestack", "stack emissions")
_SECONDARY_WORDS = ("secondary", "photochem", "precursor", "ozone formation", "ozone production")
# Word-boundary matched so "ratio" doesn't fire inside "concentrations".
_CHEM_RE = re.compile(
    r"\bhcho\b|\bformaldehyde\b|\bratios?\b|\bvocs?\b|\bnox-limited\b|\btitration\b"
)
_ELEVATION_RE = re.compile(rf"{_UP_ADJ}|spike|exceed|surpass|topped")
_HEAT_RE = re.compile(r"\bhot\b|\bheat\b|\bhumid")


def classify_claim(claim_text: str) -> list[ClaimType]:
    """Rule-based routing of a claim into taxonomy types, most specific first.

    A claim can match several types ("PM2.5 was elevated across all stations"
    is both background_vs_event and concentration_elevation); the first entry
    is the primary type the dispatcher scores. No match -> [UNCLASSIFIED].
    """
    text = claim_text.lower()
    openaq_metric, sentinel_metric = _resolve_pollutant(text)
    has_pollutant = openaq_metric is not None or sentinel_metric is not None

    checks: tuple[tuple[ClaimType, bool], ...] = (
        (ClaimType.ATMOSPHERIC_TRAP, any(w in text for w in _TRAP_WORDS)),
        (
            ClaimType.POINT_SOURCE_ATTRIBUTION,
            _claimed_coordinates(text) is not None
            or any(w in text for w in _ATTRIBUTION_WORDS),
        ),
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
    matches = [claim_type for claim_type, hit in checks if hit]
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
    )
