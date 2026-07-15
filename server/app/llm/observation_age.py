"""B8 source-specific observation-age gates shared by event-value scorers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ObservationAgeGates:
    openaq: float = 90.0
    tceq: float = 90.0
    purpleair: float = 90.0
    asos: float = 90.0
    openweather: float = 90.0
    epa_aqs: float = 90.0
    noaa_gfs: float = 360.0
    sentinel5p: float = 720.0

    def to_dict(self) -> dict[str, float]:
        return {source: float(value) for source, value in asdict(self).items()}

    def for_source(self, source: str) -> float:
        gates = self.to_dict()
        if source not in gates:
            raise ValueError(f"no observation-age gate declared for source {source!r}")
        return gates[source]


DEFAULT_OBSERVATION_AGE_GATES = ObservationAgeGates()


@dataclass(frozen=True)
class ObservationAgeDecision:
    source: str
    gate_minutes: float
    dt_minutes: float | None
    votes: bool
    reason: str | None


def assess_observation_age(
    source: str,
    raw_dt_minutes: Any,
    *,
    gates: ObservationAgeGates = DEFAULT_OBSERVATION_AGE_GATES,
) -> ObservationAgeDecision:
    """Apply the inclusive B8 age boundary; malformed ages abstain."""
    gate = gates.for_source(source)
    try:
        if raw_dt_minutes is None or isinstance(raw_dt_minutes, bool):
            raise TypeError
        dt_minutes = float(raw_dt_minutes)
        if not math.isfinite(dt_minutes) or dt_minutes < 0.0:
            raise ValueError
    except (TypeError, ValueError):
        return ObservationAgeDecision(
            source=source,
            gate_minutes=gate,
            dt_minutes=None,
            votes=False,
            reason="missing_or_invalid",
        )
    if dt_minutes > gate:
        return ObservationAgeDecision(
            source=source,
            gate_minutes=gate,
            dt_minutes=dt_minutes,
            votes=False,
            reason="stale",
        )
    return ObservationAgeDecision(
        source=source,
        gate_minutes=gate,
        dt_minutes=dt_minutes,
        votes=True,
        reason=None,
    )
