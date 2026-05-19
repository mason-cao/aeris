"""Consensus across the three detectors.

Each detector (Z-score, STL, IsolationForest) emits independent anomalies on
the same underlying time-series. The consensus layer groups them by exact
timestamp, counts how many methods agreed, and assigns a severity tier:
1 method = minor, 2 = moderate, 3 = severe.

The merged record is a :class:`ConsensusAnomaly` — a pure data object that
mirrors the DB ``anomalies`` table schema. Persistence is the caller's
responsibility; this module only computes.
"""

from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from .isolation_forest import IsolationForestAnomaly
from .stl import STLAnomaly
from .zscore import ZScoreAnomaly


Severity = Literal["minor", "moderate", "severe"]

# Canonical method ordering — drives ``methods_triggered`` output for stable
# downstream rendering and JSON serialization.
ALL_METHODS: tuple[str, str, str] = ("zscore", "stl", "isolation_forest")

_SEVERITY_BY_COUNT: dict[int, Severity] = {1: "minor", 2: "moderate", 3: "severe"}


class ConsensusAnomaly(BaseModel):
    """A merged anomaly record across detectors at a single (timestamp, location).

    Mirrors the ``anomalies`` DB table; per-detector raw outputs are preserved
    on the object so downstream enrichment + research analyses can inspect
    detector-level signals without re-running detection.
    """

    timestamp: datetime
    lat: float
    lon: float
    metric: str
    source: str
    value: float
    expected_value: float | None = None
    z_score: float | None = None
    methods_triggered: list[str] = Field(default_factory=list)
    severity: Severity

    zscore_detail: ZScoreAnomaly | None = None
    stl_detail: STLAnomaly | None = None
    isolation_forest_detail: IsolationForestAnomaly | None = None


def severity_for(n_methods: int) -> Severity:
    """Map the count of triggering methods to a severity tier."""
    if n_methods not in _SEVERITY_BY_COUNT:
        raise ValueError(
            f"n_methods must be 1, 2, or 3; got {n_methods}. "
            "Consensus is undefined when zero methods triggered."
        )
    return _SEVERITY_BY_COUNT[n_methods]


def merge(
    *,
    metric: str,
    source: str,
    lat: float,
    lon: float,
    zscore_results: Sequence[ZScoreAnomaly] = (),
    stl_results: Sequence[STLAnomaly] = (),
    isolation_forest_results: Sequence[IsolationForestAnomaly] = (),
) -> list[ConsensusAnomaly]:
    """Group detector outputs by timestamp and emit unified anomaly records.

    All three detector result sequences share the same (metric, source, lat,
    lon) identity — the caller is responsible for filtering inputs to a single
    station/grid cell. Matching across detectors is by exact timestamp, which
    is valid because all three detectors read from the same DataPoint series.
    """
    by_ts: dict[datetime, _Bucket] = {}

    for z in zscore_results:
        _bucket_for(by_ts, z.timestamp).zscore = z
    for s in stl_results:
        _bucket_for(by_ts, s.timestamp).stl = s
    for i in isolation_forest_results:
        _bucket_for(by_ts, i.timestamp).isolation_forest = i

    merged: list[ConsensusAnomaly] = []
    for ts in sorted(by_ts):
        bucket = by_ts[ts]
        merged.append(
            _build_consensus_anomaly(
                ts=ts,
                bucket=bucket,
                metric=metric,
                source=source,
                lat=lat,
                lon=lon,
            )
        )
    return merged


class _Bucket:
    """Internal mutable accumulator: detector outputs at a single timestamp."""

    __slots__ = ("zscore", "stl", "isolation_forest")

    def __init__(self) -> None:
        self.zscore: ZScoreAnomaly | None = None
        self.stl: STLAnomaly | None = None
        self.isolation_forest: IsolationForestAnomaly | None = None

    def methods(self) -> list[str]:
        out: list[str] = []
        if self.zscore is not None:
            out.append("zscore")
        if self.stl is not None:
            out.append("stl")
        if self.isolation_forest is not None:
            out.append("isolation_forest")
        return out


def _bucket_for(by_ts: dict[datetime, _Bucket], ts: datetime) -> _Bucket:
    bucket = by_ts.get(ts)
    if bucket is None:
        bucket = _Bucket()
        by_ts[ts] = bucket
    return bucket


def _build_consensus_anomaly(
    *,
    ts: datetime,
    bucket: _Bucket,
    metric: str,
    source: str,
    lat: float,
    lon: float,
) -> ConsensusAnomaly:
    methods = bucket.methods()
    severity = severity_for(len(methods))
    value = _resolve_value(bucket)
    expected_value = _resolve_expected_value(bucket)
    z_score = bucket.zscore.z_score if bucket.zscore is not None else None
    return ConsensusAnomaly(
        timestamp=ts,
        lat=lat,
        lon=lon,
        metric=metric,
        source=source,
        value=value,
        expected_value=expected_value,
        z_score=z_score,
        methods_triggered=methods,
        severity=severity,
        zscore_detail=bucket.zscore,
        stl_detail=bucket.stl,
        isolation_forest_detail=bucket.isolation_forest,
    )


def _resolve_value(bucket: _Bucket) -> float:
    """Pick the observed value and verify detectors agree.

    All detectors share the same source data, so any detector that flagged a
    timestamp must report the same ``value``. A disagreement signals that the
    caller fed mismatched inputs to different detectors — surface it loudly.
    """
    candidates: list[tuple[str, float]] = []
    if bucket.zscore is not None:
        candidates.append(("zscore", bucket.zscore.value))
    if bucket.stl is not None:
        candidates.append(("stl", bucket.stl.value))
    if bucket.isolation_forest is not None:
        candidates.append(("isolation_forest", bucket.isolation_forest.value))

    if not candidates:
        # Should be unreachable — a bucket exists only because a detector
        # flagged the timestamp. Defensive guard to fail loudly if invariants
        # ever break.
        raise RuntimeError("consensus bucket has no detector results")

    reference_name, reference_value = candidates[0]
    for name, v in candidates[1:]:
        if v != reference_value:
            raise ValueError(
                "inconsistent values across detectors at the same timestamp: "
                f"{reference_name}={reference_value} vs {name}={v}. "
                "Detectors must read the same DataPoint series."
            )
    return reference_value


def _resolve_expected_value(bucket: _Bucket) -> float | None:
    """Z-score's expected_value wins over STL's; IF doesn't provide one."""
    if bucket.zscore is not None:
        return bucket.zscore.expected_value
    if bucket.stl is not None:
        return bucket.stl.expected_value
    return None
