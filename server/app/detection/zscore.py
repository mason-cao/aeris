"""Rolling 7-day Z-score detector with configurable per-metric threshold (default +/- 3 sigma)."""

from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel


# Seven orders of magnitude above double-precision residue (~2e-16 relative)
# and seven below any real instrument's reporting precision, so it rejects
# quantization-flat windows without ever touching a genuinely varying series.
_FLAT_WINDOW_RELATIVE_TOLERANCE = 1e-9


class ZScoreAnomaly(BaseModel):
    timestamp: datetime
    value: float
    expected_value: float
    z_score: float
    window_n: int


class ZScoreDetector:
    def __init__(
        self,
        window: timedelta = timedelta(days=7),
        threshold: float = 3.0,
        min_points: int = 10,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        if min_points < 2:
            raise ValueError("min_points must be >= 2 for std computation")
        self.window = window
        self.threshold = threshold
        self.min_points = min_points

    def detect(
        self, series: list[tuple[datetime, float]]
    ) -> list[ZScoreAnomaly]:
        if not series:
            return []

        sorted_series = sorted(series, key=lambda p: p[0])
        times = [ts for ts, _ in sorted_series]
        values = np.asarray([v for _, v in sorted_series], dtype=float)
        anomalies: list[ZScoreAnomaly] = []

        # Two monotonic pointers replace the per-point full-series rescan: the
        # series is sorted, so as ts_i advances the half-open window [cutoff,
        # ts_i) only slides forward. `lo` tracks the inclusive cutoff bound and
        # `hi` the exclusive ts_i bound; both compare timestamps (not indices),
        # so same-timestamp siblings are excluded just as `cutoff <= ts < ts_i`
        # did. mean/std run on the contiguous slice for numerics identical to
        # the old per-window reduction.
        lo = 0
        hi = 0
        for i, (ts_i, value_i) in enumerate(sorted_series):
            cutoff = ts_i - self.window
            while lo < len(times) and times[lo] < cutoff:
                lo += 1
            while hi < len(times) and times[hi] < ts_i:
                hi += 1
            n = hi - lo
            if n < self.min_points:
                continue

            window = values[lo:hi]
            mean = float(window.mean())
            std = float(window.std(ddof=0))
            # Scaled, not `std == 0`. An exact test is defeated by float
            # residue: TCEQ reports CO at 0.1 ppm resolution, and np.std over a
            # constant window of 0.1 returns ~1.4e-17 rather than 0 because 0.1
            # is not binary-exact. The resulting z reaches 1e15 and wins any
            # ranking it enters. Scaling the floor to the window's own
            # magnitude keeps the test unit-free across ppm and molec/cm^2.
            scale = float(np.abs(window).max())
            if scale == 0.0 or std <= _FLAT_WINDOW_RELATIVE_TOLERANCE * scale:
                continue

            z = (value_i - mean) / std
            if abs(z) > self.threshold:
                anomalies.append(
                    ZScoreAnomaly(
                        timestamp=ts_i,
                        value=value_i,
                        expected_value=mean,
                        z_score=z,
                        window_n=n,
                    )
                )

        return anomalies
