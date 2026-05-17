"""Rolling 7-day Z-score detector with configurable per-metric threshold (default +/- 3 sigma)."""

from datetime import datetime, timedelta

import numpy as np
from pydantic import BaseModel


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
        anomalies: list[ZScoreAnomaly] = []

        for ts_i, value_i in sorted_series:
            cutoff = ts_i - self.window
            window_values = [v for ts, v in sorted_series if cutoff <= ts < ts_i]
            n = len(window_values)
            if n < self.min_points:
                continue

            arr = np.asarray(window_values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))
            if std == 0:
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
