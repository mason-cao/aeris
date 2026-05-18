"""STL decomposition detector (statsmodels); flags residuals beyond +/- 2.5 sigma."""

from datetime import datetime

import numpy as np
from pydantic import BaseModel
from statsmodels.tsa.seasonal import STL

_STD_EPS = 1e-9


class STLAnomaly(BaseModel):
    timestamp: datetime
    value: float
    expected_value: float
    residual: float
    residual_score: float
    period: int


class STLDetector:
    def __init__(
        self,
        period: int = 24,
        threshold: float = 2.5,
        min_points: int | None = None,
        robust: bool = True,
    ) -> None:
        if period < 2:
            raise ValueError("period must be >= 2 for seasonal decomposition")
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        floor = 2 * period + 1
        if min_points is None:
            min_points = floor
        elif min_points < floor:
            raise ValueError(
                f"min_points must be >= 2*period+1 ({floor}); got {min_points}"
            )
        self.period = period
        self.threshold = threshold
        self.min_points = min_points
        self.robust = robust

    def detect(
        self, series: list[tuple[datetime, float]]
    ) -> list[STLAnomaly]:
        if len(series) < self.min_points:
            return []

        sorted_series = sorted(series, key=lambda p: p[0])
        values = np.asarray([v for _, v in sorted_series], dtype=float)

        # Degenerate case: zero variance input. STL would produce noise via
        # its LOESS smoothers; short-circuit to avoid spurious flags.
        if float(values.std(ddof=0)) < _STD_EPS:
            return []

        fit = STL(values, period=self.period, robust=self.robust).fit()
        resid = np.asarray(fit.resid, dtype=float)
        trend = np.asarray(fit.trend, dtype=float)
        seasonal = np.asarray(fit.seasonal, dtype=float)

        resid_std = float(resid.std(ddof=0))
        if resid_std < _STD_EPS:
            return []

        anomalies: list[STLAnomaly] = []
        for i, (ts_i, value_i) in enumerate(sorted_series):
            residual = float(resid[i])
            score = residual / resid_std
            if abs(score) > self.threshold:
                anomalies.append(
                    STLAnomaly(
                        timestamp=ts_i,
                        value=value_i,
                        expected_value=float(trend[i] + seasonal[i]),
                        residual=residual,
                        residual_score=score,
                        period=self.period,
                    )
                )
        return anomalies
