"""STL decomposition detector (statsmodels); flags residuals beyond a robust
+/- 2.5 sigma, with sigma estimated from the residuals' MAD."""

from datetime import datetime

import numpy as np
from pydantic import BaseModel
from statsmodels.tsa.seasonal import STL

_STD_EPS = 1e-9

# Scale a median-absolute-deviation up to a standard-deviation-equivalent so the
# same +/- threshold (in sigmas) applies regardless of which scale is used: for
# normal data, 1.4826 * MAD -> sigma.
_MAD_TO_SIGMA = 1.4826


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

        # Robust scale: a plain std over all residuals is inflated by the very
        # anomalies being scored, raising the effective threshold until smaller
        # real anomalies fall under it (masking). The MAD about the residual
        # median is unmoved by a minority of outliers, so each keeps the score
        # it deserves; center on the median for the same reason. The MAD only
        # degenerates to ~0 on a near-perfect (e.g. noiseless) fit, where it
        # carries no spread to normalize against — there we fall back to the
        # classic std so a lone outlier on an otherwise exact fit is still
        # scored rather than silently dropped.
        resid_median = float(np.median(resid))
        robust_scale = _MAD_TO_SIGMA * float(np.median(np.abs(resid - resid_median)))
        if robust_scale < _STD_EPS:
            robust_scale = float(resid.std(ddof=0))
        if robust_scale < _STD_EPS:
            return []

        anomalies: list[STLAnomaly] = []
        for i, (ts_i, value_i) in enumerate(sorted_series):
            residual = float(resid[i])
            score = (residual - resid_median) / robust_scale
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
