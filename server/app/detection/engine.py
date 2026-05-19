"""Detection orchestrator.

Runs the configured detectors (Z-score, STL, IsolationForest) over a single
``(metric, source, lat, lon)`` time-series and routes their outputs through
:func:`app.detection.consensus.merge` to produce unified
:class:`ConsensusAnomaly` records.

The engine is a pure compute layer — it does not read from or write to the
database. Callers gather the input series (and any IsolationForest aux
features) from ``data_points`` and persist the returned records themselves.
This keeps the engine unit-testable against synthetic data and lets the API
layer decide how to scope queries.
"""

from datetime import datetime

from .consensus import ConsensusAnomaly, merge
from .isolation_forest import IsolationForestDetector, IsolationForestInput
from .stl import STLDetector
from .zscore import ZScoreDetector


class DetectionEngine:
    """Orchestrates one or more detectors over a single time-series.

    Detectors are optional at construction time — passing ``None`` (the
    default for each argument) disables that method. Use
    :meth:`with_defaults` for the standard 3-detector configuration.
    """

    def __init__(
        self,
        zscore: ZScoreDetector | None = None,
        stl: STLDetector | None = None,
        isolation_forest: IsolationForestDetector | None = None,
    ) -> None:
        if zscore is None and stl is None and isolation_forest is None:
            raise ValueError(
                "DetectionEngine requires at least one detector. "
                "Use DetectionEngine.with_defaults() for the standard "
                "3-detector configuration."
            )
        self.zscore = zscore
        self.stl = stl
        self.isolation_forest = isolation_forest

    @classmethod
    def with_defaults(cls) -> "DetectionEngine":
        """Standard 3-detector engine with each detector's library defaults."""
        return cls(
            zscore=ZScoreDetector(),
            stl=STLDetector(),
            isolation_forest=IsolationForestDetector(),
        )

    def run(
        self,
        series: list[tuple[datetime, float]],
        *,
        metric: str,
        source: str,
        lat: float,
        lon: float,
        isolation_forest_inputs: list[IsolationForestInput] | None = None,
    ) -> list[ConsensusAnomaly]:
        """Run all configured detectors on ``series`` and merge via consensus.

        ``series`` is the univariate primary metric for Z-score and STL.
        ``isolation_forest_inputs`` is the multivariate feature stream IF
        consumes; when omitted, IF is skipped even if the detector is
        configured (the engine cannot fabricate aux features).
        """
        zscore_results = (
            self.zscore.detect(series) if self.zscore is not None else []
        )
        stl_results = self.stl.detect(series) if self.stl is not None else []
        if self.isolation_forest is not None and isolation_forest_inputs is not None:
            isolation_forest_results = self.isolation_forest.detect(
                isolation_forest_inputs
            )
        else:
            isolation_forest_results = []

        return merge(
            metric=metric,
            source=source,
            lat=lat,
            lon=lon,
            zscore_results=zscore_results,
            stl_results=stl_results,
            isolation_forest_results=isolation_forest_results,
        )
