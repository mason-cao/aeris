"""Multivariate IsolationForest detector over (value, hour_of_day, day_of_week, aux features)."""

from datetime import datetime

import numpy as np
from pydantic import BaseModel, Field
from sklearn.ensemble import IsolationForest


class IsolationForestInput(BaseModel):
    timestamp: datetime
    value: float
    aux_features: dict[str, float] = Field(default_factory=dict)


class IsolationForestAnomaly(BaseModel):
    timestamp: datetime
    value: float
    decision_score: float
    anomaly_score: float
    features_used: list[str]


_INTRINSIC_FEATURES = ("value", "hour_of_day", "day_of_week")


class IsolationForestDetector:
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int | None = 42,
        min_points: int = 50,
    ) -> None:
        if not (0.0 < contamination <= 0.5):
            raise ValueError(
                "contamination must be in (0, 0.5]; sklearn IsolationForest "
                f"rejects values outside this range, got {contamination}"
            )
        if n_estimators <= 0:
            raise ValueError("n_estimators must be positive")
        if min_points < 2:
            raise ValueError("min_points must be >= 2")
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_points = min_points

    def detect(
        self, points: list[IsolationForestInput]
    ) -> list[IsolationForestAnomaly]:
        if len(points) < self.min_points:
            return []

        sorted_points = sorted(points, key=lambda p: p.timestamp)
        aux_keys = self._validated_aux_keys(sorted_points)
        feature_names = list(_INTRINSIC_FEATURES) + aux_keys

        X = self._build_feature_matrix(sorted_points, aux_keys)
        if not np.all(np.isfinite(X)):
            raise ValueError("all feature values must be finite (no NaN/inf)")

        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        labels = model.fit_predict(X)
        decision = model.decision_function(X)
        anomaly_scores = model.score_samples(X)

        anomalies: list[IsolationForestAnomaly] = []
        for i, point in enumerate(sorted_points):
            if labels[i] == -1:
                anomalies.append(
                    IsolationForestAnomaly(
                        timestamp=point.timestamp,
                        value=point.value,
                        decision_score=float(decision[i]),
                        anomaly_score=float(anomaly_scores[i]),
                        features_used=feature_names,
                    )
                )
        return anomalies

    @staticmethod
    def _validated_aux_keys(
        sorted_points: list[IsolationForestInput],
    ) -> list[str]:
        expected = set(sorted_points[0].aux_features.keys())
        for p in sorted_points[1:]:
            if set(p.aux_features.keys()) != expected:
                raise ValueError(
                    "all points must carry the same aux_features keys; "
                    f"expected {sorted(expected)}, "
                    f"saw {sorted(p.aux_features.keys())} at {p.timestamp}"
                )
        return sorted(expected)

    @staticmethod
    def _build_feature_matrix(
        sorted_points: list[IsolationForestInput],
        aux_keys: list[str],
    ) -> np.ndarray:
        rows: list[list[float]] = []
        for point in sorted_points:
            hour = float(point.timestamp.hour)
            dow = float(point.timestamp.weekday())
            row = [point.value, hour, dow]
            row.extend(point.aux_features[k] for k in aux_keys)
            rows.append(row)
        return np.asarray(rows, dtype=float)
