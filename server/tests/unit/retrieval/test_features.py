from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from app.retrieval.features import (
    FEATURE_NAMES,
    SCHEMA_VERSION,
    EventFeatures,
    FeatureError,
    Scaling,
    epoch_seconds,
    fit_scaling,
    held_out_values,
    raw_features,
    standardize,
    wind_components,
)
from tests.unit.retrieval._events import summary_with, varied_values

UTC = timezone.utc
T0 = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _features(summary: dict | None = None, *, when: datetime = T0, severity: str = "minor") -> EventFeatures:
    return raw_features(
        timestamp=when, lat=29.7, lon=-95.3, severity=severity, summary=summary or summary_with()
    )


def _value(features: EventFeatures, name: str) -> float | None:
    return features.values[FEATURE_NAMES.index(name)]


def _varied(variant: int, *, severity: str, overrides: dict | None = None) -> EventFeatures:
    """Events that differ on every feature, so scaling can be fitted."""
    return raw_features(
        timestamp=datetime(2026, 7, 1, 3 * variant, 0, tzinfo=UTC),
        lat=29.6 + 0.05 * variant,
        lon=-95.4 + 0.05 * variant,
        severity=severity,
        summary=summary_with(overrides, variant=variant),
    )


def test_wind_from_north_blows_south() -> None:
    assert wind_components(5.0, 360.0) == pytest.approx((0.0, -5.0), abs=1e-12)


def test_wind_from_east_blows_west() -> None:
    assert wind_components(5.0, 90.0) == pytest.approx((-5.0, 0.0), abs=1e-12)


def test_calm_is_zero_whatever_the_direction() -> None:
    assert wind_components(0.0, 270.0) == (0.0, 0.0)


def test_hour_encoding_wraps_midnight() -> None:
    def hour_point(hour: int) -> tuple[float, float]:
        f = _features(when=datetime(2026, 7, 1, hour, 0, tzinfo=UTC))
        return (_value(f, "hour_sin"), _value(f, "hour_cos"))  # type: ignore[return-value]

    late, midnight, noon = hour_point(23), hour_point(0), hour_point(12)
    assert math.dist(late, midnight) < math.dist(midnight, noon)


def test_log_transforms() -> None:
    f = _features(summary_with({("noaa_gfs", "pbl_height"): 500.0, ("openweather", "precipitation"): 2.0}))
    assert _value(f, "ln_pbl_height") == pytest.approx(math.log(500.0))
    assert _value(f, "ln1p_precipitation") == pytest.approx(math.log1p(2.0))


def test_missing_asos_is_imputable() -> None:
    summary = summary_with({key: None for key in [
        ("asos", "wind_speed"), ("asos", "wind_direction"), ("asos", "temperature"), ("asos", "humidity"),
    ]})
    assert _features(summary).imputed == ("wind_u", "wind_v", "temperature", "humidity")


def test_missing_wind_direction_drops_both_components() -> None:
    f = _features(summary_with({("asos", "wind_direction"): None}))
    assert f.imputed == ("wind_u", "wind_v")


@pytest.mark.parametrize(
    "key",
    [("noaa_gfs", "pbl_height"), ("openweather", "cloud_cover"), ("openweather", "precipitation")],
)
def test_missing_non_asos_feature_raises(key: tuple[str, str]) -> None:
    with pytest.raises(FeatureError):
        _features(summary_with({key: None}))


def test_nonpositive_pbl_raises() -> None:
    with pytest.raises(FeatureError):
        _features(summary_with({("noaa_gfs", "pbl_height"): 0.0}))


def test_negative_precipitation_raises() -> None:
    with pytest.raises(FeatureError):
        _features(summary_with({("openweather", "precipitation"): -0.1}))


def test_severity_tiers() -> None:
    assert _value(_features(severity="severe"), "severity_tier") == 3.0
    with pytest.raises(FeatureError):
        _features(severity="extreme")


def test_non_finite_value_raises() -> None:
    with pytest.raises(FeatureError):
        raw_features(timestamp=T0, lat=float("nan"), lon=-95.3, severity="minor", summary=summary_with())


def test_naive_timestamp_is_utc() -> None:
    naive = datetime(2026, 7, 1, 12, 0)
    assert _features(when=naive) == _features(when=T0)


def test_epoch_rounding_is_strict_in_both_directions() -> None:
    moment = datetime(2026, 7, 1, 12, 0, 0, 500000, tzinfo=UTC)
    whole = int(datetime(2026, 7, 1, 12, 0, tzinfo=UTC).timestamp())
    assert epoch_seconds(moment, round_up=False) == whole
    assert epoch_seconds(moment, round_up=True) == whole + 1


def test_fit_and_standardize() -> None:
    a = _varied(1, severity="minor")
    b = _varied(2, severity="severe")
    scaling = fit_scaling([a, b])
    i = FEATURE_NAMES.index("temperature")
    assert varied_values(1)[("asos", "temperature")] == 26.0
    assert scaling.means[i] == pytest.approx(26.5)
    assert scaling.stds[i] == pytest.approx(0.5)
    assert standardize(a, scaling)[i] == pytest.approx(-1.0)


def test_imputed_values_standardize_to_zero() -> None:
    scaling = fit_scaling([_varied(1, severity="minor"), _varied(2, severity="severe")])
    gap = _varied(3, severity="moderate", overrides={("asos", "temperature"): None})
    assert gap.imputed == ("temperature",)
    assert standardize(gap, scaling)[FEATURE_NAMES.index("temperature")] == 0.0


def test_zero_spread_raises() -> None:
    with pytest.raises(FeatureError):
        fit_scaling([_features(), _features()])


def test_scaling_json_round_trip_and_stable_hash() -> None:
    scaling = fit_scaling([_varied(1, severity="minor"), _varied(2, severity="severe")])
    again = Scaling.from_json(scaling.to_json())
    assert again == scaling
    assert again.sha256 == scaling.sha256


def test_scaling_rejects_another_schema() -> None:
    text = fit_scaling([_varied(1, severity="minor"), _varied(2, severity="severe")]).to_json()
    with pytest.raises(FeatureError):
        Scaling.from_json(text.replace(SCHEMA_VERSION, "analog-v0"))


def test_held_out_values_in_declared_order() -> None:
    assert held_out_values(summary_with()) == (290.0, 101000.0, 40.0)
    with pytest.raises(FeatureError):
        held_out_values(summary_with({("noaa_gfs", "t_850"): None}))
