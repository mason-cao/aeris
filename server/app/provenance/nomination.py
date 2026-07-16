"""B9 regulatory-monitor anomaly-nomination eligibility."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

from app.provenance.openaq_pm25 import (
    FIXTURE_PATH,
    LOCKED_SNAPSHOT_SHA256,
    load_openaq_pm25_fixture,
    verified_monitor_entity_ids,
)

OPENAQ_NOMINATING_METRICS: Final = frozenset({"pm25", "pm10", "ozone"})
TCEQ_NOMINATING_METRICS: Final = frozenset({"no2", "so2", "co"})
NOMINATING_METRICS_BY_SOURCE: Final = {
    "openaq": OPENAQ_NOMINATING_METRICS,
    "tceq": TCEQ_NOMINATING_METRICS,
}
STRICT_ELEVATION_RULE: Final = "value > expected_value"


def validate_nominating_metric_disjointness() -> None:
    """Fail if two nominating sources can emit the same metric."""
    seen: dict[str, str] = {}
    for source in sorted(NOMINATING_METRICS_BY_SOURCE):
        for metric in sorted(NOMINATING_METRICS_BY_SOURCE[source]):
            prior_source = seen.get(metric)
            if prior_source is not None:
                raise ValueError(
                    "B9 nominating metric sets overlap: "
                    f"{metric} appears under {prior_source} and {source}"
                )
            seen[metric] = source


def series_is_nomination_eligible(
    source: str,
    metric: str,
    source_entity_id: str,
) -> bool:
    """Return whether one exact series may reach an anomaly detector."""
    if not source_entity_id:
        return False
    if source == "tceq":
        return metric in TCEQ_NOMINATING_METRICS
    if source == "openaq" and metric in OPENAQ_NOMINATING_METRICS:
        return source_entity_id in verified_monitor_entity_ids(metric)
    return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nomination_manifest_payload() -> dict[str, object]:
    """Hash-link the active B6 fixture and complete B9 rule into a freeze."""
    validate_nominating_metric_disjointness()
    fixture = load_openaq_pm25_fixture()
    if fixture["snapshot_sha256"] != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("B9 OpenAQ fixture does not match the locked snapshot")
    return {
        "fixture_id": fixture["fixture_id"],
        "schema_version": fixture["schema_version"],
        "artifact": FIXTURE_PATH.name,
        "artifact_sha256": _sha256(FIXTURE_PATH),
        "snapshot_sha256": fixture["snapshot_sha256"],
        "covered_metrics": fixture["nominating_metrics"],
        "eligible_entity_counts": fixture["eligible_entity_counts"],
        "nominating_metrics_by_source": {
            source: sorted(metrics)
            for source, metrics in sorted(NOMINATING_METRICS_BY_SOURCE.items())
        },
        "strict_elevation_rule": STRICT_ELEVATION_RULE,
        "undefined_expected_value": "excluded",
        "entity_enforcement": "pre-detector exact series grouping",
        "eligible_sources": ["openaq", "tceq"],
        "excluded_sources": [
            "asos",
            "epa_aqs",
            "noaa_gfs",
            "openweather",
            "purpleair",
            "sentinel5p",
        ],
        "trigger_channel_consequence": (
            "ground_insitu trigger support is demoted for type-1 claims; "
            "positive non-circular evidence must come from another channel"
        ),
    }


validate_nominating_metric_disjointness()
