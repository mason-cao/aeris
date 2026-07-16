"""Derive and load the B6/B9 OpenAQ regulatory-monitor provenance fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Final, TypeVar

LOCKED_SNAPSHOT_SHA256: Final = (
    "8ec0bfacec592b50a31aafb9e80f61e886cfb48da030d595e89bdc0f53f9ea81"
)
STUDY_START: Final = "2026-06-01T00:00:00Z"
STUDY_END_EXCLUSIVE: Final = "2026-07-13T00:00:00Z"
FIXTURE_PATH: Final = (
    Path(__file__).parent
    / "fixtures"
    / "openaq_regulatory_entity_provenance.v2.json"
)
NOMINATING_METRICS: Final = ("ozone", "pm10", "pm25")

VERIFIED_MONITOR: Final = "verified_monitor"
NON_MONITOR_SENSOR: Final = "non_monitor_sensor"
UNMAPPABLE_ARCHIVE: Final = "unmappable_archive"

_INCLUDE_DISPOSITION: Final = "include_regulatory_ground_monitor"
_EXCLUDE_DISPOSITION: Final = "exclude_regulatory_ground_monitor"
_NON_MONITOR_PROVIDERS: Final = frozenset({"AirGradient", "Clarity"})
_MetadataValue = TypeVar("_MetadataValue")


@dataclass
class _EntityAccumulator:
    """Metadata and row counts observed for one retained OpenAQ sensor ID."""

    providers: set[str] = field(default_factory=set)
    monitor_flags: set[bool] = field(default_factory=set)
    instrument_set_counts: dict[tuple[str, ...], int] = field(default_factory=dict)
    inline_rows: int = 0
    archive_rows: int = 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_mapping(value: object, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


def _instrument_names(location: Mapping[str, Any]) -> tuple[str, ...]:
    raw_instruments = location.get("instruments")
    if not isinstance(raw_instruments, list):
        raise ValueError("OpenAQ location.instruments must be a list")
    names: list[str] = []
    for raw_instrument in raw_instruments:
        instrument = _as_mapping(
            raw_instrument, field_name="OpenAQ location instrument"
        )
        name = instrument.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("OpenAQ location instrument name is missing")
        names.append(name)
    if not names:
        raise ValueError("OpenAQ location has no retained instrument metadata")
    return tuple(sorted(set(names)))


def _monitor_flag(value: object) -> bool:
    if value is True or value == 1:
        return True
    if value is False or value == 0:
        return False
    raise ValueError(f"OpenAQ isMonitor is not boolean: {value!r}")


def _single_metadata_value(
    values: set[_MetadataValue], *, entity_id: str, field_name: str
) -> _MetadataValue:
    if len(values) != 1:
        raise ValueError(
            f"OpenAQ entity {entity_id} has conflicting {field_name}: "
            f"{sorted(values, key=repr)!r}"
        )
    return next(iter(values))


def _classification(
    *,
    entity_id: str,
    provider: str,
    is_monitor: bool,
    instrument_sets: tuple[tuple[str, ...], ...],
) -> tuple[str, str]:
    if (
        provider == "AirNow"
        and is_monitor
        and instrument_sets == (("Government Monitor",),)
    ):
        return VERIFIED_MONITOR, _INCLUDE_DISPOSITION
    if provider in _NON_MONITOR_PROVIDERS and not is_monitor:
        return NON_MONITOR_SENSOR, _EXCLUDE_DISPOSITION
    raise ValueError(
        "OpenAQ entity "
        f"{entity_id} has an undeclared provider/instrument combination: "
        f"provider={provider!r}, isMonitor={is_monitor!r}, "
        f"instrument_sets={instrument_sets!r}"
    )


def _count_rows(
    rows: Sequence[dict[str, Any]],
    *,
    key: str,
    value: str,
) -> dict[str, int]:
    selected = [row for row in rows if row[key] == value]
    return {
        "rows": sum(
            int(row["inline_rows"]) + int(row["archive_rows"])
            for row in selected
        ),
        "entities": len(selected),
    }


def _entity_sort_key(entity_id: str) -> tuple[bool, int | str]:
    return (
        not entity_id.isdigit(),
        int(entity_id) if entity_id.isdigit() else entity_id,
    )


def _build_metric_entities(
    accumulators: Mapping[str, _EntityAccumulator],
    *,
    metric: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    entities: list[dict[str, Any]] = []
    unmappable_archive_entities: list[dict[str, Any]] = []
    instrument_metadata_revisions: list[dict[str, Any]] = []
    for entity_id in sorted(accumulators, key=_entity_sort_key):
        accumulator = accumulators[entity_id]
        if accumulator.inline_rows == 0:
            row = {
                "metric": metric,
                "entity_id": entity_id,
                "provider": None,
                "is_monitor": None,
                "instrument_names": [],
                "classification": UNMAPPABLE_ARCHIVE,
                "disposition": _EXCLUDE_DISPOSITION,
                "inline_rows": 0,
                "archive_rows": accumulator.archive_rows,
            }
            unmappable_archive_entities.append(
                {
                    "metric": metric,
                    "entity_id": entity_id,
                    "archive_rows": accumulator.archive_rows,
                }
            )
        else:
            provider = _single_metadata_value(
                accumulator.providers,
                entity_id=entity_id,
                field_name=f"{metric} provider metadata",
            )
            is_monitor = _single_metadata_value(
                accumulator.monitor_flags,
                entity_id=entity_id,
                field_name=f"{metric} isMonitor metadata",
            )
            instrument_sets = tuple(sorted(accumulator.instrument_set_counts))
            instrument_names = tuple(
                sorted({name for names in instrument_sets for name in names})
            )
            classification, disposition = _classification(
                entity_id=entity_id,
                provider=provider,
                is_monitor=is_monitor,
                instrument_sets=instrument_sets,
            )
            if len(instrument_sets) > 1:
                instrument_metadata_revisions.append(
                    {
                        "metric": metric,
                        "entity_id": entity_id,
                        "provider": provider,
                        "is_monitor": is_monitor,
                        "observed_instrument_name_sets": [
                            {
                                "instrument_names": list(names),
                                "rows": accumulator.instrument_set_counts[names],
                            }
                            for names in instrument_sets
                        ],
                    }
                )
            row = {
                "metric": metric,
                "entity_id": entity_id,
                "provider": provider,
                "is_monitor": is_monitor,
                "instrument_names": list(instrument_names),
                "classification": classification,
                "disposition": disposition,
                "inline_rows": accumulator.inline_rows,
                "archive_rows": accumulator.archive_rows,
            }
        entities.append(row)
    return entities, unmappable_archive_entities, instrument_metadata_revisions


def _audit_counts(entities: Sequence[dict[str, Any]]) -> dict[str, Any]:
    inline_by_provider: dict[str, dict[str, int]] = {}
    archive_mapped_by_provider: dict[str, dict[str, int]] = {}
    for provider in sorted(
        {str(row["provider"]) for row in entities if row["provider"] is not None}
    ):
        provider_rows = [row for row in entities if row["provider"] == provider]
        inline_by_provider[provider] = {
            "rows": sum(int(row["inline_rows"]) for row in provider_rows),
            "entities": sum(
                1 for row in provider_rows if int(row["inline_rows"]) > 0
            ),
        }
        mapped_archive_rows = [
            row for row in provider_rows if int(row["archive_rows"]) > 0
        ]
        archive_mapped_by_provider[provider] = {
            "rows": sum(int(row["archive_rows"]) for row in mapped_archive_rows),
            "entities": len(mapped_archive_rows),
        }
    archive_entities = [row for row in entities if int(row["archive_rows"]) > 0]
    return {
        "inline_by_provider": inline_by_provider,
        "archive_without_inline_metadata": {
            "rows": sum(int(row["archive_rows"]) for row in archive_entities),
            "entities": len(archive_entities),
        },
        "archive_mapped_by_provider": archive_mapped_by_provider,
        "resolved_by_class": {
            classification: _count_rows(
                entities, key="classification", value=classification
            )
            for classification in (
                VERIFIED_MONITOR,
                NON_MONITOR_SENSOR,
                UNMAPPABLE_ARCHIVE,
            )
        },
    }


def derive_openaq_pm25_fixture(
    database: Path,
    *,
    expected_sha256: str = LOCKED_SNAPSHOT_SHA256,
) -> dict[str, Any]:
    """Derive one immutable-window table for every B9 OpenAQ metric."""
    before_hash = _sha256(database)
    if before_hash != expected_sha256:
        raise ValueError(
            f"snapshot SHA-256 mismatch before read: {before_hash} != {expected_sha256}"
        )

    accumulators: dict[tuple[str, str], _EntityAccumulator] = defaultdict(
        _EntityAccumulator
    )
    uri = f"{database.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(
            """
            SELECT metric, source_entity_id, raw_json
            FROM data_points
            WHERE source = 'openaq'
              AND metric IN ('ozone', 'pm10', 'pm25')
              AND datetime(timestamp) >= datetime(?)
              AND datetime(timestamp) < datetime(?)
            ORDER BY metric, source_entity_id, timestamp, id
            """,
            (STUDY_START, STUDY_END_EXCLUSIVE),
        )
        for raw_metric, raw_entity_id, raw_json in cursor:
            metric = str(raw_metric)
            entity_id = str(raw_entity_id)
            try:
                payload = json.loads(raw_json)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"OpenAQ entity {entity_id} has invalid raw_json"
                ) from exc
            raw = _as_mapping(payload, field_name="OpenAQ raw_json")
            accumulator = accumulators[(metric, entity_id)]
            if "location" in raw:
                location = _as_mapping(
                    raw["location"], field_name="OpenAQ raw_json.location"
                )
                provider = _as_mapping(
                    location.get("provider"), field_name="OpenAQ location.provider"
                ).get("name")
                if not isinstance(provider, str) or not provider:
                    raise ValueError(
                        f"OpenAQ entity {entity_id} has no provider name"
                    )
                accumulator.providers.add(provider)
                accumulator.monitor_flags.add(_monitor_flag(location.get("isMonitor")))
                instrument_names = _instrument_names(location)
                accumulator.instrument_set_counts[instrument_names] = (
                    accumulator.instrument_set_counts.get(instrument_names, 0) + 1
                )
                accumulator.inline_rows += 1
            elif "archive" in raw:
                accumulator.archive_rows += 1
            else:
                raise ValueError(
                    f"OpenAQ entity {entity_id} row has neither location nor "
                    "archive metadata"
                )
    finally:
        connection.close()
        after_hash = _sha256(database)
        if after_hash != expected_sha256:
            raise RuntimeError(
                "snapshot SHA-256 mismatch after read: "
                f"{after_hash} != {expected_sha256}"
            )

    metric_entities: list[dict[str, Any]] = []
    all_unmappable: list[dict[str, Any]] = []
    all_revisions: list[dict[str, Any]] = []
    audit_counts_by_metric: dict[str, dict[str, Any]] = {}
    for metric in NOMINATING_METRICS:
        metric_accumulators = {
            entity_id: accumulator
            for (row_metric, entity_id), accumulator in accumulators.items()
            if row_metric == metric
        }
        entities, unmappable, revisions = _build_metric_entities(
            metric_accumulators,
            metric=metric,
        )
        metric_entities.extend(entities)
        all_unmappable.extend(unmappable)
        all_revisions.extend(revisions)
        audit_counts_by_metric[metric] = _audit_counts(entities)

    pm25_entities = [
        row for row in metric_entities if row["metric"] == "pm25"
    ]
    pm25_unmappable = [
        {key: value for key, value in row.items() if key != "metric"}
        for row in all_unmappable
        if row["metric"] == "pm25"
    ]
    pm25_revisions = [
        {key: value for key, value in row.items() if key != "metric"}
        for row in all_revisions
        if row["metric"] == "pm25"
    ]
    eligible_entity_ids_by_metric = {
        metric: sorted(
            str(row["entity_id"])
            for row in metric_entities
            if row["metric"] == metric
            and row["classification"] == VERIFIED_MONITOR
        )
        for metric in NOMINATING_METRICS
    }

    return {
        "schema_version": 2,
        "fixture_id": "openaq-regulatory-entity-provenance-v2",
        "snapshot_sha256": expected_sha256,
        "study_window": {
            "start": STUDY_START,
            "end_exclusive": STUDY_END_EXCLUSIVE,
        },
        "source": "openaq",
        "nominating_metrics": list(NOMINATING_METRICS),
        "classification_rule": {
            "entity_key": "source_entity_id (OpenAQ sensor ID)",
            "verified_monitor": (
                "provider=AirNow AND isMonitor=true AND "
                "instrument_names=[Government Monitor]"
            ),
            "non_monitor_sensor": (
                "provider IN [AirGradient, Clarity] AND isMonitor=false"
            ),
            "archive_mapping": "exact (metric, source_entity_id) equality",
            "unmappable_archive_disposition": _EXCLUDE_DISPOSITION,
        },
        "eligible_entity_ids_by_metric": eligible_entity_ids_by_metric,
        "eligible_entity_counts": {
            metric: len(entity_ids)
            for metric, entity_ids in eligible_entity_ids_by_metric.items()
        },
        "audit_counts_by_metric": audit_counts_by_metric,
        "metric_entities": metric_entities,
        "instrument_metadata_revisions_by_metric": all_revisions,
        "unmappable_archive_entities_by_metric": all_unmappable,
        "metric": "pm25",
        "audit_counts": audit_counts_by_metric["pm25"],
        "entities": pm25_entities,
        "instrument_metadata_revisions": pm25_revisions,
        "unmappable_archive_entities": pm25_unmappable,
    }


def write_openaq_pm25_fixture(payload: Mapping[str, Any], output: Path) -> None:
    """Write a deterministic JSON representation of a derived fixture."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


@lru_cache(maxsize=1)
def load_openaq_pm25_fixture() -> dict[str, Any]:
    """Load and validate the one active multi-metric B6/B9 fixture."""
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    fixture = dict(_as_mapping(payload, field_name="OpenAQ provenance fixture"))
    if fixture.get("schema_version") != 2:
        raise ValueError("unsupported OpenAQ provenance fixture schema")
    if fixture.get("fixture_id") != "openaq-regulatory-entity-provenance-v2":
        raise ValueError("unexpected OpenAQ provenance fixture ID")
    if fixture.get("snapshot_sha256") != LOCKED_SNAPSHOT_SHA256:
        raise ValueError("OpenAQ fixture does not match the locked snapshot")
    if fixture.get("nominating_metrics") != list(NOMINATING_METRICS):
        raise ValueError("OpenAQ fixture nominating metrics do not match B9")
    raw_entities = fixture.get("metric_entities")
    if not isinstance(raw_entities, list):
        raise ValueError("OpenAQ fixture metric_entities must be a list")
    entity_keys: set[tuple[str, str]] = set()
    computed_eligible: dict[str, list[str]] = {
        metric: [] for metric in NOMINATING_METRICS
    }
    for raw_entity in raw_entities:
        entity = _as_mapping(raw_entity, field_name="OpenAQ fixture entity")
        metric = entity.get("metric")
        if metric not in NOMINATING_METRICS:
            raise ValueError(f"OpenAQ fixture entity has invalid metric: {metric}")
        entity_id = entity.get("entity_id")
        if not isinstance(entity_id, str) or not entity_id:
            raise ValueError("OpenAQ fixture entity ID is missing")
        entity_key = (metric, entity_id)
        if entity_key in entity_keys:
            raise ValueError(f"duplicate OpenAQ fixture entity: {entity_key}")
        entity_keys.add(entity_key)
        classification = entity.get("classification")
        if classification not in {
            VERIFIED_MONITOR,
            NON_MONITOR_SENSOR,
            UNMAPPABLE_ARCHIVE,
        }:
            raise ValueError(
                f"OpenAQ fixture entity {metric}/{entity_id} has invalid class"
            )
        if classification == VERIFIED_MONITOR and (
            entity.get("provider") != "AirNow"
            or entity.get("is_monitor") is not True
            or entity.get("instrument_names") != ["Government Monitor"]
            or entity.get("disposition") != _INCLUDE_DISPOSITION
        ):
            raise ValueError(
                f"OpenAQ fixture entity {metric}/{entity_id} is not verified"
            )
        if classification == VERIFIED_MONITOR:
            computed_eligible[metric].append(entity_id)
        if classification != VERIFIED_MONITOR and (
            entity.get("disposition") != _EXCLUDE_DISPOSITION
        ):
            raise ValueError(
                f"OpenAQ fixture entity {metric}/{entity_id} is not excluded"
            )
    computed_eligible = {
        metric: sorted(entity_ids)
        for metric, entity_ids in computed_eligible.items()
    }
    if fixture.get("eligible_entity_ids_by_metric") != computed_eligible:
        raise ValueError("OpenAQ fixture eligible entity allowlists are inconsistent")
    if fixture.get("eligible_entity_counts") != {
        metric: len(entity_ids)
        for metric, entity_ids in computed_eligible.items()
    }:
        raise ValueError("OpenAQ fixture eligible entity counts are inconsistent")
    return fixture


@lru_cache(maxsize=None)
def verified_monitor_entity_ids(metric: str = "pm25") -> frozenset[str]:
    """Exact OpenAQ entity allowlist for one B9 nominating metric."""
    if metric not in NOMINATING_METRICS:
        return frozenset()
    fixture = load_openaq_pm25_fixture()
    return frozenset(fixture["eligible_entity_ids_by_metric"][metric])


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive the active B6/B9 OpenAQ entity-provenance fixture."
    )
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument(
        "--expected-sha256", default=LOCKED_SNAPSHOT_SHA256
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; writes a fixture or prints it to standard output."""
    args = _parse_args(argv)
    payload = derive_openaq_pm25_fixture(
        args.database, expected_sha256=args.expected_sha256
    )
    if args.output is None:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        write_openaq_pm25_fixture(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
