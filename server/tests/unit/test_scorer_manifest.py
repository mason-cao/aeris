"""Live scorer-configuration manifest and its drift against a frozen fixture."""

from __future__ import annotations

import json

from app.eval.scorer_manifest import scorer_manifest_payload
from app.llm.corroboration import DEFAULT_WIND_TOLERANCE


def _fixture(tmp_path, thresholds: dict, **extra) -> "object":
    path = tmp_path / "eval.json"
    path.write_text(
        json.dumps(
            {
                "thresholds": thresholds,
                "snapshot_sha256": extra.get("snapshot_sha256", "x" * 64),
                "code_commit": "deadbeef",
                "frozen_at": "2026-08-05T00:00:00+00:00",
            }
        )
    )
    return path


def test_manifest_records_the_live_guards() -> None:
    payload = scorer_manifest_payload()

    assert payload["guards"]["wind_disagreement"]["threshold_deg"] == 90.0
    assert payload["guards"]["calm_wind"]["floor_ms"] == 1.5
    assert payload["thresholds"]["wind"]["max_disagreement_deg"] == (
        DEFAULT_WIND_TOLERANCE.max_disagreement_deg
    )
    assert payload["source_channels"]["asos"] == "met_insitu"
    # No fixture given, so there is nothing to diff against.
    assert "frozen_fixture" not in payload


def test_drift_detects_a_rule_added_after_the_freeze(tmp_path) -> None:
    live = scorer_manifest_payload()["thresholds"]
    stale = json.loads(json.dumps(live))
    del stale["wind"]["max_disagreement_deg"]

    payload = scorer_manifest_payload(frozen_fixture=_fixture(tmp_path, stale))
    drift = payload["frozen_fixture"]["threshold_drift_since_freeze"]

    assert drift["wind"]["added"] == {"max_disagreement_deg": 90.0}
    assert drift["wind"]["changed"] == {}
    assert "scorer rules, not selection rules" in payload["frozen_fixture"]["drift_note"]


def test_drift_detects_a_changed_value(tmp_path) -> None:
    live = scorer_manifest_payload()["thresholds"]
    stale = json.loads(json.dumps(live))
    stale["wind"]["bearing_deg"] = 30.0

    payload = scorer_manifest_payload(frozen_fixture=_fixture(tmp_path, stale))
    drift = payload["frozen_fixture"]["threshold_drift_since_freeze"]

    assert drift["wind"]["changed"]["bearing_deg"] == {"frozen": 30.0, "live": 45.0}


def test_no_drift_is_reported_plainly(tmp_path) -> None:
    live = scorer_manifest_payload()["thresholds"]

    payload = scorer_manifest_payload(frozen_fixture=_fixture(tmp_path, live))

    assert payload["frozen_fixture"]["threshold_drift_since_freeze"] == {}
    assert "No drift" in payload["frozen_fixture"]["drift_note"]


def test_foreign_snapshot_is_flagged_not_silently_accepted(tmp_path) -> None:
    live = scorer_manifest_payload()["thresholds"]

    payload = scorer_manifest_payload(
        frozen_fixture=_fixture(tmp_path, live, snapshot_sha256="f" * 64)
    )

    assert payload["frozen_fixture"]["snapshot_matches"] is False


def test_payload_is_json_serializable_and_deterministic() -> None:
    first = json.dumps(scorer_manifest_payload(), sort_keys=True)
    second = json.dumps(scorer_manifest_payload(), sort_keys=True)

    assert first == second
