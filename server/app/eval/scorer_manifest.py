"""Emit the live scorer configuration as a dated, checkable artifact.

`fixtures/eval50.json` records the threshold manifest **as it stood on freeze
day**. Any scorer rule declared after the freeze and before any label — which is
a legal pre-label protocol change — therefore does not appear in the frozen
fixture, even though it governed the model outputs. The
cross-source wind-disagreement guard (declared 2026-08-06, one day after the
2026-08-05 freeze) is the first such rule.

Re-running the freeze to absorb it would be the wrong fix: it would stamp the
frozen fixture with a commit that postdates the model outputs, collapsing a
legal ordering into one that merely looks tidy. Instead this module emits an
additive record of the configuration that actually scored the claims, so the
freeze artifact and the scorer artifact together describe the real timeline.

CLI: ``python -m app.eval.scorer_manifest --out <path>``
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from app.eval.freeze import threshold_manifest_payload
from app.eval.observation_age_empirics import observation_age_manifest_payload
from app.llm.corroboration import (
    SOURCE_CHANNELS,
    calm_wind_manifest_payload,
    wind_disagreement_manifest_payload,
)
from app.provenance.nomination import nomination_manifest_payload
from app.provenance.openaq_pm25 import (
    LOCKED_SNAPSHOT_SHA256,
    STUDY_END_EXCLUSIVE,
    STUDY_START,
)
from app.provenance.purpleair_qc import purpleair_qc_manifest_payload


def _git_commit() -> str | None:
    """The HEAD commit, or None outside a checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _git_is_clean() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return not result.stdout.strip()


def scorer_manifest_payload(*, frozen_fixture: Path | None = None) -> dict[str, Any]:
    """The full live scorer configuration, plus any drift from a frozen fixture."""
    thresholds = threshold_manifest_payload()
    payload: dict[str, Any] = {
        "purpose": (
            "the scorer configuration in force when model outputs were "
            "generated; complements, and does not replace, the freeze-day "
            "manifest inside the frozen fixture"
        ),
        "code_commit": _git_commit(),
        "working_tree_clean": _git_is_clean(),
        "snapshot_sha256": LOCKED_SNAPSHOT_SHA256,
        "study_start": STUDY_START,
        "study_end_exclusive": STUDY_END_EXCLUSIVE,
        "thresholds": thresholds,
        "source_channels": dict(sorted(SOURCE_CHANNELS.items())),
        "guards": {
            "calm_wind": calm_wind_manifest_payload(),
            "wind_disagreement": wind_disagreement_manifest_payload(),
            "observation_age_gates": observation_age_manifest_payload(),
            "nomination_eligibility": nomination_manifest_payload(),
            "purpleair_time_aware_qc": purpleair_qc_manifest_payload(),
        },
    }

    if frozen_fixture is not None:
        frozen = json.loads(frozen_fixture.read_text())
        frozen_thresholds = frozen.get("thresholds", {})
        drift: dict[str, Any] = {}
        for group in sorted(set(frozen_thresholds) | set(thresholds)):
            before = frozen_thresholds.get(group) or {}
            after = thresholds.get(group) or {}
            if before == after:
                continue
            drift[group] = {
                "added": {k: after[k] for k in sorted(set(after) - set(before))},
                "removed": {k: before[k] for k in sorted(set(before) - set(after))},
                "changed": {
                    k: {"frozen": before[k], "live": after[k]}
                    for k in sorted(set(before) & set(after))
                    if before[k] != after[k]
                },
            }
        payload["frozen_fixture"] = {
            "path": frozen_fixture.name,
            "snapshot_sha256": frozen.get("snapshot_sha256"),
            "code_commit": frozen.get("code_commit"),
            "frozen_at": frozen.get("frozen_at"),
            "snapshot_matches": frozen.get("snapshot_sha256")
            == LOCKED_SNAPSHOT_SHA256,
            "threshold_drift_since_freeze": drift,
            "drift_note": (
                "Rules added after the freeze and before any expert label "
                "exists. Selection of the 50 events is unaffected: these are "
                "scorer rules, not selection rules."
                if drift
                else "No drift; the live scorer matches the freeze-day manifest."
            ),
        }
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval.scorer_manifest",
        description=(
            "Emit the live scorer configuration, with drift against a frozen "
            "fixture's freeze-day threshold manifest."
        ),
    )
    parser.add_argument(
        "--anomaly-set",
        type=Path,
        default=None,
        help="frozen fixture to diff the live thresholds against",
    )
    parser.add_argument("--out", type=Path, default=None, help="JSON artifact path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    payload = scorer_manifest_payload(frozen_fixture=args.anomaly_set)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
