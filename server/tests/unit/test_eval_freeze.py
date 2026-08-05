"""Eval-set freeze: event dedup, composite ranking, fixture output.

The freeze encodes the Month 2 inclusion rule ("top-50 summer anomalies by
composite severity") in reviewable code: one physical event contributes one
anomaly, ranking is (consensus count, |z|), and the fixture records the
criteria next to the ids.
"""

import json
import subprocess
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.db import session as session_module
from app.db.models import Anomaly, EnrichmentRecord
from app.eval import freeze as freeze_module
from app.eval.freeze import (
    FreezeResult,
    allocate_strata,
    fixture_payload,
    freeze_eval_set,
    group_events,
)
from app.eval.harness import load_anomaly_set
from app.provenance.purpleair_qc import LOCKED_SNAPSHOT_SHA256
from app.llm.corroboration import (
    DEFAULT_BACKGROUND_TOLERANCE,
    DEFAULT_CHEMISTRY_TOLERANCE,
    DEFAULT_CONCENTRATION_TOLERANCE,
    DEFAULT_SECONDARY_TOLERANCE,
    DEFAULT_SOURCE_TYPE_TOLERANCE,
    DEFAULT_TEMPORAL_TOLERANCE,
    DEFAULT_TRAP_TOLERANCE,
    DEFAULT_WIND_TOLERANCE,
)

T0 = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)
HOUSTON_LAT = 29.7604
HOUSTON_LON = -95.3698
SNAPSHOT_SHA256 = LOCKED_SNAPSHOT_SHA256
CODE_COMMIT = "a" * 40


def _fixture_payload(result: FreezeResult) -> dict:
    return fixture_payload(
        result,
        snapshot_sha256=SNAPSHOT_SHA256,
        code_commit=CODE_COMMIT,
        b18_decision="accept_unstratified",
        b18_rationale="Synthetic review found no composition pathology.",
    )


def _anomaly(
    *,
    ts: datetime = T0,
    metric: str = "pm25",
    source: str = "openaq",
    lat: float = HOUSTON_LAT,
    lon: float = HOUSTON_LON,
    methods: list[str] | None = None,
    z_score: float | None = 4.0,
    severity: str = "moderate",
) -> Anomaly:
    detector_availability = {
        "zscore": {"ran": True, "skip_code": None, "detail": None},
        "stl": {"ran": True, "skip_code": None, "detail": None},
        "isolation_forest": {
            "ran": True,
            "skip_code": None,
            "detail": None,
        },
    }
    return Anomaly(
        id=uuid.uuid4(),
        timestamp=ts,
        lat=lat,
        lon=lon,
        metric=metric,
        source=source,
        source_entity_id=f"{source}-{metric}-entity",
        detector_availability_json=detector_availability,
        value=100.0,
        expected_value=20.0,
        z_score=z_score,
        methods_triggered=methods or ["zscore", "stl"],
        severity=severity,
    )


class TestGroupEvents:
    def test_nearby_same_metric_anomalies_merge(self) -> None:
        events = group_events(
            [
                _anomaly(ts=T0),
                _anomaly(ts=T0 + timedelta(minutes=20), lat=HOUSTON_LAT + 0.02),
            ]
        )
        assert len(events) == 1
        assert len(events[0]) == 2

    def test_different_metrics_never_merge(self) -> None:
        events = group_events([_anomaly(metric="pm25"), _anomaly(metric="ozone")])
        assert len(events) == 2

    def test_same_metric_cross_source_events_still_merge(self) -> None:
        events = group_events(
            [
                _anomaly(source="openaq"),
                _anomaly(source="tceq", ts=T0 + timedelta(minutes=5)),
            ]
        )

        assert len(events) == 1
        assert {anomaly.source for anomaly in events[0]} == {"openaq", "tceq"}

    def test_distant_stations_do_not_merge(self) -> None:
        # ~28 km apart at the same instant: separate events.
        events = group_events(
            [_anomaly(), _anomaly(lat=HOUSTON_LAT + 0.25)]
        )
        assert len(events) == 2

    def test_far_apart_in_time_do_not_merge(self) -> None:
        events = group_events(
            [_anomaly(ts=T0), _anomaly(ts=T0 + timedelta(hours=2))]
        )
        assert len(events) == 2

    def test_bridging_anomaly_unions_two_separate_events(self) -> None:
        # A and B are ~16 km apart (separate events); C sits ~8 km from each,
        # within the radius of both, so it bridges them into one event.
        # First-match-and-break joins C to A only and strands B as a second
        # event, under-merging one physical event into two representatives.
        a = _anomaly(ts=T0, lat=HOUSTON_LAT)
        b = _anomaly(ts=T0 + timedelta(minutes=1), lat=HOUSTON_LAT + 0.144)
        c = _anomaly(ts=T0 + timedelta(minutes=2), lat=HOUSTON_LAT + 0.072)

        events = group_events([a, b, c])

        assert len(events) == 1
        assert len(events[0]) == 3

    def test_event_grouping_is_order_independent(self) -> None:
        # Same three anomalies (A--C--B chain by space, equal timestamps so the
        # stable sort preserves input order) must yield one event regardless of
        # the order they arrive in — connected components, not first-match.
        a = _anomaly(ts=T0, lat=HOUSTON_LAT)
        b = _anomaly(ts=T0, lat=HOUSTON_LAT + 0.144)
        c = _anomaly(ts=T0, lat=HOUSTON_LAT + 0.072)

        for ordering in ([a, b, c], [c, b, a], [b, a, c]):
            events = group_events(list(ordering))
            assert len(events) == 1, ordering
            assert len(events[0]) == 3

    def test_single_linkage_chains_a_moving_event(self) -> None:
        # A->B and B->C are each within the merge window; A->C is not.
        # Chain linkage keeps the moving plume one event.
        events = group_events(
            [
                _anomaly(ts=T0),
                _anomaly(ts=T0 + timedelta(minutes=25)),
                _anomaly(ts=T0 + timedelta(minutes=50)),
            ]
        )
        assert len(events) == 1
        assert len(events[0]) == 3

    def test_consecutive_hourly_flags_chain_into_one_event(self) -> None:
        # The ground sources report hourly, so a 4-hour ozone afternoon at one
        # station flags at 60-minute spacing. The old 30-minute merge window
        # could never chain those — one physical event would occupy four top-N
        # slots. The window must exceed the coarsest live cadence.
        events = group_events(
            [_anomaly(ts=T0 + timedelta(hours=h)) for h in range(4)]
        )
        assert len(events) == 1
        assert len(events[0]) == 4


class TestFreezeEvalSet:
    @pytest.mark.asyncio
    async def test_one_event_contributes_one_anomaly(self, db_session) -> None:
        # Five station-hours of one pm25 event must not fill the set.
        event_members = [
            _anomaly(ts=T0 + timedelta(minutes=10 * i)) for i in range(5)
        ]
        lone = _anomaly(metric="ozone", ts=T0 + timedelta(hours=8), z_score=3.2)
        for a in [*event_members, lone]:
            db_session.add(a)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=50,
        )
        assert result.n_anomalies == 6
        assert result.n_events == 2
        assert len(result.selected) == 2

    @pytest.mark.asyncio
    async def test_rejects_nonpositive_top_n(self, db_session) -> None:
        # top_n=0 silently froze an empty set; a negative top sliced all-but-N.
        with pytest.raises(ValueError, match="top_n"):
            await freeze_eval_set(
                db_session,
                window_start=T0 - timedelta(days=1),
                window_end=T0 + timedelta(days=1),
                top_n=0,
            )

    @pytest.mark.asyncio
    async def test_consensus_count_outranks_z_score(self, db_session) -> None:
        three_methods = _anomaly(
            metric="ozone",
            ts=T0 + timedelta(hours=8),
            methods=["zscore", "stl", "isolation_forest"],
            z_score=3.1,
            severity="severe",
        )
        big_z_two_methods = _anomaly(z_score=9.5)
        db_session.add(three_methods)
        db_session.add(big_z_two_methods)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=1,
        )
        assert result.selected[0].id == three_methods.id

    @pytest.mark.asyncio
    async def test_event_representative_is_its_strongest_member(
        self, db_session
    ) -> None:
        weak = _anomaly(ts=T0, z_score=3.0)
        strong = _anomaly(ts=T0 + timedelta(minutes=15), z_score=8.0)
        db_session.add(weak)
        db_session.add(strong)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=10,
        )
        assert [a.id for a in result.selected] == [strong.id]

    @pytest.mark.asyncio
    async def test_window_excludes_out_of_range_anomalies(self, db_session) -> None:
        may = _anomaly(ts=datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc))
        july = _anomaly(ts=T0, metric="ozone")
        db_session.add(may)
        db_session.add(july)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
        )
        assert [a.id for a in result.selected] == [july.id]

    @pytest.mark.asyncio
    async def test_reports_selected_anomalies_missing_enrichment(
        self, db_session
    ) -> None:
        enriched = _anomaly()
        bare = _anomaly(metric="ozone", ts=T0 + timedelta(hours=8))
        db_session.add(enriched)
        db_session.add(bare)
        await db_session.commit()
        db_session.add(
            EnrichmentRecord(
                anomaly_id=enriched.id,
                context_window_start=T0 - timedelta(hours=36),
                context_window_end=T0 + timedelta(hours=36),
                cross_source_summary_json={"schema_version": 1},
            )
        )
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=50,
        )
        assert result.missing_enrichment == [bare.id]


class TestFreezeProvenance:
    def test_repository_commit_requires_clean_tree_and_returns_full_head(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            if command[1] == "status":
                return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=f"{CODE_COMMIT}\n",
                stderr="",
            )

        monkeypatch.setattr(freeze_module.subprocess, "run", fake_run)

        assert freeze_module.repository_code_commit(tmp_path) == CODE_COMMIT
        assert calls == [
            ["git", "status", "--porcelain", "--untracked-files=all"],
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        ]

    def test_repository_commit_rejects_dirty_tree_before_resolving_head(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        calls: list[list[str]] = []

        def fake_run(
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=" M server/app/eval/freeze.py\n?? untracked.py\n",
                stderr="",
            )

        monkeypatch.setattr(freeze_module.subprocess, "run", fake_run)

        with pytest.raises(RuntimeError, match="dirty.*freeze.py.*untracked.py"):
            freeze_module.repository_code_commit(tmp_path)

        assert calls == [
            ["git", "status", "--porcelain", "--untracked-files=all"]
        ]

    def test_repository_commit_wraps_git_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def fail_run(
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            raise subprocess.CalledProcessError(
                128,
                command,
                stderr="fatal: not a git repository",
            )

        monkeypatch.setattr(freeze_module.subprocess, "run", fail_run)

        with pytest.raises(RuntimeError, match="Git provenance"):
            freeze_module.repository_code_commit(tmp_path)

    def test_repository_commit_rejects_non_full_head(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        def fake_run(
            command: list[str],
            **_kwargs: object,
        ) -> subprocess.CompletedProcess[str]:
            output = "" if command[1] == "status" else "abc123\n"
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=output,
                stderr="",
            )

        monkeypatch.setattr(freeze_module.subprocess, "run", fake_run)

        with pytest.raises(ValueError, match="code commit"):
            freeze_module.repository_code_commit(tmp_path)

    def test_parse_args_requires_snapshot_hash_even_for_dry_run(self) -> None:
        with pytest.raises(SystemExit):
            freeze_module._parse_args(
                [
                    "--start",
                    "2026-06-01",
                    "--end",
                    "2026-06-30",
                    "--dry-run",
                ]
            )

    def test_parse_args_requires_b18_decision_for_real_fixture(self) -> None:
        with pytest.raises(SystemExit):
            freeze_module._parse_args(
                [
                    "--start",
                    "2026-06-01",
                    "--end",
                    "2026-06-30",
                    "--snapshot-sha256",
                    SNAPSHOT_SHA256,
                    "--out",
                    "eval.json",
                ]
            )


class TestFixturePayload:
    def test_payload_carries_required_provenance_and_live_thresholds(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        payload = _fixture_payload(result)

        assert payload["snapshot_sha256"] == SNAPSHOT_SHA256
        assert payload["code_commit"] == CODE_COMMIT
        assert payload["thresholds"] == {
            "atmospheric_trap": asdict(DEFAULT_TRAP_TOLERANCE),
            "background_vs_event": asdict(DEFAULT_BACKGROUND_TOLERANCE),
            "chemistry": asdict(DEFAULT_CHEMISTRY_TOLERANCE),
            "concentration_elevation": asdict(
                DEFAULT_CONCENTRATION_TOLERANCE
            ),
            "emissions_source_type": asdict(DEFAULT_SOURCE_TYPE_TOLERANCE),
            "secondary_formation": asdict(DEFAULT_SECONDARY_TOLERANCE),
            "temporal_pattern": asdict(DEFAULT_TEMPORAL_TOLERANCE),
            "wind": asdict(DEFAULT_WIND_TOLERANCE),
        }

    def test_payload_carries_b9_nomination_fixture_and_rule(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"]["nomination_eligibility"]

        assert block["fixture_id"] == "openaq-regulatory-entity-provenance-v2"
        assert block["schema_version"] == 2
        assert block["artifact"] == "openaq_regulatory_entity_provenance.v2.json"
        assert block["snapshot_sha256"] == SNAPSHOT_SHA256
        assert block["covered_metrics"] == ["ozone", "pm10", "pm25"]
        assert block["eligible_entity_counts"] == {
            "ozone": 18,
            "pm10": 3,
            "pm25": 12,
        }
        assert block["nominating_metrics_by_source"] == {
            "openaq": ["ozone", "pm10", "pm25"],
            "tceq": ["co", "no2", "so2"],
        }
        assert block["strict_elevation_rule"] == "value > expected_value"

    def test_payload_marks_proposed_calm_wind_floor_as_not_shipped(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"]["calm_wind_guard"]

        assert block["floor_ms"] == 1.5
        assert block["floor_status"] == "bracco_confirmed"
        assert block["bracco_amendment_confirmed"] is True
        assert block["bracco_confirmation_date"] == "2026-07-24"
        assert block["ship_status"] == "shipped_bracco_confirmed"
        assert block["raw_nonpositive_without_floor"] == "disabled_loudly"

    def test_payload_hash_links_station_baseline_evidence(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"]["baseline_locality"]
        metrics = {
            (row["source"], row["metric"]): row for row in block["metrics"]
        }

        assert block["snapshot_sha256"] == SNAPSHOT_SHA256
        assert block["artifact"] == "baseline_locality_empirics.v1.json"
        assert block["artifact_sha256"] == (
            "101c10a3516168bfd75b6f985b84fe59c3e8b5e17557d24c5ee2f66cef57e2bc"
        )
        assert block["rules"]["baseline_locality"] == "nearest_event_entity"
        assert block["rules"]["pooled_fallback"] is False
        assert metrics[("tceq", "no2")]["pooled_supporting"] == 478
        assert metrics[("tceq", "no2")]["matched_supporting"] == 197
        assert metrics[("sentinel5p", "s5p_no2_column")][
            "matched_evaluable_count"
        ] == 0

    def test_threshold_payload_reads_mutated_tolerance_owner(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            freeze_module,
            "DEFAULT_WIND_TOLERANCE",
            replace(DEFAULT_WIND_TOLERANCE, bearing_deg=17.25),
        )

        thresholds = freeze_module.threshold_manifest_payload()

        assert thresholds["wind"]["bearing_deg"] == 17.25
        assert thresholds["wind"] == asdict(
            freeze_module.DEFAULT_WIND_TOLERANCE
        )

    @pytest.mark.parametrize(
        "snapshot_sha256",
        (
            "",
            "g" * 64,
            SNAPSHOT_SHA256.upper(),
            "0" * 64,
        ),
    )
    def test_payload_rejects_noncanonical_snapshot_hash(
        self,
        snapshot_sha256: str,
    ) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        with pytest.raises(ValueError, match="snapshot SHA-256"):
            fixture_payload(
                result,
                snapshot_sha256=snapshot_sha256,
                code_commit=CODE_COMMIT,
            )

    @pytest.mark.parametrize(
        "code_commit",
        ("", "g" * 40, "a" * 39, "a" * 41),
    )
    def test_payload_rejects_invalid_full_commit_id(self, code_commit: str) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        with pytest.raises(ValueError, match="code commit"):
            fixture_payload(
                result,
                snapshot_sha256=SNAPSHOT_SHA256,
                code_commit=code_commit,
            )

    def test_payload_carries_observation_age_gate_empirics(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"]["observation_age_gates"]

        assert block["artifact"] == "observation_age_empirics.v1.json"
        assert block["artifact_sha256"] == (
            "b22bf6cbf02abf2a87c25e2fd4898a90923759107e5ed9c31211cd03fc30d27e"
        )
        assert block["anchor_count"] == 936
        assert block["gates_minutes"] == {
            "asos": 90.0,
            "epa_aqs": 90.0,
            "noaa_gfs": 360.0,
            "openaq": 90.0,
            "openweather": 90.0,
            "purpleair": 90.0,
            "sentinel5p": 720.0,
            "tceq": 90.0,
        }
        assert block["stop_rule_violations"] == []
        assert block["structurally_absent_sources"] == ["epa_aqs"]
        assert len(block["metrics"]) == 36

    def test_payload_carries_purpleair_qc_manifest_evidence(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"][
            "purpleair_time_aware_qc"
        ]

        assert block["artifact"] == "purpleair_time_aware_qc.v1.json"
        assert block["artifact_sha256"] == (
            "057c1e65e7f0b0a7cee5ba3033730777951a38731cda00ee83f82542e18d7468"
        )
        assert block["snapshot_sha256"] == LOCKED_SNAPSHOT_SHA256
        assert block["parameters"] == {
            "candidate_min_observations": 6,
            "center_step_hours": 1,
            "minimum_peer_sensors": 10,
            "network_extreme_median_ug_m3": 100.0,
            "peer_min_observations": 6,
            "saturation_ug_m3": 500.0,
            "segment_absolute_floor_ug_m3": 20.0,
            "segment_ratio_threshold": 5.0,
            "window_hours": 24,
        }
        assert block["window_evaluation"]["unevaluated_fraction"] == pytest.approx(
            0.106916820251
        )
        assert {segment["entity_id"] for segment in block["segments"]} >= {
            "165203",
            "194469",
            "288282",
        }
        assert all(
            sensor["retained_after_last_excluded"] > 0
            for sensor in block["audited_sensors"]
        )

    def test_payload_carries_censoring_rule_and_sensitivity_evidence(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"]["censoring"]

        assert block["artifact"] == "censoring_sensitivity.v1.json"
        assert block["artifact_sha256"] == (
            "93dfccdd5106769c7e08efcf21a2b22e4461c4a1c307962f2b7594c6480d6469"
        )
        assert block["snapshot_sha256"] == LOCKED_SNAPSHOT_SHA256
        assert block["unit_assertion_passed"] is True
        assert block["rules"]["primary"] == "limit_half"
        assert block["rules"]["alternative"] == "delete"
        assert block["rules"]["ground_so2_limit_ppb"] == 0.5
        assert block["rules"]["so2_quantitative_exclusion_reason"] == (
            "so2_underpowered"
        )
        assert len(block["metrics"]) == 10
        tceq_so2 = next(
            metric
            for metric in block["metrics"]
            if metric["source"] == "tceq" and metric["metric"] == "so2"
        )
        assert tceq_so2["censored_fraction"] == pytest.approx(
            0.9685603148558249
        )
        assert tceq_so2["deletion_evaluable_windows"] == 304

    def test_payload_embeds_complete_ratified_pruning_screen(self) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        block = _fixture_payload(result)["data_quality"]["variable_pruning"]

        assert block["artifact"] == "pruning_screen.run-001.json"
        assert block["screen"]["real_screen_executed"] is True
        assert block["screen"]["mechanism_review_complete"] is True
        assert block["screen"]["drop_metric_keys"] == []
        assert len(block["screen"]["cells"]) == 360

    @pytest.mark.asyncio
    async def test_freeze_cli_writes_censoring_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 6, 2, tzinfo=timezone.utc),
            top_n=1,
            n_anomalies=0,
            n_events=0,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
        )

        @asynccontextmanager
        async def fake_engine_lifecycle() -> AsyncIterator[object]:
            yield object()

        @asynccontextmanager
        async def fake_async_session() -> AsyncIterator[object]:
            yield object()

        async def fake_freeze_eval_set(
            session: object,
            *,
            window_start: datetime,
            window_end: datetime,
            top_n: int,
            min_fresh_channels: int,
        ) -> FreezeResult:
            assert session is not None
            assert window_start == result.window_start
            assert window_end == result.window_end
            assert top_n == result.top_n
            assert min_fresh_channels == result.min_fresh_channels
            return result

        monkeypatch.setattr(
            session_module,
            "engine_lifecycle",
            fake_engine_lifecycle,
        )
        monkeypatch.setattr(session_module, "async_session", fake_async_session)
        monkeypatch.setattr(
            freeze_module,
            "freeze_eval_set",
            fake_freeze_eval_set,
        )
        monkeypatch.setattr(
            freeze_module,
            "repository_code_commit",
            lambda _root: CODE_COMMIT,
            raising=False,
        )
        output = tmp_path / "eval.json"

        exit_code = await freeze_module._amain(
            [
                "--start",
                "2026-06-01",
                "--end",
                "2026-06-01",
                "--top",
                "1",
                "--snapshot-sha256",
                SNAPSHOT_SHA256,
                "--b18-decision",
                "accept_unstratified",
                "--b18-rationale",
                "Synthetic review found no composition pathology.",
                "--out",
                str(output),
            ]
        )

        payload = json.loads(output.read_text(encoding="utf-8"))
        assert exit_code == 0
        assert payload["data_quality"]["censoring"]["artifact_sha256"] == (
            "93dfccdd5106769c7e08efcf21a2b22e4461c4a1c307962f2b7594c6480d6469"
        )
        assert payload["snapshot_sha256"] == SNAPSHOT_SHA256
        assert payload["code_commit"] == CODE_COMMIT

    @pytest.mark.asyncio
    async def test_invalid_snapshot_fails_before_database_or_fixture_write(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        entered_engine = False

        @asynccontextmanager
        async def fake_engine_lifecycle() -> AsyncIterator[object]:
            nonlocal entered_engine
            entered_engine = True
            yield object()

        monkeypatch.setattr(
            session_module,
            "engine_lifecycle",
            fake_engine_lifecycle,
        )
        monkeypatch.setattr(
            freeze_module,
            "repository_code_commit",
            lambda _root: CODE_COMMIT,
            raising=False,
        )
        output = tmp_path / "must-not-exist.json"

        with pytest.raises(ValueError, match="snapshot SHA-256"):
            await freeze_module._amain(
                [
                    "--start",
                    "2026-06-01",
                    "--end",
                    "2026-06-30",
                    "--snapshot-sha256",
                    "0" * 64,
                    "--b18-decision",
                    "accept_unstratified",
                    "--b18-rationale",
                    "Synthetic review found no composition pathology.",
                    "--out",
                    str(output),
                ]
            )

        assert entered_engine is False
        assert not output.exists()

    def test_payload_loads_through_the_harness(self, tmp_path) -> None:
        selected = [_anomaly(), _anomaly(metric="ozone")]
        result = FreezeResult(
            window_start=datetime(2026, 6, 1, tzinfo=timezone.utc),
            window_end=datetime(2026, 9, 1, tzinfo=timezone.utc),
            top_n=50,
            n_anomalies=12,
            n_events=2,
            selected=selected,
            event_sizes={selected[0].id: 5, selected[1].id: 1},
            missing_enrichment=[],
        )
        payload = _fixture_payload(result)
        path = tmp_path / "eval50.json"
        path.write_text(json.dumps(payload))

        assert load_anomaly_set(path) == [a.id for a in selected]
        assert payload["criteria"]["top"] == 50
        assert payload["n_events"] == 2
        # Composition is recorded with the fixture so a source/metric skew in
        # the frozen set is visible on freeze day, not discovered post hoc.
        assert payload["composition"] == {
            "openaq/ozone/moderate": 1,
            "openaq/pm25/moderate": 1,
        }


# --- 2026-07-24 channel-coverage eligibility screen ---


def _coverage_summary(sources: dict[str, float | None]) -> dict:
    return {
        "sources": {
            source: {
                "metrics": {
                    "pm25": {"nearest_in_time": {"v": 1.0, "dt_minutes": dt}}
                }
            }
            for source, dt in sources.items()
        }
    }


def _enrichment(anomaly: Anomaly, summary: dict) -> EnrichmentRecord:
    return EnrichmentRecord(
        anomaly_id=anomaly.id,
        context_window_start=T0 - timedelta(hours=36),
        context_window_end=T0 + timedelta(hours=36),
        cross_source_summary_json=summary,
    )


def test_fresh_channel_coverage_counts_distinct_age_gated_channels() -> None:
    from app.eval.freeze import fresh_channel_coverage

    summary = _coverage_summary(
        {
            "openaq": 30.0,
            "tceq": 30.0,
            "asos": 30.0,
            "noaa_gfs": 400.0,
        }
    )
    summary["sources"]["purpleair"] = {
        "metrics": {"pm25": {"nearest_in_time": {"v": None, "dt_minutes": 5.0}}}
    }

    coverage = fresh_channel_coverage(summary)

    assert coverage["ground_insitu"] is True
    assert coverage["met_insitu"] is True
    assert coverage["nwp"] is False
    assert coverage["ground_optical"] is False
    assert sum(coverage.values()) == 2


class TestChannelCoverageScreen:
    @pytest.mark.asyncio
    async def test_low_coverage_representative_excluded_before_top_n(
        self, db_session
    ) -> None:
        rich = _anomaly(metric="pm25", methods=["zscore"], z_score=4.0)
        thin = _anomaly(
            metric="ozone",
            methods=["zscore", "stl", "isolation_forest"],
            z_score=9.0,
        )
        db_session.add_all([rich, thin])
        await db_session.flush()
        db_session.add(
            _enrichment(
                rich,
                _coverage_summary(
                    {"openaq": 30.0, "asos": 30.0, "noaa_gfs": 30.0}
                ),
            )
        )
        db_session.add(_enrichment(thin, _coverage_summary({"openaq": 30.0})))
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=1,
            min_fresh_channels=3,
        )

        assert [a.id for a in result.selected] == [rich.id]
        assert result.coverage_excluded == [thin.id]
        assert result.coverage_channel_counts[thin.id] == 1
        assert result.coverage_channel_counts[rich.id] == 3
        assert result.min_fresh_channels == 3

    @pytest.mark.asyncio
    async def test_missing_enrichment_is_not_screened(self, db_session) -> None:
        bare = _anomaly()
        db_session.add(bare)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=1),
            top_n=1,
            min_fresh_channels=5,
        )

        assert [a.id for a in result.selected] == [bare.id]
        assert bare.id in result.missing_enrichment
        assert result.coverage_excluded == []

    def test_fixture_payload_records_coverage_screen(self) -> None:
        excluded = uuid.uuid4()
        result = FreezeResult(
            window_start=T0,
            window_end=T0 + timedelta(days=1),
            top_n=1,
            n_anomalies=2,
            n_events=2,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
            min_fresh_channels=3,
            coverage_excluded=[excluded],
            coverage_channel_counts={excluded: 1},
        )

        block = _fixture_payload(result)["data_quality"][
            "channel_coverage_screen"
        ]

        assert block["min_fresh_channels"] == 3
        assert block["excluded_anomaly_ids"] == [str(excluded)]
        assert block["channel_counts"] == {str(excluded): 1}


class TestLockedSnapshotConstant:
    def test_both_provenance_modules_pin_the_same_snapshot(self) -> None:
        # The hash is defined twice, in openaq_pm25 and purpleair_qc, and
        # different subsystems validate against different copies. If they ever
        # drift, half the pipeline attests to one snapshot and half to another
        # and nothing else in the suite would notice.
        from app.provenance import openaq_pm25, purpleair_qc

        assert openaq_pm25.LOCKED_SNAPSHOT_SHA256 == purpleair_qc.LOCKED_SNAPSHOT_SHA256

    def test_every_module_pins_the_same_study_window(self) -> None:
        # openaq_coverage kept its own copy of the window and fell a month
        # behind, which surfaced as a bogus "entity-set drift" error rather
        # than as an obviously stale constant. All three read one owner now.
        from app.eval import openaq_coverage
        from app.provenance import openaq_pm25, purpleair_qc

        assert openaq_pm25.STUDY_START == _iso_z(purpleair_qc.STUDY_START)
        assert openaq_pm25.STUDY_END_EXCLUSIVE == _iso_z(
            purpleair_qc.STUDY_END_EXCLUSIVE
        )
        assert openaq_coverage.STUDY_START == purpleair_qc.STUDY_START
        assert openaq_coverage.STUDY_END_EXCLUSIVE == purpleair_qc.STUDY_END_EXCLUSIVE

    def test_the_string_and_datetime_window_forms_cannot_disagree(self) -> None:
        from app.provenance import openaq_pm25

        assert openaq_pm25.STUDY_START == _iso_z(openaq_pm25.STUDY_START_AT)
        assert openaq_pm25.STUDY_END_EXCLUSIVE == _iso_z(
            openaq_pm25.STUDY_END_EXCLUSIVE_AT
        )


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class TestAllocateStrata:
    def test_floor_rescues_a_metric_the_z_tiebreak_would_starve(self) -> None:
        # The observed July 12 eligible pools. Ozone holds the second largest
        # pool and a global rank cut gave it 2 of 50 slots.
        pools = {"no2": 139, "ozone": 55, "so2": 38, "pm25": 26, "co": 20, "pm10": 17}

        allocation = allocate_strata(pools, top_n=50, floor=5)

        assert sum(allocation.values()) == 50
        assert min(allocation.values()) >= 5
        assert allocation["ozone"] == 8
        assert allocation["no2"] == 21

    def test_allocation_never_exceeds_a_stratum_pool(self) -> None:
        allocation = allocate_strata(
            {"no2": 100, "co": 2}, top_n=50, floor=5
        )

        assert allocation["co"] == 2
        assert sum(allocation.values()) == 50

    def test_small_total_pool_is_taken_whole(self) -> None:
        assert allocate_strata({"a": 3, "b": 2}, top_n=50, floor=5) == {"a": 3, "b": 2}

    def test_floor_that_cannot_fit_falls_back_to_highest_averages(self) -> None:
        # Six strata cannot each hold 5 slots inside a budget of 4. Truncating
        # by name would make the selection depend on the metric spelling.
        allocation = allocate_strata(
            {"no2": 139, "ozone": 55, "so2": 38, "pm25": 26, "co": 20, "pm10": 17},
            top_n=4,
            floor=5,
        )

        assert sum(allocation.values()) == 4
        assert allocation["no2"] == 3

    def test_allocation_is_independent_of_mapping_order(self) -> None:
        pools = {"no2": 139, "ozone": 55, "so2": 38, "pm25": 26, "co": 20, "pm10": 17}
        reversed_pools = dict(reversed(list(pools.items())))

        assert allocate_strata(pools, top_n=50, floor=5) == allocate_strata(
            reversed_pools, top_n=50, floor=5
        )

    def test_empty_pool_allocates_nothing(self) -> None:
        assert allocate_strata({}, top_n=50, floor=5) == {}


class TestStratifiedSelection:
    @pytest.mark.asyncio
    async def test_stratifying_admits_a_metric_the_global_cut_drops(
        self, db_session
    ) -> None:
        # Three high-|z| SO2 events and one lower-|z| ozone event, one slot
        # short of taking everything: the global cut fills on SO2 alone.
        so2 = [
            _anomaly(
                metric="so2",
                ts=T0 + timedelta(hours=6 * i),
                lat=HOUSTON_LAT + 0.5 * i,
                methods=["zscore", "stl", "isolation_forest"],
                z_score=40.0 + i,
            )
            for i in range(3)
        ]
        ozone = _anomaly(
            metric="ozone",
            ts=T0 + timedelta(hours=30),
            lat=HOUSTON_LAT + 3.0,
            methods=["zscore", "stl", "isolation_forest"],
            z_score=4.0,
        )
        db_session.add_all([*so2, ozone])
        await db_session.commit()

        window = {
            "window_start": T0 - timedelta(days=1),
            "window_end": T0 + timedelta(days=3),
        }
        unstratified = await freeze_eval_set(db_session, top_n=3, **window)
        stratified = await freeze_eval_set(
            db_session, top_n=3, stratify=True, stratum_floor=1, **window
        )

        assert {a.metric for a in unstratified.selected} == {"so2"}
        assert sorted(a.metric for a in stratified.selected) == ["ozone", "so2", "so2"]
        assert stratified.stratum_pool_sizes == {"ozone": 1, "so2": 3}
        assert stratified.stratum_allocation == {"ozone": 1, "so2": 2}

    @pytest.mark.asyncio
    async def test_within_a_metric_the_rank_order_still_decides(
        self, db_session
    ) -> None:
        weak, strong = (
            _anomaly(
                metric="so2",
                ts=T0 + timedelta(hours=6 * i),
                lat=HOUSTON_LAT + 0.5 * i,
                methods=["zscore", "stl", "isolation_forest"],
                z_score=z,
            )
            for i, z in enumerate((5.0, 40.0))
        )
        db_session.add_all([weak, strong])
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=3),
            top_n=1,
            stratify=True,
            stratum_floor=1,
        )

        assert [a.id for a in result.selected] == [strong.id]

    @pytest.mark.asyncio
    async def test_selection_stays_in_global_rank_order(self, db_session) -> None:
        anomalies = [
            _anomaly(
                metric=metric,
                ts=T0 + timedelta(hours=6 * i),
                lat=HOUSTON_LAT + 0.5 * i,
                methods=["zscore", "stl", "isolation_forest"],
                z_score=z,
            )
            for i, (metric, z) in enumerate(
                [("so2", 40.0), ("ozone", 30.0), ("so2", 20.0), ("ozone", 10.0)]
            )
        ]
        db_session.add_all(anomalies)
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=3),
            top_n=4,
            stratify=True,
            stratum_floor=1,
        )

        z_scores = [abs(a.z_score) for a in result.selected]
        assert z_scores == sorted(z_scores, reverse=True)

    @pytest.mark.asyncio
    async def test_unstratified_remains_the_default(self, db_session) -> None:
        db_session.add(_anomaly(metric="so2"))
        await db_session.commit()

        result = await freeze_eval_set(
            db_session,
            window_start=T0 - timedelta(days=1),
            window_end=T0 + timedelta(days=3),
            top_n=1,
        )

        assert result.stratified is False
        assert result.stratum_allocation == {}


class TestB18DecisionMatchesSelection:
    def _result(self, *, stratified: bool) -> FreezeResult:
        return FreezeResult(
            window_start=T0,
            window_end=T0 + timedelta(days=1),
            top_n=1,
            n_anomalies=1,
            n_events=1,
            selected=[],
            event_sizes={},
            missing_enrichment=[],
            stratified=stratified,
            stratum_allocation={"so2": 1} if stratified else {},
            stratum_pool_sizes={"so2": 1} if stratified else {},
        )

    def test_claiming_stratified_over_a_global_cut_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match the selection"):
            fixture_payload(
                self._result(stratified=False),
                snapshot_sha256=SNAPSHOT_SHA256,
                code_commit=CODE_COMMIT,
                b18_decision="stratify",
                b18_rationale="Balancing pollutant coverage.",
            )

    def test_claiming_unstratified_over_a_stratified_cut_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="does not match the selection"):
            fixture_payload(
                self._result(stratified=True),
                snapshot_sha256=SNAPSHOT_SHA256,
                code_commit=CODE_COMMIT,
                b18_decision="accept_unstratified",
                b18_rationale="No composition pathology.",
            )

    def test_stratified_fixture_records_the_allocation(self) -> None:
        payload = fixture_payload(
            self._result(stratified=True),
            snapshot_sha256=SNAPSHOT_SHA256,
            code_commit=CODE_COMMIT,
            b18_decision="stratify",
            b18_rationale="Balancing pollutant coverage.",
        )

        selection = payload["criteria"]["selection"]
        assert selection["stratified"] is True
        assert selection["stratum"] == "metric"
        assert selection["stratum_floor"] == 5
        assert selection["slots_per_stratum"] == {"so2": 1}
        assert selection["eligible_per_stratum"] == {"so2": 1}
