"""B4 blinded PDF packet rendering and evidence-faithfulness audit."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.db.models import Anomaly
from app.eval.label_cli import ClaimGroup, presentation_order
from app.eval.packet import (
    calm_wind_claim_flags,
    extract_plot_data,
    render_packet,
    unsafe_text_findings,
)
from app.eval.packet_audit import PacketAuditError, audit_packet, extract_pdf_text

ANOMALY_ID = uuid.UUID("00000000-0000-0000-0000-000000000004")
ANOMALY_TS = datetime(2026, 7, 8, 3, 0, tzinfo=UTC)


def _anomaly(*, timestamp: datetime = ANOMALY_TS) -> Anomaly:
    return Anomaly(
        id=ANOMALY_ID,
        timestamp=timestamp,
        lat=29.52,
        lon=-95.39,
        metric="ozone",
        source="openaq",
        value=0.04,
        expected_value=0.015,
        z_score=4.2,
        methods_triggered=["stl", "isolation_forest"],
        severity="moderate",
    )


def _entity(
    entity_id: str,
    lat: float,
    lon: float,
    series: list[list[str | float]],
) -> dict:
    return {
        "entity_id": entity_id,
        "lat": lat,
        "lon": lon,
        "distance_km": 1.0,
        "n_points": len(series),
        "series": series,
    }


def _metric(unit: str, entities: list[dict]) -> dict:
    values = [float(row[1]) for entity in entities for row in entity["series"]]
    nearest = entities[0]["series"][0] if entities and entities[0]["series"] else [None, None]
    return {
        "unit": unit,
        "n_points": len(values),
        "n_entities": len(entities),
        "value_range": {
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
        },
        "nearest_in_time": {
            "t": nearest[0],
            "v": nearest[1],
            "entity_id": entities[0]["entity_id"] if entities else None,
            "distance_km": 1.0,
            "dt_minutes": 0.0,
        },
        "entities": entities,
    }


def _summary() -> dict:
    t0 = "2026-07-08T03:00:00+00:00"
    t1 = "2026-07-08T04:00:00+00:00"
    return {
        "schema_version": 1,
        "window": {
            "start": "2026-07-06T15:00:00+00:00",
            "end": "2026-07-09T15:00:00+00:00",
            "spatial_radius_km": 50.0,
        },
        "coverage": {
            "asos": True,
            "noaa_gfs": True,
            "openaq": True,
            "openweather": True,
            "purpleair": True,
            "tceq": True,
        },
        "sources": {
            "asos": {
                "n_entities": 1,
                "n_points": 4,
                "metrics": {
                    "wind_speed": _metric(
                        "m/s", [_entity("AS1", 29.5, -95.4, [[t0, 5.0], [t1, 9.0]])]
                    ),
                    "wind_direction": _metric(
                        "degrees", [_entity("AS1", 29.5, -95.4, [[t0, 90.0], [t1, 180.0]])]
                    ),
                },
            },
            "noaa_gfs": {
                "n_entities": 1,
                "n_points": 4,
                "metrics": {
                    "u_10m": _metric(
                        "m/s", [_entity("G1", 29.6, -95.3, [[t0, 3.0], [t1, 8.0]])]
                    ),
                    "v_10m": _metric(
                        "m/s", [_entity("G1", 29.6, -95.3, [[t0, 4.0], [t1, 6.0]])]
                    ),
                },
            },
            "openaq": {
                "n_entities": 1,
                "n_points": 1,
                "metrics": {
                    "ozone": _metric(
                        "ppm", [_entity("OA1", 29.7, -95.2, [[t0, 0.04]])]
                    )
                },
            },
            "openweather": {
                "n_entities": 1,
                "n_points": 4,
                "metrics": {
                    "wind_speed": _metric(
                        "m/s", [_entity("OW1", 29.4, -95.5, [[t0, 2.0], [t1, 3.0]])]
                    ),
                    "wind_direction": _metric(
                        "degrees", [_entity("OW1", 29.4, -95.5, [[t0, 0.0], [t1, 45.0]])]
                    ),
                },
            },
            "purpleair": {
                "n_entities": 1,
                "n_points": 1,
                "metrics": {
                    "pm25": _metric(
                        "ug/m3", [_entity("PA1", 29.8, -95.1, [[t0, 8.0]])]
                    )
                },
            },
            "tceq": {
                "n_entities": 1,
                "n_points": 1,
                "metrics": {
                    "no2": _metric(
                        "ppb", [_entity("TQ1", 29.3, -95.6, [[t0, 7.0]])]
                    )
                },
            },
        },
    }


def _groups() -> list[ClaimGroup]:
    return [
        ClaimGroup("Ozone was above its expected value.", (uuid.UUID(int=11),)),
        ClaimGroup("Winds were easterly at the event time.", (uuid.UUID(int=12),)),
        ClaimGroup("The displayed evidence does not resolve a single cause.", (uuid.UUID(int=13),)),
    ]


def test_plot_data_golden_matches_stored_stations_and_wind_values() -> None:
    plot_data = extract_plot_data(_summary(), ANOMALY_TS)

    assert [station.to_dict() for station in plot_data.stations] == [
        {"source": "asos", "entity_id": "AS1", "lat": 29.5, "lon": -95.4},
        {"source": "openaq", "entity_id": "OA1", "lat": 29.7, "lon": -95.2},
        {"source": "purpleair", "entity_id": "PA1", "lat": 29.8, "lon": -95.1},
        {"source": "tceq", "entity_id": "TQ1", "lat": 29.3, "lon": -95.6},
    ]
    assert [vector.to_dict() for vector in plot_data.vectors] == [
        {
            "source": "asos",
            "entity_id": "AS1",
            "timestamp": "2026-07-08T03:00:00+00:00",
            "lat": 29.5,
            "lon": -95.4,
            "input_kind": "speed_direction",
            "speed": 5.0,
            "direction_from_degrees": 90.0,
            "input_u_east": None,
            "input_v_north": None,
            "u_east": -5.0,
            "v_north": 0.0,
        },
        {
            "source": "noaa_gfs",
            "entity_id": "G1",
            "timestamp": "2026-07-08T03:00:00+00:00",
            "lat": 29.6,
            "lon": -95.3,
            "input_kind": "uv",
            "speed": 5.0,
            "direction_from_degrees": None,
            "input_u_east": 3.0,
            "input_v_north": 4.0,
            "u_east": 3.0,
            "v_north": 4.0,
        },
        {
            "source": "openweather",
            "entity_id": "OW1",
            "timestamp": "2026-07-08T03:00:00+00:00",
            "lat": 29.4,
            "lon": -95.5,
            "input_kind": "speed_direction",
            "speed": 2.0,
            "direction_from_degrees": 0.0,
            "input_u_east": None,
            "input_v_north": None,
            "u_east": 0.0,
            "v_north": -2.0,
        },
    ]


def test_plot_data_handles_naive_event_time_and_empty_series() -> None:
    summary = _summary()
    summary["sources"]["openweather"]["metrics"]["wind_direction"]["entities"][0]["series"] = []

    aware = extract_plot_data(summary, ANOMALY_TS)
    naive = extract_plot_data(summary, ANOMALY_TS.replace(tzinfo=None))

    assert aware.to_dict() == naive.to_dict()
    assert {vector.source for vector in aware.vectors} == {"asos", "noaa_gfs"}


def test_render_is_byte_deterministic_blind_and_uses_label_cli_order(tmp_path: Path) -> None:
    first_pdf = tmp_path / "first.pdf"
    first_manifest = tmp_path / "first.audit.json"
    second_pdf = tmp_path / "second.pdf"
    second_manifest = tmp_path / "second.audit.json"

    first = render_packet(
        _anomaly(), _summary(), _groups(), "bracco-example", first_pdf, first_manifest
    )
    second = render_packet(
        _anomaly(), _summary(), _groups(), "bracco-example", second_pdf, second_manifest
    )

    assert first_pdf.read_bytes() == second_pdf.read_bytes()
    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    assert first.plot_data.to_dict() == second.plot_data.to_dict()
    shown = extract_pdf_text(first_pdf)
    expected = presentation_order(_groups(), ANOMALY_ID, "bracco-example")
    positions = [shown.index(group.claim_text) for group in expected]
    assert positions == sorted(positions)
    normalized = " ".join(shown.split())
    assert "Instructions" not in shown
    assert "labeling guide" in normalized
    assert "Mark exactly one box per claim" in normalized
    lowered = shown.lower()
    for leak in ("llama3:8b", "gpt-5.4", "grounded", "corroboration", "confidence"):
        assert leak not in lowered

    report = audit_packet(
        first_pdf,
        first_manifest,
        _summary(),
        ANOMALY_TS,
        model_names=("llama3:8b", "gpt-5.4", "gemini-3.5-flash"),
        claim_groups=tuple(_groups()),
    )
    assert report.station_count == 4
    assert report.vector_count == 3
    assert report.calm_wind_flag_count == 0
    assert report.blinding_leaks == ()
    assert report.text_sanitation_findings == ()


def test_text_sanitation_rejects_observed_controls_and_allows_units() -> None:
    assert unsafe_text_findings((("claim", "PM10 was 272.0 µg/m³.\t\n\r"),)) == ()
    assert unsafe_text_findings((("claim", "µ\x05g/m\x03"),)) == (
        "claim[1]=U+0005 (Cc)",
        "claim[5]=U+0003 (Cc)",
    )


def test_render_rejects_control_character_before_writing_artifacts(
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "unsafe.pdf"
    manifest_path = tmp_path / "unsafe.audit.json"
    groups = [
        ClaimGroup(
            "A PM10 value of 272.0 \x05g/m\x03 was observed.",
            (uuid.UUID(int=21),),
        )
    ]

    with pytest.raises(ValueError, match=r"U\+0005.*U\+0003"):
        render_packet(
            _anomaly(),
            _summary(),
            groups,
            "bracco-example",
            pdf_path,
            manifest_path,
        )

    assert not pdf_path.exists()
    assert not manifest_path.exists()


def test_audit_rejects_control_character_in_extracted_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "packet.pdf"
    manifest_path = tmp_path / "packet.audit.json"
    render_packet(
        _anomaly(), _summary(), _groups(), "bracco-example", pdf_path, manifest_path
    )
    monkeypatch.setattr(
        "app.eval.packet_audit.extract_pdf_text",
        lambda _path: "safe text\x05",
    )

    with pytest.raises(PacketAuditError, match=r"extracted PDF text.*U\+0005"):
        audit_packet(
            pdf_path,
            manifest_path,
            _summary(),
            ANOMALY_TS,
            model_names=(),
            claim_groups=tuple(_groups()),
        )


def test_audit_rejects_control_character_in_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf_path = tmp_path / "packet.pdf"
    manifest_path = tmp_path / "packet.audit.json"
    render_packet(
        _anomaly(), _summary(), _groups(), "bracco-example", pdf_path, manifest_path
    )
    monkeypatch.setattr(
        "app.eval.packet_audit.extract_pdf_metadata",
        lambda _path: {"/Subject": "unsafe\x03"},
    )

    with pytest.raises(PacketAuditError, match=r"metadata.*U\+0003"):
        audit_packet(
            pdf_path,
            manifest_path,
            _summary(),
            ANOMALY_TS,
            model_names=(),
            claim_groups=tuple(_groups()),
        )


def test_audit_rejects_plot_manifest_tampering(tmp_path: Path) -> None:
    pdf_path = tmp_path / "packet.pdf"
    manifest_path = tmp_path / "packet.audit.json"
    render_packet(
        _anomaly(), _summary(), _groups(), "bracco-example", pdf_path, manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["plot_data"]["vectors"][0]["speed"] = 999.0
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(PacketAuditError, match="plot data"):
        audit_packet(
            pdf_path,
            manifest_path,
            _summary(),
            ANOMALY_TS,
            model_names=("llama3:8b",),
            claim_groups=tuple(_groups()),
        )


def test_audit_rejects_missing_packet_artifacts(tmp_path: Path) -> None:
    pdf_path = tmp_path / "packet.pdf"
    manifest_path = tmp_path / "packet.audit.json"

    with pytest.raises(PacketAuditError, match="PDF does not exist"):
        audit_packet(
            pdf_path,
            manifest_path,
            _summary(),
            ANOMALY_TS,
            model_names=(),
            claim_groups=tuple(_groups()),
        )

    pdf_path.write_bytes(b"not read before the manifest check")
    with pytest.raises(PacketAuditError, match="manifest does not exist"):
        audit_packet(
            pdf_path,
            manifest_path,
            _summary(),
            ANOMALY_TS,
            model_names=(),
            claim_groups=tuple(_groups()),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 1, "unsupported packet audit schema"),
        ("pdf_sha256", "0" * 64, "PDF hash"),
        ("summary_sha256", "0" * 64, "stored summary"),
    ],
)
def test_audit_rejects_manifest_identity_tampering(
    tmp_path: Path,
    field: str,
    value: str | int,
    message: str,
) -> None:
    pdf_path = tmp_path / "packet.pdf"
    manifest_path = tmp_path / "packet.audit.json"
    render_packet(
        _anomaly(), _summary(), _groups(), "bracco-example", pdf_path, manifest_path
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(PacketAuditError, match=message):
        audit_packet(
            pdf_path,
            manifest_path,
            _summary(),
            ANOMALY_TS,
            model_names=(),
            claim_groups=tuple(_groups()),
        )


def test_calm_wind_claim_flag_renders_and_audit_recomputes_it(
    tmp_path: Path,
) -> None:
    summary = _summary()
    speed = summary["sources"]["openweather"]["metrics"]["wind_speed"]
    speed["nearest_in_time"]["v"] = 1.0
    speed["entities"][0]["series"][0][1] = 1.0
    ordered = presentation_order(_groups(), ANOMALY_ID, "bracco-example")

    flags = calm_wind_claim_flags(summary, ordered)

    assert len(flags) == 1
    assert flags[0].source_decisions[0].source == "openweather"
    assert flags[0].source_decisions[0].event_speed_ms == 1.0
    assert flags[0].source_decisions[0].effective_cutoff_ms == 1.5
    assert flags[0].floor_status == "bracco_confirmed"

    pdf_path = tmp_path / "calm.pdf"
    manifest_path = tmp_path / "calm.audit.json"
    render_packet(
        _anomaly(),
        summary,
        _groups(),
        "bracco-example",
        pdf_path,
        manifest_path,
    )
    shown = " ".join(extract_pdf_text(pdf_path).split())
    assert "default Unsure" not in shown
    assert "wind direction unstable under calm conditions" in shown
    assert "openweather" in shown
    assert "confirmed 2026-07-24" in shown
    assert "Bracco" not in shown

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["calm_wind_flags"] == [flags[0].to_dict()]

    report = audit_packet(
        pdf_path,
        manifest_path,
        summary,
        ANOMALY_TS,
        model_names=("llama3:8b", "gpt-5.4"),
        claim_groups=tuple(_groups()),
    )
    assert report.calm_wind_flag_count == 1
    assert report.blinding_leaks == ()


def test_audit_rejects_calm_wind_flag_tampering(tmp_path: Path) -> None:
    summary = _summary()
    speed = summary["sources"]["openweather"]["metrics"]["wind_speed"]
    speed["nearest_in_time"]["v"] = 1.0
    speed["entities"][0]["series"][0][1] = 1.0
    pdf_path = tmp_path / "calm.pdf"
    manifest_path = tmp_path / "calm.audit.json"
    render_packet(
        _anomaly(),
        summary,
        _groups(),
        "bracco-example",
        pdf_path,
        manifest_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["calm_wind_flags"][0]["source_decisions"][0][
        "effective_cutoff_ms"
    ] = 999.0
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PacketAuditError, match="calm-wind"):
        audit_packet(
            pdf_path,
            manifest_path,
            summary,
            ANOMALY_TS,
            model_names=("llama3:8b",),
            claim_groups=tuple(_groups()),
        )
