"""Manifiesto de videos (E0 §6): esquema, nulos explicitos, referencias y aptitud."""

from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path

import pytest

from precision_weeder.core.errors import SchemaValidationError
from precision_weeder.video.manifest import (
    MANIFEST_COLUMNS,
    UNKNOWN,
    StemBaseVisibility,
    Suitability,
    VideoManifestRecord,
    read_manifest_csv,
    write_manifest_csv,
)

SHA = "a" * 64


def _record(**overrides: object) -> VideoManifestRecord:
    values: dict[str, object] = {
        "video_id": "clip_01",
        "relative_path": "resources/clip_01.mp4",
        "sha256": SHA,
        "size_bytes": 1000,
        "session_id": None,
        "site_id": None,
        "container": "mov,mp4",
        "codec": "h264",
        "width_px": 1920,
        "height_px": 1080,
        "rotation_deg": 0.0,
        "duration_reported_s": 10.0,
        "duration_observed_s": 10.0,
        "frame_count_reported": 300,
        "frame_count_decoded": 300,
        "fps_nominal": 30.0,
        "fps_effective": 30.0,
        "timestamp_source": "pts",
        "time_base": "1/15360",
        "timestamp_anomalies_ref": "experiments/phase0/clip_01/timestamp_anomalies.csv",
        "stem_base_visibility": "unknown",
        "apparent_motion": "unknown",
        "mounting_status": "unknown",
        "scale_status": "unknown",
        "scale_evidence_ref": None,
        "homography_feasibility": "unknown",
        "observation_timing": "unknown",
        "suitability": "pending",
        "suitability_reason": "",
        "evidence_ref": None,
        "decoder_version": "pyav test",
        "config_hash": "b" * 64,
    }
    values.update(overrides)
    return VideoManifestRecord(**values)  # type: ignore[arg-type]


def test_columns_match_the_order_exactly() -> None:
    assert MANIFEST_COLUMNS[0] == "video_id"
    assert len(MANIFEST_COLUMNS) == 32
    assert {f.name for f in dataclasses.fields(VideoManifestRecord)} == set(MANIFEST_COLUMNS)


def test_round_trip_preserves_values_and_unknowns(tmp_path: Path) -> None:
    record = _record()
    path = tmp_path / "manifest.csv"
    write_manifest_csv([record], path)
    text = path.read_text()
    assert ",unknown," in text  # session_id/site_id desconocidos, no vacios ni cero
    (loaded,) = read_manifest_csv(path)
    assert loaded == record
    assert loaded.session_id is None and loaded.scale_evidence_ref is None


def test_writing_is_deterministic_and_sorted(tmp_path: Path) -> None:
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    records = [_record(video_id="z_clip"), _record(video_id="a_clip")]
    write_manifest_csv(records, first)
    write_manifest_csv(list(reversed(records)), second)
    assert first.read_bytes() == second.read_bytes()
    assert [r.video_id for r in read_manifest_csv(first)] == ["a_clip", "z_clip"]


@pytest.mark.parametrize("field", ["frame_count_reported", "width_px", "size_bytes"])
def test_zero_is_not_unknown(field: str) -> None:
    with pytest.raises(SchemaValidationError, match="unknown"):
        _record(**{field: 0})


@pytest.mark.parametrize("field", ["fps_nominal", "duration_observed_s", "fps_effective"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_non_finite_or_non_positive_rates_are_rejected(field: str, value: float) -> None:
    with pytest.raises(SchemaValidationError):
        _record(**{field: value})


def test_rotation_must_be_a_multiple_of_ninety() -> None:
    _record(rotation_deg=-90.0)
    with pytest.raises(SchemaValidationError):
        _record(rotation_deg=45.0)


@pytest.mark.parametrize("sha", ["A" * 64, "a" * 63, "g" * 64, ""])
def test_hashes_are_validated(sha: str) -> None:
    with pytest.raises(SchemaValidationError):
        _record(sha256=sha)
    with pytest.raises(SchemaValidationError):
        _record(config_hash=sha)


@pytest.mark.parametrize(
    "path", ["/abs/clip.mp4", "../outside/clip.mp4", "resources/../../x.mp4", "", "unknown"]
)
def test_paths_must_be_relative_and_contained(path: str) -> None:
    with pytest.raises(SchemaValidationError):
        _record(relative_path=path)
    with pytest.raises(SchemaValidationError):
        _record(evidence_ref=f"experiments/ok.json;{path}" if path else ";")


@pytest.mark.parametrize("enum_field", ["apparent_motion", "suitability", "scale_status"])
def test_unknown_enum_values_are_rejected(enum_field: str) -> None:
    with pytest.raises(ValueError):
        _record(**{enum_field: "certainly"})


def test_pending_is_the_only_state_without_reason_or_evidence() -> None:
    assert _record().suitability is Suitability.PENDING
    for value in ("A", "B", "C"):
        with pytest.raises(SchemaValidationError, match="motivo"):
            _record(suitability=value, evidence_ref="experiments/e.json")
        with pytest.raises(SchemaValidationError, match="evidencia"):
            _record(suitability=value, suitability_reason="motivo")


def test_class_a_requires_observed_bases_and_decoded_frames() -> None:
    base = {"suitability": "A", "suitability_reason": "r", "evidence_ref": "experiments/e.json"}
    with pytest.raises(SchemaValidationError, match="bases"):
        _record(**base, stem_base_visibility="unknown")
    with pytest.raises(SchemaValidationError, match="bases"):
        _record(**base, stem_base_visibility="not_visible")
    with pytest.raises(SchemaValidationError, match="cuadros"):
        _record(**base, stem_base_visibility="visible", frame_count_decoded=0)
    record = _record(**base, stem_base_visibility="partial")
    assert record.stem_base_visibility is StemBaseVisibility.PARTIAL


def test_class_c_is_allowed_for_undecodable_clips() -> None:
    record = _record(
        suitability="C",
        suitability_reason="no decodifica",
        evidence_ref="experiments/e.json",
        frame_count_decoded=0,
    )
    assert record.suitability is Suitability.C


def test_verified_scale_needs_evidence() -> None:
    with pytest.raises(SchemaValidationError, match="escala"):
        _record(scale_status="verified")


def test_reading_rejects_bad_headers_and_cells(tmp_path: Path) -> None:
    path = tmp_path / "m.csv"
    write_manifest_csv([_record()], path)
    header, row = path.read_text().splitlines()
    (tmp_path / "extra.csv").write_text(header + ",extra\n" + row + ",1\n")
    with pytest.raises(SchemaValidationError):
        read_manifest_csv(tmp_path / "extra.csv")
    (tmp_path / "empty.csv").write_text(header + "\n" + row.replace(",unknown,", ",,", 1) + "\n")
    with pytest.raises(SchemaValidationError, match="vacia"):
        read_manifest_csv(tmp_path / "empty.csv")
    (tmp_path / "nan.csv").write_text(header + "\n" + row.replace(",30.0,", ",nan,", 1) + "\n")
    with pytest.raises(SchemaValidationError):
        read_manifest_csv(tmp_path / "nan.csv")


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SchemaValidationError, match="duplicado"):
        write_manifest_csv([_record(), _record()], tmp_path / "m.csv")


def test_references_must_resolve_and_hash_must_match(tmp_path: Path) -> None:
    video = tmp_path / "resources" / "clip_01.mp4"
    video.parent.mkdir()
    video.write_bytes(b"contenido de prueba")
    anomalies = tmp_path / "experiments/phase0/clip_01/timestamp_anomalies.csv"
    anomalies.parent.mkdir(parents=True)
    anomalies.write_text("kind\n")
    digest = hashlib.sha256(b"contenido de prueba").hexdigest()
    good = _record(sha256=digest)
    good.check_references(tmp_path, verify_hash=True)
    with pytest.raises(SchemaValidationError, match="sha256"):
        _record().check_references(tmp_path, verify_hash=True)
    with pytest.raises(SchemaValidationError, match="no resuelta"):
        _record(sha256=digest, evidence_ref="experiments/missing.json").check_references(tmp_path)
    path = tmp_path / "m.csv"
    write_manifest_csv(
        [
            _record(
                sha256=digest,
                evidence_ref="experiments/nope.json",
                suitability="B",
                suitability_reason="r",
            )
        ],
        path,
    )
    with pytest.raises(SchemaValidationError):
        read_manifest_csv(path, base_dir=tmp_path)


def test_repository_manifest_is_valid_and_resolvable(project_root: Path) -> None:
    """El manifiesto real se valida y sus referencias resuelven (sin decodificar video)."""
    path = project_root / "data" / "video_manifest.csv"
    if not path.is_file():
        pytest.skip("manifiesto aun no generado")
    records = read_manifest_csv(path)
    for record in records:
        assert record.suitability is not Suitability.A or record.stem_base_visibility not in (
            StemBaseVisibility.NOT_VISIBLE,
            StemBaseVisibility.UNKNOWN,
        )
        refs = [record.timestamp_anomalies_ref, record.evidence_ref]
        for ref in refs:
            for part in (ref or "").split(";"):
                if part:
                    assert (project_root / part).exists(), part
        assert UNKNOWN not in (record.sha256,)
