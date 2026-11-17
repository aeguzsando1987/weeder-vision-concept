"""CLI de caracterizacion (E0) de punta a punta sobre un clip generado; sin video privado."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("av")
pytest.importorskip("cv2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import characterize_videos  # noqa: E402
from precision_weeder.video.manifest import Suitability, read_manifest_csv  # noqa: E402


def _setup(tmp_path: Path, video: Path, **video_overrides: object) -> Path:
    resources = tmp_path / "resources"
    resources.mkdir(exist_ok=True)
    shutil.copy(video, resources / "clip_test.mp4")
    config = {
        "schema_version": 1,
        "mode": "exploratory",
        "video": {
            "source": "resources/clip_test.mp4",
            "output_dir": "experiments/phase0",
            "manifest_path": "data/video_manifest.csv",
            "roi": {"v_min": 0.2, "rationale": "excluye el codigo de indice"},
            "sampling": {"uniform_count": 12, "analysis_scale": 1.0, "min_separation_frames": 2},
            "flow": {"min_valid_points": 10, "example_count": 2, "max_corners": 200},
            **video_overrides,
        },
    }
    path = tmp_path / "video.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(tmp_path: Path, config: Path, *extra: str) -> int:
    return characterize_videos.main(["--config", str(config), "--root", str(tmp_path), *extra])


def test_cli_characterizes_clip_and_leaves_suitability_pending(
    tmp_path: Path, synthetic_video: Path
) -> None:
    assert _run(tmp_path, _setup(tmp_path, synthetic_video)) == 0
    (record,) = read_manifest_csv(
        tmp_path / "data/video_manifest.csv", base_dir=tmp_path, verify_hash=True
    )
    assert record.suitability is Suitability.PENDING  # nunca A por defecto
    assert record.stem_base_visibility.value == "unknown"
    assert record.mounting_status.value == "unknown"
    assert record.scale_status.value == "unknown"
    assert record.frame_count_decoded == 24 == record.frame_count_reported
    assert record.fps_effective == pytest.approx(15.0, abs=1e-9, rel=0)
    assert record.apparent_motion.value == "up"
    out = tmp_path / "experiments/phase0/clip_test"
    results = json.loads((out / "characterization.json").read_text())["results"]
    assert results["flow_summary"]["median_dv_px"] == pytest.approx(-4.0, abs=0.5, rel=0)
    assert abs(results["flow_summary"]["median_du_px"]) < 0.5
    assert results["decode"]["eof_status"] == "eof_reached"
    assert len(results["samples"]) >= 12
    for sample in results["samples"]:
        assert (tmp_path / sample["image"]).is_file()
    for name in (
        "contact_sheet.jpg",
        "timestamps.csv",
        "flow.csv",
        "flow_timeseries.png",
        "derived_manifest.json",
        "run_clip_test.json",
        "timestamp_anomalies.csv",
    ):
        assert (out / name).is_file(), name
    derived = json.loads((out / "derived_manifest.json").read_text())
    assert not any("run_" in key for key in derived)


def test_cli_results_are_deterministic(tmp_path: Path, synthetic_video: Path) -> None:
    config = _setup(tmp_path, synthetic_video)
    out = tmp_path / "experiments/phase0/clip_test"
    assert _run(tmp_path, config) == 0
    first_results = (out / "characterization.json").read_text()
    first_derived = (out / "derived_manifest.json").read_text()
    first_manifest = (tmp_path / "data/video_manifest.csv").read_text()
    assert _run(tmp_path, config) == 0
    assert (out / "characterization.json").read_text() == first_results
    assert (out / "derived_manifest.json").read_text() == first_derived
    assert (tmp_path / "data/video_manifest.csv").read_text() == first_manifest


@pytest.mark.parametrize("case", ["missing", "corrupt", "hash_mismatch", "unknown_key"])
def test_cli_rejects_invalid_input_without_manifest_row(
    tmp_path: Path, synthetic_video: Path, case: str, capsys: pytest.CaptureFixture[str]
) -> None:
    overrides: dict[str, object] = {}
    if case == "hash_mismatch":
        overrides["expected_sha256"] = "0" * 64
    if case == "unknown_key":
        overrides["clave_desconocida"] = 1
    config = _setup(tmp_path, synthetic_video, **overrides)
    if case == "missing":
        (tmp_path / "resources/clip_test.mp4").unlink()
    if case == "corrupt":
        (tmp_path / "resources/clip_test.mp4").write_bytes(b"\x00junk" * 300)
    assert _run(tmp_path, config) == 2
    assert "error" in capsys.readouterr().err
    assert not (tmp_path / "data/video_manifest.csv").exists()


def test_cli_rejects_class_a_not_backed_by_observations(
    tmp_path: Path, synthetic_video: Path
) -> None:
    config = _setup(tmp_path, synthetic_video)
    out = tmp_path / "experiments/phase0/clip_test"
    out.mkdir(parents=True)
    source = yaml.safe_load(
        (PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/manual_assessment.yaml").read_text()
    )
    observations = out / "stem_observations.csv"
    observations.write_text(
        (PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/stem_observations.csv")
        .read_text()
        .splitlines()[0]
        + "\n"
    )
    source.update(
        video_id="clip_test",
        stem_observations_ref="experiments/phase0/clip_test/stem_observations.csv",
        suitability="A",
        apparent_motion_manual="up",
    )
    (out / "manual_assessment.yaml").write_text(yaml.safe_dump(source, allow_unicode=True))
    assert _run(tmp_path, config) == 2
    assert not (tmp_path / "data/video_manifest.csv").exists()


def _small_observations(path: Path) -> None:
    """Tres plantas dentro del cuadro de 192x128 que cruzan v=64 hacia arriba."""
    header = (
        (PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/stem_observations.csv")
        .read_text()
        .splitlines()[0]
    )
    rows = [header]
    for plant, u in (("P01", 40), ("P02", 96), ("P03", 150)):
        for k, (frame, v) in enumerate(((0, 100.0), (12, 52.0))):
            rows.append(
                f"{plant}_{k},{plant},{frame},{frame / 15:.6f},pts,partial,stem_axis_proxy,"
                f"{u},{v},4,medium,eje del tallo,manual_visual,prueba"
            )
    path.write_text("\n".join(rows) + "\n")


def test_cli_incorporates_a_valid_manual_assessment(tmp_path: Path, synthetic_video: Path) -> None:
    config = _setup(tmp_path, synthetic_video)
    out = tmp_path / "experiments/phase0/clip_test"
    out.mkdir(parents=True)
    _small_observations(out / "stem_observations.csv")
    source = yaml.safe_load(
        (PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/manual_assessment.yaml").read_text()
    )
    source.update(
        video_id="clip_test",
        video_sha256=_sha256(tmp_path / "resources/clip_test.mp4"),
        stem_observations_sha256=_sha256(out / "stem_observations.csv"),
        stem_observations_ref="experiments/phase0/clip_test/stem_observations.csv",
        # Intervalos, condiciones y contrastes del clip real no caben en 24 cuadros.
        intervals=[{"start_frame": 0, "end_frame": 23, "suitability": "B", "reason": "prueba"}],
        conditions=[],
        flow_manual_checks=[],
        crossing_line={
            "v_px": 64.0,
            "u_min_px": 0.0,
            "u_max_px": 192.0,
            "crossing_direction": "up",
            "rationale": "centro del cuadro de prueba",
        },
    )
    (out / "manual_assessment.yaml").write_text(yaml.safe_dump(source, allow_unicode=True))
    assert _run(tmp_path, config) == 0
    (record,) = read_manifest_csv(tmp_path / "data/video_manifest.csv", base_dir=tmp_path)
    assert record.suitability is Suitability.B
    assert "manual_assessment.yaml" in (record.evidence_ref or "")
    results = json.loads((out / "characterization.json").read_text())["results"]
    events = {e["plant_id"]: e for e in results["manual"]["crossing_events"]}
    assert set(events) == {"P01", "P02", "P03"}
    # Oraculo: v 100 -> 52 entre t=0 y t=0.8 s; cruza 64 en t = 0.8 * 36/48 = 0.6 s.
    assert events["P01"]["crossing_time_s"] == pytest.approx(0.6, abs=1e-9, rel=0)
    # Es el cruce de un proxy de eje, con su intervalo de soporte y su incertidumbre.
    assert events["P01"]["observed_target"] == "stem_axis_proxy"
    assert (events["P01"]["support_from_s"], events["P01"]["support_to_s"]) == pytest.approx(
        (0.0, 0.8), abs=1e-6, rel=0
    )
    # 4 px de lectura a 48 px / 0.8 s => 4 * 0.8 / 48 s.
    assert events["P01"]["time_uncertainty_s"] == pytest.approx(4 * 0.8 / 48, abs=1e-6, rel=0)
    assert results["manual"]["tracked_plants_ground_contact"] == {}
    assert set(results["manual"]["tracked_plants_stem_axis_proxy"]) == {"P01", "P02", "P03"}


def test_cli_rejects_observations_outside_the_frame(
    tmp_path: Path, synthetic_video: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Las posiciones del clip real (1920x1080) no caben en el clip de prueba: error."""
    config = _setup(tmp_path, synthetic_video)
    out = tmp_path / "experiments/phase0/clip_test"
    out.mkdir(parents=True)
    shutil.copy(
        PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/stem_observations.csv",
        out / "stem_observations.csv",
    )
    source = yaml.safe_load(
        (PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/manual_assessment.yaml").read_text()
    )
    source.update(
        video_id="clip_test",
        video_sha256=_sha256(tmp_path / "resources/clip_test.mp4"),
        stem_observations_ref="experiments/phase0/clip_test/stem_observations.csv",
    )
    (out / "manual_assessment.yaml").write_text(yaml.safe_dump(source, allow_unicode=True))
    assert _run(tmp_path, config) == 2
    error = capsys.readouterr().err
    assert "fuera de la imagen" in error
    # Nada publicado: la validacion contra el clip ocurre antes de escribir derivados.
    assert not (out / "characterization.json").exists()
    assert not (tmp_path / "data/video_manifest.csv").exists()
