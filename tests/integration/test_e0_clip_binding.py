"""Regresiones de E0-R2 (dictamen E0 rev. 1): la evaluacion manual se vincula al clip.

Sobre el clip generado (24 cuadros, 15 fps, 192x128) se construye una evaluacion A
coherente que se acepta (control positivo). Cada corrupcion cambia un unico aspecto
y debe rechazarse por esa causa, antes de publicar derivados, tanto en el
caracterizador como en el renderer de evidencia.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("av")
pytest.importorskip("cv2")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import render_stem_observations  # noqa: E402
from precision_weeder.core.config import VideoConfig  # noqa: E402
from precision_weeder.core.errors import ConfigError, PrecisionWeederError  # noqa: E402
from precision_weeder.video.characterize import characterize_video  # noqa: E402
from precision_weeder.video.manifest import Suitability  # noqa: E402
from precision_weeder.video.observations import OBSERVATION_COLUMNS  # noqa: E402

FPS = 15


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(target: str = "ground_contact") -> list[dict[str, Any]]:
    """Tres plantas que cruzan v=64 hacia arriba entre los cuadros 2 y 3."""
    rows = []
    for plant in range(3):
        for index, v in ((2, 70.0), (3, 50.0)):
            rows.append(
                {
                    "event_id": f"p{plant}_{index}",
                    "plant_id": f"p{plant}",
                    "frame_index": index,
                    # Redondeo a 6 decimales, como la tabla real: debe aceptarse.
                    "timestamp_s": f"{index / FPS:.6f}",
                    "timestamp_method": "pts",
                    "base_visibility": "visible",
                    "observed_target": target,
                    "u_px": 30 + plant * 60,
                    "v_px": v,
                    "uncertainty_px": 2,
                    "confidence": "high",
                    "ambiguity_cause": "none",
                    "source": "manual_visual",
                    "notes": "fixture",
                }
            )
    return rows


class Case:
    """Clip de prueba con evaluacion coherente y ganchos para corromperla."""

    def __init__(self, root: Path, video: Path) -> None:
        self.root = root
        self.video = root / "clip.mp4"
        shutil.copy(video, self.video)
        self.out = root / "out" / "clip"
        self.table = root / "observations.csv"
        self.assessment = root / "assessment.yaml"

    def write(
        self,
        rows: list[dict[str, Any]] | None = None,
        **overrides: Any,
    ) -> Path:
        with self.table.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
            writer.writeheader()
            writer.writerows(rows if rows is not None else _rows())
        payload = yaml.safe_load(
            (PROJECT_ROOT / "experiments/phase0/moringa_sin_ai/manual_assessment.yaml").read_text()
        )
        payload.update(
            video_id="clip",
            video_sha256=_sha256(self.video),
            stem_observations_ref=self.table.name,
            stem_observations_sha256=_sha256(self.table),
            suitability="A",
            replay_suitability="A",
            suitability_reason="tres bases visibles en el clip de prueba",
            apparent_motion_manual="unknown",
            flow_manual_checks=[],
            conditions=[],
            intervals=[{"start_frame": 0, "end_frame": 23, "suitability": "A", "reason": "t"}],
            crossing_line={
                "v_px": 64.0,
                "u_min_px": 0.0,
                "u_max_px": 192.0,
                "crossing_direction": "up",
                "rationale": "centro",
            },
        )
        payload.update(overrides)
        self.assessment.write_text(yaml.safe_dump(payload, allow_unicode=True))
        return self.assessment

    def characterize(self, **timestamps: Any) -> Any:
        config = VideoConfig(
            source="clip.mp4",
            output_dir="out",
            manifest_path="manifest.csv",
            timestamps=timestamps,
            sampling={"analysis_scale": 1.0, "uniform_count": 12, "min_separation_frames": 1},
            flow={"example_count": 0, "min_valid_points": 10},
        )
        return characterize_video(self.video, config, self.root, self.assessment)


@pytest.fixture
def case(tmp_path: Path, synthetic_video: Path) -> Case:
    return Case(tmp_path, synthetic_video)


def test_coherent_assessment_with_rounded_times_is_accepted(case: Case) -> None:
    case.write()
    result = case.characterize()
    assert result.record.suitability is Suitability.A
    assert (case.out / "characterization.json").is_file()


def _shift_frame(rows: list[dict[str, Any]]) -> None:
    rows[0]["frame_index"] = 1002


def _time_far(rows: list[dict[str, Any]]) -> None:
    rows[0]["timestamp_s"] = 102.0


def _time_one_frame_off(rows: list[dict[str, Any]]) -> None:
    # El tiempo del cuadro vecino: un error de cuadro, no de redondeo.
    rows[0]["timestamp_s"] = f"{3 / FPS:.6f}"


def _estimated_method(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["timestamp_method"] = "index_fps_estimate"


def _point_outside(rows: list[dict[str, Any]]) -> None:
    rows[0]["u_px"] = 500


ROW_CORRUPTIONS: dict[str, tuple[Callable[[list[dict[str, Any]]], None], str]] = {
    "frame_index": (_shift_frame, "fuera del clip"),
    "timestamp_far": (_time_far, "no corresponde al cuadro"),
    "timestamp_neighbor_frame": (_time_one_frame_off, "no corresponde al cuadro"),
    "timestamp_method": (_estimated_method, "timestamp_method"),
    "point_outside": (_point_outside, "fuera de la imagen"),
}


@pytest.mark.parametrize("name", sorted(ROW_CORRUPTIONS))
def test_table_incompatible_with_clip_is_rejected_before_publishing(case: Case, name: str) -> None:
    corrupt, reason = ROW_CORRUPTIONS[name]
    rows = _rows()
    corrupt(rows)
    case.write(rows)
    with pytest.raises(ConfigError, match=reason):
        case.characterize()
    assert not case.out.exists()


ASSESSMENT_CORRUPTIONS: dict[str, tuple[dict[str, Any], str]] = {
    "video_id": ({"video_id": "wrong_clip"}, "video_id"),
    "video_sha256": ({"video_sha256": "0" * 64}, "video_sha256"),
    "crossing_line_below_image": (
        {
            "crossing_line": {
                "v_px": 400.0,
                "u_min_px": 0.0,
                "u_max_px": 192.0,
                "crossing_direction": "up",
                "rationale": "fuera",
            }
        },
        "linea virtual",
    ),
    "crossing_line_too_wide": (
        {
            "crossing_line": {
                "v_px": 64.0,
                "u_min_px": 0.0,
                "u_max_px": 1920.0,
                "crossing_direction": "up",
                "rationale": "fuera",
            }
        },
        "linea virtual",
    ),
    "interval_beyond_clip": (
        {"intervals": [{"start_frame": 0, "end_frame": 971, "suitability": "A", "reason": "t"}]},
        "intervalo",
    ),
    "overlapping_intervals": (
        {
            "intervals": [
                {"start_frame": 0, "end_frame": 12, "suitability": "A", "reason": "t"},
                {"start_frame": 10, "end_frame": 23, "suitability": "B", "reason": "t"},
            ]
        },
        "solapa",
    ),
    "condition_frame_beyond_clip": (
        {"conditions": [{"condition": "blur", "observed": "no", "frames": [400], "notes": "t"}]},
        "condicion",
    ),
    "flow_check_outside": (
        {
            "flow_manual_checks": [
                {
                    "check_id": "c",
                    "feature": "f",
                    "frame_from": 0,
                    "frame_to": 6,
                    "uv_from_px": [1278.0, 555.0],
                    "uv_to_px": [1275.0, 386.0],
                    "uncertainty_px": 3.0,
                    "automated_median_du_px": 0.0,
                    "automated_median_dv_px": -4.0,
                    "notes": "",
                }
            ]
        },
        "contraste",
    ),
}


@pytest.mark.parametrize("name", sorted(ASSESSMENT_CORRUPTIONS))
def test_assessment_incompatible_with_clip_is_rejected(case: Case, name: str) -> None:
    overrides, reason = ASSESSMENT_CORRUPTIONS[name]
    case.write(**overrides)
    with pytest.raises(ConfigError, match=reason):
        case.characterize()
    assert not case.out.exists()


def test_changed_video_content_is_rejected(case: Case, video_fixture_module: Any) -> None:
    """Mismo nombre y mismos cuadros validos, distinto contenido: el hash lo detecta."""
    case.write()
    video_fixture_module.write_video(case.video, frames=24, fps=FPS, shift_px=3)
    with pytest.raises(ConfigError, match="video_sha256"):
        case.characterize()


def test_changed_table_is_rejected_by_its_hash(case: Case) -> None:
    case.write()
    case.table.write_text(case.table.read_text().replace(",70.0,", ",71.0,", 1))
    with pytest.raises(ConfigError, match="sha256"):
        case.characterize()


def test_tolerance_that_would_accept_the_wrong_frame_is_rejected(case: Case) -> None:
    case.write()
    with pytest.raises(ConfigError, match="medio intervalo"):
        case.characterize(annotation_tolerance_s=0.04)


def test_proxy_points_cannot_back_class_a(case: Case) -> None:
    case.write(_rows("stem_axis_proxy"))
    with pytest.raises(ConfigError, match="ground_contact"):
        case.characterize()
    case.write(_rows("stem_axis_proxy"), suitability="B", replay_suitability="B")
    result = case.characterize()
    assert result.record.suitability is Suitability.B


def test_schema_version_1_requires_explicit_migration(case: Case) -> None:
    case.write(schema_version=1)
    with pytest.raises(ConfigError, match="Migre a la version 2"):
        case.characterize()


# --- renderer: misma validacion antes de escribir evidencia -------------------------


def _render(case: Case, video: Path) -> int:
    return render_stem_observations.main(
        ["--assessment", str(case.assessment), "--video", str(video), "--root", str(case.root)]
    )


def test_renderer_accepts_the_bound_clip(case: Case) -> None:
    case.write()
    assert _render(case, case.video) == 0
    assert sorted(p.name for p in (case.root / "manual_evidence").glob("strip_*.jpg")) == [
        "strip_p0.jpg",
        "strip_p1.jpg",
        "strip_p2.jpg",
    ]


@pytest.mark.parametrize("corruption", ["other_video", "frame_index"])
def test_renderer_rejects_unbound_input_without_writing(
    case: Case, corruption: str, capsys: pytest.CaptureFixture[str]
) -> None:
    rows = _rows()
    if corruption == "frame_index":
        _shift_frame(rows)
    case.write(rows)
    video = case.video
    if corruption == "other_video":
        video = case.root / "clip_copy.mp4"
        shutil.copy(case.video, video)  # mismo contenido, otro ID
    evidence = case.root / "manual_evidence"
    evidence.mkdir()
    (evidence / "previous.jpg").write_bytes(b"x")
    assert _render(case, video) == 2
    assert "incompatible" in capsys.readouterr().err
    assert [p.name for p in evidence.iterdir()] == ["previous.jpg"]


def test_load_without_clip_context_is_not_publication(case: Case) -> None:
    """`load` valida esquema y tabla; las incoherencias con el clip las detecta la
    validacion contra cuadros decodificados, que `characterize_video` siempre aplica."""
    from precision_weeder.video.observations import ManualAssessment

    rows = _rows()
    _shift_frame(rows)
    case.write(rows)
    assessment, observations = ManualAssessment.load(case.assessment, case.root)
    assert assessment.suitability is Suitability.A and len(observations) == 6
    with pytest.raises(PrecisionWeederError):
        case.characterize()
