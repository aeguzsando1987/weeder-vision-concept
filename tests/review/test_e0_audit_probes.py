"""Sondas de auditoria E0 del revisor tecnico, integradas en la suite.

Origen: la evidencia de la revision (no versionada) (E0-R1
integridad temporal, E0-R2 vinculacion evaluacion-clip), conservado sin modificar.
Unica adaptacion: `ROOT` se calcula desde la nueva ubicacion de este archivo (el
formato lo normaliza Ruff solo en esta copia). Las causas especificas de rechazo de
cada corrupcion, con control positivo, estan en `tests/integration/test_e0_clip_binding.py`.
"""

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

from precision_weeder.core.config import VideoConfig
from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.video.characterize import characterize_video
from precision_weeder.video.observations import OBSERVATION_COLUMNS
from precision_weeder.video.timing import analyze_timestamps

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "values,kind", [([0.0, 0.0, 0.0], "duplicate"), ([2.0, 1.0, 0.0], "regression")]
)
def test_nonpositive_intervals_keep_anomaly_rows(values, kind):
    report = analyze_timestamps(values)
    assert len(report.anomalies) == 2
    assert all(row["kind"] == kind for row in report.anomalies)


def test_missing_pts_does_not_erase_observed_regression():
    try:
        report = analyze_timestamps([0.0, 0.1, 0.05, None, 0.2], fps_nominal=10.0)
    except PrecisionWeederError:
        return
    assert any(row["kind"] == "regression" for row in report.anomalies)


def test_missing_pts_does_not_hide_nonfinite_pts():
    with pytest.raises(PrecisionWeederError):
        analyze_timestamps([0.0, float("nan"), None, 0.3], fps_nominal=10.0)


@pytest.mark.parametrize("corruption", ["video_id", "frame_index", "timestamp"])
def test_assessment_must_match_decoded_clip(tmp_path, corruption):
    spec = importlib.util.spec_from_file_location(
        "audit_fixture", ROOT / "tests/fixtures/video_fixture.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    video = module.write_video(tmp_path / "clip.mp4")
    rows = []
    for plant in range(3):
        for index, v in [(2, 70.0), (3, 50.0)]:
            rows.append(
                dict(
                    event_id=f"p{plant}_{index}",
                    plant_id=f"p{plant}",
                    frame_index=1000 + index if corruption == "frame_index" else index,
                    timestamp_s=100 + index if corruption == "timestamp" else index / 15,
                    timestamp_method="pts",
                    base_visibility="visible",
                    u_px=30 + plant * 30,
                    v_px=v,
                    uncertainty_px=2,
                    confidence="high",
                    ambiguity_cause="none",
                    source="manual_visual",
                    notes="fixture",
                )
            )
    with (tmp_path / "observations.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    assessment = yaml.safe_load(
        (ROOT / "experiments/phase0/moringa_sin_ai/manual_assessment.yaml").read_text()
    )
    assessment.update(
        video_id="wrong_clip" if corruption == "video_id" else "clip",
        stem_observations_ref="observations.csv",
        suitability="A",
        replay_suitability="A",
        suitability_reason="three visible bases in fixture",
        apparent_motion_manual="unknown",
        flow_manual_checks=[],
        conditions=[],
        intervals=[],
        crossing_line=dict(
            v_px=64, u_min_px=0, u_max_px=192, crossing_direction="up", rationale="fixture"
        ),
    )
    path = tmp_path / "assessment.yaml"
    path.write_text(yaml.safe_dump(assessment))
    config = VideoConfig(
        source="clip.mp4",
        output_dir="out",
        manifest_path="manifest.csv",
        sampling={"analysis_scale": 1.0, "uniform_count": 12},
        flow={"example_count": 0},
    )
    with pytest.raises(PrecisionWeederError):
        characterize_video(video, config, tmp_path, path)
