"""Tiempo, muestreo, flujo aparente y observaciones manuales (E0), con oraculos conocidos."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from precision_weeder.core.config import FlowConfig, RoiConfig, VideoFile
from precision_weeder.core.errors import (
    ConfigError,
    PrecisionWeederError,
    SchemaValidationError,
    TemporalOrderError,
)
from precision_weeder.video.manifest import ApparentMotion, Suitability
from precision_weeder.video.observations import (
    OBSERVATION_COLUMNS,
    CrossingLine,
    FlowManualCheck,
    ManualAssessment,
    StemBaseObservation,
    crossing_events,
    read_stem_observations,
)
from precision_weeder.video.sampling import select_samples
from precision_weeder.video.timing import analyze_timestamps

# --- tiempo -------------------------------------------------------------------


def test_regular_pts_give_fps_and_observed_duration() -> None:
    report = analyze_timestamps([i / 30 for i in range(300)])
    assert report.source == "pts"
    assert report.fps_effective == pytest.approx(30.0, abs=1e-9, rel=0)
    assert report.duration_observed_s == pytest.approx(10.0, abs=1e-9, rel=0)
    assert report.monotonic_increasing and report.gaps == report.duplicates == 0


def test_gap_duplicate_and_regression_are_reported_not_normalized() -> None:
    times = [0.0, 0.1, 0.2, 0.2, 0.15, 0.6, 0.7]
    report = analyze_timestamps(times, gap_factor=1.5)
    kinds = sorted(a["kind"] for a in report.anomalies)
    assert kinds == ["duplicate", "gap", "regression"]
    assert not report.monotonic_increasing
    assert report.duplicates == 1 and report.regressions == 1 and report.gaps == 1


def test_missing_pts_use_labeled_index_estimate_only_when_allowed() -> None:
    report = analyze_timestamps([0.0, None, 0.2], fps_nominal=10.0)
    assert report.source == "index_fps_estimate"
    assert "index_fps_estimate" in report.fps_effective_method
    with pytest.raises(TemporalOrderError):
        analyze_timestamps([0.0, None], fps_nominal=10.0, allow_index_estimate=False)
    with pytest.raises(TemporalOrderError):
        analyze_timestamps([0.0, None], fps_nominal=None)


@pytest.mark.parametrize("bad", [[], [0.0, float("nan")], [0.0, float("inf")]])
def test_empty_or_non_finite_timestamps_are_rejected(bad: list[float]) -> None:
    with pytest.raises(TemporalOrderError):
        analyze_timestamps(bad)


# --- muestreo -------------------------------------------------------------------


def test_sampling_is_deterministic_and_includes_extremes() -> None:
    values = [float((i * 37) % 101) for i in range(300)]
    criteria = [("low", values, lambda v: v), ("high", values, lambda v: -v)]
    first = select_samples(300, 12, criteria, 2, 15)
    second = select_samples(300, 12, criteria, 2, 15)
    assert first == second
    indices = [s.index for s in first]
    assert indices[0] == 0 and indices[-1] == 299
    assert sum("uniform" in r for s in first for r in s.reasons) == 12
    problem = [s.index for s in first if not any(r.startswith("uniform") for r in s.reasons)]
    for index in problem:
        others = [i for i in indices if i != index]
        assert min(abs(index - o) for o in others) >= 15


def test_sampling_handles_short_clips_and_none_values() -> None:
    samples = select_samples(5, 12, [("x", [None, 1.0, None, 3.0, None], lambda v: v)], 2, 0)
    assert [s.index for s in samples] == [0, 1, 2, 3, 4]
    assert select_samples(0, 12, [], 2, 0) == []


# --- flujo -------------------------------------------------------------------


def test_classify_direction_oracles() -> None:
    from precision_weeder.video.apparent_motion import classify_direction

    up = np.full(50, -5.0)
    zero = np.zeros(50)
    assert classify_direction(zero, up, 0.7, 0.5)[0] == "up"
    assert classify_direction(zero, -up, 0.7, 0.5)[0] == "down"
    assert classify_direction(up, zero, 0.7, 0.5)[0] == "left"
    assert classify_direction(-up, zero, 0.7, 0.5)[0] == "right"
    assert classify_direction(zero + 0.1, zero, 0.7, 0.5)[0] == "none"
    mixed = np.concatenate([np.full(30, -5.0), np.full(20, 5.0)])
    assert classify_direction(zero, mixed, 0.7, 0.5)[0] == "mixed"
    assert classify_direction(np.array([]), np.array([]), 0.7, 0.5)[0] == "unknown"


@pytest.mark.parametrize(("shift_u", "shift_v"), [(0, -8), (0, 6), (5, 0), (3, -4)])
def test_flow_recovers_a_known_translation(shift_u: int, shift_v: int) -> None:
    """Oraculo: textura trasladada un numero entero de pixeles, sin compresion."""
    cv2 = pytest.importorskip("cv2")
    from precision_weeder.video.apparent_motion import estimate_flow

    rng = np.random.default_rng(3)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (300, 400)).astype(np.float32), (0, 0), 2)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    previous = texture[50:250, 50:350]
    current = texture[50 - shift_v : 250 - shift_v, 50 - shift_u : 350 - shift_u]
    bgr = np.dstack([previous] * 3)
    estimate, vectors = estimate_flow(
        previous, current, bgr, FlowConfig(), RoiConfig(), 1.0, 0, 1, 1 / 30
    )
    assert estimate.median_du_px == pytest.approx(shift_u, abs=0.1, rel=0)
    assert estimate.median_dv_px == pytest.approx(shift_v, abs=0.1, rel=0)
    assert estimate.valid_soil >= FlowConfig().min_valid_points
    assert vectors.origins.shape == vectors.displacements.shape


def test_flow_rescales_to_original_pixels() -> None:
    cv2 = pytest.importorskip("cv2")
    from precision_weeder.video.apparent_motion import estimate_flow

    rng = np.random.default_rng(5)
    texture = cv2.GaussianBlur(rng.uniform(0, 255, (300, 400)).astype(np.float32), (0, 0), 2)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # Un punto en la fila 100 de la textura esta en v=50 y luego en v=40: sube 10 px.
    previous, current = texture[50:250, 50:350], texture[60:260, 50:350]
    estimate, _ = estimate_flow(
        previous, current, np.dstack([previous] * 3), FlowConfig(), RoiConfig(), 0.25, 0, 1, None
    )
    assert estimate.median_dv_px == pytest.approx(-40.0, abs=0.4, rel=0)
    assert estimate.direction == "up"


def test_featureless_frames_give_unknown_direction() -> None:
    pytest.importorskip("cv2")
    from precision_weeder.video.apparent_motion import estimate_flow

    flat = np.full((100, 100), 120, np.uint8)
    estimate, _ = estimate_flow(
        flat, flat, np.dstack([flat] * 3), FlowConfig(), RoiConfig(), 1.0, 0, 1, None
    )
    assert estimate.direction == "unknown" and estimate.median_dv_px is None


def test_roi_rejects_empty_box() -> None:
    with pytest.raises(ValueError, match="ROI"):
        RoiConfig(u_min=0.5, u_max=0.5)


def test_repository_video_config_is_valid(project_root: Path) -> None:
    payload = yaml.safe_load((project_root / "configs/video.yaml").read_text())
    config = VideoFile.model_validate(payload).video
    assert config.sampling.uniform_count >= 12
    assert config.expected_sha256 is not None and len(config.expected_sha256) == 64
    payload["video"]["clave_inesperada"] = 1
    with pytest.raises(ValueError):
        VideoFile.model_validate(payload)


# --- observaciones manuales ------------------------------------------------------


def _obs(**kw: object) -> StemBaseObservation:
    values: dict[str, object] = {
        "event_id": "E1",
        "plant_id": "P1",
        "frame_index": 10,
        "timestamp_s": 1.0,
        "timestamp_method": "pts",
        "base_visibility": "partial",
        "observed_target": "stem_axis_proxy",
        "u_px": 100.0,
        "v_px": 600.0,
        "uncertainty_px": 10.0,
        "confidence": "medium",
        "ambiguity_cause": "",
        "source": "manual_visual",
        "notes": "",
    }
    values.update(kw)
    return StemBaseObservation(**values)  # type: ignore[arg-type]


def test_observation_rules() -> None:
    _obs()
    _obs(observed_target="ground_contact")
    _obs(
        base_visibility="hidden", observed_target="none", u_px=None, v_px=None, uncertainty_px=None
    )
    with pytest.raises(SchemaValidationError, match="oculta"):
        _obs(base_visibility="hidden", observed_target="none")
    # El objetivo anotado y la visibilidad deben concordar: oculto <=> sin objetivo.
    with pytest.raises(SchemaValidationError, match="observed_target"):
        _obs(base_visibility="hidden", u_px=None, v_px=None, uncertainty_px=None)
    with pytest.raises(SchemaValidationError, match="observed_target"):
        _obs(observed_target="none")
    with pytest.raises(SchemaValidationError):
        _obs(observed_target="base_guess")
    with pytest.raises(SchemaValidationError):
        _obs(u_px=None)
    with pytest.raises(SchemaValidationError, match="uncertainty"):
        _obs(uncertainty_px=None)
    with pytest.raises(SchemaValidationError, match="detector"):
        _obs(source="yolo")
    with pytest.raises(SchemaValidationError):
        _obs(timestamp_method="frame_index")
    with pytest.raises(SchemaValidationError):
        _obs(timestamp_s=math.nan)


def _write_obs(path: Path, rows: list[dict[str, object]]) -> Path:
    import csv

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OBSERVATION_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _row(
    event: str,
    plant: str,
    frame: int,
    v: float | None,
    vis: str = "partial",
    target: str = "ground_contact",
) -> dict:
    return {
        "event_id": event,
        "plant_id": plant,
        "frame_index": frame,
        "timestamp_s": frame / 30,
        "timestamp_method": "pts",
        "base_visibility": vis,
        "observed_target": "none" if v is None else target,
        "u_px": "unknown" if v is None else 500,
        "v_px": "unknown" if v is None else v,
        "uncertainty_px": "unknown" if v is None else 10,
        "confidence": "medium",
        "ambiguity_cause": "",
        "source": "manual_visual",
        "notes": "",
    }


def test_reading_observations_checks_bounds_and_duplicates(tmp_path: Path) -> None:
    good = _write_obs(tmp_path / "o.csv", [_row("E1", "P1", 0, 700), _row("E2", "P1", 6, 500)])
    assert len(read_stem_observations(good, (1920, 1080))) == 2
    out = _write_obs(tmp_path / "out.csv", [_row("E1", "P1", 0, 2000)])
    with pytest.raises(SchemaValidationError, match="fuera"):
        read_stem_observations(out, (1920, 1080))
    dup = _write_obs(tmp_path / "d.csv", [_row("E1", "P1", 0, 700), _row("E1", "P2", 3, 600)])
    with pytest.raises(SchemaValidationError, match="duplicado"):
        read_stem_observations(dup)
    same = _write_obs(tmp_path / "s.csv", [_row("E1", "P1", 0, 700), _row("E2", "P1", 0, 600)])
    with pytest.raises(SchemaValidationError, match="mismo cuadro"):
        read_stem_observations(same)


def test_crossing_time_is_linearly_interpolated() -> None:
    """Oraculo: v pasa de 700 a 500 entre t=0 y t=0.2; cruza 540 en t=0.16."""
    line = CrossingLine(v_px=540, u_min_px=0, u_max_px=1920, crossing_direction="up", rationale="t")
    observations = [
        _obs(event_id="a", frame_index=0, timestamp_s=0.0, v_px=700.0),
        _obs(event_id="b", frame_index=6, timestamp_s=0.2, v_px=500.0),
        _obs(event_id="c", plant_id="P2", frame_index=0, timestamp_s=0.0, v_px=900.0),
        _obs(event_id="d", plant_id="P2", frame_index=6, timestamp_s=0.2, v_px=800.0),
        _obs(
            event_id="e",
            plant_id="P3",
            frame_index=0,
            timestamp_s=0.0,
            base_visibility="hidden",
            observed_target="none",
            u_px=None,
            v_px=None,
            uncertainty_px=None,
        ),
    ]
    events = {e.plant_id: e for e in crossing_events(observations, line)}
    assert events["P1"].status == "crossed"
    assert events["P1"].crossing_time_s == pytest.approx(0.16, abs=1e-12, rel=0)
    assert events["P2"].status == "not_bracketed"
    assert "P3" not in events


def test_flow_manual_check_agreement_math() -> None:
    check = FlowManualCheck(
        check_id="c",
        feature="f",
        frame_from=0,
        frame_to=10,
        uv_from_px=(100, 600),
        uv_to_px=(110, 340),
        uncertainty_px=5.0,
        automated_median_du_px=1.0,
        automated_median_dv_px=-25.0,
        notes="",
    )
    result = check.agreement()
    assert result["manual_dv_per_frame_px"] == pytest.approx(-26.0, abs=1e-12, rel=0)
    assert result["tolerance_per_frame_px"] == pytest.approx(1.0, abs=1e-12, rel=0)
    assert result["within_tolerance"] is True and result["same_vertical_sign"] is True
    with pytest.raises(ValueError):
        FlowManualCheck(**{**check.model_dump(), "frame_to": 0})


def _assessment(tmp_path: Path, observations: Path, **overrides: object) -> Path:
    unknown = {"status": "unknown", "value": None, "source": "sin ficha"}
    payload: dict[str, object] = {
        "schema_version": 2,
        "video_id": "clip",
        "video_sha256": "0" * 64,
        "stem_observations_sha256": (
            hashlib.sha256(observations.read_bytes()).hexdigest()
            if observations.is_file()
            else "0" * 64
        ),
        "assessor": "test",
        "method": "test",
        "stem_base_visibility": "partial",
        "stem_observations_ref": observations.name,
        "crossing_line": {
            "v_px": 540,
            "u_min_px": 0,
            "u_max_px": 1920,
            "crossing_direction": "up",
            "rationale": "centro",
        },
        "apparent_motion_manual": "up",
        "flow_manual_checks": [],
        "mounting": {
            "camera_rigid_to_vehicle": unknown,
            "orientation": unknown,
            "height": unknown,
            "position_relative_to_tool": unknown,
            "tool_visible_in_frame": False,
        },
        "observation_timing": "unknown",
        "replay_suitability": "B",
        "anticipation_validation": "not_possible",
        "anticipation_reason": "montaje desconocido",
        "scale": {
            "status": "unknown",
            "evidence_ref": None,
            "candidates_examined": [],
            "notes": "sin referencia",
        },
        "homography": {
            "feasibility": "possible_with_additional_data",
            "planar_ground_in_roi": "plausible",
            "notes": "n",
        },
        "conditions": [],
        "intervals": [],
        "suitability": "B",
        "suitability_reason": "experimentos limitados",
    }
    payload.update(overrides)
    path = tmp_path / "assessment.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True))
    return path


def test_class_a_needs_three_tracked_plants(tmp_path: Path) -> None:
    two = _write_obs(
        tmp_path / "two.csv",
        [
            _row("E1", "P1", 0, 700),
            _row("E2", "P1", 6, 500),
            _row("E3", "P2", 0, 800),
            _row("E4", "P2", 6, 600),
            _row("E5", "P3", 0, None, "hidden"),
            _row("E6", "P3", 6, None, "hidden"),
        ],
    )
    with pytest.raises(ConfigError, match=">= 3"):
        ManualAssessment.load(_assessment(tmp_path, two, suitability="A"), tmp_path)
    three = _write_obs(
        tmp_path / "three.csv",
        [
            _row("E1", "P1", 0, 700),
            _row("E2", "P1", 6, 500),
            _row("E3", "P2", 0, 800),
            _row("E4", "P2", 6, 600),
            _row("E5", "P3", 0, 900),
            _row("E6", "P3", 6, 700),
        ],
    )
    loaded, _ = ManualAssessment.load(_assessment(tmp_path, three, suitability="A"), tmp_path)
    assert loaded.suitability is Suitability.A
    # Los mismos puntos tipados como proxy de eje/copa no son bases en el suelo.
    proxies = _write_obs(
        tmp_path / "proxies.csv",
        [
            _row(f"E{k}", plant, frame, v, target="stem_axis_proxy")
            for k, (plant, frame, v) in enumerate(
                [
                    ("P1", 0, 700),
                    ("P1", 6, 500),
                    ("P2", 0, 800),
                    ("P2", 6, 600),
                    ("P3", 0, 900),
                    ("P3", 6, 700),
                ]
            )
        ],
    )
    with pytest.raises(ConfigError, match="ground_contact"):
        ManualAssessment.load(_assessment(tmp_path, proxies, suitability="A"), tmp_path)


def test_class_a_needs_a_crossing_line(tmp_path: Path) -> None:
    obs = _write_obs(tmp_path / "o.csv", [_row("E1", "P1", 0, 700)])
    with pytest.raises(ConfigError, match="linea virtual"):
        ManualAssessment.load(
            _assessment(tmp_path, obs, suitability="A", crossing_line=None), tmp_path
        )


@pytest.mark.parametrize(
    "override",
    [
        {
            "scale": {
                "status": "verified",
                "evidence_ref": "x",
                "candidates_examined": [],
                "notes": "",
            }
        },
        {
            "homography": {
                "feasibility": "validated",
                "planar_ground_in_roi": "plausible",
                "notes": "",
            }
        },
        {"observation_timing": "before_intervention"},
        {"anticipation_validation": "possible"},
        {"clave_desconocida": 1},
    ],
)
def test_e0_cannot_certify_scale_homography_or_anticipation(
    tmp_path: Path, override: dict[str, object]
) -> None:
    obs = _write_obs(tmp_path / "o.csv", [_row("E1", "P1", 0, 700)])
    with pytest.raises(ConfigError):
        ManualAssessment.load(_assessment(tmp_path, obs, **override), tmp_path)


def test_unknown_mounting_dominates_and_motion_conflicts_raise(tmp_path: Path) -> None:
    obs = _write_obs(tmp_path / "o.csv", [_row("E1", "P1", 0, 700)])
    assessment, _ = ManualAssessment.load(_assessment(tmp_path, obs), tmp_path)
    assert assessment.mounting_status.value == "unknown"
    assert assessment.reconcile_motion(ApparentMotion.UP) is ApparentMotion.UP
    with pytest.raises(PrecisionWeederError, match="distinta"):
        assessment.reconcile_motion(ApparentMotion.DOWN)


def test_missing_observation_table_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="no resuelve"):
        ManualAssessment.load(_assessment(tmp_path, tmp_path / "missing.csv"), tmp_path)
