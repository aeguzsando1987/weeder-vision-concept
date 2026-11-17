"""Matriz MP-01..MP-09 de la orden M0-P sobre fixtures sinteticos congelados."""

from __future__ import annotations

import itertools
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from m0p_testing import M0_DIR, PROFILES, ROOT, load, read_sidecar

sys.path.insert(0, str(M0_DIR))

import render_route_video
from m0p.fixtures import EXPECTED, fixture_spec
from m0p.mapbuilder import build_local_map
from m0p.motion import GroundMotionEstimator
from m0p.observations import Observation, ObservationSet, build_export
from m0p.pipeline import run_pipeline
from m0p.profile import FusionConfig, TrackingConfig
from m0p.projection import SegmentProjection
from m0p.route import assumed_tool_profile, chain_hash, chain_x_at
from m0p.synthetic import decode_frame_code
from m0p.tracking import Tracker, fuse_observations
from verify_run import _presentation_end

from precision_weeder.core.errors import PrecisionWeederError, SchemaValidationError
from precision_weeder.core.trajectory import TaskPath
from precision_weeder.mapping.local_map import LocalMap
from precision_weeder.validation.swept_volume import validate_task_path
from precision_weeder.video.reader import iter_frames, sha256_file

# --- MP-01: misma CLI en clips distintos; configuracion invalida ------------------------


def _cli(video, profile, export, output) -> int:
    return render_route_video.main(
        [
            "--input",
            str(video),
            "--profile",
            str(profile),
            "--export",
            str(export),
            "--output",
            str(output),
            "--root",
            str(ROOT),
        ]
    )


@pytest.mark.parametrize("name", ["cfr_640", "vfr_960"])
def test_same_cli_and_code_on_two_generated_clips(fixture_files, tmp_path, name) -> None:
    video, export = fixture_files[name]
    assert _cli(video, PROFILES[name], export, tmp_path / "out.mp4") == 0
    assert (tmp_path / "out.mp4").is_file()
    assert (tmp_path / "out.sidecar.jsonl").is_file()


@pytest.mark.parametrize("name", ["cfr_640", "vfr_960"])
def test_left_and_right_row_profiles_need_no_code_change(fixture_files, name) -> None:
    video, export = fixture_files[name]
    profile, resolved = load(name)
    observations = ObservationSet.from_payload(json.loads(export.read_text()))
    left = profile.model_copy(
        update={
            "row": profile.row.model_copy(
                update={
                    "name": "izquierdo",
                    "u_min_norm": 0.05,
                    "u_max_norm": 0.40,
                    "axis_u_norm": 0.225,
                }
            )
        }
    )
    for variant in (profile, left):
        out = Path(video).with_name(f"{name}_{variant.row.name}.mp4")
        summary = run_pipeline(video, variant, resolved, observations, out, ROOT)
        assert summary.valid_frames, variant.row.name


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("projection", "scale_m_per_px"), float("nan"), "finito"),
        (("projection", "scale_m_per_px"), 0.0, "greater than 0"),
        (("projection", "scale_m_per_px"), -0.003, "greater than 0"),
        (("row", "u_max_norm"), 0.5, "u_max_norm"),
        (("motion", "max_mad_px"), float("inf"), "finito"),
        (("clave_inventada",), 1, "Extra inputs"),
    ],
)
def test_invalid_profile_is_rejected_before_output(fixture_files, tmp_path, path, value, message):
    payload = yaml.safe_load(PROFILES["cfr_640"].read_text())
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(payload))
    video, export = fixture_files["cfr_640"]
    assert _cli(video, bad, export, tmp_path / "o" / "out.mp4") == 2
    assert not (tmp_path / "o" / "out.mp4").exists()


def test_singular_projection_is_rejected() -> None:
    with pytest.raises(SchemaValidationError):
        SegmentProjection("W", 0, 0, 0.0, 1, (0.0, 0.0), (0.0, 0.0), (640, 480))
    with pytest.raises(SchemaValidationError):
        SegmentProjection("W", 0, 0, 0.003, 0, (0.0, 0.0), (0.0, 0.0), (640, 480))


def test_export_for_other_video_or_frame_count_is_rejected(fixture_files, tmp_path) -> None:
    video, _ = fixture_files["cfr_640"]
    _, other_export = fixture_files["vfr_960"]
    assert _cli(video, PROFILES["cfr_640"], other_export, tmp_path / "a.mp4") == 2
    payload = json.loads(fixture_files["cfr_640"][1].read_text())
    body = {k: v for k, v in payload.items() if k != "content_hash"}
    short = build_export(
        body["provider"],
        body["video"]["sha256"],
        39,
        (body["video"]["width"], body["video"]["height"]),
        body["frames"][:39],
    )
    short_path = tmp_path / "short.json"
    short_path.write_text(json.dumps(short))
    assert _cli(video, PROFILES["cfr_640"], short_path, tmp_path / "b.mp4") == 2
    assert not list(tmp_path.glob("*.mp4")) and not list(tmp_path.glob(".*partial*"))


# --- MP-02: contrato de observaciones -----------------------------------------------------


def test_observation_contract_and_targets(fixture_files) -> None:
    payload = json.loads(fixture_files["cfr_640"][1].read_text())
    observations = ObservationSet.from_payload(payload)
    assert all(o.target == "canopy_proxy" for f in observations.by_frame.values() for o in f)
    tampered = dict(payload, target="ground_contact")
    with pytest.raises(SchemaValidationError, match="content_hash"):
        ObservationSet.from_payload(tampered)
    with pytest.raises(SchemaValidationError):
        Observation(0, "base_guess", "moringa", 0.9, np.zeros((3, 2)))
    with pytest.raises(SchemaValidationError):
        Observation(0, "canopy_proxy", "moringa", 1.5, np.array([[0, 0], [1, 0], [0, 1]]))


def test_alternative_provider_and_confidence_do_not_change_planning(fixture_files, tmp_path):
    """Un proveedor distinto con las mismas mascaras (y otras confianzas >= umbral)
    produce los mismos mapas y rutas: la confianza no es incertidumbre metrica."""
    video, export = fixture_files["cfr_640"]
    payload = json.loads(export.read_text())
    frames = json.loads(json.dumps(payload["frames"]))
    for frame in frames:
        for det in frame["detections"]:
            det["confidence"] = 0.99 if det["confidence"] >= 0.25 else det["confidence"]
    meta = payload["video"]
    mock = build_export(
        {"kind": "mock", "name": "proveedor alternativo"},
        meta["sha256"],
        meta["frame_count"],
        (meta["width"], meta["height"]),
        frames,
    )
    profile, resolved = load("cfr_640")
    hashes = []
    for name, data in (("a", payload), ("b", mock)):
        summary = run_pipeline(
            video,
            profile,
            resolved,
            ObservationSet.from_payload(data),
            tmp_path / f"{name}.mp4",
            ROOT,
        )
        _, records = read_sidecar(Path(summary.sidecar))
        hashes.append([(r.get("map_hash"), r.get("route_hash"), r["status"]) for r in records])
    assert hashes[0] == hashes[1]


# --- MP-03: fusion, asociacion y caducidad ------------------------------------------------


def _obs(u, v, r, conf=0.9):
    angles = np.linspace(0, 2 * math.pi, 24, endpoint=False)
    polygon = np.column_stack([u + r * np.cos(angles), v + r * np.sin(angles)])
    return Observation(0, "canopy_proxy", "moringa", conf, polygon)


def test_duplicate_masks_fuse_and_close_plants_stay_apart() -> None:
    config = FusionConfig()
    same = fuse_observations((_obs(100, 100, 20), _obs(104, 100, 18, 0.6)), config)
    assert len(same) == 1 and same[0].members == 2 and same[0].confidence == 0.9
    close = fuse_observations((_obs(100, 100, 20), _obs(142, 100, 20)), config)
    assert len(close) == 2


def test_ids_persist_under_known_translation_and_coast_without_new_gap(fixture_files) -> None:
    profile, _ = load("cfr_640")
    spec = fixture_spec("cfr_640")
    tracker = Tracker(TrackingConfig(gate_m=0.06, timeout_frames=3))
    observations = ObservationSet.from_payload(json.loads(fixture_files["cfr_640"][1].read_text()))
    anchor = SegmentProjection(
        "W", 0, 0, profile.projection.scale_m_per_px, 1, (320.0, 240.0), (0.0, 0.0), (640, 480)
    )
    truth = anchor.image_to_ground(np.array([[p.u0, p.v0] for p in spec.plants]))
    seen_ids: dict[int, set[int]] = {}
    for index in range(spec.frames):
        projection = SegmentProjection(
            "W",
            0,
            index,
            profile.projection.scale_m_per_px,
            1,
            (320.0, 240.0),
            (index * spec.shift_px[0], index * spec.shift_px[1]),
            (640, 480),
        )
        fused = fuse_observations(observations.for_frame(index, 0.25), profile.fusion)
        tracker.update(index, fused, projection)
        for track in tracker.tracks:
            nearest = int(np.argmin(np.linalg.norm(truth - track.centroid, axis=1)))
            seen_ids.setdefault(track.track_id, set()).add(nearest)
    # Cero intercambios: cada id corresponde a una sola planta real durante todo el
    # fixture (el centroide puede moverse al recortarse la mascara en el borde).
    assert all(len(plants) == 1 for plants in seen_ids.values())
    plants_seen = {
        p.plant_id
        for p in spec.plants
        if any(0 <= p.v0 + t * spec.shift_px[1] <= 480 for t in range(spec.frames))
    }
    assert tracker._next_id - 1 == len(plants_seen)


def test_dropout_keeps_protection_then_marks_unknown(fixture_files) -> None:
    profile, resolved = load("cfr_640")
    tracker = Tracker(TrackingConfig(gate_m=0.06, timeout_frames=2))
    projection = SegmentProjection("W", 0, 0, 0.003, 1, (320.0, 240.0), (0.0, 0.0), (640, 480))
    plant = _obs(496, 240, 18)
    tracker.update(0, fuse_observations((plant,), profile.fusion), projection)
    for frame in (1, 2):
        tracker.update(frame, [], projection)
        outcome = build_local_map(
            profile,
            resolved["safety"].safety,
            projection,
            tracker.tracks,
            tracker.expired,
            projection,
        )
        assert len(outcome.local_map.exclusions) == 1  # proteccion conservada
    tracker.update(3, [], projection)
    assert not tracker.tracks and tracker.expired
    outcome = build_local_map(
        profile, resolved["safety"].safety, projection, tracker.tracks, tracker.expired, projection
    )
    # Sin plantas vivas: no se declara libre; con supuesto libre, el hueco es desconocido.
    assert outcome.local_map is None
    free = profile.model_copy(
        update={"map": profile.map.model_copy(update={"assume_free_when_empty": True})}
    )
    outcome = build_local_map(
        free, resolved["safety"].safety, projection, [], tracker.expired, projection
    )
    assert outcome.local_map.unknown_regions


def test_false_positive_creates_protection_not_a_hole() -> None:
    tracker = Tracker(TrackingConfig(gate_m=0.06, timeout_frames=1))
    projection = SegmentProjection("W", 0, 0, 0.003, 1, (320.0, 240.0), (0.0, 0.0), (640, 480))
    info = tracker.update(0, fuse_observations((_obs(300, 100, 10),), FusionConfig()), projection)
    assert info["created"] == [1]
    tracker.update(1, [], projection)
    assert tracker.tracks  # en espera: sigue protegido


# --- MP-04: proyeccion y movimiento ---------------------------------------------------------


def test_projection_round_trip_and_rasterization() -> None:
    rng = np.random.default_rng(0)
    for size in ((640, 480), (1920, 1080)):
        for forward in (1, -1):
            projection = SegmentProjection(
                "W", 0, 7, 0.0013, forward, (size[0] / 2, size[1] / 2), (13.25, -411.5), size
            )
            points = rng.uniform(0, size[0], (500, 2))
            back = projection.ground_to_image(projection.image_to_ground(points))
            assert np.max(np.abs(back - points)) <= 1e-6
            assert np.max(np.abs(np.round(back) - points)) <= 1.0
    # Signo de avance: con avance hacia abajo, la herramienta progresa en +y.
    base = SegmentProjection("W", 0, 0, 0.001, 1, (320, 240), (0, 0), (640, 480))
    later = replace(base, frame_index=10, displacement_px=(0.0, -60.0))
    assert later.row_v_to_ground_y(0.05) > base.row_v_to_ground_y(0.05)


@pytest.mark.parametrize("name", ["cfr_640", "vfr_960"])
def test_accumulated_motion_error_within_two_pixels(runs, name) -> None:
    spec = fixture_spec(name)
    _, _, records, _ = runs[name]
    for record in records:
        projection = record.get("projection")
        if not projection:
            continue
        index = record["frame_index"]
        truth = np.array(spec.shift_px) * (index - projection["anchor_frame"])
        assert np.max(np.abs(np.array(projection["displacement_px"]) - truth)) <= 2.0, index


def _first_frame(fixture_files, name="cfr_640"):
    video, _ = fixture_files[name]
    for frame in iter_frames(video, image_format="bgr24"):
        return frame.image


@pytest.mark.parametrize("kind", ["rotation", "cut", "textureless", "backward"])
def test_unsupported_motion_invalidates_projection(fixture_files, kind) -> None:
    profile, _ = load("cfr_640")
    image = _first_frame(fixture_files)
    if kind == "rotation":
        matrix = cv2.getRotationMatrix2D((320, 240), 8.0, 1.0)
        second = cv2.warpAffine(image, matrix, (640, 480), borderMode=cv2.BORDER_REFLECT)
    elif kind == "cut":
        second = np.ascontiguousarray(np.flipud(np.fliplr(image)))
    elif kind == "textureless":
        image = np.full_like(image, 120)
        second = image.copy()
    else:
        second = np.roll(image, 6, axis=0)  # el suelo baja: retroceso respecto del supuesto
    estimator = GroundMotionEstimator(profile.motion, forward_sign=1)
    assert estimator.step(0, image) is None
    step = estimator.step(1, second)
    assert step is not None and not step.valid, step


# --- MP-05/06: certificado, vigencia y retirada ------------------------------------------


@pytest.mark.parametrize("name", list(PROFILES))
def test_every_valid_route_is_bound_and_independently_valid(runs, name) -> None:
    summary, header, records, _ = runs[name]
    store = Path(summary.store)
    _, resolved = load(name)
    tool = assumed_tool_profile(resolved["tool"].tool)
    assert header["tool_profile_hash"] == tool.content_hash()
    checked = 0
    for record in records:
        if record["status"] != "vigente":
            assert "route_hash" not in record
            continue
        local_map = LocalMap.from_dict(json.loads((store / record["map"]).read_text()))
        assert local_map.content_hash() == record["map_hash"]
        routes = tuple(
            TaskPath.from_dict(json.loads((store / leg["route"]).read_text()))
            for leg in record["legs"]
        )
        assert routes and chain_hash(routes) == record["route_hash"]
        for route in routes:
            assert route.map_hash == record["map_hash"]
            report = validate_task_path(
                route, local_map, tool, resolved["safety"].safety, resolved["validation"].validation
            )
            assert report.accepted
        checked += 1
    assert (checked > 0) == (EXPECTED[name] == "route")


def test_route_is_retired_on_lost_projection_and_invalid_timestamp(
    fixture_files, tmp_path, monkeypatch
) -> None:
    import m0p.pipeline as pipeline

    video, export = fixture_files["cfr_640"]
    original = pipeline.iter_frames

    def corrupted(path, image_format=None):
        for frame in original(path, image_format=image_format):
            if frame.index == 12:  # corte: cuadro sin relacion con el anterior
                frame = replace(frame, image=np.ascontiguousarray(np.flipud(frame.image)))
            if frame.index == 25:
                frame = replace(frame, timestamp_s=0.0)  # tiempo no creciente
            yield frame

    monkeypatch.setattr(pipeline, "iter_frames", corrupted)
    profile, resolved = load("cfr_640")
    observations = ObservationSet.from_payload(json.loads(export.read_text()))
    summary = run_pipeline(video, profile, resolved, observations, tmp_path / "o.mp4", ROOT)
    _, records = read_sidecar(Path(summary.sidecar))
    by_index = {r["frame_index"]: r for r in records}
    assert by_index[11]["status"] == "vigente"
    assert by_index[12]["status"] == "proyeccion_perdida" and "route_hash" not in by_index[12]
    assert by_index[13]["status"] == "segmento_iniciado"
    assert by_index[13]["segment_id"] != by_index[11]["segment_id"]
    assert by_index[25]["status"] == "timestamp_invalido" and "route_hash" not in by_index[25]


def test_no_frame_shows_a_route_certified_against_another_map(runs, fixture_files, tmp_path):
    """Con replanificacion cada 3 cuadros, los tramos reutilizados de otro cuadro solo se
    muestran revalidados contra el mapa propio (nuevo certificado), nunca con el viejo.
    Como la proyeccion avanza, la cadena reutilizada se prolonga hasta el horizonte
    actual (R1 rev. 2); la revalidacion pura con proyeccion fija se prueba aparte."""
    video, export = fixture_files["cfr_640"]
    profile, resolved = load("cfr_640")
    slow = profile.model_copy(
        update={
            "planning": profile.planning.model_copy(
                update={"replan_every_frames": 3, "max_route_age_frames": 2}
            )
        }
    )
    observations = ObservationSet.from_payload(json.loads(export.read_text()))
    summary = run_pipeline(video, slow, resolved, observations, tmp_path / "s.mp4", ROOT)
    store = Path(summary.store)
    _, records = read_sidecar(Path(summary.sidecar))
    conserved = 0
    for record in records:
        if record["status"] != "vigente":
            continue
        for leg in record["legs"]:
            report = json.loads((store / leg["report"]).read_text())
            assert report["map_hash"] == record["map_hash"]
            conserved += leg["origin"] == "conservado"
    assert conserved > 0  # la reutilizacion entre cuadros se ejercita realmente


# --- MP-07: salida de video -----------------------------------------------------------------


@pytest.mark.parametrize("name", list(PROFILES))
def test_output_preserves_frames_order_resolution_and_pts(runs, fixture_files, name) -> None:
    _, header, records, output = runs[name]
    video, _ = fixture_files[name]
    source = list(iter_frames(video))
    result = list(iter_frames(output, image_format="bgr24"))
    assert len(result) == len(source) == len(records)
    for a, b in zip(source, result, strict=True):
        assert (a.width, a.height) == (b.width, b.height)
        assert a.time_base == b.time_base and a.pts == b.pts
        assert abs(a.timestamp_s - b.timestamp_s) <= float(b.time_base)
        assert decode_frame_code(b.image) == b.index
    assert [r["frame_index"] for r in records] == list(range(len(source)))
    assert header["audio"] == "omit"
    # Fin efectivo de presentacion (PTS + duracion del ultimo cuadro), dentro de un tick.
    (src_end, src_tb), (out_end, out_tb) = _presentation_end(video), _presentation_end(output)
    assert abs(float(src_end * src_tb) - float(out_end * out_tb)) <= float(out_tb)


# --- MP-08: exito util y bloqueo ------------------------------------------------------------


@pytest.mark.parametrize("name", list(PROFILES))
def test_free_fixtures_route_around_plants_and_blocked_do_not(runs, name) -> None:
    summary, _, records, _ = runs[name]
    if EXPECTED[name] == "no_route":
        assert not summary.valid_frames
        return
    assert summary.longest_valid_run >= 10
    store = Path(summary.store)
    lateral_span = 0.0
    for record in records:
        if record["status"] == "vigente" and "planning" in record:
            for leg in record["legs"]:
                route = TaskPath.from_dict(json.loads((store / leg["route"]).read_text()))
                xs = [route.evaluate(s)[0] for s in np.linspace(0, 1, 50)]
                lateral_span = max(lateral_span, max(xs) - min(xs))
    assert lateral_span > 0.05  # la ruta se desvia respecto de una exclusion


@pytest.mark.parametrize("name", ["cfr_640", "vfr_960"])
def test_route_works_the_gaps_of_the_planting_line(runs, name) -> None:
    """Orden v1.2 §9 (MP-08): con la curva publicada y la huella, la cadena vuelve al
    eje en un gap intermedio despues de rodear una planta de la linea."""
    summary, _, records, _ = runs[name]
    store = Path(summary.store)
    _, resolved = load(name)
    half_tool = resolved["tool"].tool.length_x_m / 2
    returns = 0
    for record in records:
        if record["status"] != "vigente":
            continue
        assert record["route_kind"] in ("trabajo_en_gaps", "rodeo_lateral")
        paths = tuple(
            TaskPath.from_dict(json.loads((store / leg["route"]).read_text()))
            for leg in record["legs"]
        )
        local_map = LocalMap.from_dict(json.loads((store / record["map"]).read_text()))
        axis = record["row_axis"]["task_x_m"]
        gaps = [leg["goal_m"][1] for leg in record["legs"] if leg["target"] == "gap"]
        for y_gap in gaps:
            assert abs(chain_x_at(paths, y_gap) - axis) < 1e-9  # curva real sobre el eje
        for a, b in itertools.pairwise(gaps):
            plants = [
                e
                for e in local_map.exclusions
                if a < e.center.y < b and abs(e.center.x - axis) < e.total_radius
            ]
            for plant in plants:  # entre dos gaps, la huella rodea la planta de la linea
                x = chain_x_at(paths, plant.center.y)
                assert abs(x - plant.center.x) >= plant.total_radius + half_tool - 1e-6
                returns += 1
    assert returns > 0


def test_cli_exit_code_three_when_no_route(fixture_files, tmp_path) -> None:
    video, export = fixture_files["cfr_640_blocked"]
    assert _cli(video, PROFILES["cfr_640_blocked"], export, tmp_path / "b.mp4") == 3
    assert (tmp_path / "b.mp4").is_file()


# --- MP-09: determinismo y procedencia ----------------------------------------------------


def test_replay_is_deterministic_and_input_untouched(fixture_files, tmp_path) -> None:
    video, export = fixture_files["vfr_960"]
    before = sha256_file(video)
    profile, resolved = load("vfr_960")
    observations = ObservationSet.from_payload(json.loads(export.read_text()))
    outputs = []
    for name in ("one", "two"):
        summary = run_pipeline(
            video, profile, resolved, observations, tmp_path / f"{name}.mp4", ROOT
        )
        header, records = read_sidecar(Path(summary.sidecar))
        outputs.append(
            (
                header,
                records,
                sorted(
                    p.name
                    for p in itertools.chain.from_iterable(
                        (Path(summary.store) / kind).iterdir()
                        for kind in ("maps", "routes", "reports")
                    )
                ),
            )
        )
    assert outputs[0] == outputs[1]
    assert sha256_file(video) == before
    header = outputs[0][0]
    assert header["video"]["sha256"] == before
    assert header["observation_export_hash"] == observations.export_hash
    assert header["profile_hash"] and header["resolved_config_hash"]


def test_output_cannot_overwrite_input(fixture_files) -> None:
    video, export = fixture_files["cfr_640"]
    profile, resolved = load("cfr_640")
    observations = ObservationSet.from_payload(json.loads(export.read_text()))
    with pytest.raises(PrecisionWeederError, match="sobrescribir"):
        run_pipeline(video, profile, resolved, observations, video, ROOT)
