"""Dictamen M0-P rev. 1: sondas del director (R1-R4, API equivalente) y continuidad R5.

Las sondas originales siguen en la evidencia de la revision (no versionada);
aqui se integran con la firma actual de `_route_for_frame` (referencia explicita) y con
controles positivos. R5 sigue la orden v1.2 §10: estabilidad del costado ante pequenas
perturbaciones y abandono forzado enlazado desde la referencia o abstencion.
"""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from m0p.observations import ObservationSet, build_export
from m0p.pipeline import _route_for_frame, _Store
from m0p.profile import load_profile
from m0p.projection import SegmentProjection
from m0p.render import VideoWriter
from m0p.route import (
    Reference,
    RoutePlanner,
    _sides,
    assumed_tool_profile,
    chain_hash,
    chain_parametrization,
    chain_x_at,
    path_x_at,
    trim_path,
)
from m0p_testing import PROFILES, ROOT, load

from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.core.types import (
    CircularExclusion,
    Corridor,
    Frame,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
)
from precision_weeder.core.units import SpatialDomain
from precision_weeder.mapping.local_map import LocalMap, rectangle_ring

D = SpatialDomain.GROUND_ASSUMED
BOX = (0, 1, 0, 1)


def setup(exclusions=(), cadence: int | None = None):
    """Escena de las sondas del director: corredor [0,1]^2, escala 1 mm/px."""
    profile, resolved = load_profile(PROFILES["cfr_640"], ROOT)
    planning = {"goal_margin_norm": 0.1}
    if cadence:
        planning |= {"replan_every_frames": cadence, "max_route_age_frames": cadence - 1}
    profile = profile.model_copy(
        update={
            "tool_line": profile.tool_line.model_copy(update={"v_norm": 0.1}),
            "planning": profile.planning.model_copy(update=planning),
        }
    )
    tool = assumed_tool_profile(resolved["tool"].tool)
    planner = RoutePlanner(
        profile,
        resolved["planner"].planner,
        resolved["validation"].validation,
        resolved["safety"].safety,
        tool,
    )
    return profile, planner, scene(exclusions), projection(0)


def scene(exclusions, map_id="audit") -> LocalMap:
    return LocalMap(
        map_id,
        "W",
        D,
        Corridor(0, 1, 0, 1, domain=D),
        rectangle_ring(0, 1, 0, 1),
        "assumed_test",
        tuple(exclusions),
        UncertaintyPolicy(UncertaintyBound(0, 0, 0, D), 0),
        free_space_certified=not exclusions,
    )


def projection(frame: int, advance_px: float = 0.0) -> SegmentProjection:
    return SegmentProjection(
        "W", 0, frame, 0.001, 1, (0, 0), (0, -advance_px * frame), (1000, 1000)
    )


def exclusion(name, x, y, r):
    return CircularExclusion(name, Point2(x, y, Frame.WORLD, D), r, 0)


def two_plants(map_id="audit") -> LocalMap:
    """Dos plantas sobre el eje con un gap intermedio admisible para la huella."""
    return scene([exclusion("a", 0.5, 0.4, 0.08), exclusion("b", 0.5, 0.7, 0.08)], map_id)


def reference_of(attempt, frame: int) -> Reference:
    return Reference(attempt.legs, attempt.axis_x, frame, chain_hash(attempt.paths))


# --- R1: cadena completa hasta el horizonte -------------------------------------------------


def test_partial_chain_cannot_be_returned_as_complete() -> None:
    wall = [exclusion(str(i), x, 0.8, 0.25) for i, x in enumerate((0.1, 0.5, 0.9))]
    plant = exclusion("first", 0.5, 0.25, 0.08)
    _, planner, _, proj = setup()
    partial = planner.plan(scene([*wall, plant]), proj, BOX, 0.5)
    assert partial.paths is None and not partial.complete
    complete = planner.plan(scene([plant]), proj, BOX, 0.5)  # control positivo
    assert complete.paths and complete.kind == "rodeo_lateral"  # sin gap intermedio
    in_gap = planner.plan(two_plants(), proj, BOX, 0.5)
    assert in_gap.paths and in_gap.kind == "trabajo_en_gaps"


def test_partial_chain_is_diagnostic_in_the_sidecar(tmp_path) -> None:
    wall = [exclusion(str(i), x, 0.8, 0.25) for i, x in enumerate((0.1, 0.5, 0.9))]
    profile, planner, _, proj = setup()
    record: dict = {}
    paths, status, _, reference = _route_for_frame(
        planner,
        profile,
        scene([*wall, exclusion("first", 0.5, 0.25, 0.08)]),
        proj,
        BOX,
        (0.5, "test"),
        None,
        1,
        record,
        _Store(tmp_path),
    )
    assert paths is None and reference is None
    assert status in ("cadena_incompleta", "sin_ruta")
    assert "route_hash" not in record and "legs" not in record


def test_chain_has_global_c2_parametrization() -> None:
    _, planner, _, proj = setup()
    attempt = planner.plan(two_plants(), proj, BOX, 0.5)
    assert any(leg.target == "gap" for leg in attempt.legs)
    param = attempt.parametrization
    assert param["c2_ok"] and param["breakpoints"][0] == 0.0 and param["breakpoints"][-1] == 1.0
    assert len(param["joints"]) == len(attempt.legs) - 1 >= 1
    # Control negativo: unir un camino con un recorte de si mismo rompe C0.
    path = attempt.paths[0]
    y_mid = 0.5 * (path.start_point[1] + path.end_point[1])
    assert not chain_parametrization((path, trim_path(path, y_mid)))["c2_ok"]


def test_trim_keeps_the_exact_curve_and_its_certificate() -> None:
    _, planner, _, proj = setup()
    attempt = planner.plan(two_plants(), proj, BOX, 0.5)

    def excursion(path):
        return max(abs(path.evaluate(t)[0] - path.start_point[0]) for t in np.linspace(0, 1, 50))

    path = max(attempt.paths, key=excursion)  # el tramo que rodea una planta
    assert excursion(path) > 0.05
    y0, y1 = path.start_point[1], path.end_point[1]
    cut = trim_path(path, y0 + 0.37 * (y1 - y0))
    for y in np.linspace(cut.start_point[1], y1, 25):
        assert abs(path_x_at(cut, y) - path_x_at(path, y)) < 1e-9
    _, report = planner._rebind(cut, two_plants())
    assert report.accepted


# --- R2: identidad de la revalidacion y cambio de tarea -------------------------------------


def test_revalidation_sidecar_references_the_certified_path(tmp_path) -> None:
    profile, planner, local_map, proj = setup(cadence=3)
    first = planner.plan(local_map, proj, BOX, 0.5)
    assert first.paths
    updated = replace(local_map, map_id="audit:f2")
    record: dict = {}
    paths, status, _, _ = _route_for_frame(
        planner,
        profile,
        updated,
        replace(proj, frame_index=2),
        BOX,
        (0.5, "test"),
        reference_of(first, 1),
        2,
        record,
        _Store(tmp_path),
    )
    assert status == "vigente" and record["continuity"]["decision"] == "revalidada"
    for path, leg in zip(paths, record["legs"], strict=True):
        report = json.loads((tmp_path / leg["report"]).read_text())
        assert path.content_hash() == leg["route_hash"] == report["path_hash"]
        assert path.map_hash == report["map_hash"] == updated.content_hash()
    assert record["route_hash"] == chain_hash(paths)


def test_changed_axis_replans_instead_of_reusing_old_task(tmp_path) -> None:
    profile, planner, local_map, proj = setup(cadence=3)
    first = planner.plan(local_map, proj, BOX, 0.5)
    record: dict = {}
    _, status, _, _ = _route_for_frame(
        planner,
        profile,
        replace(local_map, map_id="f2"),
        replace(proj, frame_index=2),
        BOX,
        (0.7, "test"),
        reference_of(first, 1),
        2,
        record,
        _Store(tmp_path),
    )
    decision = record["continuity"]["decision"]
    assert decision in ("reseleccion_enlazada", "discontinuidad"), decision
    assert record["row_axis"]["task_x_m"] == 0.7
    assert status != "vigente" or record["route_planned_frame"] == 2


def _reuse(profile, planner, local_map, current, reference, tmp_path):
    record: dict = {}
    paths, status, _, _ = _route_for_frame(
        planner,
        profile,
        local_map,
        current,
        BOX,
        (0.5, "test"),
        reference,
        2,
        record,
        _Store(tmp_path),
    )
    return paths, status, record


def test_reused_chain_reaches_the_current_horizon(tmp_path) -> None:
    """Sonda del director (rev. 2): entre replanificaciones, si la proyeccion avanzo, la
    cadena reutilizada debe llegar a la meta actual; con proyeccion fija se revalida."""
    profile, planner, local_map, proj = setup(cadence=3)
    first = planner.plan(local_map, proj, BOX, 0.5)
    assert first.paths
    reference = reference_of(first, 1)
    current = projection(2, advance_px=5)
    goal = current.row_v_to_ground_y(0.9)
    assert first.paths[-1].end_point[1] < goal - 1e-6  # la cadena anterior no llega
    paths, status, record = _reuse(
        profile, planner, replace(local_map, map_id="f2"), current, reference, tmp_path
    )
    assert status == "vigente", record["continuity"]
    assert abs(paths[-1].end_point[1] - goal) < 1e-9
    assert record["continuity"]["decision"] == "prefijo_conservado"
    assert record["route_planned_frame"] == 2
    # Control: proyeccion fija, misma meta; se revalida sin replanificar.
    _, status, record = _reuse(
        profile, planner, replace(local_map, map_id="f2"), proj, reference, tmp_path
    )
    assert status == "vigente" and record["continuity"]["decision"] == "revalidada"
    # Control: obstaculo nuevo en la zona de extension; no se publica la cadena corta.
    wall = [exclusion(str(i), x, 1.0, 0.12) for i, x in enumerate((0.1, 0.5, 0.9))]
    paths, status, record = _reuse(
        profile, planner, scene(wall, "f2"), current, reference, tmp_path
    )
    assert paths is None and status != "vigente", record["continuity"]


# --- R3: indices de la exportacion ----------------------------------------------------------


@pytest.mark.parametrize("indices", [(0, 0, 2), (0, 1, 3), (-1, 1, 2), (0, 2), (0.5, 1, 2)])
def test_invalid_observation_frame_indexing_is_rejected(indices) -> None:
    payload = build_export(
        {"kind": "synthetic"},
        "a" * 64,
        3,
        (100, 100),
        [{"frame_index": i, "detections": []} for i in indices],
    )
    with pytest.raises(PrecisionWeederError, match=r"frame_index|faltan"):
        ObservationSet.from_payload(payload)


def test_complete_export_without_detections_is_valid() -> None:
    payload = build_export(
        {"kind": "synthetic"},
        "a" * 64,
        3,
        (100, 100),
        [{"frame_index": i, "detections": []} for i in range(3)],
    )
    observations = ObservationSet.from_payload(payload)
    assert all(observations.for_frame(i, 0.0) == () for i in range(3))


# --- R4: duracion efectiva de presentacion --------------------------------------------------


@pytest.mark.parametrize(
    ("pts", "last", "expected"),
    [((0, 3000, 6000), None, 0.1), ((0, 3000, 6600, 9000), 4500, 0.15)],
)
def test_video_duration_includes_last_frame(tmp_path, pts, last, expected) -> None:
    import av

    file = tmp_path / "duration.mp4"
    writer = VideoWriter(file, 64, 64, Fraction(1, 90000), 20)
    for value in pts:
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8), value)
    writer.close(last)
    with av.open(str(file)) as container:
        stream = container.streams.video[0]
        frames = [f.pts for f in container.decode(stream)]
        duration = float(stream.duration * stream.time_base)
        tick = float(stream.time_base)
    assert frames == list(pts)
    assert abs(duration - expected) <= tick, duration


def test_audio_copy_is_rejected_until_implemented() -> None:
    profile, _ = load("cfr_640")
    with pytest.raises(ValueError):
        profile.render.model_validate(
            {**profile.render.model_dump(), "audio": "copy_when_supported"}
        )


# --- R5: continuidad entre propuestas (orden v1.2 §10) ---------------------------------------


def test_side_is_kept_under_small_perturbations() -> None:
    """Dos rodeos validos; perturbaciones pequenas de la planta y del eje no cambian el
    costado elegido ni la posicion en la linea de herramienta (cero alternancias)."""
    _, planner, _, _ = setup()
    rng = np.random.default_rng(7)
    reference, sides, decisions = None, [], []
    for frame in range(8):
        dx, axis_noise = rng.uniform(-0.002, 0.002), rng.uniform(-0.01, 0.01)
        local_map = scene([exclusion("p", 0.5 + dx, 0.55, 0.08)], map_id=f"f{frame}")
        proj = projection(frame, advance_px=5)
        attempt = planner.plan(local_map, proj, BOX, 0.5 + axis_noise, reference)
        assert attempt.paths, attempt.summary()
        y_tool = proj.row_v_to_ground_y(0.1)
        if reference is not None:
            before = chain_x_at(tuple(leg.path for leg in reference.legs), y_tool)
            assert abs(chain_x_at(attempt.paths, y_tool) - before) < 1e-9
        sides.append(_sides(attempt.paths, local_map, (0, 1))["p"])
        decisions.append(attempt.decision)
        reference = reference_of(attempt, frame)
    assert len(set(sides)) == 1, sides
    assert decisions[0] == "inicio" and set(decisions[1:]) == {"prefijo_conservado"}


def test_forced_change_is_linked_from_the_reference_or_abstains() -> None:
    """Un obstaculo nuevo sobre la propuesta obliga a abandonarla: la nueva parte de la
    posicion de la referencia en la linea de herramienta (enlace) o no hay ruta."""
    _, planner, _, proj = setup()
    first = planner.plan(scene([exclusion("p", 0.5, 0.55, 0.08)]), proj, BOX, 0.5)
    paths = first.paths
    y_tool = proj.row_v_to_ground_y(0.1)
    x_on_route = chain_x_at(paths, 0.3)
    blocked = scene(
        [exclusion("p", 0.5, 0.55, 0.08), exclusion("new", x_on_route, 0.3, 0.05)], "f1"
    )
    attempt = planner.plan(blocked, replace(proj, frame_index=1), BOX, 0.5, reference_of(first, 0))
    assert attempt.decision in ("reseleccion_enlazada", "discontinuidad")
    if attempt.paths:
        assert abs(attempt.paths[0].start_point[0] - chain_x_at(paths, y_tool)) < 1e-9
        assert all(leg.report.accepted for leg in attempt.legs)
    else:
        assert attempt.decision == "discontinuidad"


def test_reference_off_the_tool_line_withdraws_the_fresh_route(tmp_path) -> None:
    """Sonda del director (rev. 2): si la referencia no cubre la linea de herramienta no
    hay enlace certificado; la solucion fresca es diagnostico, no ruta vigente ni
    referencia. Control: con la referencia cubierta se conserva su prefijo."""
    profile, planner, local_map, proj = setup()
    first = planner.plan(local_map, proj, BOX, 0.5)
    assert first.paths
    wide = replace(
        local_map,
        map_id="f1",
        corridor=Corridor(0, 1, 0, 2, domain=local_map.domain),
        known_region=rectangle_ring(0, 1, 0, 2),
    )

    def frame_one(advance_px):
        record: dict = {}
        result = _route_for_frame(
            planner,
            profile,
            wide,
            projection(1, advance_px=advance_px),
            (0, 1, 0, 2),
            (0.5, "test"),
            reference_of(first, 0),
            1,
            record,
            _Store(tmp_path),
        )
        return result, record

    (paths, status, _, new_reference), record = frame_one(850)
    assert paths is None and new_reference is None and status == "discontinuidad"
    assert record["continuity"]["decision"] == "discontinuidad"
    assert record["diagnostic_legs"] and "route_hash" not in record
    (paths, status, _, new_reference), record = frame_one(5)  # control positivo
    assert status == "vigente" and new_reference is not None
    assert record["continuity"]["decision"] == "prefijo_conservado"
    assert not record["continuity"]["discontinuity"]
