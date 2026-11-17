"""Patron alternante V1 (orden V1, criterio V1-01) sobre escenas sinteticas.

El patron solo propone el costado; E2 genera cada tramo y el validador lo certifica
contra el mapa real. Se comprueba con espaciamientos variados, no con un clip.
"""

from __future__ import annotations

from itertools import pairwise

import pytest
from m0p.profile import load_profile
from m0p.projection import SegmentProjection
from m0p.route import Reference, RoutePlanner, assumed_tool_profile, chain_hash
from m0p_testing import M0_DIR, PROFILES, ROOT

from precision_weeder.core.config import ToolFile, load_yaml_mapping, validate_config_payload
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
BOX = (0.0, 1.0, 0.0, 1.0)
AXIS = 0.5
TOOL_V1 = "experiments/m0_exploratory/profiles/tool_v1.yaml"


def planner_for(pattern: str, **planning) -> tuple[RoutePlanner, SegmentProjection]:
    """Corredor [0,1]^2, escala 1 mm/px, herramienta V1 de 0.10 x 0.04 m."""
    profile, resolved = load_profile(PROFILES["cfr_640"], ROOT)
    profile = profile.model_copy(
        update={
            "tool_line": profile.tool_line.model_copy(update={"v_norm": 0.05}),
            "planning": profile.planning.model_copy(
                update={"goal_margin_norm": 0.05, "pattern": pattern, **planning}
            ),
        }
    )
    tool_file = validate_config_payload(ToolFile, load_yaml_mapping(ROOT / TOOL_V1), TOOL_V1)
    planner = RoutePlanner(
        profile,
        resolved["planner"].planner,
        resolved["validation"].validation,
        resolved["safety"].safety,
        assumed_tool_profile(tool_file.tool),
    )
    return planner, SegmentProjection("W", 0, 0, 0.001, 1, (0, 0), (0, 0), (1000, 1000))


def scene(ys, radius=0.05, x=AXIS, map_id="v1", length=1.0) -> LocalMap:
    exclusions = tuple(
        CircularExclusion(f"p{k}", Point2(x, y, Frame.WORLD, D), radius, 0)
        for k, y in enumerate(ys)
    )
    return LocalMap(
        map_id,
        "W",
        D,
        Corridor(0, 1, 0, length, domain=D),
        rectangle_ring(0, 1, 0, length),
        "assumed_test",
        exclusions,
        UncertaintyPolicy(UncertaintyBound(0, 0, 0, D), 0),
        free_space_certified=not exclusions,
    )


def sides_in_order(attempt) -> list[int]:
    sides = attempt.pattern["plant_sides"]
    return [sides[key] for key in sorted(sides, key=lambda name: int(name[1:]))]


@pytest.mark.parametrize(
    "ys",
    [
        (0.25, 0.50, 0.75),  # espaciamiento regular
        (0.20, 0.42, 0.78),  # irregular
        (0.18, 0.38, 0.58, 0.80),  # cuatro plantas
    ],
)
def test_alternating_switches_side_at_every_admissible_gap(ys) -> None:
    planner, projection = planner_for("alternating")
    attempt = planner.plan(scene(ys), projection, BOX, AXIS)
    assert attempt.paths and attempt.kind == "trabajo_en_gaps"
    assert all(leg.report.accepted for leg in attempt.legs)
    assert attempt.parametrization["c2_ok"]
    sides = sides_in_order(attempt)
    assert len(sides) == len(ys)
    assert all(a == -b for a, b in pairwise(sides)), sides
    assert attempt.pattern["omitted"] == []


def test_default_pattern_is_unchanged_and_reports_no_pattern() -> None:
    planner, projection = planner_for("planner_choice")
    attempt = planner.plan(scene((0.25, 0.50, 0.75)), projection, BOX, AXIS)
    assert attempt.paths and attempt.pattern is None
    assert "pattern" not in attempt.summary()


def test_plants_too_close_are_passed_on_one_side() -> None:
    """Un gap donde la huella no cabe no admite cruce: el grupo se rodea por un costado."""
    planner, projection = planner_for("alternating")
    attempt = planner.plan(scene((0.30, 0.42, 0.75)), projection, BOX, AXIS)
    assert attempt.paths
    first, second, third = sides_in_order(attempt)
    assert first == second  # sin cruce entre plantas pegadas (side cut)
    assert third == -second  # y se alterna en el gap admisible siguiente


def test_blocked_side_degrades_and_declares_the_omission() -> None:
    """Si el costado deseado esta cerrado, se conserva la eleccion de E2 y se declara."""
    planner, projection = planner_for("alternating")
    free = planner.plan(scene((0.25, 0.60)), projection, BOX, AXIS)
    first, second = sides_in_order(free)
    assert second == -first
    # Pared pegada a la segunda planta por el costado que el patron pediria.
    wall_x = AXIS + second * 0.30
    base = scene((0.25, 0.60))
    wall = CircularExclusion("w9", Point2(wall_x, 0.60, Frame.WORLD, D), 0.24, 0)
    blocked = LocalMap(
        "v1_wall",
        "W",
        D,
        base.corridor,
        rectangle_ring(0, 1, 0, 1),
        "assumed_test",
        (*base.exclusions, wall),
        base.uncertainty,
    )
    attempt = planner.plan(blocked, projection, BOX, AXIS)
    assert attempt.paths, "debe degradar, no quedarse sin ruta"
    sides = attempt.pattern["plant_sides"]
    assert sides["p1"] == sides["p0"] == first
    omitted = attempt.pattern["omitted"]
    assert len(omitted) == 1 and omitted[0]["plants"] == ["p1"]
    assert omitted[0]["wanted_side"] == second and omitted[0]["cause"]


def test_committed_sides_are_kept_between_frames() -> None:
    """Al avanzar, la parte ya decidida no cambia de costado (prefijo conservado)."""
    planner, _ = planner_for("alternating")
    local_map = scene((0.25, 0.50, 0.75), length=1.2)  # suelo conocido por delante

    def projection(frame: int) -> SegmentProjection:
        return SegmentProjection("W", 0, frame, 0.001, 1, (0, 0), (0, -8.0 * frame), (1000, 1000))

    attempt = planner.plan(local_map, projection(0), BOX, AXIS)
    reference = Reference(attempt.legs, attempt.axis_x, 0, chain_hash(attempt.paths))
    before = attempt.pattern["plant_sides"]
    for frame in (1, 2, 3):
        attempt = planner.plan(local_map, projection(frame), BOX, AXIS, reference)
        assert attempt.paths and attempt.decision == "prefijo_conservado"
        for plant, side in attempt.pattern["plant_sides"].items():
            assert before[plant] == side
        reference = Reference(attempt.legs, attempt.axis_x, frame, chain_hash(attempt.paths))


def projection_at(tool_y: float) -> SegmentProjection:
    """Proyeccion cuya linea de herramienta (v = 0.05) queda en la ordenada `tool_y`."""
    shift_px = -(tool_y - 0.05) / 0.001
    return SegmentProjection("W", 0, 1, 0.001, 1, (0, 0), (0, shift_px), (1000, 1000))


@pytest.mark.parametrize("drift_per_frame", [0.002, -0.001])
def test_knots_ahead_follow_plants_that_drift_in_the_map(drift_per_frame) -> None:
    """Las plantas seguidas no quedan fijas en `W`: los nudos por delante se vuelven a
    anclar a su planta en cada cuadro, sin cambiar de costado ni romper la consigna."""
    from m0p.route import chain_x_at

    planner, _ = planner_for("alternating")
    base = (0.30, 0.55, 0.80, 1.05)
    attempt = planner.plan(scene(base, length=2.0), projection_at(0.05), BOX, AXIS)
    reference = Reference(attempt.legs, attempt.axis_x, 0, chain_hash(attempt.paths))
    committed = dict(attempt.pattern["plant_sides"])
    for frame in range(1, 25):
        tool_y = 0.05 + 0.025 * frame
        ys = tuple(y - drift_per_frame * frame for y in base)
        previous_x = chain_x_at(attempt.paths, tool_y)
        attempt = planner.plan(scene(ys, length=2.0), projection_at(tool_y), BOX, AXIS, reference)
        assert attempt.paths and attempt.decision == "prefijo_conservado", attempt.cause
        assert chain_x_at(attempt.paths, tool_y) == pytest.approx(previous_x, abs=1e-9)
        assert attempt.parametrization["c2_ok"]
        for leg in attempt.legs[1:]:
            if leg.target != "cierre":
                assert min(abs(leg.goal[1] - y) for y in ys) < 1e-9
        for plant, side in attempt.pattern["plant_sides"].items():
            assert committed.setdefault(plant, side) == side
        reference = Reference(attempt.legs, attempt.axis_x, frame, chain_hash(attempt.paths))


def test_plant_entering_after_a_long_straight_run_is_crossed() -> None:
    """Un cierre recto en curso es provisional: al entrar una planta nueva tras un gap
    largo, la ruta cruza hacia el costado opuesto desde la posicion de la herramienta."""
    from m0p.route import chain_x_at

    planner, _ = planner_for("alternating")
    attempt = planner.plan(scene((0.25,), length=2.0), projection_at(0.05), BOX, AXIS)
    assert attempt.paths and attempt.legs[-1].target == "cierre"
    reference = Reference(attempt.legs, attempt.axis_x, 0, chain_hash(attempt.paths))
    x_tool = chain_x_at(attempt.paths, 0.40)
    assert abs(x_tool - AXIS) > 0.10  # la herramienta ya paso la planta, por un costado
    attempt = planner.plan(
        scene((0.25, 0.95), length=2.0), projection_at(0.40), BOX, AXIS, reference
    )
    assert attempt.paths and attempt.decision == "prefijo_conservado", attempt.cause
    assert chain_x_at(attempt.paths, 0.40) == pytest.approx(x_tool, abs=1e-9)
    assert (chain_x_at(attempt.paths, 0.95) - AXIS) * (x_tool - AXIS) < 0  # costado opuesto
    assert attempt.kind == "trabajo_en_gaps"


def test_route_ends_before_a_plant_that_is_still_beyond_the_horizon() -> None:
    """Una planta con el centro mas alla del horizonte aun no se rodea, pero su zona
    protegida asoma: el cierre termina antes de ella en lugar de quedarse sin ruta."""
    planner, projection = planner_for("alternating")
    alone = planner.plan(scene((0.30,), length=1.2), projection, BOX, AXIS)
    side = alone.pattern["plant_sides"]["p0"]
    assert alone.legs[-1].goal[1] == pytest.approx(0.95)
    assert alone.pattern["horizon_limited_by"] is None
    base = scene((0.30,), length=1.2)
    entering = CircularExclusion("p1", Point2(AXIS + side * 0.05, 0.99, Frame.WORLD, D), 0.05, 0)
    local_map = LocalMap(
        "v1_horizon",
        "W",
        D,
        base.corridor,
        rectangle_ring(0, 1, 0, 1.2),
        "assumed_test",
        (*base.exclusions, entering),
        base.uncertainty,
    )
    attempt = planner.plan(local_map, projection, BOX, AXIS)
    assert attempt.paths and all(leg.report.accepted for leg in attempt.legs)
    assert attempt.legs[-1].target == "cierre"
    assert 0.60 < attempt.legs[-1].goal[1] < 0.99 - 0.05
    assert attempt.pattern["horizon_limited_by"] == "p1"
    assert "p1" not in attempt.pattern["plant_sides"]


def test_a_new_crossing_needs_spare_clearance_to_be_committed() -> None:
    """Un cruce certificado con menos holgura que la del perfil no se compromete (se
    declara), salvo que sea la unica forma de rodear la planta."""
    ys = (0.25, 0.60)
    planner, projection = planner_for("alternating")
    free = planner.plan(scene(ys), projection, BOX, AXIS)
    first, second = sides_in_order(free)
    assert second == -first
    strict, _ = planner_for("alternating", pattern_commit_clearance_m=0.05)
    attempt = strict.plan(scene(ys), projection, BOX, AXIS)
    assert attempt.paths and sides_in_order(attempt) == [first, first]
    assert attempt.pattern["omitted"][0]["cause"].startswith("cruce con holgura insuficiente")
    # Con el mismo costado cerrado por una pared, el cruce justo es la unica opcion.
    base = scene(ys)
    wall = CircularExclusion("w9", Point2(AXIS + first * 0.30, 0.60, Frame.WORLD, D), 0.24, 0)
    blocked = LocalMap(
        "v1_tight",
        "W",
        D,
        base.corridor,
        rectangle_ring(0, 1, 0, 1),
        "assumed_test",
        (*base.exclusions, wall),
        base.uncertainty,
    )
    strict, _ = planner_for("alternating", pattern_commit_clearance_m=0.05)
    attempt = strict.plan(blocked, projection, BOX, AXIS)
    assert attempt.paths
    sides = attempt.pattern["plant_sides"]
    assert sides["p1"] == -sides["p0"] and attempt.pattern["omitted"] == []


def test_v1_profiles_load() -> None:
    for name in ("real_left_row_v1.yaml", "real_right_row_v1.yaml"):
        profile, resolved = load_profile(M0_DIR / "profiles" / name, ROOT)
        assert profile.planning.pattern == "alternating"
        assert profile.map.protection == "assumed_stem"
        assert resolved["tool"].tool.length_x_m == pytest.approx(0.10)


def test_alternating_route_is_a_smooth_weave_close_to_the_plants() -> None:
    """El trazo del patron usa transiciones largas y se aparta lo justo del eje."""
    import numpy as np
    from m0p.route import chain_x_at

    def shape(pattern: str) -> tuple[float, float]:
        planner, projection = planner_for(pattern)
        attempt = planner.plan(scene((0.25, 0.50, 0.75)), projection, BOX, AXIS)
        ys = np.linspace(0.05, 0.95, 2001)
        xs = np.array([chain_x_at(attempt.paths, y) for y in ys])
        return float(np.abs(np.gradient(xs, ys)).max()), float(np.abs(xs - AXIS).max())

    slope, amplitude = shape("alternating")
    planner_slope, planner_amplitude = shape("planner_choice")
    assert slope < 0.5 * planner_slope and amplitude < planner_amplitude
    # Radio 0.05 + media herramienta 0.05 + holgura de propuesta (~0.03): menos de 0.14 m.
    assert 0.10 < amplitude < 0.14
