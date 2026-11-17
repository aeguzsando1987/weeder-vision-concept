"""E2-01, E2-02, E2-04, E2-06, E2-09: planificador de punta a punta con oraculos."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from precision_weeder.core.config import PlannerConfig, UncertaintyConfig
from precision_weeder.core.types import AbstentionReason, PlanStatus
from precision_weeder.geometry.tool_profile import ToolGeometry
from precision_weeder.mapping.synthetic_adapter import local_map_from_scene
from precision_weeder.scenarios import build_scene
from precision_weeder.scenarios.planning_cases import (
    build_planning_case,
    planning_case_names,
    random_planning_case,
    scene_config,
)
from precision_weeder.validation.swept_volume import (
    ValidationStatus,
    polynomial_min,
    validate_task_path,
)

POSITIVES = (
    "nominal_three_plants",
    "empty_corridor_certified",
    "isolated_plant",
    "straight_lane_free",
)
NEGATIVES = {
    "blocked_corridor": AbstentionReason.NO_PATH_FOUND,
    "start_in_collision": AbstentionReason.START_INADMISSIBLE,
    "unknown_band": AbstentionReason.UNKNOWN_SPACE,
    "confined_straight_passage": AbstentionReason.NO_PATH_FOUND,
}


def test_catalog_covers_exactly_the_mandatory_cases() -> None:
    assert set(planning_case_names()) == set(POSITIVES) | set(NEGATIVES)
    for name in planning_case_names():
        case = build_planning_case(name)
        expected = "ACCEPTED" if name in POSITIVES else "ABSTAIN"
        assert case.expected_status == expected  # congelado antes de ajustar el algoritmo


def _assert_certified_route(planning, result, case) -> None:
    assert result.status is PlanStatus.ACCEPTED and result.reason is None
    path = result.path
    # Revalidacion independiente: llamada nueva, sin informacion del planificador.
    report = validate_task_path(
        path,
        case.local_map,
        planning.TOOL,
        planning.SAFETY,
        planning.VALIDATION,
        expected_start=case.start_xy_m,
        expected_goal=case.goal_xy_m,
    )
    assert report.status is ValidationStatus.ACCEPTED
    assert report.clearance_lower_bound_m is None or report.clearance_lower_bound_m > 1e-4
    _assert_c2_and_progress(path, case)


def _assert_c2_and_progress(path, case) -> None:
    """E2-04: union en p, dp/ds, d2p/ds2 con |error| <= 1e-8; extremos <= 1e-9 m;
    dy/ds >= 0 por tramo con minimo analitico."""
    for left, right in itertools.pairwise(path.segments):
        for order in (0, 1, 2):
            jump = np.subtract(left.derivative_s(1.0, order), right.derivative_s(0.0, order))
            assert np.all(np.abs(jump) <= 1e-8), (order, jump)
    for segment in path.segments:
        assert np.all(np.isfinite(segment.x_coefficients))
        assert np.all(np.isfinite(segment.y_coefficients))
        dy = np.polynomial.polynomial.polyder(segment.y_coefficients) / segment.length_s
        assert polynomial_min(dy, 0.0, 1.0) >= 0.0
    assert np.abs(np.subtract(path.start_point, case.start_xy_m)).max() <= 1e-9
    assert np.abs(np.subtract(path.end_point, case.goal_xy_m)).max() <= 1e-9


@pytest.mark.parametrize("name", POSITIVES)
def test_mandatory_positive_cases_are_accepted_with_full_progress(planning, name) -> None:
    case = build_planning_case(name)
    result = planning.run_case(case)
    _assert_certified_route(planning, result, case)


def test_nominal_uses_unmodified_e1_scene_and_demo_extremes(planning) -> None:
    case = build_planning_case("nominal_three_plants")
    assert (case.start_xy_m, case.goal_xy_m, case.theta_rad) == ((0.0, 0.10), (0.0, 4.90), 0.0)
    demo = planning.PLANNER_FILE.demo
    assert (demo.scene, demo.start_xy_m, demo.goal_xy_m) == (
        "nominal_three_plants",
        (0.0, 0.10),
        (0.0, 4.90),
    )
    corridor = case.local_map.corridor
    assert (corridor.width, corridor.length) == (1.0, 5.0)
    assert [e.center.y for e in case.local_map.exclusions] == [1.0, 2.0, 3.0]
    assert all(e.total_radius == pytest.approx(0.18) for e in case.local_map.exclusions)
    result = planning.run_case(case)
    # La ruta rodea las plantas por un lado: en y=1,2,3 el origen B esta a |x| >= 0.28 m.
    for y in (1.0, 2.0, 3.0):
        s = (y - 0.1) / 4.8
        assert abs(result.path.evaluate(s)[0]) >= 0.28


def test_straight_lane_matches_analytic_free_rectangle(planning) -> None:
    case = build_planning_case("straight_lane_free")
    result = planning.run_case(case)
    _assert_certified_route(planning, result, case)
    assert {round(x, 12) for x, _ in result.chosen.waypoints} == {0.25}
    oracle = Polygon([(0.15, 0.06), (0.35, 0.06), (0.35, 4.94), (0.15, 4.94)])
    for exclusion in case.local_map.exclusions:  # circulos exactos
        circle = Point(exclusion.center.x, exclusion.center.y).buffer(exclusion.total_radius, 256)
        assert oracle.distance(circle) == pytest.approx(0.22, abs=1e-3)
    assert result.validation["clearance_lower_bound_m"] == pytest.approx(0.22, abs=1e-3)


@pytest.mark.parametrize("name", sorted(NEGATIVES))
def test_mandatory_negative_cases_are_rejected_without_route(planning, name) -> None:
    case = build_planning_case(name)
    result = planning.run_case(case)
    assert result.status is PlanStatus.ABSTAIN
    assert result.reason is NEGATIVES[name]
    assert result.path is None and result.chosen is None
    assert result.to_dict()["path"] is None
    if result.reason is AbstentionReason.NO_PATH_FOUND:
        assert "no es prueba de imposibilidad" in result.detail


def test_confined_passage_oracle_proves_impossibility() -> None:
    """La fila y=2.0 deja huecos < 0.20 m (seccion de la cuchilla) con circulos exactos."""
    case = build_planning_case("confined_straight_passage")
    corridor = case.local_map.corridor
    intervals = sorted(
        (e.center.x - e.total_radius, e.center.x + e.total_radius)
        for e in case.local_map.exclusions
    )
    assert all(e.center.y == 2.0 for e in case.local_map.exclusions)
    gaps = [
        intervals[0][0] - corridor.x_min,
        intervals[1][0] - intervals[0][1],
        corridor.x_max - intervals[1][1],
    ]
    assert gaps == pytest.approx([0.065, 0.15, 0.065], abs=1e-12)
    assert max(gaps) < 0.20
    assert case.start_xy_m[1] + 0.04 < 2.0 - 0.18 and case.goal_xy_m[1] - 0.04 > 2.0 + 0.18


def test_start_and_goal_collisions_have_distinct_reasons(planning) -> None:
    case = build_planning_case("nominal_three_plants")
    for start, goal, reason in (
        ((0.0, 2.0), (0.0, 4.9), AbstentionReason.START_INADMISSIBLE),
        ((0.0, 0.1), (0.0, 3.0), AbstentionReason.GOAL_INADMISSIBLE),
        ((0.45, 0.1), (0.0, 4.9), AbstentionReason.START_INADMISSIBLE),  # fuera del corredor
    ):
        request = planning.request(case.local_map, start, goal)
        result = planning.planner().plan(request, case.local_map, planning.TOOL)
        assert result.reason is reason and result.path is None


# --- E2-06: politicas de incertidumbre en el planificador -------------------------------


@pytest.mark.parametrize("inflate", ["exclusions", "tool"])
def test_planner_accepts_nominal_under_both_inflation_policies(planning, inflate) -> None:
    margin = 0.03 if inflate == "exclusions" else 0.0
    config = scene_config("nominal_policy", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), margin_m=margin)
    local_map = local_map_from_scene(
        build_scene(config, uncertainty=UncertaintyConfig(inflate=inflate))
    )
    request = planning.request(local_map, (0.0, 0.1), (0.0, 4.9))
    result = planning.planner().plan(request, local_map, planning.TOOL)
    assert result.accepted
    effective = result.validation["effective_geometry"]
    assert effective["uncertainty_inflate"] == inflate
    assert effective["tool_inflation_m"] == pytest.approx(0.03 if inflate == "tool" else 0.0)


# --- E2-09: determinismo y generalizacion --------------------------------------------------


def test_identical_inputs_give_identical_results(planning) -> None:
    case = build_planning_case("nominal_three_plants")
    first = planning.run_case(case).to_dict(include_timing=False)
    second = planning.run_case(build_planning_case("nominal_three_plants")).to_dict(
        include_timing=False
    )
    assert first == second


def test_translated_fixture_keeps_invariants(planning) -> None:
    case = build_planning_case("nominal_three_plants")
    dx, dy = 3.7, -12.1
    moved = case.local_map.translated(dx, dy)
    request = planning.request(moved, (dx, 0.1 + dy), (dx, 4.9 + dy))
    result = planning.planner().plan(request, moved, planning.TOOL)
    assert result.accepted
    original = planning.run_case(case)
    # Mismas invariantes; la ruta trasladada coincide con la original desplazada.
    for s in np.linspace(0.0, 1.0, 11):
        x0, y0 = original.path.evaluate(s)
        x1, y1 = result.path.evaluate(s)
        assert (x1 - dx, y1 - dy) == pytest.approx((x0, y0), abs=1e-6)
    assert result.validation["clearance_lower_bound_m"] == pytest.approx(
        original.validation["clearance_lower_bound_m"], abs=1e-6
    )


@pytest.mark.parametrize(
    ("width", "positions", "offsets"),
    [
        (0.8, (1.0, 2.5), (0.0, 0.0)),
        (1.4, (1.0, 2.0, 3.0, 4.0), (0.2, -0.2, 0.2, -0.2)),
        (1.0, (1.2, 3.4), (0.15, -0.1)),
    ],
)
@pytest.mark.parametrize(("length_x", "width_y"), [(0.20, 0.08), (0.12, 0.05)])
def test_configured_variations_need_no_code_change(
    planning, width, positions, offsets, length_x, width_y
) -> None:
    tool = ToolGeometry.rectangular("blade_variant", length_x, width_y, cutting_inset=0.005)
    config = scene_config("variation", positions, offsets, width=width)
    local_map = local_map_from_scene(build_scene(config))
    request = planning.request(local_map, (0.0, 0.1), (0.0, 4.9), profile=tool)
    result = planning.planner().plan(request, local_map, tool)
    assert result.status in (PlanStatus.ACCEPTED, PlanStatus.ABSTAIN, PlanStatus.UNCERTIFIED)
    if result.accepted:
        report = validate_task_path(
            result.path, local_map, tool, planning.SAFETY, planning.VALIDATION
        )
        assert report.accepted
    else:
        assert result.path is None


def test_wider_blade_still_finds_route_when_gap_allows(planning) -> None:
    """Hueco lateral de la escena nominal: 0.32 m; cuchilla de 0.26 m mas holguras cabe."""
    tool = ToolGeometry.rectangular("blade_260x100", 0.26, 0.10, cutting_inset=0.005)
    case = build_planning_case("nominal_three_plants")
    request = planning.request(case.local_map, case.start_xy_m, case.goal_xy_m, profile=tool)
    result = planning.planner().plan(request, case.local_map, tool)
    assert result.accepted


def test_search_budget_is_deterministic(planning) -> None:
    case = build_planning_case("nominal_three_plants")
    tight = PlannerConfig(max_graph_edges=5)
    results = [planning.run_case(case, planner_config=tight) for _ in range(2)]
    assert results[0].to_dict(include_timing=False) == results[1].to_dict(include_timing=False)
    assert results[0].search["budget_exhausted"] is True
    assert results[0].status is PlanStatus.ABSTAIN and "presupuesto" in results[0].detail


def test_rejected_candidates_are_kept_as_diagnostics(planning) -> None:
    """Con holgura de generacion casi nula, los candidatos rozan la protección; los que
    el validador rechace quedan registrados sin marcarse como aceptados."""
    case = build_planning_case("nominal_three_plants")
    result = planning.run_case(case)
    payload = result.to_dict()
    for record in payload["rejected_candidates"]:
        assert record["outcome"].startswith("validator_")
    assert payload["chosen"]["outcome"] == "accepted"
    assert math.isfinite(payload["chosen"]["cost"]["total"])


def test_passage_only_inside_overlap_band_is_found(planning) -> None:
    """Regresion (semilla 7 de E2-08, generador congelado): las plantas en
    (-0.407, 1.456) y (0.264, 1.741) se solapan en `y`; el paso entre ambas solo existe
    en su franja comun. Antes de muestrear la descomposicion vertical, ningun nivel ni
    candidato lateral caia en esa franja y el planificador devolvia NO_PATH_FOUND."""
    case = random_planning_case(7, planning.RANDOM)
    centers = [(round(e.center.x, 3), round(e.center.y, 3)) for e in case.local_map.exclusions]
    assert centers[:2] == [(-0.407, 1.456), (0.264, 1.741)]
    result = planning.run_case(case)
    assert result.accepted
    # En la franja comun (y = 1.6) el origen B pasa entre ambas plantas.
    x, _ = result.path.evaluate(
        (1.6 - case.start_xy_m[1]) / (case.goal_xy_m[1] - case.start_xy_m[1])
    )
    assert -0.407 + 0.18 + 0.10 < x < 0.264 - 0.18 - 0.10
