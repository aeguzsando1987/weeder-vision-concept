"""E2-03, E2-05, E2-06, E2-07: validador independiente con oraculos analiticos.

Los caminos se construyen a mano (no los propone el planificador). Cuchilla E1:
colision 0.20 m (x) x 0.08 m (y); corte 0.19 x 0.07 m.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from precision_weeder.core.config import UncertaintyConfig, ValidationConfig
from precision_weeder.mapping.synthetic_adapter import local_map_from_scene
from precision_weeder.scenarios import build_fixture, build_scene
from precision_weeder.scenarios.planning_cases import scene_config
from precision_weeder.validation.swept_volume import (
    ValidationStatus,
    _chord_hull,
    polynomial_abs_max,
    polynomial_min,
    validate_task_path,
)

EPS = 1.0e-4


def _validate(planning, path, local_map, config=None, **kwargs):
    return validate_task_path(
        path, local_map, planning.TOOL, planning.SAFETY, config or planning.VALIDATION, **kwargs
    )


def _kinds(report) -> set[str]:
    return {v["kind"] for v in report.violations}


# --- cotas analiticas -----------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_polynomial_bounds_are_exact_extremes(seed) -> None:
    rng = np.random.default_rng(seed)
    coefficients = rng.normal(size=4)
    a, b = sorted(rng.uniform(0, 1, 2))
    grid = np.linspace(a, b, 20001)
    values = np.polynomial.polynomial.polyval(grid, coefficients)
    assert polynomial_abs_max(coefficients, a, b) >= np.abs(values).max() - 1e-12
    assert polynomial_abs_max(coefficients, a, b) <= np.abs(values).max() + 1e-6
    assert polynomial_min(coefficients, a, b) <= values.min() + 1e-12


# --- E2-03: casos adversos del validador -------------------------------------------


def test_center_free_but_edge_collides(planning) -> None:
    """Oraculo: centro a 0.25 m del centro de la planta (> 0.18), pero el borde de la
    cuchilla (semiancho 0.10) queda a 0.25 - 0.10 - 0.18 = -0.03 m: invade."""
    local_map = planning.custom_map([planning.exclusion("P1", 0.25, 2.5, 0.18)])
    path = planning.linear_path([(0.0, 0.1), (0.0, 4.9)], local_map)
    assert math.hypot(0.25, 0.0) > 0.18  # el centro de la ruta esta libre
    report = _validate(planning, path, local_map)
    assert report.status is ValidationStatus.REJECTED
    assert "COLLISION" in _kinds(report)


def _tangent_x(radius: float, segments: int = 64) -> float:
    """x del vertice derecho del poligono exterior de la exclusion centrada en 0."""
    return radius / math.cos(math.pi / segments)


@pytest.mark.parametrize(
    ("gap", "expected"),
    [
        (0.0, ValidationStatus.REJECTED),  # tangencia exacta
        (EPS, ValidationStatus.REJECTED),  # despeje igual a epsilon: no es > epsilon
        (0.5 * EPS, ValidationStatus.REJECTED),
        (3.0 * EPS, ValidationStatus.ACCEPTED),
    ],
)
def test_tangency_and_clearance_epsilon_are_strict(planning, gap, expected) -> None:
    """Oraculo: el vertice derecho del 64-gono exterior esta en x = R/cos(pi/64); la
    cara izquierda de la cuchilla esta en x_path - 0.10."""
    local_map = planning.custom_map([planning.exclusion("P1", 0.0, 2.5, 0.18)])
    x_path = _tangent_x(0.18) + 0.10 + gap
    path = planning.linear_path([(x_path, 0.1), (x_path, 4.9)], local_map)
    report = _validate(planning, path, local_map)
    assert report.status is expected
    if expected is ValidationStatus.ACCEPTED:
        assert report.clearance_lower_bound_m == pytest.approx(gap, abs=1e-9)
        assert report.clearance_lower_bound_m > EPS


def test_invading_corner_is_rejected(planning) -> None:
    """Oraculo: la esquina superior derecha de la cuchilla en la meta, (0.10, 4.54),
    queda a 0.6·√2·r = 0.85 r del centro de la exclusion: dentro del circulo. Los
    puntos medios de los lados quedan fuera."""
    r = 0.05
    corner = np.array([0.10, 4.5 + 0.04])
    center = corner + 0.6 * r
    local_map = planning.custom_map([planning.exclusion("P1", *center, r)])
    assert np.linalg.norm(corner - center) < r
    assert np.linalg.norm(np.array([0.0, 4.54]) - center) > r  # punto medio del lado superior
    assert np.linalg.norm(np.array([0.10, 4.5]) - center) > r  # punto medio del lado derecho
    path = planning.linear_path([(0.0, 0.1), (0.0, 4.5)], local_map)
    report = _validate(planning, path, local_map)
    assert report.status is ValidationStatus.REJECTED
    assert "COLLISION" in _kinds(report)


def test_collision_between_two_free_samples_is_rejected(planning) -> None:
    """Oraculo: muestras en y=0.1 y y=4.9 libres; la planta en y=2.5 corta la recta."""
    local_map = planning.custom_map([planning.exclusion("P1", 0.0, 2.5, 0.18)])
    path = planning.linear_path([(0.0, 0.1), (0.0, 4.9)], local_map)
    tool = np.asarray(planning.TOOL.collision_polygon)
    exclusion = local_map.exclusion_polygons()[0]
    for y in (0.1, 4.9):
        assert Polygon(tool + np.array([0.0, y])).distance(exclusion) > 1.0
    report = _validate(planning, path, local_map)
    assert report.status is ValidationStatus.REJECTED
    (violation,) = [v for v in report.violations if v["kind"] == "COLLISION"]
    assert violation["evidence"] == "exact_linear_sweep" and violation["plant_id"] == "P1"


def test_leaving_corridor_and_unknown_space_are_rejected(planning) -> None:
    band = [[-0.5, 2.0], [0.5, 2.0], [0.5, 2.3], [-0.5, 2.3]]
    local_map = planning.custom_map(unknown=[band])
    path = planning.linear_path([(0.0, 0.1), (0.0, 4.9)], local_map)
    assert "UNKNOWN_SPACE" in _kinds(_validate(planning, path, local_map))
    empty = planning.custom_map()
    outside = planning.linear_path([(0.45, 0.1), (0.45, 4.9)], empty)  # borde en x=0.55
    assert "OUT_OF_DOMAIN" in _kinds(_validate(planning, outside, empty))
    # El limite es cerrado: la cara de la cuchilla sobre la pared del corredor se admite.
    flush = planning.linear_path([(0.40, 0.1), (0.40, 4.9)], empty)
    assert _validate(planning, flush, empty).status is ValidationStatus.ACCEPTED


def test_identity_mismatch_is_invalid_input(planning) -> None:
    a = planning.custom_map()
    b = planning.custom_map([planning.exclusion("P1", 0.0, 2.5, 0.18)])
    path = planning.linear_path([(0.3, 0.1), (0.3, 4.9)], a)
    with pytest.raises(Exception, match="otro mapa"):
        _validate(planning, path, b)


def test_discontinuity_and_regression_are_rejected(planning) -> None:
    from precision_weeder.core.trajectory import PathSegment

    local_map = planning.custom_map()
    # Cambio brusco de pendiente: C0 pero no C1.
    kink = planning.linear_path([(0.0, 0.1), (0.2, 2.0), (0.2, 4.9)], local_map)
    report = _validate(planning, kink, local_map)
    assert "DISCONTINUITY" in _kinds(report)
    # y retrocede dentro de un tramo: dy/du = 4.8 - 2·6u < 0 para u > 0.4.
    regress = planning.task_path(
        [PathSegment(0.0, 1.0, [0, 0, 0, 0, 0, 0], [0.1, 4.8 + 6.0, -6.0, 0, 0, 0])], local_map
    )
    report = _validate(planning, regress, local_map)
    assert "NON_MONOTONIC_PROGRESS" in _kinds(report)
    assert report.progress["min_dy_ds"] < 0


def test_endpoint_mismatch_is_rejected(planning) -> None:
    local_map = planning.custom_map()
    path = planning.linear_path([(0.0, 0.1), (0.0, 4.9)], local_map)
    report = _validate(
        planning, path, local_map, expected_start=(0.0, 0.1), expected_goal=(0.0, 4.9 + 1e-6)
    )
    assert "ENDPOINT_MISMATCH" in _kinds(report)


# --- E2-05: certificacion -------------------------------------------------------------


def test_linear_sweep_matches_translated_rectangle_oracle(planning) -> None:
    """Oraculo: la cuchilla 0.20 x 0.08 trasladada de (x, y0) a (x, y1) barre
    exactamente el rectangulo [x-0.10, x+0.10] x [y0-0.04, y1+0.04]."""
    tool = np.asarray(planning.TOOL.collision_polygon)
    hull = _chord_hull(tool, (0.1, 0.5), (0.1, 3.0))
    oracle = box(0.0, 0.46, 0.2, 3.04)
    assert hull.symmetric_difference(oracle).area <= 1e-12
    assert hull.area == pytest.approx(0.2 * 2.58, abs=1e-12)
    # Diagonal: area = area(K) + |d x b| con d el desplazamiento y b el ancho proyectado.
    diagonal = _chord_hull(tool, (0.0, 0.0), (0.3, 0.4))
    projected = 0.20 * 0.4 + 0.08 * 0.3
    assert diagonal.area == pytest.approx(0.2 * 0.08 + projected, abs=1e-12)


def test_smoothed_curve_invading_where_polyline_is_free_is_rejected(planning) -> None:
    """El tramo en S se separa de la cuerda; un obstaculo fuera del barrido de la cuerda
    pero dentro de la huella real de la curva hace rechazar la curva suavizada."""
    start, goal = (0.0, 0.1), (0.5, 1.1)
    probe = planning.custom_map(corridor=(-0.5, 1.0, 0.0, 1.3))
    curve = planning.smooth_path([start, goal], probe)
    u = 0.21
    x_curve, y_curve = curve.evaluate(u)
    corner = np.array([x_curve - 0.10, y_curve + 0.04])
    normal = np.array([1.0, -0.5]) / math.hypot(1.0, 0.5)
    center = corner + 0.034 * normal
    radius = 0.01
    local_map = planning.custom_map(
        [planning.exclusion("P1", *center, radius)], corridor=(-0.5, 1.0, 0.0, 1.3)
    )
    tool = np.asarray(planning.TOOL.collision_polygon)
    chord_sweep = _chord_hull(tool, start, goal)
    exclusion = local_map.exclusion_polygons()[0]
    assert chord_sweep.distance(exclusion) > 0.01  # precondicion: la polilinea no invade
    assert Polygon(tool + np.array([x_curve, y_curve])).intersects(exclusion)  # la curva si
    polyline = planning.linear_path([start, goal], local_map)
    assert _validate(planning, polyline, local_map).status is ValidationStatus.ACCEPTED
    smoothed = planning.smooth_path([start, goal], local_map)
    report = _validate(planning, smoothed, local_map)
    assert report.status is ValidationStatus.REJECTED and "COLLISION" in _kinds(report)


def test_subdivision_certifies_with_budget_and_is_uncertified_without(planning) -> None:
    local_map = planning.custom_map([planning.exclusion("P1", 0.0, 2.5, 0.18)])
    path = planning.smooth_path([(0.0, 0.1), (0.35, 1.5), (0.35, 3.5), (0.0, 4.9)], local_map)
    full = _validate(planning, path, local_map)
    assert full.status is ValidationStatus.ACCEPTED
    assert 0.0 < full.envelope_error_max_m <= planning.SAFETY.geometric_approximation_tolerance_m
    assert full.evaluated_intervals > len(path.segments)  # necesito subdividir
    shallow = ValidationConfig(max_subdivision_depth=2)
    report = _validate(planning, path, local_map, config=shallow)
    assert report.status is ValidationStatus.UNCERTIFIED
    assert {i["cause"] for i in report.uncertified_intervals} == {"max_subdivision_depth"}
    tiny = ValidationConfig(max_intervals=10)
    report = _validate(planning, path, local_map, config=tiny)
    assert report.status is ValidationStatus.UNCERTIFIED
    assert "max_intervals" in {i["cause"] for i in report.uncertified_intervals}


def test_accepted_envelope_never_exceeds_declared_budget(planning) -> None:
    local_map = planning.custom_map()
    for offset in (0.1, 0.3, 0.38):
        path = planning.smooth_path([(0.0, 0.1), (offset, 0.6), (offset, 4.9)], local_map)
        report = _validate(planning, path, local_map)
        assert report.status is ValidationStatus.ACCEPTED
        assert report.envelope_error_max_m <= report.envelope_error_budget_m


# --- E2-06: incertidumbre ---------------------------------------------------------------


def _nominal(inflate: str):
    """Escena nominal (plantas en y=1,2,3 centradas) bajo la politica pedida."""
    uncertainty = UncertaintyConfig(inflate=inflate)
    if inflate == "exclusions":
        return local_map_from_scene(build_fixture("nominal_three_plants", uncertainty=uncertainty))
    config = scene_config("nominal_inflate_tool", (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), margin_m=0.0)
    return local_map_from_scene(build_scene(config, uncertainty=uncertainty))


@pytest.mark.parametrize("inflate", ["exclusions", "tool"])
def test_margin_is_applied_once_under_both_policies(planning, inflate) -> None:
    """Oraculo: recta en x=0.35; cara izquierda de la cuchilla en 0.25; planta en 0 con
    proteccion 0.15 y margen 0.03 => despeje 0.25 - 0.18 = 0.07 m (no 0.04 m, que seria
    aplicar el margen dos veces), salvo el exceso del poligono exterior."""
    local_map = _nominal(inflate)
    path = planning.linear_path([(0.35, 0.1), (0.35, 4.9)], local_map)
    report = _validate(planning, path, local_map)
    assert report.status is ValidationStatus.ACCEPTED
    effective = report.effective_geometry
    if inflate == "tool":
        assert effective["tool_inflation_m"] == pytest.approx(0.03)
        assert all(v == pytest.approx(0.15) for v in effective["exclusion_total_radii_m"].values())
    else:
        assert effective["tool_inflation_m"] == 0.0
        assert all(v == pytest.approx(0.18) for v in effective["exclusion_total_radii_m"].values())
    excess = report.approximations["max_exclusion_discretization_excess_m"]
    assert 0.07 - excess - 1e-3 <= report.clearance_lower_bound_m <= 0.07 + 1e-9


@pytest.mark.parametrize("seed", range(40))
def test_enlarging_an_exclusion_never_accepts_a_previously_invalid_path(planning, seed) -> None:
    rng = np.random.default_rng(1000 + seed)
    x, y, r = rng.uniform(-0.3, 0.3), rng.uniform(1.0, 4.0), rng.uniform(0.05, 0.2)
    waypoints = [(0.0, 0.1)]
    for level in sorted(rng.uniform(0.4, 4.6, 3)):
        waypoints.append((float(rng.uniform(-0.35, 0.35)), float(level)))
    waypoints.append((0.0, 4.9))
    reports = []
    for growth in (0.0, 0.02, 0.1):
        local_map = planning.custom_map([planning.exclusion("P1", x, y, r + growth)])
        path = planning.smooth_path(waypoints, local_map)
        reports.append(_validate(planning, path, local_map).status)
    for before, after in itertools.pairwise(reports):
        if before is not ValidationStatus.ACCEPTED:
            assert after is not ValidationStatus.ACCEPTED


# --- E2-07: cobertura y despeje ------------------------------------------------------------


def test_rectilinear_coverage_matches_analytic_area_without_double_counting(planning) -> None:
    """Oraculo: corte 0.19 x 0.07 trasladado 4.8 m => area 0.19 · (4.8 + 0.07)."""
    local_map = planning.custom_map()
    expected = 0.19 * (4.8 + 0.07)
    for points in (
        [(0.0, 0.1), (0.0, 4.9)],
        [
            (0.0, 0.1),
            (0.0, 1.0),
            (0.0, 2.5),
            (0.0, 4.9),
        ],  # uniones solapadas: no se suman dos veces
    ):
        report = _validate(planning, planning.linear_path(points, local_map), local_map)
        coverage = report.coverage
        assert coverage["exact"] is True and coverage["error_bound_m2"] == 0.0
        assert coverage["swept_cut_area_m2"] == pytest.approx(expected, abs=1e-9, rel=0)
        assert coverage["covered_target_area_m2"] == pytest.approx(expected, abs=1e-9, rel=0)
    assert report.clearance_lower_bound_m is None  # sin exclusiones no hay cota que dar


def test_cut_and_collision_are_distinct_and_target_excludes_protection(planning) -> None:
    local_map = planning.custom_map([planning.exclusion("P1", 0.0, 2.5, 0.18)])
    path = planning.linear_path([(0.35, 0.1), (0.35, 4.9)], local_map)
    report = _validate(planning, path, local_map)
    geometry = report.effective_geometry
    collision = Polygon(geometry["tool_collision_polygon_W_orientation"]).area
    cut = Polygon(geometry["tool_cutting_polygon_W_orientation"]).area
    assert collision == pytest.approx(0.2 * 0.08) and cut == pytest.approx(0.19 * 0.07)
    assert report.coverage["target_area_m2"] < local_map.allowed_region().area


def test_curved_coverage_is_labelled_estimate_and_clearance_is_certified(planning) -> None:
    local_map = planning.custom_map([planning.exclusion("P1", 0.0, 2.5, 0.18)])
    path = planning.smooth_path([(0.0, 0.1), (0.35, 1.5), (0.35, 3.5), (0.0, 4.9)], local_map)
    report = _validate(planning, path, local_map)
    assert report.coverage["exact"] is False
    # A1E2-R2: estimacion rotulada con resolucion espacial declarada, sin cota de area.
    assert report.coverage["error_bound_m2"] is None
    assert 0.0 < report.coverage["chord_deviation_max_m"] <= 1e-5
    assert "estimacion" in report.coverage["label"] and "sin cota" in report.coverage["label"]
    assert report.clearance_lower_bound_m > planning.SAFETY.clearance_epsilon_m
    # La cota inferior certificada no supera la distancia real de muestras densas.
    tool = np.asarray(planning.TOOL.collision_polygon)
    exclusion = local_map.exclusion_polygons()[0]
    sampled = min(
        Polygon(tool + path.evaluate(s)).distance(exclusion) for s in np.linspace(0, 1, 2001)
    )
    assert report.clearance_lower_bound_m <= sampled + 1e-12
