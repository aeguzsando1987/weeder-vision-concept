"""Regresiones de los hallazgos R1-R12 del dictamen E1 (`docs/reviews/E1_review.md`).

Cada bloque cubre el hallazgo con controles positivos y rechazos, por constructor
directo y, cuando aplica, por configuracion. La ruta CLI se cubre en
`tests/regression/test_demo_validation.py`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from shapely.geometry import Polygon

from precision_weeder.core.config import (
    CalibrationConfig,
    ExclusionConfig,
    ProjectConfig,
    RunMode,
    SafetyConfig,
    SceneConfig,
    ToolConfig,
    UncertaintyConfig,
    load_project_config,
)
from precision_weeder.core.errors import (
    ConfigError,
    FrameMismatchError,
    InvalidPolygonError,
    MissingCalibrationError,
    NonFiniteValueError,
    PrecisionWeederError,
    ScenarioError,
    SchemaValidationError,
    UncertaintyBudgetError,
    UnitMismatchError,
)
from precision_weeder.core.types import (
    CalibrationReference,
    CalibrationStatus,
    Corridor,
    Frame,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
    circumscribed_radial_excess,
    exterior_circle_polygon,
    minimum_segments_for_excess,
    validate_polygon_points,
)
from precision_weeder.core.units import (
    CLEARANCE_EPSILON_M,
    GEOMETRIC_APPROXIMATION_TOLERANCE_M,
    LIMIT_COMPARISON_TOLERANCE,
    Provenance,
    SpatialDomain,
)
from precision_weeder.geometry.frames import Transform2D, world_from_vehicle
from precision_weeder.scenarios.fixtures import build_fixture
from precision_weeder.scenarios.synthetic import build_scene, require_discretization_budget
from precision_weeder.simulation.tool import ToolGeometry, ToolState

D = SpatialDomain.GROUND_METRIC_SYNTHETIC
VERIFIED = SpatialDomain.GROUND_METRIC_VERIFIED


def _scene_config(margin: float = 0.03) -> SceneConfig:
    return SceneConfig(exclusion=ExclusionConfig(uncertainty_margin_m=margin))


def _uncertainty(each: float, budget: float, inflate: str = "exclusions") -> UncertaintyConfig:
    return UncertaintyConfig(
        localization_bound_m=each,
        calibration_bound_m=each,
        tracking_bound_m=each,
        bound_budget_m=budget,
        inflate=inflate,  # type: ignore[arg-type]
    )


def _write_yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


# ============================================================================
# R1 - incertidumbre efectiva: una fuente de verdad y aplicacion coherente
# ============================================================================


def test_r1_default_scene_applies_the_declared_policy() -> None:
    scene = build_scene(SceneConfig())
    payload = scene.to_dict()["uncertainty"]
    assert scene.uncertainty.total == pytest.approx(0.03, abs=1e-15, rel=0)
    assert payload["budget"] == pytest.approx(0.03, abs=1e-15, rel=0)
    assert payload["inflate"] == "exclusions"
    assert payload["applied_tool_inflation"] == 0.0
    assert set(payload["applied_exclusion_margins"].values()) == {0.03}


def test_r1_bound_larger_than_applied_margin_is_rejected() -> None:
    with pytest.raises(UncertaintyBudgetError, match="margen"):
        build_scene(_scene_config(0.03), uncertainty=_uncertainty(0.02, 0.06))


def test_r1_margin_matching_raised_bounds_is_applied_and_recorded() -> None:
    scene = build_scene(_scene_config(0.06), uncertainty=_uncertainty(0.02, 0.06))
    payload = scene.to_dict()
    assert payload["uncertainty"]["total"] == pytest.approx(0.06, abs=1e-12, rel=0)
    assert all(e["uncertainty_margin"] == 0.06 for e in payload["exclusions"])
    radii = [e["total_radius"] for e in payload["exclusions"]]
    assert radii == pytest.approx([0.21] * 3, abs=1e-12, rel=0)


def test_r1_budget_is_a_maximum_not_the_inflation() -> None:
    """Con presupuesto holgado, el margen solo debe cubrir la suma de cotas."""
    scene = build_scene(_scene_config(0.03), uncertainty=_uncertainty(0.01, 0.06))
    assert scene.uncertainty.budget == 0.06
    assert scene.exclusions[0].uncertainty_margin == 0.03


def test_r1_bounds_above_budget_are_rejected() -> None:
    with pytest.raises(ValueError, match=r"presupuesto|excede"):
        _uncertainty(0.03, 0.06)
    bound = UncertaintyBound(0.03, 0.03, 0.03, D)
    with pytest.raises(UncertaintyBudgetError):
        UncertaintyPolicy(bound=bound, budget=0.06)


def test_r1_tool_inflation_moves_protection_without_duplicating_it() -> None:
    scene = build_scene(_scene_config(0.0), uncertainty=_uncertainty(0.01, 0.03, "tool"))
    assert scene.tool_inflation == pytest.approx(0.03, abs=1e-15, rel=0)
    assert all(e.uncertainty_margin == 0.0 for e in scene.exclusions)
    with pytest.raises(UncertaintyBudgetError, match="duplicar"):
        build_scene(_scene_config(0.03), uncertainty=_uncertainty(0.01, 0.03, "tool"))
    with pytest.raises(UncertaintyBudgetError):
        replace(scene, tool_inflation=0.0)


def test_r1_exclusion_policy_rejects_extra_tool_inflation() -> None:
    scene = build_scene(SceneConfig())
    with pytest.raises(UncertaintyBudgetError, match="dos veces"):
        replace(scene, tool_inflation=0.03)


def test_r1_inflated_tool_is_conservative() -> None:
    blade = ToolGeometry.rectangular("blade", 0.20, 0.08)
    inflated = blade.inflated_collision_shape(0.03)
    exact_round = blade.collision_shape.buffer(0.03, quad_segs=256)
    assert inflated.covers(exact_round)
    assert blade.inflated_collision_shape(0.0).equals(blade.collision_shape)


def test_r1_project_config_rejects_underprotecting_safety(
    configs_dir: Path, tmp_path: Path
) -> None:
    payload = yaml.safe_load((configs_dir / "safety.yaml").read_text())
    payload["safety"]["uncertainty"].update(
        localization_bound_m=0.02,
        calibration_bound_m=0.02,
        tracking_bound_m=0.02,
        bound_budget_m=0.06,
    )
    safety = _write_yaml(tmp_path / "safety.yaml", payload)
    with pytest.raises(ConfigError, match="incertidumbre"):
        load_project_config(
            configs_dir / "synthetic.yaml",
            configs_dir / "tool.yaml",
            safety,
            configs_dir / "kinematics.yaml",
        )


def test_r1_project_config_rejects_double_inflation() -> None:
    with pytest.raises(ValueError, match="duplicaria"):
        ProjectConfig(safety=SafetyConfig(uncertainty=_uncertainty(0.01, 0.03, "tool")))
    config = ProjectConfig(
        scene=_scene_config(0.0),
        safety=SafetyConfig(uncertainty=_uncertainty(0.01, 0.03, "tool")),
    )
    scene = build_scene(config.scene, uncertainty=config.safety.uncertainty)
    assert scene.tool_inflation == pytest.approx(0.03, abs=1e-15, rel=0)


def test_r1_metric_bounds_cannot_be_applied_to_graphic_units() -> None:
    graphic = SceneConfig(domain=SpatialDomain.GRAPHIC_UNSCALED)
    with pytest.raises(PrecisionWeederError):
        build_scene(graphic, uncertainty=UncertaintyConfig())
    with pytest.raises(ScenarioError):
        build_scene(graphic)


# ============================================================================
# R2 - escala verificada solo con evidencia y procedencia coherentes
# ============================================================================


def test_r2_verified_reference_requires_non_blank_evidence_and_verified_domain() -> None:
    with pytest.raises(MissingCalibrationError):
        CalibrationReference(
            "cam", status=CalibrationStatus.VERIFIED, scale_evidence="  \t", domain=VERIFIED
        )
    with pytest.raises(UnitMismatchError):
        CalibrationReference(
            "cam",
            status=CalibrationStatus.VERIFIED,
            scale_evidence="patron",
            domain=SpatialDomain.IMAGE_PX,
        )
    reference = CalibrationReference(
        "cam", status=CalibrationStatus.VERIFIED, scale_evidence="  patron 1 m  ", domain=VERIFIED
    )
    assert reference.scale_evidence == "patron 1 m"


def test_r2_calibrated_mode_without_evidence_is_rejected() -> None:
    with pytest.raises(MissingCalibrationError):
        ProjectConfig(
            mode=RunMode.CALIBRATED,
            scene=SceneConfig(domain=VERIFIED),
            tool=ToolConfig(domain=VERIFIED),
            calibration=CalibrationConfig(
                calibration_id="cam",
                status=CalibrationStatus.VERIFIED,
                scale_evidence=" ",
                domain=VERIFIED,
            ),
        )


def test_r2_declared_calibration_is_validated_in_every_mode() -> None:
    with pytest.raises(MissingCalibrationError):
        ProjectConfig(
            calibration=CalibrationConfig(calibration_id="cam", status=CalibrationStatus.VERIFIED)
        )


def test_r2_calibrated_yaml_with_evidence_loads(
    invalid_configs_dir: Path, configs_dir: Path
) -> None:
    config = load_project_config(
        invalid_configs_dir / "calibrated_without_calibration.yaml",
        invalid_configs_dir / "tool_verified.yaml",
        configs_dir / "safety.yaml",
        configs_dir / "kinematics.yaml",
        calibration={
            "calibration_id": "cam0",
            "status": "verified",
            "scale_evidence": "patron metrico independiente",
            "domain": "ground_metric_verified",
        },
    )
    assert config.calibration_reference.is_verified


def test_r2_synthetic_scenes_never_claim_verified_scale() -> None:
    with pytest.raises(UnitMismatchError, match="medicion"):
        build_scene(SceneConfig(domain=VERIFIED))
    for name in ("nominal_three_plants", "svg_graphic"):
        assert build_fixture(name).to_dict()["metric_scale_verified"] is False


def test_r2_synthetic_scene_rejects_measured_provenance() -> None:
    scene = build_scene(SceneConfig())
    with pytest.raises(ScenarioError, match="procedencia"):
        replace(scene, provenance=Provenance.MEASURED)
    measured_plant = replace(scene.plants[0], provenance=Provenance.MEASURED)
    with pytest.raises(ScenarioError, match="procedencia"):
        replace(scene, plants=(measured_plant, *scene.plants[1:]))


# ============================================================================
# R3 - marco y dominio verificados juntos
# ============================================================================


def test_r3_corridor_requires_same_frame_and_accepts_explicit_transform() -> None:
    corridor = Corridor.centered(1.0, 5.0, frame=Frame.WORLD, domain=D)
    in_vehicle = Point2(0.1, 0.2, Frame.VEHICLE, D)
    with pytest.raises(FrameMismatchError):
        corridor.contains_point(in_vehicle)
    in_world = world_from_vehicle(0.0, 2.0, 0.0, D).apply_point(in_vehicle)
    assert corridor.contains_point(in_world)
    vehicle_corridor = Corridor.centered(1.0, 5.0, frame=Frame.VEHICLE, domain=D)
    assert vehicle_corridor.contains_point(in_vehicle)


def test_r3_transform_comparison_requires_same_domain() -> None:
    metric = Transform2D(Frame.WORLD, Frame.TOOL, 1.0, 2.0, 0.3, D)
    pixels = Transform2D(Frame.WORLD, Frame.TOOL, 1.0, 2.0, 0.3, SpatialDomain.IMAGE_PX)
    assert not metric.allclose(pixels)
    assert metric.allclose(Transform2D(Frame.WORLD, Frame.TOOL, 1.0, 2.0, 0.3, D))


def test_r3_scene_components_share_the_scene_frame() -> None:
    scene = build_scene(SceneConfig())
    with pytest.raises(FrameMismatchError):
        replace(scene, frame=Frame.VEHICLE)
    vehicle_scene = build_scene(SceneConfig(frame=Frame.VEHICLE))
    assert vehicle_scene.corridor.frame is Frame.VEHICLE
    vehicle_position = replace(scene.plants[0].position, frame=Frame.VEHICLE)
    moved = replace(scene.plants[0], position=vehicle_position)
    with pytest.raises(FrameMismatchError):
        replace(scene, plants=(moved, *scene.plants[1:]))


# ============================================================================
# R4 - escena estructuralmente valida e inmutable
# ============================================================================


@pytest.mark.parametrize("field", ["advance_speed", "advance_heading"])
@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_r4_advance_must_be_finite(field: str, value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        replace(build_scene(SceneConfig()), **{field: value})


@pytest.mark.parametrize("value", [0.0, -0.2])
def test_r4_advance_speed_must_be_positive(value: float) -> None:
    with pytest.raises(SchemaValidationError):
        replace(build_scene(SceneConfig()), advance_speed=value)


def test_r4_every_plant_needs_exactly_one_matching_exclusion() -> None:
    scene = build_scene(SceneConfig())
    with pytest.raises(ScenarioError, match="sin exclusion"):
        replace(scene, exclusions=scene.exclusions[:-1])
    orphan = replace(scene.exclusions[0], plant_id="ghost")
    with pytest.raises(ScenarioError, match="sin planta"):
        replace(scene, exclusions=(*scene.exclusions, orphan))
    with pytest.raises(ScenarioError, match="duplicada"):
        replace(scene, exclusions=(*scene.exclusions, scene.exclusions[0]))


def test_r4_exclusion_must_be_centered_and_cover_required_protection() -> None:
    scene = build_scene(SceneConfig())
    first = scene.exclusions[0]
    displaced = replace(first, center=replace(first.center, y=first.center.y + 1e-9))
    with pytest.raises(ScenarioError, match="centrada"):
        replace(scene, exclusions=(displaced, *scene.exclusions[1:]))
    small = replace(first, protection_radius=0.149)
    with pytest.raises(ScenarioError, match="protege"):
        replace(scene, exclusions=(small, *scene.exclusions[1:]))
    larger = replace(first, protection_radius=0.2)
    assert replace(scene, exclusions=(larger, *scene.exclusions[1:])).exclusions[0] == larger


def test_r4_unknown_empty_scene_is_rejected_certified_empty_is_valid() -> None:
    scene = build_scene(SceneConfig())
    with pytest.raises(ScenarioError, match="no certifica"):
        replace(scene, plants=(), exclusions=())
    certified = build_fixture("empty_corridor_certified")
    assert certified.target_area() == pytest.approx(5.0, abs=1e-12, rel=0)


def test_r4_collections_are_owned_tuples() -> None:
    scene = build_scene(SceneConfig())
    plants, exclusions, notes = list(scene.plants), list(scene.exclusions), ["nota"]
    owned = replace(scene, plants=plants, exclusions=exclusions, notes=notes)
    plants.clear()
    exclusions.clear()
    notes.clear()
    assert isinstance(owned.plants, tuple) and len(owned.plants) == 3
    assert isinstance(owned.exclusions, tuple) and len(owned.exclusions) == 3
    assert owned.notes == ("nota",)


def test_r4_collection_elements_are_type_checked() -> None:
    scene = build_scene(SceneConfig())
    with pytest.raises(SchemaValidationError):
        replace(scene, plants=(*scene.plants, "planta"))


# ============================================================================
# R5 - origen de B = pivote; radio respecto del mismo origen
# ============================================================================


def test_r5_pivot_must_be_origin() -> None:
    polygon = np.array([[-0.1, -0.04], [0.1, -0.04], [0.1, 0.04], [-0.1, 0.04]])
    with pytest.raises(SchemaValidationError, match="pivote"):
        ToolGeometry("t", polygon, polygon, Point2(0.0, 1e-12, Frame.TOOL, D))


def test_r5_offset_blade_radius_contains_every_rotated_vertex() -> None:
    blade = ToolGeometry.rectangular("offset", 0.20, 0.08, pivot_offset=(0.15, 0.05))
    radius = blade.circumscribed_radius()
    expected = max(np.hypot(*vertex) for vertex in blade.collision_polygon)
    assert radius == pytest.approx(expected, abs=0, rel=0)
    for theta in np.linspace(-math.pi, math.pi, 73):
        placed = blade.collision_polygon_in(Transform2D(Frame.VEHICLE, Frame.TOOL, 0, 0, theta, D))
        assert max(math.hypot(x, y) for x, y in placed.exterior.coords) <= radius + 1e-12


# ============================================================================
# R6 - poligonos propios e inmutables
# ============================================================================


def test_r6_tool_polygons_are_isolated_from_inputs_and_outputs() -> None:
    collision = np.array([[-0.1, -0.04], [0.1, -0.04], [0.1, 0.04], [-0.1, 0.04]])
    cutting = collision * 0.5
    tool = ToolGeometry("owned", collision, cutting, Point2(0, 0, Frame.TOOL, D))
    area, cut_area = tool.collision_area, tool.cutting_area
    collision[:] = np.nan
    cutting[:] = 99.0
    with pytest.raises(ValueError):
        tool.collision_polygon[0, 0] = 5.0
    with pytest.raises(ValueError):
        tool.cutting_polygon.flags.writeable = True
    copy = tool.collision_polygon.copy()
    copy[:] = 0.0
    assert tool.collision_area == area and tool.cutting_area == cut_area
    assert np.all(np.isfinite(tool.collision_polygon))
    assert not np.shares_memory(tool.collision_polygon, tool.cutting_polygon)


def test_r6_same_input_array_for_both_polygons_is_not_aliased() -> None:
    square = np.array([[-0.1, -0.04], [0.1, -0.04], [0.1, 0.04], [-0.1, 0.04]])
    tool = ToolGeometry("shared", square, square, Point2(0, 0, Frame.TOOL, D))
    assert not np.shares_memory(tool.collision_polygon, tool.cutting_polygon)
    assert replace(tool, tool_id="copia").collision_area == tool.collision_area


def test_r6_validated_polygon_is_a_frozen_copy() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    frozen = validate_polygon_points(source, "tri")
    source[0, 0] = 7.0
    assert frozen[0, 0] == 0.0
    assert not frozen.flags.writeable


# ============================================================================
# R9 - vectores de estado y contacto inequivocos
# ============================================================================


def _state(**kwargs: object) -> ToolState:
    return ToolState(0.0, Point2(0, 0, Frame.VEHICLE, D), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["velocity", "acceleration"])
@pytest.mark.parametrize("value", [(), (1.0,), (1.0, 2.0, 3.0), ((1.0, 2.0),), "12", None])
def test_r9_vectors_need_exactly_two_components(field: str, value: object) -> None:
    with pytest.raises(SchemaValidationError):
        _state(**{field: value})


@pytest.mark.parametrize("field", ["velocity", "acceleration"])
@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_r9_each_component_must_be_finite(field: str, index: int, bad: float) -> None:
    vector = [0.5, -0.25]
    vector[index] = bad
    with pytest.raises(NonFiniteValueError):
        _state(**{field: tuple(vector)})


def test_r9_valid_vectors_are_normalized_to_float_tuples() -> None:
    state = _state(velocity=np.array([1, 2]), acceleration=[0.5, -0.5])
    assert state.velocity == (1.0, 2.0)
    assert state.acceleration == (0.5, -0.5)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, 0.0])
def test_r9_contact_rejects_non_booleans(value: object) -> None:
    with pytest.raises(SchemaValidationError, match="booleano"):
        _state(contact=value)


@pytest.mark.parametrize("value", [True, False, np.bool_(True), np.bool_(False)])
def test_r9_contact_accepts_real_booleans(value: object) -> None:
    state = _state(contact=value)
    assert state.contact is bool(value)


# ============================================================================
# R10 - validez poligonal invariante a traslacion
# ============================================================================


@pytest.mark.parametrize("offset", [0.0, 1e4, -1e6, 1e8])
def test_r10_small_triangle_is_valid_anywhere(offset: float) -> None:
    triangle = np.array([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02]]) + offset
    ring = validate_polygon_points(triangle, "tri")
    assert ring.shape == (3, 2)


def test_r10_explicit_closure_is_dropped_exactly() -> None:
    closed = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]
    assert validate_polygon_points(closed, "closed").shape == (3, 2)
    nearly_closed = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1e-9, 1e-9]]
    assert validate_polygon_points(nearly_closed, "near").shape == (4, 2)


def test_r10_repeated_vertices_do_not_create_a_polygon() -> None:
    assert validate_polygon_points([[0, 0], [1, 0], [1, 0], [1, 1], [0, 1]], "repeated").shape == (
        5,
        2,
    )
    with pytest.raises(InvalidPolygonError, match="distintos"):
        validate_polygon_points([[0, 0], [1, 0], [1, 0], [0, 0]], "two")
    with pytest.raises(InvalidPolygonError):
        validate_polygon_points([[1e4, 1e4], [1e4 + 1, 1e4], [1e4 + 2, 1e4]], "collinear")


@pytest.mark.parametrize("offset", [0.0, 1e4])
def test_r10_self_intersection_is_rejected_anywhere(offset: float) -> None:
    bowtie = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]) + offset
    with pytest.raises(InvalidPolygonError):
        validate_polygon_points(bowtie, "bowtie")


# ============================================================================
# R11 - corte cubierto por la huella de colision almacenada
# ============================================================================


@pytest.mark.parametrize("shift", [(5e-6, 0), (-5e-6, 0), (0, 5e-6), (0, -5e-6), (1e-12, 0)])
def test_r11_any_protrusion_is_rejected(shift: tuple[float, float]) -> None:
    tool = ToolGeometry.rectangular("blade", 0.2, 0.08)
    moved = tool.collision_polygon + np.array(shift)
    with pytest.raises(InvalidPolygonError, match="sobresale"):
        replace(tool, cutting_polygon=moved)


def test_r11_identical_and_interior_cuts_are_accepted() -> None:
    tool = ToolGeometry.rectangular("blade", 0.2, 0.08)
    same = replace(tool, cutting_polygon=tool.collision_polygon)
    assert same.collision_shape.covers(same.cutting_shape)
    inner = ToolGeometry.rectangular("blade", 0.2, 0.08, cutting_inset=0.01)
    assert inner.collision_shape.covers(inner.cutting_shape)


def test_r11_rule_is_unit_independent() -> None:
    tool = ToolGeometry.rectangular("svg", 30.0, 10.0, domain=SpatialDomain.GRAPHIC_UNSCALED)
    with pytest.raises(InvalidPolygonError):
        replace(tool, cutting_polygon=tool.collision_polygon + np.array([1e-3, 0.0]))


# ============================================================================
# R12 - exceso conservador de discretizacion explicito y acotado
# ============================================================================


def test_r12_radial_excess_matches_analytic_oracle() -> None:
    excess = circumscribed_radial_excess(0.18, 64)
    assert excess == pytest.approx(2.17079364670658e-4, abs=1e-15, rel=0)
    vertices = exterior_circle_polygon((0.0, 0.0), 0.18, 64)
    measured = float(np.max(np.hypot(vertices[:, 0], vertices[:, 1]))) - 0.18
    assert measured == pytest.approx(excess, abs=1e-15, rel=0)
    assert excess > GEOMETRIC_APPROXIMATION_TOLERANCE_M  # no se afirma lo contrario


def test_r12_minimum_segments_for_a_linear_budget() -> None:
    assert minimum_segments_for_excess(0.18, 1e-5) == 299
    assert circumscribed_radial_excess(0.18, 299) <= 1e-5 < circumscribed_radial_excess(0.18, 298)


def test_r12_excess_is_exported_per_exclusion() -> None:
    payload = build_fixture("nominal_three_plants").to_dict()
    for exclusion in payload["exclusions"]:
        assert exclusion["discretization_excess"] == pytest.approx(
            circumscribed_radial_excess(0.18, 64), abs=0, rel=0
        )


def test_r12_project_config_bounds_the_conservative_excess() -> None:
    coarse = SceneConfig(exclusion=ExclusionConfig(polygon_segments=8))
    with pytest.raises(ValueError, match="discretizacion"):
        ProjectConfig(scene=coarse)
    ProjectConfig(scene=SceneConfig(exclusion=ExclusionConfig(polygon_segments=64)))


def test_r12_tolerance_hierarchy_keeps_tolerances_below_clearance() -> None:
    assert LIMIT_COMPARISON_TOLERANCE < GEOMETRIC_APPROXIMATION_TOLERANCE_M < CLEARANCE_EPSILON_M
    with pytest.raises(ValueError, match="contacto"):
        SafetyConfig(geometric_approximation_tolerance_m=1e-3, clearance_epsilon_m=1e-4)


def test_r12_conservative_excess_narrows_the_passage_as_documented() -> None:
    """El exceso conservador estrecha el paso: 0.22 m nominal menos dos excesos."""
    scene = build_fixture("narrow_passage")
    left, right = scene.exclusions
    gap = left.as_polygon().distance(right.as_polygon())
    expected = 0.22 - 2 * circumscribed_radial_excess(0.18, 64)
    assert gap == pytest.approx(expected, abs=1e-12, rel=0)
    assert gap < 0.22


def test_r12_tangency_is_never_accepted_by_tolerance() -> None:
    """Una huella a distancia exactamente cero de una exclusion no tiene despeje."""
    exclusion = Polygon(exterior_circle_polygon((0.0, 0.0), 0.18, 64))
    tool = ToolGeometry.rectangular("blade", 0.2, 0.08)
    apothem_x = 0.18 + 0.1
    placed = tool.collision_polygon_in(Transform2D(Frame.WORLD, Frame.TOOL, apothem_x, 0, 0, D))
    assert placed.distance(exclusion) <= CLEARANCE_EPSILON_M


# ============================================================================
# R12 (cierre) - presupuesto de discretizacion sobre exclusiones construidas
# ============================================================================


def _limit(value: float) -> SafetyConfig:
    return SafetyConfig(max_conservative_discretization_excess_m=value)


def test_r12_scene_check_uses_the_built_geometry() -> None:
    scene = build_fixture("nominal_three_plants")
    excess = circumscribed_radial_excess(0.18, 64)
    require_discretization_budget(scene, _limit(5e-4))
    require_discretization_budget(scene, _limit(excess))  # el limite exacto se admite
    with pytest.raises(ConfigError, match="discretizacion"):
        require_discretization_budget(scene, _limit(excess * (1 - 1e-9)))


def test_r12_scene_check_follows_uncertainty_overrides() -> None:
    """Un margen mayor cambia el radio real: el mismo limite puede dejar de cumplirse."""
    limit = _limit(2.3e-4)  # entre el exceso para r=0.18 (2.17e-4) y r=0.21 (2.53e-4)
    require_discretization_budget(build_fixture("nominal_three_plants"), limit)
    raised = build_scene(_scene_config(0.06), uncertainty=_uncertainty(0.02, 0.06))
    with pytest.raises(ConfigError, match=r"radio 0\.21 m"):
        require_discretization_budget(raised, limit)


@pytest.mark.parametrize("name", ["nominal_three_plants", "overlapping_exclusions"])
def test_r12_every_metric_catalog_scene_is_checked(name: str) -> None:
    with pytest.raises(ConfigError):
        require_discretization_budget(build_fixture(name), _limit(1e-5))


def test_r12_graphic_scene_is_exempt_by_domain() -> None:
    """El presupuesto esta en metros: no se aplica a unidades graficas del SVG."""
    require_discretization_budget(build_fixture("svg_graphic"), _limit(1e-9))
    require_discretization_budget(build_fixture("empty_corridor_certified"), _limit(1e-9))


def test_r12_config_and_scene_checks_agree() -> None:
    """La regla comun produce el mismo veredicto antes y despues de construir."""
    for segments, limit in ((64, 5e-4), (64, 1e-5), (299, 1e-5), (298, 1e-5)):
        config = SceneConfig(exclusion=ExclusionConfig(polygon_segments=segments))
        try:
            ProjectConfig(scene=config, safety=_limit(limit))
            config_ok = True
        except ValueError:
            config_ok = False
        try:
            require_discretization_budget(build_scene(config), _limit(limit))
            scene_ok = True
        except ConfigError:
            scene_ok = False
        assert config_ok == scene_ok == (circumscribed_radial_excess(0.18, segments) <= limit)


# ============================================================================
# N1 - pivot_offset con forma exacta (2,) y finito
# ============================================================================


@pytest.mark.parametrize(
    "offset", [(), (0.0,), (0.0, 0.0, 0.0), ((0.0, 0.0),), np.zeros((2, 1)), "00", None]
)
def test_n1_pivot_offset_needs_exactly_two_components(offset: object) -> None:
    with pytest.raises(SchemaValidationError, match="pivot_offset"):
        ToolGeometry.rectangular("blade", 0.2, 0.08, pivot_offset=offset)  # type: ignore[arg-type]


@pytest.mark.parametrize("index", [0, 1])
@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_n1_pivot_offset_components_must_be_finite(index: int, bad: float) -> None:
    offset = [0.05, -0.02]
    offset[index] = bad
    with pytest.raises(NonFiniteValueError, match="pivot_offset"):
        ToolGeometry.rectangular("blade", 0.2, 0.08, pivot_offset=tuple(offset))


def test_n1_three_component_offset_is_not_truncated() -> None:
    with pytest.raises(SchemaValidationError):
        ToolGeometry.rectangular("blade", 0.2, 0.08, pivot_offset=(0.0, 0.0, math.nan))


@pytest.mark.parametrize("offset", [(0.1, 0.0), [0.05, -0.02], np.array([-0.03, 0.01])])
def test_n1_valid_nonzero_offsets_are_kept(offset: object) -> None:
    blade = ToolGeometry.rectangular("blade", 0.2, 0.08, pivot_offset=offset)  # type: ignore[arg-type]
    centroid = blade.collision_shape.centroid
    ox, oy = (float(v) for v in offset)  # type: ignore[union-attr]
    assert (centroid.x, centroid.y) == pytest.approx((-ox, -oy), abs=1e-12, rel=0)
    assert (blade.pivot.x, blade.pivot.y) == (0.0, 0.0)
