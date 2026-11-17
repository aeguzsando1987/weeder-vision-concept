"""Escenas sinteticas: determinismo, geometria conocida y casos adversos."""

from __future__ import annotations

import json
import math

import pytest
from shapely.geometry import Point as ShapelyPoint

from precision_weeder.core.config import SceneConfig
from precision_weeder.core.errors import ScenarioError, UnitMismatchError
from precision_weeder.core.types import Frame, Point2
from precision_weeder.core.units import SpatialDomain
from precision_weeder.scenarios.fixtures import (
    SCENE_BUILDERS,
    build_fixture,
    default_tool_geometry,
    fixture_names,
    svg_tool_geometry,
    svg_tool_transform,
)
from precision_weeder.scenarios.synthetic import SyntheticScene, build_scene

TOOL_EXTENT_X_M = 0.20


# --- determinismo ------------------------------------------------------------


@pytest.mark.parametrize("name", list(SCENE_BUILDERS))
def test_same_seed_produces_identical_scene(name: str) -> None:
    """Misma semilla, entradas exactamente iguales (§5 E1)."""
    first, second = build_fixture(name), build_fixture(name)
    assert first.to_json() == second.to_json()
    assert first == second


def test_different_seed_changes_jittered_scene() -> None:
    base = SceneConfig(scene_id="jitter", jitter_x_m=0.02, jitter_y_m=0.05, seed=1)
    other = base.model_copy(update={"seed": 2})
    assert build_scene(base).to_json() != build_scene(other).to_json()


def test_jitter_is_reproducible_across_instances() -> None:
    config = SceneConfig(scene_id="jitter", jitter_x_m=0.02, jitter_y_m=0.05, seed=7)
    positions = [
        tuple((plant.position.x, plant.position.y) for plant in build_scene(config).plants)
        for _ in range(3)
    ]
    assert positions[0] == positions[1] == positions[2]


# --- escena nominal ----------------------------------------------------------


def test_nominal_scene_matches_declared_synthetic_values() -> None:
    scene = build_fixture("nominal_three_plants")
    assert len(scene.plants) == 3
    assert [plant.position.y for plant in scene.plants] == [1.0, 2.0, 3.0]
    assert all(plant.position.x == 0.0 for plant in scene.plants)
    assert scene.corridor.width == pytest.approx(1.0)
    assert scene.corridor.length == pytest.approx(5.0)
    assert scene.advance_speed == pytest.approx(0.20)
    assert all(exclusion.total_radius == pytest.approx(0.18) for exclusion in scene.exclusions)


def test_nominal_scene_is_feasible_for_the_default_blade() -> None:
    """Existe al menos un caso factible: el hueco supera la extension lateral."""
    scene = build_fixture("nominal_three_plants")
    assert scene.lateral_gaps()["widest"] > default_tool_geometry().extent_x


def test_target_area_oracle_matches_analytic_value() -> None:
    """Oraculo: area del corredor menos tres discos disjuntos de radio 0.18 m."""
    scene = build_fixture("nominal_three_plants")
    analytic = 5.0 - 3.0 * math.pi * 0.18**2
    # El poligono exterior excede ligeramente el circulo; el area objetivo es menor.
    assert scene.target_area() < analytic
    assert scene.target_area() == pytest.approx(analytic, rel=2e-3)


def test_uncertainty_split_sums_to_declared_margin() -> None:
    scene = build_fixture("nominal_three_plants")
    assert scene.uncertainty.total == pytest.approx(0.03)
    assert scene.exclusions[0].uncertainty_margin == pytest.approx(0.03)


def test_scene_separates_reference_truth_from_estimation() -> None:
    """Las plantas son verdad de referencia con procedencia sintetica explicita."""
    scene = build_fixture("nominal_three_plants")
    payload = scene.to_dict()
    assert payload["provenance"] == "synthetic"
    assert payload["metric_scale_verified"] is False
    assert all(plant["provenance"] == "synthetic" for plant in payload["plants"])


# --- casos adversos ----------------------------------------------------------


def test_empty_corridor_is_certified_not_merely_empty() -> None:
    scene = build_fixture("empty_corridor_certified")
    assert scene.plants == ()
    assert scene.free_space_certified is True
    assert scene.target_area() == pytest.approx(scene.corridor.as_polygon().area)


def test_isolated_plant_has_single_band() -> None:
    scene = build_fixture("isolated_plant")
    assert len(scene.plants) == 1
    assert scene.lateral_gaps()["bands"] == 1.0


def test_overlapping_exclusions_actually_overlap() -> None:
    """Oraculo independiente: interseccion de discos no vacia."""
    scene = build_fixture("overlapping_exclusions")
    first, second = scene.exclusions
    distance = math.dist((first.center.x, first.center.y), (second.center.x, second.center.y))
    assert distance < first.total_radius + second.total_radius
    assert first.as_polygon().intersects(second.as_polygon())
    assert scene.lateral_gaps()["between"] < 0.0


def test_narrow_passage_is_narrow_but_not_blocked() -> None:
    scene = build_fixture("narrow_passage")
    gaps = scene.lateral_gaps()
    assert gaps["between"] == pytest.approx(0.22, abs=1e-12, rel=0)
    assert TOOL_EXTENT_X_M < gaps["widest"] < TOOL_EXTENT_X_M + 0.03


def test_blocked_corridor_leaves_no_gap_for_the_blade() -> None:
    """Caso adverso: debe conducir a abstencion, no a una ruta de menor costo."""
    scene = build_fixture("blocked_corridor")
    assert scene.lateral_gaps()["widest"] < default_tool_geometry().extent_x


def test_irregular_spacing_has_uneven_intervals() -> None:
    scene = build_fixture("irregular_spacing")
    gaps = [
        second.position.y - first.position.y
        for first, second in zip(scene.plants, scene.plants[1:], strict=False)
    ]
    assert len({round(gap, 3) for gap in gaps}) == len(gaps)


def test_off_center_plants_are_laterally_displaced() -> None:
    scene = build_fixture("off_center_plant")
    offsets = sorted(plant.position.x for plant in scene.plants)
    assert offsets == pytest.approx([-0.25, 0.30])


def test_plant_on_corridor_edge_pushes_exclusion_outside() -> None:
    scene = build_fixture("plant_on_corridor_edge")
    exclusion = scene.exclusions[0]
    assert exclusion.center.x == pytest.approx(scene.corridor.x_max)
    assert not scene.corridor.as_polygon().covers(exclusion.as_polygon())
    assert scene.lateral_gaps()["right"] < 0.0


def test_plant_outside_corridor_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="fuera del corredor"):
        build_scene(
            SceneConfig(
                scene_id="out_of_bounds",
                plants={"positions_y_m": (2.0,), "offsets_x_m": (0.9,)},  # type: ignore[arg-type]
            )
        )


# --- escena grafica del SVG --------------------------------------------------


def test_svg_scene_keeps_graphic_units() -> None:
    """No se convierten unidades graficas a metros (§1.2)."""
    scene = build_fixture("svg_graphic")
    assert scene.domain is SpatialDomain.GRAPHIC_UNSCALED
    assert scene.unit.value == "svg_unit"
    assert scene.is_metric is False
    assert scene.to_dict()["metric_scale_verified"] is False


def test_svg_scene_reproduces_source_geometry() -> None:
    scene = build_fixture("svg_graphic")
    assert [plant.position.y for plant in scene.plants] == [65.0, 165.0, 265.0]
    assert all(plant.position.x == pytest.approx(169.0) for plant in scene.plants)
    assert scene.corridor.x_min == pytest.approx(90.0)
    assert scene.corridor.width == pytest.approx(160.0)
    assert scene.corridor.length == pytest.approx(330.0)
    assert scene.exclusions[0].total_radius == pytest.approx(32.0)
    assert scene.plants[0].stem_radius == pytest.approx(4.5)


def test_svg_scene_does_not_flip_axes() -> None:
    """Se conserva el sentido del eje `y` del SVG; la inversion es solo de dibujo."""
    scene = build_fixture("svg_graphic")
    assert scene.plants[0].position.y < scene.plants[-1].position.y
    assert any("no se invierte ningun eje" in note for note in scene.notes)


def test_svg_tool_matches_source_rectangle() -> None:
    geometry, transform = svg_tool_geometry(), svg_tool_transform()
    placed = geometry.collision_polygon_in(transform)
    bounds = placed.bounds
    assert bounds == pytest.approx((104.0, 243.0, 134.0, 253.0), abs=1e-9, rel=0)
    pivot = transform.apply_point(geometry.pivot)
    assert (pivot.x, pivot.y) == pytest.approx((119.0, 248.0), abs=1e-9, rel=0)


def test_svg_scene_rejects_mixing_with_metric_geometry() -> None:
    scene = build_fixture("svg_graphic")
    metric_point = Point2(
        x=0.0, y=1.0, frame=Frame.WORLD, domain=SpatialDomain.GROUND_METRIC_SYNTHETIC
    )
    with pytest.raises(UnitMismatchError):
        scene.corridor.contains_point(metric_point)


# --- invariantes del contrato de escena --------------------------------------


@pytest.mark.parametrize("name", list(SCENE_BUILDERS))
def test_every_scene_is_internally_consistent(name: str) -> None:
    scene = build_fixture(name)
    assert len(scene.exclusions) == len(scene.plants)
    for plant, exclusion in zip(scene.plants, scene.exclusions, strict=True):
        assert exclusion.plant_id == plant.plant_id
        assert exclusion.total_radius >= plant.protection_radius
        assert exclusion.as_polygon().covers(ShapelyPoint(plant.position.x, plant.position.y))
    assert scene.target_area() >= 0.0


@pytest.mark.parametrize("name", list(SCENE_BUILDERS))
def test_every_scene_serializes_to_stable_json(name: str) -> None:
    scene = build_fixture(name)
    payload = json.loads(scene.to_json())
    assert payload["scene_id"] == scene.scene_id
    assert payload["unit"] == scene.unit.value
    assert payload["advance"]["speed_unit"] == f"{scene.unit.value}/s"
    assert payload["corridor"]["unit"] == scene.unit.value
    assert payload["uncertainty"]["unit"] == scene.unit.value


def test_unknown_fixture_name_is_rejected() -> None:
    with pytest.raises(ScenarioError, match="escena desconocida"):
        build_fixture("no_existe")


def test_fixture_catalog_covers_required_cases() -> None:
    required = {
        "nominal_three_plants",
        "five_plants",
        "empty_corridor_certified",
        "isolated_plant",
        "irregular_spacing",
        "off_center_plant",
        "overlapping_exclusions",
        "narrow_passage",
        "blocked_corridor",
        "svg_graphic",
    }
    assert required <= set(fixture_names())


def test_scene_rejects_duplicate_plant_ids() -> None:
    scene = build_fixture("nominal_three_plants")
    with pytest.raises(ScenarioError, match="plant_id duplicado"):
        SyntheticScene(
            scene_id="dup",
            description="",
            frame=scene.frame,
            domain=scene.domain,
            corridor=scene.corridor,
            plants=(scene.plants[0], scene.plants[0]),
            exclusions=(),
            advance_speed=scene.advance_speed,
            advance_heading=0.0,
            uncertainty=scene.uncertainty,
            seed=0,
        )


def test_scene_rejects_mixed_domains() -> None:
    scene = build_fixture("nominal_three_plants")
    graphic = build_fixture("svg_graphic")
    with pytest.raises(UnitMismatchError):
        SyntheticScene(
            scene_id="mixed",
            description="",
            frame=scene.frame,
            domain=scene.domain,
            corridor=graphic.corridor,
            plants=scene.plants,
            exclusions=scene.exclusions,
            advance_speed=scene.advance_speed,
            advance_heading=0.0,
            uncertainty=scene.uncertainty,
            seed=0,
        )
