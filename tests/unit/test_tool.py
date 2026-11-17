"""Geometria y estado de la herramienta."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from precision_weeder.core.config import ToolConfig
from precision_weeder.core.errors import (
    InvalidDimensionError,
    InvalidPolygonError,
    NonFiniteValueError,
    SchemaValidationError,
)
from precision_weeder.core.types import Frame, OrientationMode, Point2, SimulationState
from precision_weeder.core.units import TRANSFORM_TOLERANCE, SpatialDomain
from precision_weeder.geometry.frames import tool_from_vehicle, world_from_vehicle
from precision_weeder.simulation.tool import (
    ToolGeometry,
    ToolState,
    tool_geometry_from_config,
    world_footprints,
)

DOMAIN = SpatialDomain.GROUND_METRIC_SYNTHETIC
TOL = TRANSFORM_TOLERANCE


def _blade(cutting_inset: float = 0.005) -> ToolGeometry:
    return ToolGeometry.rectangular(
        tool_id="blade", length_x=0.20, width_y=0.08, cutting_inset=cutting_inset, domain=DOMAIN
    )


# --- construccion ------------------------------------------------------------


def test_rectangular_blade_has_long_side_on_x() -> None:
    blade = _blade()
    assert blade.extent_x == pytest.approx(0.20, abs=TOL, rel=0)
    assert blade.extent_y == pytest.approx(0.08, abs=TOL, rel=0)
    assert blade.extent_x > blade.extent_y


def test_collision_and_cutting_footprints_may_differ() -> None:
    blade = _blade(cutting_inset=0.005)
    assert blade.collision_area == pytest.approx(0.20 * 0.08, abs=1e-12, rel=0)
    assert blade.cutting_area == pytest.approx(0.19 * 0.07, abs=1e-12, rel=0)
    assert blade.cutting_area < blade.collision_area
    assert blade.collision_shape.covers(blade.cutting_shape)


def test_pivot_is_origin_of_tool_frame() -> None:
    blade = _blade()
    assert (blade.pivot.x, blade.pivot.y) == (0.0, 0.0)
    assert blade.pivot.frame is Frame.TOOL
    centroid = blade.collision_shape.centroid
    assert (centroid.x, centroid.y) == pytest.approx((0.0, 0.0), abs=TOL, rel=0)


def test_pivot_offset_shifts_polygon_not_origin() -> None:
    """Con desplazamiento, el origen de B sigue siendo el pivote."""
    blade = ToolGeometry.rectangular(
        tool_id="offset_blade",
        length_x=0.20,
        width_y=0.08,
        pivot_offset=(0.10, 0.0),
        domain=DOMAIN,
    )
    assert (blade.pivot.x, blade.pivot.y) == (0.0, 0.0)
    centroid = blade.collision_shape.centroid
    assert centroid.x == pytest.approx(-0.10, abs=TOL, rel=0)
    assert blade.collision_polygon[:, 0].max() == pytest.approx(0.0, abs=TOL, rel=0)


def test_circumscribed_radius_matches_analytic_oracle() -> None:
    """Oraculo: media diagonal del rectangulo centrado en el pivote."""
    blade = _blade()
    assert blade.circumscribed_radius() == pytest.approx(
        0.5 * math.hypot(0.20, 0.08), abs=1e-12, rel=0
    )


def test_geometry_from_config_matches_defaults() -> None:
    blade = tool_geometry_from_config(ToolConfig())
    assert blade.tool_id == "blade_rect_200x80"
    assert blade.orientation_mode is OrientationMode.FIXED
    assert blade.extent_x == pytest.approx(0.20, abs=TOL, rel=0)


# --- rechazo de entradas invalidas -------------------------------------------


@pytest.mark.parametrize("value", [0.0, -0.2])
def test_rectangular_rejects_non_positive_dimensions(value: float) -> None:
    with pytest.raises(InvalidDimensionError):
        ToolGeometry.rectangular(tool_id="bad", length_x=value, width_y=0.08, domain=DOMAIN)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rectangular_rejects_non_finite_dimensions(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        ToolGeometry.rectangular(tool_id="bad", length_x=value, width_y=0.08, domain=DOMAIN)


def test_rectangular_rejects_degenerate_cutting_inset() -> None:
    with pytest.raises(InvalidPolygonError):
        ToolGeometry.rectangular(
            tool_id="bad", length_x=0.20, width_y=0.08, cutting_inset=0.04, domain=DOMAIN
        )


def test_geometry_rejects_self_intersecting_polygon() -> None:
    bowtie = np.array([[0.0, 0.0], [0.1, 0.1], [0.1, 0.0], [0.0, 0.1]])
    with pytest.raises(InvalidPolygonError):
        ToolGeometry(
            tool_id="bowtie",
            collision_polygon=bowtie,
            cutting_polygon=bowtie,
            pivot=Point2(x=0.0, y=0.0, frame=Frame.TOOL, domain=DOMAIN),
            domain=DOMAIN,
        )


def test_geometry_rejects_cutting_outside_collision() -> None:
    small = np.array([[-0.05, -0.02], [0.05, -0.02], [0.05, 0.02], [-0.05, 0.02]])
    big = np.array([[-0.5, -0.2], [0.5, -0.2], [0.5, 0.2], [-0.5, 0.2]])
    with pytest.raises(InvalidPolygonError):
        ToolGeometry(
            tool_id="inverted",
            collision_polygon=small,
            cutting_polygon=big,
            pivot=Point2(x=0.0, y=0.0, frame=Frame.TOOL, domain=DOMAIN),
            domain=DOMAIN,
        )


def test_geometry_rejects_pivot_in_wrong_frame() -> None:
    square = np.array([[-0.1, -0.04], [0.1, -0.04], [0.1, 0.04], [-0.1, 0.04]])
    with pytest.raises(SchemaValidationError):
        ToolGeometry(
            tool_id="blade",
            collision_polygon=square,
            cutting_polygon=square,
            pivot=Point2(x=0.0, y=0.0, frame=Frame.WORLD, domain=DOMAIN),
            domain=DOMAIN,
        )


# --- colocacion en el mundo --------------------------------------------------


def test_placed_footprint_is_translated_rectangle() -> None:
    """Oraculo: rectangulo trasladado, con solapamiento conocido."""
    blade = _blade(cutting_inset=0.0)
    transform = tool_from_vehicle(0.30, 1.00, 0.0, DOMAIN)
    placed = blade.collision_polygon_in(transform)
    expected = Polygon([[0.20, 0.96], [0.40, 0.96], [0.40, 1.04], [0.20, 1.04]])
    assert placed.area == pytest.approx(expected.area, abs=1e-12, rel=0)
    assert placed.symmetric_difference(expected).area == pytest.approx(0.0, abs=1e-12, rel=0)


def test_rotated_footprint_preserves_area() -> None:
    blade = _blade()
    rotated = blade.collision_polygon_in(tool_from_vehicle(0.0, 0.0, 0.7, DOMAIN))
    assert rotated.area == pytest.approx(blade.collision_area, abs=1e-12, rel=0)


def test_placement_rejects_transform_with_wrong_child() -> None:
    blade = _blade()
    with pytest.raises(SchemaValidationError):
        blade.collision_polygon_in(world_from_vehicle(0.0, 0.0, 0.0, DOMAIN))


# --- ToolState ---------------------------------------------------------------


def _state(**kwargs: object) -> ToolState:
    defaults: dict[str, object] = {
        "time_s": 0.0,
        "position": Point2(x=0.0, y=0.0, frame=Frame.VEHICLE, domain=DOMAIN),
    }
    defaults.update(kwargs)
    return ToolState(**defaults)  # type: ignore[arg-type]


def test_tool_state_defaults_are_idle_and_without_contact() -> None:
    state = _state()
    assert state.machine_state is SimulationState.IDLE
    assert state.contact is False
    assert state.depth_z is None
    assert state.speed == pytest.approx(0.0)


def test_tool_state_rejects_depth_in_h1() -> None:
    """La interfaz Z esta reservada: no se simulan elevaciones instantaneas (§2)."""
    with pytest.raises(SchemaValidationError):
        _state(depth_z=-0.03)


def test_tool_state_rejects_negative_time() -> None:
    with pytest.raises(InvalidDimensionError):
        _state(time_s=-0.01)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_tool_state_rejects_non_finite_derivatives(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        _state(velocity=(value, 0.0))
    with pytest.raises(NonFiniteValueError):
        _state(acceleration=(0.0, value))


def test_tool_state_rejects_tool_frame_position() -> None:
    with pytest.raises(SchemaValidationError):
        _state(position=Point2(x=0.0, y=0.0, frame=Frame.TOOL, domain=DOMAIN))


# --- composicion W <- V <- B -------------------------------------------------


def test_world_footprints_apply_vehicle_advance_once() -> None:
    blade = _blade(cutting_inset=0.0)
    state = _state(position=Point2(x=0.30, y=0.0, frame=Frame.VEHICLE, domain=DOMAIN))
    collision, cutting = world_footprints(blade, state, world_from_vehicle(0.0, 2.0, 0.0, DOMAIN))
    expected = Polygon([[0.20, 1.96], [0.40, 1.96], [0.40, 2.04], [0.20, 2.04]])
    assert collision.symmetric_difference(expected).area == pytest.approx(0.0, abs=1e-12, rel=0)
    assert cutting.area == pytest.approx(collision.area, abs=1e-12, rel=0)


def test_world_footprints_require_t_wv_for_vehicle_state() -> None:
    with pytest.raises(SchemaValidationError):
        world_footprints(_blade(), _state())


def test_world_footprints_reject_double_counting_advance() -> None:
    """Un estado ya expresado en W no admite T_WV: sumaria el avance dos veces."""
    state = _state(position=Point2(x=0.0, y=2.0, frame=Frame.WORLD, domain=DOMAIN))
    with pytest.raises(SchemaValidationError):
        world_footprints(_blade(), state, world_from_vehicle(0.0, 2.0, 0.0, DOMAIN))


def test_world_footprints_accept_world_state_directly() -> None:
    blade = _blade(cutting_inset=0.0)
    state = _state(position=Point2(x=0.0, y=2.0, frame=Frame.WORLD, domain=DOMAIN))
    collision, _ = world_footprints(blade, state)
    assert collision.centroid.y == pytest.approx(2.0, abs=TOL, rel=0)
