"""Marcos y transformaciones rigidas: identidad, traslacion, rotacion e ida/vuelta."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import Polygon

from precision_weeder.core.errors import (
    FrameMismatchError,
    MissingCalibrationError,
    NonFiniteValueError,
    SchemaValidationError,
    UnitMismatchError,
)
from precision_weeder.core.types import CalibrationReference, CalibrationStatus, Frame, Point2
from precision_weeder.core.units import TRANSFORM_TOLERANCE, SpatialDomain
from precision_weeder.geometry.frames import (
    ImageVAxisOnGround,
    Pose2,
    Transform2D,
    advance_sign_under_declared_mounting,
    metric_ground_displacement_from_image,
    tool_from_vehicle,
    world_from_vehicle,
)

DOMAIN = SpatialDomain.GROUND_METRIC_SYNTHETIC
TOL = TRANSFORM_TOLERANCE


def _tool_point(x: float, y: float) -> Point2:
    return Point2(x=x, y=y, frame=Frame.TOOL, domain=DOMAIN)


# --- casos conocidos ---------------------------------------------------------


def test_identity_preserves_points_exactly() -> None:
    identity = Transform2D.identity(Frame.WORLD, DOMAIN)
    point = Point2(x=0.37, y=-1.21, frame=Frame.WORLD, domain=DOMAIN)
    moved = identity.apply_point(point)
    assert abs(moved.x - point.x) <= TOL
    assert abs(moved.y - point.y) <= TOL


def test_known_translation() -> None:
    transform = Transform2D.from_translation(Frame.VEHICLE, Frame.TOOL, 0.25, -0.1, DOMAIN)
    moved = transform.apply_point(_tool_point(0.1, 0.2))
    assert moved.x == pytest.approx(0.35, abs=TOL, rel=0)
    assert moved.y == pytest.approx(0.1, abs=TOL, rel=0)
    assert moved.frame is Frame.VEHICLE


@pytest.mark.parametrize(
    ("angle", "expected"),
    [
        (math.pi / 2.0, (0.0, 1.0)),
        (math.pi, (-1.0, 0.0)),
        (-math.pi / 2.0, (0.0, -1.0)),
    ],
)
def test_known_rotation(angle: float, expected: tuple[float, float]) -> None:
    transform = Transform2D.from_rotation(Frame.VEHICLE, Frame.TOOL, angle, DOMAIN)
    moved = transform.apply_point(_tool_point(1.0, 0.0))
    assert moved.x == pytest.approx(expected[0], abs=TOL, rel=0)
    assert moved.y == pytest.approx(expected[1], abs=TOL, rel=0)


def test_rotation_then_translation_order_is_explicit() -> None:
    """`T` rota y despues traslada: `p_parent = R(theta) p_child + t`."""
    transform = Transform2D(
        parent=Frame.WORLD, child=Frame.TOOL, tx=1.0, ty=2.0, theta=math.pi / 2.0, domain=DOMAIN
    )
    moved = transform.apply_point(_tool_point(1.0, 0.0))
    assert (moved.x, moved.y) == pytest.approx((1.0, 3.0), abs=TOL, rel=0)


# --- ida y vuelta ------------------------------------------------------------


def test_inverse_round_trip_preserves_points() -> None:
    transform = Transform2D(
        parent=Frame.WORLD, child=Frame.TOOL, tx=-3.14, ty=2.71, theta=0.9273, domain=DOMAIN
    )
    original = _tool_point(0.11, -0.47)
    recovered = transform.inverse().apply_point(transform.apply_point(original))
    assert abs(recovered.x - original.x) <= TOL
    assert abs(recovered.y - original.y) <= TOL
    assert recovered.frame is Frame.TOOL


def test_inverse_round_trip_preserves_footprint() -> None:
    """La huella vuelve a su lugar: area, perimetro y vertices se conservan."""
    footprint = np.array([[-0.10, -0.04], [0.10, -0.04], [0.10, 0.04], [-0.10, 0.04]])
    transform = Transform2D(
        parent=Frame.WORLD, child=Frame.TOOL, tx=0.5, ty=4.2, theta=-1.2, domain=DOMAIN
    )
    placed = transform.apply_array(footprint)
    recovered = transform.inverse().apply_array(placed)
    assert np.max(np.abs(recovered - footprint)) <= TOL
    assert Polygon(placed).area == pytest.approx(Polygon(footprint).area, abs=TOL, rel=0)
    assert Polygon(placed).length == pytest.approx(Polygon(footprint).length, abs=TOL, rel=0)


def test_compose_with_inverse_is_identity() -> None:
    transform = world_from_vehicle(1.5, -0.25, 0.3, DOMAIN)
    composed = transform.compose(transform.inverse())
    assert composed.allclose(Transform2D.identity(Frame.WORLD, DOMAIN))


def test_geometry_round_trip_matches_array_round_trip() -> None:
    transform = Transform2D(
        parent=Frame.WORLD, child=Frame.TOOL, tx=0.3, ty=1.7, theta=0.45, domain=DOMAIN
    )
    polygon = Polygon([[-0.1, -0.04], [0.1, -0.04], [0.1, 0.04], [-0.1, 0.04]])
    via_geometry = np.array(transform.apply_geometry(polygon).exterior.coords[:-1])
    via_array = transform.apply_array(np.array(polygon.exterior.coords[:-1]))
    assert np.max(np.abs(via_geometry - via_array)) <= TOL


# --- cadena W <- V <- B ------------------------------------------------------


def test_chain_world_vehicle_tool_adds_advance_once() -> None:
    """`T_WB = T_WV @ T_VB`: el avance del tractor entra una sola vez (§3.2)."""
    t_wv = world_from_vehicle(x=0.0, y=2.0, heading=0.0, domain=DOMAIN)
    t_vb = tool_from_vehicle(x=0.3, y=0.1, theta=0.0, domain=DOMAIN)
    t_wb = t_wv @ t_vb
    assert t_wb.parent is Frame.WORLD
    assert t_wb.child is Frame.TOOL
    assert (t_wb.tx, t_wb.ty) == pytest.approx((0.3, 2.1), abs=TOL, rel=0)


def test_compose_rejects_broken_chain() -> None:
    t_wv = world_from_vehicle(0.0, 1.0, 0.0, DOMAIN)
    with pytest.raises(FrameMismatchError):
        t_wv.compose(t_wv)


def test_compose_rejects_mixed_domains() -> None:
    t_wv = world_from_vehicle(0.0, 1.0, 0.0, DOMAIN)
    t_vb = tool_from_vehicle(0.0, 0.0, 0.0, SpatialDomain.IMAGE_PX)
    with pytest.raises(UnitMismatchError):
        t_wv.compose(t_vb)


def test_apply_point_rejects_wrong_frame_and_domain() -> None:
    transform = tool_from_vehicle(0.1, 0.0, 0.0, DOMAIN)
    with pytest.raises(FrameMismatchError):
        transform.apply_point(Point2(x=0.0, y=0.0, frame=Frame.WORLD, domain=DOMAIN))
    with pytest.raises(UnitMismatchError):
        transform.apply_point(Point2(x=0.0, y=0.0, frame=Frame.TOOL, domain=SpatialDomain.IMAGE_PX))


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_transform_rejects_non_finite(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        Transform2D(parent=Frame.WORLD, child=Frame.TOOL, tx=value, ty=0.0, domain=DOMAIN)
    with pytest.raises(NonFiniteValueError):
        Transform2D(parent=Frame.WORLD, child=Frame.TOOL, theta=value, domain=DOMAIN)


def test_pose_rejects_identical_parent_and_child() -> None:
    with pytest.raises(FrameMismatchError):
        Pose2(x=0.0, y=0.0, theta=0.0, parent=Frame.WORLD, child=Frame.WORLD, domain=DOMAIN)


def test_pose_as_transform_matches_fields() -> None:
    pose = Pose2(x=1.0, y=2.0, theta=0.5, parent=Frame.WORLD, child=Frame.VEHICLE, domain=DOMAIN)
    transform = pose.as_transform()
    assert transform.to_dict()["name"] == "T_WV"
    assert (transform.tx, transform.ty, transform.theta) == (1.0, 2.0, 0.5)


def test_transform_matrix_is_rigid() -> None:
    """Solo rotacion y traslacion: determinante 1 y columnas ortonormales."""
    transform = Transform2D(
        parent=Frame.WORLD, child=Frame.TOOL, tx=0.1, ty=0.2, theta=1.1, domain=DOMAIN
    )
    rotation = transform.rotation_matrix()
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=TOL, rel=0)
    assert np.max(np.abs(rotation @ rotation.T - np.eye(2))) <= TOL


# --- movimiento aparente en imagen ------------------------------------------


def test_upward_flow_means_forward_advance_only_with_declared_forward_mounting() -> None:
    """`v` crece hacia abajo; con `+v` hacia `+y_W`, plantas subiendo => avance +y_W."""
    forward = ImageVAxisOnGround.FORWARD
    assert advance_sign_under_declared_mounting(-12.0, forward) == 1
    assert advance_sign_under_declared_mounting(+12.0, forward) == -1
    assert advance_sign_under_declared_mounting(0.0, forward) == 0


def test_opposite_mounting_reverses_the_sign() -> None:
    """Una camara girada 180 grados invierte la correspondencia; no se asume ninguna (R7)."""
    backward = ImageVAxisOnGround.BACKWARD
    assert advance_sign_under_declared_mounting(-12.0, backward) == -1
    assert advance_sign_under_declared_mounting(+12.0, backward) == 1
    for delta in (-3.0, 5.0):
        assert advance_sign_under_declared_mounting(
            delta, ImageVAxisOnGround.FORWARD
        ) == -advance_sign_under_declared_mounting(delta, ImageVAxisOnGround.BACKWARD)


@pytest.mark.parametrize("delta", [-12.0, 0.0, 12.0])
def test_unknown_mounting_never_yields_a_physical_direction(delta: float) -> None:
    with pytest.raises(MissingCalibrationError, match="montaje"):
        advance_sign_under_declared_mounting(delta, ImageVAxisOnGround.UNKNOWN)


def test_mounting_must_be_a_declared_value() -> None:
    with pytest.raises(ValueError):
        advance_sign_under_declared_mounting(-1.0, "lateral")  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_apparent_motion_rejects_non_finite(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        advance_sign_under_declared_mounting(value, ImageVAxisOnGround.FORWARD)


def test_metric_displacement_requires_verified_calibration() -> None:
    """Una vista sin escala verificada no produce desplazamiento metrico."""
    absent = CalibrationReference(calibration_id="cam0")
    with pytest.raises(MissingCalibrationError):
        metric_ground_displacement_from_image((0.0, -12.0), absent)


def test_metric_displacement_is_not_implemented_before_e4() -> None:
    verified = CalibrationReference(
        calibration_id="cam0",
        status=CalibrationStatus.VERIFIED,
        scale_evidence="patron metrico independiente",
        domain=SpatialDomain.GROUND_METRIC_VERIFIED,
    )
    with pytest.raises(NotImplementedError):
        metric_ground_displacement_from_image((0.0, -12.0), verified)


def test_metric_displacement_rejects_bad_shape() -> None:
    verified = CalibrationReference(
        calibration_id="cam0",
        status=CalibrationStatus.VERIFIED,
        scale_evidence="patron metrico independiente",
        domain=SpatialDomain.GROUND_METRIC_VERIFIED,
    )
    with pytest.raises(SchemaValidationError):
        metric_ground_displacement_from_image((0.0, -12.0, 3.0), verified)


def test_module_documents_axis_conventions() -> None:
    """Las convenciones de ejes estan escritas, no implicitas."""
    source = Path(__file__).resolve().parents[2] / "src/precision_weeder/geometry/frames.py"
    text = source.read_text(encoding="utf-8")
    assert "hacia **abajo**" in text
    assert "hacia **delante**" in text
