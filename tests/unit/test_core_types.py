"""Contratos de datos de `core.types`: validacion estricta y rechazo de entradas invalidas."""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from precision_weeder.core.errors import (
    InvalidDimensionError,
    InvalidPolygonError,
    MissingCalibrationError,
    NonFiniteValueError,
    SchemaValidationError,
    UncertaintyBudgetError,
    UnitMismatchError,
)
from precision_weeder.core.types import (
    CalibrationReference,
    CalibrationStatus,
    CircularExclusion,
    Corridor,
    Frame,
    PlantReference,
    Point2,
    UncertaintyBound,
    covariance_to_bound,
    exterior_circle_polygon,
    validate_polygon_points,
)
from precision_weeder.core.units import Provenance, SpatialDomain

DOMAIN = SpatialDomain.GROUND_METRIC_SYNTHETIC
NON_FINITE = [float("nan"), float("inf"), float("-inf")]


def _point(x: float = 0.0, y: float = 0.0) -> Point2:
    return Point2(x=x, y=y, frame=Frame.WORLD, domain=DOMAIN)


# --- Point2 -----------------------------------------------------------------


@pytest.mark.parametrize("value", NON_FINITE)
def test_point_rejects_non_finite(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        Point2(x=value, y=0.0, frame=Frame.WORLD, domain=DOMAIN)
    with pytest.raises(NonFiniteValueError):
        Point2(x=0.0, y=value, frame=Frame.WORLD, domain=DOMAIN)


def test_point_is_frozen_and_serializable() -> None:
    point = _point(1.5, -2.0)
    with pytest.raises(AttributeError):
        point.x = 3.0  # type: ignore[misc]
    payload = point.to_dict()
    assert payload["unit"] == "m"
    assert payload["frame"] == "W"
    assert payload["domain"] == DOMAIN.value


# --- poligonos ---------------------------------------------------------------


def test_validate_polygon_rejects_self_intersection() -> None:
    bowtie = [[0.0, 0.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(InvalidPolygonError):
        validate_polygon_points(bowtie, "bowtie")


def test_validate_polygon_rejects_degenerate_and_short_rings() -> None:
    with pytest.raises(InvalidPolygonError):
        validate_polygon_points([[0.0, 0.0], [1.0, 0.0]], "segmento")
    with pytest.raises(InvalidPolygonError):
        validate_polygon_points([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], "colineal")


@pytest.mark.parametrize("value", NON_FINITE)
def test_validate_polygon_rejects_non_finite(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        validate_polygon_points([[0.0, 0.0], [1.0, value], [1.0, 1.0]], "poligono")


def test_validate_polygon_accepts_square() -> None:
    square = validate_polygon_points([[0, 0], [2, 0], [2, 1], [0, 1]], "cuadrado")
    assert square.shape == (4, 2)
    assert Polygon(square).area == pytest.approx(2.0)


# --- aproximacion exterior de circulos --------------------------------------


def test_exterior_circle_polygon_contains_the_true_circle() -> None:
    """Oraculo independiente: todo punto del circulo exacto cae dentro del poligono."""
    radius, segments = 0.18, 64
    polygon = Polygon(exterior_circle_polygon((0.3, -0.2), radius, segments))
    angles = np.linspace(0.0, 2.0 * math.pi, 997, endpoint=False)
    for angle in angles:
        boundary = ShapelyPoint(0.3 + radius * math.cos(angle), -0.2 + radius * math.sin(angle))
        assert polygon.covers(boundary)
    assert polygon.area > math.pi * radius**2


def test_exterior_circle_polygon_error_stays_within_budget() -> None:
    """El exceso de area por discretizacion es pequeno y acotado, nunca negativo."""
    radius, segments = 0.18, 64
    polygon = Polygon(exterior_circle_polygon((0.0, 0.0), radius, segments))
    exact = math.pi * radius**2
    assert 0.0 < polygon.area - exact < 0.002 * exact


def test_exterior_circle_polygon_rejects_bad_arguments() -> None:
    with pytest.raises(InvalidDimensionError):
        exterior_circle_polygon((0.0, 0.0), 0.0, 64)
    with pytest.raises(InvalidPolygonError):
        exterior_circle_polygon((0.0, 0.0), 1.0, 4)
    with pytest.raises(NonFiniteValueError):
        exterior_circle_polygon((float("nan"), 0.0), 1.0, 64)


# --- incertidumbre -----------------------------------------------------------


def test_uncertainty_bound_sums_conservatively() -> None:
    bound = UncertaintyBound(localization=0.01, calibration=0.01, tracking=0.01, domain=DOMAIN)
    assert bound.total == pytest.approx(0.03)
    assert bound.within_budget(0.03)
    assert not bound.within_budget(0.029)


def test_uncertainty_bound_rejects_negative_components() -> None:
    with pytest.raises(InvalidDimensionError):
        UncertaintyBound(localization=-0.01, calibration=0.0, tracking=0.0, domain=DOMAIN)


def test_covariance_to_bound_matches_analytic_oracle() -> None:
    """Oraculo analitico: para diag(a, b) la cota es k*sqrt(max(a, b))."""
    assert covariance_to_bound([[4.0, 0.0], [0.0, 9.0]], k=2.0) == pytest.approx(6.0)
    assert covariance_to_bound([[0.0, 0.0], [0.0, 0.0]], k=3.0) == pytest.approx(0.0)


def test_covariance_to_bound_rejects_invalid_matrices() -> None:
    with pytest.raises(UncertaintyBudgetError):
        covariance_to_bound([[1.0, 0.5], [-0.5, 1.0]], k=1.0)  # no simetrica
    with pytest.raises(UncertaintyBudgetError):
        covariance_to_bound([[1.0, 2.0], [2.0, 1.0]], k=1.0)  # no semidefinida positiva
    with pytest.raises(UncertaintyBudgetError):
        covariance_to_bound([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], k=1.0)


# --- calibracion -------------------------------------------------------------


def test_calibration_cannot_be_verified_without_evidence() -> None:
    """No se admite calibracion verificada por simple declaracion (§5 E1)."""
    with pytest.raises(MissingCalibrationError):
        CalibrationReference(calibration_id="cam0", status=CalibrationStatus.VERIFIED)


def test_absent_calibration_refuses_metric_requests() -> None:
    calibration = CalibrationReference(calibration_id="cam0")
    assert not calibration.is_verified
    with pytest.raises(MissingCalibrationError):
        calibration.require_verified()


def test_verified_calibration_with_evidence_is_accepted() -> None:
    calibration = CalibrationReference(
        calibration_id="cam0",
        status=CalibrationStatus.VERIFIED,
        scale_evidence="patron de 1.000 m medido con cinta calibrada",
        independent_point_error=0.004,
        domain=SpatialDomain.GROUND_METRIC_VERIFIED,
    )
    calibration.require_verified()
    assert calibration.to_dict()["status"] == "verified"


# --- plantas y exclusiones ---------------------------------------------------


def test_plant_rejects_protection_smaller_than_stem() -> None:
    with pytest.raises(SchemaValidationError):
        PlantReference(plant_id="p0", position=_point(), stem_radius=0.10, protection_radius=0.05)


def test_plant_rejects_non_positive_radius() -> None:
    with pytest.raises(InvalidDimensionError):
        PlantReference(plant_id="p0", position=_point(), stem_radius=0.0, protection_radius=0.15)


def test_plant_rejects_empty_id() -> None:
    with pytest.raises(SchemaValidationError):
        PlantReference(plant_id="  ", position=_point(), stem_radius=0.02, protection_radius=0.15)


def test_exclusion_total_radius_is_minkowski_sum_of_radii() -> None:
    plant = PlantReference(
        plant_id="p0", position=_point(0.0, 1.0), stem_radius=0.02, protection_radius=0.15
    )
    exclusion = CircularExclusion.from_plant(plant, uncertainty_margin=0.03)
    assert exclusion.total_radius == pytest.approx(0.18)
    assert exclusion.provenance is Provenance.SYNTHETIC
    # Oraculo: por construccion exterior, la apotema del poligono vale exactamente el
    # radio total, de modo que el circulo exacto queda inscrito y nunca se subestima.
    polygon = exclusion.as_polygon()
    apothem = polygon.exterior.distance(ShapelyPoint(0.0, 1.0))
    assert apothem == pytest.approx(0.18, abs=1e-12, rel=0)
    assert polygon.covers(ShapelyPoint(0.18, 1.0))


def test_exclusion_rejects_invalid_arguments() -> None:
    plant = PlantReference(
        plant_id="p0", position=_point(), stem_radius=0.02, protection_radius=0.15
    )
    with pytest.raises(InvalidDimensionError):
        CircularExclusion.from_plant(plant, uncertainty_margin=-0.01)
    with pytest.raises(InvalidPolygonError):
        CircularExclusion.from_plant(plant, uncertainty_margin=0.03, polygon_segments=3)


# --- corredor ----------------------------------------------------------------


def test_corridor_rejects_inverted_or_degenerate_bounds() -> None:
    with pytest.raises(SchemaValidationError):
        Corridor(x_min=0.5, x_max=-0.5, y_min=0.0, y_max=5.0)
    with pytest.raises(SchemaValidationError):
        Corridor(x_min=0.0, x_max=0.0, y_min=0.0, y_max=5.0)


def test_corridor_centered_dimensions() -> None:
    corridor = Corridor.centered(width=1.0, length=5.0)
    assert (corridor.width, corridor.length) == (1.0, 5.0)
    assert corridor.center_x == pytest.approx(0.0)
    assert corridor.as_polygon().area == pytest.approx(5.0)


def test_corridor_rejects_point_from_other_domain() -> None:
    corridor = Corridor.centered(width=1.0, length=5.0)
    foreign = Point2(x=0.0, y=1.0, frame=Frame.WORLD, domain=SpatialDomain.IMAGE_PX)
    with pytest.raises(UnitMismatchError):
        corridor.contains_point(foreign)


def test_corridor_contains_boundary_point() -> None:
    corridor = Corridor.centered(width=1.0, length=5.0)
    assert corridor.contains_point(_point(0.5, 0.0))
    assert not corridor.contains_point(_point(0.5 + 1e-9, 0.0))
