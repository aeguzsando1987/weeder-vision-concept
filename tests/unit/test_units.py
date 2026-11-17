"""Unidades, dominios espaciales y guardas numericas (E1)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from precision_weeder.core.errors import (
    InvalidDimensionError,
    NonFiniteValueError,
    UnitMismatchError,
)
from precision_weeder.core.units import (
    CLEARANCE_EPSILON_M,
    GEOMETRIC_APPROXIMATION_TOLERANCE_M,
    LIMIT_COMPARISON_TOLERANCE,
    TRANSFORM_TOLERANCE,
    LengthUnit,
    SpatialDomain,
    ensure_finite,
    ensure_finite_array,
    ensure_non_negative,
    ensure_positive,
    is_metric,
    require_same_domain,
    require_verified_metric,
    unit_of_domain,
)

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


@pytest.mark.parametrize("value", NON_FINITE)
def test_ensure_finite_rejects_nan_and_inf(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        ensure_finite(value, "valor")


@pytest.mark.parametrize("value", NON_FINITE)
def test_ensure_finite_array_rejects_non_finite(value: float) -> None:
    with pytest.raises(NonFiniteValueError):
        ensure_finite_array([1.0, value], "valores")


def test_ensure_finite_array_rejects_empty() -> None:
    with pytest.raises(NonFiniteValueError):
        ensure_finite_array([], "valores")


@pytest.mark.parametrize("value", [0.0, -1e-12, -3.0])
def test_ensure_positive_rejects_non_positive(value: float) -> None:
    with pytest.raises(InvalidDimensionError):
        ensure_positive(value, "dimension")


def test_ensure_non_negative_accepts_zero_rejects_negative() -> None:
    assert ensure_non_negative(0.0, "margen") == 0.0
    with pytest.raises(InvalidDimensionError):
        ensure_non_negative(-1e-9, "margen")


def test_ensure_finite_returns_float64_array() -> None:
    array = ensure_finite_array([[1, 2], [3, 4]], "puntos")
    assert array.dtype == np.float64
    assert array.shape == (2, 2)


def test_units_per_domain() -> None:
    assert unit_of_domain(SpatialDomain.IMAGE_PX) is LengthUnit.PIXEL
    assert unit_of_domain(SpatialDomain.GRAPHIC_UNSCALED) is LengthUnit.GRAPHIC_UNIT
    assert unit_of_domain(SpatialDomain.GROUND_METRIC_SYNTHETIC) is LengthUnit.METER
    assert unit_of_domain(SpatialDomain.GROUND_ASSUMED) is LengthUnit.METER


def test_ground_assumed_is_not_metric() -> None:
    """Metros supuestos no son escala verificada ni sintetica (§3.2)."""
    assert not is_metric(SpatialDomain.GROUND_ASSUMED)
    assert not is_metric(SpatialDomain.IMAGE_PX)
    assert is_metric(SpatialDomain.GROUND_METRIC_SYNTHETIC)
    assert is_metric(SpatialDomain.GROUND_METRIC_VERIFIED)


def test_require_same_domain_rejects_mixture() -> None:
    with pytest.raises(UnitMismatchError):
        require_same_domain(SpatialDomain.IMAGE_PX, SpatialDomain.GROUND_METRIC_VERIFIED)
    assert (
        require_same_domain(SpatialDomain.IMAGE_PX, SpatialDomain.IMAGE_PX)
        is SpatialDomain.IMAGE_PX
    )


@pytest.mark.parametrize(
    "domain",
    [
        SpatialDomain.IMAGE_PX,
        SpatialDomain.GROUND_ASSUMED,
        SpatialDomain.GROUND_METRIC_SYNTHETIC,
        SpatialDomain.GRAPHIC_UNSCALED,
    ],
)
def test_require_verified_metric_rejects_non_verified(domain: SpatialDomain) -> None:
    """El dominio sintetico es metrico por construccion, no por medicion."""
    with pytest.raises(UnitMismatchError):
        require_verified_metric(domain)


def test_tolerances_are_ordered_and_documented() -> None:
    assert TRANSFORM_TOLERANCE == 1e-9
    assert CLEARANCE_EPSILON_M == 1e-4
    assert GEOMETRIC_APPROXIMATION_TOLERANCE_M == 1e-5
    assert LIMIT_COMPARISON_TOLERANCE == 1e-8
    assert GEOMETRIC_APPROXIMATION_TOLERANCE_M < CLEARANCE_EPSILON_M
    assert all(math.isfinite(value) for value in (TRANSFORM_TOLERANCE, CLEARANCE_EPSILON_M))
