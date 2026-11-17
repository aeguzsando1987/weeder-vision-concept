"""Propiedades geometricas con oraculos independientes.

Se usan muestras pseudoaleatorias deterministas (`numpy.random.default_rng` con
semilla fija) en lugar de Hypothesis, que no es dependencia de E1; la cobertura
basada en propiedades se amplia en E2 junto al validador de barrido. Cada prueba
crea su propio generador, de modo que sus datos no dependen del orden de ejecucion.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon

from precision_weeder.core.types import Frame, exterior_circle_polygon
from precision_weeder.core.units import TRANSFORM_TOLERANCE, SpatialDomain
from precision_weeder.geometry.frames import Transform2D
from precision_weeder.scenarios.fixtures import default_tool_geometry

DOMAIN = SpatialDomain.GROUND_METRIC_SYNTHETIC
SAMPLES = 256
SEED = 20261024


def _random_transforms(rng: np.random.Generator, count: int) -> list[Transform2D]:
    translations = rng.uniform(-50.0, 50.0, size=(count, 2))
    angles = rng.uniform(-math.pi, math.pi, size=count)
    return [
        Transform2D(
            parent=Frame.WORLD,
            child=Frame.TOOL,
            tx=float(tx),
            ty=float(ty),
            theta=float(theta),
            domain=DOMAIN,
        )
        for (tx, ty), theta in zip(translations, angles, strict=True)
    ]


def test_round_trip_preserves_points_within_tolerance() -> None:
    rng = np.random.default_rng(SEED)
    points = rng.uniform(-10.0, 10.0, size=(32, 2))
    worst = 0.0
    for transform in _random_transforms(rng, SAMPLES):
        recovered = transform.inverse().apply_array(transform.apply_array(points))
        worst = max(worst, float(np.max(np.abs(recovered - points))))
    assert worst <= TRANSFORM_TOLERANCE


def test_rigid_transform_preserves_distances() -> None:
    """Oraculo: una transformacion rigida es una isometria."""
    rng = np.random.default_rng(SEED)
    points = rng.uniform(-5.0, 5.0, size=(16, 2))
    original = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    for transform in _random_transforms(rng, 64):
        moved = transform.apply_array(points)
        distances = np.linalg.norm(moved[:, None, :] - moved[None, :, :], axis=-1)
        assert np.max(np.abs(distances - original)) <= 1e-9


def test_rigid_transform_preserves_footprint_area_and_perimeter() -> None:
    rng = np.random.default_rng(SEED)
    blade = default_tool_geometry()
    reference = blade.collision_shape
    for transform in _random_transforms(rng, 64):
        placed = blade.collision_polygon_in(transform)
        assert placed.area == pytest.approx(reference.area, abs=1e-12, rel=0)
        assert placed.length == pytest.approx(reference.length, abs=1e-12, rel=0)
        assert placed.is_valid


def test_composition_is_associative() -> None:
    rng = np.random.default_rng(SEED)
    for _ in range(32):
        a = Transform2D(
            parent=Frame.WORLD,
            child=Frame.VEHICLE,
            tx=float(rng.uniform(-5, 5)),
            ty=float(rng.uniform(-5, 5)),
            theta=float(rng.uniform(-math.pi, math.pi)),
            domain=DOMAIN,
        )
        b = Transform2D(
            parent=Frame.VEHICLE,
            child=Frame.TOOL,
            tx=float(rng.uniform(-1, 1)),
            ty=float(rng.uniform(-1, 1)),
            theta=float(rng.uniform(-math.pi, math.pi)),
            domain=DOMAIN,
        )
        point = rng.uniform(-2.0, 2.0, size=(1, 2))
        chained = (a @ b).apply_array(point)
        stepwise = a.apply_array(b.apply_array(point))
        assert np.max(np.abs(chained - stepwise)) <= TRANSFORM_TOLERANCE


def test_exterior_circle_polygon_always_covers_its_circle() -> None:
    """Oraculo analitico: la apotema del poligono es exactamente el radio pedido."""
    rng = np.random.default_rng(SEED)
    for radius in rng.uniform(0.01, 5.0, size=64):
        for segments in (8, 16, 64, 256):
            polygon = Polygon(exterior_circle_polygon((0.0, 0.0), float(radius), segments))
            apothem = polygon.exterior.distance(ShapelyPoint(0.0, 0.0))
            circumscribed = float(radius) / math.cos(math.pi / segments)
            assert polygon.area > math.pi * radius**2
            assert np.max(np.linalg.norm(np.array(polygon.exterior.coords[:-1]), axis=1)) == (
                pytest.approx(circumscribed, rel=1e-12)
            )
            assert apothem == pytest.approx(float(radius), rel=1e-12)
            assert polygon.covers(ShapelyPoint(float(radius), 0.0))


def test_segment_circle_distance_oracle_matches_shapely() -> None:
    """Oraculo independiente: distancia analitica segmento-circulo vs. Shapely."""
    rng = np.random.default_rng(SEED)
    from shapely.geometry import LineString

    for _ in range(128):
        center = rng.uniform(-1.0, 1.0, size=2)
        radius = float(rng.uniform(0.05, 0.5))
        start = rng.uniform(-2.0, 2.0, size=2)
        end = rng.uniform(-2.0, 2.0, size=2)
        if np.allclose(start, end):
            continue
        segment = end - start
        t = np.clip(np.dot(center - start, segment) / np.dot(segment, segment), 0.0, 1.0)
        analytic = float(np.linalg.norm(center - (start + t * segment)) - radius)
        shapely_distance = LineString([start, end]).distance(
            Polygon(exterior_circle_polygon(center, radius, 4096))
        )
        if analytic <= 0.0:
            assert shapely_distance == pytest.approx(0.0, abs=1e-9, rel=0)
        else:
            assert shapely_distance == pytest.approx(analytic, abs=1e-5, rel=0)
