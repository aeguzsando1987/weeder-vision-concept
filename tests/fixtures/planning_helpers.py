"""Utilidades de prueba A1/E2: configuraciones del repositorio, mapas a medida,
caminos rectilineos/curvos construidos a mano y peticiones del planificador.

Los caminos construidos aqui no pasan por el planificador: sirven para probar el
validador de forma independiente con oraculos analiticos.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from precision_weeder.core.config import (
    PlannerFile,
    SafetyFile,
    ToolConfig,
    ValidationFile,
    load_yaml_mapping,
    validate_config_payload,
)
from precision_weeder.core.trajectory import OrientationRequirement, PathSegment, TaskPath
from precision_weeder.core.types import (
    CircularExclusion,
    Corridor,
    Frame,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
)
from precision_weeder.core.units import Provenance, SpatialDomain
from precision_weeder.geometry.tool_profile import ToolGeometry, tool_geometry_from_config
from precision_weeder.mapping.local_map import LocalMap, rectangle_ring
from precision_weeder.planning import GeometricPlanner, PlanRequest, PlanResult
from precision_weeder.planning.smoothing import quintic_lane_change_segments
from precision_weeder.scenarios.planning_cases import PlanningCase

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = SpatialDomain.GROUND_METRIC_SYNTHETIC

PLANNER_FILE = validate_config_payload(
    PlannerFile, load_yaml_mapping(ROOT / "configs/planner.yaml"), "planner.yaml"
)
PLANNER = PLANNER_FILE.planner
RANDOM = PLANNER_FILE.random_scenarios
VALIDATION = validate_config_payload(
    ValidationFile, load_yaml_mapping(ROOT / "configs/validation.yaml"), "validation.yaml"
).validation
SAFETY = validate_config_payload(
    SafetyFile, load_yaml_mapping(ROOT / "configs/safety.yaml"), "safety.yaml"
).safety
TOOL = tool_geometry_from_config(ToolConfig())


def point(x: float, y: float, frame: Frame = Frame.WORLD, domain=DOMAIN) -> Point2:
    return Point2(x=x, y=y, frame=frame, domain=domain)


def zero_policy() -> UncertaintyPolicy:
    bound = UncertaintyBound(0.0, 0.0, 0.0, DOMAIN, Provenance.SYNTHETIC)
    return UncertaintyPolicy(bound=bound, budget=0.0)


def exclusion(plant_id: str, x: float, y: float, radius: float, segments: int = 64):
    return CircularExclusion(
        plant_id=plant_id,
        center=point(x, y),
        protection_radius=radius,
        uncertainty_margin=0.0,
        polygon_segments=segments,
    )


def custom_map(
    exclusions: Sequence[CircularExclusion] = (),
    corridor: tuple[float, float, float, float] = (-0.5, 0.5, 0.0, 5.0),
    unknown: Sequence[Sequence[Sequence[float]]] = (),
    map_id: str = "map_custom",
) -> LocalMap:
    x0, x1, y0, y1 = corridor
    return LocalMap(
        map_id=map_id,
        segment_id="W_test",
        domain=DOMAIN,
        corridor=Corridor(x_min=x0, x_max=x1, y_min=y0, y_max=y1),
        known_region=rectangle_ring(x0, x1, y0, y1),
        known_region_basis="test_ground_truth",
        exclusions=tuple(exclusions),
        uncertainty=zero_policy(),
        unknown_regions=tuple(unknown),
        free_space_certified=not exclusions,
    )


def linear_segments(points: Sequence[tuple[float, float]]) -> tuple[PathSegment, ...]:
    """Polilinea exacta (tramos lineales), con s proporcional a la longitud."""
    pts = np.asarray(points, dtype=float)
    lengths = np.hypot(*np.diff(pts, axis=0).T)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)]) / lengths.sum()
    cumulative[-1] = 1.0
    segments = []
    for i in range(len(pts) - 1):
        (xa, ya), (xb, yb) = pts[i], pts[i + 1]
        segments.append(
            PathSegment(
                s_start=float(cumulative[i]),
                s_end=float(cumulative[i + 1]),
                x_coefficients=[xa, xb - xa, 0, 0, 0, 0],
                y_coefficients=[ya, yb - ya, 0, 0, 0, 0],
            )
        )
    return tuple(segments)


def task_path(
    segments: Sequence[PathSegment],
    local_map: LocalMap,
    profile: ToolGeometry = TOOL,
    theta: float = 0.0,
    path_id: str = "test_path",
) -> TaskPath:
    return TaskPath(
        path_id=path_id,
        domain=local_map.domain,
        segments=tuple(segments),
        orientation=OrientationRequirement.fixed(theta),
        profile_id=profile.tool_id,
        profile_hash=profile.content_hash(),
        map_id=local_map.map_id,
        map_hash=local_map.content_hash(),
    )


def linear_path(points, local_map, profile=TOOL, theta=0.0) -> TaskPath:
    return task_path(linear_segments(points), local_map, profile, theta)


def smooth_path(points, local_map, profile=TOOL, theta=0.0) -> TaskPath:
    return task_path(quintic_lane_change_segments(points), local_map, profile, theta)


def request(
    local_map: LocalMap,
    start: tuple[float, float],
    goal: tuple[float, float],
    profile: ToolGeometry = TOOL,
    orientation: OrientationRequirement | None = None,
    request_id: str = "test_request",
) -> PlanRequest:
    return PlanRequest(
        request_id=request_id,
        map_id=local_map.map_id,
        map_hash=local_map.content_hash(),
        profile_id=profile.tool_id,
        profile_hash=profile.content_hash(),
        start=point(*start, domain=local_map.domain),
        goal=point(*goal, domain=local_map.domain),
        orientation=orientation or OrientationRequirement.fixed(0.0),
    )


def planner(planner_config=PLANNER, validation=VALIDATION, safety=SAFETY) -> GeometricPlanner:
    return GeometricPlanner(planner_config, validation, safety)


def run_case(case: PlanningCase, profile: ToolGeometry = TOOL, **kwargs) -> PlanResult:
    req = request(case.local_map, case.start_xy_m, case.goal_xy_m, profile)
    return planner(**kwargs).plan(req, case.local_map, profile)
