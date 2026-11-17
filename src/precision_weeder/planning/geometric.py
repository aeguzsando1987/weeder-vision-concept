"""Planificador geometrico E2 en suelo `W` (orden A1/E2 §4; plan maestro §3.4).

Algoritmo base, determinista y no completo:

1. **Admisibilidad:** la huella en inicio y meta debe tener despeje, estar en el
   dominio conocido y no pisar suelo desconocido.
2. **Obstaculos de configuracion (solo para proponer):** para huella convexa `K`
   y exclusion convexa `E`, `E ⊕ (-K)` es la envolvente convexa de las diferencias
   de vertices. El suelo desconocido se aproxima por su envolvente convexa
   (conservador). Todo se ensancha con `generation_margin_m`.
3. **Niveles longitudinales:** inicio, meta y, por obstaculo, sus bordes en `y`
   ensanchados por `level_padding_m` y su centro; ademas, el punto medio de cada
   franja de la descomposicion vertical (entre `y` criticas consecutivas de todos
   los obstaculos), donde el conjunto de obstaculos de la fila es constante.
4. **Candidatos laterales:** en cada nivel, intervalos libres del corredor
   erosionado; se muestrean `lateral_candidates_per_interval` puntos por intervalo.
   El conjunto global de `x` se prueba en todos los niveles; se admite quedarse del
   mismo lado o alternar.
5. **Aristas hacia delante** entre niveles consecutivos. Una arista se admite si la
   huella barrida sobre el **rectangulo** que contiene su tramo suavizado (ver
   `smoothing`) tiene despeje y holgura de generacion: es una comprobacion
   conservadora con la geometria real, no con el disco circunscrito.
6. **Busqueda** de los `max_candidates` mejores caminos por programacion dinamica
   sobre el grafo aciclico, con costo normalizado (longitud, desviacion respecto de
   la recta inicio-meta y suavidad) y desempate estable por coordenadas.
7. **Suavizado C2 y validacion independiente** de cada candidato en orden; el primero
   que el validador acepta se devuelve. La seguridad nunca es un termino de costo.

`NO_PATH_FOUND` significa que este algoritmo no encontro ruta, no que no exista.
No hay ramas por nombre de escena, clip, resolucion o coordenadas concretas.
Este modulo no importa simulador, cinematica, actuadores ni video.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry

from ..core.config import PlannerConfig, SafetyConfig, ValidationConfig
from ..core.errors import SchemaValidationError, UnitMismatchError
from ..core.identity import content_hash, thaw
from ..core.trajectory import E2_SUPPORTED_ROLES, OrientationPolicy, TaskPath
from ..core.types import AbstentionReason, PlanStatus
from ..core.units import require_same_domain
from ..geometry.tool_profile import ToolGeometry
from ..mapping.local_map import LocalMap
from ..validation.swept_volume import (
    ValidationStatus,
    require_discretization_budget,
    validate_task_path,
)
from .base import CandidateRecord, PlanRequest, PlanResult
from .smoothing import quintic_lane_change_segments

__all__ = ["GeometricPlanner", "plan_path"]

_LEVEL_MERGE_M = 1e-9


@dataclass(frozen=True)
class _Problem:
    tool: np.ndarray
    allowed: BaseGeometry
    allowed_boundary: BaseGeometry
    exclusions: BaseGeometry
    unknown: BaseGeometry
    epsilon: float
    margin: float
    x_bounds: tuple[float, float]


def _effective_tool(profile: ToolGeometry, theta: float, tool_inflation: float) -> np.ndarray:
    shape = affinity.rotate(
        profile.inflated_collision_shape(tool_inflation), theta, origin=(0.0, 0.0), use_radians=True
    )
    return np.asarray(shape.exterior.coords[:-1], dtype=np.float64)


def _region_ok(problem: _Problem, region: Polygon, margin: float) -> str | None:
    """`None` si `region` (huella o barrido) tiene despeje y holgura; si no, la causa."""
    if not problem.allowed.covers(region):
        if region.intersection(problem.unknown).area > 0.0:
            return "unknown_space"
        return "out_of_domain"
    if margin > 0.0 and region.distance(problem.allowed_boundary) < margin:
        return "domain_margin"
    if not problem.exclusions.is_empty and (
        region.distance(problem.exclusions) <= problem.epsilon + margin
    ):
        return "collision"
    return None


def _box_sweep(tool: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> Polygon:
    corners = np.array([[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]])
    points = (tool[None, :, :] + corners[:, None, :]).reshape(-1, 2)
    return MultiPoint(points).convex_hull


def _config_obstacles(problem: _Problem, local_map: LocalMap) -> list[Polygon]:
    """Obstaculos de configuracion `O ⊕ (-K)`, ensanchados por la holgura de generacion."""
    negative_tool = -problem.tool
    sources = list(local_map.exclusion_polygons())
    unknown = problem.unknown
    if not unknown.is_empty:
        parts = getattr(unknown, "geoms", [unknown])
        sources.extend(part.convex_hull for part in parts if part.area > 0.0)
    obstacles = []
    for source in sources:
        hull_points = np.asarray(source.convex_hull.exterior.coords[:-1])
        points = (hull_points[:, None, :] + negative_tool[None, :, :]).reshape(-1, 2)
        obstacles.append(MultiPoint(points).convex_hull.buffer(problem.epsilon + problem.margin))
    return obstacles


def _free_intervals(
    obstacles: list[Polygon], y: float, x_bounds: tuple[float, float]
) -> list[tuple[float, float]]:
    low, high = x_bounds
    blocked: list[tuple[float, float]] = []
    for obstacle in obstacles:
        min_x, min_y, max_x, max_y = obstacle.bounds
        if not min_y <= y <= max_y:
            continue
        line = Polygon(
            [
                (min_x - 1.0, y - 1e-12),
                (max_x + 1.0, y - 1e-12),
                (max_x + 1.0, y + 1e-12),
                (min_x - 1.0, y + 1e-12),
            ]
        )
        cut = obstacle.intersection(line)
        if not cut.is_empty:
            blocked.append((cut.bounds[0], cut.bounds[2]))
    blocked.sort()
    intervals = []
    cursor = low
    for start, end in blocked:
        if start > cursor:
            intervals.append((cursor, min(start, high)))
        cursor = max(cursor, end)
        if cursor >= high:
            break
    if cursor < high:
        intervals.append((cursor, high))
    return [(a, b) for a, b in intervals if b > a]


class GeometricPlanner:
    """Planificador E2. `plan` devuelve un `PlanResult`; nunca acepta sin el validador."""

    def __init__(
        self,
        planner: PlannerConfig,
        validation: ValidationConfig,
        safety: SafetyConfig,
    ) -> None:
        if not isinstance(planner, PlannerConfig) or not isinstance(validation, ValidationConfig):
            raise SchemaValidationError("GeometricPlanner: configuraciones invalidas")
        if not isinstance(safety, SafetyConfig):
            raise SchemaValidationError("GeometricPlanner: se espera SafetyConfig")
        self.planner = planner
        self.validation = validation
        self.safety = safety
        self.config_hash = content_hash(
            {
                "planner": planner.model_dump(mode="json"),
                "validation": validation.model_dump(mode="json"),
                "safety": safety.model_dump(mode="json"),
            }
        )

    # --- API ------------------------------------------------------------------

    def plan(self, request: PlanRequest, local_map: LocalMap, profile: ToolGeometry) -> PlanResult:
        """Planifica una ruta geometrica.

        Raises:
            SchemaValidationError / UnitMismatchError: entrada invalida (marcos,
                dominios, identidades). Una peticion valida sin ruta devuelve
                `PlanResult` con razon, no excepcion.
            ConfigError: alguna exclusion efectiva excede el presupuesto de exceso de
                discretizacion (`safety.max_conservative_discretization_excess_m`):
                entrada incompatible, comprobada antes de buscar o exportar.
        """
        started = time.perf_counter()
        self._check_inputs(request, local_map, profile)
        require_discretization_budget(local_map, self.safety)
        base = {
            "request_hash": request.content_hash(),
            "map_hash": local_map.content_hash(),
            "profile_hash": profile.content_hash(),
            "config_hash": self.config_hash,
        }
        unsupported = []
        if request.orientation.policy is not OrientationPolicy.FIXED:
            unsupported.append(
                f"orientacion {request.orientation.policy.value} aun no implementada; solo FIXED"
            )
        if not profile.is_convex():
            unsupported.append("huella de colision no convexa")
        if not profile.is_cut_convex():
            unsupported.append("huella de corte no convexa (su cobertura no es la envolvente)")
        if request.role not in E2_SUPPORTED_ROLES:
            unsupported.append(
                f"rol {request.role!r} no soportado; E2 solo planifica {sorted(E2_SUPPORTED_ROLES)}"
            )
        if unsupported:
            return PlanResult(
                status=PlanStatus.UNSUPPORTED,
                reason=AbstentionReason.UNSUPPORTED_FEATURE,
                detail="; ".join(unsupported) + ". No es una imposibilidad geometrica.",
                **base,
            )
        theta = float(request.orientation.theta_rad)  # type: ignore[arg-type]
        tool = _effective_tool(profile, theta, local_map.tool_inflation)
        allowed = local_map.allowed_region()
        corridor = local_map.corridor
        problem = _Problem(
            tool=tool,
            allowed=allowed,
            allowed_boundary=allowed.boundary,
            exclusions=local_map.exclusion_union(),
            unknown=local_map.unknown_union(),
            epsilon=self.safety.clearance_epsilon_m,
            margin=self.planner.generation_margin_m,
            x_bounds=(
                corridor.x_min - float(tool[:, 0].min()) + self.planner.generation_margin_m,
                corridor.x_max - float(tool[:, 0].max()) - self.planner.generation_margin_m,
            ),
        )
        start = (request.start.x, request.start.y)
        goal = (request.goal.x, request.goal.y)
        for name, point, reason in (
            ("inicio", start, AbstentionReason.START_INADMISSIBLE),
            ("meta", goal, AbstentionReason.GOAL_INADMISSIBLE),
        ):
            cause = _region_ok(problem, Polygon(tool + np.asarray(point)), 0.0)
            if cause is not None:
                return PlanResult(
                    status=PlanStatus.ABSTAIN,
                    reason=reason,
                    detail=f"huella en el {name} {point}: {cause}",
                    search={"admissibility": {name: cause}},
                    timing={"wall_s": time.perf_counter() - started},
                    **base,
                )

        candidates, search = self._search(problem, local_map, start, goal, include_unknown=True)
        rejected: list[CandidateRecord] = []
        uncertified = False
        for rank, (cost, waypoints) in enumerate(candidates):
            path = self._task_path(request, local_map, profile, waypoints, rank)
            report = validate_task_path(
                path,
                local_map,
                profile,
                self.safety,
                self.validation,
                expected_start=start,
                expected_goal=goal,
            )
            summary = {
                "validation_status": report.status.value,
                "violations": thaw(report.violations[:5]),
                "uncertified_intervals": len(report.uncertified_intervals),
                "path_hash": report.path_hash,
            }
            if report.status is ValidationStatus.ACCEPTED:
                return PlanResult(
                    status=PlanStatus.ACCEPTED,
                    reason=None,
                    detail="ruta geometrica certificada por el validador independiente",
                    path=path,
                    validation=report,
                    chosen=CandidateRecord(rank, waypoints, cost, "accepted", summary),
                    rejected=tuple(rejected),
                    search=search,
                    timing={"wall_s": time.perf_counter() - started},
                    **base,
                )
            uncertified = uncertified or report.status is ValidationStatus.UNCERTIFIED
            rejected.append(
                CandidateRecord(rank, waypoints, cost, f"validator_{report.status.value}", summary)
            )

        if rejected:
            status = PlanStatus.UNCERTIFIED if uncertified else PlanStatus.ABSTAIN
            reason = (
                AbstentionReason.CERTIFICATION_BUDGET_EXHAUSTED
                if uncertified
                else AbstentionReason.COLLISION
            )
            detail = (
                f"{len(rejected)} candidatos suavizados no certificados o rechazados por el "
                "validador; no se acepta ninguna ruta"
            )
        else:
            status = PlanStatus.ABSTAIN
            reason = AbstentionReason.NO_PATH_FOUND
            detail = "el algoritmo no encontro candidatos; no es prueba de imposibilidad"
            if not problem.unknown.is_empty:
                relaxed, _ = self._search(problem, local_map, start, goal, include_unknown=False)
                if relaxed:
                    reason = AbstentionReason.UNKNOWN_SPACE
                    detail = (
                        "solo habria candidatos atravesando suelo desconocido, que se trata "
                        "como prohibido"
                    )
            if search.get("budget_exhausted"):
                detail += f" (presupuesto de busqueda agotado: {search['edges_evaluated']} aristas)"
        return PlanResult(
            status=status,
            reason=reason,
            detail=detail,
            rejected=tuple(rejected),
            search=search,
            timing={"wall_s": time.perf_counter() - started},
            **base,
        )

    # --- internos -------------------------------------------------------------

    def _check_inputs(
        self, request: PlanRequest, local_map: LocalMap, profile: ToolGeometry
    ) -> None:
        if not isinstance(request, PlanRequest):
            raise SchemaValidationError("plan: se espera PlanRequest")
        if not isinstance(local_map, LocalMap):
            raise SchemaValidationError("plan: se espera LocalMap")
        if not isinstance(profile, ToolGeometry):
            raise SchemaValidationError("plan: se espera ToolGeometry (perfil)")
        require_same_domain(request.domain, local_map.domain, profile.domain, context="plan")
        if request.map_hash != local_map.content_hash() or request.map_id != local_map.map_id:
            raise SchemaValidationError("plan: la peticion referencia otro mapa")
        if request.profile_hash != profile.content_hash() or request.profile_id != profile.tool_id:
            raise SchemaValidationError("plan: la peticion referencia otro perfil")
        if local_map.domain is not profile.domain:
            raise UnitMismatchError("plan: mapa y perfil en dominios distintos")

    def _levels(
        self, obstacles: list[Polygon], start: tuple[float, float], goal: tuple[float, float]
    ) -> list[float]:
        pad = self.planner.level_padding_m
        raw = {start[1], goal[1]}
        critical = set()
        for obstacle in obstacles:
            _, min_y, _, max_y = obstacle.bounds
            raw.update((min_y - pad, 0.5 * (min_y + max_y), max_y + pad))
            critical.update((min_y, max_y))
        # Descomposicion vertical: en cada franja entre `y` criticas consecutivas el
        # conjunto de obstaculos que corta la fila es constante; su punto medio
        # representa la franja (p. ej. el paso entre dos obstaculos solapados en `y`).
        ordered = sorted(critical)
        raw.update(0.5 * (a + b) for a, b in itertools.pairwise(ordered))
        inside = sorted(y for y in raw if start[1] <= y <= goal[1])
        levels: list[float] = []
        for y in inside:
            if not levels or y - levels[-1] > _LEVEL_MERGE_M:
                levels.append(y)
        levels[0], levels[-1] = start[1], goal[1]
        if levels[0] == levels[-1]:  # pragma: no cover - PlanRequest exige goal.y > start.y
            raise SchemaValidationError("niveles degenerados")
        return levels

    def _search(
        self,
        problem: _Problem,
        local_map: LocalMap,
        start: tuple[float, float],
        goal: tuple[float, float],
        include_unknown: bool,
    ) -> tuple[list[tuple[dict[str, float], tuple[tuple[float, float], ...]]], dict[str, Any]]:
        if not include_unknown:
            # Diagnostico: el mismo grafo tratando como libre el desconocido interior.
            relaxed_allowed = local_map.corridor.as_polygon().intersection(
                Polygon(local_map.known_region)
            )
            problem = _Problem(
                tool=problem.tool,
                allowed=relaxed_allowed,
                allowed_boundary=relaxed_allowed.boundary,
                exclusions=problem.exclusions,
                unknown=Polygon(),
                epsilon=problem.epsilon,
                margin=problem.margin,
                x_bounds=problem.x_bounds,
            )
        obstacles = _config_obstacles(problem, local_map)
        levels = self._levels(obstacles, start, goal)
        k = self.planner.lateral_candidates_per_interval
        xs: set[float] = {start[0], goal[0]}
        for y in levels:
            for low, high in _free_intervals(obstacles, y, problem.x_bounds):
                xs.update(low + (high - low) * (j + 1) / (k + 1) for j in range(k))
        x_values = sorted(xs)
        margin = problem.margin
        nodes: list[list[float]] = []
        for index, y in enumerate(levels):
            if index == 0:
                nodes.append([start[0]])
            elif index == len(levels) - 1:
                nodes.append([goal[0]])
            else:
                nodes.append(
                    [
                        x
                        for x in x_values
                        if _region_ok(problem, Polygon(problem.tool + np.array([x, y])), margin)
                        is None
                    ]
                )
        straight = math.hypot(goal[0] - start[0], goal[1] - start[1])
        weights, norms = self.planner.weights, self.planner.normalizers
        slope = (goal[0] - start[0]) / (goal[1] - start[1])

        def edge_cost(a: tuple[float, float], b: tuple[float, float]) -> dict[str, float]:
            length = math.hypot(b[0] - a[0], b[1] - a[1]) / straight
            line_a = start[0] + slope * (a[1] - start[1])
            line_b = start[0] + slope * (b[1] - start[1])
            deviation = 0.5 * (abs(a[0] - line_a) + abs(b[0] - line_b)) * (b[1] - a[1])
            deviation /= straight * norms.deviation_m
            heading = math.atan2(abs(b[0] - a[0]), b[1] - a[1])
            smoothness = 2.0 * heading**2 / norms.smoothness_rad2
            return {"length": length, "deviation": deviation, "smoothness": smoothness}

        best: dict[tuple[int, float], list[tuple[float, tuple[tuple[float, float], ...], dict]]] = {
            (0, start[0]): [(0.0, (start,), {"length": 0.0, "deviation": 0.0, "smoothness": 0.0})]
        }
        evaluated = 0
        exhausted = False
        edge_rejections: dict[str, int] = {}
        limit = self.planner.max_candidates
        for level in range(len(levels) - 1):
            y0, y1 = levels[level], levels[level + 1]
            for x1 in nodes[level + 1]:
                incoming = []
                for x0 in nodes[level]:
                    entries = best.get((level, x0))
                    if not entries:
                        continue
                    if evaluated >= self.planner.max_graph_edges:
                        exhausted = True
                        break
                    evaluated += 1
                    cause = _region_ok(
                        problem, _box_sweep(problem.tool, (x0, y0), (x1, y1)), margin
                    )
                    if cause is not None:
                        edge_rejections[cause] = edge_rejections.get(cause, 0) + 1
                        continue
                    parts = edge_cost((x0, y0), (x1, y1))
                    increment = (
                        weights.length * parts["length"]
                        + weights.deviation * parts["deviation"]
                        + weights.smoothness * parts["smoothness"]
                    )
                    for total, path, breakdown in entries:
                        incoming.append(
                            (
                                total + increment,
                                (*path, (x1, y1)),
                                {key: breakdown[key] + parts[key] for key in parts},
                            )
                        )
                if incoming:
                    incoming.sort(key=lambda item: (item[0], item[1]))
                    best[(level + 1, x1)] = incoming[:limit]
        final = best.get((len(levels) - 1, goal[0]), [])
        candidates = []
        for _, path, breakdown in final:
            cost = dict(breakdown)
            cost["length"] -= 1.0  # exceso relativo sobre la recta
            cost["total"] = (
                weights.length * cost["length"]
                + weights.deviation * cost["deviation"]
                + weights.smoothness * cost["smoothness"]
            )
            candidates.append((cost, path))
        search = {
            "levels": len(levels),
            "level_y_m": levels,
            "lateral_candidates": len(x_values),
            "nodes": sum(len(n) for n in nodes),
            "edges_evaluated": evaluated,
            "edge_rejections": dict(sorted(edge_rejections.items())),
            "budget_exhausted": exhausted,
            "candidates_found": len(candidates),
            "config_obstacles": len(obstacles),
            "include_unknown_as_forbidden": include_unknown,
        }
        return candidates, search

    def _task_path(
        self,
        request: PlanRequest,
        local_map: LocalMap,
        profile: ToolGeometry,
        waypoints: tuple[tuple[float, float], ...],
        rank: int,
    ) -> TaskPath:
        return TaskPath(
            path_id=f"{request.request_id}:candidate{rank}",
            domain=local_map.domain,
            segments=quintic_lane_change_segments(waypoints),
            orientation=request.orientation,
            profile_id=profile.tool_id,
            profile_hash=profile.content_hash(),
            map_id=local_map.map_id,
            map_hash=local_map.content_hash(),
            role=request.role,
            working_contact=True,
            task_tolerance_m=0.0,
            continuity_declared="C2",
            provenance=local_map.provenance,
        )


def plan_path(
    request: PlanRequest,
    local_map: LocalMap,
    profile: ToolGeometry,
    planner: PlannerConfig,
    validation: ValidationConfig,
    safety: SafetyConfig,
) -> PlanResult:
    """Atajo funcional de `GeometricPlanner(...).plan(...)`."""
    return GeometricPlanner(planner, validation, safety).plan(request, local_map, profile)
