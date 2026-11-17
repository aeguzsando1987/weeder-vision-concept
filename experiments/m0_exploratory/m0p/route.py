"""Planificacion por cuadro con E2 y vigencia de la ruta (orden M0-P §4.7-§4.8, MP-05/06).

E2 se usa **sin modificar su algoritmo**: el planificador propone y el validador
independiente certifica. La tarea intrafilar es trabajar los **gaps entre plantas de
la linea de plantacion** (directriz del usuario, 26-10-2026): la ruta es una cadena de
tramos E2 entre puntos de paso sobre el eje de la hilera, desde la linea de
herramienta hacia el borde delantero visible, rodeando cada planta. Cada tramo se
vuelve a validar con una llamada nueva a E2 (MP-05). La cadena solo es vigente en el
cuadro cuyo mapa la certifico, o en cuadros siguientes si **todos** sus tramos se
revalidan contra el mapa nuevo dentro de la edad maxima; si no, se retira (MP-06).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from numpy.polynomial import Polynomial
from shapely import affinity

from precision_weeder.core.config import (
    PlannerConfig,
    SafetyConfig,
    ToolConfig,
    ValidationConfig,
)
from precision_weeder.core.identity import content_hash
from precision_weeder.core.trajectory import OrientationRequirement, PathSegment, TaskPath
from precision_weeder.core.types import Frame, Point2
from precision_weeder.core.units import Provenance, SpatialDomain
from precision_weeder.geometry.tool_profile import ToolGeometry
from precision_weeder.mapping.local_map import LocalMap
from precision_weeder.planning import GeometricPlanner, PlanRequest, PlanResult
from precision_weeder.planning.smoothing import quintic_lane_change_segments
from precision_weeder.validation.swept_volume import ValidationReport, validate_task_path

from .profile import M0Profile
from .projection import SegmentProjection

__all__ = [
    "Reference",
    "RouteAttempt",
    "RouteLeg",
    "RoutePlanner",
    "assumed_tool_profile",
    "chain_hash",
    "chain_parametrization",
    "chain_x_at",
    "gap_waypoints",
    "lateral_candidates",
    "trim_path",
]

DOMAIN = SpatialDomain.GROUND_ASSUMED

QUINTIC_PEAK_SLOPE = 1.875
"""Maximo de `h'(u)` del cambio lateral quintico: pendiente pico = 1.875 * dx / dy."""


def assumed_tool_profile(config: ToolConfig) -> ToolGeometry:
    """Cuchilla de `configs/tool.yaml` en el dominio `ground_assumed` (dimensiones supuestas)."""
    return ToolGeometry.rectangular(
        tool_id=config.tool_id,
        length_x=config.length_x_m,
        width_y=config.width_y_m,
        pivot_offset=config.pivot_offset_m,
        cutting_inset=config.cutting_inset_m,
        orientation_mode=config.orientation_mode,
        domain=DOMAIN,
        provenance=Provenance.ASSUMED,
    )


def lateral_candidates(
    corridor_box: tuple[float, float, float, float],
    tool: ToolGeometry,
    samples: int,
    axis_x: float,
    margin: float = 0.0,
) -> list[float]:
    """`x` candidatas del origen `B`: el eje de la hilera y posiciones uniformes donde
    la huella cabe en el corredor, ordenadas por distancia al eje (desempate: menor `x`).
    Asi la ruta se plantea sobre la linea de plantacion y E2 la desvia alrededor de
    cada planta; solo se aparta del eje si inicio o meta caen dentro de una exclusion.
    `margin` es la holgura de generacion de E2 respecto del borde del corredor: como el
    corredor es solidario al vehiculo, absorbe su deriva lateral entre cuadros."""
    polygon = np.asarray(tool.collision_polygon)
    low = corridor_box[0] - float(polygon[:, 0].min()) + margin
    high = corridor_box[1] - float(polygon[:, 0].max()) - margin
    if high < low:
        return []
    axis = min(max(axis_x, low), high)
    values = [low + (high - low) * (k + 0.5) / samples for k in range(samples)]
    values.append(axis)
    unique = sorted(set(values), key=lambda x: (abs(x - axis), x))
    return unique


def gap_waypoints(
    local_map: LocalMap,
    tool: ToolGeometry,
    axis_x: float,
    y_from: float,
    y_to: float,
    step: float,
    margin: float,
) -> list[float]:
    """`y` de los puntos de paso sobre el eje de la hilera, ordenados de `y_from` a `y_to`.

    Se muestrea el eje cada `step`; una muestra es candidata si la huella de colision
    (con la inflacion del mapa) cabe en la region permitida y queda a mas de `margin`
    de toda exclusion. Cada tramo contiguo de muestras candidatas es un gap y aporta su
    centro; los gaps que tocan los extremos aportan el extremo. Es solo un filtro de
    candidatos: E2 decide la admisibilidad de cada tramo."""
    shape = tool.inflated_collision_shape(local_map.tool_inflation)
    allowed = local_map.allowed_region()
    exclusions = local_map.exclusion_union()
    count = max(2, math.ceil(abs(y_to - y_from) / step) + 1)
    ys = np.linspace(y_from, y_to, count)

    def candidate(y: float) -> bool:
        region = affinity.translate(shape, axis_x, float(y))
        if not allowed.covers(region):
            return False
        return exclusions.is_empty or region.distance(exclusions) > margin

    flags = [candidate(y) for y in ys]
    points: list[float] = []
    k = 0
    while k < count:
        if not flags[k]:
            k += 1
            continue
        end = k
        while end + 1 < count and flags[end + 1]:
            end += 1
        if k == 0:
            points.append(float(ys[0]))
        if end == count - 1:
            points.append(float(ys[-1]))
        if k != 0 and end != count - 1:
            points.append(float(0.5 * (ys[k] + ys[end])))
        k = end + 1
    return sorted(set(points), key=lambda y: abs(y - y_from))


def path_x_at(path: TaskPath, y: float) -> float:
    """`x` del camino en la ordenada `y` (en E2, `y` es lineal en `u` dentro de cada tramo)."""
    for segment in path.segments:
        y0, y1 = segment.derivative_u(0.0)[1], segment.derivative_u(1.0)[1]
        if min(y0, y1) - 1e-12 <= y <= max(y0, y1) + 1e-12:
            u = 0.0 if y1 == y0 else min(max((y - y0) / (y1 - y0), 0.0), 1.0)
            return segment.derivative_u(u)[0]
    raise ValueError(f"y={y} fuera del camino")


def chain_x_at(paths: tuple[TaskPath, ...], y: float) -> float | None:
    for path in paths:
        if path.start_point[1] - 1e-12 <= y <= path.end_point[1] + 1e-12:
            return path_x_at(path, y)
    return None


def trim_path(path: TaskPath, y_cut: float) -> TaskPath:
    """Mismo camino a partir de la ordenada `y_cut` (recorte exacto, no aproximacion).

    El tramo cortado se reparametriza por composicion polinomica `u = u0 + (1-u0) v`
    y el parametro `s` restante se reescala linealmente a `[0, 1]`. Como en E2 `y` es
    lineal en `s`, la curva y su continuidad C2 se conservan; el resultado se vuelve
    a validar como cualquier camino."""
    start_y, end_y = path.start_point[1], path.end_point[1]
    if y_cut <= start_y:
        return path
    if y_cut >= end_y:
        raise ValueError("recorte fuera del camino")
    s_cut = None
    segments: list[PathSegment] = []
    for segment in path.segments:
        y0, y1 = segment.derivative_u(0.0)[1], segment.derivative_u(1.0)[1]
        if y1 <= y_cut:
            continue
        if s_cut is None:
            u0 = max(0.0, (y_cut - y0) / (y1 - y0))
            s_cut = segment.s_start + u0 * segment.length_s
            mapping = Polynomial([u0, 1.0 - u0])
            x = Polynomial(segment.x_coefficients)(mapping).coef
            y = Polynomial(segment.y_coefficients)(mapping).coef
            segment = PathSegment(
                s_start=segment.s_start,
                s_end=segment.s_end,
                x_coefficients=np.pad(x, (0, 6 - len(x))),
                y_coefficients=np.pad(y, (0, 6 - len(y))),
            )
            segments.append(segment)
        else:
            segments.append(segment)
    scale = 1.0 - s_cut
    rebuilt = []
    for k, segment in enumerate(segments):
        new_start = 0.0 if k == 0 else rebuilt[-1].s_end
        new_end = 1.0 if k == len(segments) - 1 else (segment.s_end - s_cut) / scale
        rebuilt.append(
            PathSegment(new_start, new_end, segment.x_coefficients, segment.y_coefficients)
        )
    return replace(path, segments=tuple(rebuilt))


def chain_parametrization(paths: tuple[TaskPath, ...]) -> dict[str, Any]:
    """Parametro global `S` en `[0, 1]` de la cadena, proporcional al avance en `y`.

    Cada camino E2 tiene `y` lineal en su `s` local; con intervalos de `S` de longitud
    `L_i = dy_i / dy_total`, `dy/dS` es la misma constante en toda la cadena y
    `d^k p/dS^k = (d^k p/ds^k) / L_i^k`. Se comprueban posicion, primera y segunda
    derivada en cada union (C2 global), ademas de progreso estricto en `y`."""
    y0, y1 = paths[0].start_point[1], paths[-1].end_point[1]
    total = y1 - y0
    breakpoints, joints, ok = [0.0], [], total > 0.0
    lengths = []
    for path in paths:
        span = path.end_point[1] - path.start_point[1]
        ok = ok and span > 0.0
        lengths.append(span / total if total > 0 else 0.0)
        breakpoints.append(breakpoints[-1] + lengths[-1])
    breakpoints[-1] = 1.0
    for k in range(len(paths) - 1):
        left, right = paths[k], paths[k + 1]
        la, lb = lengths[k], lengths[k + 1]
        if la <= 0 or lb <= 0:
            ok = False
            continue
        gaps = []
        for order in (0, 1, 2):
            a = np.array(left.evaluate(1.0, order)) / la**order
            b = np.array(right.evaluate(0.0, order)) / lb**order
            scale = max(1.0, float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            gaps.append(float(np.max(np.abs(a - b))) / (scale if order else 1.0))
        joint_ok = gaps[0] <= 1e-9 and gaps[1] <= 1e-6 and gaps[2] <= 1e-6
        ok = ok and joint_ok
        joints.append({"c0_m": gaps[0], "c1_rel": gaps[1], "c2_rel": gaps[2], "ok": joint_ok})
    return {
        "definition": "S en [0,1]; intervalo de cada tramo proporcional a su avance en y",
        "breakpoints": breakpoints,
        "joints": joints,
        "c2_ok": bool(ok),
    }


@dataclass(frozen=True)
class RouteLeg:
    """Tramo certificado de una cadena contra el mapa del cuadro."""

    path: TaskPath
    report: ValidationReport
    start: tuple[float, float]
    goal: tuple[float, float]
    target: str
    """`gap` (punto de paso sobre el eje), `eje_horizonte` o `cierre` (meta lateral)."""
    origin: str = "planificado"
    """`planificado` en este cuadro o `conservado` (prefijo recortado y revalidado)."""
    plant_id: str | None = None
    """Planta a cuyo costado termina el tramo (solo nudos del patron alternante)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_m": list(self.start),
            "goal_m": list(self.goal),
            "target": self.target,
            "origin": self.origin,
        }


@dataclass(frozen=True)
class RouteAttempt:
    """Cadena del cuadro: completa y certificada (vigente) o diagnostico no vigente."""

    legs: tuple[RouteLeg, ...]
    skipped: tuple[tuple[float, float], ...]
    complete: bool
    axis_x: float
    decision: str
    cause: str
    last: PlanResult | None
    attempts: tuple[dict[str, Any], ...]
    no_gaps: bool = False
    parametrization: dict[str, Any] | None = None
    pattern: dict[str, Any] | None = None
    """Solo con patron `alternating`: costado por planta de la cadena y propuestas de
    alternancia que E2 no certifico (oportunidades omitidas, con su causa)."""

    @property
    def paths(self) -> tuple[TaskPath, ...] | None:
        """Solo una cadena completa hasta el horizonte, con todos sus tramos aceptados y
        C2 global, es una ruta vigente; un prefijo es diagnostico (R1)."""
        if not (self.complete and self.legs):
            return None
        if not all(leg.report.accepted for leg in self.legs):
            return None
        if not (self.parametrization and self.parametrization["c2_ok"]):
            return None
        return tuple(leg.path for leg in self.legs)

    @property
    def kind(self) -> str | None:
        """`trabajo_en_gaps` si la cadena alcanza al menos un gap intermedio entre plantas
        sobre el eje; `rodeo_lateral` en otro caso. Volver al eje solo en el borde del
        horizonte visible no acredita trabajo en un gap."""
        if self.paths is None:
            return None
        in_gap = any(leg.target == "gap" for leg in self.legs)
        return "trabajo_en_gaps" if in_gap else "rodeo_lateral"

    def summary(self) -> dict[str, Any]:
        last = self.last
        if self.legs and not self.complete:
            reason = "cadena incompleta: no alcanza el horizonte solicitado"
        elif self.no_gaps and not self.legs:
            reason = "sin candidato admisible encontrado sobre la linea de plantacion"
        else:
            reason = last.reason.value if last and last.reason else None
        summary = {
            "status": last.status.value if last else None,
            "reason": reason,
            "detail": last.detail if last else None,
            "kind": self.kind,
            "complete": self.complete,
            "axis_x_m": self.axis_x,
            "legs": [leg.to_dict() for leg in self.legs],
            "skipped_gaps_m": [list(w) for w in self.skipped],
            "parametrization": self.parametrization,
            "attempts": list(self.attempts),
        }
        if self.pattern is not None:
            summary["pattern"] = self.pattern
        return summary


@dataclass(frozen=True)
class Reference:
    """Propuesta anterior vigente: contexto para la continuidad entre cuadros (R5)."""

    legs: tuple[RouteLeg, ...]
    axis_x: float
    frame_index: int
    chain_hash: str


def chain_hash(paths: tuple[TaskPath, ...]) -> str:
    """Identidad de la cadena: hash de los hashes de sus tramos, en orden."""
    return content_hash({"legs": [path.content_hash() for path in paths]})


def _sides(paths: tuple[TaskPath, ...], local_map: LocalMap, y_range) -> dict[str, int]:
    """Costado (+1 derecha, -1 izquierda) por el que la cadena pasa cada exclusion."""
    sides = {}
    for exclusion in local_map.exclusions:
        cy = exclusion.center.y
        if y_range[0] <= cy <= y_range[1]:
            x = chain_x_at(paths, cy)
            if x is not None:
                sides[exclusion.plant_id] = 1 if x >= exclusion.center.x else -1
    return sides


class RoutePlanner:
    """Envuelve E2 para un perfil M0-P."""

    def __init__(
        self,
        profile: M0Profile,
        planner: PlannerConfig,
        validation: ValidationConfig,
        safety: SafetyConfig,
        tool: ToolGeometry,
    ) -> None:
        self.profile = profile
        self.planner_config = planner
        self.planner = GeometricPlanner(planner, validation, safety)
        self.validation = validation
        self.safety = safety
        self.tool = tool
        self._half_x = float(np.abs(np.asarray(tool.collision_polygon)[:, 0]).max())
        self._half_y = float(np.abs(np.asarray(tool.collision_polygon)[:, 1]).max())
        self._alternating = profile.planning.pattern == "alternating"
        self._map: LocalMap | None = None  # mapa del cuadro en curso (patron)
        self._omitted: dict[tuple, dict[str, Any]] = {}
        self._committed: dict[str, int] = {}  # costado decidido por planta (segmento actual)
        self._committed_segment: str | None = None
        self._last_failure: str | None = None
        self._horizon_limit: str | None = None  # planta que acorta el cierre del cuadro

    def _request(self, local_map: LocalMap, start, goal, rid: str) -> PlanRequest:
        return PlanRequest(
            request_id=rid,
            map_id=local_map.map_id,
            map_hash=local_map.content_hash(),
            profile_id=self.tool.tool_id,
            profile_hash=self.tool.content_hash(),
            start=Point2(x=start[0], y=start[1], frame=Frame.WORLD, domain=DOMAIN),
            goal=Point2(x=goal[0], y=goal[1], frame=Frame.WORLD, domain=DOMAIN),
            orientation=OrientationRequirement.fixed(0.0),
        )

    def _validate(self, path: TaskPath, local_map: LocalMap, start=None, goal=None):
        return validate_task_path(
            path,
            local_map,
            self.tool,
            self.safety,
            self.validation,
            expected_start=start,
            expected_goal=goal,
        )

    def _rebind(self, path: TaskPath, local_map: LocalMap) -> tuple[TaskPath, ValidationReport]:
        """Mismo camino ligado al mapa nuevo y su certificado (el que se publica, R2)."""
        rebound = replace(path, map_id=local_map.map_id, map_hash=local_map.content_hash())
        return rebound, self._validate(rebound, local_map)

    # --- Patron alternante (orden V1, paso 2) ---------------------------------------------

    def _blocking(self, local_map: LocalMap, axis_x: float, y_low: float, y_high: float):
        """Plantas que obligan a apartarse del eje entre dos ordenadas, ordenadas por `y`."""
        plants = [
            e
            for e in local_map.exclusions
            if y_low + 1e-9 < e.center.y < y_high - 1e-9
            and abs(e.center.x - axis_x) < e.total_radius + self._half_x
        ]
        return sorted(plants, key=lambda e: e.center.y)

    def _side_before(self, prior, local_map: LocalMap, axis_x: float, y: float) -> int | None:
        """Costado por el que la cadena `prior` paso la ultima planta anterior a `y`."""
        if not prior:
            return None
        y0 = prior[0].start_point[1]
        plants = self._blocking(local_map, axis_x, y0 - 1e-6, y + 1e-6)
        sides = _sides(prior, local_map, (y0, y))
        for plant in reversed(plants):
            if plant.plant_id in sides:
                return sides[plant.plant_id]
        return None

    def _knot_leg(self, local_map, a, b, target, tried):
        """Tramo quintico C2 entre dos nudos del patron, o la causa por la que no se usa.

        Se rechaza sin validar si su pendiente lateral maxima supera la del perfil: un
        cruce mas brusco exigiria al actuador un movimiento lateral demasiado rapido para
        el avance. Si la cumple, solo se acepta con el certificado del validador de E2."""
        key = ("knot", a, b)
        if key not in tried:
            dy = b[1] - a[1]
            limit = self.profile.planning.pattern_max_slope
            slope = math.inf if dy <= 1e-9 else QUINTIC_PEAK_SLOPE * abs(b[0] - a[0]) / dy
            if slope > limit:
                tried[key] = f"cruce demasiado brusco (pendiente {slope:.1f} > {limit:g})"
            else:
                request = self._request(local_map, a, b, local_map.map_id)
                path = TaskPath(
                    path_id=f"{local_map.map_id}:patron",
                    domain=local_map.domain,
                    segments=quintic_lane_change_segments([a, b]),
                    orientation=request.orientation,
                    profile_id=self.tool.tool_id,
                    profile_hash=self.tool.content_hash(),
                    map_id=local_map.map_id,
                    map_hash=local_map.content_hash(),
                    role=request.role,
                    working_contact=True,
                    task_tolerance_m=0.0,
                    continuity_declared="C2",
                    provenance=local_map.provenance,
                )
                report = self._validate(path, local_map, a, b)
                if report.accepted:
                    tried[key] = RouteLeg(path, report, a, b, target)
                else:
                    tried[key] = f"tramo no certificado ({report.status.value})"
        outcome = tried[key]
        if isinstance(outcome, RouteLeg) and outcome.target != target:
            outcome = replace(outcome, target=target)
        return outcome

    def _extend_pattern(
        self, local_map, start, axis_x, y_goal, tried, prior, side_hint=None, passed=None
    ):
        """Cadena del patron alternante desde `start` hasta el horizonte `y_goal`.

        Un nudo por planta, a su costado y a la altura de su centro; entre nudos, un
        cambio lateral quintico tan largo como la separacion entre plantas. El costado
        de cada planta se decide una vez y se recuerda: el opuesto al de la planta
        anterior si el cruce cumple la pendiente maxima y se certifica; si no, el mismo
        costado (sin cruce) y la omision queda registrada. Nada depende de E2 como
        generador; todo tramo lleva el certificado del validador de E2.

        Al continuar una ruta, `side_hint` es el costado en que esta el punto de partida
        y `passed` la planta a cuyo costado termina el tramo conservado: ya esta rodeada
        aunque su centro estimado se haya movido por delante del nudo."""
        reach = (
            self.planner_config.generation_margin_m
            + self.profile.planning.pattern_side_margin_m
            + self.safety.clearance_epsilon_m
        )
        side = self._side_before(prior, local_map, axis_x, start[1])
        if side is None:
            side = side_hint
        legs: list[RouteLeg] = []
        current = start
        plants = [
            plant
            for plant in self._blocking(local_map, axis_x, start[1], y_goal)
            if plant.plant_id != passed
        ]
        for plant in plants:
            committed = self._committed.get(plant.plant_id)
            if committed is not None:
                order = [committed, -committed]
            elif side is not None:
                order = [-side, side]
            else:
                near = 1 if current[0] >= plant.center.x else -1
                order = [near, -near]
            failures = []
            chosen = tight = None
            for candidate in order:
                offset = plant.total_radius + self._half_x + reach
                knot = (plant.center.x + candidate * offset, plant.center.y)
                crossing = side is not None and candidate != side
                outcome = self._knot_leg(
                    local_map, current, knot, "gap" if crossing else "costado", tried
                )
                if isinstance(outcome, RouteLeg) and crossing and committed is None:
                    # Un cruce nuevo solo se compromete si le sobra holgura: uno certificado
                    # por milimetros deja de serlo al moverse las plantas seguidas.
                    slack = self.profile.planning.pattern_commit_clearance_m
                    bound = outcome.report.clearance_lower_bound_m
                    if bound is not None and bound < slack:
                        tight = (candidate, knot, outcome)
                        outcome = f"cruce con holgura insuficiente ({bound:.3f} < {slack:g} m)"
                if isinstance(outcome, RouteLeg):
                    chosen = (candidate, knot, outcome)
                    break
                failures.append(outcome)
            if chosen is None and tight is not None:
                chosen, failures = tight, []  # sin otra opcion, vale el cruce certificado
            if chosen is None:
                self._last_failure = "; ".join(failures)
                return legs, [], False, None, False
            candidate, knot, outcome = chosen
            if failures:
                self._omitted[(current, knot)] = {
                    "plants": [plant.plant_id],
                    "wanted_side": order[0],
                    "taken_side": candidate,
                    "cause": failures[0],
                }
            legs.append(replace(outcome, plant_id=plant.plant_id))
            current, side = knot, candidate
        # Una planta cuyo centro aun esta mas alla del horizonte no se puede rodear todavia
        # (se ve solo en parte), pero su zona protegida puede asomar: el cierre recto
        # termina antes de ella y la ruta se prolonga cuando la planta entra al horizonte.
        y_end = y_goal
        for exclusion in local_map.exclusions:
            if (
                exclusion.center.y >= y_goal
                and abs(exclusion.center.x - current[0])
                < exclusion.total_radius + self._half_x + reach
            ):
                edge = exclusion.center.y - exclusion.total_radius - self._half_y - reach
                if edge < y_end:
                    y_end, self._horizon_limit = edge, exclusion.plant_id
        if y_end - current[1] > 1e-3:
            closing = self._knot_leg(local_map, current, (current[0], y_end), "cierre", tried)
            if not isinstance(closing, RouteLeg):
                self._last_failure = closing
                return legs, [], False, None, not plants
            legs.append(closing)
        elif not legs:
            self._last_failure = "sin recorrido libre hasta la planta del horizonte"
            return legs, [], False, None, not plants
        return legs, [], True, None, not plants

    def _pattern_summary(self, legs, axis_x: float) -> dict[str, Any] | None:
        if not self._alternating or self._map is None:
            return None
        paths = tuple(leg.path for leg in legs)
        sides: dict[str, int] = {}
        if paths:
            span = (paths[0].start_point[1], paths[-1].end_point[1])
            blocking = self._blocking(self._map, axis_x, span[0] - 1e-6, span[1] + 1e-6)
            taken = _sides(paths, self._map, span)
            sides = {p.plant_id: taken[p.plant_id] for p in blocking if p.plant_id in taken}
        keys = {(leg.start, leg.goal) for leg in legs}
        return {
            "name": "alternating",
            "plant_sides": sides,
            "omitted": [event for key, event in self._omitted.items() if key in keys],
            "failure": self._last_failure,
            "horizon_limited_by": self._horizon_limit,
        }

    def _extend(self, local_map, start, axis_x, y_goal, lateral, attempts, tried, prior=()):
        """Cadena desde `start` hasta el horizonte `y_goal` por los gaps del eje.

        Devuelve `(legs, skipped, complete, last, no_gaps)`. Desde cada punto se
        planifica al siguiente gap alcanzable; los no alcanzables se registran. Tras el
        ultimo gap, la meta del horizonte sobre el eje o, si no es admisible (plantas
        juntas), la meta lateral mas cercana al eje. `complete` exige llegar a `y_goal`."""
        last: PlanResult | None = None

        def leg(a, b, target):
            nonlocal last
            if (a, b) not in tried:
                result = self.planner.plan(
                    self._request(local_map, a, b, local_map.map_id), local_map, self.tool
                )
                attempts.append(
                    {
                        "start": list(a),
                        "goal": list(b),
                        "status": result.status.value,
                        "reason": result.reason.value if result.reason else None,
                    }
                )
                if result.accepted:
                    report = self._validate(result.path, local_map, a, b)
                    tried[(a, b)] = RouteLeg(result.path, report, a, b, target)
                else:
                    tried[(a, b)] = result
            outcome = tried[(a, b)]
            if isinstance(outcome, PlanResult):
                last = outcome
            elif outcome.target != target:
                outcome = replace(outcome, target=target)
            return outcome

        if start[1] >= y_goal:
            return [], [], True, None, False
        if self._alternating:
            return self._extend_pattern(local_map, start, axis_x, y_goal, tried, prior)
        ys = gap_waypoints(
            local_map,
            self.tool,
            axis_x,
            start[1],
            y_goal,
            self.profile.planning.gap_step_m,
            self.planner_config.generation_margin_m + self.safety.clearance_epsilon_m,
        )
        targets = [(axis_x, y) for y in ys if y > start[1] + 1e-9]
        legs: list[RouteLeg] = []
        skipped: list[tuple[float, float]] = []
        current, pending = start, list(targets)
        while pending:
            for k, target in enumerate(pending):
                kind = "eje_horizonte" if target[1] == y_goal else "gap"
                outcome = leg(current, target, kind)
                if isinstance(outcome, PlanResult):
                    if outcome.reason is not None and outcome.reason.value == "START_INADMISSIBLE":
                        return legs, skipped, False, last, not targets
                    continue
                if not outcome.report.accepted:
                    return legs, skipped, False, last, not targets  # sin certificado
                legs.append(outcome)
                skipped.extend(pending[:k])
                current, pending = target, pending[k + 1 :]
                break
            else:
                skipped.extend(pending)
                pending = []
        if current[1] != y_goal:
            for goal in [(x, y_goal) for x in lateral]:
                outcome = leg(current, goal, "cierre")
                if isinstance(outcome, RouteLeg):
                    if outcome.report.accepted:
                        legs.append(outcome)
                        current = goal
                    break
                if outcome.reason is not None and outcome.reason.value == "START_INADMISSIBLE":
                    break
        return legs, skipped, current[1] == y_goal, last, not targets

    def _frame_geometry(self, projection, corridor_box, axis_x):
        cfg = self.profile.planning
        y_start = projection.row_v_to_ground_y(self.profile.tool_line.v_norm)
        goal_v = 1.0 - cfg.goal_margin_norm if projection.forward_sign > 0 else cfg.goal_margin_norm
        y_goal = projection.row_v_to_ground_y(goal_v)
        lateral = lateral_candidates(
            corridor_box,
            self.tool,
            cfg.lateral_samples,
            axis_x,
            self.planner_config.generation_margin_m,
        )
        return y_start, y_goal, lateral

    def _attempt(self, legs, skipped, complete, axis_x, decision, cause, last, attempts, no_gaps):
        legs = tuple(legs)
        param = chain_parametrization(tuple(leg.path for leg in legs)) if legs else None
        pattern = self._pattern_summary(legs, axis_x)
        if pattern is not None and complete and legs:
            self._committed.update(pattern["plant_sides"])
        return RouteAttempt(
            legs,
            tuple(skipped),
            complete,
            axis_x,
            decision,
            cause,
            last,
            tuple(attempts),
            no_gaps,
            param,
            pattern,
        )

    def plan(
        self,
        local_map: LocalMap,
        projection: SegmentProjection,
        corridor_box: tuple[float, float, float, float],
        axis_x: float,
        reference: Reference | None = None,
        extend: bool = True,
    ) -> RouteAttempt:
        """Cadena de tramos E2 por los gaps de la linea de plantacion, con continuidad.

        Sin referencia: se arranca sobre el eje si es candidato; si no, en la posicion
        lateral mas cercana al eje desde la que E2 complete la cadena.

        Con referencia (propuesta vigente anterior del mismo segmento, R5): si el eje
        observado no se aparta mas de `axis_hysteresis_m` del de la tarea, se conserva
        la tarea y el prefijo de la cadena anterior, recortado en la linea de
        herramienta y revalidado contra el mapa actual; solo se planifica lo nuevo por
        delante, sin cambiar el costado por el que se rodean las plantas ya decididas.
        Si eso no es posible (nuevo obstaculo, cambio de eje, sin cierre), se
        reselecciona **enlazando** desde la posicion de la referencia en la linea de
        herramienta; si E2 no lo logra, se declara discontinuidad y no hay ruta.

        Ningun tramo se acepta sin su certificado; solo una cadena completa hasta el
        horizonte es vigente (R1)."""
        y_start, y_goal, lateral = self._frame_geometry(projection, corridor_box, axis_x)
        attempts: list[dict[str, Any]] = []
        tried: dict = {}
        self._map, self._omitted, self._last_failure = local_map, {}, None
        self._horizon_limit = None
        if self._committed_segment != local_map.segment_id:
            # Los identificadores de planta se reinician con cada segmento `W`.
            self._committed, self._committed_segment = {}, local_map.segment_id
        if reference is None:
            return self._fresh(local_map, axis_x, y_start, y_goal, lateral, attempts, tried)
        x_ref = chain_x_at(tuple(leg.path for leg in reference.legs), y_start)
        if x_ref is None:
            attempt = self._fresh(local_map, axis_x, y_start, y_goal, lateral, attempts, tried)
            decision = "discontinuidad" if attempt.paths else attempt.decision
            return replace(
                attempt, decision=decision, cause="la referencia no cubre la linea de herramienta"
            )
        if self._alternating and extend:
            # El eje solo decide que plantas pertenecen a la hilera: su cambio no reinicia
            # la ruta, que esta anclada a las plantas.
            return self._continue_pattern(
                local_map, reference, x_ref, y_start, y_goal, axis_x, attempts, tried
            )
        tolerance = self.profile.planning.axis_hysteresis_m
        if abs(axis_x - reference.axis_x) > tolerance:
            return self._linked(
                local_map,
                (x_ref, y_start),
                axis_x,
                y_goal,
                attempts,
                tried,
                f"eje de la tarea cambiado ({reference.axis_x:.3f} -> {axis_x:.3f} m)",
            )
        axis_task = reference.axis_x
        lateral = lateral_candidates(
            corridor_box,
            self.tool,
            self.profile.planning.lateral_samples,
            axis_task,
            self.planner_config.generation_margin_m,
        )
        kept: list[RouteLeg] = []
        broken = None
        for leg in reference.legs:
            if leg.goal[1] <= y_start + 1e-9:
                continue
            path = trim_path(leg.path, y_start) if leg.start[1] < y_start else leg.path
            start = (path.start_point[0], path.start_point[1])
            rebound, report = self._rebind(path, local_map)
            if not report.accepted:
                broken = f"tramo conservado invalido en el mapa actual ({report.status.value})"
                break
            kept.append(RouteLeg(rebound, report, start, leg.goal, leg.target, "conservado"))
        if not kept:
            return self._linked(
                local_map,
                (x_ref, y_start),
                axis_task,
                y_goal,
                attempts,
                tried,
                broken or "sin prefijo conservable",
            )
        # Reutilizar sin extender solo si el prefijo sigue libre y llega al horizonte
        # actual; si la proyeccion avanzo y falta recorrido, se prolonga como en una
        # replanificacion (R1).
        reaches = kept[-1].goal[1] >= y_goal - 1e-9
        if not extend and (broken is not None or reaches):
            return self._attempt(
                kept,
                (),
                broken is None,
                axis_task,
                "revalidada",
                broken or f"ruta del cuadro {reference.frame_index} revalidada",
                None,
                attempts,
                False,
            )
        # Planificar solo por delante. Un cierre aun no alcanzado se replanifica desde su
        # inicio si conserva los costados ya decididos; si no, se prolonga desde su final.
        while kept:
            tail = kept[-1]
            if tail.target == "cierre" and tail.start[1] > y_start + 1e-9 and len(kept) > 1:
                legs, skipped, complete, last, no_gaps = self._extend(
                    local_map,
                    tail.start,
                    axis_task,
                    y_goal,
                    lateral,
                    attempts,
                    tried,
                    tuple(leg.path for leg in kept[:-1]),
                )
                if complete:
                    old = (tail.path,)
                    new = tuple(leg.path for leg in legs)
                    span = (tail.start[1], tail.goal[1])
                    if _sides(old, local_map, span) == _sides(new, local_map, span):
                        return self._attempt(
                            kept[:-1] + legs,
                            skipped,
                            True,
                            axis_task,
                            "prefijo_conservado",
                            "costados conservados",
                            last,
                            attempts,
                            no_gaps,
                        )
            legs, skipped, complete, last, no_gaps = self._extend(
                local_map,
                tail.goal,
                axis_task,
                y_goal,
                lateral,
                attempts,
                tried,
                tuple(leg.path for leg in kept),
            )
            if complete:
                cause = broken or "costados conservados"
                return self._attempt(
                    kept + legs,
                    skipped,
                    True,
                    axis_task,
                    "prefijo_conservado",
                    cause,
                    last,
                    attempts,
                    no_gaps,
                )
            if len(kept) == 1:
                break
            kept.pop()  # el tramo retirado esta por delante de la herramienta
            broken = "sin cierre desde el prefijo; se acorta por delante de la herramienta"
        return self._linked(
            local_map,
            (x_ref, y_start),
            axis_task,
            y_goal,
            attempts,
            tried,
            broken or "sin cierre desde el prefijo",
        )

    def _continue_pattern(
        self, local_map, reference, x_ref, y_start, y_goal, axis_x, attempts, tried
    ):
        """Continuacion del patron alternante: se conserva solo el tramo en curso y el
        resto se vuelve a anclar a las plantas del cuadro.

        Las plantas seguidas no quedan fijas en `W` (la copa se desplaza respecto del
        suelo estimado al avanzar), asi que un nudo fijado cuando la planta entro al
        cuadro queda atrasado al llegar la herramienta. Lo comprometido es el costado de
        cada planta y el tramo bajo la herramienta; los nudos por delante siguen a su
        planta. Los tramos se unen con pendiente y curvatura nulas, de modo que mover un
        nudo posterior no rompe la continuidad C2. Un cierre recto en curso es
        provisional: se corta en la linea de herramienta para poder cruzar hacia la
        siguiente planta."""
        kept: list[RouteLeg] = []
        start = (x_ref, y_start)
        leg = next((leg for leg in reference.legs if leg.goal[1] > y_start + 1e-9), None)
        if leg is not None and leg.target != "cierre":
            path = trim_path(leg.path, y_start) if leg.start[1] < y_start else leg.path
            rebound, report = self._rebind(path, local_map)
            if not report.accepted:
                return self._linked(
                    local_map,
                    start,
                    axis_x,
                    y_goal,
                    attempts,
                    tried,
                    f"tramo en curso invalido en el mapa actual ({report.status.value})",
                )
            origin = (path.start_point[0], path.start_point[1])
            kept = [replace(leg, path=rebound, report=report, start=origin, origin="conservado")]
            start = leg.goal
        hint = None
        if abs(start[0] - axis_x) > self._half_x:
            hint = 1 if start[0] > axis_x else -1
        if start[1] >= y_goal:
            legs, skipped, complete, last, no_gaps = [], [], True, None, False
        else:
            legs, skipped, complete, last, no_gaps = self._extend_pattern(
                local_map,
                start,
                axis_x,
                y_goal,
                tried,
                tuple(leg.path for leg in kept),
                side_hint=hint,
                passed=kept[0].plant_id if kept else None,
            )
        if complete and (kept or legs):
            return self._attempt(
                kept + legs,
                skipped,
                True,
                axis_x,
                "prefijo_conservado",
                "tramo en curso conservado; nudos por delante anclados a las plantas",
                last,
                attempts,
                no_gaps,
            )
        return self._linked(
            local_map,
            (x_ref, y_start),
            axis_x,
            y_goal,
            attempts,
            tried,
            self._last_failure or "sin cierre desde el tramo en curso",
        )

    def _fresh(self, local_map, axis_x, y_start, y_goal, lateral, attempts, tried):
        starts = [(x, y_start) for x in lateral]
        ys = gap_waypoints(
            local_map,
            self.tool,
            axis_x,
            y_start,
            y_goal,
            self.profile.planning.gap_step_m,
            self.planner_config.generation_margin_m + self.safety.clearance_epsilon_m,
        )
        if ys and ys[0] == y_start:
            starts = [(axis_x, y_start)] + [s for s in starts if s[0] != axis_x]
        best = None
        for start in starts:
            legs, skipped, complete, last, no_gaps = self._extend(
                local_map, start, axis_x, y_goal, lateral, attempts, tried
            )
            if complete and legs:
                return self._attempt(
                    legs,
                    skipped,
                    True,
                    axis_x,
                    "inicio",
                    "sin propuesta anterior en el segmento",
                    last,
                    attempts,
                    no_gaps,
                )
            if best is None or len(legs) > len(best[0]):
                best = (legs, skipped, complete, last, no_gaps)
        legs, skipped, _, last, no_gaps = best or ([], [], False, None, True)
        return self._attempt(
            legs, skipped, False, axis_x, "inicio", "sin cadena completa", last, attempts, no_gaps
        )

    def _linked(self, local_map, start, axis_x, y_goal, attempts, tried, cause):
        """Reseleccion enlazada desde la posicion de la referencia (no salto libre)."""
        corridor = local_map.corridor
        lateral = lateral_candidates(
            (corridor.x_min, corridor.x_max, corridor.y_min, corridor.y_max),
            self.tool,
            self.profile.planning.lateral_samples,
            axis_x,
            self.planner_config.generation_margin_m,
        )
        legs, skipped, complete, last, no_gaps = self._extend(
            local_map, start, axis_x, y_goal, lateral, attempts, tried
        )
        if complete and legs:
            return self._attempt(
                legs, skipped, True, axis_x, "reseleccion_enlazada", cause, last, attempts, no_gaps
            )
        return self._attempt(
            legs,
            skipped,
            False,
            axis_x,
            "discontinuidad",
            f"{cause}; sin enlace certificable desde la referencia",
            last,
            attempts,
            no_gaps,
        )
