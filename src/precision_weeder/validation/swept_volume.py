"""Validador geometrico independiente del barrido continuo (orden A1/E2 §4.3).

Decide la aceptacion **reconstruyendo** las huellas desde la curva, el perfil y el
mapa: no confia en flags, aristas aprobadas ni costos del planificador.

Certificado por intervalos, para orientacion fija y huella convexa `K`:

- En un subintervalo `[a, b]` del parametro local de un tramo, la cuerda
  `p(a)->p(b)` barre exactamente `conv((K + p(a)) U (K + p(b)))`: es el barrido
  exacto de una traslacion rectilinea de un convexo.
- La curva real se separa de la cuerda a lo sumo `delta = M (b - a)^2 / 8`, con
  `M >= max ||p''(u)||` en `[a, b]`. `M` se obtiene **analiticamente**: cada
  componente de `p''` es un polinomio cubico y su maximo absoluto se alcanza en un
  extremo del intervalo o en una raiz de su derivada (nunca por muestreo).
- Por tanto el barrido real esta contenido en `hull ⊕ D(delta)`. Se certifica el
  intervalo si `delta <= geometric_approximation_tolerance_m` (presupuesto de
  `configs/safety.yaml`), `dist(hull, exclusiones) > clearance_epsilon + delta`, y
  `hull ⊕ D(delta)` cabe en el dominio permitido (cerrado: `covers` y distancia al
  borde `>= delta`). Si no, se biseca; al agotar la profundidad o el presupuesto
  total, el intervalo queda `UNCERTIFIED`.
- Una violacion es **definitiva** si la huella real en una muestra viola el
  despeje o sale del dominio, o si el intervalo es rectilineo (`delta = 0`) y su
  barrido exacto lo hace: entonces se rechaza con evidencia.

El despeje minimo reportado es una **cota inferior certificada** (`dist - delta` en
los intervalos certificados), no una distancia exacta. La cobertura se calcula con
la union de barridos de cuerda de la huella de corte: exacta en tramos rectilineos;
en tramos curvos es una estimacion rotulada con su cota de error.

Este modulo no importa el planificador, el simulador, la cinematica ni video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPoint, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..core.config import SafetyConfig, ValidationConfig, discretization_excess_violation
from ..core.errors import ConfigError, SchemaValidationError
from ..core.identity import content_hash, freeze, is_sha256, thaw
from ..core.trajectory import E2_SUPPORTED_ROLES, OrientationPolicy, TaskPath
from ..core.units import require_same_domain
from ..geometry.tool_profile import ToolGeometry
from ..mapping.local_map import LocalMap

__all__ = [
    "ValidationReport",
    "ValidationStatus",
    "polynomial_abs_max",
    "polynomial_min",
    "require_discretization_budget",
    "unsupported_capabilities",
    "validate_task_path",
    "validation_config_hash",
]

VALIDATION_REPORT_SCHEMA_VERSION: Final[int] = 2
"""v2 (A1E2-R1): informe inmutable, `from_dict` y `content_hash`."""
CLEARANCE_LABEL: Final[str] = (
    "cota inferior certificada (dist(barrido de cuerda) - delta); no es distancia exacta"
)
_BOUND_PAD_REL: Final[float] = 1e-9
_BOUND_PAD_ABS: Final[float] = 1e-15


class ValidationStatus(StrEnum):
    """Estado geometrico. `ACCEPTED` no implica viabilidad mecanica."""

    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    UNCERTIFIED = "UNCERTIFIED"


# --- cotas analiticas de polinomios ------------------------------------------------


def _derivative(coefficients: np.ndarray) -> np.ndarray:
    """Derivada de un polinomio en base ascendente."""
    if coefficients.size <= 1:
        return np.zeros(1)
    return coefficients[1:] * np.arange(1, coefficients.size)


def _evaluate(coefficients: np.ndarray, u: float) -> float:
    return float(np.polynomial.polynomial.polyval(u, coefficients))


def _real_roots_in(coefficients: np.ndarray, low: float, high: float) -> list[float]:
    trimmed = np.trim_zeros(np.asarray(coefficients, dtype=np.float64), trim="b")
    if trimmed.size <= 1:
        return []
    roots = np.polynomial.polynomial.polyroots(trimmed)
    return [
        float(r.real)
        for r in roots
        if abs(r.imag) <= 1e-12 * max(1.0, abs(r.real)) and low <= r.real <= high
    ]


def polynomial_abs_max(coefficients: np.ndarray, low: float, high: float) -> float:
    """Maximo de `|q(u)|` en `[low, high]`: extremos y raices de `q'` (analitico)."""
    candidates = [low, high, *_real_roots_in(_derivative(coefficients), low, high)]
    return max(abs(_evaluate(coefficients, u)) for u in candidates)


def polynomial_min(coefficients: np.ndarray, low: float, high: float) -> float:
    """Minimo de `q(u)` en `[low, high]`: extremos y raices de `q'` (analitico)."""
    candidates = [low, high, *_real_roots_in(_derivative(coefficients), low, high)]
    return min(_evaluate(coefficients, u) for u in candidates)


# --- informe ---------------------------------------------------------------------------


_STATUS_RULES: Final[str] = (
    "ACCEPTED: sin violaciones ni intervalos no certificados; REJECTED: al menos una "
    "violacion; UNCERTIFIED: intervalos no certificados y ninguna violacion"
)


@dataclass(frozen=True, eq=False)
class ValidationReport:
    """Informe de validacion geometrica ligado por hash a ruta, mapa, perfil y config.

    Es un certificado terminado e inmutable (A1E2-R1): sus colecciones se congelan
    al construirlo (`core.identity.freeze`) y `to_dict` exporta copias profundas
    independientes. `config_hash` identifica `validation + safety`; no es el hash de
    la configuracion del planificador.
    """

    status: ValidationStatus
    path_hash: str
    map_hash: str
    profile_hash: str
    config_hash: str
    method: str
    violations: tuple[Any, ...] = ()
    uncertified_intervals: tuple[Any, ...] = ()
    certified_intervals: int = 0
    evaluated_intervals: int = 0
    envelope_error_max_m: float = 0.0
    envelope_error_budget_m: float = 0.0
    clearance_required_m: float = 0.0
    clearance_lower_bound_m: float | None = None
    continuity: Any = field(default_factory=dict)
    progress: Any = field(default_factory=dict)
    endpoints: Any = field(default_factory=dict)
    coverage: Any = field(default_factory=dict)
    approximations: Any = field(default_factory=dict)
    effective_geometry: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", ValidationStatus(self.status))
        except ValueError as exc:
            raise SchemaValidationError(f"ValidationReport: estado invalido ({exc})") from exc
        for name in ("path_hash", "map_hash", "profile_hash", "config_hash"):
            if not is_sha256(getattr(self, name)):
                raise SchemaValidationError(f"ValidationReport.{name}: se espera SHA-256")
        if not str(self.method).strip():
            raise SchemaValidationError("ValidationReport.method: cadena vacia")
        for name in ("certified_intervals", "evaluated_intervals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SchemaValidationError(f"ValidationReport.{name}: entero >= 0")
        for name in ("envelope_error_max_m", "envelope_error_budget_m", "clearance_required_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchemaValidationError(f"ValidationReport.{name}: real")
            if not math.isfinite(value) or value < 0:
                raise SchemaValidationError(f"ValidationReport.{name}: finito y >= 0")
            object.__setattr__(self, name, float(value))
        bound = self.clearance_lower_bound_m
        if bound is not None:
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise SchemaValidationError("ValidationReport.clearance_lower_bound_m: real")
            if not math.isfinite(bound):
                raise SchemaValidationError("ValidationReport.clearance_lower_bound_m: finito")
            object.__setattr__(self, "clearance_lower_bound_m", float(bound))
        for name in (
            "violations",
            "uncertified_intervals",
            "continuity",
            "progress",
            "endpoints",
            "coverage",
            "approximations",
            "effective_geometry",
        ):
            object.__setattr__(self, name, freeze(getattr(self, name)))
        self._check_status_coherence()

    def _check_status_coherence(self) -> None:
        has_violations = bool(self.violations)
        has_uncertified = bool(self.uncertified_intervals)
        coherent = {
            ValidationStatus.ACCEPTED: not has_violations and not has_uncertified,
            ValidationStatus.REJECTED: has_violations,
            ValidationStatus.UNCERTIFIED: has_uncertified and not has_violations,
        }[self.status]
        if not coherent:
            raise SchemaValidationError(
                f"ValidationReport: estado {self.status.value} incoherente ({_STATUS_RULES})"
            )
        if self.status is ValidationStatus.ACCEPTED:
            if self.envelope_error_max_m > self.envelope_error_budget_m:
                raise SchemaValidationError(
                    "ValidationReport ACCEPTED con envolvente mayor que el presupuesto"
                )
            if (
                self.clearance_lower_bound_m is not None
                and self.clearance_lower_bound_m <= self.clearance_required_m
            ):
                raise SchemaValidationError(
                    "ValidationReport ACCEPTED con cota de despeje <= clearance_epsilon"
                )

    @property
    def accepted(self) -> bool:
        return self.status is ValidationStatus.ACCEPTED

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": VALIDATION_REPORT_SCHEMA_VERSION,
            "status": self.status.value,
            "validation_scope": "geometric",
            "mechanical_feasibility": "not_evaluated",
            "path_hash": self.path_hash,
            "map_hash": self.map_hash,
            "profile_hash": self.profile_hash,
            "config_hash": self.config_hash,
            "config_hash_scope": "validation + safety (no incluye el planificador)",
            "method": self.method,
            "violations": thaw(self.violations),
            "uncertified_intervals": thaw(self.uncertified_intervals),
            "certified_intervals": self.certified_intervals,
            "evaluated_intervals": self.evaluated_intervals,
            "envelope_error_max_m": self.envelope_error_max_m,
            "envelope_error_budget_m": self.envelope_error_budget_m,
            "clearance_required_m": self.clearance_required_m,
            "clearance_lower_bound_m": self.clearance_lower_bound_m,
            "clearance_lower_bound_label": CLEARANCE_LABEL,
            "continuity": thaw(self.continuity),
            "progress": thaw(self.progress),
            "endpoints": thaw(self.endpoints),
            "coverage": thaw(self.coverage),
            "approximations": thaw(self.approximations),
            "effective_geometry": thaw(self.effective_geometry),
        }

    def content_hash(self) -> str:
        return content_hash(self._payload())

    def to_dict(self) -> dict[str, Any]:
        """Exportacion como copia profunda independiente, con identidad de contenido."""
        payload = self._payload()
        payload["content_hash"] = content_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidationReport:
        """Reconstruye un informe exportado y verifica version, alcance, estado e identidad.

        Raises:
            SchemaValidationError: campo ausente, version/alcance distintos, valores no
                finitos, estado incoherente o `content_hash` que no coincide.
        """
        try:
            if payload["schema_version"] != VALIDATION_REPORT_SCHEMA_VERSION:
                raise SchemaValidationError(
                    f"ValidationReport: schema_version {payload['schema_version']!r} no soportada"
                )
            if (
                payload["validation_scope"] != "geometric"
                or payload["mechanical_feasibility"] != "not_evaluated"
            ):
                raise SchemaValidationError("ValidationReport: alcance distinto del geometrico")
            report = cls(
                status=payload["status"],
                path_hash=payload["path_hash"],
                map_hash=payload["map_hash"],
                profile_hash=payload["profile_hash"],
                config_hash=payload["config_hash"],
                method=payload["method"],
                violations=payload["violations"],
                uncertified_intervals=payload["uncertified_intervals"],
                certified_intervals=payload["certified_intervals"],
                evaluated_intervals=payload["evaluated_intervals"],
                envelope_error_max_m=payload["envelope_error_max_m"],
                envelope_error_budget_m=payload["envelope_error_budget_m"],
                clearance_required_m=payload["clearance_required_m"],
                clearance_lower_bound_m=payload["clearance_lower_bound_m"],
                continuity=payload["continuity"],
                progress=payload["progress"],
                endpoints=payload["endpoints"],
                coverage=payload["coverage"],
                approximations=payload["approximations"],
                effective_geometry=payload["effective_geometry"],
            )
        except KeyError as exc:
            raise SchemaValidationError(f"ValidationReport: falta el campo {exc}") from exc
        if payload.get("content_hash") != report.content_hash():
            raise SchemaValidationError(
                "ValidationReport: content_hash no coincide con el contenido"
            )
        return report


# --- geometria efectiva ----------------------------------------------------------------


def _effective_tool(
    profile: ToolGeometry, theta: float, tool_inflation: float
) -> tuple[np.ndarray, np.ndarray]:
    """Huellas efectivas en orientacion `theta`: colision (inflada si procede) y corte."""
    collision = profile.inflated_collision_shape(tool_inflation)
    rotated_collision = affinity.rotate(collision, theta, origin=(0.0, 0.0), use_radians=True)
    rotated_cut = affinity.rotate(profile.cutting_shape, theta, origin=(0.0, 0.0), use_radians=True)
    return (
        np.asarray(rotated_collision.exterior.coords[:-1], dtype=np.float64),
        np.asarray(rotated_cut.exterior.coords[:-1], dtype=np.float64),
    )


def _placed(vertices: np.ndarray, point: tuple[float, float]) -> Polygon:
    return Polygon(vertices + np.asarray(point))


def _chord_hull(vertices: np.ndarray, a: tuple[float, float], b: tuple[float, float]) -> Polygon:
    points = np.vstack([vertices + np.asarray(a), vertices + np.asarray(b)])
    hull = MultiPoint(points).convex_hull
    return hull if isinstance(hull, Polygon) else Polygon(vertices + np.asarray(a))


@dataclass
class _Context:
    tool: np.ndarray
    cut: np.ndarray
    exclusions: tuple[tuple[str, Polygon], ...]
    exclusion_union: BaseGeometry
    allowed: BaseGeometry
    allowed_boundary: BaseGeometry
    unknown: BaseGeometry
    epsilon: float
    budget: float
    max_depth: int
    max_intervals: int
    evaluated: int = 0
    certified: int = 0
    max_delta: float = 0.0
    clearance_bound: float = math.inf
    violations: list[dict[str, Any]] = field(default_factory=list)
    uncertified: list[dict[str, Any]] = field(default_factory=list)
    cut_hulls: list[Polygon] = field(default_factory=list)
    budget_exhausted: bool = False


def _footprint_violation(
    ctx: _Context, footprint: Polygon, s: float, kind_prefix: str
) -> dict[str, Any] | None:
    """Violacion definitiva de una huella real (muestra) o de un barrido exacto."""
    if not ctx.allowed.covers(footprint):
        outside_unknown = footprint.intersection(ctx.unknown).area > 0.0
        return {
            "kind": "UNKNOWN_SPACE" if outside_unknown else "OUT_OF_DOMAIN",
            "evidence": kind_prefix,
            "s": s,
            "detail": "la huella sale del dominio conocido permitido",
        }
    if ctx.exclusions and footprint.distance(ctx.exclusion_union) <= ctx.epsilon:
        nearest = min(ctx.exclusions, key=lambda item: footprint.distance(item[1]))
        return {
            "kind": "COLLISION",
            "evidence": kind_prefix,
            "s": s,
            "plant_id": nearest[0],
            "distance_m": float(footprint.distance(nearest[1])),
            "required_gt_m": ctx.epsilon,
            "detail": "despeje <= clearance_epsilon (contacto, tangencia o invasion)",
        }
    return None


def _validate_segment(ctx: _Context, path: TaskPath, index: int) -> None:
    segment = path.segments[index]
    ddx = _derivative(_derivative(segment.x_coefficients))
    ddy = _derivative(_derivative(segment.y_coefficients))

    def point(u: float) -> tuple[float, float]:
        return segment.derivative_u(u, 0)

    def s_of(u: float) -> float:
        return segment.s_start + u * segment.length_s

    # Muestras reales en los extremos del tramo: evidencia directa.
    for u in (0.0, 1.0):
        violation = _footprint_violation(ctx, _placed(ctx.tool, point(u)), s_of(u), "sample")
        if violation is not None:
            violation["segment"] = index
            ctx.violations.append(violation)
            return

    stack: list[tuple[float, float, int]] = [(0.0, 1.0, 0)]
    while stack:
        a, b, depth = stack.pop()
        if ctx.evaluated >= ctx.max_intervals:
            ctx.budget_exhausted = True
            ctx.uncertified.append(
                {"segment": index, "s_from": s_of(a), "s_to": s_of(b), "cause": "max_intervals"}
            )
            continue
        ctx.evaluated += 1
        bound_x = polynomial_abs_max(ddx, a, b)
        bound_y = polynomial_abs_max(ddy, a, b)
        curvature_bound = math.hypot(bound_x, bound_y)
        curvature_bound = curvature_bound * (1.0 + _BOUND_PAD_REL) + (
            _BOUND_PAD_ABS if curvature_bound > 0.0 else 0.0
        )
        delta = curvature_bound * (b - a) ** 2 / 8.0
        pa, pb = point(a), point(b)
        hull = _chord_hull(ctx.tool, pa, pb)

        if delta == 0.0:
            # Traslacion rectilinea: el barrido de la cuerda es exacto.
            violation = _footprint_violation(ctx, hull, s_of(a), "exact_linear_sweep")
            if violation is not None:
                violation.update({"segment": index, "s_from": s_of(a), "s_to": s_of(b)})
                ctx.violations.append(violation)
                return
            distance = hull.distance(ctx.exclusion_union) if ctx.exclusions else math.inf
            ctx.certified += 1
            ctx.clearance_bound = min(ctx.clearance_bound, distance)
            ctx.cut_hulls.append(_chord_hull(ctx.cut, pa, pb))
            continue

        certified = False
        inside = (
            delta <= ctx.budget
            and ctx.allowed.covers(hull)
            and hull.distance(ctx.allowed_boundary) >= delta
        )
        if inside:
            distance = hull.distance(ctx.exclusion_union) if ctx.exclusions else math.inf
            if distance > ctx.epsilon + delta:
                certified = True
                ctx.certified += 1
                ctx.max_delta = max(ctx.max_delta, delta)
                ctx.clearance_bound = min(ctx.clearance_bound, distance - delta)
                ctx.cut_hulls.append(_chord_hull(ctx.cut, pa, pb))
        if certified:
            continue
        # Evidencia definitiva en el punto medio real antes de subdividir.
        middle = 0.5 * (a + b)
        violation = _footprint_violation(
            ctx, _placed(ctx.tool, point(middle)), s_of(middle), "sample"
        )
        if violation is not None:
            violation["segment"] = index
            ctx.violations.append(violation)
            return
        if depth >= ctx.max_depth:
            ctx.uncertified.append(
                {
                    "segment": index,
                    "s_from": s_of(a),
                    "s_to": s_of(b),
                    "delta_m": delta,
                    "cause": "max_subdivision_depth",
                }
            )
            continue
        # Orden determinista: primero la mitad inferior.
        stack.append((middle, b, depth + 1))
        stack.append((a, middle, depth + 1))


def _continuity(path: TaskPath, tolerance: float) -> dict[str, Any]:
    jumps = {0: 0.0, 1: 0.0, 2: 0.0}
    for left, right in zip(path.segments[:-1], path.segments[1:], strict=True):
        for order in jumps:
            lv = np.array(left.derivative_s(1.0, order))
            rv = np.array(right.derivative_s(0.0, order))
            jumps[order] = max(jumps[order], float(np.max(np.abs(lv - rv))))
    finite = all(
        np.all(np.isfinite(seg.x_coefficients)) and np.all(np.isfinite(seg.y_coefficients))
        for seg in path.segments
    )
    required = {"C0": 0, "C1": 1, "C2": 2}[path.continuity_declared]
    ok = finite and all(jumps[o] <= tolerance for o in range(required + 1))
    return {
        "declared": path.continuity_declared,
        "max_jump_position_m": jumps[0],
        "max_jump_first_derivative": jumps[1],
        "max_jump_second_derivative": jumps[2],
        "tolerance_abs": tolerance,
        "finite": finite,
        "satisfied": ok,
        "joins": len(path.segments) - 1,
    }


def _progress(path: TaskPath) -> dict[str, Any]:
    minimum = math.inf
    for segment in path.segments:
        dy = _derivative(segment.y_coefficients) / segment.length_s
        minimum = min(minimum, polynomial_min(dy, 0.0, 1.0))
    start_y, end_y = path.start_point[1], path.end_point[1]
    return {
        "min_dy_ds": minimum,
        "method": "minimo analitico de dy/ds por tramo (extremos y raices)",
        "non_decreasing": minimum >= 0.0,
        "start_y": start_y,
        "end_y": end_y,
        "advances": end_y > start_y,
    }


def validate_task_path(
    path: TaskPath,
    local_map: LocalMap,
    profile: ToolGeometry,
    safety: SafetyConfig,
    config: ValidationConfig,
    expected_start: tuple[float, float] | None = None,
    expected_goal: tuple[float, float] | None = None,
) -> ValidationReport:
    """Valida un `TaskPath` contra el mapa y el perfil, sin informacion del planificador.

    Capacidades E2 (fuera de ellas no se emite certificado de trabajo): orientacion
    `FIXED`, huellas de colision **y** de corte convexas, rol `pasada_principal`,
    contacto de trabajo y tolerancia de tarea 0.

    Raises:
        SchemaValidationError: identidad o dominio incoherentes entre ruta, mapa y
            perfil (dato invalido: no se valida contra otro mapa o herramienta).
        ConfigError: una exclusion efectiva excede
            `safety.max_conservative_discretization_excess_m` (entrada incompatible
            con el presupuesto declarado; no es ausencia de ruta).
    """
    if not isinstance(path, TaskPath) or not isinstance(local_map, LocalMap):
        raise SchemaValidationError("validate_task_path: se esperan TaskPath y LocalMap")
    if not isinstance(profile, ToolGeometry):
        raise SchemaValidationError("validate_task_path: se espera ToolGeometry")
    require_same_domain(path.domain, local_map.domain, profile.domain, context="validacion")
    map_hash, profile_hash = local_map.content_hash(), profile.content_hash()
    if path.map_hash != map_hash:
        raise SchemaValidationError("la ruta referencia otro mapa (map_hash distinto)")
    if path.profile_hash != profile_hash:
        raise SchemaValidationError("la ruta referencia otro perfil (profile_hash distinto)")
    require_discretization_budget(local_map, safety)
    common: dict[str, Any] = {
        "path_hash": path.content_hash(),
        "map_hash": map_hash,
        "profile_hash": profile_hash,
        "config_hash": validation_config_hash(safety, config),
        "method": (
            "biseccion con envolvente exterior conv(K+p(a), K+p(b)) ⊕ D(M h^2/8); M analitico; "
            "barrido rectilineo exacto"
        ),
        "envelope_error_budget_m": safety.geometric_approximation_tolerance_m,
        "clearance_required_m": safety.clearance_epsilon_m,
    }
    unsupported = unsupported_capabilities(path, profile)
    if unsupported:
        return ValidationReport(
            status=ValidationStatus.REJECTED,
            violations=[{"kind": "UNSUPPORTED_FEATURE", "detail": item} for item in unsupported],
            **common,
        )

    theta = float(path.orientation.theta_rad)  # type: ignore[arg-type]
    tool, cut = _effective_tool(profile, theta, local_map.tool_inflation)
    allowed = local_map.allowed_region()
    exclusions = tuple(
        (e.plant_id, poly)
        for e, poly in zip(local_map.exclusions, local_map.exclusion_polygons(), strict=True)
    )
    ctx = _Context(
        tool=tool,
        cut=cut,
        exclusions=exclusions,
        exclusion_union=local_map.exclusion_union(),
        allowed=allowed,
        allowed_boundary=allowed.boundary,
        unknown=local_map.unknown_union(),
        epsilon=safety.clearance_epsilon_m,
        budget=safety.geometric_approximation_tolerance_m,
        max_depth=config.max_subdivision_depth,
        max_intervals=config.max_intervals,
    )
    effective_geometry = {
        "tool_collision_polygon_W_orientation": tool.tolist(),
        "tool_cutting_polygon_W_orientation": cut.tolist(),
        "theta_rad": theta,
        "tool_inflation_m": local_map.tool_inflation,
        "uncertainty_inflate": local_map.uncertainty.inflate.value,
        "exclusion_total_radii_m": {e.plant_id: e.total_radius for e in local_map.exclusions},
        "allowed_region_area_m2": float(allowed.area),
    }
    approximations = {
        "exclusions": "poligonos exteriores (contienen el circulo exacto)",
        "max_exclusion_discretization_excess_m": max(
            (e.discretization_excess for e in local_map.exclusions), default=0.0
        ),
        "max_conservative_discretization_excess_m": safety.max_conservative_discretization_excess_m,
        "tool_inflation": "buffer en inglete (contiene el buffer circular)",
    }

    violations: list[dict[str, Any]] = []
    continuity = _continuity(path, config.continuity_tolerance)
    if not continuity["satisfied"]:
        violations.append({"kind": "DISCONTINUITY", "detail": continuity})
    progress = _progress(path)
    if not progress["non_decreasing"] or not progress["advances"]:
        violations.append({"kind": "NON_MONOTONIC_PROGRESS", "detail": progress})
    endpoints: dict[str, Any] = {"start": list(path.start_point), "end": list(path.end_point)}
    for name, expected, actual in (
        ("start", expected_start, path.start_point),
        ("goal", expected_goal, path.end_point),
    ):
        if expected is not None:
            error = float(np.max(np.abs(np.asarray(actual) - np.asarray(expected))))
            endpoints[f"{name}_error_m"] = error
            if error > config.endpoint_tolerance_m:
                violations.append({"kind": "ENDPOINT_MISMATCH", "endpoint": name, "error_m": error})
    endpoints["tolerance_m"] = config.endpoint_tolerance_m

    for index in range(len(path.segments)):
        _validate_segment(ctx, path, index)
    violations.extend(ctx.violations)

    if violations:
        status = ValidationStatus.REJECTED
    elif ctx.uncertified:
        status = ValidationStatus.UNCERTIFIED
    else:
        status = ValidationStatus.ACCEPTED
    return ValidationReport(
        status=status,
        violations=violations,
        uncertified_intervals=ctx.uncertified,
        certified_intervals=ctx.certified,
        evaluated_intervals=ctx.evaluated,
        envelope_error_max_m=ctx.max_delta,
        clearance_lower_bound_m=(
            None if math.isinf(ctx.clearance_bound) else float(ctx.clearance_bound)
        ),
        continuity=continuity,
        progress=progress,
        endpoints=endpoints,
        coverage=_coverage(ctx, local_map) if status is ValidationStatus.ACCEPTED else {},
        approximations=approximations,
        effective_geometry=effective_geometry,
        **common,
    )


def validation_config_hash(safety: SafetyConfig, config: ValidationConfig) -> str:
    """Identidad de la configuracion del validador (`validation + safety`)."""
    return content_hash(
        {"validation": config.model_dump(mode="json"), "safety": safety.model_dump(mode="json")}
    )


def unsupported_capabilities(path: TaskPath, profile: ToolGeometry) -> list[str]:
    """Capacidades pedidas fuera del perfil E2; lista vacia si todo esta soportado.

    Un rechazo por capacidad del software no es una imposibilidad geometrica.
    """
    issues = []
    if path.orientation.policy is not OrientationPolicy.FIXED:
        issues.append(f"orientacion {path.orientation.policy.value}: solo FIXED es validable")
    if not profile.is_convex():
        issues.append("huella de colision no convexa")
    if not profile.is_cut_convex():
        issues.append(
            "huella de corte no convexa: su barrido no es la envolvente convexa (cobertura no "
            "soportada)"
        )
    if path.role not in E2_SUPPORTED_ROLES:
        issues.append(
            f"rol {path.role!r} no soportado por el perfil E2 (solo {sorted(E2_SUPPORTED_ROLES)})"
        )
    if not path.working_contact:
        issues.append("working_contact=False: el perfil E2 solo certifica trabajo con contacto")
    if path.task_tolerance_m != 0.0:
        issues.append("task_tolerance_m > 0: E2 valida la curva exacta (tolerancia 0)")
    return issues


def require_discretization_budget(local_map: LocalMap, safety: SafetyConfig) -> None:
    """R12 sobre las exclusiones **efectivas** del mapa (regla unica de E1).

    Raises:
        ConfigError: alguna exclusion excede el presupuesto de exceso conservador.
    """
    limit = safety.max_conservative_discretization_excess_m
    for exclusion in local_map.exclusions:
        violation = discretization_excess_violation(
            exclusion.total_radius, exclusion.polygon_segments, limit
        )
        if violation is not None:
            raise ConfigError(
                f"mapa {local_map.map_id}, exclusion {exclusion.plant_id}: {violation}; "
                "entrada incompatible con el presupuesto declarado"
            )


def _coverage(ctx: _Context, local_map: LocalMap) -> dict[str, Any]:
    swept = unary_union(ctx.cut_hulls) if ctx.cut_hulls else Polygon()
    target = local_map.target_region()
    covered = swept.intersection(target)
    exact = ctx.max_delta == 0.0
    return {
        "method": "union de barridos de cuerda de la huella de corte",
        "exact": exact,
        "label": (
            "exacta (tramos rectilineos, huella de corte convexa)"
            if exact
            else (
                "estimacion rotulada: barridos de cuerdas a <= chord_deviation_max_m de la "
                "curva; sin cota de area garantizada"
            )
        ),
        "error_bound_m2": 0.0 if exact else None,
        "chord_deviation_max_m": ctx.max_delta,
        "swept_cut_area_m2": float(swept.area),
        "covered_target_area_m2": float(covered.area),
        "target_area_m2": float(target.area),
        "covered_fraction": float(covered.area / target.area) if target.area > 0 else None,
        "note": "la cobertura es objetivo secundario; nunca compensa una colision",
    }
