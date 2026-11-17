"""Contratos temporales de tarea y de vehiculo (A3/E3, orden §3.1; ADR 0001).

- `TimePolynomial`: funcion vectorial del tiempo, polinomica por tramos en `t` (s), con
  coeficientes en base de potencias ascendente sobre `u = (t - t_k)/(t_{k+1} - t_k)`.
  La interpolacion entre nodos esta definida (no hay muestras sueltas) y las derivadas
  son exactas. Fuera de `[t_start, t_end]` no se evalua: no se extrapola.
- `TaskTrajectory` (`TIMED_REFERENCE`): un `TaskPath` E2 referenciado por hash y una ley
  `s(t)` explicita. `s` sigue siendo adimensional; `t` es el reloj declarado. El
  compilador no cambia tiempos, perfil, orientacion ni tolerancias.
- `VehicleMotion`: `T_WV(t)` como `(x_V, y_V, psi)` y sus derivadas, con reloj,
  horizonte, cota de incertidumbre y origen (referencia o prediccion).
- `TaskBinding`: rol de tarea -> herramienta y perfil compatibles.

Los limites de una ley (rango de `s`, monotonia, extremos de derivadas) se comprueban
sobre todo el intervalo mediante las raices del polinomio derivado, no solo en nodos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

import numpy as np

from .errors import SchemaValidationError, TemporalOrderError, UnitMismatchError
from .identity import content_hash
from .trajectory import POLYNOMIAL_COEFFICIENTS, TASK_DOMAINS, TaskPath
from .types import frozen_float_array
from .units import Provenance, SpatialDomain, ensure_finite, ensure_non_negative

__all__ = [
    "TEMPORAL_SCHEMA_VERSION",
    "MotionSource",
    "RoleBinding",
    "TaskBinding",
    "TaskTrajectory",
    "TemporalMode",
    "TimePolynomial",
    "VehicleMotion",
    "WorkRequirement",
    "quintic_segment",
]

TEMPORAL_SCHEMA_VERSION: Final[int] = 1

_JOIN_TOLERANCES: Final[tuple[float, float, float]] = (1e-9, 1e-6, 1e-6)
"""Discontinuidad admitida en las uniones: valor, primera y segunda derivada (orden §7)."""

_CONTINUITY_ORDER: Final[dict[str, int]] = {"C0": 0, "C1": 1, "C2": 2}


def _text(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise SchemaValidationError(f"{name}: cadena vacia")
    return text


def _segment_value(coefficients: np.ndarray, u: float, order: int) -> np.ndarray:
    """Derivada `order` respecto de `u` de cada componente, en base ascendente."""
    value = np.zeros(coefficients.shape[0])
    for power in range(order, POLYNOMIAL_COEFFICIENTS):
        value += math.perm(power, order) * coefficients[:, power] * u ** (power - order)
    return value


def quintic_segment(duration: float, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Coeficientes `(dims, 6)` del quintico de Hermite en `u` entre dos estados.

    `start` y `end` son `(3, dims)`: valor, derivada y segunda derivada **respecto de
    t**. La duracion convierte las derivadas a `u`.
    """
    duration = ensure_finite(duration, "duracion")
    if duration <= 0.0:
        raise TemporalOrderError("quintic_segment: duracion no positiva")
    p0, v0, a0 = (np.asarray(row, dtype=np.float64) for row in start)
    p1, v1, a1 = (np.asarray(row, dtype=np.float64) for row in end)
    v0, v1 = v0 * duration, v1 * duration
    a0, a1 = a0 * duration**2, a1 * duration**2
    c3 = 10 * (p1 - p0) - 6 * v0 - 4 * v1 - 1.5 * a0 + 0.5 * a1
    c4 = -15 * (p1 - p0) + 8 * v0 + 7 * v1 + 1.5 * a0 - a1
    c5 = 6 * (p1 - p0) - 3 * v0 - 3 * v1 - 0.5 * a0 + 0.5 * a1
    return np.stack([p0, v0, a0 / 2, c3, c4, c5], axis=-1)


@dataclass(frozen=True, slots=True, eq=False)
class TimePolynomial:
    """Funcion `f(t)` de `dims` componentes, polinomica (grado <= 5) por tramos.

    `breakpoints` estrictamente crecientes `(n+1,)`; `coefficients` `(n, dims, 6)`.
    La continuidad declarada (`C0`/`C1`/`C2`) se comprueba en las uniones con las
    tolerancias de la orden §7.
    """

    breakpoints: np.ndarray
    coefficients: np.ndarray
    continuity: str = "C2"

    def __post_init__(self) -> None:
        times = frozen_float_array(np.atleast_1d(self.breakpoints), "TimePolynomial.breakpoints")
        if times.ndim != 1 or times.size < 2:
            raise SchemaValidationError("TimePolynomial: se exigen al menos dos nodos")
        if np.any(np.diff(times) <= 0.0):
            raise TemporalOrderError("TimePolynomial: nodos de tiempo no estrictamente crecientes")
        coefficients = frozen_float_array(self.coefficients, "TimePolynomial.coefficients")
        if coefficients.ndim != 3 or coefficients.shape[0] != times.size - 1:
            raise SchemaValidationError(
                f"TimePolynomial: coeficientes {coefficients.shape} incompatibles con "
                f"{times.size - 1} tramos (se exige (tramos, dims, 6))"
            )
        if coefficients.shape[2] != POLYNOMIAL_COEFFICIENTS:
            raise SchemaValidationError("TimePolynomial: se exigen 6 coeficientes por componente")
        if self.continuity not in _CONTINUITY_ORDER:
            raise SchemaValidationError("TimePolynomial.continuity: C0, C1 o C2")
        object.__setattr__(self, "breakpoints", times)
        object.__setattr__(self, "coefficients", coefficients)
        for k in range(1, times.size - 1):
            for order in range(_CONTINUITY_ORDER[self.continuity] + 1):
                left = self._local(k - 1, 1.0, order)
                right = self._local(k, 0.0, order)
                if float(np.max(np.abs(left - right))) > _JOIN_TOLERANCES[order]:
                    raise SchemaValidationError(
                        f"TimePolynomial: discontinuidad de orden {order} en t={times[k]}"
                    )

    @property
    def dims(self) -> int:
        return int(self.coefficients.shape[1])

    @property
    def t_start(self) -> float:
        return float(self.breakpoints[0])

    @property
    def t_end(self) -> float:
        return float(self.breakpoints[-1])

    def _duration(self, k: int) -> float:
        return float(self.breakpoints[k + 1] - self.breakpoints[k])

    def _local(self, k: int, u: float, order: int) -> np.ndarray:
        return _segment_value(self.coefficients[k], u, order) / self._duration(k) ** order

    def segment_index(self, t: float) -> int:
        t = ensure_finite(t, "t")
        if not self.t_start <= t <= self.t_end:
            raise TemporalOrderError(
                f"t={t} fuera del intervalo [{self.t_start}, {self.t_end}]: no se extrapola"
            )
        index = int(np.searchsorted(self.breakpoints, t, side="right")) - 1
        return min(index, self.breakpoints.size - 2)

    def evaluate(self, t: float, order: int = 0) -> np.ndarray:
        """Valor (orden 0) o derivada respecto de `t` (orden 1, 2...) en `t`."""
        k = self.segment_index(t)
        u = (t - self.breakpoints[k]) / self._duration(k)
        return self._local(k, u, order)

    def bounds(self, component: int, order: int = 0) -> tuple[float, float]:
        """Minimo y maximo exactos de una derivada sobre todo el intervalo.

        Se evaluan extremos de tramo y raices reales en `[0, 1]` de la derivada
        siguiente: el resultado no depende de un muestreo.
        """
        low, high = math.inf, -math.inf
        for k in range(self.coefficients.shape[0]):
            poly = np.polynomial.Polynomial(self.coefficients[k, component]).deriv(order)
            candidates = [0.0, 1.0]
            for root in poly.deriv().roots():
                if abs(root.imag) <= 1e-12 and 0.0 <= root.real <= 1.0:
                    candidates.append(float(root.real))
            scale = self._duration(k) ** order
            for u in candidates:
                value = float(poly(u)) / scale
                low, high = min(low, value), max(high, value)
        return low, high

    @classmethod
    def from_states(
        cls, times: np.ndarray, states: np.ndarray, continuity: str = "C2"
    ) -> TimePolynomial:
        """Quinticos de Hermite por tramos; `states` es `(n+1, 3, dims)` (valor, d/dt, d2/dt2)."""
        times = np.asarray(times, dtype=np.float64)
        states = np.asarray(states, dtype=np.float64)
        segments = [
            quintic_segment(times[k + 1] - times[k], states[k], states[k + 1])
            for k in range(times.size - 1)
        ]
        return cls(times, np.stack(segments), continuity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "breakpoints_s": self.breakpoints.tolist(),
            "coefficients": self.coefficients.tolist(),
            "continuity": self.continuity,
            "interpolation": "polinomio por tramos en t, base ascendente en u local",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TimePolynomial:
        return cls(
            np.asarray(payload["breakpoints_s"]),
            np.asarray(payload["coefficients"]),
            payload["continuity"],
        )


class TemporalMode(StrEnum):
    """Modo temporal de la tarea. Solo `TIMED_REFERENCE` esta implementado."""

    TIMED_REFERENCE = "TIMED_REFERENCE"
    PATH_WITH_WINDOWS = "PATH_WITH_WINDOWS"


@dataclass(frozen=True, slots=True)
class WorkRequirement:
    """Condicion minima de trabajo, declarada antes de compilar o ensayar.

    `target_region` es un anillo en `W` (metros) fijado antes del ensayo; la cobertura
    cuenta la huella de corte ejecutada dentro de el.
    """

    min_advance_m: float = 0.0
    min_coverage_m2: float = 0.0
    target_region: tuple[tuple[float, float], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "min_advance_m", ensure_non_negative(self.min_advance_m, "min_advance_m")
        )
        object.__setattr__(
            self, "min_coverage_m2", ensure_non_negative(self.min_coverage_m2, "min_coverage_m2")
        )
        if self.target_region is not None:
            ring = tuple(
                (ensure_finite(x, "target_region"), ensure_finite(y, "target_region"))
                for x, y in self.target_region
            )
            if len(ring) < 3:
                raise SchemaValidationError("WorkRequirement.target_region: anillo degenerado")
            object.__setattr__(self, "target_region", ring)

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_advance_m": self.min_advance_m,
            "min_coverage_m2": self.min_coverage_m2,
            "target_region": [list(p) for p in self.target_region] if self.target_region else None,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> WorkRequirement:
        region = payload.get("target_region")
        return cls(
            payload["min_advance_m"],
            payload["min_coverage_m2"],
            tuple(tuple(p) for p in region) if region else None,
        )


@dataclass(frozen=True, slots=True, eq=False)
class TaskTrajectory:
    """Referencia temporal de tarea `p_d(t) = p(s(t))` en `W` (`TIMED_REFERENCE`).

    Invariantes: `s(t)` escalar, en `[0, 1]` y no decreciente en todo el intervalo
    (sin retroceso sobre el camino); tolerancia de tarea (tubo) no negativa. El
    `TaskPath` se conserva entero y se identifica por su hash; su parametrizacion
    global no se altera.
    """

    trajectory_id: str
    path: TaskPath
    timing: TimePolynomial
    clock_id: str
    work: WorkRequirement = field(default_factory=WorkRequirement)
    tolerance_m: float = 0.0
    mode: TemporalMode = TemporalMode.TIMED_REFERENCE
    provenance: Provenance = Provenance.SYNTHETIC
    schema_version: int = TEMPORAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "trajectory_id", _text(self.trajectory_id, "trajectory_id"))
        object.__setattr__(self, "clock_id", _text(self.clock_id, "clock_id"))
        object.__setattr__(self, "mode", TemporalMode(self.mode))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if self.schema_version != TEMPORAL_SCHEMA_VERSION:
            raise SchemaValidationError("TaskTrajectory.schema_version no soportada")
        if self.mode is not TemporalMode.TIMED_REFERENCE:
            raise SchemaValidationError(
                f"TaskTrajectory: modo {self.mode.value} sin implementar (solo TIMED_REFERENCE)"
            )
        if not isinstance(self.path, TaskPath):
            raise SchemaValidationError("TaskTrajectory.path: se espera TaskPath")
        if not isinstance(self.timing, TimePolynomial) or self.timing.dims != 1:
            raise SchemaValidationError("TaskTrajectory.timing: se espera s(t) escalar")
        if not isinstance(self.work, WorkRequirement):
            raise SchemaValidationError("TaskTrajectory.work: se espera WorkRequirement")
        object.__setattr__(
            self, "tolerance_m", ensure_non_negative(self.tolerance_m, "tolerance_m")
        )
        tol = _JOIN_TOLERANCES[0]
        low, high = self.timing.bounds(0)
        if low < -tol or high > 1.0 + tol:
            raise SchemaValidationError(f"s(t) sale de [0, 1]: [{low}, {high}]")
        if self.timing.bounds(0, order=1)[0] < 0.0:
            raise SchemaValidationError("s(t) decrece: retroceso sobre el camino no admitido")

    @property
    def t_start(self) -> float:
        return self.timing.t_start

    @property
    def t_end(self) -> float:
        return self.timing.t_end

    @property
    def domain(self) -> SpatialDomain:
        return self.path.domain

    def s_at(self, t: float) -> float:
        return min(max(float(self.timing.evaluate(t)[0]), 0.0), 1.0)

    def position(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`p_d`, `dp_d/dt` y `d2p_d/dt2` en `W` (regla de la cadena sobre `p(s)`)."""
        s = self.s_at(t)
        s_dot = float(self.timing.evaluate(t, 1)[0])
        s_ddot = float(self.timing.evaluate(t, 2)[0])
        p = np.array(self.path.evaluate(s, 0))
        dp = np.array(self.path.evaluate(s, 1))
        ddp = np.array(self.path.evaluate(s, 2))
        return p, dp * s_dot, ddp * s_dot**2 + dp * s_ddot

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "mode": self.mode.value,
            "clock_id": self.clock_id,
            "interval_s": [self.t_start, self.t_end],
            "timing": self.timing.to_dict(),
            "task_path": self.path.to_dict(),
            "task_path_hash": self.path.content_hash(),
            "map_hash": self.path.map_hash,
            "profile_hash": self.path.profile_hash,
            "domain": self.domain.value,
            "orientation": self.path.orientation.to_dict(),
            "work": self.work.to_dict(),
            "tolerance_m": self.tolerance_m,
            "provenance": self.provenance.value,
        }
        payload["content_hash"] = content_hash(payload)
        return payload

    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskTrajectory:
        path = TaskPath.from_dict(payload["task_path"])
        if path.content_hash() != payload["task_path_hash"]:
            raise SchemaValidationError("TaskTrajectory: hash del TaskPath no coincide")
        trajectory = cls(
            payload["trajectory_id"],
            path,
            TimePolynomial.from_dict(payload["timing"]),
            payload["clock_id"],
            WorkRequirement.from_dict(payload["work"]),
            payload["tolerance_m"],
            payload["mode"],
            payload["provenance"],
            payload["schema_version"],
        )
        if payload.get("content_hash") not in (None, trajectory.content_hash()):
            raise SchemaValidationError("TaskTrajectory: content_hash no coincide")
        return trajectory


class MotionSource(StrEnum):
    """Origen de `T_WV(t)`: referencia del escenario o prediccion."""

    REFERENCE = "reference"
    PREDICTED = "predicted"


@dataclass(frozen=True, slots=True, eq=False)
class VehicleMotion:
    """`T_WV(t)` como `(x_V, y_V, psi)` en `W`, con derivadas exactas.

    El avance del vehiculo entra una sola vez, aqui. Fuera del horizonte no se
    evalua. `position_bound_m`/`yaw_bound_rad` son cotas declaradas de incertidumbre
    del movimiento; no se suman de nuevo si la traza ya las incluye.
    """

    motion_id: str
    clock_id: str
    pose: TimePolynomial
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC
    position_bound_m: float = 0.0
    yaw_bound_rad: float = 0.0
    source: MotionSource = MotionSource.REFERENCE
    provenance: Provenance = Provenance.SYNTHETIC
    schema_version: int = TEMPORAL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "motion_id", _text(self.motion_id, "motion_id"))
        object.__setattr__(self, "clock_id", _text(self.clock_id, "clock_id"))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "source", MotionSource(self.source))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if self.schema_version != TEMPORAL_SCHEMA_VERSION:
            raise SchemaValidationError("VehicleMotion.schema_version no soportada")
        if self.domain not in TASK_DOMAINS:
            raise UnitMismatchError(f"VehicleMotion: dominio {self.domain.value} no metrico")
        if not isinstance(self.pose, TimePolynomial) or self.pose.dims != 3:
            raise SchemaValidationError(
                "VehicleMotion.pose: se espera (x, y, psi) de 3 componentes"
            )
        for name in ("position_bound_m", "yaw_bound_rad"):
            object.__setattr__(self, name, ensure_non_negative(getattr(self, name), name))

    @property
    def t_start(self) -> float:
        return self.pose.t_start

    @property
    def t_end(self) -> float:
        return self.pose.t_end

    def state(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(x, y, psi)` y sus dos primeras derivadas respecto de `t`."""
        return self.pose.evaluate(t, 0), self.pose.evaluate(t, 1), self.pose.evaluate(t, 2)

    def to_vehicle(
        self, t: float, p: np.ndarray, dp: np.ndarray, ddp: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Punto de `W` y sus derivadas expresados en `V`: `w = R(psi)^T (p - p_V)`.

        Incluye traslacion **y rotacion** del vehiculo:
        `w' = R^T d' - psi' J w`, `w'' = R^T d'' - psi'' J w - psi' J (w' + R^T d')`,
        con `d = p - p_V` y `J` la rotacion de +90 grados.
        """
        pose, pose_d, pose_dd = self.state(t)
        psi, psi_d, psi_dd = pose[2], pose_d[2], pose_dd[2]
        c, s = math.cos(psi), math.sin(psi)
        rt = np.array([[c, s], [-s, c]])
        j = np.array([[0.0, -1.0], [1.0, 0.0]])
        d = np.asarray(p) - pose[:2]
        dd = np.asarray(dp) - pose_d[:2]
        ddd = np.asarray(ddp) - pose_dd[:2]
        w = rt @ d
        w_d = rt @ dd - psi_d * (j @ w)
        w_dd = rt @ ddd - psi_dd * (j @ w) - psi_d * (j @ (w_d + rt @ dd))
        return w, w_d, w_dd

    def to_world(self, t: float, w: np.ndarray, theta_v: float) -> tuple[np.ndarray, float]:
        """Posicion y orientacion en `W` de un punto/orientacion dados en `V`."""
        pose = self.pose.evaluate(t)
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        return pose[:2] + rotation @ np.asarray(w), float(pose[2] + theta_v)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "motion_id": self.motion_id,
            "clock_id": self.clock_id,
            "frame": "T_WV: (x_V, y_V, psi) en W",
            "interval_s": [self.t_start, self.t_end],
            "pose": self.pose.to_dict(),
            "domain": self.domain.value,
            "position_bound_m": self.position_bound_m,
            "yaw_bound_rad": self.yaw_bound_rad,
            "source": self.source.value,
            "provenance": self.provenance.value,
        }
        payload["content_hash"] = content_hash(payload)
        return payload

    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VehicleMotion:
        motion = cls(
            payload["motion_id"],
            payload["clock_id"],
            TimePolynomial.from_dict(payload["pose"]),
            payload["domain"],
            payload["position_bound_m"],
            payload["yaw_bound_rad"],
            payload["source"],
            payload["provenance"],
            payload["schema_version"],
        )
        if payload.get("content_hash") not in (None, motion.content_hash()):
            raise SchemaValidationError("VehicleMotion: content_hash no coincide")
        return motion


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """Un rol de tarea servido por una herramienta con perfil identificado."""

    role: str
    tool_id: str
    tool_profile_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "role", _text(self.role, "RoleBinding.role"))
        object.__setattr__(self, "tool_id", _text(self.tool_id, "RoleBinding.tool_id"))
        value = str(self.tool_profile_hash)
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise SchemaValidationError("RoleBinding.tool_profile_hash: se espera SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "tool_id": self.tool_id,
            "tool_profile_hash": self.tool_profile_hash,
        }


@dataclass(frozen=True, slots=True)
class TaskBinding:
    """Rol -> herramienta. `policy` declara exclusividad, simultaneidad o secuencia.

    El compilador inicial solo resuelve un rol exclusivo; otras combinaciones son
    `UNSUPPORTED`, no factibles ni imposibles.
    """

    bindings: tuple[RoleBinding, ...]
    policy: str = "exclusive"

    def __post_init__(self) -> None:
        bindings = tuple(self.bindings)
        if not bindings or not all(isinstance(b, RoleBinding) for b in bindings):
            raise SchemaValidationError("TaskBinding: se exige al menos un RoleBinding")
        if self.policy not in ("exclusive", "simultaneous", "sequence"):
            raise SchemaValidationError("TaskBinding.policy: exclusive, simultaneous o sequence")
        object.__setattr__(self, "bindings", bindings)

    @property
    def is_single_exclusive(self) -> bool:
        return self.policy == "exclusive" and len(self.bindings) == 1

    def for_role(self, role: str) -> RoleBinding | None:
        matches = [b for b in self.bindings if b.role == role]
        return matches[0] if len(matches) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        payload = {"policy": self.policy, "bindings": [b.to_dict() for b in self.bindings]}
        payload["content_hash"] = content_hash(payload)
        return payload

    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]
