"""Contratos de tarea en suelo `W` (A1, orden A1/E2 §3; ADR 0001).

`TaskPath` es la salida geometrica de E2: una curva `p(s)` en `W`, con `s` en
`[0, 1]` **adimensional**. `s` no es tiempo ni la coordenada `y`, y las derivadas
respecto de `s` no son velocidades fisicas. La ley temporal (`TaskTrajectory`,
modo `TIMED_REFERENCE`) es un contrato posterior que referenciara este camino; su
esquema esta documentado en `docs/data_contracts.md` y no se materializa aun.

La curva es polinomica por tramos. Cada tramo cubre `[s_start, s_end]` y define
`x(u)`, `y(u)` con `u = (s - s_start) / (s_end - s_start)` en `[0, 1]` y
coeficientes en base de potencias **ascendente** (`c0 + c1 u + ... + c5 u^5`). De
ahi `dp/ds = (dp/du) / L` y `d2p/ds2 = (d2p/du2) / L^2`, con `L = s_end - s_start`.
La interpolacion es inequivoca: no hay puntos sueltos.

Solo admite dominios metricos de suelo (`ground_metric_synthetic`,
`ground_assumed`, `ground_metric_verified`). Pixeles (`image_px`) y unidades
graficas (`graphic_unscaled`) pertenecen a otros contratos de visualizacion y nunca
se compilan como metros.

No contiene configuraciones articulares, tiempos ni identidad de mecanismo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np

from .errors import FrameMismatchError, SchemaValidationError, UnitMismatchError
from .identity import content_hash
from .types import Frame, frozen_float_array
from .units import Provenance, SpatialDomain, ensure_finite, ensure_non_negative, unit_of_domain

__all__ = [
    "E2_SUPPORTED_ROLES",
    "POLYNOMIAL_COEFFICIENTS",
    "TASK_DOMAINS",
    "TASK_PATH_SCHEMA_VERSION",
    "OrientationPolicy",
    "OrientationRequirement",
    "PathSegment",
    "TaskPath",
]

TASK_PATH_SCHEMA_VERSION: Final[int] = 1
POLYNOMIAL_COEFFICIENTS: Final[int] = 6
"""Coeficientes por coordenada y tramo: grado maximo 5 (quintica)."""

TASK_DOMAINS: Final[frozenset[SpatialDomain]] = frozenset(
    {
        SpatialDomain.GROUND_METRIC_SYNTHETIC,
        SpatialDomain.GROUND_ASSUMED,
        SpatialDomain.GROUND_METRIC_VERIFIED,
    }
)
"""Dominios admitidos para coordenadas de tarea: suelo en metros (sintetico, supuesto
o verificado). Son modos separados: supuesto no es medido."""

E2_SUPPORTED_ROLES: Final[frozenset[str]] = frozenset({"pasada_principal"})
"""Roles que el perfil E2 sabe planificar y certificar. El contrato admite otros
nombres para A3 (extensible), pero E2 los rechaza como `UNSUPPORTED_FEATURE`: un
nombre de rol no acredita capacidad de elevacion ni otro papel de herramienta."""

_JOIN_POSITION_TOLERANCE_M: Final[float] = 1e-9
"""Discontinuidad maxima de posicion admitida entre tramos contiguos del contrato."""


def _hash_text(value: Any, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise SchemaValidationError(f"{name}: se espera un SHA-256 hexadecimal en minusculas")
    return text


def _required_text(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise SchemaValidationError(f"{name}: cadena vacia")
    return text


class OrientationPolicy(StrEnum):
    """Restriccion de orientacion de la herramienta en `W` (plan adaptado §4.2).

    `FREE` no significa cero. E2 solo implementa `FIXED`; `INTERVAL` y `FREE` se
    rechazan con `UNSUPPORTED_FEATURE`, nunca se aproximan en silencio.
    """

    FIXED = "FIXED"
    INTERVAL = "INTERVAL"
    FREE = "FREE"


@dataclass(frozen=True, slots=True)
class OrientationRequirement:
    """Orientacion pedida para el marco `B` respecto de `W`, en radianes (antihorario)."""

    policy: OrientationPolicy
    theta_rad: float | None = None
    interval_rad: tuple[float, float] | None = None
    frame: Frame = Frame.WORLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", OrientationPolicy(self.policy))
        object.__setattr__(self, "frame", Frame(self.frame))
        if self.frame is not Frame.WORLD:
            raise FrameMismatchError("OrientationRequirement: la orientacion se declara en W")
        if self.policy is OrientationPolicy.FIXED:
            if self.theta_rad is None or self.interval_rad is not None:
                raise SchemaValidationError("orientacion FIXED exige theta_rad y no intervalo")
            object.__setattr__(
                self, "theta_rad", ensure_finite(self.theta_rad, "OrientationRequirement.theta")
            )
        elif self.policy is OrientationPolicy.INTERVAL:
            if self.interval_rad is None or self.theta_rad is not None:
                raise SchemaValidationError("orientacion INTERVAL exige interval_rad y no theta")
            low, high = (
                ensure_finite(v, "OrientationRequirement.interval") for v in self.interval_rad
            )
            if high < low:
                raise SchemaValidationError("orientacion INTERVAL: limite superior < inferior")
            object.__setattr__(self, "interval_rad", (low, high))
        elif self.theta_rad is not None or self.interval_rad is not None:
            raise SchemaValidationError("orientacion FREE no admite theta ni intervalo")

    @classmethod
    def fixed(cls, theta_rad: float) -> OrientationRequirement:
        return cls(policy=OrientationPolicy.FIXED, theta_rad=theta_rad)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "theta_rad": self.theta_rad,
            "interval_rad": list(self.interval_rad) if self.interval_rad is not None else None,
            "frame": self.frame.value,
            "unit": "rad",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> OrientationRequirement:
        interval = payload.get("interval_rad")
        return cls(
            policy=payload["policy"],
            theta_rad=payload.get("theta_rad"),
            interval_rad=tuple(interval) if interval is not None else None,
            frame=payload.get("frame", Frame.WORLD.value),
        )


def _poly_eval(coefficients: np.ndarray, u: float, order: int) -> float:
    """Derivada `order` (0..5) en `u` de un polinomio en base ascendente."""
    value = 0.0
    for power in range(order, POLYNOMIAL_COEFFICIENTS):
        factor = math.perm(power, order)
        value += factor * float(coefficients[power]) * u ** (power - order)
    return value


@dataclass(frozen=True, slots=True, eq=False)
class PathSegment:
    """Tramo polinomico `[s_start, s_end]` de un `TaskPath`.

    `x_coefficients` e `y_coefficients` son 6 coeficientes en base de potencias
    ascendente sobre el parametro local `u` en `[0, 1]`; copias propias no
    modificables.
    """

    s_start: float
    s_end: float
    x_coefficients: np.ndarray
    y_coefficients: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "s_start", ensure_finite(self.s_start, "PathSegment.s_start"))
        object.__setattr__(self, "s_end", ensure_finite(self.s_end, "PathSegment.s_end"))
        if not self.s_end > self.s_start:
            raise SchemaValidationError(
                f"PathSegment: intervalo de s vacio o invertido [{self.s_start}, {self.s_end}]"
            )
        for name in ("x_coefficients", "y_coefficients"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != (POLYNOMIAL_COEFFICIENTS,):
                raise SchemaValidationError(
                    f"PathSegment.{name}: se exigen {POLYNOMIAL_COEFFICIENTS} coeficientes, "
                    f"se recibio forma {values.shape}"
                )
            object.__setattr__(self, name, frozen_float_array(values, f"PathSegment.{name}"))

    @property
    def length_s(self) -> float:
        return self.s_end - self.s_start

    def local_u(self, s: float) -> float:
        return (s - self.s_start) / self.length_s

    def derivative_u(self, u: float, order: int = 0) -> tuple[float, float]:
        """Derivada respecto del parametro local `u`."""
        return (
            _poly_eval(self.x_coefficients, u, order),
            _poly_eval(self.y_coefficients, u, order),
        )

    def derivative_s(self, u: float, order: int = 0) -> tuple[float, float]:
        """Derivada respecto de `s` global en el parametro local `u`."""
        dx, dy = self.derivative_u(u, order)
        scale = self.length_s**order
        return dx / scale, dy / scale

    def to_dict(self) -> dict[str, Any]:
        return {
            "s_start": self.s_start,
            "s_end": self.s_end,
            "x_coefficients": self.x_coefficients.tolist(),
            "y_coefficients": self.y_coefficients.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PathSegment:
        return cls(
            s_start=payload["s_start"],
            s_end=payload["s_end"],
            x_coefficients=payload["x_coefficients"],
            y_coefficients=payload["y_coefficients"],
        )


@dataclass(frozen=True, slots=True, eq=False)
class TaskPath:
    """Camino geometrico de tarea `p(s)`, `s` en `[0, 1]`, en `W` y metros.

    Invariantes de construccion: marco `W`; dominio metrico de suelo; tramos que
    particionan `[0, 1]` sin huecos ni solapes (`s_end` de un tramo es exactamente el
    `s_start` del siguiente); posicion continua en las uniones (`<= 1e-9 m`);
    identidad de perfil y mapa por SHA-256. La continuidad C1/C2 declarada la
    comprueba el validador, no este contrato.

    Para E2 la tolerancia de tarea es cero: se valida y se ejecutaria la curva
    exacta entregada.
    """

    path_id: str
    domain: SpatialDomain
    segments: tuple[PathSegment, ...]
    orientation: OrientationRequirement
    profile_id: str
    profile_hash: str
    map_id: str
    map_hash: str
    role: str = "pasada_principal"
    working_contact: bool = True
    task_tolerance_m: float = 0.0
    continuity_declared: str = "C2"
    frame: Frame = Frame.WORLD
    provenance: Provenance = Provenance.SYNTHETIC
    schema_version: int = TASK_PATH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "path_id", _required_text(self.path_id, "TaskPath.path_id"))
        object.__setattr__(self, "frame", Frame(self.frame))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if self.schema_version != TASK_PATH_SCHEMA_VERSION:
            raise SchemaValidationError(
                f"TaskPath.schema_version {self.schema_version!r} no soportada"
            )
        if self.frame is not Frame.WORLD:
            raise FrameMismatchError(
                f"TaskPath: el camino de tarea se expresa en W, se recibio {self.frame.value}"
            )
        if self.domain not in TASK_DOMAINS:
            raise UnitMismatchError(
                f"TaskPath: dominio {self.domain.value} no admitido; las coordenadas de tarea "
                "son metros de suelo (pixeles y unidades graficas usan otro contrato)"
            )
        if not isinstance(self.orientation, OrientationRequirement):
            raise SchemaValidationError("TaskPath.orientation: se espera OrientationRequirement")
        object.__setattr__(self, "profile_id", _required_text(self.profile_id, "profile_id"))
        object.__setattr__(self, "map_id", _required_text(self.map_id, "map_id"))
        object.__setattr__(self, "profile_hash", _hash_text(self.profile_hash, "profile_hash"))
        object.__setattr__(self, "map_hash", _hash_text(self.map_hash, "map_hash"))
        object.__setattr__(self, "role", _required_text(self.role, "TaskPath.role"))
        if not isinstance(self.working_contact, bool):
            raise SchemaValidationError("TaskPath.working_contact: se exige un booleano")
        object.__setattr__(
            self,
            "task_tolerance_m",
            ensure_non_negative(self.task_tolerance_m, "TaskPath.task_tolerance_m"),
        )
        if self.continuity_declared not in ("C0", "C1", "C2"):
            raise SchemaValidationError("TaskPath.continuity_declared: C0, C1 o C2")
        segments = tuple(self.segments) if not isinstance(self.segments, (str, bytes)) else ()
        if not segments or not all(isinstance(seg, PathSegment) for seg in segments):
            raise SchemaValidationError("TaskPath.segments: se exige al menos un PathSegment")
        object.__setattr__(self, "segments", segments)
        self._check_partition()

    def _check_partition(self) -> None:
        if self.segments[0].s_start != 0.0 or self.segments[-1].s_end != 1.0:
            raise SchemaValidationError("TaskPath: los tramos deben cubrir exactamente s en [0, 1]")
        for index in range(len(self.segments) - 1):
            left, right = self.segments[index], self.segments[index + 1]
            if left.s_end != right.s_start:
                raise SchemaValidationError(
                    f"TaskPath: hueco o solape de s entre los tramos {index} y {index + 1}"
                )
            end = np.array(left.derivative_u(1.0))
            start = np.array(right.derivative_u(0.0))
            if float(np.max(np.abs(end - start))) > _JOIN_POSITION_TOLERANCE_M:
                raise SchemaValidationError(
                    f"TaskPath: discontinuidad de posicion entre los tramos {index} y {index + 1}"
                )

    # --- evaluacion -----------------------------------------------------------

    @property
    def unit(self) -> str:
        return unit_of_domain(self.domain).value

    def segment_at(self, s: float) -> tuple[int, PathSegment]:
        """Tramo que contiene `s` (intervalos semiabiertos; el ultimo incluye 1)."""
        s = ensure_finite(s, "s")
        if not 0.0 <= s <= 1.0:
            raise SchemaValidationError(f"s={s} fuera de [0, 1]")
        for index, segment in enumerate(self.segments):
            if s < segment.s_end or index == len(self.segments) - 1:
                return index, segment
        raise AssertionError("particion de s incoherente")  # pragma: no cover

    def evaluate(self, s: float, order: int = 0) -> tuple[float, float]:
        """`p(s)` (orden 0) o su derivada respecto de `s` (orden 1, 2...)."""
        _, segment = self.segment_at(s)
        return segment.derivative_s(segment.local_u(s), order)

    @property
    def start_point(self) -> tuple[float, float]:
        return self.segments[0].derivative_u(0.0)

    @property
    def end_point(self) -> tuple[float, float]:
        return self.segments[-1].derivative_u(1.0)

    # --- identidad y serializacion -------------------------------------------

    def identity_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload.pop("content_hash", None)
        return payload

    def content_hash(self) -> str:
        return content_hash(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "path_id": self.path_id,
            "frame": self.frame.value,
            "domain": self.domain.value,
            "unit": self.unit,
            "provenance": self.provenance.value,
            "parameter": "s adimensional en [0, 1]; no es tiempo",
            "interpolation": "polinomio por tramos, base de potencias ascendente en u local",
            "continuity_declared": self.continuity_declared,
            "segments": [segment.to_dict() for segment in self.segments],
            "orientation": self.orientation.to_dict(),
            "profile_id": self.profile_id,
            "profile_hash": self.profile_hash,
            "map_id": self.map_id,
            "map_hash": self.map_hash,
            "role": self.role,
            "working_contact": self.working_contact,
            "task_tolerance_m": self.task_tolerance_m,
        }
        payload["content_hash"] = content_hash(payload)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskPath:
        """Reconstruye y comprueba la identidad declarada, si viene incluida."""
        try:
            path = cls(
                path_id=payload["path_id"],
                domain=payload["domain"],
                segments=tuple(PathSegment.from_dict(item) for item in payload["segments"]),
                orientation=OrientationRequirement.from_dict(payload["orientation"]),
                profile_id=payload["profile_id"],
                profile_hash=payload["profile_hash"],
                map_id=payload["map_id"],
                map_hash=payload["map_hash"],
                role=payload["role"],
                working_contact=payload["working_contact"],
                task_tolerance_m=payload["task_tolerance_m"],
                continuity_declared=payload["continuity_declared"],
                frame=payload["frame"],
                provenance=payload["provenance"],
                schema_version=payload["schema_version"],
            )
        except KeyError as exc:
            raise SchemaValidationError(f"TaskPath: falta el campo {exc}") from exc
        declared = payload.get("content_hash")
        if declared is not None and declared != path.content_hash():
            raise SchemaValidationError("TaskPath: content_hash no coincide con el contenido")
        return path

    @classmethod
    def from_segments(
        cls, segments: Sequence[PathSegment], **fields: Any
    ) -> TaskPath:  # pragma: no cover - azucar
        return cls(segments=tuple(segments), **fields)
