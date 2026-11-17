"""Contrato `MechanismModel` (A3/E3, orden §3.1; adaptacion §4.4).

Un modelo declara su esquema de configuracion (juntas, tipo, unidades, topes y
limites de referencia), sus cuerpos de espesor finito y los pares estructurales que
se permiten en contacto. `forward(q)` devuelve la pose en `V` de **cada** cuerpo por
`body_id` (y `tool_id` cuando es herramienta): validar brazo-cultivo exige eslabones,
no solo la cuchilla. La dimension de `q` es independiente del numero de herramientas.

El planificador E2 no importa este modulo; el compilador lo consume.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ..core.compilation import CompileStatus, InfeasibilityCode
from ..core.errors import SchemaValidationError
from ..core.identity import content_hash
from ..core.types import validate_polygon_points
from ..core.units import ensure_finite, ensure_finite_array

__all__ = [
    "BodyPose",
    "BodySpec",
    "ConfigurationCheck",
    "ConfigurationSchema",
    "InverseResult",
    "JointKind",
    "JointSpec",
    "MechanismModel",
    "place_polygon",
]


class JointKind(StrEnum):
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


@dataclass(frozen=True, slots=True)
class JointSpec:
    """Junta activa: topes de posicion y limites de la referencia de velocidad/aceleracion."""

    name: str
    kind: JointKind
    lower: float
    upper: float
    velocity_limit: float
    acceleration_limit: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", JointKind(self.kind))
        for name in ("lower", "upper", "velocity_limit", "acceleration_limit"):
            object.__setattr__(self, name, ensure_finite(getattr(self, name), f"joint.{name}"))
        if not self.lower < self.upper:
            raise SchemaValidationError(f"junta {self.name}: tope inferior >= superior")
        if self.velocity_limit <= 0.0 or self.acceleration_limit <= 0.0:
            raise SchemaValidationError(f"junta {self.name}: limites deben ser positivos")

    @property
    def unit(self) -> str:
        return "rad" if self.kind is JointKind.REVOLUTE else "m"

    def to_dict(self) -> dict[str, Any]:
        unit = self.unit
        return {
            "name": self.name,
            "kind": self.kind.value,
            "unit": unit,
            "lower": self.lower,
            "upper": self.upper,
            "velocity_limit": self.velocity_limit,
            "velocity_unit": f"{unit}/s",
            "acceleration_limit": self.acceleration_limit,
            "acceleration_unit": f"{unit}/s^2",
        }


@dataclass(frozen=True, slots=True)
class ConfigurationSchema:
    """Juntas en orden de `q`; todas activas (sin pasivas ni cadenas cerradas en A3)."""

    joints: tuple[JointSpec, ...]
    assembly_modes: tuple[str, ...] = ("unico",)

    @property
    def dimension(self) -> int:
        return len(self.joints)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joints": [joint.to_dict() for joint in self.joints],
            "assembly_modes": list(self.assembly_modes),
            "passive_joints": [],
            "closure_constraints": [],
        }


@dataclass(frozen=True, slots=True, eq=False)
class BodySpec:
    """Cuerpo rigido: poligono de colision en su marco local; corte si es herramienta."""

    body_id: str
    collision_polygon: np.ndarray
    tool_id: str | None = None
    cutting_polygon: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "collision_polygon",
            validate_polygon_points(self.collision_polygon, f"{self.body_id}.collision"),
        )
        if self.cutting_polygon is not None:
            object.__setattr__(
                self,
                "cutting_polygon",
                validate_polygon_points(self.cutting_polygon, f"{self.body_id}.cutting"),
            )
        if (self.tool_id is None) != (self.cutting_polygon is None):
            raise SchemaValidationError(f"{self.body_id}: herramienta exige huella de corte")

    def to_dict(self) -> dict[str, Any]:
        return {
            "body_id": self.body_id,
            "tool_id": self.tool_id,
            "collision_polygon": self.collision_polygon.tolist(),
            "cutting_polygon": None
            if self.cutting_polygon is None
            else self.cutting_polygon.tolist(),
        }


def place_polygon(points: np.ndarray, x: float, y: float, theta: float) -> np.ndarray:
    """Lleva un anillo local al marco padre con la pose `(x, y, theta)`."""
    c, s = math.cos(theta), math.sin(theta)
    return np.asarray(points) @ np.array([[c, s], [-s, c]]) + np.array([x, y])


@dataclass(frozen=True, slots=True)
class BodyPose:
    """Pose de un cuerpo en `V`: origen de su marco local y orientacion (rad)."""

    body_id: str
    x: float
    y: float
    theta: float
    tool_id: str | None = None

    def in_world(self, vehicle: np.ndarray) -> tuple[float, float, float]:
        """Pose en `W` dada `T_WV = (x_V, y_V, psi)`: `T_WB = T_WV T_VB`."""
        c, s = math.cos(vehicle[2]), math.sin(vehicle[2])
        return (
            float(vehicle[0] + c * self.x - s * self.y),
            float(vehicle[1] + s * self.x + c * self.y),
            float(vehicle[2] + self.theta),
        )


@dataclass(frozen=True, slots=True)
class ConfigurationCheck:
    """Resultado de `validate_configuration`: violaciones `(codigo, junta, valor, limite)`."""

    valid: bool
    violations: tuple[tuple[str, str, float, float], ...] = ()


@dataclass(frozen=True, slots=True)
class InverseResult:
    """Inversa puntual. `INFEASIBLE_PROVEN` lleva certificado analitico en `detail`."""

    status: CompileStatus
    q: tuple[float, ...] | None = None
    code: InfeasibilityCode | None = None
    value: float | None = None
    limit: float | None = None
    residual: float | None = None
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status is CompileStatus.FEASIBLE_CANDIDATE


class MechanismModel(ABC):
    """Modelo mecanico intercambiable: esquema, cuerpos, FK y (opcional) inversa."""

    model_id: str
    version: int

    @property
    @abstractmethod
    def configuration_schema(self) -> ConfigurationSchema: ...

    @property
    @abstractmethod
    def bodies(self) -> tuple[BodySpec, ...]: ...

    @property
    def allowed_pairs(self) -> frozenset[frozenset[str]]:
        """Pares estructurales que pueden tocarse; cualquier otro par se comprueba."""
        return frozenset()

    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """Parametros que definen el modelo (entran en su identidad)."""

    @abstractmethod
    def forward(self, q: np.ndarray, assembly_mode: str = "unico") -> dict[str, BodyPose]:
        """Pose en `V` de cada cuerpo por `body_id`."""

    def inverse(
        self, target_v: np.ndarray, seed: np.ndarray, theta_v: float | None = None
    ) -> InverseResult:
        """Inversa de la herramienta; un modelo sin ella responde `UNSUPPORTED`."""
        return InverseResult(CompileStatus.UNSUPPORTED, detail="inversa no implementada")

    @property
    def tool_ids(self) -> tuple[str, ...]:
        return tuple(body.tool_id for body in self.bodies if body.tool_id is not None)

    def body(self, body_id: str) -> BodySpec:
        for body in self.bodies:
            if body.body_id == body_id:
                return body
        raise SchemaValidationError(f"{self.model_id}: cuerpo {body_id} inexistente")

    def tool_body(self, tool_id: str) -> BodySpec:
        for body in self.bodies:
            if body.tool_id == tool_id:
                return body
        raise SchemaValidationError(f"{self.model_id}: herramienta {tool_id} inexistente")

    def as_configuration(self, q: Any) -> np.ndarray:
        """`q` como vector finito de la dimension declarada."""
        array = ensure_finite_array(np.atleast_1d(np.asarray(q, dtype=np.float64)), "q")
        if array.shape != (self.configuration_schema.dimension,):
            raise SchemaValidationError(
                f"{self.model_id}: q de forma {array.shape}, se exige "
                f"({self.configuration_schema.dimension},)"
            )
        return array

    def validate_configuration(self, q: Any, assembly_mode: str = "unico") -> ConfigurationCheck:
        """Topes de posicion de cada junta (comparacion exacta, sin holgura)."""
        values = self.as_configuration(q)
        if assembly_mode not in self.configuration_schema.assembly_modes:
            raise SchemaValidationError(f"{self.model_id}: modo de ensamblaje {assembly_mode}")
        violations = []
        for value, joint in zip(values, self.configuration_schema.joints, strict=True):
            if value < joint.lower:
                violations.append(("JOINT_LIMIT", joint.name, float(value), joint.lower))
            elif value > joint.upper:
                violations.append(("JOINT_LIMIT", joint.name, float(value), joint.upper))
        return ConfigurationCheck(not violations, tuple(violations))

    def identity_payload(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "configuration_schema": self.configuration_schema.to_dict(),
            "bodies": [body.to_dict() for body in self.bodies],
            "allowed_pairs": sorted(sorted(pair) for pair in self.allowed_pairs),
            "parameters": self.parameters(),
        }

    def content_hash(self) -> str:
        return content_hash(self.identity_payload())
