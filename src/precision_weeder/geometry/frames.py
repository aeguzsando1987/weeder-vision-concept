"""Marcos de coordenadas y transformaciones rigidas 2D (`docs/technical_plan.md` §3.2).

Convenciones, declaradas una sola vez y nunca invertidas de forma implicita:

- `I` (imagen): `(u, v)` en pixeles, origen arriba a la izquierda, `u` hacia la
  derecha, `v` hacia **abajo**.
- `W` (suelo/mundo): `x` lateral a la derecha del sentido de avance, `y`
  longitudinal hacia **delante**; marco fijo durante cada segmento local.
- `V` (vehiculo): su origen y orientacion se expresan respecto a `W` mediante
  `T_WV(t)`.
- `B` (herramienta): origen en el pivote; su poligono local se lleva a `W`
  mediante `T_WV(t) * T_VB(q(t))`. El avance del tractor entra una sola vez, a
  traves de `T_WV`.

El desplazamiento aparente de las plantas "hacia arriba" en los videos
(`dv < 0`) no tiene significado fisico por si solo: depende del montaje de la
camara. `advance_sign_under_declared_mounting` solo devuelve un signo cuando el
montaje se declara explicitamente, y aun asi no da escala ni velocidad fisica
(R7). Vease tambien `metric_ground_displacement_from_image`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry.base import BaseGeometry

from ..core.errors import FrameMismatchError, MissingCalibrationError, SchemaValidationError
from ..core.types import CalibrationReference, Frame, Point2
from ..core.units import (
    TRANSFORM_TOLERANCE,
    SpatialDomain,
    ensure_finite,
    ensure_finite_array,
    require_same_domain,
)

__all__ = [
    "ImageVAxisOnGround",
    "Pose2",
    "Transform2D",
    "advance_sign_under_declared_mounting",
    "metric_ground_displacement_from_image",
    "tool_from_vehicle",
    "world_from_vehicle",
]


@dataclass(frozen=True, slots=True)
class Pose2:
    """Pose plana `(x, y, theta)` de un marco hijo expresada en un marco padre."""

    x: float
    y: float
    theta: float
    parent: Frame
    child: Frame
    domain: SpatialDomain

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", ensure_finite(self.x, "Pose2.x"))
        object.__setattr__(self, "y", ensure_finite(self.y, "Pose2.y"))
        object.__setattr__(self, "theta", ensure_finite(self.theta, "Pose2.theta"))
        object.__setattr__(self, "parent", Frame(self.parent))
        object.__setattr__(self, "child", Frame(self.child))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        if self.parent is self.child:
            raise FrameMismatchError(
                f"Pose2: parent y child no pueden ser el mismo marco ({self.parent.value})"
            )

    def as_transform(self) -> Transform2D:
        """Transformacion rigida `T_parent_child` equivalente."""
        return Transform2D(
            parent=self.parent,
            child=self.child,
            tx=self.x,
            ty=self.y,
            theta=self.theta,
            domain=self.domain,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "theta_rad": self.theta,
            "parent": self.parent.value,
            "child": self.child.value,
            "domain": self.domain.value,
        }


@dataclass(frozen=True, slots=True)
class Transform2D:
    """Transformacion rigida `T_parent_child`: rota por `theta` y traslada `(tx, ty)`.

    Aplicada a un punto expresado en el marco `child`, devuelve el punto en el
    marco `parent`. Solo rotacion y traslacion: no hay escalado ni reflexion, de
    modo que no puede invertir ejes de forma silenciosa.
    """

    parent: Frame
    child: Frame
    tx: float = 0.0
    ty: float = 0.0
    theta: float = 0.0
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC

    def __post_init__(self) -> None:
        object.__setattr__(self, "parent", Frame(self.parent))
        object.__setattr__(self, "child", Frame(self.child))
        object.__setattr__(self, "tx", ensure_finite(self.tx, "Transform2D.tx"))
        object.__setattr__(self, "ty", ensure_finite(self.ty, "Transform2D.ty"))
        object.__setattr__(self, "theta", ensure_finite(self.theta, "Transform2D.theta"))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))

    # --- constructores ------------------------------------------------------

    @classmethod
    def identity(
        cls,
        frame: Frame,
        domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
    ) -> Transform2D:
        """Identidad de un marco sobre si mismo."""
        return cls(parent=frame, child=frame, tx=0.0, ty=0.0, theta=0.0, domain=domain)

    @classmethod
    def from_translation(
        cls,
        parent: Frame,
        child: Frame,
        tx: float,
        ty: float,
        domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
    ) -> Transform2D:
        return cls(parent=parent, child=child, tx=tx, ty=ty, theta=0.0, domain=domain)

    @classmethod
    def from_rotation(
        cls,
        parent: Frame,
        child: Frame,
        theta: float,
        domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
    ) -> Transform2D:
        return cls(parent=parent, child=child, tx=0.0, ty=0.0, theta=theta, domain=domain)

    # --- algebra ------------------------------------------------------------

    @property
    def is_identity_frames(self) -> bool:
        return self.parent is self.child

    def rotation_matrix(self) -> np.ndarray:
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        return np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float64)

    def matrix(self) -> np.ndarray:
        """Matriz homogenea 3x3 de la transformacion."""
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :2] = self.rotation_matrix()
        matrix[:2, 2] = (self.tx, self.ty)
        return matrix

    def inverse(self) -> Transform2D:
        """`T_child_parent`, la transformacion inversa exacta."""
        rotation_t = self.rotation_matrix().T
        translation = -rotation_t @ np.array([self.tx, self.ty], dtype=np.float64)
        return Transform2D(
            parent=self.child,
            child=self.parent,
            tx=float(translation[0]),
            ty=float(translation[1]),
            theta=-self.theta,
            domain=self.domain,
        )

    def compose(self, other: Transform2D) -> Transform2D:
        """`self @ other`: requiere `self.child == other.parent`.

        Raises:
            FrameMismatchError: si la cadena de marcos no encaja.
            UnitMismatchError: si los dominios espaciales difieren.
        """
        if not isinstance(other, Transform2D):
            raise SchemaValidationError("Transform2D.compose: se espera otro Transform2D")
        if self.child is not other.parent:
            raise FrameMismatchError(
                f"composicion invalida: T_{self.parent.value}{self.child.value} @ "
                f"T_{other.parent.value}{other.child.value}; "
                f"se exige child={self.child.value} == parent={other.parent.value}"
            )
        domain = require_same_domain(self.domain, other.domain, context="Transform2D.compose")
        product = self.matrix() @ other.matrix()
        return Transform2D(
            parent=self.parent,
            child=other.child,
            tx=float(product[0, 2]),
            ty=float(product[1, 2]),
            theta=_wrap_angle(self.theta + other.theta),
            domain=domain,
        )

    def __matmul__(self, other: Transform2D) -> Transform2D:
        return self.compose(other)

    # --- aplicacion ---------------------------------------------------------

    def apply_array(self, points: Sequence[Sequence[float]]) -> np.ndarray:
        """Transforma puntos `(n, 2)` del marco hijo al marco padre."""
        array = ensure_finite_array(points, "Transform2D.apply_array")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != 2:
            raise SchemaValidationError(
                f"Transform2D.apply_array: se espera (n, 2), se recibio {array.shape}"
            )
        return array @ self.rotation_matrix().T + np.array([self.tx, self.ty], dtype=np.float64)

    def apply_point(self, point: Point2) -> Point2:
        """Transforma un `Point2`, verificando marco y dominio.

        Raises:
            FrameMismatchError: si el punto no esta en el marco hijo.
        """
        if not isinstance(point, Point2):
            raise SchemaValidationError("Transform2D.apply_point: se espera Point2")
        if point.frame is not self.child:
            raise FrameMismatchError(
                f"se esperaba un punto en el marco {self.child.value}, "
                f"se recibio {point.frame.value}"
            )
        require_same_domain(self.domain, point.domain, context="Transform2D.apply_point")
        transformed = self.apply_array([[point.x, point.y]])[0]
        return Point2(
            x=float(transformed[0]),
            y=float(transformed[1]),
            frame=self.parent,
            domain=point.domain,
        )

    def apply_geometry(self, geometry: BaseGeometry) -> BaseGeometry:
        """Transforma una geometria de Shapely preservando su forma."""
        if not isinstance(geometry, BaseGeometry):
            raise SchemaValidationError("Transform2D.apply_geometry: se espera geometria Shapely")
        cos_t, sin_t = math.cos(self.theta), math.sin(self.theta)
        return affine_transform(geometry, [cos_t, -sin_t, sin_t, cos_t, self.tx, self.ty])

    def allclose(self, other: Transform2D, tolerance: float = TRANSFORM_TOLERANCE) -> bool:
        """Comparacion numerica de dos transformaciones del mismo par de marcos y dominio.

        Transformaciones de dominios distintos (por ejemplo metros y pixeles) nunca
        son equivalentes, aunque sus numeros coincidan (R3).
        """
        if not isinstance(other, Transform2D):
            raise SchemaValidationError("Transform2D.allclose: se espera otro Transform2D")
        if self.parent is not other.parent or self.child is not other.child:
            return False
        if self.domain is not other.domain:
            return False
        angle_difference = abs(_wrap_angle(self.theta - other.theta))
        return bool(
            abs(self.tx - other.tx) <= tolerance
            and abs(self.ty - other.ty) <= tolerance
            and angle_difference <= tolerance
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": f"T_{self.parent.value}{self.child.value}",
            "parent": self.parent.value,
            "child": self.child.value,
            "tx": self.tx,
            "ty": self.ty,
            "theta_rad": self.theta,
            "domain": self.domain.value,
        }


def _wrap_angle(angle: float) -> float:
    """Normaliza un angulo a `(-pi, pi]`."""
    wrapped = math.remainder(angle, 2.0 * math.pi)
    return math.pi if wrapped == -math.pi else wrapped


def world_from_vehicle(
    x: float,
    y: float,
    heading: float = 0.0,
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
) -> Transform2D:
    """`T_WV`: pose del vehiculo en el suelo. El avance del tractor entra solo aqui."""
    return Transform2D(
        parent=Frame.WORLD, child=Frame.VEHICLE, tx=x, ty=y, theta=heading, domain=domain
    )


def tool_from_vehicle(
    x: float,
    y: float,
    theta: float = 0.0,
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
) -> Transform2D:
    """`T_VB`: pose del pivote de la herramienta respecto al vehiculo, `q(t)`."""
    return Transform2D(
        parent=Frame.VEHICLE, child=Frame.TOOL, tx=x, ty=y, theta=theta, domain=domain
    )


class ImageVAxisOnGround(StrEnum):
    """Declaracion de montaje: hacia donde apunta `+v` de la imagen sobre el suelo.

    - `backward`: `+v` apunta hacia `-y_W` (la parte superior de la imagen mira
      hacia delante).
    - `forward`: `+v` apunta hacia `+y_W` (la parte superior de la imagen mira
      hacia atras, por ejemplo una camara girada 180 grados).
    - `unknown`: montaje no declarado; no se puede deducir direccion fisica.

    Ninguna opcion se asume por defecto: el desplazamiento aparente "hacia arriba"
    de los videos existentes solo implica avance en `+y_W` si se declara
    `forward`.

    Una camara cuyo eje `v` no sea longitudinal no tiene valor aqui: la regla de
    signo no le es aplicable.
    """

    BACKWARD = "backward"
    FORWARD = "forward"
    UNKNOWN = "unknown"


def advance_sign_under_declared_mounting(
    delta_v_px: float,
    v_axis_on_ground: ImageVAxisOnGround,
) -> int:
    """Signo de avance del vehiculo bajo un montaje de camara **declarado** (R7).

    Supuestos explicitos: escena estatica, flujo vertical dominado por la
    traslacion del vehiculo (sin rotacion, movimiento propio de las plantas ni
    paralaje relevante) y eje `v` alineado con el eje longitudinal segun
    `v_axis_on_ground`. Con `forward`, plantas que suben en pantalla
    (`delta_v_px < 0`) implican avance en `+y_W`; con `backward`, el mismo flujo
    implica avance en `-y_W`.

    Returns:
        `+1` (avance hacia `+y_W`), `-1` (hacia `-y_W`) o `0` (sin flujo).

    Raises:
        MissingCalibrationError: si el montaje es `unknown`; un montaje no
            declarado no produce una direccion fisica.

    Note:
        Es un signo bajo hipotesis declaradas, no una direccion certificada ni una
        velocidad; no autoriza afirmar anticipacion fisica (§1.1).
    """
    delta = ensure_finite(delta_v_px, "delta_v_px")
    mounting = ImageVAxisOnGround(v_axis_on_ground)
    if mounting is ImageVAxisOnGround.UNKNOWN:
        raise MissingCalibrationError(
            "montaje de camara desconocido: el signo del flujo aparente no determina la "
            "direccion de avance sin declarar la orientacion del eje v sobre el suelo"
        )
    if delta == 0.0:
        return 0
    image_sign = 1 if delta > 0.0 else -1
    # El suelo se desplaza en imagen en sentido opuesto al avance del vehiculo.
    ground_flow_sign = -image_sign if mounting is ImageVAxisOnGround.BACKWARD else image_sign
    return -ground_flow_sign


def metric_ground_displacement_from_image(
    delta_uv_px: Sequence[float],
    calibration: CalibrationReference,
) -> np.ndarray:
    """Desplazamiento metrico en el suelo a partir de un desplazamiento en imagen.

    Raises:
        MissingCalibrationError: siempre que la calibracion no este verificada; una
            vista estabilizada sin escala no es metrica (§3.2). La conversion
            efectiva se implementa en E4 con la homografia
            `H_ground_from_undistorted_image`.

    Note:
        Bajo una homografia proyectiva un mismo delta de pixeles no determina un
        desplazamiento metrico en cualquier posicion: la API definitiva de E4
        necesitara un punto de anclaje o ambos puntos transformados.
    """
    delta = ensure_finite_array(delta_uv_px, "delta_uv_px")
    if delta.shape != (2,):
        raise SchemaValidationError(f"delta_uv_px: se espera (2,), se recibio {delta.shape}")
    if not isinstance(calibration, CalibrationReference):
        raise SchemaValidationError("calibration: se espera CalibrationReference")
    calibration.require_verified("conversion imagen -> suelo metrico")
    raise NotImplementedError(
        "la conversion metrica imagen -> suelo requiere "
        "H_ground_from_undistorted_image, entregada en E4"
    )
