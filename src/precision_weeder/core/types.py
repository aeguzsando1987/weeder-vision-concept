"""Contratos de datos tipados e independientes de vision (`docs/technical_plan.md` §3.2).

Los contratos de esta etapa (E1) cubren marcos, puntos, incertidumbre, plantas de
referencia, exclusiones circulares y el corredor. Los contratos que dependen de
video, calibracion o percepcion (`FramePacket`, `StemObservation`, `Calibration`,
`PlantTrack`, `LocalMap`, `PlanResult`, `ValidationReport`) estan especificados en
`docs/data_contracts.md` y se materializan en su etapa, para no crear modulos
vacios que simulen avance (§4).

Todos los contratos son `frozen`: validan en construccion y luego son inmutables.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from shapely.geometry import Polygon

from .errors import (
    FrameMismatchError,
    InvalidPolygonError,
    MissingCalibrationError,
    SchemaValidationError,
    UncertaintyBudgetError,
    UnitMismatchError,
)
from .units import (
    LIMIT_COMPARISON_TOLERANCE,
    LengthUnit,
    Provenance,
    SpatialDomain,
    ensure_finite,
    ensure_finite_array,
    ensure_non_negative,
    ensure_positive,
    is_metric,
    require_same_domain,
    unit_of_domain,
)

__all__ = [
    "MIN_CIRCLE_SEGMENTS",
    "AbstentionReason",
    "CalibrationReference",
    "CalibrationStatus",
    "CircularExclusion",
    "Corridor",
    "Frame",
    "InflationTarget",
    "OrientationMode",
    "PlanStatus",
    "PlantReference",
    "Point2",
    "SimulationState",
    "UncertaintyBound",
    "UncertaintyPolicy",
    "circumscribed_radial_excess",
    "covariance_to_bound",
    "exterior_circle_polygon",
    "frozen_float_array",
    "minimum_segments_for_excess",
    "validate_polygon_points",
]


MIN_CIRCLE_SEGMENTS: Final[int] = 8
"""Minimo de segmentos de una aproximacion poligonal de circulo."""


class Frame(StrEnum):
    """Marcos de coordenadas del proyecto (§3.2).

    - `W`: suelo/mundo. `x` lateral a la derecha del sentido de avance, `y`
      longitudinal hacia delante. Fijo durante cada segmento local.
    - `V`: vehiculo. Origen y orientacion documentados respecto a `W`.
    - `B`: herramienta/cuchilla. Origen en su pivote; poligono local.
    - `I`: imagen. `(u, v)` en pixeles, origen arriba a la izquierda, `u` a la
      derecha y `v` hacia abajo.
    """

    WORLD = "W"
    VEHICLE = "V"
    TOOL = "B"
    IMAGE = "I"


class OrientationMode(StrEnum):
    """Politica de orientacion de la herramienta. H1 usa `fixed` (§3.3)."""

    FIXED = "fixed"
    VARIABLE = "variable"


class PlanStatus(StrEnum):
    """Estados del resultado de planificacion (§3.5).

    `ACCEPTED` en E2 significa aceptacion **geometrica** (`validation_scope=geometric`,
    `mechanical_feasibility=not_evaluated`), nunca ejecucion aprobada. `UNSUPPORTED`
    (A1) distingue una capacidad aun no implementada de una imposibilidad geometrica.
    """

    ACCEPTED = "ACCEPTED"
    ABSTAIN = "ABSTAIN"
    INVALID_INPUT = "INVALID_INPUT"
    UNCERTIFIED = "UNCERTIFIED"
    UNSUPPORTED = "UNSUPPORTED"


class AbstentionReason(StrEnum):
    """Razones estables de abstencion o rechazo (§3.5).

    `NO_PATH_FOUND` significa que el algoritmo no encontro ruta; no es una prueba
    universal de imposibilidad.
    """

    NO_PATH_FOUND = "NO_PATH_FOUND"
    INSUFFICIENT_LOOKAHEAD = "INSUFFICIENT_LOOKAHEAD"
    UNKNOWN_SCALE = "UNKNOWN_SCALE"
    STALE_MAP = "STALE_MAP"
    EXCESS_UNCERTAINTY = "EXCESS_UNCERTAINTY"
    LIMIT_VIOLATION = "LIMIT_VIOLATION"
    COLLISION = "COLLISION"
    NO_SAFE_TRANSITION = "NO_SAFE_TRANSITION"
    NO_TARGET_AREA = "NO_TARGET_AREA"
    # A1/E2: razones del planificador geometrico.
    UNSUPPORTED_FEATURE = "UNSUPPORTED_FEATURE"
    """La peticion usa una capacidad aun no implementada; no es imposibilidad geometrica."""
    START_INADMISSIBLE = "START_INADMISSIBLE"
    """La huella en el inicio colisiona, sale del dominio o pisa suelo desconocido."""
    GOAL_INADMISSIBLE = "GOAL_INADMISSIBLE"
    """La huella en la meta colisiona, sale del dominio o pisa suelo desconocido."""
    UNKNOWN_SPACE = "UNKNOWN_SPACE"
    """Solo habria paso atravesando suelo desconocido, que se trata como prohibido."""
    CERTIFICATION_BUDGET_EXHAUSTED = "CERTIFICATION_BUDGET_EXHAUSTED"
    """El validador agoto su presupuesto de subdivision sin certificar ni refutar."""


class SimulationState(StrEnum):
    """Estados de la maquina de simulacion (§3.5)."""

    IDLE = "IDLE"
    WORKING = "WORKING"
    RETREATING = "RETREATING"
    ABSTAINED = "ABSTAINED"
    UNSAFE_STATE = "UNSAFE_STATE"


class CalibrationStatus(StrEnum):
    """Estado de una referencia de calibracion."""

    ABSENT = "absent"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"


class InflationTarget(StrEnum):
    """Donde se aplica la cota de incertidumbre de localizacion (§3.3).

    Se infla **una** de las dos geometrias, nunca ambas: las exclusiones
    (`exclusions`) o la huella de colision de la herramienta (`tool`).
    """

    EXCLUSIONS = "exclusions"
    TOOL = "tool"


# --- Utilidades geometricas de contrato -------------------------------------


def frozen_float_array(values: Sequence[Sequence[float]] | np.ndarray, name: str) -> np.ndarray:
    """Copia propia, finita, `float64` y **no modificable** de `values`.

    La memoria se respalda en un objeto `bytes` inmutable: ni el llamador, que
    conserva su array original, ni quien reciba este array pueden alterar su
    contenido, y el indicador `writeable` no puede reactivarse (R6). Para trabajar
    con los datos basta `array.copy()`.
    """
    array = ensure_finite_array(values, name)
    contiguous = np.ascontiguousarray(array, dtype=np.float64)
    frozen = np.frombuffer(contiguous.tobytes(), dtype=np.float64).reshape(contiguous.shape)
    return frozen


def validate_polygon_points(points: Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Valida un anillo poligonal y devuelve una copia propia `(n, 2)` no modificable.

    Reglas (R10): un vertice final **exactamente igual** al primero se interpreta
    como cierre explicito y se descarta; no se fusionan vertices por proximidad
    relativa, de modo que la validez no depende de la posicion global (una
    traslacion bien condicionada no cambia el resultado). Despues se exigen al
    menos 3 vertices distintos, validez topologica sin autointersecciones y area
    estrictamente positiva.

    Raises:
        NonFiniteValueError: si hay NaN/inf.
        InvalidPolygonError: si es degenerado o autointersectado.
    """
    array = ensure_finite_array(points, name)
    if array.ndim != 2 or array.shape[1] != 2:
        raise InvalidPolygonError(f"{name}: se espera forma (n, 2), se recibio {array.shape}")
    ring = array[:-1] if array.shape[0] > 1 and np.array_equal(array[0], array[-1]) else array
    distinct = np.unique(ring, axis=0).shape[0]
    if distinct < 3:
        raise InvalidPolygonError(
            f"{name}: se exigen al menos 3 vertices distintos, hay {distinct}"
        )
    polygon = Polygon(ring)
    if not polygon.is_valid or not polygon.is_simple:
        raise InvalidPolygonError(
            f"{name}: poligono invalido o autointersectado "
            f"({polygon.is_valid=}, {polygon.is_simple=})"
        )
    if polygon.area <= 0.0:
        raise InvalidPolygonError(f"{name}: area no positiva ({polygon.area!r})")
    return frozen_float_array(ring, name)


def circumscribed_radial_excess(radius: float, segments: int) -> float:
    """Exceso radial maximo del poligono circunscrito sobre el circulo exacto.

    Oraculo analitico (R12): `r * (sec(pi/n) - 1)`. Es un exceso **conservador**
    (la exclusion representada es mayor que la exacta), no un error de
    subproteccion.
    """
    radius = ensure_positive(radius, "radius")
    if int(segments) != segments or segments < MIN_CIRCLE_SEGMENTS:
        raise InvalidPolygonError(f"segments: se exige entero >= {MIN_CIRCLE_SEGMENTS}")
    return radius * (1.0 / math.cos(math.pi / int(segments)) - 1.0)


def minimum_segments_for_excess(radius: float, max_excess: float) -> int:
    """Menor numero de lados cuyo exceso radial conservador no supera `max_excess`."""
    radius = ensure_positive(radius, "radius")
    max_excess = ensure_positive(max_excess, "max_excess")
    segments = max(
        MIN_CIRCLE_SEGMENTS, math.ceil(math.pi / math.acos(radius / (radius + max_excess)))
    )
    while circumscribed_radial_excess(radius, segments) > max_excess:
        segments += 1
    return segments


def exterior_circle_polygon(
    center: Sequence[float],
    radius: float,
    segments: int = 64,
) -> np.ndarray:
    """Aproximacion poligonal **exterior** de un circulo.

    Un poligono regular inscrito subestima el circulo; aqui se escala el radio por
    `1/cos(pi/n)` para que el poligono contenga el circulo, de modo que ninguna
    exclusion se represente mas pequena de lo que es (§3.4: las aproximaciones de
    circulos y buffers deben ser exteriores).

    Returns:
        Vertices `(segments, 2)` en sentido antihorario, sin repetir el primero.
    """
    cx = ensure_finite(center[0], "center.x")
    cy = ensure_finite(center[1], "center.y")
    radius = ensure_positive(radius, "radius")
    if int(segments) != segments or segments < MIN_CIRCLE_SEGMENTS:
        raise InvalidPolygonError(
            f"segments: se exige entero >= {MIN_CIRCLE_SEGMENTS}, se recibio {segments!r}"
        )
    segments = int(segments)
    circumscribed = radius / math.cos(math.pi / segments)
    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    return np.column_stack(
        (cx + circumscribed * np.cos(angles), cy + circumscribed * np.sin(angles))
    )


def covariance_to_bound(covariance: Sequence[Sequence[float]], k: float) -> float:
    """Convierte una covarianza 2x2 en una cota espacial declarada `k*sqrt(lambda_max)`.

    Es un modelo de incertidumbre, no una garantia probabilistica (§3.3). El
    multiplicador `k` queda registrado por quien lo invoca.

    Raises:
        UncertaintyBudgetError: si la matriz no es simetrica o no es semidefinida
            positiva.
    """
    matrix = ensure_finite_array(covariance, "covariance")
    if matrix.shape != (2, 2):
        raise UncertaintyBudgetError(f"covariance: se espera (2, 2), se recibio {matrix.shape}")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise UncertaintyBudgetError("covariance: no es simetrica")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if float(eigenvalues.min()) < -1e-12:
        raise UncertaintyBudgetError(
            f"covariance: no es semidefinida positiva (autovalor minimo {eigenvalues.min()!r})"
        )
    k = ensure_positive(k, "k")
    return float(k * math.sqrt(max(float(eigenvalues.max()), 0.0)))


# --- Contratos --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Point2:
    """Punto 2D con marco y dominio espacial explicitos."""

    x: float
    y: float
    frame: Frame
    domain: SpatialDomain

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", ensure_finite(self.x, "Point2.x"))
        object.__setattr__(self, "y", ensure_finite(self.y, "Point2.y"))
        object.__setattr__(self, "frame", Frame(self.frame))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))

    @property
    def unit(self) -> LengthUnit:
        return unit_of_domain(self.domain)

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "frame": self.frame.value,
            "domain": self.domain.value,
            "unit": self.unit.value,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyBound:
    """Cota conservadora de localizacion, expresada como radio en la unidad del dominio.

    La primera version suma cotas de localizacion, calibracion y seguimiento
    (§3.3). El error de ejecucion infla la herramienta **o** las exclusiones, nunca
    ambas; esa eleccion vive en `SafetyConfig`.
    """

    localization: float
    calibration: float
    tracking: float
    domain: SpatialDomain
    provenance: Provenance = Provenance.ASSUMED
    k_sigma: float | None = None

    def __post_init__(self) -> None:
        for name in ("localization", "calibration", "tracking"):
            object.__setattr__(
                self,
                name,
                ensure_non_negative(getattr(self, name), f"UncertaintyBound.{name}"),
            )
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if self.k_sigma is not None:
            object.__setattr__(
                self, "k_sigma", ensure_positive(self.k_sigma, "UncertaintyBound.k_sigma")
            )

    @property
    def total(self) -> float:
        """Suma conservadora de las cotas, en la unidad del dominio."""
        return self.localization + self.calibration + self.tracking

    def within_budget(self, budget: float) -> bool:
        """`True` si la cota total cabe en el presupuesto declarado."""
        return self.total <= ensure_non_negative(budget, "budget")

    def to_dict(self) -> dict[str, Any]:
        return {
            "localization": self.localization,
            "calibration": self.calibration,
            "tracking": self.tracking,
            "total": self.total,
            "domain": self.domain.value,
            "unit": unit_of_domain(self.domain).value,
            "provenance": self.provenance.value,
            "k_sigma": self.k_sigma,
        }


@dataclass(frozen=True, slots=True)
class UncertaintyPolicy:
    """Fuente unica de la incertidumbre efectiva de una escena (R1).

    - `bound`: cotas declaradas de localizacion, calibracion y seguimiento; su suma
      conservadora es la proteccion minima exigida. No incluye el error de
      ejecucion del actuador, que se modela por separado en E3.
    - `budget`: maximo permitido para esa suma. Es un limite, **no** el margen que
      se infla; superarlo invalida la intervencion.
    - `inflate`: geometria que recibe la proteccion (exclusiones o herramienta),
      nunca ambas.
    """

    bound: UncertaintyBound
    budget: float
    inflate: InflationTarget = InflationTarget.EXCLUSIONS

    def __post_init__(self) -> None:
        if not isinstance(self.bound, UncertaintyBound):
            raise SchemaValidationError("UncertaintyPolicy.bound: se espera UncertaintyBound")
        object.__setattr__(
            self, "budget", ensure_non_negative(self.budget, "UncertaintyPolicy.budget")
        )
        object.__setattr__(self, "inflate", InflationTarget(self.inflate))
        if self.bound.total > self.budget + LIMIT_COMPARISON_TOLERANCE:
            raise UncertaintyBudgetError(
                f"incertidumbre: la suma de cotas ({self.bound.total!r}) excede el presupuesto "
                f"({self.budget!r}); superar el presupuesto invalida la intervencion"
            )

    @classmethod
    def from_bound(cls, bound: UncertaintyBound) -> UncertaintyPolicy:
        """Politica que infla exclusiones con presupuesto igual a la cota declarada."""
        return cls(bound=bound, budget=bound.total, inflate=InflationTarget.EXCLUSIONS)

    @property
    def domain(self) -> SpatialDomain:
        return self.bound.domain

    @property
    def total(self) -> float:
        """Proteccion minima exigida por incertidumbre, en la unidad del dominio."""
        return self.bound.total

    def to_dict(self) -> dict[str, Any]:
        payload = self.bound.to_dict()
        payload.update({"budget": self.budget, "inflate": self.inflate.value})
        return payload


@dataclass(frozen=True, slots=True)
class CalibrationReference:
    """Referencia minima de calibracion/escala para decidir dominios licitos.

    No implementa calibracion (E4); solo permite rechazar explicitamente que se
    solicite un resultado metrico verificado sin evidencia (§5 E1, §7).
    """

    calibration_id: str
    status: CalibrationStatus = CalibrationStatus.ABSENT
    scale_evidence: str | None = None
    independent_point_error: float | None = None
    domain: SpatialDomain = SpatialDomain.IMAGE_PX

    def __post_init__(self) -> None:
        if not str(self.calibration_id).strip():
            raise SchemaValidationError("CalibrationReference.calibration_id: cadena vacia")
        object.__setattr__(self, "status", CalibrationStatus(self.status))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        if self.independent_point_error is not None:
            object.__setattr__(
                self,
                "independent_point_error",
                ensure_non_negative(
                    self.independent_point_error, "CalibrationReference.independent_point_error"
                ),
            )
        if self.scale_evidence is not None:
            object.__setattr__(self, "scale_evidence", str(self.scale_evidence).strip() or None)
        if self.status is CalibrationStatus.VERIFIED:
            if not self.scale_evidence:
                raise MissingCalibrationError(
                    f"CalibrationReference({self.calibration_id}): estado 'verified' sin "
                    "evidencia de escala no vacia; no se admite calibracion verificada por "
                    "declaracion"
                )
            if self.domain is not SpatialDomain.GROUND_METRIC_VERIFIED:
                raise UnitMismatchError(
                    f"CalibrationReference({self.calibration_id}): una calibracion verificada "
                    f"autoriza el dominio {SpatialDomain.GROUND_METRIC_VERIFIED.value}, "
                    f"no {self.domain.value}"
                )

    @property
    def is_verified(self) -> bool:
        return self.status is CalibrationStatus.VERIFIED

    def require_verified(self, context: str = "operacion metrica") -> None:
        """Raises:
        MissingCalibrationError: si la calibracion no esta verificada.
        """
        if not self.is_verified:
            raise MissingCalibrationError(
                f"{context}: la calibracion '{self.calibration_id}' esta en estado "
                f"'{self.status.value}'; no autoriza resultados metricos verificados"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration_id": self.calibration_id,
            "status": self.status.value,
            "scale_evidence": self.scale_evidence,
            "independent_point_error": self.independent_point_error,
            "domain": self.domain.value,
        }


@dataclass(frozen=True, slots=True)
class PlantReference:
    """Planta de la verdad de referencia del escenario sintetico.

    Es referencia de verdad, no una deteccion: se mantiene separada del mapa
    estimado (§3.1).
    """

    plant_id: str
    position: Point2
    stem_radius: float
    protection_radius: float
    provenance: Provenance = Provenance.SYNTHETIC

    def __post_init__(self) -> None:
        if not str(self.plant_id).strip():
            raise SchemaValidationError("PlantReference.plant_id: cadena vacia")
        if not isinstance(self.position, Point2):
            raise SchemaValidationError("PlantReference.position: se espera Point2")
        object.__setattr__(
            self, "stem_radius", ensure_positive(self.stem_radius, "PlantReference.stem_radius")
        )
        object.__setattr__(
            self,
            "protection_radius",
            ensure_positive(self.protection_radius, "PlantReference.protection_radius"),
        )
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if self.protection_radius < self.stem_radius:
            raise SchemaValidationError(
                f"PlantReference({self.plant_id}): protection_radius "
                f"({self.protection_radius}) < stem_radius ({self.stem_radius})"
            )

    @property
    def domain(self) -> SpatialDomain:
        return self.position.domain

    def to_dict(self) -> dict[str, Any]:
        return {
            "plant_id": self.plant_id,
            "position": self.position.to_dict(),
            "stem_radius": self.stem_radius,
            "protection_radius": self.protection_radius,
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True, slots=True)
class CircularExclusion:
    """Exclusion circular `E_i = P_i (+) D(m_i)` (§3.3).

    `protection_radius` es la proteccion agronomica nominal y `uncertainty_margin`
    la cota conservadora de localizacion/calibracion/seguimiento. Su suma de
    Minkowski en el caso circular es la suma de radios.
    """

    plant_id: str
    center: Point2
    protection_radius: float
    uncertainty_margin: float
    polygon_segments: int = 64
    provenance: Provenance = Provenance.SYNTHETIC

    def __post_init__(self) -> None:
        if not str(self.plant_id).strip():
            raise SchemaValidationError("CircularExclusion.plant_id: cadena vacia")
        if not isinstance(self.center, Point2):
            raise SchemaValidationError("CircularExclusion.center: se espera Point2")
        object.__setattr__(
            self,
            "protection_radius",
            ensure_positive(self.protection_radius, "CircularExclusion.protection_radius"),
        )
        object.__setattr__(
            self,
            "uncertainty_margin",
            ensure_non_negative(self.uncertainty_margin, "CircularExclusion.uncertainty_margin"),
        )
        if int(self.polygon_segments) != self.polygon_segments:
            raise InvalidPolygonError("CircularExclusion.polygon_segments: se exige entero")
        object.__setattr__(self, "polygon_segments", int(self.polygon_segments))
        if self.polygon_segments < MIN_CIRCLE_SEGMENTS:
            raise InvalidPolygonError(
                f"CircularExclusion.polygon_segments: se exige >= {MIN_CIRCLE_SEGMENTS}"
            )
        object.__setattr__(self, "provenance", Provenance(self.provenance))

    @property
    def domain(self) -> SpatialDomain:
        return self.center.domain

    @property
    def total_radius(self) -> float:
        """Radio efectivo de la exclusion, proteccion mas margen."""
        return self.protection_radius + self.uncertainty_margin

    @property
    def discretization_excess(self) -> float:
        """Exceso radial conservador de la aproximacion poligonal (R12)."""
        return circumscribed_radial_excess(self.total_radius, self.polygon_segments)

    def polygon_points(self) -> np.ndarray:
        """Aproximacion exterior de la exclusion como anillo `(n, 2)`."""
        return exterior_circle_polygon(
            (self.center.x, self.center.y), self.total_radius, self.polygon_segments
        )

    def as_polygon(self) -> Polygon:
        """Aproximacion exterior de la exclusion como poligono de Shapely."""
        return Polygon(self.polygon_points())

    @classmethod
    def from_plant(
        cls,
        plant: PlantReference,
        uncertainty_margin: float,
        polygon_segments: int = 64,
    ) -> CircularExclusion:
        """Construye la exclusion de una planta de referencia."""
        return cls(
            plant_id=plant.plant_id,
            center=plant.position,
            protection_radius=plant.protection_radius,
            uncertainty_margin=uncertainty_margin,
            polygon_segments=polygon_segments,
            provenance=plant.provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plant_id": self.plant_id,
            "center": self.center.to_dict(),
            "protection_radius": self.protection_radius,
            "uncertainty_margin": self.uncertainty_margin,
            "total_radius": self.total_radius,
            "polygon_segments": self.polygon_segments,
            "discretization_excess": self.discretization_excess,
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True, slots=True)
class Corridor:
    """Franja de trabajo rectangular alineada con los ejes del marco declarado.

    El corredor delimita el dominio permitido; no determina alcance mecanico
    (§1.2).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    frame: Frame = Frame.WORLD
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC
    provenance: Provenance = Provenance.SYNTHETIC

    def __post_init__(self) -> None:
        for name in ("x_min", "x_max", "y_min", "y_max"):
            object.__setattr__(self, name, ensure_finite(getattr(self, name), f"Corridor.{name}"))
        object.__setattr__(self, "frame", Frame(self.frame))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if self.x_max <= self.x_min:
            raise SchemaValidationError(
                f"Corridor: x_max ({self.x_max}) debe ser > x_min ({self.x_min})"
            )
        if self.y_max <= self.y_min:
            raise SchemaValidationError(
                f"Corridor: y_max ({self.y_max}) debe ser > y_min ({self.y_min})"
            )

    @property
    def width(self) -> float:
        """Ancho lateral del corredor (eje `x`)."""
        return self.x_max - self.x_min

    @property
    def length(self) -> float:
        """Largo longitudinal del corredor (eje `y`)."""
        return self.y_max - self.y_min

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def unit(self) -> LengthUnit:
        return unit_of_domain(self.domain)

    def polygon_points(self) -> np.ndarray:
        return np.array(
            [
                [self.x_min, self.y_min],
                [self.x_max, self.y_min],
                [self.x_max, self.y_max],
                [self.x_min, self.y_max],
            ],
            dtype=np.float64,
        )

    def as_polygon(self) -> Polygon:
        return Polygon(self.polygon_points())

    def contains_point(self, point: Point2) -> bool:
        """`True` si el punto esta dentro o en el borde del corredor.

        Raises:
            UnitMismatchError: si el punto pertenece a otro dominio espacial.
            FrameMismatchError: si el punto esta expresado en otro marco; debe
                transformarse explicitamente antes de compararlo (R3).
        """
        if not isinstance(point, Point2):
            raise SchemaValidationError("Corridor.contains_point: se espera Point2")
        require_same_domain(self.domain, point.domain, context="Corridor.contains_point")
        if point.frame is not self.frame:
            raise FrameMismatchError(
                f"Corridor.contains_point: corredor en el marco {self.frame.value}, punto en "
                f"{point.frame.value}; transformelo explicitamente antes de comparar"
            )
        return (
            self.x_min <= point.x <= self.x_max and self.y_min <= point.y <= self.y_max
        )

    @classmethod
    def centered(
        cls,
        width: float,
        length: float,
        center_x: float = 0.0,
        start_y: float = 0.0,
        frame: Frame = Frame.WORLD,
        domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
        provenance: Provenance = Provenance.SYNTHETIC,
    ) -> Corridor:
        """Corredor de ancho/largo dados, centrado lateralmente en `center_x`."""
        width = ensure_positive(width, "Corridor.width")
        length = ensure_positive(length, "Corridor.length")
        center_x = ensure_finite(center_x, "Corridor.center_x")
        start_y = ensure_finite(start_y, "Corridor.start_y")
        return cls(
            x_min=center_x - 0.5 * width,
            x_max=center_x + 0.5 * width,
            y_min=start_y,
            y_max=start_y + length,
            frame=frame,
            domain=domain,
            provenance=provenance,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "width": self.width,
            "length": self.length,
            "frame": self.frame.value,
            "domain": self.domain.value,
            "unit": self.unit.value,
            "is_metric": is_metric(self.domain),
            "provenance": self.provenance.value,
        }
