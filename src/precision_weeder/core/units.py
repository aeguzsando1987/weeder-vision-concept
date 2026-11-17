"""Unidades, dominios espaciales, procedencia, tolerancias y guardas numericas.

Reglas rectoras (`docs/technical_plan.md` §3.2 y §6):

- `image_px` y `ground_assumed` son dominios exploratorios; no se mezclan con
  `ground_metric_verified`.
- `ground_metric_synthetic` es metrico por construccion del banco sintetico, no
  por medicion fisica: no equivale a `ground_metric_verified`.
- `graphic_unscaled` reproduce literalmente las unidades del SVG de referencia y
  nunca se convierte a metros.
- Toda magnitud declarada registra su procedencia (`measured`, `synthetic`,
  `assumed`, `unknown`).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from enum import StrEnum
from typing import Final

import numpy as np

from .errors import InvalidDimensionError, NonFiniteValueError, UnitMismatchError

__all__ = [
    "CLEARANCE_EPSILON_M",
    "GEOMETRIC_APPROXIMATION_TOLERANCE_M",
    "LIMIT_COMPARISON_TOLERANCE",
    "TRANSFORM_TOLERANCE",
    "LengthUnit",
    "Provenance",
    "SpatialDomain",
    "ensure_finite",
    "ensure_finite_array",
    "ensure_non_negative",
    "ensure_positive",
    "is_metric",
    "require_same_domain",
    "require_verified_metric",
    "unit_of_domain",
]


# --- Tolerancias de software para fixtures sinteticos (§6) -------------------
# No sustituyen un margen agronomico; son presupuestos numericos.
TRANSFORM_TOLERANCE: Final[float] = 1e-9
"""Tolerancia de las transformaciones rigidas ida/vuelta en fixtures bien condicionados."""

CLEARANCE_EPSILON_M: Final[float] = 1e-4
"""Despeje minimo estricto entre huella de colision y exclusiones [m]."""

GEOMETRIC_APPROXIMATION_TOLERANCE_M: Final[float] = 1e-5
"""Cota maxima admitida de error por discretizacion geometrica [m]."""

LIMIT_COMPARISON_TOLERANCE: Final[float] = 1e-8
"""Tolerancia de comparacion numerica de limites, en la unidad correspondiente."""


class LengthUnit(StrEnum):
    """Unidad de longitud de una magnitud espacial."""

    METER = "m"
    PIXEL = "px"
    GRAPHIC_UNIT = "svg_unit"


class SpatialDomain(StrEnum):
    """Dominio espacial de un conjunto de coordenadas.

    El dominio, no solo la unidad, decide que afirmaciones son licitas.
    """

    IMAGE_PX = "image_px"
    GROUND_ASSUMED = "ground_assumed"
    GROUND_METRIC_SYNTHETIC = "ground_metric_synthetic"
    GROUND_METRIC_VERIFIED = "ground_metric_verified"
    GRAPHIC_UNSCALED = "graphic_unscaled"


class Provenance(StrEnum):
    """Origen del valor de un parametro (`docs/assumptions.md`)."""

    MEASURED = "measured"
    SYNTHETIC = "synthetic"
    ASSUMED = "assumed"
    UNKNOWN = "unknown"


_DOMAIN_UNITS: Final[dict[SpatialDomain, LengthUnit]] = {
    SpatialDomain.IMAGE_PX: LengthUnit.PIXEL,
    SpatialDomain.GROUND_ASSUMED: LengthUnit.METER,
    SpatialDomain.GROUND_METRIC_SYNTHETIC: LengthUnit.METER,
    SpatialDomain.GROUND_METRIC_VERIFIED: LengthUnit.METER,
    SpatialDomain.GRAPHIC_UNSCALED: LengthUnit.GRAPHIC_UNIT,
}

_METRIC_DOMAINS: Final[frozenset[SpatialDomain]] = frozenset(
    {SpatialDomain.GROUND_METRIC_SYNTHETIC, SpatialDomain.GROUND_METRIC_VERIFIED}
)


def unit_of_domain(domain: SpatialDomain) -> LengthUnit:
    """Unidad de longitud canonica del dominio.

    `ground_assumed` se expresa en metros supuestos: la unidad no certifica
    escala verificada.
    """
    return _DOMAIN_UNITS[SpatialDomain(domain)]


def is_metric(domain: SpatialDomain) -> bool:
    """`True` si el dominio tiene escala metrica (sintetica o verificada)."""
    return SpatialDomain(domain) in _METRIC_DOMAINS


def require_same_domain(*domains: SpatialDomain, context: str = "operacion") -> SpatialDomain:
    """Exige un unico dominio espacial comun y lo devuelve.

    Raises:
        UnitMismatchError: si se recibe mas de un dominio distinto, o ninguno.
    """
    unique = {SpatialDomain(d) for d in domains}
    if not unique:
        raise UnitMismatchError(f"{context}: no se recibio ningun dominio espacial")
    if len(unique) > 1:
        names = ", ".join(sorted(d.value for d in unique))
        raise UnitMismatchError(
            f"{context}: dominios espaciales incompatibles ({names}); "
            "no se mezclan dominios sin una transformacion declarada"
        )
    return unique.pop()


def require_verified_metric(domain: SpatialDomain, context: str = "operacion") -> SpatialDomain:
    """Exige `ground_metric_verified`.

    Raises:
        UnitMismatchError: para cualquier otro dominio, incluido el sintetico.
    """
    domain = SpatialDomain(domain)
    if domain is not SpatialDomain.GROUND_METRIC_VERIFIED:
        raise UnitMismatchError(
            f"{context}: requiere dominio {SpatialDomain.GROUND_METRIC_VERIFIED.value}, "
            f"se recibio {domain.value}"
        )
    return domain


# --- Guardas numericas ------------------------------------------------------


def ensure_finite(value: float, name: str) -> float:
    """Devuelve `value` como `float` finito.

    Raises:
        NonFiniteValueError: ante NaN, +inf, -inf o valor no numerico.
    """
    try:
        as_float = float(value)
    except (TypeError, ValueError) as exc:
        raise NonFiniteValueError(f"{name}: no es un numero real ({value!r})") from exc
    if not math.isfinite(as_float):
        raise NonFiniteValueError(f"{name}: se exige un valor finito, se recibio {as_float!r}")
    return as_float


def ensure_finite_array(values: Iterable[float], name: str) -> np.ndarray:
    """Devuelve un `np.ndarray` `float64` tras verificar finitud elemento a elemento."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise NonFiniteValueError(f"{name}: secuencia vacia")
    if not np.all(np.isfinite(array)):
        raise NonFiniteValueError(f"{name}: contiene valores no finitos (NaN o inf)")
    return array


def ensure_positive(value: float, name: str) -> float:
    """Devuelve `value` finito y estrictamente positivo."""
    as_float = ensure_finite(value, name)
    if as_float <= 0.0:
        raise InvalidDimensionError(f"{name}: se exige > 0, se recibio {as_float!r}")
    return as_float


def ensure_non_negative(value: float, name: str) -> float:
    """Devuelve `value` finito y mayor o igual que cero."""
    as_float = ensure_finite(value, name)
    if as_float < 0.0:
        raise InvalidDimensionError(f"{name}: se exige >= 0, se recibio {as_float!r}")
    return as_float
