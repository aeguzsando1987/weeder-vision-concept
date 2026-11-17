"""Jerarquia de errores del dominio.

Las excepciones de este modulo expresan violaciones de contrato detectadas en la
frontera de los esquemas (E1). Una abstencion del planificador NO es una
excepcion: es un resultado del dominio (`PlanStatus.ABSTAIN`), segun
`docs/technical_plan.md` §7.
"""

from __future__ import annotations

__all__ = [
    "ConfigError",
    "FrameMismatchError",
    "InvalidDimensionError",
    "InvalidPolygonError",
    "MissingCalibrationError",
    "NonFiniteValueError",
    "PrecisionWeederError",
    "ScenarioError",
    "SchemaValidationError",
    "TemporalOrderError",
    "UncertaintyBudgetError",
    "UnitMismatchError",
    "UnknownConfigKeyError",
]


class PrecisionWeederError(Exception):
    """Raiz de todos los errores propios del proyecto."""


class SchemaValidationError(PrecisionWeederError, ValueError):
    """Un contrato de datos recibio un valor que no puede representar."""


class NonFiniteValueError(SchemaValidationError):
    """Se recibio NaN, +inf o -inf donde el contrato exige un valor finito."""


class InvalidDimensionError(SchemaValidationError):
    """Dimension nula o negativa donde el contrato exige magnitud positiva."""


class InvalidPolygonError(SchemaValidationError):
    """Poligono vacio, degenerado, autointersectado o con area no positiva."""


class TemporalOrderError(SchemaValidationError):
    """Tiempos no finitos, negativos o decrecientes en una secuencia temporal."""


class UnitMismatchError(SchemaValidationError):
    """Mezcla de unidades o de dominios espaciales incompatibles."""


class FrameMismatchError(SchemaValidationError):
    """Composicion o comparacion entre marcos de coordenadas incompatibles."""


class UncertaintyBudgetError(SchemaValidationError):
    """Covarianza no simetrica/no semidefinida positiva, o cota fuera de presupuesto."""


class MissingCalibrationError(PrecisionWeederError):
    """Se solicito un resultado metrico verificado sin calibracion valida.

    Se usa tambien cuando una configuracion declara `mode: calibrated` sin
    evidencia de escala: la degradacion silenciosa esta prohibida
    (`docs/technical_plan.md` §7).
    """


class ConfigError(PrecisionWeederError):
    """Configuracion invalida o inconsistente."""


class UnknownConfigKeyError(ConfigError):
    """Clave desconocida en la configuracion: siempre es error, nunca se ignora."""


class ScenarioError(PrecisionWeederError):
    """El escenario sintetico solicitado no existe o es inconsistente."""
