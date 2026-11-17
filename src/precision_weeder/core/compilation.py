"""Estados y codigos de compilacion/ejecucion (A3/E3; contratos §5 bis).

Separan causa y fuerza de la conclusion: `INFEASIBLE_PROVEN` exige certificado (por
ejemplo alcance radial analitico); agotar una busqueda es `NOT_FOUND_WITHIN_LIMIT`;
una capacidad no implementada es `UNSUPPORTED`, nunca imposibilidad.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["CompileStatus", "InfeasibilityCode"]


class CompileStatus(StrEnum):
    FEASIBLE_CANDIDATE = "FEASIBLE_CANDIDATE"
    INFEASIBLE_PROVEN = "INFEASIBLE_PROVEN"
    NOT_FOUND_WITHIN_LIMIT = "NOT_FOUND_WITHIN_LIMIT"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_INPUT = "INVALID_INPUT"
    UNCERTIFIED = "UNCERTIFIED"


class InfeasibilityCode(StrEnum):
    RADIAL_REACH = "RADIAL_REACH"
    ORIENTATION_CONFLICT = "ORIENTATION_CONFLICT"
    JOINT_LIMIT = "JOINT_LIMIT"
    VELOCITY_LIMIT = "VELOCITY_LIMIT"
    ACCELERATION_LIMIT = "ACCELERATION_LIMIT"
    CLOSURE_ERROR = "CLOSURE_ERROR"
    BRANCH_DISCONTINUITY = "BRANCH_DISCONTINUITY"
    COLLISION = "COLLISION"
    INSUFFICIENT_LOOKAHEAD = "INSUFFICIENT_LOOKAHEAD"
    UNKNOWN_SPACE = "UNKNOWN_SPACE"
