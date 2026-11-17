"""Planificacion geometrica en suelo `W` (E2). Salida: `TaskPath` dentro de `PlanResult`.

El planificador no importa cinematica, actuadores, simulador ni video.
"""

from __future__ import annotations

from .base import CandidateRecord, PlanRequest, PlanResult
from .geometric import GeometricPlanner, plan_path

__all__ = ["CandidateRecord", "GeometricPlanner", "PlanRequest", "PlanResult", "plan_path"]
