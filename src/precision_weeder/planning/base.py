"""Solicitud y resultado del planificador geometrico (A1, orden A1/E2 §3.2).

`PlanRequest` solo contiene tarea en suelo: identidad de mapa y perfil, inicio y
meta del origen `B` en `W`, orientacion pedida y rol. No admite `ToolState`
articular, mecanismo, video ni calibracion implicita.

`PlanResult` lleva estado, razon, candidato elegido y candidatos rechazados como
diagnostico. Un resultado que no es `ACCEPTED` **no** contiene ruta. Una ruta
aceptada lo es solo geometricamente (`validation_scope=geometric`,
`mechanical_feasibility=not_evaluated`); nunca se exporta como ejecucion aprobada.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final

from ..core.errors import FrameMismatchError, SchemaValidationError, UnitMismatchError
from ..core.identity import content_hash, freeze, is_sha256, thaw
from ..core.trajectory import TASK_DOMAINS, OrientationRequirement, TaskPath
from ..core.types import AbstentionReason, Frame, PlanStatus, Point2
from ..core.units import require_same_domain
from ..validation.swept_volume import ValidationReport

__all__ = [
    "PLAN_RESULT_SCHEMA_VERSION",
    "CandidateRecord",
    "PlanRequest",
    "PlanResult",
]

PLAN_REQUEST_SCHEMA_VERSION: Final[int] = 1
PLAN_RESULT_SCHEMA_VERSION: Final[int] = 2
"""v2 (A1E2-R1): colecciones congeladas, ligadura ruta-informe, `from_dict`, hash."""


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """Peticion de tarea: llevar el origen `B` de `start` a `goal` con progreso en `y`.

    «Progreso completo» significa ir del inicio a la meta pedidos con `y` no
    decreciente y barrido valido entre ambos; por eso se exige `goal.y > start.y`.
    """

    request_id: str
    map_id: str
    map_hash: str
    profile_id: str
    profile_hash: str
    start: Point2
    goal: Point2
    orientation: OrientationRequirement
    role: str = "pasada_principal"
    schema_version: int = PLAN_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("request_id", "map_id", "profile_id", "role"):
            if not str(getattr(self, name)).strip():
                raise SchemaValidationError(f"PlanRequest.{name}: cadena vacia")
        for name in ("map_hash", "profile_hash"):
            value = str(getattr(self, name))
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise SchemaValidationError(f"PlanRequest.{name}: se espera SHA-256")
        if self.schema_version != PLAN_REQUEST_SCHEMA_VERSION:
            raise SchemaValidationError("PlanRequest.schema_version no soportada")
        if not isinstance(self.start, Point2) or not isinstance(self.goal, Point2):
            raise SchemaValidationError("PlanRequest: start y goal deben ser Point2")
        if not isinstance(self.orientation, OrientationRequirement):
            raise SchemaValidationError("PlanRequest.orientation: OrientationRequirement")
        for name, point in (("start", self.start), ("goal", self.goal)):
            if point.frame is not Frame.WORLD:
                raise FrameMismatchError(f"PlanRequest.{name}: se expresa en W")
            if point.domain not in TASK_DOMAINS:
                raise UnitMismatchError(
                    f"PlanRequest.{name}: dominio {point.domain.value}; la tarea es en metros"
                )
        require_same_domain(self.start.domain, self.goal.domain, context="PlanRequest")
        if not self.goal.y > self.start.y:
            raise SchemaValidationError(
                f"PlanRequest: la meta (y={self.goal.y}) debe estar por delante del inicio "
                f"(y={self.start.y}); el progreso longitudinal es no decreciente"
            )

    @property
    def domain(self) -> Any:
        return self.start.domain

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "map_id": self.map_id,
            "map_hash": self.map_hash,
            "profile_id": self.profile_id,
            "profile_hash": self.profile_hash,
            "start": self.start.to_dict(),
            "goal": self.goal.to_dict(),
            "orientation": self.orientation.to_dict(),
            "role": self.role,
        }
        payload["content_hash"] = content_hash(payload)
        return payload

    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanRequest:
        try:
            request = cls(
                request_id=payload["request_id"],
                map_id=payload["map_id"],
                map_hash=payload["map_hash"],
                profile_id=payload["profile_id"],
                profile_hash=payload["profile_hash"],
                start=Point2(**{k: payload["start"][k] for k in ("x", "y", "frame", "domain")}),
                goal=Point2(**{k: payload["goal"][k] for k in ("x", "y", "frame", "domain")}),
                orientation=OrientationRequirement.from_dict(payload["orientation"]),
                role=payload["role"],
                schema_version=payload["schema_version"],
            )
        except KeyError as exc:
            raise SchemaValidationError(f"PlanRequest: falta el campo {exc}") from exc
        declared = payload.get("content_hash")
        if declared is not None and declared != request.content_hash():
            raise SchemaValidationError("PlanRequest: content_hash no coincide con el contenido")
        return request


@dataclass(frozen=True, eq=False)
class CandidateRecord:
    """Candidato del planificador: polilinea, costo desglosado y destino.

    Inmutable: `waypoints` en tuplas y `cost`/`detail` congelados (A1E2-R1).
    """

    rank: int
    waypoints: tuple[tuple[float, float], ...]
    cost: Any
    outcome: str
    detail: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0:
            raise SchemaValidationError("CandidateRecord.rank: entero >= 0")
        try:
            points = tuple((float(x), float(y)) for x, y in self.waypoints)
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"CandidateRecord.waypoints: {exc}") from exc
        if len(points) < 2 or not all(math.isfinite(v) for p in points for v in p):
            raise SchemaValidationError("CandidateRecord.waypoints: >= 2 puntos finitos")
        object.__setattr__(self, "waypoints", points)
        if not str(self.outcome).strip():
            raise SchemaValidationError("CandidateRecord.outcome: cadena vacia")
        cost = freeze(self.cost)
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in cost.values()):
            raise SchemaValidationError("CandidateRecord.cost: valores numericos finitos")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "detail", freeze(self.detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "waypoints": [list(p) for p in self.waypoints],
            "cost": thaw(self.cost),
            "outcome": self.outcome,
            "detail": thaw(self.detail),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateRecord:
        try:
            return cls(
                rank=payload["rank"],
                waypoints=tuple(tuple(p) for p in payload["waypoints"]),
                cost=payload["cost"],
                outcome=payload["outcome"],
                detail=payload["detail"],
            )
        except KeyError as exc:
            raise SchemaValidationError(f"CandidateRecord: falta el campo {exc}") from exc


@dataclass(frozen=True, eq=False)
class PlanResult:
    """Resultado del planificador. Sin ruta salvo `ACCEPTED`.

    Invariantes (A1E2-R1):

    - `ACCEPTED` exige ruta e informe; el informe se reconstruye y verifica
      (`ValidationReport.from_dict`: version, coherencia de estado y `content_hash`)
      y debe estar ligado a **esta** ruta, mapa y perfil. Sustituir la ruta, o
      presentar un informe de otra ruta, incompleto, rechazado o no certificado,
      se rechaza.
    - Un resultado no aceptado no contiene ruta, y su informe (si lo hay) no es
      `ACCEPTED`.
    - Colecciones congeladas; `to_dict` exporta copias profundas independientes.

    `config_hash` identifica `planner + validation + safety`; el informe lleva su
    propio `config_hash` (`validation + safety`). Son hashes distintos a proposito.
    """

    status: PlanStatus
    reason: AbstentionReason | None
    detail: str
    request_hash: str | None
    map_hash: str
    profile_hash: str
    config_hash: str
    path: TaskPath | None = None
    validation: Any = None
    chosen: CandidateRecord | None = None
    rejected: tuple[CandidateRecord, ...] = ()
    search: Any = field(default_factory=dict)
    timing: Any = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "status", PlanStatus(self.status))
            if self.reason is not None:
                object.__setattr__(self, "reason", AbstentionReason(self.reason))
        except ValueError as exc:
            raise SchemaValidationError(f"PlanResult: estado o razon invalidos ({exc})") from exc
        for name in ("map_hash", "profile_hash", "config_hash"):
            if not is_sha256(getattr(self, name)):
                raise SchemaValidationError(f"PlanResult.{name}: se espera SHA-256")
        if self.request_hash is not None and not is_sha256(self.request_hash):
            raise SchemaValidationError("PlanResult.request_hash: se espera SHA-256")
        if self.path is not None and not isinstance(self.path, TaskPath):
            raise SchemaValidationError("PlanResult.path: se espera TaskPath")
        if self.chosen is not None and not isinstance(self.chosen, CandidateRecord):
            raise SchemaValidationError("PlanResult.chosen: se espera CandidateRecord")
        rejected = tuple(self.rejected)
        if not all(isinstance(r, CandidateRecord) for r in rejected):
            raise SchemaValidationError("PlanResult.rejected: se esperan CandidateRecord")
        object.__setattr__(self, "rejected", rejected)
        if isinstance(self.validation, ValidationReport):
            object.__setattr__(self, "validation", self.validation.to_dict())
        object.__setattr__(self, "validation", freeze(self.validation))
        object.__setattr__(self, "search", freeze(self.search))
        object.__setattr__(self, "timing", freeze(self.timing))
        report = (
            ValidationReport.from_dict(thaw(self.validation))
            if self.validation is not None
            else None
        )
        accepted = self.status is PlanStatus.ACCEPTED
        if accepted != (self.path is not None):
            raise SchemaValidationError(
                "PlanResult: solo un resultado ACCEPTED contiene ruta, y siempre la contiene"
            )
        if accepted:
            self._check_accepted_binding(report)
        elif report is not None and report.accepted:
            raise SchemaValidationError("PlanResult no aceptado con informe ACCEPTED")
        if not accepted and self.reason is None:
            raise SchemaValidationError("PlanResult no aceptado exige razon")

    def _check_accepted_binding(self, report: ValidationReport | None) -> None:
        path = self.path
        assert path is not None
        if self.reason is not None:
            raise SchemaValidationError("PlanResult ACCEPTED no lleva razon de abstencion")
        if report is None or not report.accepted:
            raise SchemaValidationError("PlanResult ACCEPTED exige validacion ACCEPTED")
        if self.request_hash is None:
            raise SchemaValidationError("PlanResult ACCEPTED exige request_hash")
        if self.chosen is None:
            raise SchemaValidationError("PlanResult ACCEPTED exige el candidato elegido")
        bindings = {
            "path_hash": (report.path_hash, path.content_hash()),
            "map_hash (informe/resultado)": (report.map_hash, self.map_hash),
            "map_hash (ruta/resultado)": (path.map_hash, self.map_hash),
            "profile_hash (informe/resultado)": (report.profile_hash, self.profile_hash),
            "profile_hash (ruta/resultado)": (path.profile_hash, self.profile_hash),
        }
        broken = [name for name, (a, b) in bindings.items() if a != b]
        if broken:
            raise SchemaValidationError(
                f"PlanResult ACCEPTED: el informe no corresponde a esta ruta/mapa/perfil ({broken})"
            )

    @property
    def accepted(self) -> bool:
        return self.status is PlanStatus.ACCEPTED

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_RESULT_SCHEMA_VERSION,
            "status": self.status.value,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "validation_scope": "geometric",
            "mechanical_feasibility": "not_evaluated",
            "output_kind": "propuesta geometrica (TaskPath); no es trayectoria ejecutable",
            "request_hash": self.request_hash,
            "map_hash": self.map_hash,
            "profile_hash": self.profile_hash,
            "config_hash": self.config_hash,
            "config_hash_scope": "planner + validation + safety",
            "path": self.path.to_dict() if self.path else None,
            "validation": thaw(self.validation),
            "chosen": self.chosen.to_dict() if self.chosen else None,
            "rejected_candidates": [c.to_dict() for c in self.rejected],
            "search": thaw(self.search),
        }

    def content_hash(self) -> str:
        """Identidad del contenido determinista (excluye el tiempo de pared)."""
        return content_hash(self._payload())

    def to_dict(self, include_timing: bool = True) -> dict[str, Any]:
        """Copia profunda independiente; `include_timing=False` deja solo lo determinista."""
        payload = self._payload()
        payload["content_hash"] = content_hash(payload)
        if include_timing:
            payload["timing"] = thaw(self.timing)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PlanResult:
        """Reconstruye un resultado exportado y verifica version, alcance e identidad.

        Raises:
            SchemaValidationError: campo ausente, version/alcance distintos, informe
                incoherente o desligado de la ruta, o `content_hash` que no coincide.
        """
        try:
            if payload["schema_version"] != PLAN_RESULT_SCHEMA_VERSION:
                raise SchemaValidationError(
                    f"PlanResult: schema_version {payload['schema_version']!r} no soportada"
                )
            if (
                payload["validation_scope"] != "geometric"
                or payload["mechanical_feasibility"] != "not_evaluated"
            ):
                raise SchemaValidationError("PlanResult: alcance distinto del geometrico")
            result = cls(
                status=payload["status"],
                reason=payload["reason"],
                detail=payload["detail"],
                request_hash=payload["request_hash"],
                map_hash=payload["map_hash"],
                profile_hash=payload["profile_hash"],
                config_hash=payload["config_hash"],
                path=TaskPath.from_dict(payload["path"]) if payload["path"] else None,
                validation=payload["validation"],
                chosen=CandidateRecord.from_dict(payload["chosen"]) if payload["chosen"] else None,
                rejected=tuple(
                    CandidateRecord.from_dict(c) for c in payload["rejected_candidates"]
                ),
                search=payload["search"],
                timing=payload.get("timing", {}),
            )
        except KeyError as exc:
            raise SchemaValidationError(f"PlanResult: falta el campo {exc}") from exc
        if payload.get("content_hash") != result.content_hash():
            raise SchemaValidationError("PlanResult: content_hash no coincide con el contenido")
        return result
