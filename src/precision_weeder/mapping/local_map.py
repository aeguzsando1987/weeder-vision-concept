"""Mapa local en suelo `W` para el planificador geometrico (A1/E2, orden §3.2 y §4.1).

`LocalMap` declara **explicitamente** que parte del corredor es conocida. El dominio
permitido para la huella es `corredor ∩ region_conocida - regiones_desconocidas`:
el suelo desconocido interior funciona como region prohibida. Que no haya plantas
no certifica espacio libre: un mapa sin exclusiones exige `free_space_certified`.

La incertidumbre aplicada sigue la politica de E1: infla las exclusiones **o** la
herramienta (`tool_inflation`), nunca ambas, y cubre la cota declarada. Las
geometrias efectivas que llegan al validador se obtienen de este contrato
(`exclusion_polygons`, `allowed_region`, `tool_inflation`).

Este modulo no importa escenarios, simulador, cinematica ni video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..core.errors import (
    FrameMismatchError,
    ScenarioError,
    SchemaValidationError,
    UncertaintyBudgetError,
    UnitMismatchError,
)
from ..core.identity import content_hash
from ..core.trajectory import TASK_DOMAINS
from ..core.types import (
    CircularExclusion,
    Corridor,
    Frame,
    InflationTarget,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
    validate_polygon_points,
)
from ..core.units import (
    LIMIT_COMPARISON_TOLERANCE,
    Provenance,
    SpatialDomain,
    ensure_finite,
    ensure_non_negative,
    require_same_domain,
    unit_of_domain,
)

__all__ = ["LOCAL_MAP_SCHEMA_VERSION", "LocalMap", "rectangle_ring"]

LOCAL_MAP_SCHEMA_VERSION: Final[int] = 1


def _text(value: Any, name: str) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        raise SchemaValidationError(f"{name}: cadena vacia")
    return text


@dataclass(frozen=True, slots=True, eq=False)
class LocalMap:
    """Mapa local estatico o con vigencia declarada, en `W` y metros.

    Attributes:
        map_id, version: Identidad; el contenido se identifica ademas por SHA-256.
        segment_id: Segmento de `W` al que pertenece; dos `W` de origenes distintos
            no son el mismo marco aunque compartan nombre.
        corridor: Dominio lateral/longitudinal permitido.
        known_region: Anillo `(n, 2)` del suelo conocido. Se intersecta con el
            corredor; lo que queda fuera es desconocido.
        unknown_regions: Anillos de suelo desconocido interior (prohibido).
        known_region_basis: Por que se declara conocida esa region (p. ej.
            `synthetic_ground_truth`); no se infiere de la ausencia de plantas.
        exclusions: Exclusiones identificadas, ya con su margen efectivo.
        uncertainty: Politica que se aplico; `tool_inflation` es la parte aplicada a
            la herramienta cuando `inflate=tool`.
        free_space_certified: Necesario si no hay exclusiones.
        static: `True` para un fixture estatico; si es `False`, exige vigencia.
    """

    map_id: str
    segment_id: str
    domain: SpatialDomain
    corridor: Corridor
    known_region: np.ndarray
    known_region_basis: str
    exclusions: tuple[CircularExclusion, ...]
    uncertainty: UncertaintyPolicy
    tool_inflation: float = 0.0
    unknown_regions: tuple[np.ndarray, ...] = ()
    free_space_certified: bool = False
    static: bool = True
    valid_from_s: float | None = None
    valid_to_s: float | None = None
    version: int = 1
    frame: Frame = Frame.WORLD
    provenance: Provenance = Provenance.SYNTHETIC
    notes: tuple[str, ...] = ()
    schema_version: int = LOCAL_MAP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "map_id", _text(self.map_id, "LocalMap.map_id"))
        object.__setattr__(self, "segment_id", _text(self.segment_id, "LocalMap.segment_id"))
        object.__setattr__(
            self, "known_region_basis", _text(self.known_region_basis, "known_region_basis")
        )
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "frame", Frame(self.frame))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        if self.schema_version != LOCAL_MAP_SCHEMA_VERSION:
            raise SchemaValidationError(f"LocalMap.schema_version {self.schema_version!r}")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 1:
            raise SchemaValidationError("LocalMap.version: entero >= 1")
        if self.frame is not Frame.WORLD:
            raise FrameMismatchError("LocalMap: el mapa de tarea se expresa en W")
        if self.domain not in TASK_DOMAINS:
            raise UnitMismatchError(
                f"LocalMap: dominio {self.domain.value} no admitido para planificar en metros"
            )
        if not isinstance(self.corridor, Corridor):
            raise SchemaValidationError("LocalMap.corridor: se espera Corridor")
        if not isinstance(self.uncertainty, UncertaintyPolicy):
            raise SchemaValidationError("LocalMap.uncertainty: se espera UncertaintyPolicy")
        for flag in ("free_space_certified", "static"):
            if not isinstance(getattr(self, flag), bool):
                raise SchemaValidationError(f"LocalMap.{flag}: se exige un booleano")
        object.__setattr__(
            self, "known_region", validate_polygon_points(self.known_region, "known_region")
        )
        object.__setattr__(
            self,
            "unknown_regions",
            tuple(
                validate_polygon_points(ring, f"unknown_regions[{i}]")
                for i, ring in enumerate(self.unknown_regions)
            ),
        )
        exclusions = tuple(self.exclusions)
        if not all(isinstance(e, CircularExclusion) for e in exclusions):
            raise SchemaValidationError("LocalMap.exclusions: se esperan CircularExclusion")
        object.__setattr__(self, "exclusions", exclusions)
        object.__setattr__(
            self, "tool_inflation", ensure_non_negative(self.tool_inflation, "tool_inflation")
        )
        self._check_frames_and_domains()
        self._check_uncertainty()
        self._check_validity_window()
        if not exclusions and not self.free_space_certified:
            raise ScenarioError(
                f"LocalMap({self.map_id}): sin exclusiones y sin free_space_certified; la "
                "ausencia de plantas no certifica suelo libre"
            )
        if exclusions and self.free_space_certified:
            raise ScenarioError(
                f"LocalMap({self.map_id}): free_space_certified es incompatible con exclusiones"
            )
        ids = [e.plant_id for e in exclusions]
        if len(set(ids)) != len(ids):
            raise ScenarioError(f"LocalMap({self.map_id}): exclusion duplicada")
        if self.allowed_region().is_empty:
            raise ScenarioError(f"LocalMap({self.map_id}): no queda dominio conocido permitido")

    def _check_frames_and_domains(self) -> None:
        domains = [self.domain, self.corridor.domain, self.uncertainty.domain]
        domains.extend(e.domain for e in self.exclusions)
        require_same_domain(*domains, context=f"LocalMap({self.map_id})")
        frames = {self.frame, self.corridor.frame}
        frames.update(e.center.frame for e in self.exclusions)
        if frames != {Frame.WORLD}:
            raise FrameMismatchError(
                f"LocalMap({self.map_id}): corredor y exclusiones deben estar en W"
            )

    def _check_uncertainty(self) -> None:
        required = self.uncertainty.total
        if self.uncertainty.inflate is InflationTarget.EXCLUSIONS:
            if self.tool_inflation != 0.0:
                raise UncertaintyBudgetError(
                    "LocalMap: inflate=exclusions con tool_inflation > 0 aplicaria el margen dos "
                    "veces"
                )
            for exclusion in self.exclusions:
                if exclusion.uncertainty_margin + LIMIT_COMPARISON_TOLERANCE < required:
                    raise UncertaintyBudgetError(
                        f"LocalMap: margen de {exclusion.plant_id} menor que la cota declarada"
                    )
        else:
            if any(e.uncertainty_margin != 0.0 for e in self.exclusions):
                raise UncertaintyBudgetError(
                    "LocalMap: inflate=tool exige exclusiones sin margen (no duplicar)"
                )
            if self.tool_inflation + LIMIT_COMPARISON_TOLERANCE < required:
                raise UncertaintyBudgetError("LocalMap: tool_inflation menor que la cota declarada")

    def _check_validity_window(self) -> None:
        if self.static:
            if self.valid_from_s is not None or self.valid_to_s is not None:
                raise SchemaValidationError("LocalMap estatico: sin ventana de vigencia")
            return
        if self.valid_from_s is None or self.valid_to_s is None:
            raise SchemaValidationError("LocalMap no estatico: exige valid_from_s y valid_to_s")
        start = ensure_finite(self.valid_from_s, "valid_from_s")
        end = ensure_finite(self.valid_to_s, "valid_to_s")
        if end <= start:
            raise SchemaValidationError("LocalMap: vigencia vacia o invertida")

    # --- geometrias efectivas -------------------------------------------------

    @property
    def unit(self) -> str:
        return unit_of_domain(self.domain).value

    def exclusion_polygons(self) -> tuple[Polygon, ...]:
        """Aproximaciones **exteriores** de las exclusiones efectivas."""
        return tuple(exclusion.as_polygon() for exclusion in self.exclusions)

    def exclusion_union(self) -> BaseGeometry:
        polygons = self.exclusion_polygons()
        return unary_union(polygons) if polygons else Polygon()

    def unknown_union(self) -> BaseGeometry:
        """Suelo desconocido dentro del corredor: interior declarado y lo no conocido."""
        corridor = self.corridor.as_polygon()
        outside_known = corridor.difference(Polygon(self.known_region))
        interior = [Polygon(ring) for ring in self.unknown_regions]
        return unary_union([outside_known, *interior]).intersection(corridor)

    def allowed_region(self) -> BaseGeometry:
        """Dominio cerrado donde puede estar la huella: corredor ∩ conocido - desconocido."""
        corridor = self.corridor.as_polygon()
        known = corridor.intersection(Polygon(self.known_region))
        for ring in self.unknown_regions:
            known = known.difference(Polygon(ring))
        return known

    def target_region(self) -> BaseGeometry:
        """Suelo objetivo de cobertura: dominio permitido menos exclusiones."""
        return self.allowed_region().difference(self.exclusion_union())

    # --- identidad y serializacion --------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "map_id": self.map_id,
            "version": self.version,
            "segment_id": self.segment_id,
            "frame": self.frame.value,
            "domain": self.domain.value,
            "unit": self.unit,
            "provenance": self.provenance.value,
            "static": self.static,
            "valid_from_s": self.valid_from_s,
            "valid_to_s": self.valid_to_s,
            "corridor": self.corridor.to_dict(),
            "known_region": self.known_region.tolist(),
            "known_region_basis": self.known_region_basis,
            "unknown_regions": [ring.tolist() for ring in self.unknown_regions],
            "free_space_certified": self.free_space_certified,
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
            "uncertainty": self.uncertainty.to_dict(),
            "tool_inflation": self.tool_inflation,
            "notes": list(self.notes),
        }
        payload["content_hash"] = content_hash(payload)
        return payload

    def content_hash(self) -> str:
        return self.to_dict()["content_hash"]

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LocalMap:
        """Reconstruye el mapa y comprueba su identidad declarada."""
        try:
            corridor_d = payload["corridor"]
            corridor = Corridor(
                x_min=corridor_d["x_min"],
                x_max=corridor_d["x_max"],
                y_min=corridor_d["y_min"],
                y_max=corridor_d["y_max"],
                frame=corridor_d["frame"],
                domain=corridor_d["domain"],
                provenance=corridor_d["provenance"],
            )
            unc = payload["uncertainty"]
            policy = UncertaintyPolicy(
                bound=UncertaintyBound(
                    localization=unc["localization"],
                    calibration=unc["calibration"],
                    tracking=unc["tracking"],
                    domain=unc["domain"],
                    provenance=unc["provenance"],
                    k_sigma=unc["k_sigma"],
                ),
                budget=unc["budget"],
                inflate=unc["inflate"],
            )
            exclusions = tuple(
                CircularExclusion(
                    plant_id=item["plant_id"],
                    center=Point2(
                        x=item["center"]["x"],
                        y=item["center"]["y"],
                        frame=item["center"]["frame"],
                        domain=item["center"]["domain"],
                    ),
                    protection_radius=item["protection_radius"],
                    uncertainty_margin=item["uncertainty_margin"],
                    polygon_segments=item["polygon_segments"],
                    provenance=item["provenance"],
                )
                for item in payload["exclusions"]
            )
            local_map = cls(
                map_id=payload["map_id"],
                version=payload["version"],
                segment_id=payload["segment_id"],
                frame=payload["frame"],
                domain=payload["domain"],
                provenance=payload["provenance"],
                static=payload["static"],
                valid_from_s=payload["valid_from_s"],
                valid_to_s=payload["valid_to_s"],
                corridor=corridor,
                known_region=payload["known_region"],
                known_region_basis=payload["known_region_basis"],
                unknown_regions=tuple(payload["unknown_regions"]),
                free_space_certified=payload["free_space_certified"],
                exclusions=exclusions,
                uncertainty=policy,
                tool_inflation=payload["tool_inflation"],
                notes=tuple(payload["notes"]),
                schema_version=payload["schema_version"],
            )
        except KeyError as exc:
            raise SchemaValidationError(f"LocalMap: falta el campo {exc}") from exc
        declared = payload.get("content_hash")
        if declared is not None and declared != local_map.content_hash():
            raise SchemaValidationError("LocalMap: content_hash no coincide con el contenido")
        return local_map

    def translated(self, dx: float, dy: float, map_id: str | None = None) -> LocalMap:
        """Copia trasladada rigidamente (prueba de generalizacion por cambio de origen)."""
        dx, dy = ensure_finite(dx, "dx"), ensure_finite(dy, "dy")
        shift = np.array([dx, dy])
        corridor = Corridor(
            x_min=self.corridor.x_min + dx,
            x_max=self.corridor.x_max + dx,
            y_min=self.corridor.y_min + dy,
            y_max=self.corridor.y_max + dy,
            frame=self.corridor.frame,
            domain=self.corridor.domain,
            provenance=self.corridor.provenance,
        )
        exclusions = tuple(
            CircularExclusion(
                plant_id=e.plant_id,
                center=Point2(
                    x=e.center.x + dx, y=e.center.y + dy, frame=e.center.frame, domain=e.domain
                ),
                protection_radius=e.protection_radius,
                uncertainty_margin=e.uncertainty_margin,
                polygon_segments=e.polygon_segments,
                provenance=e.provenance,
            )
            for e in self.exclusions
        )
        return LocalMap(
            map_id=map_id or f"{self.map_id}+translated",
            segment_id=self.segment_id,
            domain=self.domain,
            corridor=corridor,
            known_region=self.known_region + shift,
            known_region_basis=self.known_region_basis,
            exclusions=exclusions,
            uncertainty=self.uncertainty,
            tool_inflation=self.tool_inflation,
            unknown_regions=tuple(ring + shift for ring in self.unknown_regions),
            free_space_certified=self.free_space_certified,
            static=self.static,
            valid_from_s=self.valid_from_s,
            valid_to_s=self.valid_to_s,
            version=self.version,
            frame=self.frame,
            provenance=self.provenance,
            notes=(*self.notes, f"trasladado ({dx}, {dy}) m"),
        )


def rectangle_ring(x_min: float, x_max: float, y_min: float, y_max: float) -> list[list[float]]:
    """Anillo rectangular alineado con los ejes (utilidad para regiones conocidas)."""
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
