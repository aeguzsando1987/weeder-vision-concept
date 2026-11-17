"""Perfil geometrico de la herramienta en su marco local `B` (A1, orden A1/E2 §3.2).

`ToolGeometry` es el `ToolProfile` del plan adaptado: poligonos locales de colision
y de corte, referencia `B`, unidades, procedencia e identidad de contenido. No
contiene configuraciones articulares, topologia, montaje ni limites de motor: eso
pertenece al modelo mecanico (A3). Se extrajo de `simulation/tool.py` para que el
planificador no dependa del simulador; `simulation.tool` la reexporta, de modo que
existe una sola implementacion.

`ToolGeometry` describe la herramienta en su propio marco `B`, con el origen en el
pivote declarado. Se distinguen dos poligonos: la **huella de colision**, que es la
que debe mantenerse lejos de las exclusiones, y la **huella activa de corte**, que
es la que acumula cobertura. Pueden diferir, pero el corte siempre queda
cubierto por la huella de colision **realmente almacenada** (R11).

Los poligonos se guardan como copias propias no modificables (R6): ni el array
del llamador ni el devuelto por el atributo pueden alterar una huella validada.

`orientation_mode` es un campo heredado de E1. La politica de orientacion de una
tarea la declara la tarea (`core.trajectory.OrientationRequirement`), no la forma.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import JOIN_STYLE, Polygon

from ..core.config import ToolConfig
from ..core.errors import InvalidPolygonError, SchemaValidationError
from ..core.identity import content_hash
from ..core.types import (
    Frame,
    OrientationMode,
    Point2,
    validate_polygon_points,
)
from ..core.units import (
    LengthUnit,
    Provenance,
    SpatialDomain,
    ensure_finite_array,
    ensure_non_negative,
    ensure_positive,
    require_same_domain,
    unit_of_domain,
)
from .frames import Transform2D

__all__ = [
    "TOOL_PROFILE_SCHEMA_VERSION",
    "ToolGeometry",
    "ToolProfile",
    "tool_geometry_from_config",
]

TOOL_PROFILE_SCHEMA_VERSION = 1
"""Version del esquema serializado del perfil de herramienta."""


@dataclass(frozen=True, slots=True, eq=False)
class ToolGeometry:
    """Geometria de la herramienta en el marco `B`.

    Attributes:
        tool_id: Identificador estable de la geometria.
        collision_polygon: Anillo `(n, 2)` en `B` usado para no colision; copia
            propia de solo lectura.
        cutting_polygon: Anillo `(m, 2)` en `B` de la huella activa de corte;
            copia propia de solo lectura, cubierta por `collision_polygon`.
        pivot: Referencia local de `B`; **debe** ser `(0, 0)` (R5). El
            desplazamiento de la cuchilla respecto al pivote se expresa en los
            poligonos, no moviendo el pivote. El pivote mecanico de un brazo es otro
            marco y no se modela aqui.
        orientation_mode: `fixed` en H1.
        domain: Dominio espacial de las coordenadas.
    """

    tool_id: str
    collision_polygon: np.ndarray
    cutting_polygon: np.ndarray
    pivot: Point2
    orientation_mode: OrientationMode = OrientationMode.FIXED
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC
    provenance: Provenance = Provenance.SYNTHETIC

    def __post_init__(self) -> None:
        if not str(self.tool_id).strip():
            raise SchemaValidationError("ToolGeometry.tool_id: cadena vacia")
        collision = validate_polygon_points(
            self.collision_polygon, "ToolGeometry.collision_polygon"
        )
        cutting = validate_polygon_points(self.cutting_polygon, "ToolGeometry.cutting_polygon")
        object.__setattr__(self, "collision_polygon", collision)
        object.__setattr__(self, "cutting_polygon", cutting)
        object.__setattr__(self, "orientation_mode", OrientationMode(self.orientation_mode))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        if not isinstance(self.pivot, Point2):
            raise SchemaValidationError("ToolGeometry.pivot: se espera Point2")
        if self.pivot.frame is not Frame.TOOL:
            raise SchemaValidationError(
                f"ToolGeometry.pivot debe estar en el marco {Frame.TOOL.value}, "
                f"se recibio {self.pivot.frame.value}"
            )
        require_same_domain(self.domain, self.pivot.domain, context="ToolGeometry")
        if self.pivot.x != 0.0 or self.pivot.y != 0.0:
            raise SchemaValidationError(
                f"ToolGeometry({self.tool_id}): el pivote es el origen de B y debe valer "
                f"(0, 0), se recibio ({self.pivot.x}, {self.pivot.y}); exprese el "
                "desplazamiento de la cuchilla en sus poligonos"
            )
        # Inclusion exacta, sin tolerancia: el contacto de bordes esta permitido
        # (el corte sobre el contorno sigue dentro de la huella de colision), la
        # protrusion no, por pequena que sea y en cualquier unidad (R11).
        if not Polygon(collision).covers(Polygon(cutting)):
            raise InvalidPolygonError(
                f"ToolGeometry({self.tool_id}): la huella de corte sobresale de la huella "
                "de colision almacenada; pueden diferir, pero el corte debe quedar cubierto"
            )

    # --- constructores ------------------------------------------------------

    @classmethod
    def rectangular(
        cls,
        tool_id: str,
        length_x: float,
        width_y: float,
        pivot_offset: Sequence[float] = (0.0, 0.0),
        cutting_inset: float = 0.0,
        orientation_mode: OrientationMode = OrientationMode.FIXED,
        domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC,
        provenance: Provenance = Provenance.SYNTHETIC,
    ) -> ToolGeometry:
        """Cuchilla rectangular con el lado largo sobre `x_B` (§5 E1).

        El rectangulo se centra en el pivote y luego se desplaza por
        `-pivot_offset`, de modo que el origen de `B` siempre coincide con el
        pivote declarado.
        """
        length_x = ensure_positive(length_x, "length_x")
        width_y = ensure_positive(width_y, "width_y")
        cutting_inset = ensure_non_negative(cutting_inset, "cutting_inset")
        # Forma exacta (2,) y finitud antes de indexar: nada de truncar ni IndexError (N1).
        offset_x, offset_y = _planar_vector(pivot_offset, "pivot_offset")
        if 2.0 * cutting_inset >= min(length_x, width_y):
            raise InvalidPolygonError(
                f"cutting_inset ({cutting_inset}) degenera la huella de corte de una "
                f"cuchilla {length_x}x{width_y}"
            )
        collision = _centered_rectangle(length_x, width_y, offset_x, offset_y)
        cutting = _centered_rectangle(
            length_x - 2.0 * cutting_inset, width_y - 2.0 * cutting_inset, offset_x, offset_y
        )
        return cls(
            tool_id=tool_id,
            collision_polygon=collision,
            cutting_polygon=cutting,
            pivot=Point2(x=0.0, y=0.0, frame=Frame.TOOL, domain=domain),
            orientation_mode=orientation_mode,
            domain=domain,
            provenance=provenance,
        )

    # --- propiedades --------------------------------------------------------

    @property
    def unit(self) -> LengthUnit:
        return unit_of_domain(self.domain)

    @property
    def collision_shape(self) -> Polygon:
        return Polygon(self.collision_polygon)

    @property
    def cutting_shape(self) -> Polygon:
        return Polygon(self.cutting_polygon)

    @property
    def collision_area(self) -> float:
        return float(self.collision_shape.area)

    @property
    def cutting_area(self) -> float:
        return float(self.cutting_shape.area)

    @property
    def extent_x(self) -> float:
        """Extension de la huella de colision sobre `x_B`."""
        return float(np.ptp(self.collision_polygon[:, 0]))

    @property
    def extent_y(self) -> float:
        """Extension de la huella de colision sobre `y_B`."""
        return float(np.ptp(self.collision_polygon[:, 1]))

    def circumscribed_radius(self) -> float:
        """Radio del disco circunscrito centrado en el origen de `B` (el pivote).

        Es el mismo origen respecto del cual se rota y traslada la huella, de modo
        que el disco contiene la huella en cualquier orientacion (R5).
        Aproximacion conservadora admitida en H1 para inflar obstaculos; su
        conservadurismo puede rechazar rutas validas (§3.4).
        """
        return float(np.max(np.linalg.norm(self.collision_polygon, axis=1)))

    def inflated_collision_shape(self, distance: float) -> Polygon:
        """Huella de colision inflada por `distance`, de forma conservadora.

        Se usa cuando la politica de incertidumbre infla la herramienta en lugar de
        las exclusiones. La union con esquinas en inglete contiene el buffer
        circular exacto: nunca subestima la huella inflada.
        """
        distance = ensure_non_negative(distance, "distance")
        if distance == 0.0:
            return self.collision_shape
        return self.collision_shape.buffer(distance, join_style=JOIN_STYLE.mitre, mitre_limit=10.0)

    # --- transformacion ----------------------------------------------------

    def _placed(self, polygon: np.ndarray, transform: Transform2D) -> Polygon:
        if transform.child is not Frame.TOOL:
            raise SchemaValidationError(
                f"se espera una transformacion con child={Frame.TOOL.value}, "
                f"se recibio child={transform.child.value}"
            )
        require_same_domain(self.domain, transform.domain, context="ToolGeometry.placed")
        return Polygon(transform.apply_array(polygon))

    def collision_polygon_in(self, transform: Transform2D) -> Polygon:
        """Huella de colision llevada al marco padre de `transform`."""
        return self._placed(self.collision_polygon, transform)

    def cutting_polygon_in(self, transform: Transform2D) -> Polygon:
        """Huella activa de corte llevada al marco padre de `transform`."""
        return self._placed(self.cutting_polygon, transform)

    def is_convex(self) -> bool:
        """`True` si la huella de colision almacenada es convexa (sin tolerancia relativa)."""
        shape = self.collision_shape
        return math.isclose(shape.convex_hull.area, shape.area, rel_tol=0.0, abs_tol=1e-15)

    def is_cut_convex(self) -> bool:
        """`True` si la huella activa de corte es convexa.

        El barrido exacto por envolvente convexa (E2) solo vale para huellas
        convexas: un corte concavo se declara capacidad no soportada en lugar de
        sobreestimar su cobertura (A1E2-R2).
        """
        shape = self.cutting_shape
        return math.isclose(shape.convex_hull.area, shape.area, rel_tol=0.0, abs_tol=1e-15)

    def identity_payload(self) -> dict[str, Any]:
        """Contenido que define la identidad del perfil (geometria, unidades, procedencia)."""
        return {
            "schema_version": TOOL_PROFILE_SCHEMA_VERSION,
            "tool_id": self.tool_id,
            "collision_polygon": self.collision_polygon.tolist(),
            "cutting_polygon": self.cutting_polygon.tolist(),
            "domain": self.domain.value,
            "provenance": self.provenance.value,
        }

    def content_hash(self) -> str:
        """SHA-256 del contenido: cambia si cambia cualquier vertice, unidad o procedencia."""
        return content_hash(self.identity_payload())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "collision_polygon": self.collision_polygon.tolist(),
            "cutting_polygon": self.cutting_polygon.tolist(),
            "pivot": self.pivot.to_dict(),
            "orientation_mode": self.orientation_mode.value,
            "domain": self.domain.value,
            "unit": self.unit.value,
            "provenance": self.provenance.value,
            "extent_x": self.extent_x,
            "extent_y": self.extent_y,
            "collision_area": self.collision_area,
            "cutting_area": self.cutting_area,
            "circumscribed_radius": self.circumscribed_radius(),
            "schema_version": TOOL_PROFILE_SCHEMA_VERSION,
            "content_hash": self.content_hash(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ToolGeometry:
        """Reconstruye el perfil exportado por `to_dict` y verifica su identidad.

        Comprueba version de esquema, unidad coherente con el dominio y que todos los
        campos derivados (extensiones, areas, radio, hash) coincidan exactamente con
        los recalculados: una exportacion alterada se rechaza.

        Raises:
            SchemaValidationError: campo ausente, version o unidad incoherentes, o
                contenido que no coincide con la identidad declarada.
        """
        try:
            if payload["schema_version"] != TOOL_PROFILE_SCHEMA_VERSION:
                raise SchemaValidationError(
                    f"ToolGeometry: schema_version {payload['schema_version']!r} no soportada"
                )
            pivot = payload["pivot"]
            profile = cls(
                tool_id=payload["tool_id"],
                collision_polygon=payload["collision_polygon"],
                cutting_polygon=payload["cutting_polygon"],
                pivot=Point2(
                    x=pivot["x"], y=pivot["y"], frame=pivot["frame"], domain=pivot["domain"]
                ),
                orientation_mode=payload["orientation_mode"],
                domain=payload["domain"],
                provenance=payload["provenance"],
            )
        except KeyError as exc:
            raise SchemaValidationError(f"ToolGeometry: falta el campo {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"ToolGeometry: contenido invalido ({exc})") from exc
        if payload.get("unit") != profile.unit.value:
            raise SchemaValidationError(
                f"ToolGeometry: unidad {payload.get('unit')!r} incoherente con el dominio "
                f"{profile.domain.value}"
            )
        if payload.get("content_hash") != profile.content_hash():
            raise SchemaValidationError("ToolGeometry: content_hash no coincide con el contenido")
        if profile.to_dict() != payload:
            raise SchemaValidationError(
                "ToolGeometry: campos derivados (extensiones, areas, radio) alterados"
            )
        return profile


ToolProfile = ToolGeometry
"""Nombre del contrato en el plan adaptado; es la misma clase, no una copia."""


def _planar_vector(values: Any, name: str) -> tuple[float, float]:
    """Vector 2D finito de dimension exacta; valida antes de indexar (R9)."""
    if isinstance(values, (str, bytes)):
        raise SchemaValidationError(f"{name}: se espera un vector numerico de 2 componentes")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise SchemaValidationError(f"{name}: no es un vector numerico ({values!r})") from exc
    if array.shape != (2,):
        raise SchemaValidationError(
            f"{name}: se exigen exactamente 2 componentes, se recibio forma {array.shape}"
        )
    finite = ensure_finite_array(array, name)
    return float(finite[0]), float(finite[1])


def _centered_rectangle(
    length_x: float, width_y: float, offset_x: float, offset_y: float
) -> np.ndarray:
    half_x, half_y = 0.5 * length_x, 0.5 * width_y
    return np.array(
        [
            [-half_x - offset_x, -half_y - offset_y],
            [half_x - offset_x, -half_y - offset_y],
            [half_x - offset_x, half_y - offset_y],
            [-half_x - offset_x, half_y - offset_y],
        ],
        dtype=np.float64,
    )


def tool_geometry_from_config(config: ToolConfig) -> ToolGeometry:
    """Construye la geometria declarada en `configs/tool.yaml`."""
    if not isinstance(config, ToolConfig):
        raise SchemaValidationError("tool_geometry_from_config: se espera ToolConfig")
    return ToolGeometry.rectangular(
        tool_id=config.tool_id,
        length_x=config.length_x_m,
        width_y=config.width_y_m,
        pivot_offset=config.pivot_offset_m,
        cutting_inset=config.cutting_inset_m,
        orientation_mode=config.orientation_mode,
        domain=config.domain,
    )
