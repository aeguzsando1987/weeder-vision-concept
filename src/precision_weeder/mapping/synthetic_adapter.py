"""Adaptador de escena sintetica E1 a `LocalMap` (orden A1/E2 §4.1).

La escena sintetica es verdad de referencia completa: todas sus plantas son
conocidas. Por eso el adaptador declara conocido el corredor entero, con la base
`synthetic_ground_truth`, o `certified_free` para el corredor vacio certificado.
Esa declaracion es explicita y queda registrada; no se deduce de la ausencia de
plantas. Las regiones desconocidas interiores se pasan aparte y se tratan como
prohibidas.

Las exclusiones, la politica de incertidumbre y `tool_inflation` se copian tal
cual de la escena: el margen no se recalcula ni se aplica dos veces.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..core.errors import UnitMismatchError
from ..core.trajectory import TASK_DOMAINS
from ..scenarios.synthetic import SyntheticScene
from .local_map import LocalMap, rectangle_ring

__all__ = ["local_map_from_scene"]


def local_map_from_scene(
    scene: SyntheticScene,
    *,
    unknown_regions: Sequence[Sequence[Sequence[float]]] = (),
    map_id: str | None = None,
    segment_id: str = "W_synthetic",
    version: int = 1,
) -> LocalMap:
    """Mapa local de una escena sintetica metrica.

    Raises:
        UnitMismatchError: si la escena no es metrica (por ejemplo, el SVG grafico).
    """
    if scene.domain not in TASK_DOMAINS:
        raise UnitMismatchError(
            f"escena {scene.scene_id} en {scene.domain.value}: el planificador trabaja en "
            "metros de suelo; una escena grafica o en pixeles no se planifica como tarea"
        )
    corridor = scene.corridor
    basis = "certified_free" if scene.free_space_certified else "synthetic_ground_truth"
    notes = [f"escena de origen: {scene.scene_id} (semilla {scene.seed})"]
    if unknown_regions:
        notes.append("regiones desconocidas interiores declaradas: prohibidas para la huella")
    return LocalMap(
        map_id=map_id or f"map_{scene.scene_id}",
        version=version,
        segment_id=segment_id,
        domain=scene.domain,
        corridor=corridor,
        known_region=rectangle_ring(corridor.x_min, corridor.x_max, corridor.y_min, corridor.y_max),
        known_region_basis=basis,
        unknown_regions=tuple(unknown_regions),
        exclusions=scene.exclusions,
        uncertainty=scene.uncertainty,
        tool_inflation=scene.tool_inflation,
        free_space_certified=scene.free_space_certified,
        provenance=scene.provenance,
        notes=tuple(notes),
    )
