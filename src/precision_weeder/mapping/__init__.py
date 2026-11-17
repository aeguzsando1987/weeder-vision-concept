"""Mapa local en suelo `W` (A1/E2).

`mapping.local_map` no depende de escenarios ni del simulador; el adaptador de
escenas sinteticas vive aparte (`mapping.synthetic_adapter`) para que importar el
planificador no cargue el catalogo de escenas.
"""

from __future__ import annotations

from .local_map import LOCAL_MAP_SCHEMA_VERSION, LocalMap, rectangle_ring

__all__ = ["LOCAL_MAP_SCHEMA_VERSION", "LocalMap", "rectangle_ring"]
