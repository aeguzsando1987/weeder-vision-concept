"""Estado de la cuchilla virtual (`docs/technical_plan.md` §3.2, §3.3).

`ToolGeometry` vive en `geometry/tool_profile.py` desde A1 (el planificador no debe
depender del simulador) y se reexporta aqui sin cambios: hay una sola
implementacion, y las importaciones de E1 siguen funcionando.

Nada de este modulo simula elevacion ni profundidad: `ToolState.depth_z` esta
reservada y vale `None` en H1 (§2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from ..core.errors import SchemaValidationError
from ..core.types import Frame, Point2, SimulationState
from ..core.units import ensure_finite, ensure_non_negative
from ..geometry.frames import Transform2D
from ..geometry.tool_profile import ToolGeometry, _planar_vector, tool_geometry_from_config

__all__ = ["ToolGeometry", "ToolState", "tool_geometry_from_config", "world_footprints"]


@dataclass(frozen=True, slots=True)
class ToolState:
    """Estado instantaneo de la herramienta en el marco declarado.

    `position` es la posicion del pivote; `velocity` y `acceleration` son derivadas
    relativas a ese marco. `depth_z` esta reservada: en H1 debe ser `None`.
    """

    time_s: float
    position: Point2
    theta: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)
    acceleration: tuple[float, float] = (0.0, 0.0)
    theta_rate: float = 0.0
    contact: bool = False
    machine_state: SimulationState = SimulationState.IDLE
    depth_z: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time_s", ensure_non_negative(self.time_s, "ToolState.time_s"))
        if not isinstance(self.position, Point2):
            raise SchemaValidationError("ToolState.position: se espera Point2")
        if self.position.frame not in (Frame.VEHICLE, Frame.WORLD):
            raise SchemaValidationError(
                "ToolState.position debe expresarse en el marco V o W, "
                f"se recibio {self.position.frame.value}"
            )
        object.__setattr__(self, "theta", ensure_finite(self.theta, "ToolState.theta"))
        object.__setattr__(self, "velocity", _planar_vector(self.velocity, "ToolState.velocity"))
        object.__setattr__(
            self, "acceleration", _planar_vector(self.acceleration, "ToolState.acceleration")
        )
        object.__setattr__(
            self, "theta_rate", ensure_finite(self.theta_rate, "ToolState.theta_rate")
        )
        if not isinstance(self.contact, (bool, np.bool_)):
            raise SchemaValidationError(
                f"ToolState.contact: se exige un booleano, se recibio {self.contact!r} "
                f"({type(self.contact).__name__}); el contacto no se infiere por veracidad"
            )
        object.__setattr__(self, "contact", bool(self.contact))
        object.__setattr__(self, "machine_state", SimulationState(self.machine_state))
        if self.depth_z is not None:
            raise SchemaValidationError(
                "ToolState.depth_z debe ser None en H1: la interfaz de profundidad Z esta "
                "reservada y no se simulan elevaciones instantaneas (§2)"
            )

    @property
    def frame(self) -> Frame:
        return self.position.frame

    @property
    def speed(self) -> float:
        """Modulo de la velocidad relativa en el marco declarado."""
        return math.hypot(*self.velocity)

    def transform_to_tool(self) -> Transform2D:
        """`T_{frame}B` que coloca el marco de la herramienta segun este estado."""
        return Transform2D(
            parent=self.position.frame,
            child=Frame.TOOL,
            tx=self.position.x,
            ty=self.position.y,
            theta=self.theta,
            domain=self.position.domain,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_s": self.time_s,
            "position": self.position.to_dict(),
            "theta_rad": self.theta,
            "velocity": list(self.velocity),
            "acceleration": list(self.acceleration),
            "theta_rate": self.theta_rate,
            "contact": self.contact,
            "machine_state": self.machine_state.value,
            "depth_z": self.depth_z,
            "frame": self.frame.value,
        }


def world_footprints(
    geometry: ToolGeometry,
    state: ToolState,
    world_from_vehicle_transform: Transform2D | None = None,
) -> tuple[Polygon, Polygon]:
    """Huellas de colision y de corte en el marco `W`.

    Si el estado esta expresado en `V`, se exige `T_WV` para situarlo en el suelo:
    el avance del tractor entra una sola vez, a traves de esa transformacion
    (§3.2). Si el estado ya esta en `W`, `T_WV` debe omitirse.
    """
    tool_transform = state.transform_to_tool()
    if state.frame is Frame.VEHICLE:
        if world_from_vehicle_transform is None:
            raise SchemaValidationError(
                "estado en el marco V: se requiere T_WV para obtener huellas en W"
            )
        if (
            world_from_vehicle_transform.parent is not Frame.WORLD
            or world_from_vehicle_transform.child is not Frame.VEHICLE
        ):
            raise SchemaValidationError(
                "se espera T_WV (parent=W, child=V), se recibio "
                f"T_{world_from_vehicle_transform.parent.value}"
                f"{world_from_vehicle_transform.child.value}"
            )
        tool_transform = world_from_vehicle_transform.compose(tool_transform)
    elif world_from_vehicle_transform is not None:
        raise SchemaValidationError(
            "estado ya expresado en W: aportar T_WV sumaria el avance del tractor dos veces"
        )
    return (
        geometry.collision_polygon_in(tool_transform),
        geometry.cutting_polygon_in(tool_transform),
    )
