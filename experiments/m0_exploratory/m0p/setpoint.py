"""Senal para el brazo derivada de la ruta vigente (orden V1, paso 3).

Por cuadro se muestrea la cadena vigente en la **linea de herramienta** y se exporta la
consigna lateral que seguiria un actuador: donde debe estar la herramienta, de lado a
lado, en el instante del cuadro. Es una salida de vision:

- unidades en metros **supuestos** (`ground_assumed`); `tool_u_px` es la misma posicion
  en pixeles de la imagen, sin escala supuesta;
- `lateral_camera_m` se mide desde la columna central de la imagen (camara supuesta
  rigida al vehiculo); `offset_from_row_axis_m`, desde la linea de plantacion estimada;
- la tasa entre cuadros es la que exigiria la consigna, no una velocidad ejecutada ni
  una comprobacion mecanica (eso pertenece a A3/E3, en pausa);
- sin ruta vigente no se inventa consigna: el estado lo declara y los campos quedan vacios.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from precision_weeder.core.trajectory import TaskPath

from .profile import M0Profile
from .projection import SegmentProjection
from .route import chain_x_at

__all__ = ["FIELDS", "SetpointLog"]

FIELDS = (
    "frame_index",
    "timestamp_s",
    "segment_id",
    "state",
    "travel_m",
    "lateral_camera_m",
    "offset_from_row_axis_m",
    "tool_u_px",
    "lateral_step_m",
    "lateral_rate_m_s",
    "route_hash",
)

WORKING, WITHDRAWN, NO_ROUTE = "trabajando", "retirada", "sin_ruta"


@dataclass
class SetpointLog:
    """Acumula la consigna cuadro a cuadro y sus metricas de continuidad."""

    rows: list[dict[str, Any]] = field(default_factory=list)
    travel_m: float = 0.0
    travel_gaps: int = 0
    """Cuadros cuyo avance no se pudo medir (cambio o perdida de segmento `W`)."""
    max_step_m: float = 0.0
    max_rate_m_s: float = 0.0
    working_frames: int = 0
    _previous: dict[str, Any] | None = None
    _previous_tool_y: tuple[str, float] | None = None

    def update(
        self,
        frame_index: int,
        timestamp_s: float | None,
        projection: SegmentProjection | None,
        profile: M0Profile,
        paths: tuple[TaskPath, ...] | None,
        axis_x: float | None,
        route_hash: str | None,
    ) -> dict[str, Any]:
        """Consigna del cuadro; `paths` es la cadena vigente o `None`."""
        row: dict[str, Any] = dict.fromkeys(FIELDS)
        row.update(frame_index=frame_index, timestamp_s=timestamp_s)
        tool_y = None
        if projection is not None:
            row["segment_id"] = projection.segment_id
            tool_y = projection.row_v_to_ground_y(profile.tool_line.v_norm)
            previous = self._previous_tool_y
            if previous is not None and previous[0] == projection.segment_id:
                self.travel_m += tool_y - previous[1]
            elif self.rows:
                self.travel_gaps += 1
            self._previous_tool_y = (projection.segment_id, tool_y)
        else:
            self._previous_tool_y = None
            if self.rows:
                self.travel_gaps += 1
        row["travel_m"] = self.travel_m
        x_tool = chain_x_at(paths, tool_y) if paths and tool_y is not None else None
        last = self._previous
        if x_tool is None:
            was_working = last is not None and last["state"] == WORKING
            row["state"] = WITHDRAWN if was_working else NO_ROUTE
        else:
            row["state"] = WORKING
            row["route_hash"] = route_hash
            row["lateral_camera_m"] = x_tool - projection.column_u_to_ground_x(0.5)
            if axis_x is not None:
                row["offset_from_row_axis_m"] = x_tool - axis_x
            pixel = projection.ground_to_image(np.array([[x_tool, tool_y]]))
            row["tool_u_px"] = float(pixel[0][0])
            self.working_frames += 1
            consecutive = (
                last is not None
                and last["state"] == WORKING
                and last["segment_id"] == row["segment_id"]
                and last["frame_index"] == frame_index - 1
            )
            if consecutive:
                step = row["lateral_camera_m"] - last["lateral_camera_m"]
                row["lateral_step_m"] = step
                self.max_step_m = max(self.max_step_m, abs(step))
                if timestamp_s is not None and last["timestamp_s"] is not None:
                    dt = timestamp_s - last["timestamp_s"]
                    if dt > 0.0:
                        row["lateral_rate_m_s"] = step / dt
                        self.max_rate_m_s = max(self.max_rate_m_s, abs(step / dt))
        self._previous = row
        self.rows.append(row)
        return row

    def summary(self) -> dict[str, Any]:
        return {
            "domain": "ground_assumed",
            "working_frames": self.working_frames,
            "travel_m": self.travel_m,
            "frames_without_measured_travel": self.travel_gaps,
            "max_lateral_step_m": self.max_step_m,
            "max_lateral_rate_m_s": self.max_rate_m_s,
            "note": "consigna de vision; no es velocidad ejecutada ni validacion mecanica",
        }

    def write_csv(self, path: Path) -> None:
        with Path(path).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for row in self.rows:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
