"""Proyeccion supuesta imagen <-> suelo `W_segment` (orden M0-P §4.1).

Modelo: suelo plano paralelo a la imagen, escala uniforme **supuesta** `s` [m/px] y
movimiento del suelo en la imagen como traslacion acumulada `D_t` desde el cuadro
ancla del segmento. Un punto de suelo fijo que en el ancla se veia en `q` se ve en
el cuadro `t` en `p = q + D_t`. Sus coordenadas en `W_segment` son

    x = s * (q_u - c_u)
    y = s * sigma * (q_v - c_v),   sigma = +1 si el avance es hacia abajo, -1 si arriba

con `c` el centro del cuadro ancla. Asi `y` crece en el sentido de avance supuesto.
Es una transformacion afin explicita (escala por eje + traslacion), **no** un
`Transform2D` rigido ni una homografia calibrada; cambia en cada cuadro con `D_t`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from precision_weeder.core.errors import SchemaValidationError

__all__ = ["SegmentProjection"]


@dataclass(frozen=True)
class SegmentProjection:
    """Proyeccion de un segmento `W_segment` para un cuadro concreto."""

    segment_id: str
    anchor_frame: int
    frame_index: int
    scale_m_per_px: float
    forward_sign: int
    center_px: tuple[float, float]
    displacement_px: tuple[float, float]
    image_size: tuple[int, int]

    def __post_init__(self) -> None:
        if not (math.isfinite(self.scale_m_per_px) and self.scale_m_per_px > 0):
            raise SchemaValidationError("SegmentProjection: escala no finita o <= 0")
        if self.forward_sign not in (-1, 1):
            raise SchemaValidationError("SegmentProjection: forward_sign debe ser +1 o -1")
        values = (*self.center_px, *self.displacement_px)
        if not all(math.isfinite(v) for v in values):
            raise SchemaValidationError("SegmentProjection: centro o desplazamiento no finitos")
        width, height = self.image_size
        if width <= 0 or height <= 0:
            raise SchemaValidationError("SegmentProjection: tamano de imagen invalido")

    @property
    def _diag(self) -> np.ndarray:
        return np.array([self.scale_m_per_px, self.scale_m_per_px * self.forward_sign])

    def image_to_ground(self, points_px: np.ndarray) -> np.ndarray:
        """Pixeles del cuadro `frame_index` -> metros en `W_segment` (`(n, 2)`)."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        anchor = points - np.asarray(self.displacement_px) - np.asarray(self.center_px)
        return anchor * self._diag

    def ground_to_image(self, points_m: np.ndarray) -> np.ndarray:
        """Metros en `W_segment` -> pixeles del cuadro `frame_index` (`(n, 2)`)."""
        points = np.asarray(points_m, dtype=np.float64).reshape(-1, 2)
        return points / self._diag + np.asarray(self.center_px) + np.asarray(self.displacement_px)

    def row_v_to_ground_y(self, v_norm: float) -> float:
        """`y` de suelo de la fila normalizada `v_norm` del cuadro actual."""
        _, height = self.image_size
        return float(self.image_to_ground(np.array([[0.0, v_norm * height]]))[0, 1])

    def column_u_to_ground_x(self, u_norm: float) -> float:
        """`x` de suelo de la columna normalizada `u_norm` del cuadro actual."""
        width, _ = self.image_size
        return float(self.image_to_ground(np.array([[u_norm * width, 0.0]]))[0, 0])

    def visible_ground_box(self) -> tuple[float, float, float, float]:
        """Rectangulo de suelo visible `(x_min, x_max, y_min, y_max)` en este cuadro."""
        width, height = self.image_size
        corners = self.image_to_ground(np.array([[0, 0], [width, 0], [width, height], [0, height]]))
        return (
            float(corners[:, 0].min()),
            float(corners[:, 0].max()),
            float(corners[:, 1].min()),
            float(corners[:, 1].max()),
        )

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "anchor_frame": self.anchor_frame,
            "frame_index": self.frame_index,
            "model": "suelo plano paralelo a la imagen, escala uniforme supuesta + traslacion",
            "scale_m_per_px": self.scale_m_per_px,
            "forward_sign": self.forward_sign,
            "center_px": list(self.center_px),
            "displacement_px": list(self.displacement_px),
            "image_size": list(self.image_size),
            "domain": "ground_assumed",
        }
