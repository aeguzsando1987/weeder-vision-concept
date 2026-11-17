"""Suavizado C2 de candidatos poligonales (orden A1/E2 §4.2).

Estrategia `quintic_lane_change`: cada arista `(x_i, y_i) -> (x_{i+1}, y_{i+1})` del
candidato (con `y` estrictamente creciente) se convierte en un tramo

    x(u) = x_i + Δx · h(u),   h(u) = 10u^3 - 15u^4 + 6u^5
    y(u) = y_i + Δy · u

con `u` en `[0, 1]` y longitud de tramo en `s` proporcional a `Δy`
(`L_i = Δy_i / (y_n - y_0)`). Entonces:

- `dy/ds = y_n - y_0` es constante y `d2y/ds2 = 0` en todo el camino;
- `h'(0) = h'(1) = h''(0) = h''(1) = 0`, asi que `dx/ds = d2x/ds2 = 0` en cada union;

por tanto `p(s)` es C2 en `s` global, el progreso en `y` es estrictamente
creciente y los extremos coinciden exactamente con inicio y meta. La curva no es
la polilinea: se revalida completa (un tramo en S puede invadir donde la cuerda no).
Para cada tramo la curva queda dentro del rectangulo `[x_i, x_{i+1}] x [y_i, y_{i+1}]`,
porque `h` es monotona en `[0, 1]`.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..core.errors import SchemaValidationError
from ..core.trajectory import PathSegment

__all__ = ["SMOOTHSTEP_COEFFICIENTS", "quintic_lane_change_segments"]

SMOOTHSTEP_COEFFICIENTS = np.array([0.0, 0.0, 0.0, 10.0, -15.0, 6.0])
"""`h(u)` en base ascendente: transicion 0 -> 1 con h', h'' nulas en los extremos."""


def quintic_lane_change_segments(
    waypoints: Sequence[tuple[float, float]],
) -> tuple[PathSegment, ...]:
    """Tramos quinticos C2 a partir de una polilinea con `y` estrictamente creciente.

    Raises:
        SchemaValidationError: menos de dos puntos o `y` no estrictamente creciente.
    """
    points = np.asarray(waypoints, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2:
        raise SchemaValidationError("suavizado: se exigen al menos dos puntos (x, y)")
    if not np.all(np.isfinite(points)):
        raise SchemaValidationError("suavizado: puntos no finitos")
    dy = np.diff(points[:, 1])
    if np.any(dy <= 0.0):
        raise SchemaValidationError("suavizado: y debe ser estrictamente creciente")
    total = float(points[-1, 1] - points[0, 1])
    boundaries = [0.0]
    for index in range(1, points.shape[0] - 1):
        boundaries.append(float((points[index, 1] - points[0, 1]) / total))
    boundaries.append(1.0)
    segments = []
    for index in range(points.shape[0] - 1):
        x0, y0 = points[index]
        x1, y1 = points[index + 1]
        x_coefficients = SMOOTHSTEP_COEFFICIENTS * (x1 - x0)
        x_coefficients[0] = x0
        y_coefficients = np.array([y0, y1 - y0, 0.0, 0.0, 0.0, 0.0])
        segments.append(
            PathSegment(
                s_start=boundaries[index],
                s_end=boundaries[index + 1],
                x_coefficients=x_coefficients,
                y_coefficients=y_coefficients,
            )
        )
    return tuple(segments)
