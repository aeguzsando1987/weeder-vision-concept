"""Seleccion determinista de muestras (E0, orden §3).

Muestras uniformes sobre los cuadros **decodificados** mas cuadros extremos de
metricas indicativas (posible desenfoque, cambio de iluminacion, dispersion de
flujo, pocos puntos validos, mucha o poca vegetacion). Desempates por indice;
misma entrada y configuracion producen la misma seleccion.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

__all__ = ["SampleChoice", "select_samples"]


@dataclass(frozen=True)
class SampleChoice:
    index: int
    reasons: tuple[str, ...]


def select_samples(
    frame_count: int,
    uniform_count: int,
    criteria: Sequence[tuple[str, Sequence[float | None], Callable[[float], float]]],
    per_criterion: int,
    min_separation: int,
) -> list[SampleChoice]:
    """Selecciona indices de muestra.

    Args:
        frame_count: Cuadros decodificados.
        uniform_count: Muestras uniformes (incluyen primer y ultimo cuadro).
        criteria: `(nombre, valores_por_cuadro, clave)`; se eligen los cuadros de
            menor `clave(valor)`. Los `None` se ignoran.
        per_criterion: Cuadros extremos por criterio.
        min_separation: Separacion minima entre una muestra de problema y
            cualquier muestra ya elegida.
    """
    if frame_count <= 0:
        return []
    reasons: dict[int, list[str]] = {}
    count = min(uniform_count, frame_count)
    for position, index in enumerate(np.linspace(0, frame_count - 1, count).round().astype(int)):
        reasons.setdefault(int(index), []).append(f"uniform_{position:02d}")
    for name, values, key in criteria:
        ranked = sorted(
            (key(float(value)), index)
            for index, value in enumerate(values)
            if value is not None and np.isfinite(value)
        )
        taken = 0
        for _, index in ranked:
            if taken >= per_criterion:
                break
            if any(abs(index - chosen) < min_separation for chosen in reasons):
                continue
            reasons.setdefault(index, []).append(name)
            taken += 1
    return [SampleChoice(index, tuple(reasons[index])) for index in sorted(reasons)]
