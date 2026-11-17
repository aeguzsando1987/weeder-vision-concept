"""Catalogo congelado de fixtures M0-P (orden §6: fijado antes de ajustar el pipeline).

- `cfr_640`: 640x480, 40 cuadros CFR (base 1/15), suelo desplazandose (+1, -6) px/cuadro.
- `vfr_960`: 960x540, 53 cuadros VFR (base 1/90000, intervalos 3000/3600/2400/4500),
  suelo desplazandose (-1, -8) px/cuadro.

Cada uno tiene dos hileras (izquierda en u=0.225, derecha en u=0.775 del ancho) y
variante `blocked`: lineas de plantas que cierran la banda derecha cada 200 px, de
modo que toda ventana visible contiene al menos una fila sin paso para la cuchilla.

Resultados esperados (congelados): las variantes libres producen ruta vigente con
desvio respecto de una exclusion; las bloqueadas no producen ruta.
"""

from __future__ import annotations

from fractions import Fraction

from .synthetic import FixtureSpec, PlantSpec

__all__ = ["EXPECTED", "fixture_spec"]


def _rows(width: int, radius: float, right_v: list[float], left_v: list[float]) -> list[PlantSpec]:
    plants = [PlantSpec(0.775 * width, v, radius, f"r{i}") for i, v in enumerate(right_v)]
    plants += [PlantSpec(0.225 * width, v, radius, f"l{i}") for i, v in enumerate(left_v)]
    return plants


def _blockers(width: int, radius: float, v_values: list[float]) -> list[PlantSpec]:
    plants = []
    for row, v in enumerate(v_values):
        for column, u_norm in enumerate((0.625, 0.775, 0.925)):
            plants.append(PlantSpec(u_norm * width, v, radius, f"b{row}_{column}"))
    return plants


def fixture_spec(name: str) -> FixtureSpec:
    """Construye un fixture del catalogo por nombre."""
    if name in ("cfr_640", "cfr_640_blocked"):
        plants = _rows(640, 18.0, [80, 220, 360, 500, 640], [150, 300, 450, 600])
        if name.endswith("blocked"):
            plants = _blockers(640, 18.0, [60, 260, 460, 660]) + plants[5:]
        return FixtureSpec(640, 480, 40, (1, -6), tuple(plants), Fraction(1, 15))
    if name in ("vfr_960", "vfr_960_blocked"):
        cycle = (3000, 3600, 2400, 4500)
        pts = [0]
        for index in range(1, 53):
            pts.append(pts[-1] + cycle[index % len(cycle)])
        plants = _rows(960, 27.0, [100, 260, 420, 580, 740, 900], [180, 340, 500, 660, 820])
        if name.endswith("blocked"):
            plants = _blockers(960, 27.0, [80, 280, 480, 680, 880]) + plants[6:]
        return FixtureSpec(
            960, 540, 53, (-1, -8), tuple(plants), Fraction(1, 90000), pts=tuple(pts)
        )
    raise KeyError(f"fixture desconocido: {name}")


EXPECTED = {
    "cfr_640": "route",
    "vfr_960": "route",
    "cfr_640_blocked": "no_route",
    "vfr_960_blocked": "no_route",
}
