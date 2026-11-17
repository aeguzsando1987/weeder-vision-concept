"""Casos de aceptacion E2 y generador congelado de escenarios (orden A1/E2 §6).

Cada caso fija mapa, inicio/meta del origen `B` en `W`, orientacion y el resultado
esperado **antes** de ajustar el algoritmo. Los extremos pertenecen al caso, no a la
escena E1: las configuraciones E1 no se modifican.

Los nombres de caso existen solo en este catalogo; el planificador no los ve ni los
trata de forma especial.

Oraculos de los casos construidos aqui (cuchilla de E1: 0.20 m en `x`, 0.08 m en `y`):

- `straight_lane_free`: plantas en `x = -0.25` con radio total 0.18 m ocupan
  `x <= -0.07`. Una traslacion recta en `x = +0.25` barre exactamente el rectangulo
  `[0.15, 0.35] x [0.06, 4.94]`, a 0.22 m de los circulos exactos: libre.
- `confined_straight_passage`: dos plantas en `y = 2.0`, `x = ±0.255`, radio total
  0.18 m. En la fila `y = 2.0` los huecos libres miden 0.065 (izquierda), 0.15
  (centro) y 0.065 m (derecha), todos menores que 0.20 m. Con orientacion fija 0, la
  seccion horizontal de la cuchilla en cualquier fila que cubra mide 0.20 m, y para
  ir de `y = 0.10` a `y = 4.90` la cuchilla debe cubrir la fila `y = 2.0`: no existe
  ruta. (Las exclusiones poligonales exteriores solo estrechan mas los huecos.)
- `unknown_band`: corredor vacio certificado con una franja desconocida de ancho
  completo en `y ∈ [2.0, 2.3]`; atravesarla exige pisar suelo desconocido.
- `start_in_collision`: escena nominal con inicio en el centro de la primera planta.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

import numpy as np

from ..core.config import (
    AdvanceConfig,
    CorridorConfig,
    ExclusionConfig,
    PlantsConfig,
    RandomScenariosConfig,
    SceneConfig,
    UncertaintyConfig,
)
from ..core.types import Frame
from ..core.units import SpatialDomain
from ..mapping.local_map import LocalMap, rectangle_ring
from ..mapping.synthetic_adapter import local_map_from_scene
from .fixtures import build_fixture
from .synthetic import build_scene

__all__ = [
    "PLANNING_CASES",
    "PlanningCase",
    "build_planning_case",
    "planning_case_names",
    "random_planning_case",
    "scene_config",
]

_DOMAIN: Final[SpatialDomain] = SpatialDomain.GROUND_METRIC_SYNTHETIC


@dataclass(frozen=True)
class PlanningCase:
    """Caso de planificacion con resultado esperado congelado."""

    name: str
    local_map: LocalMap
    start_xy_m: tuple[float, float]
    goal_xy_m: tuple[float, float]
    theta_rad: float
    expected_status: str
    expected_reason: str | None
    oracle: str


def scene_config(
    scene_id: str,
    positions_y: tuple[float, ...],
    offsets_x: tuple[float, ...],
    *,
    width: float = 1.0,
    length: float = 5.0,
    seed: int = 20261025,
    free: bool = False,
    margin_m: float = 0.03,
) -> SceneConfig:
    """Escena metrica E2 con los parametros de exclusion de E1 (proteccion 0.15 m).

    `margin_m` es el margen de incertidumbre de las exclusiones: 0.03 m para inflar
    exclusiones, 0.0 para la politica `inflate=tool` (la cota pasa a la herramienta).
    """
    return SceneConfig(
        scene_id=scene_id,
        description=f"Caso E2 {scene_id}",
        domain=_DOMAIN,
        frame=Frame.WORLD,
        seed=seed,
        free_space_certified=free,
        corridor=CorridorConfig(width_m=width, length_m=length, center_x_m=0.0, start_y_m=0.0),
        plants=PlantsConfig(positions_y_m=positions_y, offsets_x_m=offsets_x, stem_radius_m=0.02),
        exclusion=ExclusionConfig(
            protection_radius_m=0.15, uncertainty_margin_m=margin_m, polygon_segments=64
        ),
        advance=AdvanceConfig(speed_mps=0.20, heading_rad=0.0),
    )


def _e1(name: str, uncertainty: UncertaintyConfig | None) -> LocalMap:
    return local_map_from_scene(build_fixture(name, uncertainty=uncertainty))


def _custom(config: SceneConfig, uncertainty: UncertaintyConfig | None, **kwargs) -> LocalMap:
    return local_map_from_scene(build_scene(config, uncertainty=uncertainty), **kwargs)


_NOMINAL_START: Final = (0.0, 0.10)
_NOMINAL_GOAL: Final = (0.0, 4.90)

CaseBuilder = Callable[[UncertaintyConfig | None], PlanningCase]


def _nominal(u: UncertaintyConfig | None) -> PlanningCase:
    return PlanningCase(
        "nominal_three_plants",
        _e1("nominal_three_plants", u),
        _NOMINAL_START,
        _NOMINAL_GOAL,
        0.0,
        "ACCEPTED",
        None,
        "escena E1 sin modificar; hay hueco lateral de 0.32 m para una cuchilla de 0.20 m",
    )


def _empty(u: UncertaintyConfig | None) -> PlanningCase:
    return PlanningCase(
        "empty_corridor_certified",
        _e1("empty_corridor_certified", u),
        _NOMINAL_START,
        _NOMINAL_GOAL,
        0.0,
        "ACCEPTED",
        None,
        "corredor vacio certificado: recta central libre",
    )


def _isolated(u: UncertaintyConfig | None) -> PlanningCase:
    return PlanningCase(
        "isolated_plant",
        _e1("isolated_plant", u),
        _NOMINAL_START,
        _NOMINAL_GOAL,
        0.0,
        "ACCEPTED",
        None,
        "una planta central; huecos laterales de 0.32 m",
    )


def _straight_lane(u: UncertaintyConfig | None) -> PlanningCase:
    config = scene_config("straight_lane_free", (1.0, 2.0, 3.0), (-0.25, -0.25, -0.25))
    return PlanningCase(
        "straight_lane_free",
        _custom(config, u),
        (0.25, 0.10),
        (0.25, 4.90),
        0.0,
        "ACCEPTED",
        None,
        "recta x=0.25: barrido exacto [0.15,0.35]x[0.06,4.94] a 0.22 m de las exclusiones",
    )


def _blocked(u: UncertaintyConfig | None) -> PlanningCase:
    return PlanningCase(
        "blocked_corridor",
        _e1("blocked_corridor", u),
        _NOMINAL_START,
        _NOMINAL_GOAL,
        0.0,
        "ABSTAIN",
        "NO_PATH_FOUND",
        "escena E1: huecos de 0.12 m laterales y 0.04 m central < 0.20 m de cuchilla",
    )


def _start_collision(u: UncertaintyConfig | None) -> PlanningCase:
    return PlanningCase(
        "start_in_collision",
        _e1("nominal_three_plants", u),
        (0.0, 1.0),
        _NOMINAL_GOAL,
        0.0,
        "ABSTAIN",
        "START_INADMISSIBLE",
        "inicio sobre el centro de la planta P1 (y=1.0)",
    )


def _unknown_band(u: UncertaintyConfig | None) -> PlanningCase:
    config = scene_config("unknown_band", (), (), free=True)
    band = rectangle_ring(-0.5, 0.5, 2.0, 2.3)
    return PlanningCase(
        "unknown_band",
        _custom(config, u, unknown_regions=(band,)),
        _NOMINAL_START,
        _NOMINAL_GOAL,
        0.0,
        "ABSTAIN",
        "UNKNOWN_SPACE",
        "franja desconocida de ancho completo en y=[2.0,2.3]: cruzarla pisa suelo desconocido",
    )


def _confined(u: UncertaintyConfig | None) -> PlanningCase:
    config = scene_config("confined_straight_passage", (2.0, 2.0), (-0.255, 0.255))
    return PlanningCase(
        "confined_straight_passage",
        _custom(config, u),
        _NOMINAL_START,
        _NOMINAL_GOAL,
        0.0,
        "ABSTAIN",
        "NO_PATH_FOUND",
        "fila y=2.0: huecos 0.065/0.15/0.065 m < 0.20 m de seccion de la cuchilla",
    )


PLANNING_CASES: Final[dict[str, CaseBuilder]] = {
    "nominal_three_plants": _nominal,
    "empty_corridor_certified": _empty,
    "isolated_plant": _isolated,
    "straight_lane_free": _straight_lane,
    "blocked_corridor": _blocked,
    "start_in_collision": _start_collision,
    "unknown_band": _unknown_band,
    "confined_straight_passage": _confined,
}
"""Casos obligatorios E2-01 (positivos) y E2-02 (negativos), con resultado congelado."""


def planning_case_names() -> tuple[str, ...]:
    return tuple(PLANNING_CASES)


def build_planning_case(name: str, uncertainty: UncertaintyConfig | None = None) -> PlanningCase:
    """Construye un caso del catalogo con la politica de incertidumbre dada."""
    try:
        builder = PLANNING_CASES[name]
    except KeyError as exc:
        from ..core.errors import ScenarioError

        raise ScenarioError(
            f"caso desconocido '{name}'; disponibles: {', '.join(PLANNING_CASES)}"
        ) from exc
    return builder(uncertainty)


def random_planning_case(
    seed: int,
    config: RandomScenariosConfig,
    uncertainty: UncertaintyConfig | None = None,
) -> PlanningCase:
    """Escenario aleatorio reproducible (E2-08): depende solo de `seed` y `config`.

    El resultado esperado es desconocido (`expected_status="ANY"`): no se impone un
    porcentaje de exito; se verifican invariantes de lo que se acepte.
    """
    rng = np.random.default_rng(seed)
    width = float(rng.uniform(*config.corridor_width_m))
    count = int(rng.integers(config.plant_count[0], config.plant_count[1] + 1))
    length = config.corridor_length_m
    positions = tuple(
        float(v)
        for v in np.sort(
            rng.uniform(config.plant_y_margin_m, length - config.plant_y_margin_m, count)
        )
    )
    limit = config.plant_offset_fraction * width
    offsets = tuple(float(v) for v in rng.uniform(-limit, limit, count))
    config_random = scene_config(
        f"random_{seed:03d}",
        positions,
        offsets,
        width=width,
        length=length,
        seed=seed,
        free=count == 0,
    )
    return PlanningCase(
        name=f"random_{seed:03d}",
        local_map=_custom(config_random, uncertainty),
        start_xy_m=(0.0, config.start_y_m),
        goal_xy_m=(0.0, length - config.goal_margin_m),
        theta_rad=0.0,
        expected_status="ANY",
        expected_reason=None,
        oracle="sin oraculo de existencia; invariantes del validador sobre lo aceptado",
    )
