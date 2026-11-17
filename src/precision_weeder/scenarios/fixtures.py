"""Catalogo de escenas sinteticas y casos adversos (`docs/technical_plan.md` §5 E1, §6).

Cada constructor es determinista y no depende de estado global. Ningun valor de
este modulo es una medicion: todos son sinteticos o graficos, registrados en
`docs/assumptions.md`.

Casos incluidos:

| Nombre | Proposito |
|---|---|
| `nominal_three_plants` | Escena principal factible: 3 plantas alineadas. |
| `five_plants` | Escena principal ampliada a 5 plantas. |
| `empty_corridor_certified` | Corredor conocido y certificado como libre. |
| `isolated_plant` | Planta aislada. |
| `irregular_spacing` | Espaciamiento irregular con perturbacion sembrada. |
| `off_center_plant` | Planta descentrada. |
| `plant_on_corridor_edge` | Planta en el limite del corredor. |
| `overlapping_exclusions` | Exclusiones superpuestas. |
| `narrow_passage` | Paso estrecho, apenas mayor que la cuchilla. |
| `blocked_corridor` | Corredor bloqueado: no existe hueco para la cuchilla. |
| `svg_graphic` | Escena grafica del SVG de referencia, sin escala. |
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final

from ..core.config import (
    AdvanceConfig,
    CorridorConfig,
    ExclusionConfig,
    PlantsConfig,
    SceneConfig,
    ToolConfig,
    UncertaintyConfig,
)
from ..core.errors import ScenarioError
from ..core.types import (
    Frame,
    InflationTarget,
    OrientationMode,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
)
from ..core.units import Provenance, SpatialDomain
from ..geometry.frames import Transform2D
from ..simulation.tool import ToolGeometry, ToolState, tool_geometry_from_config
from .synthetic import SyntheticScene, build_scene

__all__ = [
    "SCENE_BUILDERS",
    "SVG_VIEWBOX",
    "build_fixture",
    "default_tool_geometry",
    "default_tool_state",
    "fixture_names",
    "svg_tool_geometry",
    "svg_tool_transform",
]

# --- parametros sinteticos compartidos (metricos) ---------------------------

_CORRIDOR_WIDTH_M: Final[float] = 1.0
_CORRIDOR_LENGTH_M: Final[float] = 5.0
_STEM_RADIUS_M: Final[float] = 0.02
_PROTECTION_RADIUS_M: Final[float] = 0.15
_UNCERTAINTY_MARGIN_M: Final[float] = 0.03
_ADVANCE_SPEED_MPS: Final[float] = 0.20
_METRIC_DOMAIN: Final[SpatialDomain] = SpatialDomain.GROUND_METRIC_SYNTHETIC

# Radio total de exclusion resultante: 0.15 + 0.03 = 0.18 m.
_TOTAL_EXCLUSION_RADIUS_M: Final[float] = _PROTECTION_RADIUS_M + _UNCERTAINTY_MARGIN_M

# --- parametros graficos del SVG de referencia (sin escala) -----------------

UncertaintySource = UncertaintyPolicy | UncertaintyBound | UncertaintyConfig | None
"""Fuente de incertidumbre aceptada por las escenas metricas del catalogo (R1)."""

SVG_VIEWBOX: Final[tuple[float, float]] = (340.0, 330.0)
"""Lienzo del SVG en unidades graficas; no tiene escala fisica (§1.2)."""

_SVG_CORRIDOR_X: Final[tuple[float, float]] = (90.0, 250.0)
_SVG_CORRIDOR_Y: Final[tuple[float, float]] = (0.0, 330.0)
_SVG_PLANT_X: Final[float] = 169.0
_SVG_PLANT_Y: Final[tuple[float, ...]] = (65.0, 165.0, 265.0)
_SVG_PROTECTION_RADIUS: Final[float] = 32.0
_SVG_STEM_RADIUS: Final[float] = 4.5
_SVG_TOOL_SIZE: Final[tuple[float, float]] = (30.0, 10.0)
_SVG_TOOL_PIVOT: Final[tuple[float, float]] = (119.0, 248.0)
_SVG_DOMAIN: Final[SpatialDomain] = SpatialDomain.GRAPHIC_UNSCALED


def _metric_scene_config(
    scene_id: str,
    description: str,
    positions_y_m: tuple[float, ...],
    offsets_x_m: tuple[float, ...],
    *,
    seed: int = 20261024,
    jitter_x_m: float = 0.0,
    jitter_y_m: float = 0.0,
    free_space_certified: bool = False,
) -> SceneConfig:
    return SceneConfig(
        scene_id=scene_id,
        description=description,
        domain=_METRIC_DOMAIN,
        frame=Frame.WORLD,
        seed=seed,
        jitter_x_m=jitter_x_m,
        jitter_y_m=jitter_y_m,
        free_space_certified=free_space_certified,
        corridor=CorridorConfig(
            width_m=_CORRIDOR_WIDTH_M, length_m=_CORRIDOR_LENGTH_M, center_x_m=0.0, start_y_m=0.0
        ),
        plants=PlantsConfig(
            positions_y_m=positions_y_m,
            offsets_x_m=offsets_x_m,
            stem_radius_m=_STEM_RADIUS_M,
        ),
        exclusion=ExclusionConfig(
            protection_radius_m=_PROTECTION_RADIUS_M,
            uncertainty_margin_m=_UNCERTAINTY_MARGIN_M,
            polygon_segments=64,
        ),
        advance=AdvanceConfig(speed_mps=_ADVANCE_SPEED_MPS, heading_rad=0.0),
    )


# --- escenas metricas sinteticas -------------------------------------------


def nominal_three_plants(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Escena principal: 3 plantas centradas en `y = 1, 2, 3 m` (§5 E1)."""
    return build_scene(
        _metric_scene_config(
            "nominal_three_plants",
            "Escena nominal sintetica: 3 plantas centradas y corredor de 1 x 5 m.",
            positions_y_m=(1.0, 2.0, 3.0),
            offsets_x_m=(0.0, 0.0, 0.0),
        ),
        uncertainty=uncertainty,
        notes=(
            "Caso factible de referencia: el hueco lateral libre (0.32 m) supera la "
            "extension lateral de la cuchilla (0.20 m).",
        ),
    )


def five_plants(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Escena principal ampliada a 5 plantas regularmente espaciadas."""
    return build_scene(
        _metric_scene_config(
            "five_plants",
            "Escena nominal ampliada: 5 plantas regularmente espaciadas cada 0.8 m.",
            positions_y_m=(0.6, 1.4, 2.2, 3.0, 3.8),
            offsets_x_m=(0.0,) * 5,
        ),
        uncertainty=uncertainty,
    )


def empty_corridor_certified(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Corredor vacio **certificado** como libre.

    La certificacion es una declaracion explicita del escenario; una lista de
    plantas vacia por si sola no prueba ausencia de cultivo (§3.2).
    """
    return build_scene(
        _metric_scene_config(
            "empty_corridor_certified",
            "Corredor sintetico sin plantas, declarado explicitamente como libre.",
            positions_y_m=(),
            offsets_x_m=(),
            free_space_certified=True,
        ),
        uncertainty=uncertainty,
        notes=("La region objetivo coincide con el corredor completo: no hay exclusiones.",),
    )


def isolated_plant(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Una sola planta en el centro del corredor."""
    return build_scene(
        _metric_scene_config(
            "isolated_plant",
            "Planta aislada centrada en y = 2.5 m.",
            positions_y_m=(2.5,),
            offsets_x_m=(0.0,),
        ),
        uncertainty=uncertainty,
    )


def irregular_spacing(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Espaciamiento irregular con perturbacion sembrada y reproducible."""
    return build_scene(
        _metric_scene_config(
            "irregular_spacing",
            "Espaciamiento irregular con perturbacion lateral y longitudinal sembrada.",
            positions_y_m=(0.8, 1.35, 2.9, 4.1),
            offsets_x_m=(0.0, 0.05, -0.04, 0.02),
            seed=20261024,
            jitter_x_m=0.02,
            jitter_y_m=0.05,
        ),
        uncertainty=uncertainty,
        notes=("La perturbacion depende solo de la semilla 20261024.",),
    )


def off_center_plant(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Plantas descentradas respecto al eje del corredor."""
    return build_scene(
        _metric_scene_config(
            "off_center_plant",
            "Plantas descentradas: desplazamientos laterales de 0.30 y -0.25 m.",
            positions_y_m=(1.5, 3.0),
            offsets_x_m=(0.30, -0.25),
        ),
        uncertainty=uncertainty,
    )


def plant_on_corridor_edge(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Planta justo en el limite lateral del corredor.

    Su exclusion sobresale del corredor: el area objetivo se reduce y parte de la
    exclusion queda fuera del dominio permitido.
    """
    return build_scene(
        _metric_scene_config(
            "plant_on_corridor_edge",
            "Planta centrada en el borde lateral del corredor (x = +0.50 m).",
            positions_y_m=(2.0,),
            offsets_x_m=(0.5,),
        ),
        uncertainty=uncertainty,
        notes=(
            "La exclusion excede el corredor; el despeje se evalua solo en el dominio conocido.",
        ),
    )


def overlapping_exclusions(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Dos plantas tan proximas que sus exclusiones se superponen."""
    return build_scene(
        _metric_scene_config(
            "overlapping_exclusions",
            "Dos plantas separadas 0.25 m en y: sus exclusiones de radio 0.18 m se superponen.",
            positions_y_m=(2.0, 2.25),
            offsets_x_m=(0.0, 0.0),
        ),
        uncertainty=uncertainty,
        notes=("Distancia entre centros (0.25 m) < suma de radios totales (0.36 m).",),
    )


def narrow_passage(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Paso central estrecho, apenas mayor que la extension lateral de la cuchilla.

    Huecos: central 0.22 m frente a una cuchilla de 0.20 m de extension lateral.
    """
    return build_scene(
        _metric_scene_config(
            "narrow_passage",
            "Paso estrecho: dos plantas enfrentadas dejan un hueco central de 0.22 m.",
            positions_y_m=(2.0, 2.0),
            offsets_x_m=(-0.29, 0.29),
        ),
        uncertainty=uncertainty,
        notes=(
            "Hueco central 0.22 m, cuchilla 0.20 m: factible solo con despeje casi nulo.",
            "Los huecos laterales (0.03 m) no admiten la cuchilla.",
        ),
    )


def blocked_corridor(uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Corredor bloqueado: ningun hueco admite la cuchilla.

    Caso adverso que debe conducir a abstencion; no se modifica para que apruebe
    (§5 E1).
    """
    return build_scene(
        _metric_scene_config(
            "blocked_corridor",
            "Corredor bloqueado: huecos de 0.12 m laterales y 0.04 m central.",
            positions_y_m=(2.0, 2.0),
            offsets_x_m=(-0.20, 0.20),
        ),
        uncertainty=uncertainty,
        notes=(
            "Hueco mas ancho 0.12 m < extension lateral de la cuchilla (0.20 m).",
            "El resultado esperado es abstencion, no una ruta de costo menor.",
        ),
    )


# --- escena grafica del SVG (sin escala) ------------------------------------


def svg_graphic() -> SyntheticScene:
    """Escena grafica inspirada literalmente en `.context/descarga.svg`.

    Conserva las unidades graficas del `viewBox` y el sentido del eje `v` del SVG
    (`y` creciente hacia abajo en pantalla): no se invierte ningun eje ni se
    convierten las 32 unidades de proteccion en centimetros (§1.2). Sirve como
    fixture visual de regresion, no como escenario metrico.
    """
    center_x = 0.5 * (_SVG_CORRIDOR_X[0] + _SVG_CORRIDOR_X[1])
    config = SceneConfig(
        scene_id="svg_graphic",
        description=(
            "Escena grafica del SVG de referencia en unidades del viewBox 340x330, sin escala."
        ),
        domain=_SVG_DOMAIN,
        frame=Frame.WORLD,
        seed=0,
        corridor=CorridorConfig(
            width_m=_SVG_CORRIDOR_X[1] - _SVG_CORRIDOR_X[0],
            length_m=_SVG_CORRIDOR_Y[1] - _SVG_CORRIDOR_Y[0],
            center_x_m=center_x,
            start_y_m=_SVG_CORRIDOR_Y[0],
        ),
        plants=PlantsConfig(
            positions_y_m=_SVG_PLANT_Y,
            offsets_x_m=(_SVG_PLANT_X - center_x,) * len(_SVG_PLANT_Y),
            stem_radius_m=_SVG_STEM_RADIUS,
        ),
        exclusion=ExclusionConfig(
            protection_radius_m=_SVG_PROTECTION_RADIUS,
            uncertainty_margin_m=0.0,
            polygon_segments=64,
        ),
        advance=AdvanceConfig(speed_mps=1.0, heading_rad=0.0),
    )
    # El SVG no declara incertidumbre: politica nula explicita en unidades graficas,
    # sin convertir cotas metricas a unidades del dibujo.
    graphic_policy = UncertaintyPolicy(
        bound=UncertaintyBound(
            localization=0.0,
            calibration=0.0,
            tracking=0.0,
            domain=_SVG_DOMAIN,
            provenance=Provenance.SYNTHETIC,
        ),
        budget=0.0,
        inflate=InflationTarget.EXCLUSIONS,
    )
    return build_scene(
        config,
        uncertainty=graphic_policy,
        notes=(
            "Unidades graficas del SVG; sin escala fisica ni metadatos de calibracion.",
            "Incertidumbre nula declarada: el dibujo no aporta cotas de localizacion.",
            "El eje y crece hacia abajo igual que en el SVG; no se invierte ningun eje.",
            "El avance 1.0 unidad/s es un marcador sin significado fisico.",
            "La curva Bezier del SVG no se importa: no es una trayectoria aprobada.",
        ),
    )


# --- herramientas asociadas -------------------------------------------------


def default_tool_geometry() -> ToolGeometry:
    """Cuchilla sintetica por defecto: rectangulo `0.20 x 0.08 m`, lado largo sobre `x`."""
    return tool_geometry_from_config(ToolConfig())


def default_tool_state(
    x: float = 0.0,
    y: float = 0.0,
    time_s: float = 0.0,
) -> ToolState:
    """Estado inicial de la herramienta en el marco del vehiculo, sin contacto."""
    return ToolState(
        time_s=time_s,
        position=Point2(x=x, y=y, frame=Frame.VEHICLE, domain=_METRIC_DOMAIN),
    )


def svg_tool_geometry() -> ToolGeometry:
    """Cuchilla grafica del SVG: `30 x 10` unidades, pivote en su centro."""
    return ToolGeometry.rectangular(
        tool_id="svg_blade_30x10",
        length_x=_SVG_TOOL_SIZE[0],
        width_y=_SVG_TOOL_SIZE[1],
        cutting_inset=0.0,
        orientation_mode=OrientationMode.FIXED,
        domain=_SVG_DOMAIN,
    )


def svg_tool_transform() -> Transform2D:
    """`T_WB` que coloca la cuchilla grafica en el pivote `(119, 248)` del SVG.

    El SVG dibuja la curva y el rectangulo de forma independiente: esta pose
    reproduce el dibujo, no certifica ausencia de colision (§1.2).
    """
    return Transform2D(
        parent=Frame.WORLD,
        child=Frame.TOOL,
        tx=_SVG_TOOL_PIVOT[0],
        ty=_SVG_TOOL_PIVOT[1],
        theta=0.0,
        domain=_SVG_DOMAIN,
    )


SCENE_BUILDERS: Final[Mapping[str, Callable[..., SyntheticScene]]] = {
    "nominal_three_plants": nominal_three_plants,
    "five_plants": five_plants,
    "empty_corridor_certified": empty_corridor_certified,
    "isolated_plant": isolated_plant,
    "irregular_spacing": irregular_spacing,
    "off_center_plant": off_center_plant,
    "plant_on_corridor_edge": plant_on_corridor_edge,
    "overlapping_exclusions": overlapping_exclusions,
    "narrow_passage": narrow_passage,
    "blocked_corridor": blocked_corridor,
    "svg_graphic": svg_graphic,
}
"""Registro nombre -> constructor de escena."""


def fixture_names() -> tuple[str, ...]:
    """Nombres de escena disponibles, en orden estable."""
    return tuple(SCENE_BUILDERS)


def build_fixture(name: str, uncertainty: UncertaintySource = None) -> SyntheticScene:
    """Construye la escena registrada con ese nombre.

    Args:
        name: Nombre registrado en `SCENE_BUILDERS`.
        uncertainty: Politica de incertidumbre para escenas metricas (por ejemplo,
            `safety.uncertainty` validada). Si se omite, se usan los valores por
            defecto de `configs/safety.yaml`. La escena grafica del SVG declara su
            propia politica en unidades graficas y no admite una metrica.

    Raises:
        ScenarioError: si el nombre no esta registrado o se aplica una politica
            metrica a la escena grafica.
    """
    try:
        builder = SCENE_BUILDERS[name]
    except KeyError as exc:
        raise ScenarioError(
            f"escena desconocida '{name}'; disponibles: {', '.join(fixture_names())}"
        ) from exc
    if name == "svg_graphic":
        if uncertainty is not None:
            raise ScenarioError(
                "svg_graphic declara su incertidumbre en unidades graficas; no admite una "
                "politica metrica"
            )
        return builder()
    return builder(uncertainty=uncertainty)
