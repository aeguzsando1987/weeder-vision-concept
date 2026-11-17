"""Escenas sinteticas reproducibles (`docs/technical_plan.md` §3.1, §5 E1).

La escena es **verdad de referencia**: plantas, corredor, exclusiones y avance
conocidos. Se mantiene separada de cualquier mapa estimado (§3.1). Una lista de
plantas vacia NO certifica que el corredor este libre: esa afirmacion exige
`free_space_certified=True` de forma explicita (§3.2).

Determinismo: dada la misma `SceneConfig` (incluida su semilla), `build_scene`
produce exactamente los mismos valores; el ruido usa
`numpy.random.default_rng(seed)` y ningun estado global.

Procedencia: una escena sintetica nunca porta escala verificada. El dominio
`ground_metric_verified` solo puede provenir de una calibracion valida (E4), no
de una etiqueta en la configuracion de una escena sintetica (R2).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..core.config import (
    SafetyConfig,
    SceneConfig,
    UncertaintyConfig,
    discretization_excess_violation,
)
from ..core.errors import (
    ConfigError,
    FrameMismatchError,
    ScenarioError,
    SchemaValidationError,
    UncertaintyBudgetError,
    UnitMismatchError,
)
from ..core.types import (
    CircularExclusion,
    Corridor,
    Frame,
    InflationTarget,
    PlantReference,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
)
from ..core.units import (
    LIMIT_COMPARISON_TOLERANCE,
    LengthUnit,
    Provenance,
    SpatialDomain,
    ensure_finite,
    ensure_non_negative,
    ensure_positive,
    is_metric,
    require_same_domain,
    unit_of_domain,
)

__all__ = ["SyntheticScene", "build_scene", "require_discretization_budget"]

SCENE_SCHEMA_VERSION = 1
"""Version del esquema JSON exportado de la escena."""


@dataclass(frozen=True, slots=True)
class SyntheticScene:
    """Escena sintetica completa, lista para alimentar mapa, planificador y validador.

    Invariantes comprobados en construccion (R1-R4):

    - Un unico marco y un unico dominio para escena, corredor, plantas,
      exclusiones e incertidumbre; nunca `ground_metric_verified`.
    - Avance y rumbo finitos; velocidad de avance positiva.
    - Colecciones propias e inmutables (tuplas), aunque se reciban listas.
    - Exactamente una exclusion por planta, con el mismo centro y una proteccion
      nominal no menor que la de la planta.
    - La proteccion por incertidumbre se aplica en un solo lugar segun
      `uncertainty.inflate`: margen de cada exclusion o `tool_inflation`, y cubre
      `uncertainty.total`.
    - Una escena sin plantas solo es valida si el corredor se certifica libre.
    """

    scene_id: str
    description: str
    frame: Frame
    domain: SpatialDomain
    corridor: Corridor
    plants: tuple[PlantReference, ...]
    exclusions: tuple[CircularExclusion, ...]
    advance_speed: float
    advance_heading: float
    uncertainty: UncertaintyPolicy
    seed: int
    free_space_certified: bool = False
    provenance: Provenance = Provenance.SYNTHETIC
    notes: tuple[str, ...] = ()
    tool_inflation: float = 0.0

    def __post_init__(self) -> None:
        if not str(self.scene_id).strip():
            raise ScenarioError("SyntheticScene.scene_id: cadena vacia")
        context = f"SyntheticScene({self.scene_id})"
        object.__setattr__(self, "frame", Frame(self.frame))
        object.__setattr__(self, "domain", SpatialDomain(self.domain))
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        object.__setattr__(self, "plants", _owned_tuple(self.plants, PlantReference, "plants"))
        object.__setattr__(
            self, "exclusions", _owned_tuple(self.exclusions, CircularExclusion, "exclusions")
        )
        object.__setattr__(self, "notes", tuple(str(note) for note in self.notes))
        object.__setattr__(
            self, "advance_speed", ensure_positive(self.advance_speed, "advance_speed")
        )
        object.__setattr__(
            self, "advance_heading", ensure_finite(self.advance_heading, "advance_heading")
        )
        object.__setattr__(
            self, "tool_inflation", ensure_non_negative(self.tool_inflation, "tool_inflation")
        )
        if not isinstance(self.corridor, Corridor):
            raise SchemaValidationError(f"{context}: corridor debe ser Corridor")
        if not isinstance(self.uncertainty, UncertaintyPolicy):
            raise SchemaValidationError(f"{context}: uncertainty debe ser UncertaintyPolicy")
        if not isinstance(self.free_space_certified, bool):
            raise SchemaValidationError(f"{context}: free_space_certified debe ser booleano")

        self._check_provenance(context)
        self._check_frames_and_domains(context)
        self._check_plant_exclusion_correspondence(context)
        self._check_uncertainty_protection(context)
        if not self.plants and not self.free_space_certified:
            raise ScenarioError(
                f"{context}: escena sin plantas y sin free_space_certified=True; la ausencia "
                "de plantas no certifica un corredor libre ni un area de trabajo conocida"
            )
        if self.free_space_certified and self.plants:
            raise ScenarioError(
                f"{context}: free_space_certified=True es incompatible con plantas declaradas"
            )

    def _check_provenance(self, context: str) -> None:
        if self.domain is SpatialDomain.GROUND_METRIC_VERIFIED:
            raise UnitMismatchError(
                f"{context}: una escena sintetica no puede declararse en "
                f"{SpatialDomain.GROUND_METRIC_VERIFIED.value}; la escala verificada solo "
                "proviene de una calibracion valida"
            )
        if self.provenance is not Provenance.SYNTHETIC:
            raise ScenarioError(
                f"{context}: procedencia {self.provenance.value}; una escena sintetica es "
                "siempre synthetic"
            )
        for plant in self.plants:
            if plant.provenance is not Provenance.SYNTHETIC:
                raise ScenarioError(
                    f"{context}: la planta {plant.plant_id} declara procedencia "
                    f"{plant.provenance.value} dentro de una escena sintetica"
                )

    def _check_frames_and_domains(self, context: str) -> None:
        domains = [self.domain, self.corridor.domain, self.uncertainty.domain]
        domains.extend(plant.domain for plant in self.plants)
        domains.extend(exclusion.domain for exclusion in self.exclusions)
        require_same_domain(*domains, context=context)
        frames = {self.frame, self.corridor.frame}
        frames.update(plant.position.frame for plant in self.plants)
        frames.update(exclusion.center.frame for exclusion in self.exclusions)
        if len(frames) != 1:
            raise FrameMismatchError(
                f"{context}: marcos incompatibles {sorted(f.value for f in frames)}; escena, "
                "corredor, plantas y exclusiones deben compartir el marco declarado"
            )

    def _check_plant_exclusion_correspondence(self, context: str) -> None:
        plant_ids = [plant.plant_id for plant in self.plants]
        if len(set(plant_ids)) != len(plant_ids):
            raise ScenarioError(f"{context}: plant_id duplicado")
        exclusion_ids = [exclusion.plant_id for exclusion in self.exclusions]
        if len(set(exclusion_ids)) != len(exclusion_ids):
            raise ScenarioError(f"{context}: exclusion duplicada para una misma planta")
        if set(exclusion_ids) != set(plant_ids):
            missing = sorted(set(plant_ids) - set(exclusion_ids))
            orphan = sorted(set(exclusion_ids) - set(plant_ids))
            raise ScenarioError(
                f"{context}: cada planta de referencia exige exactamente una exclusion "
                f"(sin exclusion: {missing}; sin planta: {orphan})"
            )
        plants = {plant.plant_id: plant for plant in self.plants}
        for exclusion in self.exclusions:
            plant = plants[exclusion.plant_id]
            if exclusion.center != plant.position:
                raise ScenarioError(
                    f"{context}: la exclusion de {plant.plant_id} no esta centrada en su planta"
                )
            if exclusion.protection_radius < plant.protection_radius:
                raise ScenarioError(
                    f"{context}: la exclusion de {plant.plant_id} protege "
                    f"{exclusion.protection_radius} < proteccion requerida "
                    f"{plant.protection_radius}"
                )

    def _check_uncertainty_protection(self, context: str) -> None:
        required = self.uncertainty.total
        if self.uncertainty.inflate is InflationTarget.EXCLUSIONS:
            if self.tool_inflation != 0.0:
                raise UncertaintyBudgetError(
                    f"{context}: inflate=exclusions con tool_inflation="
                    f"{self.tool_inflation}; la incertidumbre se aplicaria dos veces"
                )
            for exclusion in self.exclusions:
                if exclusion.uncertainty_margin + LIMIT_COMPARISON_TOLERANCE < required:
                    raise UncertaintyBudgetError(
                        f"{context}: margen de incertidumbre de {exclusion.plant_id} "
                        f"({exclusion.uncertainty_margin}) menor que la cota declarada "
                        f"({required})"
                    )
        else:
            if any(exclusion.uncertainty_margin != 0.0 for exclusion in self.exclusions):
                raise UncertaintyBudgetError(
                    f"{context}: inflate=tool exige exclusiones sin margen de incertidumbre; "
                    "inflar ambas geometrias duplicaria la proteccion"
                )
            if self.tool_inflation + LIMIT_COMPARISON_TOLERANCE < required:
                raise UncertaintyBudgetError(
                    f"{context}: tool_inflation ({self.tool_inflation}) menor que la cota "
                    f"declarada ({required})"
                )

    # --- propiedades --------------------------------------------------------

    @property
    def unit(self) -> LengthUnit:
        return unit_of_domain(self.domain)

    @property
    def is_metric(self) -> bool:
        """`True` solo si el dominio tiene escala metrica (sintetica o verificada)."""
        return is_metric(self.domain)

    def exclusion_union(self) -> Polygon | Any:
        """Union de las aproximaciones exteriores de las exclusiones."""
        if not self.exclusions:
            return Polygon()
        return unary_union([exclusion.as_polygon() for exclusion in self.exclusions])

    def target_region(self) -> Polygon | Any:
        """Region objetivo fija: corredor evaluable menos las exclusiones de referencia.

        El denominador de cobertura se define aqui y no depende de decisiones del
        planificador (§6).
        """
        return self.corridor.as_polygon().difference(self.exclusion_union())

    def target_area(self) -> float:
        """Area de la region objetivo. Puede ser `0.0` (corredor bloqueado)."""
        return float(self.target_region().area)

    def exclusion_bands(self) -> tuple[tuple[CircularExclusion, ...], ...]:
        """Agrupa exclusiones en bandas longitudinales que se solapan en `y`.

        Dos exclusiones comparten banda si sus intervalos en `y` se intersecan.
        Las bandas se devuelven ordenadas por `y` creciente y, dentro de cada
        banda, por `x` creciente.
        """
        ordered = sorted(self.exclusions, key=lambda e: e.center.y - e.total_radius)
        bands: list[list[CircularExclusion]] = []
        band_top = -math.inf
        for exclusion in ordered:
            low = exclusion.center.y - exclusion.total_radius
            high = exclusion.center.y + exclusion.total_radius
            if bands and low <= band_top:
                bands[-1].append(exclusion)
                band_top = max(band_top, high)
            else:
                bands.append([exclusion])
                band_top = high
        return tuple(
            tuple(sorted(band, key=lambda e: e.center.x)) for band in bands
        )

    def lateral_gaps(self) -> dict[str, float]:
        """Huecos laterales libres del corredor, en la unidad del dominio.

        Para cada banda longitudinal de exclusiones se miden los intervalos libres
        sobre el eje `x`: contra la pared izquierda, contra la derecha y entre
        exclusiones consecutivas de la banda.

        Returns:
            - `widest`: cuello de botella del escenario, es decir el minimo entre
              bandas del intervalo libre mas ancho de cada banda. Es la cota
              superior del ancho lateral que una herramienta podria atravesar.
            - `left`, `right`: minimo entre bandas del hueco contra cada pared.
            - `between`: minimo entre bandas del mayor hueco interior de la banda
              (`0.0` si todas las bandas tienen una sola exclusion). Un valor
              negativo indica exclusiones superpuestas.
            - `bands`: numero de bandas longitudinales.

        Es un diagnostico del escenario, no una decision de factibilidad: la
        factibilidad la decide el validador con la huella completa (§3.4).
        """
        if not self.exclusions:
            width = self.corridor.width
            return {
                "widest": width,
                "left": width,
                "right": width,
                "between": width,
                "bands": 0.0,
            }
        bands = self.exclusion_bands()
        widest_per_band: list[float] = []
        left_per_band: list[float] = []
        right_per_band: list[float] = []
        between_per_band: list[float] = []
        for band in bands:
            left = (band[0].center.x - band[0].total_radius) - self.corridor.x_min
            right = self.corridor.x_max - (band[-1].center.x + band[-1].total_radius)
            interior = [
                (second.center.x - second.total_radius)
                - (first.center.x + first.total_radius)
                for first, second in pairwise(band)
            ]
            left_per_band.append(left)
            right_per_band.append(right)
            between_per_band.append(max(interior) if interior else 0.0)
            widest_per_band.append(max([left, right, *interior]))
        return {
            "widest": min(widest_per_band),
            "left": min(left_per_band),
            "right": min(right_per_band),
            "between": min(between_per_band),
            "bands": float(len(bands)),
        }

    # --- serializacion ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Representacion serializable con unidades y procedencia explicitas."""
        return {
            "schema_version": SCENE_SCHEMA_VERSION,
            "scene_id": self.scene_id,
            "description": self.description,
            "frame": self.frame.value,
            "domain": self.domain.value,
            "unit": self.unit.value,
            "is_metric": self.is_metric,
            "metric_scale_verified": False,
            "provenance": self.provenance.value,
            "seed": self.seed,
            "free_space_certified": self.free_space_certified,
            "corridor": self.corridor.to_dict(),
            "plants": [plant.to_dict() for plant in self.plants],
            "exclusions": [exclusion.to_dict() for exclusion in self.exclusions],
            "advance": {
                "speed": self.advance_speed,
                "speed_unit": f"{self.unit.value}/s",
                "heading_rad": self.advance_heading,
            },
            "uncertainty": {
                **self.uncertainty.to_dict(),
                "applied_exclusion_margins": {
                    exclusion.plant_id: exclusion.uncertainty_margin
                    for exclusion in self.exclusions
                },
                "applied_tool_inflation": self.tool_inflation,
            },
            "target_area": self.target_area(),
            "lateral_gaps": self.lateral_gaps(),
            "notes": list(self.notes),
        }

    def to_json(self, indent: int = 2) -> str:
        """Escena serializada como JSON determinista."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, ensure_ascii=True)


def build_scene(
    config: SceneConfig,
    uncertainty: UncertaintyPolicy | UncertaintyBound | UncertaintyConfig | None = None,
    notes: tuple[str, ...] = (),
) -> SyntheticScene:
    """Construye una `SyntheticScene` determinista a partir de su configuracion.

    Args:
        config: Configuracion validada de la escena.
        uncertainty: Fuente unica de la incertidumbre efectiva (R1). Puede ser una
            `UncertaintyPolicy`, la seccion `safety.uncertainty` validada, o una
            `UncertaintyBound` (presupuesto igual a su suma, inflando exclusiones).
            Si se omite, se usan los valores por defecto de `configs/safety.yaml`
            (solo en dominios metricos). La politica se aplica, no se reparte: con
            `inflate=exclusions` el margen configurado debe cubrir la cota; con
            `inflate=tool` el margen debe ser cero y la cota pasa a
            `tool_inflation`. Una cota mayor que la proteccion se rechaza.
        notes: Notas que acompanan la escena en su exportacion.

    Raises:
        UncertaintyBudgetError: si la proteccion aplicada no cubre la cota.
        UnitMismatchError: si se pide escala verificada o se mezclan dominios.
        ScenarioError: si la escena resulta inconsistente (por ejemplo, plantas
            fuera del corredor).
    """
    if not isinstance(config, SceneConfig):
        raise ScenarioError("build_scene: se espera SceneConfig")

    domain = config.domain
    frame = config.frame
    if domain is SpatialDomain.GROUND_METRIC_VERIFIED:
        raise UnitMismatchError(
            f"escena '{config.scene_id}': el constructor sintetico no produce escenas "
            f"{SpatialDomain.GROUND_METRIC_VERIFIED.value}; una etiqueta no es una medicion"
        )
    policy = _resolve_policy(uncertainty, domain, config.scene_id)

    corridor = Corridor.centered(
        width=config.corridor.width_m,
        length=config.corridor.length_m,
        center_x=config.corridor.center_x_m,
        start_y=config.corridor.start_y_m,
        frame=frame,
        domain=domain,
    )

    margin = config.exclusion.uncertainty_margin_m
    tool_inflation = policy.total if policy.inflate is InflationTarget.TOOL else 0.0

    rng = np.random.default_rng(config.seed)
    count = len(config.plants.positions_y_m)
    jitter_x = (
        rng.uniform(-config.jitter_x_m, config.jitter_x_m, size=count)
        if config.jitter_x_m > 0.0
        else np.zeros(count)
    )
    jitter_y = (
        rng.uniform(-config.jitter_y_m, config.jitter_y_m, size=count)
        if config.jitter_y_m > 0.0
        else np.zeros(count)
    )

    plants: list[PlantReference] = []
    exclusions: list[CircularExclusion] = []
    for index in range(count):
        position = Point2(
            x=(
                config.corridor.center_x_m
                + config.plants.offsets_x_m[index]
                + float(jitter_x[index])
            ),
            y=config.plants.positions_y_m[index] + float(jitter_y[index]),
            frame=frame,
            domain=domain,
        )
        if not corridor.contains_point(position):
            raise ScenarioError(
                f"escena '{config.scene_id}': la planta {index} en "
                f"({position.x:.4f}, {position.y:.4f}) cae fuera del corredor"
            )
        plant = PlantReference(
            plant_id=f"{config.scene_id}_p{index:02d}",
            position=position,
            stem_radius=config.plants.stem_radius_m,
            protection_radius=config.exclusion.protection_radius_m,
            provenance=Provenance.SYNTHETIC,
        )
        plants.append(plant)
        exclusions.append(
            CircularExclusion.from_plant(
                plant,
                uncertainty_margin=margin,
                polygon_segments=config.exclusion.polygon_segments,
            )
        )

    return SyntheticScene(
        scene_id=config.scene_id,
        description=config.description,
        frame=frame,
        domain=domain,
        corridor=corridor,
        plants=tuple(plants),
        exclusions=tuple(exclusions),
        advance_speed=config.advance.speed_mps,
        advance_heading=config.advance.heading_rad,
        uncertainty=policy,
        seed=config.seed,
        free_space_certified=config.free_space_certified,
        provenance=Provenance.SYNTHETIC,
        notes=notes,
        tool_inflation=tool_inflation,
    )


def require_discretization_budget(scene: SyntheticScene, safety: SafetyConfig) -> None:
    """Comprueba R12 sobre cada exclusion **construida** de una escena metrica.

    Se evalua la geometria real (radio total y lados de cada exclusion), no los
    parametros de configuracion, de modo que tambien cubre escenas de catalogo,
    radios distintos y politicas de incertidumbre que cambien el margen.

    Excepcion explicita por dominio: el presupuesto esta en metros y no se aplica a
    escenas no metricas (por ejemplo la escena grafica del SVG); no se convierten
    metros a unidades graficas.

    Raises:
        ConfigError: si alguna exclusion excede el presupuesto.
    """
    if not isinstance(safety, SafetyConfig):
        raise SchemaValidationError("require_discretization_budget: se espera SafetyConfig")
    if not scene.is_metric:
        return
    limit = safety.max_conservative_discretization_excess_m
    for exclusion in scene.exclusions:
        violation = discretization_excess_violation(
            exclusion.total_radius, exclusion.polygon_segments, limit
        )
        if violation is not None:
            raise ConfigError(
                f"escena '{scene.scene_id}', exclusion {exclusion.plant_id}: {violation}"
            )


def _resolve_policy(
    uncertainty: UncertaintyPolicy | UncertaintyBound | UncertaintyConfig | None,
    domain: SpatialDomain,
    scene_id: str,
) -> UncertaintyPolicy:
    if uncertainty is None:
        if not is_metric(domain):
            raise ScenarioError(
                f"escena '{scene_id}': el dominio {domain.value} no es metrico; declare una "
                "politica de incertidumbre en su propia unidad"
            )
        return UncertaintyConfig().to_policy(domain)
    if isinstance(uncertainty, UncertaintyConfig):
        return uncertainty.to_policy(domain)
    if isinstance(uncertainty, UncertaintyBound):
        return UncertaintyPolicy.from_bound(uncertainty)
    if isinstance(uncertainty, UncertaintyPolicy):
        return uncertainty
    raise SchemaValidationError(
        f"escena '{scene_id}': uncertainty debe ser UncertaintyPolicy, UncertaintyConfig "
        f"o UncertaintyBound, se recibio {type(uncertainty).__name__}"
    )


def _owned_tuple(values: Any, kind: type, name: str) -> tuple[Any, ...]:
    """Copia propia e inmutable de una coleccion, validando el tipo de cada elemento."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise SchemaValidationError(f"{name}: se espera una coleccion de {kind.__name__}")
    owned = tuple(values)
    for item in owned:
        if not isinstance(item, kind):
            raise SchemaValidationError(
                f"{name}: elemento {type(item).__name__}, se espera {kind.__name__}"
            )
    return owned
