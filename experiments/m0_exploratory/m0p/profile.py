"""Perfil validado de M0-P (orden §3): todo parametro ajustable vive aqui.

Reglas: claves desconocidas son error; valores finitos; unidades en el nombre del
campo. Ningun algoritmo tiene ramas por nombre, hash, resolucion o duracion del
clip: un clip distinto se atiende con otro perfil. El surco de trabajo se declara
como banda normalizada de la imagen de referencia, no como `u > 960`.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from precision_weeder.core.config import (
    FlowConfig,
    PlannerFile,
    RoiConfig,
    SafetyFile,
    ToolFile,
    UncertaintyConfig,
    ValidationFile,
    load_yaml_mapping,
    validate_config_payload,
)
from precision_weeder.core.errors import ConfigError

__all__ = ["M0Profile", "load_profile"]

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
Fraction = Annotated[float, Field(ge=0.0, le=1.0)]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, validate_default=True)

    @field_validator("*", mode="after")
    @classmethod
    def _finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("se exige un valor finito")
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError("se exige una secuencia de valores finitos")
        return value


class ProviderConfig(_Strict):
    """Proveedor de observaciones intercambiable (orden §3)."""

    kind: Literal["replay"] = "replay"
    export: str
    """Exportacion versionada (JSON con mascaras poligonales), relativa a la raiz."""
    min_confidence: Fraction = 0.25


class ProjectionConfig(_Strict):
    """Proyeccion supuesta imagen -> suelo `W_segment` (orden §4.1).

    Plano de suelo paralelo a la imagen con escala uniforme **supuesta**: no es una
    calibracion. `forward_image_direction` declara hacia que borde de la imagen
    avanza el vehiculo (supuesto de montaje); el movimiento medido debe ser
    coherente con el o se invalida la proyeccion.
    """

    scale_m_per_px: PositiveFloat
    scale_rationale: str
    forward_image_direction: Literal["down", "up"] = "down"


class MotionConfig(_Strict):
    """Estimador de traslacion 2D robusta del suelo con el flujo de E0 (orden §4.2)."""

    analysis_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    roi: RoiConfig = RoiConfig()
    flow: FlowConfig = FlowConfig()
    min_inliers: Annotated[int, Field(ge=3)] = 20
    max_mad_px: PositiveFloat = 6.0
    """Dispersion maxima (MAD, px del original) admitida para una traslacion; por
    encima se asume giro/paralaje/cambio de plano y se pierde la proyeccion."""
    max_step_px: PositiveFloat = 120.0
    """Desplazamiento maximo por cuadro; un salto mayor reinicia el segmento."""
    max_segment_frames: Annotated[int, Field(ge=2)] = 300
    """Longitud maxima de un segmento: se reinicia `W_segment` para no integrar
    indefinidamente una traslacion (su error se acumula)."""
    min_forward_step_px: NonNegativeFloat = 0.0
    """Paso minimo en el sentido de avance; un retroceso invalida la proyeccion."""


class RowConfig(_Strict):
    """Surco de trabajo: banda normalizada de la imagen (orden §3).

    Con la camara supuesta rigida al vehiculo, la banda es la misma en cada cuadro
    (corredor solidario al vehiculo); no se interpreta como `u > 960` ni como lado
    del tractor sin evidencia."""

    name: str
    u_min_norm: Fraction
    u_max_norm: Fraction
    axis_u_norm: Fraction
    """Linea de plantacion supuesta (columna normalizada) cuando no hay plantas
    suficientes para estimarla; la ruta se plantea sobre esa linea (intrafilar)."""
    axis_min_plants: Annotated[int, Field(ge=1)] = 2
    """Plantas seguidas en la banda necesarias para estimar el eje por su mediana."""
    rationale: str

    @model_validator(mode="after")
    def _band(self) -> RowConfig:
        if self.u_max_norm <= self.u_min_norm:
            raise ValueError("u_max_norm debe ser > u_min_norm")
        if not self.u_min_norm <= self.axis_u_norm <= self.u_max_norm:
            raise ValueError("axis_u_norm debe estar dentro de la banda del surco")
        return self


class ToolLineConfig(_Strict):
    """Posicion supuesta de la herramienta en la imagen (fila normalizada)."""

    v_norm: Fraction = 0.05
    rationale: str = "supuesto de montaje: la herramienta actua en el borde trasero visible"


class PlanningConfig(_Strict):
    """Uso de E2 por cuadro (sin modificar su algoritmo)."""

    goal_margin_norm: Fraction = 0.05
    """Distancia normalizada al borde delantero visible donde se fija la meta."""
    lateral_samples: Annotated[int, Field(ge=1, le=41)] = 9
    """Posiciones laterales de inicio/meta que se prueban, repartidas uniformemente en
    la banda donde la cuchilla cabe dentro del corredor y ordenadas desde el centro.
    Se derivan del corredor y la herramienta de cada cuadro: no hay offsets por clip."""
    axis_hysteresis_m: Annotated[float, Field(gt=0)] = 0.03
    """Cambio del eje estimado por debajo del cual se conserva la tarea (eje y prefijo)
    de la propuesta anterior; por encima, la tarea cambia y se reselecciona enlazando
    desde la referencia (orden v1.2 §9-§10)."""
    gap_step_m: Annotated[float, Field(gt=0)] = 0.01
    """Paso de muestreo del eje de la hilera para encontrar gaps donde cabe la huella."""
    replan_every_frames: Annotated[int, Field(ge=1)] = 1
    max_route_age_frames: Annotated[int, Field(ge=0)] = 0
    """Edad maxima de una ruta mostrada como vigente. Con 0 solo es vigente en el
    cuadro cuyo mapa certifico."""
    pattern: Literal["planner_choice", "alternating"] = "planner_choice"
    """Patron de la ruta (orden V1 §3). `planner_choice`: E2 elige el costado de cada
    planta por costo (M0-P). `alternating`: un nudo al costado de cada planta, por
    el lado opuesto al de la anterior, unidos por cambios laterales suaves; el costado
    se decide una vez por planta. Si el cruce es demasiado brusco o no se certifica, la
    planta se pasa por el mismo costado y la omision se registra."""
    pattern_side_margin_m: NonNegativeFloat = 0.02
    """Holgura de las propuestas del patron respecto de las exclusiones: absorbe el
    movimiento de las plantas seguidas entre cuadros. Solo afecta a lo que se propone;
    el validador certifica con los margenes de siempre."""
    pattern_max_slope: PositiveFloat = 2.5
    """Pendiente lateral maxima (dx/dy, adimensional) de un tramo del patron. La tasa
    lateral que exigiria la consigna es esta pendiente por la velocidad de avance: un
    cruce que la supere no se propone y la planta se pasa por el mismo costado."""


    pattern_commit_clearance_m: NonNegativeFloat = 0.0
    """Holgura certificada minima para comprometer un cruce nuevo. Un cruce aceptado por
    menos deja de certificarse cuando las plantas seguidas se mueven unos milimetros y la
    ruta ya trazada cambiaria de costado; con 0 basta el certificado del validador. Solo
    restringe lo que se propone: no cambia el validador ni sus margenes."""


class FusionConfig(_Strict):
    """Fusion de duplicados por cuadro (orden §4.3)."""

    iou_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.3
    containment_threshold: Annotated[float, Field(gt=0.0, le=1.0)] = 0.8
    """Fraccion de la mascara menor contenida en la mayor para considerarlas la misma."""


class TrackingConfig(_Strict):
    """Asociacion uno a uno en `W_segment` y caducidad (orden §4.3-§4.4)."""

    gate_m: PositiveFloat = 0.06
    timeout_frames: Annotated[int, Field(ge=0)] = 8
    """Cuadros sin observacion durante los que se conserva la proteccion; despues la
    region pasa a desconocida (no se aprueba el hueco)."""


class MapConfig(_Strict):
    """Mapa local supuesto (orden §4.5-§4.6)."""

    proxy_margin_m: NonNegativeFloat = 0.01
    """Margen de representacion de la copa proyectada (aparte de la incertidumbre)."""
    polygon_segments: Annotated[int, Field(ge=8)] = 64
    assume_free_when_empty: bool = False
    """Solo con `true` un mapa sin plantas se declara libre (supuesto rotulado)."""
    protection: Literal["canopy", "assumed_stem"] = "canopy"
    """Objeto protegido. `canopy`: circulo que contiene la copa detectada (M0-P).
    `assumed_stem`: circulo de radio fijo `stem_protection_radius_m` centrado en el
    centroide de la copa, bajo el SUPUESTO de que el tallo esta ahi y la herramienta
    trabaja por debajo de la copa. No es una localizacion de tallo."""
    stem_protection_radius_m: PositiveFloat | None = None

    @model_validator(mode="after")
    def _stem(self) -> MapConfig:
        if self.protection == "assumed_stem" and self.stem_protection_radius_m is None:
            raise ValueError("protection assumed_stem exige stem_protection_radius_m")
        return self


class RenderConfig(_Strict):
    audio: Literal["omit"] = "omit"
    """Unica politica implementada: el audio se omite (y se declara). Copiarlo no esta
    implementado, asi que `copy_when_supported` se rechaza en lugar de anunciarse."""
    codec: Literal["libx264"] = "libx264"
    crf: Annotated[int, Field(ge=0, le=51)] = 20
    footprint_samples: Annotated[int, Field(ge=0, le=100)] = 12
    setpoint_overlay: bool = False
    """Dibuja la herramienta en la linea de herramienta y la franja con la consigna
    lateral reciente (orden V1, paso 3)."""


class AssumedUncertaintyConfig(_Strict):
    """Politica de incertidumbre **declarada para `ground_assumed`** (orden §4.5).

    E1 no aplica las cotas metricas de `safety.yaml` a un dominio no metrico: se
    declaran aqui, en metros supuestos, con su justificacion. Se aplican una sola vez.
    """

    policy: UncertaintyConfig
    rationale: str


class ConfigRefs(_Strict):
    """YAML del producto reutilizados (no se duplican tolerancias)."""

    tool: str = "configs/tool.yaml"
    safety: str = "configs/safety.yaml"
    planner: str = "configs/planner.yaml"
    validation: str = "configs/validation.yaml"


class M0Profile(_Strict):
    schema_version: Literal[1] = 1
    profile_id: str
    mode: Literal["exploratory"] = "exploratory"
    domain: Literal["ground_assumed"] = "ground_assumed"
    description: str
    provider: ProviderConfig
    expected_video_sha256: str | None = None
    projection: ProjectionConfig
    motion: MotionConfig = MotionConfig()
    row: RowConfig
    tool_line: ToolLineConfig = ToolLineConfig()
    planning: PlanningConfig = PlanningConfig()
    fusion: FusionConfig = FusionConfig()
    tracking: TrackingConfig = TrackingConfig()
    map: MapConfig = MapConfig()
    uncertainty: AssumedUncertaintyConfig
    render: RenderConfig = RenderConfig()
    configs: ConfigRefs = ConfigRefs()

    @model_validator(mode="after")
    def _check(self) -> M0Profile:
        if not self.profile_id.strip():
            raise ValueError("profile_id vacio")
        if self.expected_video_sha256 is not None and (
            len(self.expected_video_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.expected_video_sha256)
        ):
            raise ValueError("expected_video_sha256: SHA-256 hexadecimal en minusculas")
        goal_v = (
            1.0 - self.planning.goal_margin_norm
            if self.projection.forward_image_direction == "down"
            else self.planning.goal_margin_norm
        )
        if (goal_v - self.tool_line.v_norm) * (
            1 if self.projection.forward_image_direction == "down" else -1
        ) <= 0:
            raise ValueError("la meta debe quedar por delante de la herramienta en la imagen")
        return self


def load_profile(path: str | Path, root: str | Path) -> tuple[M0Profile, dict[str, Any]]:
    """Carga el perfil y los YAML del producto que referencia; todo validado.

    Returns:
        Perfil y configuracion resuelta (`tool`, `safety`, `planner`, `validation`).

    Raises:
        ConfigError / UnknownConfigKeyError: perfil o configuracion invalidos.
    """
    file_path = Path(path)
    try:
        payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        profile = M0Profile.model_validate(payload)
    except (OSError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{file_path}: perfil M0-P invalido\n{exc}") from exc
    base = Path(root)
    refs = profile.configs
    resolved = {
        "tool": validate_config_payload(ToolFile, load_yaml_mapping(base / refs.tool), refs.tool),
        "safety": validate_config_payload(
            SafetyFile, load_yaml_mapping(base / refs.safety), refs.safety
        ),
        "planner": validate_config_payload(
            PlannerFile, load_yaml_mapping(base / refs.planner), refs.planner
        ),
        "validation": validate_config_payload(
            ValidationFile, load_yaml_mapping(base / refs.validation), refs.validation
        ),
    }
    return profile, resolved
