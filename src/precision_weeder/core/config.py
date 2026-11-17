"""Configuracion validada y resuelta (`docs/technical_plan.md` §7).

Reglas:

- Las claves desconocidas son error (`extra="forbid"`).
- Todo valor por defecto se materializa en la configuracion resuelta.
- `mode: calibrated` sin calibracion verificada falla de forma explicita; no hay
  degradacion silenciosa a dominio supuesto.
- YAML se carga con `yaml.safe_load`.
"""

from __future__ import annotations

import math
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError, UnknownConfigKeyError
from .types import (
    CalibrationReference,
    CalibrationStatus,
    Frame,
    InflationTarget,
    OrientationMode,
    UncertaintyBound,
    UncertaintyPolicy,
    circumscribed_radial_excess,
)
from .units import (
    CLEARANCE_EPSILON_M,
    GEOMETRIC_APPROXIMATION_TOLERANCE_M,
    LIMIT_COMPARISON_TOLERANCE,
    Provenance,
    SpatialDomain,
    is_metric,
)

__all__ = [
    "SCHEMA_VERSION",
    "AdvanceConfig",
    "CalibrationConfig",
    "CorridorConfig",
    "CostNormalizersConfig",
    "CostWeightsConfig",
    "ExclusionConfig",
    "JointConfig",
    "KinematicsConfig",
    "KinematicsFile",
    "MechanismConfig",
    "MechanismFile",
    "PlanDemoConfig",
    "PlannerConfig",
    "PlannerFile",
    "PlantsConfig",
    "ProjectConfig",
    "RandomScenariosConfig",
    "RunMode",
    "SafetyConfig",
    "SafetyFile",
    "SceneConfig",
    "StartStateConfig",
    "SyntheticFile",
    "ToolConfig",
    "ToolFile",
    "UncertaintyConfig",
    "ValidationConfig",
    "ValidationFile",
    "VideoConfig",
    "VideoFile",
    "discretization_excess_violation",
    "load_project_config",
    "load_yaml_mapping",
    "validate_config_payload",
]

SCHEMA_VERSION: Final[int] = 1
"""Version de esquema de configuracion registrada en la configuracion resuelta."""

PositiveFloat = Annotated[float, Field(gt=0.0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]


class RunMode(StrEnum):
    """Modo de ejecucion (§7). Determina que dominios espaciales son licitos."""

    SYNTHETIC = "synthetic"
    EXPLORATORY = "exploratory"
    CALIBRATED = "calibrated"


_ALLOWED_DOMAINS: Final[dict[RunMode, frozenset[SpatialDomain]]] = {
    RunMode.SYNTHETIC: frozenset(
        {SpatialDomain.GROUND_METRIC_SYNTHETIC, SpatialDomain.GRAPHIC_UNSCALED}
    ),
    RunMode.EXPLORATORY: frozenset({SpatialDomain.IMAGE_PX, SpatialDomain.GROUND_ASSUMED}),
    RunMode.CALIBRATED: frozenset({SpatialDomain.GROUND_METRIC_VERIFIED}),
}


class _StrictModel(BaseModel):
    """Base comun: claves desconocidas prohibidas, inmutable, sin coercion laxa."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=False,
        validate_default=True,
        use_enum_values=False,
    )

    @field_validator("*", mode="after")
    @classmethod
    def _reject_non_finite(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("se exige un valor finito (NaN e inf estan prohibidos)")
        if isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError("se exige una secuencia de valores finitos")
        return value


# --- configs/synthetic.yaml -------------------------------------------------


class CorridorConfig(_StrictModel):
    """Corredor sintetico. Valores exclusivamente sinteticos (§5 E1)."""

    width_m: PositiveFloat = 1.0
    length_m: PositiveFloat = 5.0
    center_x_m: float = 0.0
    start_y_m: float = 0.0


class PlantsConfig(_StrictModel):
    """Posiciones de plantas de la verdad de referencia."""

    positions_y_m: tuple[float, ...] = (1.0, 2.0, 3.0)
    offsets_x_m: tuple[float, ...] = (0.0, 0.0, 0.0)
    stem_radius_m: PositiveFloat = 0.02

    @model_validator(mode="after")
    def _check_lengths(self) -> PlantsConfig:
        if len(self.positions_y_m) != len(self.offsets_x_m):
            raise ValueError(
                "positions_y_m y offsets_x_m deben tener la misma longitud "
                f"({len(self.positions_y_m)} != {len(self.offsets_x_m)})"
            )
        return self


class ExclusionConfig(_StrictModel):
    """Proteccion nominal y margen de incertidumbre de cada exclusion."""

    protection_radius_m: PositiveFloat = 0.15
    uncertainty_margin_m: NonNegativeFloat = 0.03
    polygon_segments: Annotated[int, Field(ge=8, le=4096)] = 64


class AdvanceConfig(_StrictModel):
    """Avance del vehiculo. Es una entrada del escenario, no una variable libre (§3.5)."""

    speed_mps: PositiveFloat = 0.20
    heading_rad: float = 0.0


class SceneConfig(_StrictModel):
    """Escena sintetica reproducible."""

    scene_id: str = "nominal_3_plants"
    description: str = "Escena nominal sintetica de tres plantas alineadas."
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC
    frame: Frame = Frame.WORLD
    seed: Annotated[int, Field(ge=0)] = 20261024
    jitter_x_m: NonNegativeFloat = 0.0
    jitter_y_m: NonNegativeFloat = 0.0
    free_space_certified: bool = False
    corridor: CorridorConfig = CorridorConfig()
    plants: PlantsConfig = PlantsConfig()
    exclusion: ExclusionConfig = ExclusionConfig()
    advance: AdvanceConfig = AdvanceConfig()

    @field_validator("scene_id")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("scene_id no puede estar vacio")
        return value

    @model_validator(mode="after")
    def _check_free_space_claim(self) -> SceneConfig:
        empty = len(self.plants.positions_y_m) == 0
        if empty and not self.free_space_certified:
            raise ValueError(
                "plants.positions_y_m vacio sin free_space_certified=True: la ausencia de "
                "plantas declaradas no certifica que el corredor este libre (§3.2)"
            )
        if self.free_space_certified and not empty:
            raise ValueError(
                "free_space_certified=True exige una lista de plantas vacia: no se certifica "
                "libre un corredor con plantas de referencia"
            )
        return self


class SyntheticFile(_StrictModel):
    """Contenido de `configs/synthetic.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    mode: RunMode = RunMode.SYNTHETIC
    scene: SceneConfig = SceneConfig()
    provenance: Provenance = Provenance.SYNTHETIC


# --- configs/tool.yaml ------------------------------------------------------


class ToolConfig(_StrictModel):
    """Geometria de la cuchilla. Dimensiones sinteticas, no de diseno mecanico."""

    tool_id: str = "blade_rect_200x80"
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC
    shape: Literal["rectangle"] = "rectangle"
    length_x_m: PositiveFloat = 0.20
    width_y_m: PositiveFloat = 0.08
    pivot_offset_m: tuple[float, float] = (0.0, 0.0)
    orientation_mode: OrientationMode = OrientationMode.FIXED
    cutting_inset_m: NonNegativeFloat = 0.005
    depth_z_m: float | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> ToolConfig:
        if self.orientation_mode is OrientationMode.VARIABLE:
            raise ValueError(
                "orientation_mode 'variable' es una extension con sus propios tests; "
                "H1 usa orientacion fija (§3.3)"
            )
        half_min = 0.5 * min(self.length_x_m, self.width_y_m)
        if self.cutting_inset_m >= half_min:
            raise ValueError(
                f"cutting_inset_m ({self.cutting_inset_m}) deja una huella de corte "
                f"degenerada para una cuchilla de {self.length_x_m}x{self.width_y_m} m"
            )
        if self.depth_z_m is not None:
            raise ValueError(
                "depth_z_m debe ser null en H1: la interfaz de profundidad Z esta "
                "reservada y no se simulan elevaciones instantaneas (§2)"
            )
        return self


class ToolFile(_StrictModel):
    """Contenido de `configs/tool.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    tool: ToolConfig = ToolConfig()
    provenance: Provenance = Provenance.SYNTHETIC


# --- configs/safety.yaml ----------------------------------------------------


class UncertaintyConfig(_StrictModel):
    """Presupuesto de incertidumbre y donde se aplica la inflacion."""

    localization_bound_m: NonNegativeFloat = 0.01
    calibration_bound_m: NonNegativeFloat = 0.01
    tracking_bound_m: NonNegativeFloat = 0.01
    bound_budget_m: PositiveFloat = 0.03
    sigma_multiplier_k: PositiveFloat = 3.0
    inflate: InflationTarget = InflationTarget.EXCLUSIONS

    @property
    def total_bound_m(self) -> float:
        return self.localization_bound_m + self.calibration_bound_m + self.tracking_bound_m

    def to_policy(self, domain: SpatialDomain) -> UncertaintyPolicy:
        """Politica de incertidumbre efectiva para una escena del dominio dado (R1).

        Las cotas de este archivo estan en metros: solo se aplican a dominios
        metricos. Un dominio no metrico necesita una politica declarada en su
        propia unidad, salvo que todas las cotas sean cero.
        """
        domain = SpatialDomain(domain)
        if not is_metric(domain) and self.total_bound_m > 0.0:
            raise ConfigError(
                f"safety.uncertainty se expresa en metros y no puede aplicarse al dominio "
                f"{domain.value}; declare una politica de incertidumbre en su unidad"
            )
        return UncertaintyPolicy(
            bound=UncertaintyBound(
                localization=self.localization_bound_m,
                calibration=self.calibration_bound_m,
                tracking=self.tracking_bound_m,
                domain=domain,
                provenance=Provenance.ASSUMED,
                k_sigma=self.sigma_multiplier_k,
            ),
            budget=self.bound_budget_m,
            inflate=self.inflate,
        )

    @model_validator(mode="after")
    def _check_budget(self) -> UncertaintyConfig:
        if self.total_bound_m > self.bound_budget_m + LIMIT_COMPARISON_TOLERANCE:
            raise ValueError(
                f"la suma de cotas ({self.total_bound_m} m) excede bound_budget_m "
                f"({self.bound_budget_m} m): superar el presupuesto invalida la intervencion"
            )
        return self


class SafetyConfig(_StrictModel):
    """Tolerancias de software y politica de dominio conocido."""

    clearance_epsilon_m: PositiveFloat = CLEARANCE_EPSILON_M
    geometric_approximation_tolerance_m: PositiveFloat = GEOMETRIC_APPROXIMATION_TOLERANCE_M
    limit_comparison_tolerance: PositiveFloat = LIMIT_COMPARISON_TOLERANCE
    max_conservative_discretization_excess_m: PositiveFloat = 5.0e-4
    require_known_domain: bool = True
    uncertainty: UncertaintyConfig = UncertaintyConfig()

    @model_validator(mode="after")
    def _check_tolerance_hierarchy(self) -> SafetyConfig:
        """Las tolerancias numericas no pueden absorber el despeje exigido (R12)."""
        if not (
            self.limit_comparison_tolerance
            < self.geometric_approximation_tolerance_m
            < self.clearance_epsilon_m
        ):
            raise ValueError(
                "se exige limit_comparison_tolerance < geometric_approximation_tolerance_m "
                "< clearance_epsilon_m: una tolerancia nunca puede aceptar contacto"
            )
        return self


class SafetyFile(_StrictModel):
    """Contenido de `configs/safety.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    safety: SafetyConfig = SafetyConfig()
    provenance: Provenance = Provenance.ASSUMED


# --- configs/kinematics.yaml ------------------------------------------------


class AxisLimits(_StrictModel):
    """Limites por eje en el marco declarado."""

    x: PositiveFloat
    y: PositiveFloat


class PositionLimits(_StrictModel):
    """Recorrido relativo admisible de la herramienta en el marco `V`."""

    x_min: float = -0.35
    x_max: float = 0.35
    y_min: float = -0.20
    y_max: float = 0.20

    @model_validator(mode="after")
    def _check_ranges(self) -> PositionLimits:
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("los limites de posicion deben definir un rectangulo no vacio")
        return self


class LatencyConfig(_StrictModel):
    """Latencias en segundos; se registran por separado (§3.5)."""

    acquisition_s: NonNegativeFloat = 0.03
    inference_s: NonNegativeFloat = 0.05
    planning_s: NonNegativeFloat = 0.02
    communication_s: NonNegativeFloat = 0.01
    mechanical_response_s: NonNegativeFloat = 0.08

    @property
    def total_s(self) -> float:
        return (
            self.acquisition_s
            + self.inference_s
            + self.planning_s
            + self.communication_s
            + self.mechanical_response_s
        )


class StartStateConfig(_StrictModel):
    """Estado inicial de la herramienta respecto al vehiculo."""

    x_m: float = 0.0
    y_m: float = 0.0
    contact: bool = False
    parked: bool = True


class KinematicsConfig(_StrictModel):
    """Limites cinematicos relativos al vehiculo (§3.3 condicion 3)."""

    frame: Literal[Frame.VEHICLE] = Frame.VEHICLE
    orientation_enabled: bool = False
    position_limits_m: PositionLimits = PositionLimits()
    velocity_limits_mps: AxisLimits = AxisLimits(x=0.60, y=0.40)
    acceleration_limits_mps2: AxisLimits = AxisLimits(x=3.0, y=2.0)
    latencies: LatencyConfig = LatencyConfig()
    horizon_s: PositiveFloat = 2.0
    start_state: StartStateConfig = StartStateConfig()

    @model_validator(mode="after")
    def _check_start_state(self) -> KinematicsConfig:
        limits = self.position_limits_m
        if not (limits.x_min <= self.start_state.x_m <= limits.x_max):
            raise ValueError("start_state.x_m esta fuera de position_limits_m")
        if not (limits.y_min <= self.start_state.y_m <= limits.y_max):
            raise ValueError("start_state.y_m esta fuera de position_limits_m")
        return self


class KinematicsFile(_StrictModel):
    """Contenido de `configs/kinematics.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    kinematics: KinematicsConfig = KinematicsConfig()
    provenance: Provenance = Provenance.ASSUMED


# --- configs/mechanism.yaml (A3/E3) -------------------------------------------


class JointConfig(_StrictModel):
    """Junta rotacional del baseline 1R: topes y limites de la referencia articular."""

    name: str = "swing"
    lower_rad: float = -1.2
    upper_rad: float = 1.2
    velocity_limit_rad_s: PositiveFloat = 1.0
    acceleration_limit_rad_s2: PositiveFloat = 3.0

    @model_validator(mode="after")
    def _ordered(self) -> JointConfig:
        if not self.lower_rad < self.upper_rad:
            raise ValueError("joint: lower_rad debe ser menor que upper_rad")
        return self


class MechanismConfig(_StrictModel):
    """Brazo oscilante de un pivote (orden A3/E3 §4, §7). Valores sinteticos.

    `T_VB(q) = Trans(a) Rot(q) Trans(L, 0) Rot(beta)`. La cuchilla es la herramienta
    `tool_id` de `configs/tool.yaml`, con su origen `B` en el extremo del brazo.
    """

    model: Literal["single_pivot_swing"] = "single_pivot_swing"
    model_id: str = "single_pivot_swing_bench"
    version: Annotated[int, Field(ge=1)] = 1
    domain: SpatialDomain = SpatialDomain.GROUND_METRIC_SYNTHETIC
    pivot_m: tuple[float, float] = (0.0, 0.0)
    length_m: PositiveFloat = 0.4
    beta_rad: float = 0.0
    arm_width_m: PositiveFloat = 0.02
    joint: JointConfig = JointConfig()
    tool_id: str = "blade_rect_200x80"


class MechanismFile(_StrictModel):
    """Contenido de `configs/mechanism.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    provenance: Provenance = Provenance.SYNTHETIC
    mechanism: MechanismConfig = MechanismConfig()


# --- configs/video.yaml (E0) -------------------------------------------------


class TimestampPolicyConfig(_StrictModel):
    """Politica temporal: PTS medidos primero; estimacion por indice solo etiquetada."""

    preferred_source: Literal["pts"] = "pts"
    allow_index_fps_estimate: bool = True
    gap_factor: Annotated[float, Field(gt=1.0)] = 1.5
    """Un intervalo mayor que `gap_factor * mediana` se registra como salto."""
    annotation_tolerance_s: Annotated[float, Field(gt=0.0, allow_inf_nan=False)] = 1.0e-6
    """Diferencia maxima entre el `timestamp_s` de una anotacion manual y el tiempo del
    cuadro decodificado; cubre el redondeo del CSV (6 decimales, error <= 5e-7 s).
    Debe ser menor que medio intervalo entre cuadros (se comprueba contra el clip)."""
    content_repeat_ratio: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.1
    """Un cuadro cuya diferencia media con el anterior es menor que esta fraccion de la
    mediana del clip se registra como repeticion de contenido (PTS regular)."""


class RoiConfig(_StrictModel):
    """Region de interes normalizada en imagen (`u` derecha, `v` abajo), en [0, 1]."""

    u_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    u_max: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    v_min: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    v_max: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0
    rationale: str = "fotograma completo"

    @model_validator(mode="after")
    def _check_box(self) -> RoiConfig:
        if self.u_max <= self.u_min or self.v_max <= self.v_min:
            raise ValueError("ROI vacia: se exige u_min < u_max y v_min < v_max")
        return self


class SamplingConfig(_StrictModel):
    """Muestreo determinista de fotogramas."""

    uniform_count: Annotated[int, Field(ge=1, le=500)] = 12
    problem_count_per_metric: Annotated[int, Field(ge=0, le=20)] = 2
    min_separation_frames: Annotated[int, Field(ge=0)] = 15
    analysis_scale: Annotated[float, Field(gt=0.0, le=1.0)] = 0.25
    jpeg_quality: Annotated[int, Field(ge=50, le=100)] = 92


class FlowConfig(_StrictModel):
    """Flujo optico exploratorio (Lucas-Kanade piramidal con verificacion ida/vuelta)."""

    method: Literal["lk_pyramidal_fb"] = "lk_pyramidal_fb"
    frame_step: Annotated[int, Field(ge=1)] = 1
    max_corners: Annotated[int, Field(ge=10)] = 400
    quality_level: Annotated[float, Field(gt=0.0, lt=1.0)] = 0.01
    min_distance_px: PositiveFloat = 7.0
    window_px: Annotated[int, Field(ge=5)] = 21
    pyramid_levels: Annotated[int, Field(ge=0, le=8)] = 3
    fb_max_error_px: PositiveFloat = 1.0
    min_valid_points: Annotated[int, Field(ge=3)] = 20
    direction_dominance: Annotated[float, Field(gt=0.5, lt=1.0)] = 0.7
    """Fraccion minima de vectores validos en un sentido para declararlo dominante."""
    min_motion_px: PositiveFloat = 0.5
    """Por debajo de este modulo mediano (pixeles de analisis) el par es `none`."""
    example_count: Annotated[int, Field(ge=0, le=50)] = 6


class VideoConfig(_StrictModel):
    """Caracterizacion exploratoria de video (E0). Todo en pixeles; sin escala."""

    source: str = "resources/moringa_sin_ai.mp4"
    expected_sha256: str | None = None
    output_dir: str = "experiments/phase0"
    manifest_path: str = "data/video_manifest.csv"
    timestamps: TimestampPolicyConfig = TimestampPolicyConfig()
    roi: RoiConfig = RoiConfig()
    sampling: SamplingConfig = SamplingConfig()
    flow: FlowConfig = FlowConfig()

    @field_validator("expected_sha256")
    @classmethod
    def _check_sha(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("expected_sha256 debe ser 64 caracteres hexadecimales en minuscula")
        return value


class VideoFile(_StrictModel):
    """Contenido de `configs/video.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    mode: Literal[RunMode.EXPLORATORY] = RunMode.EXPLORATORY
    video: VideoConfig = VideoConfig()
    provenance: Provenance = Provenance.ASSUMED


# --- calibracion y configuracion de proyecto --------------------------------


class CalibrationConfig(_StrictModel):
    """Referencia declarada de calibracion; ausente por defecto."""

    calibration_id: str = "absent"
    status: CalibrationStatus = CalibrationStatus.ABSENT
    scale_evidence: str | None = None
    independent_point_error: NonNegativeFloat | None = None
    domain: SpatialDomain = SpatialDomain.IMAGE_PX

    def to_reference(self) -> CalibrationReference:
        """Construye el contrato `CalibrationReference` correspondiente."""
        return CalibrationReference(
            calibration_id=self.calibration_id,
            status=self.status,
            scale_evidence=self.scale_evidence,
            independent_point_error=self.independent_point_error,
            domain=self.domain,
        )


class ProjectConfig(_StrictModel):
    """Configuracion resuelta del banco sintetico de E1."""

    schema_version: Literal[1] = SCHEMA_VERSION
    mode: RunMode = RunMode.SYNTHETIC
    scene: SceneConfig = SceneConfig()
    tool: ToolConfig = ToolConfig()
    safety: SafetyConfig = SafetyConfig()
    kinematics: KinematicsConfig = KinematicsConfig()
    calibration: CalibrationConfig = CalibrationConfig()

    @model_validator(mode="after")
    def _check_mode_and_domains(self) -> ProjectConfig:
        allowed = _ALLOWED_DOMAINS[self.mode]
        pairs = (("scene.domain", self.scene.domain), ("tool.domain", self.tool.domain))
        for label, domain in pairs:
            if domain not in allowed:
                raise ValueError(
                    f"{label}={domain.value} no es licito en mode={self.mode.value}; "
                    f"dominios admitidos: {sorted(d.value for d in allowed)}"
                )
        if self.scene.domain is not self.tool.domain:
            raise ValueError(
                f"scene.domain ({self.scene.domain.value}) y tool.domain "
                f"({self.tool.domain.value}) deben coincidir: no se mezclan dominios"
            )
        # R2: la referencia se valida siempre (evidencia no vacia y dominio coherente);
        # el modo calibrado ademas exige que este verificada. MissingCalibrationError
        # no es ValueError: se propaga sin quedar oculta como error generico.
        reference = self.calibration.to_reference()
        if self.mode is RunMode.CALIBRATED:
            reference.require_verified("mode=calibrated")
        self._check_uncertainty_coherence()
        self._check_discretization_excess()
        return self

    def _check_uncertainty_coherence(self) -> None:
        """R1: el margen aplicado por la escena cubre la cota declarada en safety."""
        uncertainty = self.safety.uncertainty
        total = uncertainty.total_bound_m
        margin = self.scene.exclusion.uncertainty_margin_m
        if not is_metric(self.scene.domain):
            if total > 0.0:
                raise ValueError(
                    f"incertidumbre: safety.uncertainty esta en metros y la escena usa el "
                    f"dominio no metrico {self.scene.domain.value}"
                )
            return
        if uncertainty.inflate is InflationTarget.EXCLUSIONS:
            if margin + LIMIT_COMPARISON_TOLERANCE < total:
                raise ValueError(
                    f"incertidumbre: scene.exclusion.uncertainty_margin_m ({margin} m) es menor "
                    f"que la suma de cotas declaradas en safety ({total} m); la proteccion "
                    "efectiva debe cubrir la incertidumbre de localizacion"
                )
        elif margin != 0.0:
            raise ValueError(
                "incertidumbre: safety.uncertainty.inflate=tool exige "
                "scene.exclusion.uncertainty_margin_m=0; inflar exclusiones y herramienta "
                "duplicaria el margen"
            )

    def _check_discretization_excess(self) -> None:
        """R12: el exceso conservador de los poligonos de exclusion queda acotado."""
        if not is_metric(self.scene.domain):
            return
        exclusion = self.scene.exclusion
        violation = discretization_excess_violation(
            exclusion.protection_radius_m + exclusion.uncertainty_margin_m,
            exclusion.polygon_segments,
            self.safety.max_conservative_discretization_excess_m,
        )
        if violation is not None:
            raise ValueError(violation)

    @property
    def calibration_reference(self) -> CalibrationReference:
        return self.calibration.to_reference()

    def resolved_dict(self) -> dict[str, Any]:
        """Configuracion resuelta con todos los valores por defecto materializados."""
        return self.model_dump(mode="json")

    def resolved_yaml(self) -> str:
        """Configuracion resuelta serializada como YAML (`runs/<id>/resolved_config.yaml`)."""
        return yaml.safe_dump(self.resolved_dict(), sort_keys=True, allow_unicode=True)


def discretization_excess_violation(radius: float, segments: int, limit: float) -> str | None:
    """Regla unica de R12: exceso radial conservador frente a su presupuesto.

    Devuelve el motivo del rechazo, o `None` si `r*(sec(pi/n)-1) <= limit`. La usan
    `ProjectConfig` (antes de construir) y `scenarios.synthetic` (sobre las
    exclusiones construidas), para que ninguna ruta diverja.
    """
    excess = circumscribed_radial_excess(radius, segments)
    if excess <= limit:
        return None
    return (
        f"discretizacion: {segments} lados sobre radio {radius} m producen un exceso "
        f"radial conservador de {excess:.3e} m > max_conservative_discretization_excess_m "
        f"{limit:.3e} m"
    )


# --- configs/planner.yaml y configs/validation.yaml (A1/E2) ---------------------


class CostWeightsConfig(_StrictModel):
    """Pesos del costo normalizado. La seguridad no es un termino de costo (§4.2)."""

    length: NonNegativeFloat = 1.0
    deviation: NonNegativeFloat = 1.0
    smoothness: NonNegativeFloat = 0.25

    @model_validator(mode="after")
    def _not_all_zero(self) -> CostWeightsConfig:
        if self.length + self.deviation + self.smoothness <= 0.0:
            raise ValueError("al menos un peso de costo debe ser positivo")
        return self


class CostNormalizersConfig(_StrictModel):
    """Escalas que hacen adimensionales los terminos de costo.

    - longitud: exceso relativo sobre la distancia recta inicio-meta (sin escala).
    - desviacion: area lateral media respecto de la recta inicio-meta / `deviation_m`.
    - suavidad: suma de cuadrados de cambios de rumbo / `smoothness_rad2`.
    """

    deviation_m: PositiveFloat = 0.25
    smoothness_rad2: PositiveFloat = 1.0


class PlannerConfig(_StrictModel):
    """Planificador geometrico E2: niveles longitudinales y candidatos laterales (§3.4)."""

    strategy: Literal["longitudinal_levels"] = "longitudinal_levels"
    smoothing: Literal["quintic_lane_change"] = "quintic_lane_change"
    generation_margin_m: PositiveFloat = 0.01
    """Holgura extra sobre los obstaculos de configuracion al **generar** candidatos.
    Solo afecta la propuesta; la aceptacion la decide el validador sin ella."""
    level_padding_m: NonNegativeFloat = 0.05
    """Distancia longitudinal entre el borde de un obstaculo de configuracion y los
    niveles que lo rodean."""
    lateral_candidates_per_interval: Annotated[int, Field(ge=1, le=9)] = 3
    max_candidates: Annotated[int, Field(ge=1, le=50)] = 8
    """Candidatos ordenados por costo que se intentan suavizar y validar."""
    max_graph_edges: Annotated[int, Field(ge=1)] = 200_000
    """Presupuesto determinista de aristas evaluadas por la busqueda."""
    seed: Annotated[int, Field(ge=0)] = 0
    """El algoritmo es determinista y no usa aleatoriedad; la semilla se registra."""
    weights: CostWeightsConfig = CostWeightsConfig()
    normalizers: CostNormalizersConfig = CostNormalizersConfig()


class PlanDemoConfig(_StrictModel):
    """Caso de demostracion: escena del catalogo e inicio/meta del origen `B` en `W`.

    Los extremos pertenecen a este perfil de demo, no a la escena E1.
    """

    scene: str = "nominal_three_plants"
    start_xy_m: tuple[float, float] = (0.0, 0.10)
    goal_xy_m: tuple[float, float] = (0.0, 4.90)
    theta_rad: float = 0.0


class RandomScenariosConfig(_StrictModel):
    """Generador congelado de escenarios E2-08 (semillas `seed_start..seed_start+count-1`)."""

    seed_start: Annotated[int, Field(ge=0)] = 0
    count: Annotated[int, Field(ge=1)] = 100
    corridor_width_m: tuple[PositiveFloat, PositiveFloat] = (0.8, 1.4)
    corridor_length_m: PositiveFloat = 5.0
    plant_count: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] = (0, 6)
    plant_y_margin_m: PositiveFloat = 0.6
    plant_offset_fraction: Annotated[float, Field(ge=0.0, le=0.5)] = 0.35
    """Desplazamiento lateral maximo de una planta, como fraccion del ancho."""
    start_y_m: PositiveFloat = 0.10
    goal_margin_m: PositiveFloat = 0.10

    @model_validator(mode="after")
    def _ranges(self) -> RandomScenariosConfig:
        if self.corridor_width_m[1] < self.corridor_width_m[0]:
            raise ValueError("corridor_width_m: maximo < minimo")
        if self.plant_count[1] < self.plant_count[0]:
            raise ValueError("plant_count: maximo < minimo")
        if 2.0 * self.plant_y_margin_m >= self.corridor_length_m:
            raise ValueError("plant_y_margin_m deja un tramo de plantas vacio")
        return self


class PlannerFile(_StrictModel):
    """Contenido de `configs/planner.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    provenance: Provenance = Provenance.ASSUMED
    planner: PlannerConfig = PlannerConfig()
    demo: PlanDemoConfig = PlanDemoConfig()
    random_scenarios: RandomScenariosConfig = RandomScenariosConfig()


class ValidationConfig(_StrictModel):
    """Validador independiente de barrido (§4.3).

    El presupuesto de error de envolvente es `safety.geometric_approximation_tolerance_m`
    y el despeje `safety.clearance_epsilon_m`: no se duplican aqui.
    """

    max_subdivision_depth: Annotated[int, Field(ge=0, le=40)] = 16
    max_intervals: Annotated[int, Field(ge=1)] = 400_000
    """Presupuesto total de intervalos evaluados por validacion; agotarlo da UNCERTIFIED."""
    continuity_tolerance: PositiveFloat = 1.0e-8
    """Error absoluto admitido en p, dp/ds y d2p/ds2 en las uniones (E2-04)."""
    endpoint_tolerance_m: PositiveFloat = 1.0e-9
    """Error absoluto admitido entre extremos del camino e inicio/meta pedidos."""
    coverage_method: Literal["chord_hull_union"] = "chord_hull_union"


class ValidationFile(_StrictModel):
    """Contenido de `configs/validation.yaml`."""

    schema_version: Literal[1] = SCHEMA_VERSION
    provenance: Provenance = Provenance.ASSUMED
    validation: ValidationConfig = ValidationConfig()


# --- carga ------------------------------------------------------------------


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    """Carga un YAML como mapeo mediante `safe_load`.

    Raises:
        ConfigError: si el archivo no existe, no es YAML valido o no es un mapeo.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigError(f"configuracion inexistente: {file_path}")
    try:
        loaded = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{file_path}: YAML invalido ({exc})") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{file_path}: se espera un mapeo en la raiz, se obtuvo {type(loaded)}")
    return loaded


def validate_config_payload(
    model: type[BaseModel], payload: dict[str, Any], source: str = "configuracion"
) -> Any:
    """Valida un mapeo contra un modelo de configuracion.

    Traduce los errores de pydantic a la jerarquia del proyecto: una clave
    desconocida produce `UnknownConfigKeyError`, nunca se ignora (§7).
    """
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        if any(error["type"] == "extra_forbidden" for error in exc.errors()):
            keys = ", ".join(
                ".".join(str(part) for part in error["loc"])
                for error in exc.errors()
                if error["type"] == "extra_forbidden"
            )
            raise UnknownConfigKeyError(f"{source}: claves desconocidas ({keys})") from exc
        raise ConfigError(f"{source}: configuracion invalida\n{exc}") from exc


def load_project_config(
    synthetic_path: str | Path,
    tool_path: str | Path,
    safety_path: str | Path,
    kinematics_path: str | Path,
    calibration: dict[str, Any] | None = None,
) -> ProjectConfig:
    """Carga y resuelve los cuatro YAML de E1 en una `ProjectConfig`.

    Raises:
        ConfigError / UnknownConfigKeyError: configuracion invalida o clave
            desconocida en cualquiera de los archivos.
        MissingCalibrationError: si el modo resultante exige calibracion
            verificada y no se aporta, o si se declara verificada sin evidencia.
    """
    synthetic = validate_config_payload(
        SyntheticFile, load_yaml_mapping(synthetic_path), str(synthetic_path)
    )
    tool = validate_config_payload(ToolFile, load_yaml_mapping(tool_path), str(tool_path))
    safety = validate_config_payload(SafetyFile, load_yaml_mapping(safety_path), str(safety_path))
    kinematics = validate_config_payload(
        KinematicsFile, load_yaml_mapping(kinematics_path), str(kinematics_path)
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "mode": synthetic.mode,
        "scene": synthetic.scene.model_dump(),
        "tool": tool.tool.model_dump(),
        "safety": safety.safety.model_dump(),
        "kinematics": kinematics.kinematics.model_dump(),
    }
    if calibration is not None:
        payload["calibration"] = calibration
    try:
        return ProjectConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(f"configuracion de proyecto invalida\n{exc}") from exc
