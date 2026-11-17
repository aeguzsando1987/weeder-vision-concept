"""Observaciones manuales de bases de tallo y evaluacion manual del clip (E0, §3-§6).

Todo lo de este modulo es **observacion visual manual**, no salida de detector:
`source` es siempre `manual_visual`. Las posiciones son pixeles del cuadro
decodificado (`u` derecha, `v` abajo) con una incertidumbre declarada; una
inspeccion visual no acredita precision centimetrica.

Cada fila declara **que punto** marca (`observed_target`): `ground_contact` es el
contacto tallo-suelo; `stem_axis_proxy` es un punto del eje/copa visto desde arriba,
a altura desconocida, que no equivale a una base en el suelo; `none` indica que no se
localizo ningun punto. `base_visibility` es la visibilidad del punto anotado tal
como se registro. El objetivo lo fija el anotador; nunca se infiere del texto libre.

La evaluacion manual (`manual_assessment.yaml`) es la unica via para fijar
visibilidad de bases, montaje, escala, homografia, temporalidad y aptitud A/B/C.
Esta vinculada al clip por `video_id` y SHA-256, y a su tabla por SHA-256. Validar
el esquema (`ManualAssessment.load`) no basta para publicar resultados: antes hay
que contrastarla con los cuadros decodificados (`validate_against_clip`). Una aptitud
`A` exige al menos tres plantas con contacto tallo-suelo localizado en dos o mas
cuadros, una linea virtual de cruce y decodificacion hasta EOF.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ..core.errors import ConfigError, PrecisionWeederError, SchemaValidationError
from .manifest import (
    UNKNOWN,
    ApparentMotion,
    HomographyFeasibility,
    MountingStatus,
    ObservationTiming,
    ScaleStatus,
    StemBaseVisibility,
    Suitability,
)

__all__ = [
    "CROSSING_COLUMNS",
    "OBSERVATION_COLUMNS",
    "BaseVisibility",
    "ClipFacts",
    "CrossingEvent",
    "CrossingLine",
    "ManualAssessment",
    "ObservedTarget",
    "StemBaseObservation",
    "crossing_events",
    "read_stem_observations",
    "validate_against_clip",
]

ASSESSMENT_SCHEMA_VERSION = 2

OBSERVATION_COLUMNS = (
    "event_id",
    "plant_id",
    "frame_index",
    "timestamp_s",
    "timestamp_method",
    "base_visibility",
    "observed_target",
    "u_px",
    "v_px",
    "uncertainty_px",
    "confidence",
    "ambiguity_cause",
    "source",
    "notes",
)


class BaseVisibility(StrEnum):
    VISIBLE = "visible"
    PARTIAL = "partial"
    HIDDEN = "hidden"


class ObservedTarget(StrEnum):
    GROUND_CONTACT = "ground_contact"
    STEM_AXIS_PROXY = "stem_axis_proxy"
    NONE = "none"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class StemBaseObservation:
    """Un evento manual: base de una planta en un cuadro concreto."""

    event_id: str
    plant_id: str
    frame_index: int
    timestamp_s: float
    timestamp_method: str
    base_visibility: BaseVisibility
    observed_target: ObservedTarget
    u_px: float | None
    v_px: float | None
    uncertainty_px: float | None
    confidence: Confidence
    ambiguity_cause: str
    source: str
    notes: str

    def __post_init__(self) -> None:
        context = f"observacion {self.event_id}"
        if not self.event_id or not self.plant_id:
            raise SchemaValidationError(f"{context}: event_id y plant_id son obligatorios")
        if self.frame_index < 0:
            raise SchemaValidationError(f"{context}: frame_index negativo")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise SchemaValidationError(f"{context}: timestamp_s invalido")
        if self.timestamp_method not in ("pts", "index_fps_estimate"):
            raise SchemaValidationError(f"{context}: timestamp_method desconocido")
        try:
            object.__setattr__(self, "base_visibility", BaseVisibility(self.base_visibility))
            object.__setattr__(self, "observed_target", ObservedTarget(self.observed_target))
            object.__setattr__(self, "confidence", Confidence(self.confidence))
        except ValueError as exc:
            raise SchemaValidationError(f"{context}: {exc}") from exc
        if self.source != "manual_visual":
            raise SchemaValidationError(
                f"{context}: source debe ser manual_visual; no se mezclan salidas de detector"
            )
        located = (self.u_px, self.v_px)
        hidden = self.base_visibility is BaseVisibility.HIDDEN
        if hidden != (self.observed_target is ObservedTarget.NONE):
            raise SchemaValidationError(
                f"{context}: base_visibility hidden exige observed_target none, y al reves"
            )
        if hidden:
            if any(value is not None for value in (*located, self.uncertainty_px)):
                raise SchemaValidationError(f"{context}: base oculta sin posicion medida")
        else:
            if any(value is None or not math.isfinite(value) for value in located):
                raise SchemaValidationError(f"{context}: base visible/parcial exige u_px y v_px")
            if self.uncertainty_px is None or not (
                math.isfinite(self.uncertainty_px) and self.uncertainty_px > 0
            ):
                raise SchemaValidationError(f"{context}: uncertainty_px > 0 obligatorio")


def _parse_optional_float(raw: str, name: str) -> float | None:
    text = raw.strip()
    if text == UNKNOWN:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise SchemaValidationError(f"{name}: real invalido {raw!r}") from exc
    if not math.isfinite(value):
        raise SchemaValidationError(f"{name}: valor no finito")
    return value


def read_stem_observations(
    path: str | Path, frame_size: tuple[int, int] | None = None
) -> list[StemBaseObservation]:
    """Lee la tabla manual; con `frame_size` verifica que las posiciones caen en el cuadro."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OBSERVATION_COLUMNS:
            raise SchemaValidationError(f"cabecera de observaciones invalida: {reader.fieldnames}")
        rows = list(reader)
    observations = []
    for row in rows:
        try:
            frame_index = int(row["frame_index"])
            timestamp_s = float(row["timestamp_s"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"{row.get('event_id')}: indice o tiempo invalido") from exc
        observation = StemBaseObservation(
            event_id=row["event_id"].strip(),
            plant_id=row["plant_id"].strip(),
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            timestamp_method=row["timestamp_method"].strip(),
            base_visibility=row["base_visibility"].strip(),  # type: ignore[arg-type]
            observed_target=row["observed_target"].strip(),  # type: ignore[arg-type]
            u_px=_parse_optional_float(row["u_px"], "u_px"),
            v_px=_parse_optional_float(row["v_px"], "v_px"),
            uncertainty_px=_parse_optional_float(row["uncertainty_px"], "uncertainty_px"),
            confidence=row["confidence"].strip(),  # type: ignore[arg-type]
            ambiguity_cause=row["ambiguity_cause"].strip(),
            source=row["source"].strip(),
            notes=row["notes"].strip(),
        )
        if frame_size is not None and observation.u_px is not None:
            width, height = frame_size
            if not (0 <= observation.u_px < width and 0 <= observation.v_px < height):  # type: ignore[operator]
                raise SchemaValidationError(f"{observation.event_id}: posicion fuera del cuadro")
        observations.append(observation)
    ids = [o.event_id for o in observations]
    if len(set(ids)) != len(ids):
        raise SchemaValidationError("event_id duplicado en observaciones")
    keys = [(o.plant_id, o.frame_index) for o in observations]
    if len(set(keys)) != len(keys):
        raise SchemaValidationError("misma planta observada dos veces en el mismo cuadro")
    return observations


# --- evaluacion manual -------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CrossingLine(_Strict):
    """Linea virtual horizontal de cruce en pixeles de imagen (offline, sin escala)."""

    v_px: Annotated[float, Field(ge=0.0)]
    u_min_px: Annotated[float, Field(ge=0.0)]
    u_max_px: Annotated[float, Field(gt=0.0)]
    crossing_direction: Literal["up", "down"]
    rationale: str

    @model_validator(mode="after")
    def _check(self) -> CrossingLine:
        if self.u_max_px <= self.u_min_px:
            raise ValueError("u_max_px debe ser > u_min_px")
        return self


class KnowledgeItem(_Strict):
    status: MountingStatus
    value: str | None = None
    source: str


class Mounting(_Strict):
    camera_rigid_to_vehicle: KnowledgeItem
    orientation: KnowledgeItem
    height: KnowledgeItem
    position_relative_to_tool: KnowledgeItem
    tool_visible_in_frame: bool

    @property
    def overall(self) -> MountingStatus:
        statuses = [
            item.status
            for item in (
                self.camera_rigid_to_vehicle,
                self.orientation,
                self.height,
                self.position_relative_to_tool,
            )
        ]
        if MountingStatus.UNKNOWN in statuses:
            return MountingStatus.UNKNOWN
        if MountingStatus.ASSUMED in statuses:
            return MountingStatus.ASSUMED
        return MountingStatus.KNOWN


class ScaleAssessment(_Strict):
    status: ScaleStatus
    evidence_ref: str | None = None
    candidates_examined: list[str]
    notes: str

    @model_validator(mode="after")
    def _check(self) -> ScaleAssessment:
        if self.status is ScaleStatus.VERIFIED:
            raise ValueError("E0 no valida escala: la verificacion metrica pertenece a E4")
        return self


class HomographyAssessment(_Strict):
    feasibility: HomographyFeasibility
    planar_ground_in_roi: Literal["plausible", "doubtful", "unknown"]
    notes: str

    @model_validator(mode="after")
    def _check(self) -> HomographyAssessment:
        if self.feasibility is HomographyFeasibility.VALIDATED:
            raise ValueError("E0 no valida homografias: la validacion pertenece a E4")
        return self


class FlowManualCheck(_Strict):
    """Contraste manual de un intervalo: rasgo seguido a ojo entre dos cuadros."""

    check_id: str
    feature: str
    frame_from: Annotated[int, Field(ge=0)]
    frame_to: Annotated[int, Field(ge=0)]
    uv_from_px: tuple[float, float]
    uv_to_px: tuple[float, float]
    uncertainty_px: Annotated[float, Field(gt=0.0)]
    automated_median_du_px: float
    automated_median_dv_px: float
    notes: str

    @model_validator(mode="after")
    def _check(self) -> FlowManualCheck:
        if self.frame_to <= self.frame_from:
            raise ValueError("frame_to debe ser > frame_from")
        return self

    @property
    def manual_per_frame(self) -> tuple[float, float]:
        frames = self.frame_to - self.frame_from
        return (
            (self.uv_to_px[0] - self.uv_from_px[0]) / frames,
            (self.uv_to_px[1] - self.uv_from_px[1]) / frames,
        )

    def agreement(self) -> dict[str, float | bool]:
        """Compara el desplazamiento manual por cuadro con la mediana automatica."""
        frames = self.frame_to - self.frame_from
        manual_du, manual_dv = self.manual_per_frame
        tolerance = 2.0 * self.uncertainty_px / frames
        return {
            "manual_du_per_frame_px": manual_du,
            "manual_dv_per_frame_px": manual_dv,
            "difference_dv_px": manual_dv - self.automated_median_dv_px,
            "difference_du_px": manual_du - self.automated_median_du_px,
            "tolerance_per_frame_px": tolerance,
            "same_vertical_sign": (manual_dv > 0) == (self.automated_median_dv_px > 0),
            "within_tolerance": abs(manual_dv - self.automated_median_dv_px) <= tolerance
            and abs(manual_du - self.automated_median_du_px) <= tolerance,
        }


class ConditionCheck(_Strict):
    condition: Literal[
        "shadow_sun",
        "blur",
        "occlusion",
        "vibration",
        "dense_weeds",
        "illumination_change",
        "plant_entry_exit",
    ]
    observed: Literal["yes", "no", "inconclusive"]
    frames: list[int]
    notes: str


class IntervalAssessment(_Strict):
    start_frame: Annotated[int, Field(ge=0)]
    end_frame: Annotated[int, Field(ge=0)]
    suitability: Suitability
    reason: str

    @model_validator(mode="after")
    def _check(self) -> IntervalAssessment:
        if self.end_frame < self.start_frame:
            raise ValueError("end_frame < start_frame")
        return self


class ManualAssessment(_Strict):
    """Contenido de `experiments/phase0/<video_id>/manual_assessment.yaml`."""

    schema_version: Literal[2]
    video_id: str
    video_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    """SHA-256 del clip anotado; la evaluacion no se aplica a otro contenido."""
    stem_observations_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    """SHA-256 de la tabla de observaciones referida en `stem_observations_ref`."""
    assessor: str
    method: str
    session_id: str | None = None
    site_id: str | None = None
    stem_base_visibility: StemBaseVisibility
    stem_observations_ref: str
    crossing_line: CrossingLine | None
    apparent_motion_manual: ApparentMotion
    flow_manual_checks: list[FlowManualCheck]
    mounting: Mounting
    observation_timing: ObservationTiming
    replay_suitability: Suitability
    anticipation_validation: Literal["possible", "not_possible", "unknown"]
    anticipation_reason: str
    scale: ScaleAssessment
    homography: HomographyAssessment
    conditions: list[ConditionCheck]
    intervals: list[IntervalAssessment]
    suitability: Suitability
    suitability_reason: str

    @property
    def mounting_status(self) -> MountingStatus:
        return self.mounting.overall

    @model_validator(mode="after")
    def _check_consistency(self) -> ManualAssessment:
        if self.suitability is not Suitability.PENDING and not self.suitability_reason.strip():
            raise ValueError("suitability sin motivo")
        if self.suitability is Suitability.A and self.crossing_line is None:
            raise ValueError("suitability A exige una linea virtual de cruce")
        if (
            self.observation_timing is ObservationTiming.BEFORE_INTERVENTION
            and self.mounting.position_relative_to_tool.status is MountingStatus.UNKNOWN
        ):
            raise ValueError(
                "observation_timing before_intervention exige conocer la posicion de la "
                "camara respecto de la herramienta"
            )
        if self.anticipation_validation == "possible" and (
            self.mounting.position_relative_to_tool.status is not MountingStatus.KNOWN
        ):
            raise ValueError("anticipacion validable exige posicion camara-herramienta conocida")
        return self

    @classmethod
    def load(
        cls, path: str | Path, repo_root: str | Path
    ) -> tuple[ManualAssessment, list[StemBaseObservation]]:
        """Carga la evaluacion y su tabla; valida esquema, hash de la tabla y la regla A.

        Es una validacion **sin contexto de clip**: no basta para publicar resultados.
        Antes hay que llamar a `validate_against_clip` con los cuadros decodificados.

        Raises:
            ConfigError: YAML fuera de esquema (incluida la version 1 sin migrar), tabla
                inexistente o con hash distinto, o tabla que no respalda la aptitud A.
        """
        file_path = Path(path)
        try:
            payload = yaml.safe_load(file_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigError(f"{file_path}: evaluacion manual ilegible\n{exc}") from exc
        if isinstance(payload, dict) and payload.get("schema_version") == 1:
            raise ConfigError(
                f"{file_path}: schema_version 1 no vincula la evaluacion al clip. Migre a la "
                "version 2: anada video_sha256 y stem_observations_sha256, y la columna "
                "observed_target en la tabla (ver docs/data_contracts.md)"
            )
        try:
            assessment = cls.model_validate(payload)
        except ValidationError as exc:
            raise ConfigError(f"{file_path}: evaluacion manual invalida\n{exc}") from exc
        observations_path = Path(repo_root) / assessment.stem_observations_ref
        if not observations_path.is_file():
            raise ConfigError(f"{file_path}: stem_observations_ref no resuelve")
        table_digest = hashlib.sha256(observations_path.read_bytes()).hexdigest()
        if table_digest != assessment.stem_observations_sha256:
            raise ConfigError(
                f"{file_path}: la tabla {assessment.stem_observations_ref} tiene sha256 "
                f"{table_digest}, distinto del declarado {assessment.stem_observations_sha256}"
            )
        try:
            observations = read_stem_observations(observations_path)
        except SchemaValidationError as exc:
            raise ConfigError(f"{observations_path}: {exc}") from exc
        if assessment.suitability is Suitability.A:
            tracked = assessment.tracked_plants(observations)
            if len(tracked) < 3:
                raise ConfigError(
                    f"{file_path}: suitability A exige >= 3 plantas con contacto tallo-suelo "
                    f"(ground_contact) localizado en >= 2 cuadros; hay {len(tracked)}"
                )
        return assessment, observations

    @staticmethod
    def tracked_plants(
        observations: list[StemBaseObservation],
        target: ObservedTarget = ObservedTarget.GROUND_CONTACT,
    ) -> dict[str, int]:
        """Plantas con el objetivo indicado localizado en al menos dos cuadros distintos.

        Por defecto cuenta solo contactos tallo-suelo; un proxy de eje/copa no se
        cuenta como base en el suelo.
        """
        frames: dict[str, set[int]] = {}
        for observation in observations:
            if observation.observed_target is target:
                frames.setdefault(observation.plant_id, set()).add(observation.frame_index)
        return {plant: len(f) for plant, f in sorted(frames.items()) if len(f) >= 2}

    def reconcile_motion(self, automated: ApparentMotion) -> ApparentMotion:
        """La direccion se publica solo si el contraste manual coincide con el flujo.

        Raises:
            PrecisionWeederError: si manual y automatico discrepan; se resuelve
                inspeccionando, no eligiendo uno en silencio.
        """
        if self.apparent_motion_manual is ApparentMotion.UNKNOWN:
            return automated
        if self.apparent_motion_manual is not automated:
            raise PrecisionWeederError(
                f"direccion aparente manual ({self.apparent_motion_manual.value}) distinta de "
                f"la automatica ({automated.value}); revise el contraste manual"
            )
        return automated


@dataclass(frozen=True)
class ClipFacts:
    """Hechos del clip decodificado contra los que se valida una evaluacion manual."""

    video_id: str
    sha256: str
    frame_count: int
    width_px: int
    height_px: int
    timeline_s: tuple[float, ...]
    timestamp_source: str
    eof_reached: bool


def validate_against_clip(
    assessment: ManualAssessment,
    observations: list[StemBaseObservation],
    clip: ClipFacts,
    timestamp_tolerance_s: float,
) -> None:
    """Contrasta evaluacion y tabla con los cuadros realmente decodificados.

    Comprueba identidad (ID y SHA-256), indices de cuadro existentes, tiempos
    coherentes con la linea temporal del clip (con tolerancia explicita para tiempos
    redondeados en el CSV, siempre menor que medio intervalo entre cuadros), metodo
    temporal, posiciones, linea virtual, intervalos, condiciones y contrastes de flujo
    dentro de imagen/clip, y EOF para la aptitud A. Se usa en el caracterizador y en
    el renderer antes de escribir resultados.

    Raises:
        ConfigError: con la lista completa de incoherencias encontradas.
    """
    problems: list[str] = []
    if len(clip.timeline_s) != clip.frame_count:
        raise ConfigError("ClipFacts incoherente: timeline_s no cubre todos los cuadros")
    if assessment.video_id != clip.video_id:
        problems.append(f"video_id {assessment.video_id!r} != clip {clip.video_id!r}")
    if assessment.video_sha256 != clip.sha256:
        problems.append(f"video_sha256 {assessment.video_sha256} != clip {clip.sha256}")

    if not (math.isfinite(timestamp_tolerance_s) and timestamp_tolerance_s > 0):
        problems.append(f"tolerancia temporal invalida: {timestamp_tolerance_s!r}")
    steps = [b - a for a, b in itertools.pairwise(clip.timeline_s) if b > a]
    if steps and timestamp_tolerance_s >= 0.5 * min(steps):
        problems.append(
            f"tolerancia temporal {timestamp_tolerance_s} s >= medio intervalo minimo "
            f"({0.5 * min(steps)} s): toleraria un cuadro equivocado"
        )

    def frame_ok(index: int, what: str) -> bool:
        if not 0 <= index < clip.frame_count:
            problems.append(f"{what}: cuadro {index} fuera del clip (0..{clip.frame_count - 1})")
            return False
        return True

    def point_ok(u: float, v: float, what: str) -> None:
        if not (0 <= u < clip.width_px and 0 <= v < clip.height_px):
            problems.append(
                f"{what}: punto ({u}, {v}) fuera de la imagen {clip.width_px}x{clip.height_px}"
            )

    for obs in observations:
        what = f"observacion {obs.event_id}"
        if obs.timestamp_method != clip.timestamp_source:
            problems.append(
                f"{what}: timestamp_method {obs.timestamp_method} != fuente del clip "
                f"{clip.timestamp_source}"
            )
        if frame_ok(obs.frame_index, what):
            expected = clip.timeline_s[obs.frame_index]
            if abs(obs.timestamp_s - expected) > timestamp_tolerance_s:
                problems.append(
                    f"{what}: timestamp_s {obs.timestamp_s} no corresponde al cuadro "
                    f"{obs.frame_index} ({expected} s, tolerancia {timestamp_tolerance_s} s)"
                )
        if obs.u_px is not None and obs.v_px is not None:
            point_ok(obs.u_px, obs.v_px, what)

    line = assessment.crossing_line
    if line is not None:
        if line.v_px >= clip.height_px:
            problems.append(f"linea virtual v={line.v_px} fuera de la imagen")
        if line.u_max_px > clip.width_px:
            problems.append(f"linea virtual u_max={line.u_max_px} > ancho {clip.width_px}")

    previous_end = -1
    for interval in sorted(assessment.intervals, key=lambda item: item.start_frame):
        what = f"intervalo {interval.start_frame}-{interval.end_frame}"
        frame_ok(interval.end_frame, what)
        if interval.start_frame <= previous_end:
            problems.append(f"{what}: se solapa con el intervalo anterior")
        previous_end = interval.end_frame
    for condition in assessment.conditions:
        for index in condition.frames:
            frame_ok(index, f"condicion {condition.condition}")
    for check in assessment.flow_manual_checks:
        what = f"contraste {check.check_id}"
        frame_ok(check.frame_to, what)
        point_ok(*check.uv_from_px, what)
        point_ok(*check.uv_to_px, what)

    if assessment.suitability is Suitability.A and not clip.eof_reached:
        problems.append("la decodificacion no llego a EOF: no se admite aptitud A")
    if problems:
        raise ConfigError(
            f"evaluacion manual de {assessment.video_id!r} incompatible con el clip "
            f"{clip.video_id!r}:\n- " + "\n- ".join(problems)
        )


@dataclass(frozen=True)
class CrossingEvent:
    """Cruce de la linea virtual por un objetivo anotado de una planta.

    `observed_target` dice que punto cruzo (contacto tallo-suelo o proxy de eje/copa).
    El instante es una interpolacion entre dos anotaciones, no un evento medido:
    `support_from_s`/`support_to_s` acotan el intervalo de soporte y
    `time_uncertainty_s` propaga linealmente la incertidumbre de lectura en `v`.
    """

    plant_id: str
    observed_target: str
    crossing_time_s: float | None
    frame_before: int | None
    frame_after: int | None
    support_from_s: float | None
    support_to_s: float | None
    time_uncertainty_s: float | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


CROSSING_COLUMNS = tuple(CrossingEvent.__dataclass_fields__)


def crossing_events(
    observations: list[StemBaseObservation], line: CrossingLine
) -> list[CrossingEvent]:
    """Instante de cruce de cada planta y objetivo por interpolacion lineal.

    Solo usa puntos localizados dentro de `[u_min_px, u_max_px]` y no mezcla
    objetivos distintos de una misma planta. El cruce se declara entre dos
    observaciones consecutivas que quedan a ambos lados de la linea en el sentido
    declarado; si no hay par que lo encierre, `not_bracketed`.
    """
    groups: dict[tuple[str, ObservedTarget], list[StemBaseObservation]] = {}
    for observation in observations:
        if (
            observation.observed_target is not ObservedTarget.NONE
            and line.u_min_px <= observation.u_px <= line.u_max_px  # type: ignore[operator]
        ):
            key = (observation.plant_id, observation.observed_target)
            groups.setdefault(key, []).append(observation)
    events = []
    for (plant, target), items in sorted(groups.items()):
        items.sort(key=lambda o: o.frame_index)
        event = CrossingEvent(
            plant, target.value, None, None, None, None, None, None, "not_bracketed"
        )
        for first, second in itertools.pairwise(items):
            a, b = first.v_px - line.v_px, second.v_px - line.v_px  # type: ignore[operator]
            crosses = (a > 0 >= b) if line.crossing_direction == "up" else (a < 0 <= b)
            if crosses:
                fraction = a / (a - b)
                span = second.timestamp_s - first.timestamp_s
                time = first.timestamp_s + fraction * span
                reading = max(first.uncertainty_px, second.uncertainty_px)  # type: ignore[type-var]
                event = CrossingEvent(
                    plant,
                    target.value,
                    time,
                    first.frame_index,
                    second.frame_index,
                    first.timestamp_s,
                    second.timestamp_s,
                    abs(span / (a - b)) * reading,
                    "crossed",
                )
                break
        events.append(event)
    return events
