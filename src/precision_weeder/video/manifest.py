"""Manifiesto de videos: esquema tipado, validado y reproducible (E0, orden §6).

Cada fila describe un clip. Convenciones:

- `unknown` es el unico token de valor desconocido, para cualquier tipo. Nunca se
  usa `0` para "desconocido": un recuento o una duracion desconocidos son `None`
  en Python y `unknown` en el CSV.
- `not_applicable` solo se admite donde el esquema lo declara.
- Las referencias (`*_ref`) son rutas **relativas** al directorio base del
  manifiesto (la raiz del repositorio) y deben resolverse al validar con
  `base_dir`. Varias referencias se separan con `;`.
- A/B/C es aptitud visual/experimental, no categoria de calibracion. `pending`
  indica que la evaluacion manual no se ha completado: una fila nunca es `A` por
  defecto.
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Final

from ..core.errors import SchemaValidationError

__all__ = [
    "MANIFEST_COLUMNS",
    "UNKNOWN",
    "ApparentMotion",
    "HomographyFeasibility",
    "MountingStatus",
    "ObservationTiming",
    "ScaleStatus",
    "StemBaseVisibility",
    "Suitability",
    "TimestampSource",
    "VideoManifestRecord",
    "read_manifest_csv",
    "write_manifest_csv",
]

UNKNOWN: Final[str] = "unknown"
NOT_APPLICABLE: Final[str] = "not_applicable"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class TimestampSource(StrEnum):
    PTS = "pts"
    INDEX_FPS_ESTIMATE = "index_fps_estimate"
    UNKNOWN = UNKNOWN


class StemBaseVisibility(StrEnum):
    """Visibilidad de las bases de tallo en contacto con el suelo, en el clip."""

    VISIBLE = "visible"
    MOSTLY_VISIBLE = "mostly_visible"
    PARTIAL = "partial"
    MOSTLY_OCCLUDED = "mostly_occluded"
    NOT_VISIBLE = "not_visible"
    UNKNOWN = UNKNOWN


class ApparentMotion(StrEnum):
    """Direccion aparente del suelo en imagen (`u` derecha, `v` abajo)."""

    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    MIXED = "mixed"
    NONE = "none"
    UNKNOWN = UNKNOWN


class MountingStatus(StrEnum):
    KNOWN = "known"
    ASSUMED = "assumed"
    UNKNOWN = UNKNOWN


class ScaleStatus(StrEnum):
    VERIFIED = "verified"
    ASSUMED = "assumed"
    UNVERIFIED = "unverified"
    UNKNOWN = UNKNOWN


class HomographyFeasibility(StrEnum):
    VALIDATED = "validated"
    POSSIBLE_WITH_ADDITIONAL_DATA = "possible_with_additional_data"
    ASSUMED_ESTIMATE_ONLY = "assumed_estimate_only"
    NOT_FEASIBLE = "not_feasible"
    UNKNOWN = UNKNOWN


class ObservationTiming(StrEnum):
    """Relacion temporal entre la observacion de una planta y la intervencion fisica."""

    BEFORE_INTERVENTION = "before_intervention"
    AFTER_INTERVENTION = "after_intervention"
    NO_INTERVENTION_IN_CLIP = "no_intervention_in_clip"
    UNKNOWN = UNKNOWN


class Suitability(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    PENDING = "pending"


MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "video_id",
    "relative_path",
    "sha256",
    "size_bytes",
    "session_id",
    "site_id",
    "container",
    "codec",
    "width_px",
    "height_px",
    "rotation_deg",
    "duration_reported_s",
    "duration_observed_s",
    "frame_count_reported",
    "frame_count_decoded",
    "fps_nominal",
    "fps_effective",
    "timestamp_source",
    "time_base",
    "timestamp_anomalies_ref",
    "stem_base_visibility",
    "apparent_motion",
    "mounting_status",
    "scale_status",
    "scale_evidence_ref",
    "homography_feasibility",
    "observation_timing",
    "suitability",
    "suitability_reason",
    "evidence_ref",
    "decoder_version",
    "config_hash",
)

_INT_FIELDS = ("size_bytes", "width_px", "height_px", "frame_count_reported", "frame_count_decoded")
_FLOAT_FIELDS = (
    "rotation_deg",
    "duration_reported_s",
    "duration_observed_s",
    "fps_nominal",
    "fps_effective",
)
_REF_FIELDS = ("timestamp_anomalies_ref", "scale_evidence_ref", "evidence_ref")
_ENUM_FIELDS: Final[dict[str, type[StrEnum]]] = {
    "timestamp_source": TimestampSource,
    "stem_base_visibility": StemBaseVisibility,
    "apparent_motion": ApparentMotion,
    "mounting_status": MountingStatus,
    "scale_status": ScaleStatus,
    "homography_feasibility": HomographyFeasibility,
    "observation_timing": ObservationTiming,
    "suitability": Suitability,
}


@dataclass(frozen=True)
class VideoManifestRecord:
    """Fila del manifiesto. `None` en Python equivale a `unknown` en CSV."""

    video_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    session_id: str | None
    site_id: str | None
    container: str | None
    codec: str | None
    width_px: int | None
    height_px: int | None
    rotation_deg: float | None
    duration_reported_s: float | None
    duration_observed_s: float | None
    frame_count_reported: int | None
    frame_count_decoded: int | None
    fps_nominal: float | None
    fps_effective: float | None
    timestamp_source: TimestampSource
    time_base: str | None
    timestamp_anomalies_ref: str | None
    stem_base_visibility: StemBaseVisibility
    apparent_motion: ApparentMotion
    mounting_status: MountingStatus
    scale_status: ScaleStatus
    scale_evidence_ref: str | None
    homography_feasibility: HomographyFeasibility
    observation_timing: ObservationTiming
    suitability: Suitability
    suitability_reason: str
    evidence_ref: str | None
    decoder_version: str | None
    config_hash: str | None

    def __post_init__(self) -> None:
        context = f"VideoManifestRecord({self.video_id!r})"
        if not _IDENTIFIER.match(str(self.video_id)):
            raise SchemaValidationError(f"{context}: video_id invalido")
        if not _SHA256.match(str(self.sha256)):
            raise SchemaValidationError(f"{context}: sha256 debe ser 64 hex en minuscula")
        if self.config_hash is not None and not _SHA256.match(str(self.config_hash)):
            raise SchemaValidationError(f"{context}: config_hash debe ser 64 hex en minuscula")
        _check_relative(self.relative_path, f"{context}.relative_path")
        for name in _REF_FIELDS:
            value = getattr(self, name)
            if value is not None:
                for part in value.split(";"):
                    _check_relative(part, f"{context}.{name}")
        for name in _INT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                raise SchemaValidationError(f"{context}.{name}: se espera entero")
            minimum = 0 if name == "frame_count_decoded" else 1
            if value < minimum:
                raise SchemaValidationError(
                    f"{context}.{name}: {value} < {minimum}; desconocido se expresa como unknown"
                )
        if self.size_bytes is None:
            raise SchemaValidationError(f"{context}.size_bytes: obligatorio")
        for name in _FLOAT_FIELDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(float(value)):
                raise SchemaValidationError(f"{context}.{name}: se espera un real finito")
            object.__setattr__(self, name, float(value))
            if name != "rotation_deg" and float(value) <= 0.0:
                raise SchemaValidationError(
                    f"{context}.{name}: debe ser > 0; desconocido se expresa como unknown"
                )
        if self.rotation_deg is not None and self.rotation_deg % 90.0 != 0.0:
            raise SchemaValidationError(f"{context}.rotation_deg: multiplo de 90 esperado")
        for name, enum in _ENUM_FIELDS.items():
            object.__setattr__(self, name, enum(getattr(self, name)))
        for name in ("session_id", "site_id"):
            value = getattr(self, name)
            if value is not None and not _IDENTIFIER.match(value):
                raise SchemaValidationError(f"{context}.{name}: identificador invalido")
        if self.scale_status is ScaleStatus.VERIFIED and self.scale_evidence_ref is None:
            raise SchemaValidationError(f"{context}: scale_status verified sin evidencia de escala")
        self._check_suitability(context)

    def _check_suitability(self, context: str) -> None:
        reason = (self.suitability_reason or "").strip()
        object.__setattr__(self, "suitability_reason", reason)
        if self.suitability is Suitability.PENDING:
            return
        if not reason:
            raise SchemaValidationError(f"{context}: suitability {self.suitability} sin motivo")
        if self.evidence_ref is None:
            raise SchemaValidationError(f"{context}: suitability {self.suitability} sin evidencia")
        if self.suitability in (Suitability.A, Suitability.B) and not self.frame_count_decoded:
            raise SchemaValidationError(
                f"{context}: suitability {self.suitability} sin cuadros decodificados"
            )
        if self.suitability is Suitability.A and self.stem_base_visibility in (
            StemBaseVisibility.NOT_VISIBLE,
            StemBaseVisibility.UNKNOWN,
        ):
            raise SchemaValidationError(f"{context}: suitability A exige bases de tallo observadas")

    # --- serializacion ------------------------------------------------------

    def to_row(self) -> dict[str, str]:
        row: dict[str, str] = {}
        for name in MANIFEST_COLUMNS:
            value = getattr(self, name)
            if value is None:
                row[name] = UNKNOWN
            elif isinstance(value, StrEnum):
                row[name] = value.value
            elif isinstance(value, float):
                row[name] = repr(value)
            else:
                row[name] = str(value)
        return row

    @classmethod
    def from_row(cls, row: dict[str, str]) -> VideoManifestRecord:
        missing = [name for name in MANIFEST_COLUMNS if name not in row]
        extra = [name for name in row if name not in MANIFEST_COLUMNS]
        if missing or extra:
            raise SchemaValidationError(
                f"columnas del manifiesto invalidas (faltan {missing}, sobran {extra})"
            )
        values: dict[str, Any] = {}
        for name in MANIFEST_COLUMNS:
            raw = (row[name] or "").strip()
            if name in _ENUM_FIELDS or name == "suitability_reason":
                values[name] = raw
                continue
            if raw == "":
                raise SchemaValidationError(
                    f"{name}: celda vacia; use '{UNKNOWN}' para valores desconocidos"
                )
            if raw == UNKNOWN:
                values[name] = None
            elif name in _INT_FIELDS:
                try:
                    values[name] = int(raw)
                except ValueError as exc:
                    raise SchemaValidationError(f"{name}: entero invalido {raw!r}") from exc
            elif name in _FLOAT_FIELDS:
                try:
                    values[name] = float(raw)
                except ValueError as exc:
                    raise SchemaValidationError(f"{name}: real invalido {raw!r}") from exc
            else:
                values[name] = raw
        try:
            return cls(**values)
        except ValueError as exc:
            if isinstance(exc, SchemaValidationError):
                raise
            raise SchemaValidationError(str(exc)) from exc

    def check_references(self, base_dir: str | Path, verify_hash: bool = False) -> None:
        """Comprueba que todas las rutas se resuelven bajo `base_dir`.

        Raises:
            SchemaValidationError: si alguna referencia no existe o el hash difiere.
        """
        base = Path(base_dir)
        targets = [self.relative_path]
        for name in _REF_FIELDS:
            value = getattr(self, name)
            if value is not None:
                targets.extend(value.split(";"))
        for target in targets:
            if not (base / target).exists():
                raise SchemaValidationError(f"{self.video_id}: referencia no resuelta {target}")
        if verify_hash:
            from .reader import sha256_file

            actual = sha256_file(base / self.relative_path)
            if actual != self.sha256:
                raise SchemaValidationError(
                    f"{self.video_id}: sha256 no coincide ({actual} != {self.sha256})"
                )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return {
            key: (value.value if isinstance(value, StrEnum) else value)
            for key, value in payload.items()
        }


def _check_relative(path: str, context: str) -> None:
    text = str(path).strip()
    if not text or text == UNKNOWN:
        raise SchemaValidationError(f"{context}: ruta vacia")
    pure = PurePosixPath(text)
    if pure.is_absolute() or "\\" in text or ".." in pure.parts:
        raise SchemaValidationError(f"{context}: se exige ruta relativa sin '..' ({text})")


def write_manifest_csv(records: list[VideoManifestRecord], path: str | Path) -> str:
    """Escribe el manifiesto ordenado por `video_id`; devuelve el texto escrito."""
    ids = [record.video_id for record in records]
    if len(set(ids)) != len(ids):
        raise SchemaValidationError("video_id duplicado en el manifiesto")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for record in sorted(records, key=lambda item: item.video_id):
        writer.writerow(record.to_row())
    text = buffer.getvalue()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return text


def read_manifest_csv(
    path: str | Path, base_dir: str | Path | None = None, verify_hash: bool = False
) -> list[VideoManifestRecord]:
    """Lee y valida el manifiesto; con `base_dir` tambien resuelve referencias."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise SchemaValidationError(
                f"cabecera del manifiesto distinta de la esperada: {reader.fieldnames}"
            )
        records = [VideoManifestRecord.from_row(row) for row in reader]
    ids = [record.video_id for record in records]
    if len(set(ids)) != len(ids):
        raise SchemaValidationError("video_id duplicado en el manifiesto")
    if base_dir is not None:
        for record in records:
            record.check_references(base_dir, verify_hash=verify_hash)
    return records
