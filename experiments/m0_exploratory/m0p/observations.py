"""Contrato de observaciones y proveedor de replay (orden M0-P §2-§3, MP-02).

Una observacion es una mascara poligonal en pixeles del cuadro decodificado
(`image_px`), con objetivo observado, clase, confianza y procedencia. Los pesos de
moringa producen **copa** (`canopy_proxy`): nunca se convierten en
`ground_contact`. La confianza de la red **no** es una incertidumbre de posicion.

La exportacion versionada (JSON) la escribe cualquier proveedor (aprendido en su
entorno separado, sintetico en pruebas) y la consume siempre el mismo adaptador
(`load_export` / `ObservationSet.from_payload`). El planificador no ve el proveedor.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
from shapely.geometry import Polygon

from precision_weeder.core.errors import SchemaValidationError
from precision_weeder.core.identity import content_hash, is_sha256

__all__ = [
    "EXPORT_SCHEMA_VERSION",
    "OBSERVED_TARGETS",
    "Observation",
    "ObservationSet",
    "load_export",
]

EXPORT_SCHEMA_VERSION: Final[int] = 1
OBSERVED_TARGETS: Final[frozenset[str]] = frozenset(
    {"canopy_proxy", "stem_axis_proxy", "ground_contact"}
)


@dataclass(frozen=True, eq=False)
class Observation:
    """Mascara de una planta en un cuadro, en pixeles (`image_px`)."""

    frame_index: int
    target: str
    class_name: str
    confidence: float
    polygon_px: np.ndarray

    def __post_init__(self) -> None:
        if self.target not in OBSERVED_TARGETS:
            raise SchemaValidationError(f"Observation.target desconocido: {self.target!r}")
        if not (math.isfinite(self.confidence) and 0.0 <= self.confidence <= 1.0):
            raise SchemaValidationError("Observation.confidence fuera de [0, 1]")
        polygon = np.asarray(self.polygon_px, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or polygon.shape[0] < 3:
            raise SchemaValidationError("Observation.polygon_px: se exigen >= 3 puntos (u, v)")
        if not np.all(np.isfinite(polygon)):
            raise SchemaValidationError("Observation.polygon_px: valores no finitos")
        frozen = polygon.copy()
        frozen.setflags(write=False)
        object.__setattr__(self, "polygon_px", frozen)

    def shape(self) -> Polygon:
        """Poligono valido (se repara una autointerseccion con `buffer(0)`)."""
        shape = Polygon(self.polygon_px)
        return shape if shape.is_valid else shape.buffer(0)


@dataclass(frozen=True)
class ObservationSet:
    """Observaciones de un video, indexadas por cuadro, con su procedencia."""

    provider: dict[str, Any]
    video_sha256: str
    frame_count: int
    image_size: tuple[int, int]
    by_frame: dict[int, tuple[Observation, ...]] = field(default_factory=dict)
    discarded: int = 0
    export_hash: str = ""

    def for_frame(self, index: int, min_confidence: float) -> tuple[Observation, ...]:
        return tuple(o for o in self.by_frame[index] if o.confidence >= min_confidence)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ObservationSet:
        """Adaptador unico de exportaciones (aprendidas, sinteticas o replay).

        Raises:
            SchemaValidationError: version, hash, objetivos o geometria invalidos.
        """
        try:
            if payload["schema_version"] != EXPORT_SCHEMA_VERSION:
                raise SchemaValidationError("exportacion: schema_version no soportada")
            declared = payload["content_hash"]
            body = {k: v for k, v in payload.items() if k != "content_hash"}
            if declared != content_hash(body):
                raise SchemaValidationError("exportacion: content_hash no coincide")
            if payload["coordinate_domain"] != "image_px":
                raise SchemaValidationError("exportacion: las mascaras deben estar en image_px")
            video = payload["video"]
            if not is_sha256(video["sha256"]):
                raise SchemaValidationError("exportacion: video.sha256 invalido")
            frame_count = video["frame_count"]
            if type(frame_count) is not int or frame_count <= 0:
                raise SchemaValidationError("exportacion: video.frame_count debe ser entero > 0")
            frames: dict[int, tuple[Observation, ...]] = {}
            discarded = 0
            for frame in payload["frames"]:
                index = frame["frame_index"]
                # Un hash valido no garantiza validez semantica: cada cuadro procesado
                # tiene un registro propio (lista vacia si no hubo detecciones).
                if type(index) is not int:
                    raise SchemaValidationError(f"exportacion: frame_index no entero: {index!r}")
                if not 0 <= index < frame_count:
                    raise SchemaValidationError(
                        f"exportacion: frame_index {index} fuera de 0..{frame_count - 1}"
                    )
                if index in frames:
                    raise SchemaValidationError(f"exportacion: frame_index {index} repetido")
                kept = []
                for det in frame["detections"]:
                    if len(det["polygon"]) < 3:
                        discarded += 1
                        continue
                    observation = Observation(
                        frame_index=index,
                        target=payload["target"],
                        class_name=det["class"],
                        confidence=float(det["confidence"]),
                        polygon_px=np.asarray(det["polygon"], dtype=np.float64),
                    )
                    if observation.shape().area <= 0.0:
                        discarded += 1
                        continue
                    kept.append(observation)
                frames[index] = tuple(kept)
            if len(frames) != frame_count:
                missing = sorted(set(range(frame_count)) - set(frames))
                raise SchemaValidationError(
                    f"exportacion: faltan registros de {len(missing)} cuadro(s), p. ej. "
                    f"{missing[:5]}; un cuadro no observado no equivale a cero detecciones"
                )
            return cls(
                provider=dict(payload["provider"]),
                video_sha256=video["sha256"],
                frame_count=frame_count,
                image_size=(int(video["width"]), int(video["height"])),
                by_frame=frames,
                discarded=discarded,
                export_hash=declared,
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, SchemaValidationError):
                raise
            raise SchemaValidationError(f"exportacion de observaciones invalida: {exc}") from exc


def build_export(
    provider: dict[str, Any],
    video_sha256: str,
    frame_count: int,
    image_size: tuple[int, int],
    frames: list[dict[str, Any]],
    target: str = "canopy_proxy",
) -> dict[str, Any]:
    """Construye una exportacion versionada con identidad de contenido."""
    body = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "coordinate_domain": "image_px",
        "target": target,
        "provider": provider,
        "video": {
            "sha256": video_sha256,
            "frame_count": frame_count,
            "width": image_size[0],
            "height": image_size[1],
        },
        "frames": frames,
    }
    return {**body, "content_hash": content_hash(body)}


def load_export(path: str | Path) -> ObservationSet:
    return ObservationSet.from_payload(json.loads(Path(path).read_text(encoding="utf-8")))
