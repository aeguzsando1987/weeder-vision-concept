"""Inspeccion de metadatos y decodificacion secuencial real de video (E0).

Backend: PyAV (bibliotecas FFmpeg empaquetadas). Se usa porque expone PTS,
`time_base` y el registro de FFmpeg por cuadro; el FPS nominal del contenedor no
prueba frecuencia constante (`docs/orders/E0_technical_order.md` §2).

Reglas:

- Un campo que el contenedor/decodificador no expone es `None` (desconocido); no
  se infiere de la extension ni se sustituye por cero. PyAV informa `frames=0`
  cuando el contenedor no declara el numero de cuadros: se traduce a `None`.
- El recorrido llega hasta EOF. `eof_status` distingue fin normal de error de
  decodificacion; FFmpeg no siempre distingue un archivo truncado de uno completo,
  por lo que el truncamiento solo se declara cuando hay evidencia (error o
  advertencia del demuxer/decoder); en otro caso queda como limitacion declarada.
- Nunca se interpreta una diferencia entre cuadros declarados y decodificados como
  "cuadros perdidos durante la captura": no hay referencia externa para afirmarlo.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import PrecisionWeederError

__all__ = [
    "DecodeReport",
    "DecodedFrame",
    "VideoDecodeError",
    "VideoProbe",
    "backend_versions",
    "decode_sequential",
    "iter_frames",
    "probe_video",
    "sha256_file",
]


class VideoDecodeError(PrecisionWeederError):
    """El archivo no existe, no se puede abrir o no contiene un flujo de video."""


def _require_av() -> Any:
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depende del entorno
        raise VideoDecodeError(
            "falta la dependencia de video: instale el extra `video` "
            "(uv pip install -e '.[video]'), que aporta PyAV y OpenCV headless"
        ) from exc
    return av


def backend_versions() -> dict[str, str]:
    """Versiones de PyAV, FFmpeg (libavcodec/libavformat) y OpenCV, si existen."""
    av = _require_av()
    versions = {"pyav": str(av.__version__)}
    for name, value in sorted(av.library_versions.items()):
        versions[name] = ".".join(str(part) for part in value)
    try:
        import cv2

        versions["opencv"] = str(cv2.__version__)
    except ImportError:  # pragma: no cover
        versions["opencv"] = "unavailable"
    return versions


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 del archivo, leido en bloques."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class VideoProbe:
    """Metadatos declarados por contenedor y flujo; `None` = no expuesto."""

    path: str
    size_bytes: int
    container: str
    container_long_name: str | None
    codec: str | None
    codec_tag: str | None
    profile: str | None
    pixel_format: str | None
    width_px: int | None
    height_px: int | None
    rotation_deg: float | None
    duration_reported_s: float | None
    frame_count_reported: int | None
    fps_nominal: float | None
    fps_nominal_source: str | None
    time_base: str | None
    start_time_s: float | None
    bit_rate: int | None
    other_streams: tuple[str, ...]
    container_metadata: dict[str, str]
    stream_metadata: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.__dict__)
        payload["other_streams"] = list(self.other_streams)
        return payload


@dataclass(frozen=True)
class DecodedFrame:
    """Cuadro decodificado con su procedencia temporal."""

    index: int
    pts: int | None
    dts: int | None
    time_base: Fraction | None
    timestamp_s: float | None
    key_frame: bool
    pict_type: str
    width: int
    height: int
    rotation_deg: float | None
    image: np.ndarray | None = field(default=None, repr=False)


@dataclass
class DecodeReport:
    """Resultado del recorrido secuencial completo."""

    frames_decoded: int = 0
    eof_status: str = "not_started"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    observed_sizes: set[tuple[int, int]] = field(default_factory=set)
    frames_without_pts: int = 0
    limitation: str = (
        "FFmpeg/PyAV no garantiza distinguir un archivo truncado de uno completo; "
        "el truncamiento solo se declara ante errores o advertencias explicitas."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_decoded": self.frames_decoded,
            "eof_status": self.eof_status,
            "error": self.error,
            "warning_count": len(self.warnings),
            "warnings": self.warnings[:200],
            "observed_sizes": sorted([list(size) for size in self.observed_sizes]),
            "frames_without_pts": self.frames_without_pts,
            "truncation_suspected": self.truncation_suspected,
            "limitation": self.limitation,
        }

    @property
    def truncation_suspected(self) -> bool:
        markers = ("truncat", "partial file", "moov atom not found", "invalid data", "eof")
        text = " ".join([self.error or "", *self.warnings]).lower()
        return self.eof_status == "decode_error" or any(marker in text for marker in markers)


def _open(path: str | Path) -> tuple[Any, Any]:
    av = _require_av()
    file_path = Path(path)
    if not file_path.is_file():
        raise VideoDecodeError(f"video inexistente: {file_path}")
    try:
        container = av.open(str(file_path))
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoDecodeError(f"no se pudo abrir {file_path}: {exc}") from exc
    if not container.streams.video:
        container.close()
        raise VideoDecodeError(f"{file_path}: sin flujo de video")
    return container, container.streams.video[0]


def _positive_or_none(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


_PICT_TYPES = {0: "unknown", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def _pict_type_name(value: Any) -> str:
    """Nombre del tipo de imagen (AVPictureType); PyAV puede exponer un entero."""
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    try:
        return _PICT_TYPES.get(int(value), str(value))
    except (TypeError, ValueError):
        return str(value)


def probe_video(path: str | Path) -> VideoProbe:
    """Metadatos declarados, sin decodificar todo el clip.

    Raises:
        VideoDecodeError: si el archivo no existe o no se puede abrir.
    """
    container, stream = _open(path)
    try:
        codec_context = stream.codec_context
        time_base = stream.time_base
        rate, rate_source = None, None
        for source in ("average_rate", "base_rate", "guessed_rate"):
            candidate = _positive_or_none(getattr(stream, source, None))
            if candidate is not None:
                rate, rate_source = candidate, f"stream.{source}"
                break
        duration = None
        if stream.duration is not None and time_base is not None:
            duration = _positive_or_none(stream.duration * time_base)
        if duration is None and container.duration is not None:
            duration = _positive_or_none(container.duration / 1_000_000)
        start = None
        if stream.start_time is not None and time_base is not None:
            start = float(stream.start_time * time_base)
        others = tuple(
            f"{other.index}:{other.type}:{other.codec_context.name}"
            for other in container.streams
            if other.index != stream.index
        )
        return VideoProbe(
            path=str(path),
            size_bytes=Path(path).stat().st_size,
            container=container.format.name,
            container_long_name=container.format.long_name or None,
            codec=codec_context.name or None,
            codec_tag=getattr(stream, "codec_tag", None) or None,
            profile=getattr(stream, "profile", None) or None,
            pixel_format=codec_context.pix_fmt or None,
            width_px=codec_context.width or None,
            height_px=codec_context.height or None,
            rotation_deg=None,  # el valor declarado se obtiene del primer cuadro decodificado
            duration_reported_s=duration,
            frame_count_reported=stream.frames or None,
            fps_nominal=rate,
            fps_nominal_source=rate_source,
            time_base=str(time_base) if time_base is not None else None,
            start_time_s=start,
            bit_rate=container.bit_rate or None,
            other_streams=others,
            container_metadata={str(k): str(v) for k, v in container.metadata.items()},
            stream_metadata={str(k): str(v) for k, v in stream.metadata.items()},
        )
    finally:
        container.close()


def iter_frames(
    path: str | Path,
    image_format: str | None = None,
    report: DecodeReport | None = None,
) -> Iterator[DecodedFrame]:
    """Recorre el clip secuencialmente hasta EOF.

    Args:
        path: Archivo de video.
        image_format: Formato del array (`"bgr24"`, `"gray"`) o `None` para no
            convertir pixeles (solo temporizacion).
        report: Si se aporta, acumula cuadros, advertencias de FFmpeg y estado de EOF.
    """
    av = _require_av()
    report = report if report is not None else DecodeReport()
    container, stream = _open(path)
    report.eof_status = "in_progress"
    index = 0
    # PyAV silencia el registro de FFmpeg por defecto: se activa a WARNING durante el
    # recorrido y se captura desde todos los hilos del decodificador.
    previous_level = av.logging.get_level()
    av.logging.set_level(av.logging.WARNING)
    try:
        with av.logging.Capture(False) as logs:
            try:
                for frame in container.decode(stream):
                    time_base = frame.time_base
                    timestamp = (
                        float(frame.pts * time_base)
                        if frame.pts is not None and time_base is not None
                        else None
                    )
                    if frame.pts is None:
                        report.frames_without_pts += 1
                    report.observed_sizes.add((frame.width, frame.height))
                    rotation = getattr(frame, "rotation", None)
                    image = frame.to_ndarray(format=image_format) if image_format else None
                    report.frames_decoded += 1
                    yield DecodedFrame(
                        index=index,
                        pts=frame.pts,
                        dts=frame.dts,
                        time_base=time_base,
                        timestamp_s=timestamp,
                        key_frame=bool(frame.key_frame),
                        pict_type=_pict_type_name(frame.pict_type),
                        width=frame.width,
                        height=frame.height,
                        rotation_deg=float(rotation) if rotation is not None else None,
                        image=image,
                    )
                    index += 1
                report.eof_status = "eof_reached"
            except av.error.FFmpegError as exc:
                report.eof_status = "decode_error"
                report.error = f"{type(exc).__name__}: {exc}"
            finally:
                for level, name, message in logs:
                    if level <= av.logging.WARNING:
                        report.warnings.append(f"{name}: {str(message).strip()}")
    finally:
        av.logging.set_level(previous_level)
        container.close()


def decode_sequential(
    path: str | Path,
    on_frame: Callable[[DecodedFrame], None] | None = None,
    image_format: str | None = None,
) -> tuple[list[DecodedFrame], DecodeReport]:
    """Recorrido completo; devuelve los cuadros sin pixeles y el reporte.

    `on_frame` recibe cada cuadro (con pixeles si `image_format`) para calcular
    metricas sin retener el clip completo en memoria.
    """
    report = DecodeReport()
    frames: list[DecodedFrame] = []
    for frame in iter_frames(path, image_format=image_format, report=report):
        if on_frame is not None:
            on_frame(frame)
        frames.append(
            DecodedFrame(
                index=frame.index,
                pts=frame.pts,
                dts=frame.dts,
                time_base=frame.time_base,
                timestamp_s=frame.timestamp_s,
                key_frame=frame.key_frame,
                pict_type=frame.pict_type,
                width=frame.width,
                height=frame.height,
                rotation_deg=frame.rotation_deg,
            )
        )
    return frames, report
