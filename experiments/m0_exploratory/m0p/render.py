"""Dibujo por cuadro y escritura del video anotado (orden M0-P §4.9, MP-07).

Todo se dibuja proyectando geometria de `W_segment` al cuadro actual con su
`SegmentProjection` (nunca una ruta en pixeles fijos sobre camara movil). La
etiqueta exploratoria es permanente. El video de salida conserva resolucion,
orden y PTS de entrada (misma base temporal); si el muxer no puede conservarlos,
la verificacion posterior falla de forma explicita.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from precision_weeder.core.trajectory import TaskPath
from precision_weeder.mapping.local_map import LocalMap

from . import EXPLORATORY_LABEL_ASCII
from .projection import SegmentProjection
from .tracking import Track

__all__ = ["VideoWriter", "draw_overlay"]

STATUS_COLORS = {
    "vigente": (80, 220, 80),
    "sin_ruta": (0, 140, 255),
    "observaciones_insuficientes": (0, 200, 255),
    "proyeccion_perdida": (0, 0, 255),
    "segmento_iniciado": (200, 200, 200),
    "timestamp_invalido": (0, 0, 255),
}


def _to_px(projection: SegmentProjection, points_m: np.ndarray) -> np.ndarray:
    return np.round(projection.ground_to_image(points_m)).astype(np.int32)


def draw_overlay(
    image: np.ndarray,
    projection: SegmentProjection | None,
    status: str,
    reason: str,
    frame_index: int,
    corridor_box: tuple[float, float, float, float] | None,
    axis_x: float | None,
    local_map: LocalMap | None,
    tracks: list[Track],
    paths: tuple[TaskPath, ...] | None,
    tool_polygon: np.ndarray,
    footprint_samples: int,
    setpoints: list[dict] | None = None,
    tool_line_v_norm: float | None = None,
) -> np.ndarray:
    """Copia anotada del cuadro; la geometria de suelo se proyecta con `projection`."""
    canvas = image.copy()
    height, width = canvas.shape[:2]
    thick = max(1, round(width / 640))
    if projection is not None and corridor_box is not None:
        x0, x1, y0, y1 = corridor_box
        ring = _to_px(projection, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
        cv2.polylines(canvas, [ring], True, (255, 200, 0), thick)
        if axis_x is not None:  # linea de plantacion estimada
            axis = _to_px(projection, np.array([[axis_x, y0], [axis_x, y1]]))
            cv2.polylines(canvas, [axis], False, (120, 220, 120), thick)
    if projection is not None and local_map is not None:
        for region in local_map.unknown_regions:
            cv2.polylines(canvas, [_to_px(projection, region)], True, (160, 160, 160), thick)
        for exclusion in local_map.exclusions:
            center = _to_px(projection, np.array([[exclusion.center.x, exclusion.center.y]]))[0]
            radius = round(exclusion.total_radius / projection.scale_m_per_px)
            cv2.circle(canvas, tuple(int(c) for c in center), radius, (0, 0, 230), thick)
    if projection is not None:
        for track in tracks:
            center = _to_px(projection, track.centroid[None, :])[0]
            color = (0, 230, 230) if track.missed == 0 else (0, 130, 130)
            cv2.putText(
                canvas,
                f"#{track.track_id}",
                (int(center[0]) + 4, int(center[1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45 * thick,
                color,
                thick,
            )
    for path in paths if projection is not None and paths else ():
        samples = np.array([path.evaluate(s) for s in np.linspace(0.0, 1.0, 160)])
        cv2.polylines(canvas, [_to_px(projection, samples)], False, (255, 255, 0), 2 * thick)
        for s in np.linspace(0.0, 1.0, footprint_samples) if footprint_samples else ():
            point = np.array(path.evaluate(s))
            cv2.polylines(
                canvas, [_to_px(projection, tool_polygon + point)], True, (255, 120, 0), thick
            )
    if setpoints is not None and projection is not None and tool_line_v_norm is not None:
        _draw_setpoint(canvas, projection, setpoints, tool_polygon, tool_line_v_norm, thick)
    color = STATUS_COLORS.get(status, (255, 255, 255))
    scale = 0.45 * thick
    if setpoints is not None:
        # Con la franja de consigna, el estado va pequeno encima de ella (abajo a la
        # derecha) para no tapar la linea de herramienta ni las plantas que entran.
        w, h = int(width * 0.30), int(height * 0.16)
        x0 = width - w - 8 * thick
        y1 = height - h - 10 * thick
        y0 = y1 - int(24 * thick)
        cv2.rectangle(canvas, (x0, y0), (x0 + w, y1), (25, 25, 25), -1)
        cv2.putText(
            canvas,
            f"frame {frame_index}  {status}",
            (x0 + 4, y0 + int(10 * thick)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32 * thick,
            color,
            1,
        )
        cv2.putText(
            canvas,
            reason[:60],
            (x0 + 4, y0 + int(20 * thick)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28 * thick,
            (220, 220, 220),
            1,
        )
    else:
        cv2.rectangle(canvas, (0, 0), (int(width * 0.40), int(46 * thick)), (25, 25, 25), -1)
        cv2.putText(
            canvas,
            f"frame {frame_index}  {status}",
            (6, int(18 * thick)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thick,
        )
        cv2.putText(
            canvas,
            reason[:48],
            (6, int(38 * thick)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            (220, 220, 220),
            1,
        )
    band = int(22 * thick)
    cv2.rectangle(canvas, (0, height - band), (int(width * 0.40), height), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        EXPLORATORY_LABEL_ASCII,
        (6, height - int(7 * thick)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38 * thick,
        (255, 255, 255),
        1,
    )
    return canvas


def _draw_setpoint(canvas, projection, setpoints, tool_polygon, v_norm, thick) -> None:
    """Herramienta en la linea de herramienta y franja con la consigna lateral reciente.

    La franja muestra `tool_u_px` (columna de la imagen) de los ultimos cuadros: es lo que
    se enviaria al brazo. Los cuadros sin consigna dejan un hueco; no se interpola."""
    height, width = canvas.shape[:2]
    y_tool = projection.row_v_to_ground_y(v_norm)
    line = _to_px(projection, np.array([[0.0, y_tool]]))[0][1]
    cv2.line(canvas, (0, int(line)), (width, int(line)), (0, 215, 255), thick)
    current = setpoints[-1] if setpoints else None
    if current is not None and current["tool_u_px"] is not None:
        x_tool = projection.image_to_ground(np.array([[current["tool_u_px"], float(line)]]))[0][0]
        ring = _to_px(projection, tool_polygon + np.array([x_tool, y_tool]))
        cv2.fillPoly(canvas, [ring], (0, 215, 255))
        cv2.polylines(canvas, [ring], True, (0, 0, 0), thick)
    # Franja: tiempo hacia la derecha, columna de la imagen en vertical (reescalada).
    window = 150
    w, h = int(width * 0.30), int(height * 0.16)
    x0, y0 = width - w - 8 * thick, height - h - 8 * thick
    cv2.rectangle(canvas, (x0, y0), (x0 + w, y0 + h), (25, 25, 25), -1)
    cv2.putText(
        canvas,
        "consigna lateral (linea de herramienta)",
        (x0 + 4, y0 + int(12 * thick)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35 * thick,
        (220, 220, 220),
        1,
    )
    recent = setpoints[-window:]
    values = [row["tool_u_px"] for row in recent if row["tool_u_px"] is not None]
    if not values:
        return
    low, high = min(values), max(values)
    span = max(high - low, 0.08 * width)
    middle = 0.5 * (low + high)
    top, bottom = y0 + int(16 * thick), y0 + h - 4
    previous = None
    for k, row in enumerate(recent):
        if row["tool_u_px"] is None:
            previous = None
            continue
        px = x0 + int((k + window - len(recent)) * (w - 1) / (window - 1))
        frac = 0.5 + (row["tool_u_px"] - middle) / span
        py = int(top + min(max(frac, 0.0), 1.0) * (bottom - top))
        if previous is not None:
            cv2.line(canvas, previous, (px, py), (0, 215, 255), thick)
        previous = (px, py)


class VideoWriter:
    """Escritor H.264 que conserva PTS y duraciones de presentacion de la entrada.

    Cada paquete recibe como duracion la distancia al PTS del cuadro siguiente; el
    ultimo, la `last_duration` indicada al cerrar (fin de presentacion de la fuente) o,
    si no se conoce, la del ultimo intervalo (politica declarada `ultimo_intervalo`).
    Asi el final efectivo del stream coincide con el de la entrada, tambien en VFR.
    """

    def __init__(self, path: Path, width: int, height: int, time_base: Fraction, crf: int) -> None:
        import av

        self.path = path
        self._container = av.open(str(path), "w")
        rate = Fraction(1, 1) / time_base if time_base else Fraction(30, 1)
        self._stream = self._container.add_stream("libx264", rate=rate)
        self._stream.width, self._stream.height = width, height
        self._stream.pix_fmt = "yuv420p"
        self._stream.time_base = time_base
        self._stream.codec_context.time_base = time_base
        self._stream.options = {"crf": str(crf), "preset": "veryfast", "bf": "0"}
        self.time_base = time_base
        self._last_pts: int | None = None
        self._durations: dict[int, int] = {}
        self._pending: list = []
        self.last_duration_policy: str | None = None

    def _mux(self, packets) -> None:
        self._pending.extend(packets)
        ready = [p for p in self._pending if p.pts in self._durations]
        self._pending = [p for p in self._pending if p.pts not in self._durations]
        for packet in ready:
            packet.duration = self._durations[packet.pts]
            self._container.mux(packet)

    def write(self, image_bgr: np.ndarray, pts: int) -> None:
        import av

        if self._last_pts is not None and pts <= self._last_pts:
            raise ValueError(f"PTS no creciente en la salida ({pts} <= {self._last_pts})")
        if self._last_pts is not None:
            self._durations[self._last_pts] = pts - self._last_pts
        frame = av.VideoFrame.from_ndarray(image_bgr, format="bgr24")
        frame.pts = pts
        frame.time_base = self.time_base
        self._last_pts = pts
        self._mux(self._stream.encode(frame))

    def abort(self) -> None:
        """Cierra el contenedor sin completar el video (procesamiento interrumpido)."""
        self._container.close()

    def close(self, last_duration: int | None = None) -> None:
        if self._last_pts is not None:
            if last_duration is not None and last_duration > 0:
                self.last_duration_policy = "fin_de_presentacion_de_la_fuente"
            else:
                previous = [d for d in self._durations.values()]
                last_duration = previous[-1] if previous else 1
                self.last_duration_policy = "ultimo_intervalo"
            self._durations[self._last_pts] = int(last_duration)
        self._mux(self._stream.encode())
        if self._pending:
            raise ValueError(f"{len(self._pending)} paquete(s) sin duracion conocida")
        self._container.close()
