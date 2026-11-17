"""Fixtures sinteticos de M0-P: video con suelo texturado que se traslada de forma
conocida, plantas (discos verdes) fijas al suelo, indice de cuadro codificado y la
exportacion de observaciones exacta (proveedor sintetico, mismo contrato que el
aprendido). Sirve para oraculos sin GPU ni pesos (orden §5, MP-01..MP-09).

Convencion: el suelo se desplaza en la imagen `shift_px = (du, dv)` por cuadro; un
punto de suelo visto en `q` en el cuadro 0 se ve en `q + t*shift` en el cuadro `t`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, box

from precision_weeder.video.reader import sha256_file

from .observations import build_export

__all__ = ["CODE_BITS", "FixtureSpec", "PlantSpec", "decode_frame_code", "write_fixture"]

CODE_BITS = 6


@dataclass(frozen=True)
class PlantSpec:
    """Planta fija al suelo: posicion en el cuadro 0 (px) y radio de copa (px)."""

    u0: float
    v0: float
    radius_px: float
    plant_id: str = ""


@dataclass(frozen=True)
class FixtureSpec:
    width: int
    height: int
    frames: int
    shift_px: tuple[int, int]
    plants: tuple[PlantSpec, ...]
    time_base: Fraction = Fraction(1, 15)
    pts: tuple[int, ...] | None = None
    """PTS por cuadro en `time_base`; `None` = CFR (0, 1, 2, ...)."""
    seed: int = 7
    duplicate_masks: tuple[str, ...] = ()
    """Plantas cuya mascara se emite dos veces (desplazada) para probar la fusion."""
    dropouts: dict[str, tuple[int, ...]] = field(default_factory=dict)
    """Cuadros en que el proveedor omite una planta."""
    false_positives: tuple[tuple[int, float, float, float], ...] = ()
    """(cuadro, u, v, radio) de detecciones espurias."""

    def frame_pts(self) -> tuple[int, ...]:
        return self.pts if self.pts is not None else tuple(range(self.frames))


def _code_box(width: int) -> tuple[int, int, int]:
    block = max(8, width // 48)
    left = width // 2 - (CODE_BITS * block) // 2
    return left, 0, block


def decode_frame_code(image_bgr: np.ndarray) -> int:
    """Lee el indice de cuadro codificado en la banda superior central."""
    gray = image_bgr.mean(axis=2) if image_bgr.ndim == 3 else image_bgr
    left, top, block = _code_box(gray.shape[1])
    value = 0
    for bit in range(CODE_BITS):
        u0, u1 = left + bit * block + 2, left + (bit + 1) * block - 2
        cell = gray[top + 2 : top + block - 2, u0:u1]
        if cell.mean() > 127:
            value |= 1 << bit
    return value


def _ground_canvas(spec: FixtureSpec, origin: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    height, width = size
    noise = rng.uniform(0, 255, size=(height, width)).astype(np.float32)
    texture = cv2.GaussianBlur(noise, (0, 0), 2.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    canvas = np.empty((height, width, 3), np.uint8)
    canvas[..., 0] = (40 + 70 * texture).astype(np.uint8)  # B
    canvas[..., 1] = (55 + 80 * texture).astype(np.uint8)  # G
    canvas[..., 2] = (75 + 95 * texture).astype(np.uint8)  # R: suelo pardo, ExG bajo
    for plant in spec.plants:
        center = (round(plant.u0 + origin[0]), round(plant.v0 + origin[1]))
        cv2.circle(canvas, center, round(plant.radius_px), (40, 170, 50), -1)
        cv2.circle(canvas, center, max(1, round(plant.radius_px * 0.4)), (30, 120, 35), -1)
    return canvas


def write_fixture(spec: FixtureSpec, video_path: Path) -> dict:
    """Escribe el video y devuelve la exportacion de observaciones (sin escribirla)."""
    import av

    shift = np.array(spec.shift_px, dtype=float)
    travel = np.abs(shift) * spec.frames
    margin = np.array([40.0, 40.0])
    origin = travel + margin  # posicion del cuadro 0 dentro del lienzo
    size = (
        int(spec.height + 2 * (travel[1] + margin[1])),
        int(spec.width + 2 * (travel[0] + margin[0])),
    )
    canvas = _ground_canvas(spec, origin, size)
    pts = spec.frame_pts()
    frames_payload = []
    image_box = box(0, 0, spec.width, spec.height)
    with av.open(str(video_path), "w") as container:
        stream = container.add_stream("libx264", rate=Fraction(1, 1) / spec.time_base)
        stream.width, stream.height = spec.width, spec.height
        stream.pix_fmt = "yuv420p"
        stream.time_base = spec.time_base
        stream.codec_context.time_base = spec.time_base
        stream.options = {"crf": "16", "preset": "veryfast", "bf": "0"}
        for index in range(spec.frames):
            offset = index * shift
            # Cuadro `t`: el punto de suelo q (cuadro 0) aparece en q + t*shift.
            top_left = origin - offset
            u0, v0 = round(top_left[0]), round(top_left[1])
            image = canvas[v0 : v0 + spec.height, u0 : u0 + spec.width].copy()
            left, top, block = _code_box(spec.width)
            for bit in range(CODE_BITS):
                value = 255 if (index >> bit) & 1 else 0
                image[top : top + block, left + bit * block : left + (bit + 1) * block] = value
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts = pts[index]
            frame.time_base = spec.time_base
            for packet in stream.encode(frame):
                container.mux(packet)
            frames_payload.append(
                {"frame_index": index, "detections": _detections(spec, index, offset, image_box)}
            )
        for packet in stream.encode():
            container.mux(packet)
    return build_export(
        provider={
            "kind": "synthetic",
            "name": "m0p.synthetic (mascaras exactas del fixture)",
            "weights_sha256": None,
        },
        video_sha256=sha256_file(video_path),
        frame_count=spec.frames,
        image_size=(spec.width, spec.height),
        frames=frames_payload,
    )


def _circle(u: float, v: float, r: float, image_box) -> list[list[float]] | None:
    shape = Point(u, v).buffer(r, 32).intersection(image_box)
    if shape.is_empty or shape.area < 20.0 or shape.geom_type != "Polygon":
        return None
    return [[float(x), float(y)] for x, y in shape.exterior.coords[:-1]]


def _detections(spec: FixtureSpec, index: int, offset: np.ndarray, image_box) -> list[dict]:
    detections = []
    for number, plant in enumerate(spec.plants):
        name = plant.plant_id or f"p{number}"
        if index in spec.dropouts.get(name, ()):
            continue
        u, v = plant.u0 + offset[0], plant.v0 + offset[1]
        polygon = _circle(u, v, plant.radius_px, image_box)
        if polygon is None:
            continue
        detections.append({"class": "moringa", "confidence": 0.9, "polygon": polygon})
        if name in spec.duplicate_masks:
            twin = _circle(u + 0.2 * plant.radius_px, v, plant.radius_px * 0.9, image_box)
            if twin is not None:
                detections.append({"class": "moringa", "confidence": 0.6, "polygon": twin})
    for frame, u, v, r in spec.false_positives:
        if frame == index:
            polygon = _circle(u, v, r, image_box)
            if polygon is not None:
                detections.append({"class": "moringa", "confidence": 0.5, "polygon": polygon})
    return detections
