"""Generador de videos sinteticos pequenos para pruebas de E0 (se crean durante el test).

El contenido es conocido: una textura de "suelo" sembrada que se desplaza hacia
arriba `shift_px` pixeles por cuadro (flujo esperado dv = -shift_px, du = 0) y un
codigo binario del indice del cuadro en bloques de alto contraste en la franja
superior, para recuperar el orden tras una compresion con perdida sin exigir
igualdad de pixeles.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 192, 128
CODE_BITS = 6
CODE_BLOCK = 16
CODE_BAND_PX = CODE_BLOCK  # filas superiores ocupadas por el codigo


def _texture(seed: int, height: int, width: int) -> np.ndarray:
    import cv2

    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, 255, size=(height, width)).astype(np.float32)
    blurred = cv2.GaussianBlur(noise, (0, 0), 2.0)
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min()) * 200 + 25
    return blurred.astype(np.uint8)


def frame_image(index: int, shift_px: int, seed: int = 7) -> np.ndarray:
    """Cuadro BGR esperado (antes de comprimir) para el indice dado."""
    total = HEIGHT + shift_px * 400
    texture = _texture(seed, total, WIDTH)
    top = index * shift_px
    gray = texture[top : top + HEIGHT].copy()
    for bit in range(CODE_BITS):
        value = 255 if (index >> bit) & 1 else 0
        gray[:CODE_BLOCK, bit * CODE_BLOCK : (bit + 1) * CODE_BLOCK] = value
    return np.dstack([gray, gray, gray])


def decode_index(image: np.ndarray) -> int:
    """Recupera el indice codificado en la franja superior (umbral a mitad de escala)."""
    gray = image.mean(axis=2) if image.ndim == 3 else image
    index = 0
    for bit in range(CODE_BITS):
        block = gray[3 : CODE_BLOCK - 3, bit * CODE_BLOCK + 3 : (bit + 1) * CODE_BLOCK - 3]
        if block.mean() > 127:
            index |= 1 << bit
    return index


def write_video(
    path: Path,
    frames: int = 24,
    fps: int = 15,
    shift_px: int = 4,
    pts_ticks: list[int] | None = None,
    faststart: bool = False,
    codec: str = "libx264",
) -> Path:
    """Escribe un MP4 sintetico. `pts_ticks` (en 1/fps) permite crear saltos."""
    import av

    options = {"movflags": "faststart"} if faststart else {}
    with av.open(str(path), "w", options=options) as container:
        stream = container.add_stream(codec, rate=fps)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, fps)
        if codec == "libx264":
            stream.options = {"crf": "18", "preset": "veryfast", "bf": "0"}
        for index in range(frames):
            frame = av.VideoFrame.from_ndarray(frame_image(index, shift_px), format="bgr24")
            frame.pts = pts_ticks[index] if pts_ticks is not None else index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path
