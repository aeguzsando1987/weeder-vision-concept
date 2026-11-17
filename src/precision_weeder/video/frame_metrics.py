"""Metricas por cuadro para localizar condiciones relevantes (E0, orden §3).

Se calculan sobre una version reducida del cuadro (factor `analysis_scale`). Son
indicadores para **seleccionar** muestras e intervalos a inspeccionar, no
clasificadores: una sombra, un desenfoque o maleza densa se confirman mirando la
muestra, nunca por el valor de la metrica.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FrameMetrics",
    "compute_frame_metrics",
    "detect_content_repeats",
    "vegetation_mask",
]


@dataclass(frozen=True)
class FrameMetrics:
    index: int
    timestamp_s: float | None
    brightness_mean: float
    brightness_std: float
    sharpness_laplacian_var: float
    vegetation_fraction: float
    dark_fraction: float
    bright_fraction: float
    diff_from_previous: float | None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def vegetation_mask(bgr: np.ndarray, threshold: float = 20.0) -> np.ndarray:
    """Mascara de vegetacion por exceso de verde `ExG = 2G - R - B` (proxy, no detector)."""
    image = bgr.astype(np.float32)
    blue, green, red = image[..., 0], image[..., 1], image[..., 2]
    return (2.0 * green - red - blue) > threshold


def compute_frame_metrics(
    index: int,
    timestamp_s: float | None,
    bgr_small: np.ndarray,
    gray_small: np.ndarray,
    previous_gray: np.ndarray | None,
) -> FrameMetrics:
    """Metricas de brillo, contraste, nitidez, vegetacion y cambio entre cuadros."""
    import cv2

    gray = gray_small.astype(np.float32)
    laplacian = cv2.Laplacian(gray_small, cv2.CV_64F)
    diff = (
        float(np.mean(np.abs(gray - previous_gray.astype(np.float32))))
        if previous_gray is not None
        else None
    )
    return FrameMetrics(
        index=index,
        timestamp_s=timestamp_s,
        brightness_mean=float(gray.mean()),
        brightness_std=float(gray.std()),
        sharpness_laplacian_var=float(laplacian.var()),
        vegetation_fraction=float(vegetation_mask(bgr_small).mean()),
        dark_fraction=float((gray < 40).mean()),
        bright_fraction=float((gray > 230).mean()),
        diff_from_previous=diff,
    )


def detect_content_repeats(metrics: list[FrameMetrics], ratio: float) -> list[dict[str, Any]]:
    """Candidatos a imagen casi repetida respecto del cuadro anterior, con PTS avanzando.

    Criterio: `diff_from_previous < ratio * mediana(diff_from_previous)`. Una
    diferencia baja es compatible con una repeticion introducida al codificar, pero
    tambien con una escena casi estatica entre dos capturas: es un candidato, no un
    duplicado certificado, y no permite afirmar perdida de cuadros en captura.
    """
    diffs = np.array(
        [m.diff_from_previous for m in metrics if m.diff_from_previous is not None], dtype=float
    )
    if diffs.size == 0:
        return []
    threshold = ratio * float(np.median(diffs))
    return [
        {
            "kind": "content_repeat_candidate",
            "after_index": m.index - 1,
            "before_index": m.index,
            "interval_s": None,
            "at_s": m.timestamp_s,
            "mean_abs_diff": m.diff_from_previous,
            "threshold": threshold,
        }
        for m in metrics
        if m.diff_from_previous is not None and m.diff_from_previous < threshold
    ]
