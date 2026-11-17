"""Movimiento aparente en imagen por flujo optico disperso (E0, orden §4).

Convencion de imagen: `u` hacia la derecha y `v` hacia abajo, en pixeles del
cuadro original salvo que se indique lo contrario; no se aplica rotacion de
visualizacion. Metodo: esquinas Shi-Tomasi en la ROI, Lucas-Kanade piramidal
hacia delante y hacia atras; un vector es valido si ambos pasos convergen y el
error ida/vuelta no supera `fb_max_error_px`.

Los vectores se separan en **suelo** y **vegetacion** con la mascara ExG del
cuadro anterior, para que plantas movidas por viento no dominen la estimacion.
La direccion se informa en imagen (`up/down/left/right/mixed/none/unknown`) y
**nunca** se traduce a avance fisico: eso requiere un montaje declarado (E1, R7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.config import FlowConfig, RoiConfig
from .frame_metrics import vegetation_mask

__all__ = [
    "FlowEstimate",
    "FlowVectors",
    "affine_flow_fit",
    "classify_direction",
    "estimate_flow",
    "roi_pixels",
]

DIRECTIONS = ("up", "down", "left", "right", "mixed", "none", "unknown")


@dataclass(frozen=True)
class FlowVectors:
    """Vectores de un par de cuadros, en pixeles del cuadro **original**."""

    origins: np.ndarray  # (n, 2) u, v
    displacements: np.ndarray  # (n, 2) du, dv
    valid: np.ndarray  # (n,) bool
    on_vegetation: np.ndarray  # (n,) bool
    fb_error: np.ndarray  # (n,) px (escala de analisis)


@dataclass(frozen=True)
class FlowEstimate:
    """Resumen robusto del flujo de suelo para un par de cuadros."""

    index_from: int
    index_to: int
    dt_s: float | None
    detected: int
    valid: int
    rejected: int
    valid_soil: int
    valid_vegetation: int
    median_du_px: float | None
    median_dv_px: float | None
    mad_du_px: float | None
    mad_dv_px: float | None
    median_magnitude_px: float | None
    fraction_up: float | None
    fraction_down: float | None
    fraction_left: float | None
    fraction_right: float | None
    direction: str
    vegetation_median_du_px: float | None
    vegetation_median_dv_px: float | None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def roi_pixels(roi: RoiConfig, width: int, height: int) -> tuple[int, int, int, int]:
    """ROI normalizada a pixeles enteros `(u0, v0, u1, v1)` del cuadro dado."""
    return (
        round(roi.u_min * width),
        round(roi.v_min * height),
        round(roi.u_max * width),
        round(roi.v_max * height),
    )


def classify_direction(
    du: np.ndarray, dv: np.ndarray, dominance: float, min_motion_px: float
) -> tuple[str, dict[str, float]]:
    """Direccion dominante en imagen a partir de vectores validos.

    `none` si el modulo mediano es menor que `min_motion_px`; un sentido es
    dominante si su eje domina en mediana y al menos `dominance` de los vectores
    tienen ese signo sobre ese eje; en otro caso `mixed`. Sin vectores: `unknown`.
    """
    if du.size == 0:
        return "unknown", {}
    fractions = {
        "up": float(np.mean(dv < 0)),
        "down": float(np.mean(dv > 0)),
        "left": float(np.mean(du < 0)),
        "right": float(np.mean(du > 0)),
    }
    median_du, median_dv = float(np.median(du)), float(np.median(dv))
    if float(np.median(np.hypot(du, dv))) < min_motion_px:
        return "none", fractions
    if abs(median_dv) >= abs(median_du):
        label = "up" if median_dv < 0 else "down"
    else:
        label = "left" if median_du < 0 else "right"
    return (label if fractions[label] >= dominance else "mixed"), fractions


def estimate_flow(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    previous_bgr: np.ndarray,
    config: FlowConfig,
    roi: RoiConfig,
    scale: float,
    index_from: int,
    index_to: int,
    dt_s: float | None,
) -> tuple[FlowEstimate, FlowVectors]:
    """Flujo disperso entre dos cuadros reducidos por `scale`.

    Los resultados se reescalan a pixeles del cuadro original (`/ scale`). Los
    umbrales `min_motion_px` y `fb_max_error_px` se aplican en la escala de analisis.
    """
    import cv2

    height, width = previous_gray.shape
    u0, v0, u1, v1 = roi_pixels(roi, width, height)
    mask = np.zeros_like(previous_gray, dtype=np.uint8)
    mask[v0:v1, u0:u1] = 255
    corners = cv2.goodFeaturesToTrack(
        previous_gray,
        maxCorners=config.max_corners,
        qualityLevel=config.quality_level,
        minDistance=config.min_distance_px,
        mask=mask,
    )
    empty = np.zeros((0, 2))
    if corners is None or len(corners) == 0:
        vectors = FlowVectors(empty, empty, np.zeros(0, bool), np.zeros(0, bool), np.zeros(0))
        return _summarize(vectors, config, scale, index_from, index_to, dt_s), vectors
    points = corners.reshape(-1, 2).astype(np.float32)
    params = {
        "winSize": (config.window_px, config.window_px),
        "maxLevel": config.pyramid_levels,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    }
    lk = cv2.calcOpticalFlowPyrLK
    forward, status_f, _ = lk(previous_gray, current_gray, points, None, **params)
    backward, status_b, _ = lk(current_gray, previous_gray, forward, None, **params)
    fb_error = np.linalg.norm(points - backward, axis=1)
    valid = (status_f.ravel() == 1) & (status_b.ravel() == 1) & (fb_error <= config.fb_max_error_px)
    vegetation = vegetation_mask(previous_bgr)
    rows = np.clip(np.round(points[:, 1]).astype(int), 0, height - 1)
    cols = np.clip(np.round(points[:, 0]).astype(int), 0, width - 1)
    vectors = FlowVectors(
        origins=points.astype(np.float64) / scale,
        displacements=(forward - points).astype(np.float64) / scale,
        valid=valid,
        on_vegetation=vegetation[rows, cols],
        fb_error=fb_error.astype(np.float64),
    )
    return _summarize(vectors, config, scale, index_from, index_to, dt_s), vectors


def _mad(values: np.ndarray) -> float:
    return float(np.median(np.abs(values - np.median(values))))


def _summarize(
    vectors: FlowVectors,
    config: FlowConfig,
    scale: float,
    index_from: int,
    index_to: int,
    dt_s: float | None,
) -> FlowEstimate:
    soil = vectors.valid & ~vectors.on_vegetation
    vegetation = vectors.valid & vectors.on_vegetation
    du, dv = vectors.displacements[soil, 0], vectors.displacements[soil, 1]
    enough = int(soil.sum()) >= config.min_valid_points
    if enough:
        # Umbral de movimiento expresado en pixeles originales.
        label, fractions = classify_direction(
            du, dv, config.direction_dominance, config.min_motion_px / scale
        )
    else:
        label, fractions = "unknown", {}
    veg_du = vectors.displacements[vegetation, 0]
    veg_dv = vectors.displacements[vegetation, 1]
    return FlowEstimate(
        index_from=index_from,
        index_to=index_to,
        dt_s=dt_s,
        detected=int(vectors.valid.size),
        valid=int(vectors.valid.sum()),
        rejected=int(vectors.valid.size - vectors.valid.sum()),
        valid_soil=int(soil.sum()),
        valid_vegetation=int(vegetation.sum()),
        median_du_px=float(np.median(du)) if enough else None,
        median_dv_px=float(np.median(dv)) if enough else None,
        mad_du_px=_mad(du) if enough else None,
        mad_dv_px=_mad(dv) if enough else None,
        median_magnitude_px=float(np.median(np.hypot(du, dv))) if enough else None,
        fraction_up=fractions.get("up"),
        fraction_down=fractions.get("down"),
        fraction_left=fractions.get("left"),
        fraction_right=fractions.get("right"),
        direction=label,
        vegetation_median_du_px=float(np.median(veg_du)) if veg_du.size else None,
        vegetation_median_dv_px=float(np.median(veg_dv)) if veg_dv.size else None,
    )


def affine_flow_fit(vectors: FlowVectors) -> dict[str, float | int] | None:
    """Ajuste afin del campo de suelo: `d = a + b*(u-uc) + c*(v-vc)` para du y dv.

    Descriptor del campo, no calibracion: gradientes no nulos muestran que el flujo
    no es una traslacion uniforme. No identifican su causa: un giro en el plano entre
    cuadros con escala uniforme (`x' = R x + b`, flujo `(R - I) x + b`) tambien los
    produce, igual que la perspectiva o la inclinacion. No permiten afirmar ni
    descartar una escala pixel/suelo uniforme. Un residuo robusto pequeno es
    compatible con un campo suave, sin probar planitud. Coordenadas centradas en el
    centroide de los origenes.
    """
    soil = vectors.valid & ~vectors.on_vegetation
    if int(soil.sum()) < 10:
        return None
    origins = vectors.origins[soil]
    deltas = vectors.displacements[soil]
    center = origins.mean(axis=0)
    design = np.column_stack(
        [np.ones(len(origins)), origins[:, 0] - center[0], origins[:, 1] - center[1]]
    )
    result: dict[str, float | int] = {
        "points": int(soil.sum()),
        "center_u_px": float(center[0]),
        "center_v_px": float(center[1]),
    }
    for column, name in ((0, "du"), (1, "dv")):
        coefficients, *_ = np.linalg.lstsq(design, deltas[:, column], rcond=None)
        residual = deltas[:, column] - design @ coefficients
        result.update(
            {
                f"{name}_offset_px": float(coefficients[0]),
                f"{name}_per_u_px_per_px": float(coefficients[1]),
                f"{name}_per_v_px_per_px": float(coefficients[2]),
                f"{name}_residual_mad_px": float(np.median(np.abs(residual - np.median(residual)))),
            }
        )
    return result
