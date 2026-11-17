"""Traslacion 2D robusta del suelo entre cuadros consecutivos (orden M0-P §4.2).

Reutiliza `precision_weeder.video.apparent_motion.estimate_flow` de E0: vectores de
suelo (excluye vegetacion por ExG) con verificacion ida/vuelta. La traslacion es la
mediana de esos vectores. Se **invalida** (perdida de proyeccion, sin ruta) si hay
pocos inliers, si su dispersion (MAD) indica giro/paralaje/cambio de plano, si el
salto excede el maximo, o si el sentido contradice el avance supuesto. No se
integra indefinidamente una traslacion incorrecta.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from precision_weeder.video.apparent_motion import estimate_flow

from .profile import MotionConfig

__all__ = ["GroundMotionEstimator", "MotionStep"]


@dataclass(frozen=True)
class MotionStep:
    """Paso de movimiento aparente del suelo entre `index_from` e `index_to`."""

    index_from: int
    index_to: int
    valid: bool
    du_px: float | None
    dv_px: float | None
    inliers: int
    mad_px: float | None
    reason: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


class GroundMotionEstimator:
    """Estimador incremental: recibe cuadros BGR a resolucion completa en orden."""

    def __init__(self, config: MotionConfig, forward_sign: int) -> None:
        self.config = config
        self.forward_sign = forward_sign
        self._previous: tuple[int, np.ndarray, np.ndarray] | None = None

    def _small(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        scale = self.config.analysis_scale
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    def reset(self) -> None:
        self._previous = None

    def step(self, index: int, image: np.ndarray) -> MotionStep | None:
        """Paso desde el cuadro anterior; `None` para el primer cuadro tras un reinicio."""
        small, gray = self._small(image)
        previous = self._previous
        self._previous = (index, small, gray)
        if previous is None:
            return None
        prev_index, prev_small, prev_gray = previous
        cfg = self.config
        estimate, vectors = estimate_flow(
            prev_gray,
            gray,
            prev_small,
            cfg.flow,
            cfg.roi,
            cfg.analysis_scale,
            prev_index,
            index,
            None,
        )
        soil = vectors.valid & ~vectors.on_vegetation
        inliers = int(soil.sum())
        if inliers < cfg.min_inliers:
            return MotionStep(
                prev_index,
                index,
                False,
                None,
                None,
                inliers,
                None,
                f"inliers insuficientes ({inliers} < {cfg.min_inliers})",
            )
        displacement = vectors.displacements[soil]  # ya en pixeles del original
        du, dv = (float(v) for v in np.median(displacement, axis=0))
        mad = float(np.median(np.abs(displacement - np.array([du, dv])).max(axis=1)))
        if mad > cfg.max_mad_px:
            return MotionStep(
                prev_index,
                index,
                False,
                du,
                dv,
                inliers,
                mad,
                f"dispersion {mad:.2f} px > {cfg.max_mad_px}: giro/paralaje/plano",
            )
        if max(abs(du), abs(dv)) > cfg.max_step_px:
            return MotionStep(
                prev_index,
                index,
                False,
                du,
                dv,
                inliers,
                mad,
                f"salto {max(abs(du), abs(dv)):.1f} px > {cfg.max_step_px}",
            )
        # El suelo se mueve en la imagen en sentido opuesto al avance supuesto.
        forward_progress = -self.forward_sign * dv
        if forward_progress < cfg.min_forward_step_px:
            return MotionStep(
                prev_index,
                index,
                False,
                du,
                dv,
                inliers,
                mad,
                f"avance {forward_progress:.2f} px < {cfg.min_forward_step_px}: "
                "detenido o en retroceso respecto del supuesto",
            )
        _ = estimate  # el resumen de E0 se conserva para diagnostico en el sidecar
        return MotionStep(prev_index, index, True, du, dv, inliers, mad, "ok")
