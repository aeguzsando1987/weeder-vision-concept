"""M0-P: ruta geometrica exploratoria sobre video (orden M0-P, `ground_assumed`).

Extra exploratorio aislado del producto. Consume la API publica de E0 (lectura de
video y flujo aparente) y de E2 (mapa, perfil, planificador y validador) sin
modificar sus algoritmos. Todo resultado es una **propuesta geometrica** bajo
supuestos declarados: escala supuesta, copa como proxy y mecanica no evaluada.
"""

from __future__ import annotations

EXPLORATORY_LABEL = "EXPLORATORIO · escala supuesta · copa proxy · mecánica no evaluada"
"""Etiqueta permanente de todo artefacto visual (orden M0-P §4.9)."""

EXPLORATORY_LABEL_ASCII = "EXPLORATORIO - escala supuesta - copa proxy - mecanica no evaluada"
"""Misma etiqueta en ASCII para el texto rasterizado (OpenCV no dibuja tildes)."""

__all__ = ["EXPLORATORY_LABEL", "EXPLORATORY_LABEL_ASCII"]
