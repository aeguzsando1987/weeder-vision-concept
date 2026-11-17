"""Mapa local supuesto para E2 a partir de los tracks del cuadro (orden M0-P §4.4-§4.6).

- **Corredor:** dominio de trabajo de la herramienta, **solidario al vehiculo**: con
  la camara supuesta rigida al vehiculo, es la banda normalizada del surco en la
  imagen de **cada** cuadro, proyectada a `W_segment` con la proyeccion del cuadro.
  Asi la deriva acumulada del estimador de traslacion no desplaza el corredor fuera
  de la vista; las plantas si viven en `W_segment` y se reobservan en cada cuadro.
  Region conocida = ese rectangulo, **asumido** para el experimento
  (`known_region_basis = assumed_visible_row_band`); no certifica suelo real libre.
- **Exclusiones:** cada track activo o en espera cuya proteccion toca el corredor,
  tambien si pertenece a otra hilera. Circulo centrado en el centroide de la copa
  proyectada que la contiene (radio de representacion) mas `proxy_margin_m`; la
  incertidumbre de la politica declarada en el perfil para `ground_assumed` se aplica
  una sola vez, como margen de la exclusion. La confianza de la red no entra en ninguna cota.
- **Desconocido:** tracks caducados dentro del corredor; no se aprueba el hueco.
- **Eje de la hilera:** mediana de `x` de los tracks activos cuyo centroide cae en la
  banda (al menos `axis_min_plants`); si no, la columna `axis_u_norm` del perfil.
  Sigue la deriva lateral de la hilera sin offsets por clip.
- **Mapa sin plantas:** sin la opcion rotulada `assume_free_when_empty` no se
  declara libre: el cuadro se abstiene por observaciones insuficientes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import Point, box

from precision_weeder.core.config import SafetyConfig
from precision_weeder.core.types import (
    CircularExclusion,
    Corridor,
    Frame,
    InflationTarget,
    Point2,
    UncertaintyBound,
    UncertaintyPolicy,
    exterior_circle_polygon,
)
from precision_weeder.core.units import Provenance, SpatialDomain
from precision_weeder.mapping.local_map import LocalMap, rectangle_ring

from .profile import M0Profile
from .projection import SegmentProjection
from .tracking import Track

__all__ = ["MapOutcome", "assumed_policy", "build_local_map"]

DOMAIN = SpatialDomain.GROUND_ASSUMED


def assumed_policy(profile: M0Profile) -> UncertaintyPolicy:
    """Politica del perfil en el dominio `ground_assumed` (metros supuestos)."""
    cfg = profile.uncertainty.policy
    return UncertaintyPolicy(
        bound=UncertaintyBound(
            localization=cfg.localization_bound_m,
            calibration=cfg.calibration_bound_m,
            tracking=cfg.tracking_bound_m,
            domain=DOMAIN,
            provenance=Provenance.ASSUMED,
            k_sigma=cfg.sigma_multiplier_k,
        ),
        budget=cfg.bound_budget_m,
        inflate=cfg.inflate,
    )


def _protection_radius(profile: M0Profile, track: Track) -> float:
    """Radio protegido de la planta, sin incertidumbre (perfil `map.protection`)."""
    if profile.map.protection == "assumed_stem":
        return float(profile.map.stem_protection_radius_m)
    return track.radius + profile.map.proxy_margin_m


@dataclass(frozen=True)
class MapOutcome:
    """Mapa del cuadro, o la causa por la que no se construye."""

    local_map: LocalMap | None
    reason: str
    corridor_box: tuple[float, float, float, float] | None
    axis: tuple[float, str] | None = None
    """`x` de la linea de plantacion en `W_segment` y su fuente."""


def build_local_map(
    profile: M0Profile,
    safety: SafetyConfig,
    projection: SegmentProjection,
    tracks: list[Track],
    expired: list[Track],
    anchor_projection: SegmentProjection,
) -> MapOutcome:
    """Construye el `LocalMap` E2 del cuadro (dominio `ground_assumed`)."""
    row = profile.row
    x_row = sorted(
        (
            projection.column_u_to_ground_x(row.u_min_norm),
            projection.column_u_to_ground_x(row.u_max_norm),
        )
    )
    _ = anchor_projection  # se conserva en la firma: el ancla define W_segment
    vx_min, vx_max, vy_min, vy_max = projection.visible_ground_box()
    x_min, x_max = max(x_row[0], vx_min), min(x_row[1], vx_max)
    if x_max - x_min <= 0.0:
        return MapOutcome(None, "surco fuera del suelo visible (deriva lateral)", None)
    corridor_box = (x_min, x_max, vy_min, vy_max)
    in_band = sorted(float(t.centroid[0]) for t in tracks if x_row[0] <= t.centroid[0] <= x_row[1])
    if len(in_band) >= row.axis_min_plants:
        axis = (float(np.median(in_band)), f"mediana de {len(in_band)} plantas")
    else:
        axis = (projection.column_u_to_ground_x(row.axis_u_norm), "axis_u_norm del perfil")
    corridor_shape = box(x_min, vy_min, x_max, vy_max)
    policy = assumed_policy(profile)
    on_tool = policy.inflate is InflationTarget.TOOL
    margin = 0.0 if on_tool else policy.total
    exclusions = []
    for track in sorted(tracks, key=lambda t: t.track_id):
        cx, cy = track.centroid
        radius = _protection_radius(profile, track)
        if Point(cx, cy).buffer(radius + policy.total, 64).intersects(corridor_shape):
            exclusions.append(
                CircularExclusion(
                    plant_id=f"track_{track.track_id}",
                    center=Point2(x=float(cx), y=float(cy), frame=Frame.WORLD, domain=DOMAIN),
                    protection_radius=radius,
                    uncertainty_margin=margin,
                    polygon_segments=profile.map.polygon_segments,
                    provenance=Provenance.ASSUMED,
                )
            )
    unknown = []
    for track in sorted(expired, key=lambda t: t.track_id):
        cx, cy = track.centroid
        radius = _protection_radius(profile, track) + policy.total
        region = Point(cx, cy).buffer(radius, 64)
        if region.intersects(corridor_shape):
            unknown.append(
                exterior_circle_polygon((cx, cy), radius, profile.map.polygon_segments).tolist()
            )
    if not exclusions and not profile.map.assume_free_when_empty:
        return MapOutcome(
            None,
            "sin plantas en el corredor: la ausencia de detecciones no certifica suelo libre",
            corridor_box,
            axis,
        )
    local_map = LocalMap(
        map_id=f"{projection.segment_id}:f{projection.frame_index}",
        segment_id=projection.segment_id,
        domain=DOMAIN,
        corridor=Corridor(
            x_min=x_min,
            x_max=x_max,
            y_min=vy_min,
            y_max=vy_max,
            frame=Frame.WORLD,
            domain=DOMAIN,
            provenance=Provenance.ASSUMED,
        ),
        known_region=rectangle_ring(x_min, x_max, vy_min, vy_max),
        known_region_basis=(
            "assumed_visible_row_band"
            if exclusions
            else "assumed_free_when_empty (supuesto rotulado del perfil)"
        ),
        exclusions=tuple(exclusions),
        uncertainty=policy,
        tool_inflation=policy.total if on_tool else 0.0,
        unknown_regions=tuple(unknown),
        free_space_certified=not exclusions,
        provenance=Provenance.ASSUMED,
        notes=(
            "M0-P exploratorio: escala supuesta, copa como proxy de planta"
            if profile.map.protection == "canopy"
            else "exploratorio: escala supuesta, tallo SUPUESTO en el centroide de la copa",
            f"perfil {profile.profile_id}, surco {row.name}",
        ),
    )
    return MapOutcome(local_map, "ok", corridor_box, axis)
