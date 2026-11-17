"""Fusion de duplicados y asociacion uno a uno en `W_segment` (orden M0-P §4.3-§4.4).

- **Fusion por cuadro:** dos mascaras son la misma planta si su IoU supera el umbral
  o si la menor queda contenida en la mayor por encima de la fraccion declarada. Se
  unen sus poligonos y se conserva la confianza maxima. Dos plantas cercanas que no
  cumplen esas condiciones siguen separadas: no se promedian.
- **Asociacion:** por distancia de centroides en `W_segment` con compuerta, uno a
  uno, en orden global de distancia con desempate estable (id de track, indice de
  observacion). Los tracks conservan identidad mientras haya soporte.
- **Observaciones recortadas:** si la mascara toca el borde de la imagen, la planta se
  ve solo en parte; su geometria no sustituye a una mayor ya observada (la proteccion
  no se reduce porque la planta salga del encuadre).
- **Caducidad:** un track sin observacion conserva su proteccion `timeout_frames`
  cuadros; despues caduca y su region pasa a desconocida (no se aprueba el hueco).
  Solo el cuadro actual y los anteriores intervienen: no hay observaciones futuras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from .observations import Observation
from .profile import FusionConfig, TrackingConfig
from .projection import SegmentProjection

__all__ = ["FusedObservation", "Track", "Tracker", "fuse_observations"]

BORDER_PX = 1.0
"""Una mascara a esta distancia del borde de la imagen se considera recortada."""


@dataclass(frozen=True)
class FusedObservation:
    """Una planta en un cuadro tras fusionar mascaras duplicadas (pixeles)."""

    shape_px: Polygon
    confidence: float
    target: str
    members: int


def _largest(geometry) -> Polygon:
    if isinstance(geometry, MultiPolygon):
        return max(geometry.geoms, key=lambda g: g.area)
    return geometry


def fuse_observations(
    observations: tuple[Observation, ...], config: FusionConfig
) -> list[FusedObservation]:
    """Fusiona mascaras de la misma planta con union-find determinista."""
    shapes = [o.shape() for o in observations]
    parent = list(range(len(shapes)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(shapes)):
        for j in range(i + 1, len(shapes)):
            a, b = shapes[i], shapes[j]
            if not a.intersects(b):
                continue
            inter = a.intersection(b).area
            union = a.union(b).area
            smaller = min(a.area, b.area)
            if (union > 0 and inter / union >= config.iou_threshold) or (
                smaller > 0 and inter / smaller >= config.containment_threshold
            ):
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(shapes)):
        groups.setdefault(root(i), []).append(i)
    fused = []
    for members in sorted(groups.values()):
        merged = _largest(unary_union([shapes[k] for k in members]))
        fused.append(
            FusedObservation(
                shape_px=merged,
                confidence=max(observations[k].confidence for k in members),
                target=observations[members[0]].target,
                members=len(members),
            )
        )
    return fused


@dataclass
class Track:
    """Planta seguida en `W_segment` (metros supuestos)."""

    track_id: int
    polygon_w: np.ndarray
    target: str
    first_seen: int
    last_seen: int
    hits: int = 1
    missed: int = 0

    @property
    def centroid(self) -> np.ndarray:
        shape = Polygon(self.polygon_w)
        return np.array([shape.centroid.x, shape.centroid.y])

    @property
    def radius(self) -> float:
        """Radio del circulo centrado en el centroide que contiene la mascara proyectada."""
        return float(np.max(np.linalg.norm(self.polygon_w - self.centroid, axis=1)))

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "target": self.target,
            "centroid_m": self.centroid.tolist(),
            "radius_m": self.radius,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "hits": self.hits,
            "missed": self.missed,
        }


class Tracker:
    """Tracks de un segmento; se reinicia con el segmento."""

    def __init__(self, config: TrackingConfig) -> None:
        self.config = config
        self.tracks: list[Track] = []
        self.expired: list[Track] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks, self.expired, self._next_id = [], [], 1

    def update(
        self, frame_index: int, fused: list[FusedObservation], projection: SegmentProjection
    ) -> dict:
        """Asocia las observaciones del cuadro y actualiza caducidades."""
        observed = []
        width, height = projection.image_size
        for item in fused:
            polygon = np.asarray(item.shape_px.exterior.coords[:-1])
            polygon_w = projection.image_to_ground(polygon)
            centroid = np.asarray(Polygon(polygon_w).centroid.coords[0])
            truncated = bool(
                polygon[:, 0].min() <= BORDER_PX
                or polygon[:, 1].min() <= BORDER_PX
                or polygon[:, 0].max() >= width - BORDER_PX
                or polygon[:, 1].max() >= height - BORDER_PX
            )
            observed.append((polygon_w, centroid, item, truncated))
        pairs = []
        for t_index, track in enumerate(self.tracks):
            for o_index, (_, centroid, _, _) in enumerate(observed):
                distance = float(np.linalg.norm(track.centroid - centroid))
                if distance <= self.config.gate_m:
                    pairs.append((distance, track.track_id, o_index, t_index))
        pairs.sort()
        used_tracks: set[int] = set()
        used_obs: set[int] = set()
        for _, _, o_index, t_index in pairs:
            if t_index in used_tracks or o_index in used_obs:
                continue
            used_tracks.add(t_index)
            used_obs.add(o_index)
            track = self.tracks[t_index]
            polygon_w, _, _, truncated = observed[o_index]
            # Una mascara recortada por el borde no reduce la proteccion: se conserva la
            # geometria mayor (la planta sigue ahi aunque se vea solo en parte).
            if not truncated or Polygon(polygon_w).area >= Polygon(track.polygon_w).area:
                track.polygon_w = polygon_w
            track.last_seen = frame_index
            track.hits += 1
            track.missed = 0
        created = []
        for o_index, (polygon_w, _, item, _) in enumerate(observed):
            if o_index in used_obs:
                continue
            track = Track(self._next_id, polygon_w, item.target, frame_index, frame_index)
            self._next_id += 1
            self.tracks.append(track)
            created.append(track.track_id)
        expired_now = []
        alive = []
        for t_index, track in enumerate(self.tracks):
            if t_index not in used_tracks and track.first_seen != frame_index:
                track.missed += 1
            if track.missed > self.config.timeout_frames:
                self.expired.append(track)
                expired_now.append(track.track_id)
            else:
                alive.append(track)
        self.tracks = alive
        return {
            "matched": len(used_obs),
            "created": created,
            "expired": expired_now,
            "coasting": [t.track_id for t in self.tracks if t.missed > 0],
        }
