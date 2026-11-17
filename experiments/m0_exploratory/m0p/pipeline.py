"""Orquestacion causal cuadro a cuadro de M0-P (orden §3-§4).

Para cada cuadro `t`, solo con informacion de `0..t` (sin observaciones futuras):

1. Tiempo: PTS presente y creciente; si no, discontinuidad (se retira la ruta y se
   reinicia el segmento).
2. Movimiento: traslacion robusta del suelo desde `t-1` (E0). Si no es valida, la
   proyeccion se pierde en ese cuadro (sin ruta) y el siguiente cuadro abre un
   segmento nuevo con tracks vacios; nunca se unen mapas de `W` distintos.
3. Observaciones del proveedor -> fusion -> tracks en `W_segment`.
4. Mapa local supuesto -> E2 (planificador + revalidacion independiente) o
   revalidacion de la ruta anterior si no toca replanificar.
5. Dibujo y escritura del cuadro con su PTS; registro del sidecar.

El video y el sidecar se escriben en archivos temporales y se publican solo al
completar el procesamiento: un artefacto parcial nunca queda como ejecucion final.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.core.identity import content_hash
from precision_weeder.core.trajectory import TaskPath
from precision_weeder.video.reader import iter_frames, probe_video, sha256_file

from .mapbuilder import build_local_map
from .motion import GroundMotionEstimator
from .observations import ObservationSet
from .profile import M0Profile
from .projection import SegmentProjection
from .render import VideoWriter, draw_overlay
from .route import Reference, RoutePlanner, assumed_tool_profile, chain_hash, chain_x_at
from .setpoint import SetpointLog
from .tracking import Tracker, fuse_observations

__all__ = ["PipelineSummary", "run_pipeline"]

SIDECAR_SCHEMA_VERSION = 1


@dataclass
class PipelineSummary:
    frames: int = 0
    statuses: dict[str, int] = field(default_factory=dict)
    valid_frames: list[int] = field(default_factory=list)
    longest_valid_run: int = 0
    segments: int = 0
    output_video: str = ""
    sidecar: str = ""
    store: str = ""
    last_frame_duration_policy: str | None = None
    route_kinds: dict[str, int] = field(default_factory=dict)
    continuity: dict[str, int] = field(default_factory=dict)
    discontinuities: int = 0
    max_lateral_jump_m: float = 0.0
    jumps_over_5cm: int = 0
    setpoints: str = ""
    setpoint: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_kinds": dict(sorted(self.route_kinds.items())),
            "continuity_decisions": dict(sorted(self.continuity.items())),
            "discontinuities": self.discontinuities,
            "max_lateral_jump_m": self.max_lateral_jump_m,
            "jumps_over_5cm": self.jumps_over_5cm,
            "last_frame_duration_policy": self.last_frame_duration_policy,
            "frames": self.frames,
            "statuses": dict(sorted(self.statuses.items())),
            "valid_frame_count": len(self.valid_frames),
            "longest_valid_run": self.longest_valid_run,
            "temporal_route_coverage": len(self.valid_frames) / self.frames if self.frames else 0.0,
            "segments": self.segments,
            "output_video": self.output_video,
            "sidecar": self.sidecar,
            "store": self.store,
            "setpoints": self.setpoints,
            "setpoint": self.setpoint,
        }


def _source_last_duration(probe, last_pts: int) -> int | None:
    """Duracion del ultimo cuadro segun el fin de presentacion declarado por la fuente,
    en ticks de su base temporal; `None` si la fuente no lo declara de forma util."""
    if probe.duration_reported_s is None or not probe.time_base:
        return None
    time_base = Fraction(probe.time_base)
    end = (probe.start_time_s or 0.0) + probe.duration_reported_s
    duration = round(end / time_base) - last_pts
    return duration if duration > 0 else None


@dataclass
class _Segment:
    segment_id: str
    anchor: int
    center: tuple[float, float]
    displacement: tuple[float, float] = (0.0, 0.0)


class _Store:
    """Objetos por identidad (mapas, rutas, informes) referenciados desde el sidecar."""

    def __init__(self, root: Path) -> None:
        self.root = root
        for kind in ("maps", "routes", "reports"):
            (root / kind).mkdir(parents=True, exist_ok=True)

    def put(self, kind: str, digest: str, payload: dict) -> str:
        target = self.root / kind / f"{digest}.json"
        if not target.exists():
            target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        return f"{kind}/{digest}.json"


def run_pipeline(
    video: Path,
    profile: M0Profile,
    resolved: dict[str, Any],
    observations: ObservationSet,
    output: Path,
    root: Path,
) -> PipelineSummary:
    """Procesa el video completo; publica video, sidecar y almacen al terminar.

    Si el procesamiento no termina, se borran los temporales: un artefacto parcial
    nunca queda publicado como ejecucion final (MP-09).

    Raises:
        PrecisionWeederError: entrada incompatible (hash/cuadros/tamano/video de salida).
    """
    output = Path(output)
    partials = (
        output.with_name(f".{output.stem}.partial.mp4"),
        output.with_name(f".{output.stem}.sidecar.partial.jsonl"),
        output.with_name(f".{output.stem}.setpoints.partial.csv"),
    )
    try:
        return _run(video, profile, resolved, observations, output, root)
    except BaseException:
        for partial in partials:
            partial.unlink(missing_ok=True)
        raise


def _run(
    video: Path,
    profile: M0Profile,
    resolved: dict[str, Any],
    observations: ObservationSet,
    output: Path,
    root: Path,
) -> PipelineSummary:
    video = Path(video)
    output = Path(output)
    if video.resolve() == output.resolve():
        raise PrecisionWeederError("la salida no puede sobrescribir el video de entrada")
    digest = sha256_file(video)
    if profile.expected_video_sha256 and digest != profile.expected_video_sha256:
        raise PrecisionWeederError(f"sha256 del video {digest} distinto del esperado por el perfil")
    if observations.video_sha256 != digest:
        raise PrecisionWeederError("la exportacion de observaciones corresponde a otro video")
    probe = probe_video(video)
    if (probe.width_px, probe.height_px) != observations.image_size:
        raise PrecisionWeederError("tamano de cuadro distinto entre video y exportacion")

    safety = resolved["safety"].safety
    tool = assumed_tool_profile(resolved["tool"].tool)
    route_planner = RoutePlanner(
        profile, resolved["planner"].planner, resolved["validation"].validation, safety, tool
    )
    forward = 1 if profile.projection.forward_image_direction == "down" else -1
    motion = GroundMotionEstimator(profile.motion, forward)
    tracker = Tracker(profile.tracking)
    tool_polygon = np.asarray(tool.collision_polygon)

    stem = output.stem
    tmp_video = output.with_name(f".{stem}.partial.mp4")
    sidecar_path = output.with_suffix(".sidecar.jsonl")
    tmp_sidecar = output.with_name(f".{stem}.sidecar.partial.jsonl")
    store = _Store(output.with_suffix(".store"))
    summary = PipelineSummary()
    writer: VideoWriter | None = None
    segment: _Segment | None = None
    segments_opened = 0
    previous_ts: float | None = None
    reference: Reference | None = None
    displayed: tuple[str, tuple[TaskPath, ...]] | None = None  # (segmento, cadena) previa
    run = 0
    reference_segment: str | None = None
    setpoints = SetpointLog()
    setpoints_path = output.with_suffix(".setpoints.csv")
    tmp_setpoints = output.with_name(f".{stem}.setpoints.partial.csv")

    try:
        with tmp_sidecar.open("w", encoding="utf-8") as sidecar:
            header = {
                "record": "header",
                "schema_version": SIDECAR_SCHEMA_VERSION,
                "video": {"path": str(video), "sha256": digest},
                "profile": profile.model_dump(mode="json"),
                "profile_hash": content_hash(profile.model_dump(mode="json")),
                "resolved_config_hash": content_hash(
                    {k: v.model_dump(mode="json") for k, v in resolved.items()}
                ),
                "provider": observations.provider,
                "observation_export_hash": observations.export_hash,
                "tool_profile_hash": tool.content_hash(),
                "domain": "ground_assumed",
                "label": "EXPLORATORIO: escala supuesta, copa proxy, mecanica no evaluada",
                "audio": profile.render.audio,
            }
            sidecar.write(json.dumps(header, sort_keys=True) + "\n")
            for frame in iter_frames(video, image_format="bgr24"):
                index = frame.index
                if writer is None:
                    writer = VideoWriter(
                        tmp_video, frame.width, frame.height, frame.time_base, profile.render.crf
                    )
                record: dict[str, Any] = {
                    "record": "frame",
                    "frame_index": index,
                    "pts": frame.pts,
                    "timestamp_s": frame.timestamp_s,
                }
                status, reason = "segmento_iniciado", "inicio de segmento: sin ruta en el ancla"
                projection = None
                local_map = None
                corridor_box = None
                path_now = None
                axis_x = None

                ts_ok = (
                    frame.pts is not None
                    and frame.timestamp_s is not None
                    and math.isfinite(frame.timestamp_s)
                    and (previous_ts is None or frame.timestamp_s > previous_ts)
                )
                step = motion.step(index, frame.image)
                if not ts_ok:
                    status, reason = "timestamp_invalido", "PTS ausente o no creciente"
                    segment = None
                    motion.reset()
                elif segment is None:
                    segments_opened += 1
                    segment = _Segment(
                        f"W_seg{segments_opened:03d}", index, (frame.width / 2, frame.height / 2)
                    )
                    tracker.reset()
                elif index - segment.anchor >= profile.motion.max_segment_frames:
                    status = "segmento_iniciado"
                    reason = "longitud maxima de segmento: se reinicia W para acotar la deriva"
                    segments_opened += 1
                    segment = _Segment(
                        f"W_seg{segments_opened:03d}", index, (frame.width / 2, frame.height / 2)
                    )
                    tracker.reset()
                elif step is None or not step.valid:
                    status = "proyeccion_perdida"
                    reason = step.reason if step else "sin paso de movimiento"
                    segment = None
                else:
                    du, dv = segment.displacement
                    segment.displacement = (du + step.du_px, dv + step.dv_px)
                if ts_ok:
                    previous_ts = frame.timestamp_s
                record["motion"] = step.to_dict() if step else None

                if segment is not None:
                    projection = SegmentProjection(
                        segment.segment_id,
                        segment.anchor,
                        index,
                        profile.projection.scale_m_per_px,
                        forward,
                        segment.center,
                        segment.displacement,
                        (frame.width, frame.height),
                    )
                    anchor_projection = SegmentProjection(
                        segment.segment_id,
                        segment.anchor,
                        segment.anchor,
                        profile.projection.scale_m_per_px,
                        forward,
                        segment.center,
                        (0.0, 0.0),
                        (frame.width, frame.height),
                    )
                    if index >= observations.frame_count:
                        raise PrecisionWeederError(
                            "el video tiene mas cuadros que la exportacion "
                            f"({observations.frame_count})"
                        )
                    observed = observations.for_frame(index, profile.provider.min_confidence)
                    fused = fuse_observations(observed, profile.fusion)
                    tracking = tracker.update(index, fused, projection)
                    record.update(
                        {
                            "segment_id": segment.segment_id,
                            "projection": projection.to_dict(),
                            "observations": len(observed),
                            "fused": len(fused),
                            "tracking": tracking,
                            "tracks": [t.to_dict() for t in tracker.tracks],
                        }
                    )
                    if index != segment.anchor:
                        outcome = build_local_map(
                            profile,
                            safety,
                            projection,
                            tracker.tracks,
                            tracker.expired,
                            anchor_projection,
                        )
                        corridor_box = outcome.corridor_box
                        local_map = outcome.local_map
                        if local_map is None:
                            status, reason = "observaciones_insuficientes", outcome.reason
                        else:
                            map_hash = local_map.content_hash()
                            record["map"] = store.put("maps", map_hash, local_map.to_dict())
                            record["map_hash"] = map_hash
                            axis_x = outcome.axis[0] if outcome.axis else None
                            if reference is not None and reference_segment != segment.segment_id:
                                reference = None
                            path_now, status, reason, reference = _route_for_frame(
                                route_planner,
                                profile,
                                local_map,
                                projection,
                                corridor_box,
                                outcome.axis,
                                reference,
                                index,
                                record,
                                store,
                            )
                            reference_segment = segment.segment_id
                            _continuity_metrics(
                                record, displayed, segment.segment_id, projection, profile, path_now
                            )
                if status != "vigente" or segment is None:
                    # Ruta retirada en el primer cuadro afectado (MP-06): sin referencia.
                    reference = None
                if displayed is not None and path_now is None and "continuity" not in record:
                    record["continuity"] = {
                        "decision": "retirada",
                        "discontinuity": True,
                        "cause": reason,
                    }
                displayed = (segment.segment_id, path_now) if path_now and segment else None
                _summarize(summary, record)
                record["status"], record["reason"] = status, reason
                record["setpoint"] = setpoints.update(
                    index,
                    frame.timestamp_s if ts_ok else None,
                    projection,
                    profile,
                    path_now if status == "vigente" else None,
                    (record.get("row_axis") or {}).get("task_x_m"),
                    record.get("route_hash"),
                )
                image = draw_overlay(
                    frame.image,
                    projection,
                    status,
                    reason,
                    index,
                    corridor_box,
                    axis_x,
                    local_map,
                    tracker.tracks if projection is not None else [],
                    path_now,
                    tool_polygon,
                    profile.render.footprint_samples,
                    setpoints.rows if profile.render.setpoint_overlay else None,
                    profile.tool_line.v_norm,
                )
                last_pts = frame.pts if frame.pts is not None else index
                writer.write(image, last_pts)
                sidecar.write(json.dumps(record, sort_keys=True) + "\n")
                summary.frames += 1
                summary.statuses[status] = summary.statuses.get(status, 0) + 1
                if status == "vigente":
                    summary.valid_frames.append(index)
                    run += 1
                    summary.longest_valid_run = max(summary.longest_valid_run, run)
                else:
                    run = 0
    except BaseException:
        if writer is not None:
            writer.abort()  # libera el parcial para poder borrarlo (Windows)
        raise
    if writer is None:
        raise PrecisionWeederError("el video no tiene cuadros decodificables")
    writer.close(_source_last_duration(probe, last_pts))
    summary.last_frame_duration_policy = writer.last_duration_policy
    if summary.frames != observations.frame_count:
        raise PrecisionWeederError(
            f"cuadros decodificados ({summary.frames}) distintos de la exportacion "
            f"({observations.frame_count})"
        )
    setpoints.write_csv(tmp_setpoints)
    os.replace(tmp_video, output)
    os.replace(tmp_sidecar, sidecar_path)
    os.replace(tmp_setpoints, setpoints_path)
    summary.setpoints = str(setpoints_path)
    summary.setpoint = setpoints.summary()
    summary.segments = segments_opened
    summary.output_video = str(output)
    summary.sidecar = str(sidecar_path)
    summary.store = str(store.root)
    return summary


def _continuity_metrics(record, displayed, segment_id, projection, profile, paths) -> None:
    """Salto lateral entre la propuesta previa mostrada y la actual en la linea de
    herramienta: cambio entre propuestas, no movimiento ejecutado (orden §10)."""
    continuity = record.setdefault("continuity", {})
    if paths is None or displayed is None:
        continuity.setdefault("lateral_jump_m", None)
        return
    previous_segment, previous = displayed
    if previous_segment != segment_id:
        continuity["lateral_jump_m"] = None
        continuity["discontinuity"] = True
        continuity["cause"] = "cambio de segmento W: la referencia anterior no se transporta"
        return
    y_tool = projection.row_v_to_ground_y(profile.tool_line.v_norm)
    before, now = chain_x_at(previous, y_tool), chain_x_at(paths, y_tool)
    jump = None if before is None or now is None else abs(now - before)
    continuity["lateral_jump_m"] = jump
    if jump is not None and jump > 1e-6 and continuity.get("decision") == "inicio":
        continuity["discontinuity"] = True


def _summarize(summary: PipelineSummary, record: dict[str, Any]) -> None:
    kind = record.get("route_kind")
    if kind:
        summary.route_kinds[kind] = summary.route_kinds.get(kind, 0) + 1
    continuity = record.get("continuity") or {}
    decision = continuity.get("decision")
    if decision:
        summary.continuity[decision] = summary.continuity.get(decision, 0) + 1
    if continuity.get("discontinuity"):
        summary.discontinuities += 1
    jump = continuity.get("lateral_jump_m")
    if jump is not None:
        summary.max_lateral_jump_m = max(summary.max_lateral_jump_m, jump)
        summary.jumps_over_5cm += jump > 0.05


def _store_legs(legs, store: _Store) -> list[dict[str, Any]]:
    return [
        {
            **leg.to_dict(),
            "route": store.put("routes", leg.path.content_hash(), leg.path.to_dict()),
            "route_hash": leg.path.content_hash(),
            "report": store.put("reports", leg.report.content_hash(), leg.report.to_dict()),
            "validation_status": leg.report.status.value,
            "clearance_lower_bound_m": leg.report.clearance_lower_bound_m,
        }
        for leg in legs
    ]


def _route_for_frame(
    route_planner: RoutePlanner,
    profile: M0Profile,
    local_map,
    projection: SegmentProjection,
    corridor_box,
    axis: tuple[float, str],
    reference: Reference | None,
    index: int,
    record: dict[str, Any],
    store: _Store,
) -> tuple[tuple[TaskPath, ...] | None, str, str, Reference | None]:
    """Cadena del cuadro con continuidad respecto de la referencia (orden v1.2 §9-§10).

    Con replanificacion debida se conserva y prolonga el prefijo de la referencia (o se
    reselecciona enlazando); si no toca y cabe en la edad maxima, solo se revalida.
    Devuelve la cadena publicada (los caminos religados que certifican sus informes,
    R2), estado, motivo y la nueva referencia."""
    cfg = profile.planning
    age = None if reference is None else index - reference.frame_index
    due = reference is None or age >= cfg.replan_every_frames or age > cfg.max_route_age_frames
    attempt = route_planner.plan(
        local_map, projection, corridor_box, axis[0], reference, extend=due
    )
    record["row_axis"] = {"x_m": axis[0], "source": axis[1], "task_x_m": attempt.axis_x}
    record["planning"] = attempt.summary()
    record["continuity"] = {
        "reference_chain": reference.chain_hash if reference else None,
        "reference_frame": reference.frame_index if reference else None,
        "decision": attempt.decision,
        "cause": attempt.cause,
        "discontinuity": attempt.decision == "discontinuidad",
    }
    paths = attempt.paths
    # Sin enlace certificado con la referencia (discontinuidad), una solucion fresca
    # queda como diagnostico: se retira la propuesta en este cuadro y no se adopta como
    # referencia; el cuadro siguiente reinicia declarando `inicio` (R5).
    if paths is None or attempt.decision == "discontinuidad":
        if attempt.legs:
            record["diagnostic_legs"] = _store_legs(attempt.legs, store)
        if attempt.decision == "discontinuidad":
            return None, "discontinuidad", attempt.cause, None
        if attempt.legs and not attempt.complete:
            return None, "cadena_incompleta", attempt.summary()["reason"], None
        return None, "sin_ruta", attempt.summary()["reason"] or "sin ruta", None
    record["legs"] = _store_legs(attempt.legs, store)
    record["route_hash"] = chain_hash(paths)
    record["route_kind"] = attempt.kind
    record["validation_status"] = "ACCEPTED"
    # Un tramo sin exclusiones en su alcance no tiene cota de holgura (None).
    bounds = [leg.report.clearance_lower_bound_m for leg in attempt.legs]
    bounds = [b for b in bounds if b is not None]
    record["clearance_lower_bound_m"] = min(bounds) if bounds else None
    planned_frame = reference.frame_index if attempt.decision == "revalidada" else index
    record["route_planned_frame"] = planned_frame
    record["route_age_frames"] = index - planned_frame
    if attempt.decision == "revalidada":
        record["revalidation"] = {"status": "ACCEPTED", "of_frame": planned_frame}
    labels = {"trabajo_en_gaps": "trabajo en gaps", "rodeo_lateral": "rodeo lateral"}
    reason = f"{labels[attempt.kind]}: {len(paths)} tramo(s) E2; {attempt.decision}"
    new_reference = Reference(attempt.legs, attempt.axis_x, planned_frame, record["route_hash"])
    return paths, "vigente", reason, new_reference
