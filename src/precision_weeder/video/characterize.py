"""Caracterizacion completa de un clip (E0): decodificacion, tiempo, muestras y flujo.

Este modulo orquesta las piezas del paquete `video`; el script
`scripts/characterize_videos.py` es una capa delgada sobre `characterize_video`.

Separacion de datos y ejecucion: `characterization.json` guarda el contenido
determinista en `results` y los datos de la ejecucion (fecha, comando, rutas
absolutas) en `run`, de modo que dos ejecuciones con el mismo original y la misma
configuracion producen `results` identicos.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import VideoConfig
from ..core.errors import PrecisionWeederError
from .apparent_motion import (
    FlowEstimate,
    FlowVectors,
    affine_flow_fit,
    estimate_flow,
    roi_pixels,
)
from .frame_metrics import FrameMetrics, compute_frame_metrics, detect_content_repeats
from .manifest import (
    ApparentMotion,
    TimestampSource,
    VideoManifestRecord,
)
from .observations import (
    CROSSING_COLUMNS,
    ClipFacts,
    ManualAssessment,
    ObservedTarget,
    crossing_events,
    validate_against_clip,
)
from .reader import (
    DecodedFrame,
    VideoProbe,
    backend_versions,
    iter_frames,
    probe_video,
    sha256_file,
)
from .sampling import select_samples
from .timing import TimestampReport, analyze_timestamps

__all__ = [
    "CharacterizationResult",
    "characterize_video",
    "config_hash",
    "video_id_from_path",
]

RESULTS_SCHEMA_VERSION = 1


@dataclass
class CharacterizationResult:
    video_id: str
    sha256: str
    probe: VideoProbe
    decode: dict[str, Any]
    timestamps: TimestampReport
    flow_summary: dict[str, Any]
    samples: list[dict[str, Any]]
    output_dir: Path
    record: VideoManifestRecord


def video_id_from_path(path: str | Path) -> str:
    """ID estable: nombre del archivo sin extension, saneado (mismo original, mismo ID)."""
    stem = Path(path).stem
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("_.-")
    if not clean:
        raise PrecisionWeederError(f"no se puede derivar un video_id de {path}")
    return clean


def config_hash(config: VideoConfig) -> str:
    """SHA-256 de la configuracion resuelta en JSON canonico."""
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _cell(row.get(key)) for key in columns})


def _cell(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _label_band(image: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    """Anade una banda inferior con texto; no tapa pixeles del cuadro."""
    import cv2

    band = np.full((34 * len(lines) + 12, image.shape[1], 3), 20, dtype=np.uint8)
    for row, text in enumerate(lines):
        cv2.putText(
            band, text, (12, 30 + 34 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (230, 230, 230), 2
        )
    return np.vstack([image, band])


def _draw_vectors(
    image: np.ndarray, vectors: FlowVectors, roi_box: tuple[int, int, int, int]
) -> np.ndarray:
    import cv2

    canvas = image.copy()
    u0, v0, u1, v1 = roi_box
    cv2.rectangle(canvas, (u0, v0), (u1 - 1, v1 - 1), (255, 255, 0), 2)
    for origin, delta, ok, veg in zip(
        vectors.origins, vectors.displacements, vectors.valid, vectors.on_vegetation, strict=True
    ):
        start = (round(origin[0]), round(origin[1]))
        end = (round(origin[0] + delta[0]), round(origin[1] + delta[1]))
        if not ok:
            color = (0, 0, 255)
        elif veg:
            color = (255, 0, 255)
        else:
            color = (0, 255, 0)
        cv2.arrowedLine(canvas, start, end, color, 2, tipLength=0.25)
    return canvas


def characterize_video(
    video_path: str | Path,
    config: VideoConfig,
    repo_root: str | Path,
    manual_assessment_path: str | Path | None = None,
) -> CharacterizationResult:
    """Caracteriza un clip y escribe sus derivados en `output_dir/<video_id>/`.

    La evaluacion manual, si existe, se valida primero sin contexto (esquema y tabla)
    y despues contra los cuadros decodificados (`validate_against_clip`), siempre
    antes de escribir ningun derivado.

    Raises:
        PrecisionWeederError: archivo inexistente/ilegible, hash distinto del
            esperado o evaluacion manual invalida o incompatible con el clip.
    """
    import cv2

    root = Path(repo_root)
    path = Path(video_path)
    if not path.is_absolute():
        path = root / path
    from .reader import VideoDecodeError

    if not path.is_file():
        raise VideoDecodeError(f"video inexistente: {path}")
    digest = sha256_file(path)
    if config.expected_sha256 is not None and digest != config.expected_sha256:
        raise PrecisionWeederError(
            f"sha256 de {path} ({digest}) distinto del esperado ({config.expected_sha256})"
        )
    video_id = video_id_from_path(path)
    assessment, observations = (
        ManualAssessment.load(manual_assessment_path, root)
        if manual_assessment_path is not None and Path(manual_assessment_path).exists()
        else (None, [])
    )
    out = root / config.output_dir / video_id
    probe = probe_video(path)
    scale = config.sampling.analysis_scale
    flow_cfg = config.flow

    # --- pasada 1: tiempo, metricas y flujo -----------------------------------
    from .reader import DecodeReport

    report = DecodeReport()
    frames: list[DecodedFrame] = []
    metrics: list[FrameMetrics] = []
    flows: list[FlowEstimate] = []
    history: dict[int, tuple[np.ndarray, np.ndarray, float | None]] = {}
    for frame in iter_frames(path, image_format="bgr24", report=report):
        small = cv2.resize(frame.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        previous = history.get(frame.index - 1)
        metrics.append(
            compute_frame_metrics(
                frame.index, frame.timestamp_s, small, gray, previous[1] if previous else None
            )
        )
        source = history.get(frame.index - flow_cfg.frame_step)
        if source is not None:
            dt = (
                frame.timestamp_s - source[2]
                if frame.timestamp_s is not None and source[2] is not None
                else None
            )
            estimate, _ = estimate_flow(
                source[1],
                gray,
                source[0],
                flow_cfg,
                config.roi,
                scale,
                frame.index - flow_cfg.frame_step,
                frame.index,
                dt,
            )
            flows.append(estimate)
        history[frame.index] = (small, gray, frame.timestamp_s)
        history.pop(frame.index - flow_cfg.frame_step - 1, None)
        frames.append(
            DecodedFrame(
                frame.index,
                frame.pts,
                frame.dts,
                frame.time_base,
                frame.timestamp_s,
                frame.key_frame,
                frame.pict_type,
                frame.width,
                frame.height,
                frame.rotation_deg,
            )
        )
    if not frames:
        raise VideoDecodeError(f"{path}: no se decodifico ningun cuadro ({report.error})")

    timestamps = analyze_timestamps(
        [frame.timestamp_s for frame in frames],
        fps_nominal=probe.fps_nominal,
        gap_factor=config.timestamps.gap_factor,
        allow_index_estimate=config.timestamps.allow_index_fps_estimate,
    )
    times = list(timestamps.timeline_s)
    if assessment is not None:
        validate_against_clip(
            assessment,
            observations,
            ClipFacts(
                video_id=video_id,
                sha256=digest,
                frame_count=len(frames),
                width_px=frames[0].width,
                height_px=frames[0].height,
                timeline_s=timestamps.timeline_s,
                timestamp_source=timestamps.source,
                eof_reached=report.eof_status == "eof_reached",
            ),
            config.timestamps.annotation_tolerance_s,
        )

    # --- seleccion de muestras -------------------------------------------------
    by_target = {flow.index_to: flow for flow in flows}
    brightness = [m.brightness_mean for m in metrics]
    jumps = [None] + [abs(b - a) for a, b in itertools.pairwise(brightness)]
    criteria = [
        ("low_sharpness", [m.sharpness_laplacian_var for m in metrics], lambda v: v),
        ("brightness_jump", jumps, lambda v: -v),
        ("high_vegetation", [m.vegetation_fraction for m in metrics], lambda v: -v),
        ("low_vegetation", [m.vegetation_fraction for m in metrics], lambda v: v),
        (
            "flow_dispersion",
            [(by_target[i].mad_dv_px if i in by_target else None) for i in range(len(frames))],
            lambda v: -v,
        ),
        (
            "flow_few_valid",
            [
                (float(by_target[i].valid_soil) if i in by_target else None)
                for i in range(len(frames))
            ],
            lambda v: v,
        ),
    ]
    samples = select_samples(
        len(frames),
        config.sampling.uniform_count,
        criteria,
        config.sampling.problem_count_per_metric,
        config.sampling.min_separation_frames,
    )
    sample_reasons = {choice.index: choice.reasons for choice in samples}
    pair_targets = [flow.index_to for flow in flows]
    example_targets = sorted(
        {
            pair_targets[int(i)]
            for i in np.linspace(0, len(pair_targets) - 1, flow_cfg.example_count).round()
        }
        if pair_targets and flow_cfg.example_count
        else set()
    )
    needed_small = {t - flow_cfg.frame_step for t in example_targets} | set(example_targets)

    # --- pasada 2: muestras a resolucion completa y ejemplos de flujo ----------
    frames_dir = out / "frames"
    flow_dir = out / "flow_examples"
    for directory in (frames_dir, flow_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for stale in directory.glob("*"):
            stale.unlink()
    jpeg = [int(cv2.IMWRITE_JPEG_QUALITY), config.sampling.jpeg_quality]
    sample_rows: list[dict[str, Any]] = []
    thumbs: list[np.ndarray] = []
    small_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    example_rows: list[dict[str, Any]] = []
    for frame in iter_frames(path, image_format="bgr24"):
        if frame.index in sample_reasons:
            reasons = sample_reasons[frame.index]
            name = f"frame_{frame.index:06d}.jpg"
            labeled = _label_band(
                frame.image,
                [
                    f"{video_id}  sha256:{digest[:12]}  frame {frame.index}  pts {frame.pts}"
                    f"  t={times[frame.index]:.4f} s ({timestamps.source})",
                    "motivo: " + ", ".join(reasons) + "   ejes: u derecha, v abajo; sin rotacion",
                ],
            )
            cv2.imwrite(str(frames_dir / name), labeled, jpeg)
            thumb = cv2.resize(frame.image, (384, 216), interpolation=cv2.INTER_AREA)
            cv2.putText(
                thumb,
                f"#{frame.index} t={times[frame.index]:.2f}s",
                (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            thumbs.append(thumb)
            sample_rows.append(
                {
                    "frame_index": frame.index,
                    "pts": frame.pts,
                    "timestamp_s": times[frame.index],
                    "timestamp_method": timestamps.source,
                    "reasons": ";".join(reasons),
                    "image": _relative(frames_dir / name, root),
                }
            )
        if frame.index in needed_small:
            small = cv2.resize(frame.image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            small_cache[frame.index] = (frame.image, small, cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        if frame.index in example_targets:
            src = small_cache.get(frame.index - flow_cfg.frame_step)
            if src is not None:
                estimate, vectors = estimate_flow(
                    src[2],
                    small_cache[frame.index][2],
                    src[1],
                    flow_cfg,
                    config.roi,
                    scale,
                    frame.index - flow_cfg.frame_step,
                    frame.index,
                    None,
                )
                stem = f"flow_{frame.index - flow_cfg.frame_step:06d}_{frame.index:06d}"
                full_box = roi_pixels(config.roi, frame.width, frame.height)
                drawing = _label_band(
                    _draw_vectors(src[0], vectors, full_box),
                    [
                        f"{video_id} flujo {estimate.index_from}->{estimate.index_to}: "
                        f"mediana suelo du={estimate.median_du_px} dv={estimate.median_dv_px} px "
                        f"direccion={estimate.direction}",
                        "verde: suelo valido  magenta: vegetacion valida  "
                        "rojo: rechazado (ida/vuelta)",
                    ],
                )
                cv2.imwrite(str(flow_dir / f"{stem}.jpg"), drawing, jpeg)
                _write_csv(
                    flow_dir / f"{stem}.csv",
                    [
                        {
                            "u_px": float(o[0]),
                            "v_px": float(o[1]),
                            "du_px": float(d[0]),
                            "dv_px": float(d[1]),
                            "valid": bool(ok),
                            "on_vegetation": bool(veg),
                            "fb_error_analysis_px": float(err),
                        }
                        for o, d, ok, veg, err in zip(
                            vectors.origins,
                            vectors.displacements,
                            vectors.valid,
                            vectors.on_vegetation,
                            vectors.fb_error,
                            strict=True,
                        )
                    ],
                    [
                        "u_px",
                        "v_px",
                        "du_px",
                        "dv_px",
                        "valid",
                        "on_vegetation",
                        "fb_error_analysis_px",
                    ],
                )
                example_rows.append(
                    {
                        **estimate.to_dict(),
                        "affine_fit": affine_flow_fit(vectors),
                        "image": _relative(flow_dir / f"{stem}.jpg", root),
                    }
                )
        small_cache = {
            k: v for k, v in small_cache.items() if k >= frame.index - flow_cfg.frame_step
        }

    if thumbs:
        columns = 4
        while len(thumbs) % columns:
            thumbs.append(np.zeros_like(thumbs[0]))
        rows = [np.hstack(thumbs[k : k + columns]) for k in range(0, len(thumbs), columns)]
        cv2.imwrite(str(out / "contact_sheet.jpg"), np.vstack(rows), jpeg)

    # --- tablas -----------------------------------------------------------------
    _write_csv(
        out / "timestamps.csv",
        [
            {
                "frame_index": f.index,
                "pts": f.pts,
                "dts": f.dts,
                "time_base": str(f.time_base) if f.time_base is not None else None,
                "timestamp_observed_s": f.timestamp_s,
                "timestamp_s": times[f.index],
                "timestamp_method": timestamps.source,
                "interval_s": (times[f.index] - times[f.index - 1]) if f.index else None,
                "key_frame": f.key_frame,
                "pict_type": f.pict_type,
                "width_px": f.width,
                "height_px": f.height,
            }
            for f in frames
        ],
        [
            "frame_index",
            "pts",
            "dts",
            "time_base",
            "timestamp_observed_s",
            "timestamp_s",
            "timestamp_method",
            "interval_s",
            "key_frame",
            "pict_type",
            "width_px",
            "height_px",
        ],
    )
    _write_csv(
        out / "frame_metrics.csv",
        [m.to_dict() for m in metrics],
        list(FrameMetrics.__dataclass_fields__),
    )
    _write_csv(
        out / "flow.csv", [f.to_dict() for f in flows], list(FlowEstimate.__dataclass_fields__)
    )
    content_repeats = detect_content_repeats(metrics, config.timestamps.content_repeat_ratio)
    _write_csv(
        out / "timestamp_anomalies.csv",
        [*timestamps.anomalies, *content_repeats],
        [
            "kind",
            "frame_index",
            "after_index",
            "before_index",
            "interval_s",
            "at_s",
            "mean_abs_diff",
            "threshold",
        ],
    )
    _write_csv(
        out / "samples.csv",
        sample_rows,
        ["frame_index", "pts", "timestamp_s", "timestamp_method", "reasons", "image"],
    )

    flow_summary = _flow_summary(flows, scale, config)
    _plot_flow(out / "flow_timeseries.png", flows, times, video_id)

    # --- registro del manifiesto ------------------------------------------------
    versions = backend_versions()
    rotation = frames[0].rotation_deg
    automated_motion = ApparentMotion(flow_summary["dominant_direction"])
    record = VideoManifestRecord(
        video_id=video_id,
        relative_path=_relative(path, root),
        sha256=digest,
        size_bytes=probe.size_bytes,
        session_id=assessment.session_id if assessment else None,
        site_id=assessment.site_id if assessment else None,
        container=probe.container,
        codec=probe.codec,
        width_px=probe.width_px,
        height_px=probe.height_px,
        rotation_deg=rotation,
        duration_reported_s=probe.duration_reported_s,
        duration_observed_s=timestamps.duration_observed_s,
        frame_count_reported=probe.frame_count_reported,
        frame_count_decoded=report.frames_decoded,
        fps_nominal=probe.fps_nominal,
        fps_effective=timestamps.fps_effective,
        timestamp_source=TimestampSource(timestamps.source),
        time_base=probe.time_base,
        timestamp_anomalies_ref=_relative(out / "timestamp_anomalies.csv", root),
        stem_base_visibility=assessment.stem_base_visibility if assessment else "unknown",
        apparent_motion=(
            assessment.reconcile_motion(automated_motion) if assessment else automated_motion
        ),
        mounting_status=assessment.mounting_status if assessment else "unknown",
        scale_status=assessment.scale.status if assessment else "unknown",
        scale_evidence_ref=assessment.scale.evidence_ref if assessment else None,
        homography_feasibility=assessment.homography.feasibility if assessment else "unknown",
        observation_timing=assessment.observation_timing if assessment else "unknown",
        suitability=assessment.suitability if assessment else "pending",
        suitability_reason=(
            assessment.suitability_reason
            if assessment
            else "evaluacion manual pendiente: no se clasifica sin inspeccion visual"
        ),
        evidence_ref=(
            ";".join(
                [_relative(out / "characterization.json", root)]
                + ([_relative(Path(manual_assessment_path), root)] if assessment else [])
            )
        ),
        decoder_version=(
            f"pyav {versions['pyav']}; libavcodec {versions.get('libavcodec', 'unknown')}; "
            f"libavformat {versions.get('libavformat', 'unknown')}"
        ),
        config_hash=config_hash(config),
    )

    manual: dict[str, Any] | None = None
    if assessment is not None:
        events = (
            [event.to_dict() for event in crossing_events(observations, assessment.crossing_line)]
            if assessment.crossing_line is not None
            else []
        )
        _write_csv(
            out / "crossing_events.csv",
            events,
            CROSSING_COLUMNS,
        )
        manual = {
            "source": "manual_visual",
            "assessment_ref": _relative(Path(manual_assessment_path), root),  # type: ignore[arg-type]
            "observations": len(observations),
            "plants_observed": len({o.plant_id for o in observations}),
            "video_sha256": assessment.video_sha256,
            "stem_observations_sha256": assessment.stem_observations_sha256,
            "tracked_plants_ground_contact": ManualAssessment.tracked_plants(observations),
            "tracked_plants_stem_axis_proxy": ManualAssessment.tracked_plants(
                observations, ObservedTarget.STEM_AXIS_PROXY
            ),
            "crossing_events_scope": (
                "cruce interpolado de puntos anotados en pixeles con la linea virtual; "
                "no es instante fisico de intervencion ni evento de base medido"
            ),
            "crossing_line": (
                assessment.crossing_line.model_dump() if assessment.crossing_line else None
            ),
            "crossing_events": events,
            "flow_manual_checks": [
                {"check_id": check.check_id, **check.agreement()}
                for check in assessment.flow_manual_checks
            ],
            "flow_manual_checks_scope": (
                "same_vertical_sign solo indica coherencia de direccion; within_tolerance es "
                "el unico criterio de magnitud, con la tolerancia declarada antes de comparar"
            ),
            "mounting_status": assessment.mounting_status.value,
            "replay_suitability": assessment.replay_suitability.value,
            "anticipation_validation": assessment.anticipation_validation,
            "intervals": [interval.model_dump(mode="json") for interval in assessment.intervals],
        }

    results = {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "manual": manual,
        "video_id": video_id,
        "sha256": digest,
        "probe": probe.to_dict() | {"path": _relative(path, root)},
        "rotation": {
            "value_deg": rotation,
            "source": "frame.rotation de PyAV (matriz de visualizacion; 0 si no se declara)",
            "display_rotation_applied": False,
        },
        "decode": report.to_dict(),
        "timestamps": timestamps.to_dict(),
        "content_repeats": content_repeats,
        "content_repeat_note": _content_repeat_note(content_repeats, probe),
        "image_axes": "u hacia la derecha, v hacia abajo, pixeles del cuadro decodificado",
        "roi_px": dict(
            zip(
                ("u0", "v0", "u1", "v1"),
                roi_pixels(config.roi, probe.width_px or 0, probe.height_px or 0),
                strict=True,
            )
        ),
        "flow_summary": flow_summary,
        "flow_examples": example_rows,
        "samples": sample_rows,
        "backend": versions,
        "config_hash": record.config_hash,
        "config": config.model_dump(mode="json"),
        "manifest_record": record.to_dict(),
    }
    payload = {"results": results}
    (out / "characterization.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    _write_derived_manifest(out, root)
    return CharacterizationResult(
        video_id,
        digest,
        probe,
        report.to_dict(),
        timestamps,
        flow_summary,
        sample_rows,
        out,
        record,
    )


def _flow_summary(flows: list[FlowEstimate], scale: float, config: VideoConfig) -> dict[str, Any]:
    from collections import Counter

    counts = Counter(flow.direction for flow in flows)
    usable = [f for f in flows if f.median_dv_px is not None]
    dv = np.array([f.median_dv_px for f in usable], dtype=float)
    du = np.array([f.median_du_px for f in usable], dtype=float)
    dominant = "unknown"
    if flows:
        label, count = counts.most_common(1)[0]
        dominant = label if count / len(flows) >= config.flow.direction_dominance else "mixed"
    summary: dict[str, Any] = {
        "pairs": len(flows),
        "pairs_with_estimate": len(usable),
        "direction_counts": dict(sorted(counts.items())),
        "dominant_direction": dominant,
        "dominance_rule": (
            f"una etiqueta es dominante si >= {config.flow.direction_dominance} de los pares "
            "la presentan; en otro caso 'mixed'"
        ),
        "units": "pixeles del cuadro original por paso de frame_step",
        "frame_step": config.flow.frame_step,
        "analysis_scale": scale,
        "vectors_source": "solo suelo (mascara ExG excluye vegetacion)",
    }
    if usable:
        summary.update(
            {
                "median_du_px": float(np.median(du)),
                "median_dv_px": float(np.median(dv)),
                "p05_dv_px": float(np.percentile(dv, 5)),
                "p95_dv_px": float(np.percentile(dv, 95)),
                "p05_du_px": float(np.percentile(du, 5)),
                "p95_du_px": float(np.percentile(du, 95)),
                "std_dv_px": float(np.std(dv)),
                "std_du_px": float(np.std(du)),
                "median_valid_soil": float(np.median([f.valid_soil for f in usable])),
                "median_rejected": float(np.median([f.rejected for f in usable])),
                "median_valid_vegetation": float(np.median([f.valid_vegetation for f in usable])),
                "jitter_dv_px": _jitter(dv),
                "jitter_du_px": _jitter(du),
                "jitter_method": (
                    "desviacion tipica de la serie menos su mediana movil de 15 pares; "
                    "indicador de vibracion/irregularidad, no medicion de vibracion"
                ),
            }
        )
    return summary


def _content_repeat_note(repeats: list[dict[str, Any]], probe: VideoProbe) -> str | None:
    """Explicacion prudente de candidatos a repeticion, con los metadatos reales."""
    if not repeats:
        return None
    markers = {
        key: value
        for key, value in probe.container_metadata.items()
        if key.lower() in {"encoder", "te_is_reencode", "handler_name"}
    }
    return (
        f"{len(repeats)} cuadro(s) candidatos a imagen casi repetida respecto del anterior, "
        "con PTS regular. Es compatible con una repeticion introducida al codificar a "
        "frecuencia constante, pero no se certifica como duplicado de captura ni se "
        "infiere perdida de cuadros. Metadatos declarados por el contenedor (indicios de "
        f"procesamiento, no prueba de la cadena de adquisicion): {markers or 'ninguno'}."
    )


def _jitter(series: np.ndarray, window: int = 15) -> float | None:
    if series.size < window:
        return None
    half = window // 2
    padded = np.pad(series, half, mode="edge")
    smooth = np.array([np.median(padded[i : i + window]) for i in range(series.size)])
    return float(np.std(series - smooth))


def _plot_flow(
    path: Path, flows: list[FlowEstimate], times: list[float | None], video_id: str
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = [times[f.index_to] for f in flows]
    du = [f.median_du_px if f.median_du_px is not None else np.nan for f in flows]
    dv = [f.median_dv_px if f.median_dv_px is not None else np.nan for f in flows]
    soil = [f.valid_soil for f in flows]
    veg = [f.valid_vegetation for f in flows]
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    top.plot(t, dv, label="mediana dv (suelo)", color="#1f5fa6", linewidth=1)
    top.plot(t, du, label="mediana du (suelo)", color="#c2571a", linewidth=1)
    top.axhline(0, color="#888", linewidth=0.6)
    top.set_ylabel("px / par de cuadros")
    top.set_title(f"{video_id}: flujo aparente en imagen (u derecha, v abajo; sin escala)")
    top.legend(loc="lower right", fontsize=8)
    bottom.plot(t, soil, label="vectores validos de suelo", color="#2c7a3f", linewidth=1)
    bottom.plot(t, veg, label="vectores validos de vegetacion", color="#9b3fa0", linewidth=1)
    bottom.set_xlabel("tiempo [s] (PTS)")
    bottom.set_ylabel("vectores")
    bottom.legend(loc="lower right", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _write_derived_manifest(out: Path, root: Path) -> None:
    """Hash de cada derivado para recuperar/verificar archivos fuera de control de versiones."""
    entries = {}
    for item in sorted(out.rglob("*")):
        # Se excluyen el propio manifiesto y los registros de ejecucion (fecha variable).
        if (
            item.is_file()
            and item.name != "derived_manifest.json"
            and not item.name.startswith("run_")
        ):
            entries[_relative(item, root)] = {
                "sha256": sha256_file(item),
                "size_bytes": item.stat().st_size,
            }
    (out / "derived_manifest.json").write_text(
        json.dumps(entries, indent=2, sort_keys=True), encoding="utf-8"
    )
