#!/usr/bin/env python3
"""Evidencia visual de la tabla manual de bases de tallo (E0), reproducible desde el CSV.

Para cada observacion genera un recorte ampliado con rejilla en pixeles originales y,
si la base esta localizada, un circulo del radio de incertidumbre declarado. Para
cada planta compone una tira con todos sus cuadros, y para cada cruce dibuja la
linea virtual sobre el cuadro completo. No anota nada nuevo: solo representa lo que
dice la tabla, para que el revisor pueda contrastarlo con la imagen.

Antes de escribir nada valida la evaluacion contra el clip decodificado (ID, SHA-256,
cuadros, tiempos, posiciones), con la misma funcion que el caracterizador. Codigos de
salida: 0 exito, 2 evaluacion invalida o incompatible con el video.

    python scripts/render_stem_observations.py \\
        --assessment experiments/phase0/moringa_sin_ai/manual_assessment.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from precision_weeder.core.config import VideoFile, load_yaml_mapping, validate_config_payload
from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.video.characterize import video_id_from_path
from precision_weeder.video.observations import (
    BaseVisibility,
    ClipFacts,
    ManualAssessment,
    validate_against_clip,
)
from precision_weeder.video.reader import decode_sequential, iter_frames, probe_video, sha256_file
from precision_weeder.video.timing import analyze_timestamps

REPO_ROOT = Path(__file__).resolve().parents[1]
ZOOM = 3
HALF = 60


def _crop(image: np.ndarray, uc: float, vc: float, label: str, obs) -> np.ndarray:
    u0, v0 = int(uc - HALF), int(vc - HALF)
    tile = np.zeros((2 * HALF, 2 * HALF, 3), np.uint8)
    su0, sv0 = max(u0, 0), max(v0, 0)
    su1, sv1 = min(u0 + 2 * HALF, image.shape[1]), min(v0 + 2 * HALF, image.shape[0])
    tile[sv0 - v0 : sv1 - v0, su0 - u0 : su1 - u0] = image[sv0:sv1, su0:su1]
    big = cv2.resize(tile, None, fx=ZOOM, fy=ZOOM, interpolation=cv2.INTER_CUBIC)
    for u in range((u0 // 10 + 1) * 10, u0 + 2 * HALF, 10):
        strong = u % 50 == 0
        cv2.line(
            big,
            ((u - u0) * ZOOM, 0),
            ((u - u0) * ZOOM, big.shape[0]),
            (0, 255, 255) if strong else (0, 110, 110),
            1,
        )
        if strong:
            cv2.putText(big, str(u), ((u - u0) * ZOOM + 2, 14), 0, 0.45, (0, 255, 255), 1)
    for v in range((v0 // 10 + 1) * 10, v0 + 2 * HALF, 10):
        strong = v % 50 == 0
        cv2.line(
            big,
            (0, (v - v0) * ZOOM),
            (big.shape[1], (v - v0) * ZOOM),
            (0, 255, 255) if strong else (0, 110, 110),
            1,
        )
        if strong:
            cv2.putText(big, str(v), (2, (v - v0) * ZOOM - 3), 0, 0.45, (0, 255, 255), 1)
    if obs.base_visibility is not BaseVisibility.HIDDEN:
        center = (int((obs.u_px - u0) * ZOOM), int((obs.v_px - v0) * ZOOM))
        cv2.circle(big, center, int(obs.uncertainty_px * ZOOM), (0, 0, 255), 2)
        cv2.drawMarker(big, center, (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
    cv2.putText(big, label, (4, big.shape[0] - 8), 0, 0.55, (0, 0, 255), 2)
    return big


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessment", type=Path, required=True)
    parser.add_argument("--video", type=Path, default=REPO_ROOT / "resources/moringa_sin_ai.mp4")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs/video.yaml")
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Raiz de rutas relativas.")
    args = parser.parse_args(argv)
    try:
        policy = validate_config_payload(
            VideoFile, load_yaml_mapping(args.config), str(args.config)
        ).video.timestamps
        assessment, observations = ManualAssessment.load(args.assessment, args.root)
        if not args.video.is_file():
            raise PrecisionWeederError(f"video inexistente: {args.video}")
        frames, report = decode_sequential(args.video)
        if not frames:
            raise PrecisionWeederError(f"{args.video}: no se decodifico ningun cuadro")
        timing = analyze_timestamps(
            [frame.timestamp_s for frame in frames],
            fps_nominal=probe_video(args.video).fps_nominal,
            gap_factor=policy.gap_factor,
            allow_index_estimate=policy.allow_index_fps_estimate,
        )
        validate_against_clip(
            assessment,
            observations,
            ClipFacts(
                video_id=video_id_from_path(args.video),
                sha256=sha256_file(args.video),
                frame_count=len(frames),
                width_px=frames[0].width,
                height_px=frames[0].height,
                timeline_s=timing.timeline_s,
                timestamp_source=timing.source,
                eof_reached=report.eof_status == "eof_reached",
            ),
            policy.annotation_tolerance_s,
        )
    except PrecisionWeederError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    out = args.assessment.parent / "manual_evidence"
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.jpg"):
        stale.unlink()
    by_frame: dict[int, list] = {}
    for obs in observations:
        by_frame.setdefault(obs.frame_index, []).append(obs)
    # Las bases ocultas no tienen posicion en la tabla: no se dibujan (no se inventan).
    strips: dict[str, list[np.ndarray]] = {}
    line = assessment.crossing_line
    for frame in iter_frames(args.video, image_format="bgr24"):
        for obs in by_frame.get(frame.index, []):
            if obs.base_visibility is BaseVisibility.HIDDEN:
                continue
            label = (
                f"{obs.plant_id} #{obs.frame_index} {obs.observed_target.value} "
                f"{obs.base_visibility.value} {obs.confidence.value}"
            )
            strips.setdefault(obs.plant_id, []).append(
                _crop(frame.image, obs.u_px, obs.v_px, label, obs)
            )
            if line is not None:
                full = frame.image.copy()
                cv2.line(
                    full,
                    (int(line.u_min_px), int(line.v_px)),
                    (int(line.u_max_px) - 1, int(line.v_px)),
                    (255, 255, 0),
                    2,
                )
                cv2.circle(
                    full, (int(obs.u_px), int(obs.v_px)), int(obs.uncertainty_px), (0, 0, 255), 3
                )
                cv2.putText(
                    full,
                    f"{obs.plant_id} #{obs.frame_index}  linea virtual v={line.v_px:g} px",
                    (20, 40),
                    0,
                    1.0,
                    (0, 255, 255),
                    2,
                )
                cv2.imwrite(
                    str(out / f"full_{obs.plant_id}_{obs.frame_index:06d}.jpg"),
                    cv2.resize(full, (960, 540)),
                    [cv2.IMWRITE_JPEG_QUALITY, 90],
                )
    for plant, tiles in sorted(strips.items()):
        cv2.imwrite(
            str(out / f"strip_{plant}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
    targets = sorted({o.observed_target.value for o in observations if o.u_px is not None})
    print(
        f"evidencia escrita en {out} ({len(strips)} plantas con punto anotado; "
        f"objetivos: {', '.join(targets) or 'ninguno'})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
