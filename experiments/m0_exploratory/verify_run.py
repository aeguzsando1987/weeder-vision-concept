#!/usr/bin/env python3
"""Evidencia MP-07/MP-09 de una ejecucion M0-P sobre un video sin marcas de cuadro.

    .venv/bin/python experiments/m0_exploratory/verify_run.py \\
        --input resources/moringa_sin_ai.mp4 \\
        --output experiments/m0_exploratory/output/moringa_sin_ai_route.mp4 \\
        [--replay otra_ejecucion.sidecar.jsonl]

- **MP-07:** re-decodifica entrada y salida; compara numero, orden, resolucion,
  orientacion, PTS/base temporal cuadro a cuadro y el fin efectivo de presentacion
  (PTS + duracion del ultimo cuadro, dentro de un tick), y que cada cuadro de salida se
  parezca mas a su cuadro de entrada que a los vecinos (cuadros duplicados, perdidos
  o reordenados). El clip real no lleva marca de indice como los fixtures.
- **MP-09:** con `--replay`, compara el sidecar con el de otra ejecucion: cabecera y
  registros por cuadro deben ser identicos (el sidecar no lleva tiempo de pared).

Imprime un JSON con los resultados; sale con 1 si alguna comprobacion falla.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from precision_weeder.video.reader import iter_frames, sha256_file

THUMB = (96, 54)


def _thumbs(path: Path) -> tuple[list, np.ndarray]:
    frames, thumbs = [], []
    for frame in iter_frames(path, image_format="gray"):
        thumbs.append(cv2.resize(frame.image, THUMB, interpolation=cv2.INTER_AREA))
        frames.append(frame)
    return frames, np.asarray(thumbs, dtype=np.float32)


def _presentation_end(path: Path) -> tuple[int, Fraction]:
    """Fin efectivo de presentacion: PTS + duracion del ultimo cuadro decodificado."""
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        last = None
        for frame in container.decode(stream):
            last = frame
        return int(last.pts + last.duration), stream.time_base


def check_video(source: Path, output: Path) -> dict:
    src, src_t = _thumbs(source)
    out, out_t = _thumbs(output)
    failures = []
    (src_end, src_tb), (out_end, out_tb) = _presentation_end(source), _presentation_end(output)
    end_gap_s = abs(float(src_end * src_tb) - float(out_end * out_tb))
    if end_gap_s > float(out_tb):
        failures.append(f"fin de presentacion: entrada {src_end} != salida {out_end} ticks")
    if len(src) != len(out):
        failures.append(f"cuadros: entrada {len(src)} != salida {len(out)}")
    worst_dt, misplaced, ambiguous = 0.0, [], set()
    for i, (a, b) in enumerate(zip(src, out, strict=False)):
        if (a.width, a.height, a.rotation_deg) != (b.width, b.height, b.rotation_deg):
            failures.append(f"cuadro {i}: geometria distinta")
        if a.pts != b.pts or a.time_base != b.time_base:
            failures.append(f"cuadro {i}: pts/base {a.pts}/{a.time_base} -> {b.pts}/{b.time_base}")
        worst_dt = max(worst_dt, abs(a.timestamp_s - b.timestamp_s))
        own = float(np.abs(src_t[i] - out_t[i]).mean())
        neighbours = []
        for j in (i - 1, i + 1):
            if not 0 <= j < len(out):
                continue
            # Vecinos indistinguibles en la propia fuente (cuadro duplicado de origen) no
            # permiten decidir el orden: se cuentan aparte, no como error de la salida.
            if float(np.abs(src_t[i] - src_t[j]).mean()) <= own:
                ambiguous.add(tuple(sorted((i, j))))
                continue
            neighbours.append(j)
        if any(float(np.abs(src_t[i] - out_t[j]).mean()) <= own for j in neighbours):
            misplaced.append(i)
    if misplaced:
        failures.append(f"cuadros mas parecidos a un vecino: {misplaced[:20]}")
    tick = float(out[0].time_base) if out else 0.0
    if worst_dt > tick:
        failures.append(f"desviacion temporal {worst_dt} > 1 tick {tick}")
    return {
        "frames": [len(src), len(out)],
        "resolution": [out[0].width, out[0].height] if out else None,
        "time_base": str(out[0].time_base) if out else None,
        "max_timestamp_deviation_s": worst_dt,
        "last_frame_start_s": [src[-1].timestamp_s, out[-1].timestamp_s] if out else None,
        "presentation_end_s": [float(src_end * src_tb), float(out_end * out_tb)],
        "misplaced_frames": len(misplaced),
        "source_indistinguishable_pairs": sorted(ambiguous),
        "input_sha256": sha256_file(source),
        "failures": failures,
    }


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def check_replay(first: Path, second: Path) -> dict:
    a, b = _rows(first), _rows(second)
    differing = [i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y]
    failures = []
    if len(a) != len(b):
        failures.append(f"registros: {len(a)} != {len(b)}")
    if differing:
        failures.append(f"registros distintos (0=cabecera): {differing[:20]}")
    return {"records": [len(a), len(b)], "differing": len(differing), "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description="Evidencia MP-07/MP-09 de M0-P.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay", type=Path, help="Sidecar de una segunda ejecucion.")
    args = parser.parse_args()
    result = {"mp07": check_video(args.input, args.output)}
    if args.replay:
        sidecar = args.output.with_suffix(".sidecar.jsonl")
        result["mp09"] = check_replay(sidecar, args.replay)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if any(part["failures"] for part in result.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
