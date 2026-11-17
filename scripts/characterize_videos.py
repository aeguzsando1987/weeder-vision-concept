#!/usr/bin/env python3
"""CLI de caracterizacion de videos (E0), capa delgada sobre `precision_weeder.video`.

Ejemplos:

    python scripts/characterize_videos.py --config configs/video.yaml
    python scripts/characterize_videos.py --config configs/video.yaml \\
        --input resources/moringa_sin_ai.mp4

Por cada clip escribe `experiments/phase0/<video_id>/` (tiempo, metricas, flujo,
muestras, hoja de contactos, `characterization.json`, `derived_manifest.json`) y
actualiza su fila en `data/video_manifest.csv`. Si existe
`experiments/phase0/<video_id>/manual_assessment.yaml`, la incorpora tras
validarla; si no, la fila queda `pending` con campos visuales `unknown`.

Codigos de salida: 0 exito, 2 entrada invalida o clip no caracterizable. El
registro de la ejecucion (fecha, comando, versiones) va en `run_<video_id>.json`,
separado de los resultados deterministas.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

from precision_weeder.core.config import VideoFile, load_yaml_mapping, validate_config_payload
from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.video.characterize import characterize_video, video_id_from_path
from precision_weeder.video.manifest import read_manifest_csv, write_manifest_csv
from precision_weeder.video.reader import backend_versions

REPO_ROOT = Path(__file__).resolve().parents[1]
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caracteriza videos reales (E0).")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "video.yaml")
    parser.add_argument(
        "--input",
        type=Path,
        help="Archivo o directorio; si se omite se usa video.source de la configuracion.",
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help="Raiz para rutas relativas.")
    return parser.parse_args(argv)


def _inputs(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    return [source]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.root.resolve()
    try:
        config = validate_config_payload(
            VideoFile, load_yaml_mapping(args.config), str(args.config)
        ).video
        source = args.input if args.input is not None else Path(config.source)
        source = source if source.is_absolute() else root / source
        videos = _inputs(source)
        if not videos:
            raise PrecisionWeederError(f"no hay videos en {source}")
        if len(videos) > 1 and config.expected_sha256 is not None:
            raise PrecisionWeederError(
                "expected_sha256 identifica un unico clip; quitelo para procesar un directorio"
            )
        manifest_path = root / config.manifest_path
        existing = (
            {r.video_id: r for r in read_manifest_csv(manifest_path)}
            if manifest_path.is_file()
            else {}
        )
        for video in videos:
            video_id = video_id_from_path(video)
            assessment = root / config.output_dir / video_id / "manual_assessment.yaml"
            result = characterize_video(video, config, root, assessment)
            existing[result.video_id] = result.record
            run = {
                "executed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(
                    timespec="seconds"
                ),
                "command": [sys.executable, *sys.argv] if argv is None else ["main", *argv],
                "config_path": str(args.config),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "backend": backend_versions(),
            }
            (result.output_dir / f"run_{video_id}.json").write_text(
                json.dumps(run, indent=2, sort_keys=True), encoding="utf-8"
            )
            summary = result.flow_summary
            print(
                f"{video_id}: {result.decode['frames_decoded']} cuadros "
                f"({result.decode['eof_status']}), fps_efectivo="
                f"{result.timestamps.fps_effective} ({result.timestamps.source}), "
                f"anomalias_pts={len(result.timestamps.anomalies)}, flujo="
                f"{summary['dominant_direction']} (dv~{summary.get('median_dv_px')} px), "
                f"muestras={len(result.samples)}, aptitud={result.record.suitability.value}"
            )
        write_manifest_csv(list(existing.values()), manifest_path)
        read_manifest_csv(manifest_path, base_dir=root)  # verificacion de ida y vuelta
    except PrecisionWeederError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
