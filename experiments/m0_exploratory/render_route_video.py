#!/usr/bin/env python3
"""CLI M0-P: video de entrada -> mismo video con la ruta geometrica propuesta.

    .venv/bin/python experiments/m0_exploratory/render_route_video.py \\
        --input resources/moringa_sin_ai.mp4 \\
        --profile experiments/m0_exploratory/profiles/real_right_row.yaml \\
        --output experiments/m0_exploratory/output/moringa_sin_ai_route.mp4

EXPLORATORIO (`ground_assumed`): escala supuesta, copa como proxy de planta y
mecanica no evaluada. Las observaciones vienen de la exportacion versionada del
proveedor declarada en el perfil (p. ej. YOLO ejecutado en su entorno separado con
`export_yolo_detections.py`); el planificador y el validador son los de E2 sin
cambios.

Codigos de salida:

- `0`: procesamiento completo con al menos un cuadro con propuesta vigente.
- `3`: procesamiento completo sin ninguna propuesta (video y diagnostico de abstencion).
- `2`: entrada incompatible, proveedor o configuracion invalidos; no se publica salida.

Un procesamiento incompleto nunca devuelve 0 ni deja el video como resultado final.
Estos codigos no autorizan ninguna ejecucion mecanica.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from m0p.observations import load_export  # noqa: E402
from m0p.pipeline import run_pipeline  # noqa: E402
from m0p.profile import load_profile  # noqa: E402

from precision_weeder.core.errors import PrecisionWeederError  # noqa: E402
from precision_weeder.video.reader import backend_versions  # noqa: E402

EXIT_OK, EXIT_INVALID, EXIT_NO_ROUTE = 0, 2, 3


def _parse(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M0-P: ruta exploratoria sobre video.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT, help="Raiz de rutas relativas.")
    parser.add_argument(
        "--export", type=Path, help="Exportacion de observaciones (por defecto, la del perfil)."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse(argv)
    root = args.root.resolve()
    try:
        profile, resolved = load_profile(args.profile, root)
        export = args.export or (root / profile.provider.export)
        if not export.is_file():
            raise PrecisionWeederError(f"exportacion de observaciones inexistente: {export}")
        if not args.input.is_file():
            raise PrecisionWeederError(f"video inexistente: {args.input}")
        observations = load_export(export)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary = run_pipeline(args.input, profile, resolved, observations, args.output, root)
    except PrecisionWeederError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INVALID
    report = {
        "executed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "command": ["main", *(argv or sys.argv[1:])],
        "python": platform.python_version(),
        "backend": backend_versions(),
        "profile": profile.profile_id,
        "summary": summary.to_dict(),
    }
    args.output.with_suffix(".run.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return EXIT_OK if summary.valid_frames else EXIT_NO_ROUTE


if __name__ == "__main__":
    sys.exit(main())
