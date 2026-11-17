"""Diagnostico V1 (orden V1 §3 paso 1) sobre una corrida M0-P ya generada.

Lee el sidecar y el store de `render_route_video.py`; no ejecuta percepcion ni planner.
Responde con datos por que la ruta bordea la hilera en lugar de entrar a los gaps:

- por cada par de plantas consecutivas de la hilera, cuanto eje libre queda para la
  huella de la herramienta entre sus exclusiones (negativo = la huella no cabe);
- que fraccion de gaps cabria bajo supuestos alternativos de herramienta y proteccion
  (tabla de escenarios; es aritmetica sobre el mapa, no una ruta certificada);
- causas de los cuadros sin ruta vigente;
- una hoja de contacto del video para revision visual.

EXPLORATORIO (`ground_assumed`): todas las longitudes dependen de la escala supuesta.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (nombre, semiancho lateral, semilargo longitudinal) de la huella sobre el eje, en metros.
TOOLS = (
    ("rectangulo 0.20x0.08 (M0-P)", 0.10, 0.04),
    ("rectangulo 0.10x0.04", 0.05, 0.02),
    ("disco d=0.06 (cota cuadrada)", 0.03, 0.03),
)
# (nombre, radio protegido fijo o None = copa detectada, margen de incertidumbre).
PROTECTIONS = (
    ("copa + 0.05 (M0-P)", None, 0.05),
    ("copa + 0.03", None, 0.03),
    ("tallo supuesto 0.07 + 0.05", 0.07, 0.05),
    ("tallo supuesto 0.07 + 0.03", 0.07, 0.03),
    ("tallo supuesto 0.05 + 0.03", 0.05, 0.03),
)


def blocked_interval(axis_x, center, radius, half_x, half_y):
    """Intervalo de `y` del centro de la huella (sobre el eje) que toca el circulo."""
    dx = max(abs(center[0] - axis_x) - half_x, 0.0)
    if dx >= radius:
        return None
    dy = math.sqrt(radius * radius - dx * dx) + half_y
    return center[1] - dy, center[1] + dy


def free_length(axis_x, a, b, radius_a, radius_b, half_x, half_y):
    """Largo de eje donde la huella cabe entre dos plantas consecutivas (puede ser < 0)."""
    ia = blocked_interval(axis_x, a, radius_a, half_x, half_y)
    ib = blocked_interval(axis_x, b, radius_b, half_x, half_y)
    low = ia[1] if ia else a[1]
    high = ib[0] if ib else b[1]
    return high - low


def load_run(sidecar: Path):
    """Cabecera y registros de cuadro de un sidecar."""
    with sidecar.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle]
    return records[0], [r for r in records if r.get("record") == "frame"]


def run_tool_half_extents(header: dict, root: Path) -> tuple[float, float]:
    """Semiancho lateral y semilargo longitudinal de la herramienta que uso la corrida."""
    import yaml

    tool = yaml.safe_load((root / header["profile"]["configs"]["tool"]).read_text("utf-8"))["tool"]
    return 0.5 * tool["length_x_m"], 0.5 * tool["width_y_m"]


def analyse(sidecar: Path, root: Path = ROOT) -> dict:
    store = sidecar.with_name(sidecar.name.replace(".sidecar.jsonl", ".store"))
    header, frames = load_run(sidecar)
    run_tool = run_tool_half_extents(header, root)
    pairs: dict[tuple[str, str], list] = defaultdict(list)
    scenario_hits = Counter()
    instances = 0
    protection_radii, spacings = [], []
    maps: dict[str, dict] = {}
    for record in frames:
        axis = record.get("row_axis")
        if "map" not in record or not axis:
            continue
        if record["map"] not in maps:
            maps[record["map"]] = json.loads((store / record["map"]).read_text("utf-8"))
        axis_x = axis["task_x_m"]
        plants = sorted(maps[record["map"]]["exclusions"], key=lambda e: e["center"]["y"])
        outcome = record["route_kind"] if record["status"] == "vigente" else record["status"]
        for first, second in pairwise(plants):
            a = (first["center"]["x"], first["center"]["y"])
            b = (second["center"]["x"], second["center"]["y"])
            instances += 1
            spacings.append(b[1] - a[1])
            protection_radii.extend([first["protection_radius"], second["protection_radius"]])
            current = free_length(
                axis_x, a, b, first["total_radius"], second["total_radius"], *run_tool
            )
            pairs[(first["plant_id"], second["plant_id"])].append((b[1] - a[1], current, outcome))
            for tool_name, half_x, half_y in TOOLS:
                for prot_name, fixed, margin in PROTECTIONS:
                    ra = (fixed if fixed is not None else first["protection_radius"]) + margin
                    rb = (fixed if fixed is not None else second["protection_radius"]) + margin
                    if free_length(axis_x, a, b, ra, rb, half_x, half_y) > 0.0:
                        scenario_hits[(tool_name, prot_name)] += 1
    pair_rows = []
    for (id_a, id_b), values in pairs.items():
        pair_rows.append(
            {
                "pair": f"{id_a}-{id_b}",
                "frames": len(values),
                "spacing_m": statistics.median(v[0] for v in values),
                "free_axis_m": statistics.median(v[1] for v in values),
                "outcomes": dict(Counter(v[2] for v in values)),
            }
        )
    pair_rows.sort(key=lambda row: -row["frames"])
    fits = [row for row in pair_rows if row["free_axis_m"] > 0.0]
    return {
        "frames": len(frames),
        "statuses": dict(Counter(r["status"] for r in frames)),
        "route_kinds": dict(Counter(r["route_kind"] for r in frames if r["status"] == "vigente")),
        "no_route_causes": dict(
            Counter(r["reason"].split(";")[0][:90] for r in frames if r["status"] != "vigente")
        ),
        "pair_instances": instances,
        "spacing_quartiles_m": statistics.quantiles(spacings, n=4),
        "canopy_protection_radius_quartiles_m": statistics.quantiles(protection_radii, n=4),
        "pairs_total": len(pair_rows),
        "pairs_where_tool_fits_on_axis": len(fits),
        "scenarios": [
            {
                "tool": tool_name,
                "protection": prot_name,
                "gap_instances_fitting": scenario_hits[(tool_name, prot_name)] / instances,
            }
            for tool_name, _, _ in TOOLS
            for prot_name, _, _ in PROTECTIONS
        ],
        "pairs": pair_rows,
    }


def contact_sheet(video: Path, output: Path, count: int, columns: int) -> None:
    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for k in range(count):
        index = int((k + 0.5) * total / count)
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, image = capture.read()
        if not ok:
            continue
        scale = 480 / image.shape[1]
        tile = cv2.resize(image, (480, int(image.shape[0] * scale)))
        cv2.putText(
            tile, f"cuadro {index}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
        tiles.append(tile)
    capture.release()
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k : k + columns]) for k in range(0, len(tiles), columns)]
    cv2.imwrite(str(output), np.vstack(rows))


def write_report(result: dict, path: Path) -> None:
    lines = [
        "# Diagnostico V1 de la corrida M0-P",
        "",
        "> EXPLORATORIO · `ground_assumed` · longitudes en metros supuestos.",
        "",
        f"- Cuadros: {result['frames']}; estados: {result['statuses']}",
        f"- Tipos de ruta vigente: {result['route_kinds']}",
        "- Separacion entre plantas consecutivas (cuartiles, m): "
        + ", ".join(f"{q:.3f}" for q in result["spacing_quartiles_m"]),
        "- Radio protegido por planta, sin incertidumbre (cuartiles, m): "
        + ", ".join(f"{q:.3f}" for q in result["canopy_protection_radius_quartiles_m"]),
        f"- Pares de plantas distintos: {result['pairs_total']}; la huella de la corrida cabe en "
        f"el eje en {result['pairs_where_tool_fits_on_axis']}.",
        "",
        "## Fraccion de gaps (par x cuadro) donde la huella cabe sobre el eje",
        "",
        "| Herramienta | Proteccion | Cabe |",
        "|---|---|---:|",
    ]
    for row in result["scenarios"]:
        lines.append(
            f"| {row['tool']} | {row['protection']} | {100 * row['gap_instances_fitting']:.0f}% |"
        )
    lines += ["", "## Cuadros sin ruta vigente", ""]
    for cause, count in sorted(result["no_route_causes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {count}: {cause}")
    lines += [
        "",
        "## Pares de plantas (mediana por par)",
        "",
        "| Par | Cuadros | Separacion m | Eje libre m | Resultados |",
        "|---|---:|---:|---:|---|",
    ]
    for row in result["pairs"]:
        lines.append(
            f"| {row['pair']} | {row['frames']} | {row['spacing_m']:.3f} | "
            f"{row['free_axis_m']:.3f} | {row['outcomes']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="video de la corrida, para la hoja de contacto")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sheet-frames", type=int, default=12)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyse(args.sidecar)
    (args.output_dir / "diagnosis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(result, args.output_dir / "diagnosis.md")
    if args.video:
        contact_sheet(args.video, args.output_dir / "contact_sheet.jpg", args.sheet_frames, 4)
    print(f"diagnostico escrito en {args.output_dir}")


if __name__ == "__main__":
    main()
