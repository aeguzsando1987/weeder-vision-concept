"""Evaluacion V1 por gap y de la consigna (orden V1, criterios V1-02 y V1-03).

Lee el sidecar, el store y el CSV de consigna de una corrida con patron `alternating`.
No ejecuta percepcion ni planner.

- Un **gap** es el par de plantas consecutivas de la hilera que la ruta rodea. Su
  resultado es el de la ultima ruta vigente que lo contiene (la decision con la que la
  herramienta llega a el): `alternado` si pasa las dos plantas por costados opuestos
  (cruza la linea plantada entre ellas) o `mismo_costado`.
- Es **admisible** si la huella de la herramienta cabe sobre el eje entre las dos
  exclusiones (aritmetica sobre el mapa de ese cuadro, no una ruta certificada).
- La consigna se resume sobre cuadros vigentes consecutivos.

EXPLORATORIO (`ground_assumed`): longitudes en metros supuestos.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from diagnose_route import ROOT, free_length, load_run, run_tool_half_extents
from m0p.route import chain_x_at

from precision_weeder.core.trajectory import TaskPath


def route_changes(frames: list[dict], store: Path, threshold_m: float = 0.05) -> dict:
    """Cuanto cambia la ruta dibujada POR DELANTE de la herramienta entre cuadros vigentes
    consecutivos del mismo segmento: maxima diferencia lateral en el tramo comun.

    Se mide tambien solo sobre la parte **anudada** de la ruta anterior (hasta su ultimo
    nudo): el cierre recto final es provisional y sustituirlo al entrar una planta nueva
    no es un cambio de una decision ya tomada."""
    cache: dict[str, TaskPath] = {}

    def chain(record):
        for leg in record["legs"]:
            if leg["route"] not in cache:
                payload = json.loads((store / leg["route"]).read_text("utf-8"))
                cache[leg["route"]] = TaskPath.from_dict(payload)
        return tuple(cache[leg["route"]] for leg in record["legs"])

    previous = None
    changes = []
    knotted = []

    def difference(now, before, low, high):
        ys = np.linspace(low, high, 200)
        return max(abs(chain_x_at(now, y) - chain_x_at(before, y)) for y in ys)

    for record in frames:
        if record["status"] != "vigente":
            previous = None
            continue
        now = chain(record)
        if previous is not None and previous[0] == record["segment_id"]:
            low = max(now[0].start_point[1], previous[1][0].start_point[1])
            high = min(now[-1].end_point[1], previous[1][-1].end_point[1])
            if high > low:
                changes.append(difference(now, previous[1], low, high))
            if min(high, previous[2]) > low:
                knotted.append(difference(now, previous[1], low, min(high, previous[2])))
        last = record["legs"][-1]
        knots_end = last["start_m"][1] if last["target"] == "cierre" else last["goal_m"][1]
        previous = (record["segment_id"], now, knots_end)
    return {
        "consecutive_pairs": len(changes),
        "threshold_m": threshold_m,
        "pairs_over_threshold": int(sum(change > threshold_m for change in changes)),
        "max_change_m": float(max(changes, default=0.0)),
        "knotted_pairs": len(knotted),
        "knotted_pairs_over_threshold": int(sum(change > threshold_m for change in knotted)),
        "knotted_max_change_m": float(max(knotted, default=0.0)),
    }


def evaluate(sidecar: Path, root: Path = ROOT) -> dict:
    store = sidecar.with_name(sidecar.name.replace(".sidecar.jsonl", ".store"))
    header, frames = load_run(sidecar)
    half_x, half_y = run_tool_half_extents(header, root)
    gaps: dict[tuple[str, str], dict] = {}
    causes: dict[str, str] = {}
    for record in frames:
        pattern = (record.get("planning") or {}).get("pattern")
        if record["status"] != "vigente" or not pattern:
            continue
        for event in pattern["omitted"]:
            causes[event["plants"][0]] = event["cause"]
        sides = pattern["plant_sides"]
        local_map = json.loads((store / record["map"]).read_text("utf-8"))
        plants = sorted(
            (e for e in local_map["exclusions"] if e["plant_id"] in sides),
            key=lambda e: e["center"]["y"],
        )
        axis_x = record["row_axis"]["task_x_m"]
        for first, second in pairwise(plants):
            a = (first["center"]["x"], first["center"]["y"])
            b = (second["center"]["x"], second["center"]["y"])
            free = free_length(
                axis_x, a, b, first["total_radius"], second["total_radius"], half_x, half_y
            )
            gaps[(first["plant_id"], second["plant_id"])] = {
                "pair": f"{first['plant_id']}-{second['plant_id']}",
                "last_frame": record["frame_index"],
                "spacing_m": b[1] - a[1],
                "free_axis_m": free,
                "admissible": free > 0.0,
                "alternated": sides[first["plant_id"]] != sides[second["plant_id"]],
            }
    rows = sorted(gaps.values(), key=lambda row: row["last_frame"])
    for row in rows:
        if not row["alternated"]:
            second = row["pair"].split("-")[1]
            default = "gap inadmisible: la huella no cabe sobre el eje"
            row["cause"] = causes.get(second) or (
                default if not row["admissible"] else "sin propuesta registrada"
            )
    admissible = [row for row in rows if row["admissible"]]
    result = {
        "frames": len(frames),
        "statuses": dict(Counter(r["status"] for r in frames)),
        "route_kinds": dict(Counter(r["route_kind"] for r in frames if r["status"] == "vigente")),
        "gaps_total": len(rows),
        "gaps_admissible": len(admissible),
        "gaps_admissible_alternated": sum(row["alternated"] for row in admissible),
        "gaps_inadmissible": len(rows) - len(admissible),
        "gaps_inadmissible_alternated": sum(
            row["alternated"] for row in rows if not row["admissible"]
        ),
        "omission_causes": dict(Counter(row["cause"] for row in rows if "cause" in row)),
        "route_changes_ahead": route_changes(frames, store),
        "gaps": rows,
    }
    setpoints = sidecar.with_name(sidecar.name.replace(".sidecar.jsonl", ".setpoints.csv"))
    if setpoints.is_file():
        with setpoints.open(encoding="utf-8", newline="") as handle:
            table = list(csv.DictReader(handle))
        steps = [abs(float(r["lateral_step_m"])) for r in table if r["lateral_step_m"]]
        rates = [abs(float(r["lateral_rate_m_s"])) for r in table if r["lateral_rate_m_s"]]
        offsets = [float(r["offset_from_row_axis_m"]) for r in table if r["state"] == "trabajando"]
        result["setpoint"] = {
            "rows": len(table),
            "states": dict(Counter(r["state"] for r in table)),
            "consecutive_pairs": len(steps),
            "lateral_step_m": _stats(steps),
            "lateral_rate_m_s": _stats(rates),
            "offset_from_row_axis_m": {"min": min(offsets), "max": max(offsets)},
        }
    return result


def _stats(values: list[float]) -> dict:
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "median": statistics.median(ordered),
        "p95": ordered[int(0.95 * (len(ordered) - 1))],
        "max": ordered[-1],
    }


def write_report(result: dict, path: Path) -> None:
    adm, alt = result["gaps_admissible"], result["gaps_admissible_alternated"]
    lines = [
        "# Evaluacion V1 por gap",
        "",
        "> EXPLORATORIO · `ground_assumed` · longitudes en metros supuestos.",
        "",
        f"- Cuadros: {result['frames']}; estados: {result['statuses']}",
        f"- Tipos de ruta vigente: {result['route_kinds']}",
        f"- Gaps (pares de plantas consecutivas rodeadas): {result['gaps_total']}",
        f"- Admisibles: {adm}; alternados: {alt}" + (f" ({100 * alt / adm:.0f}%)" if adm else ""),
        f"- Inadmisibles: {result['gaps_inadmissible']}; alternados pese a ello: "
        f"{result['gaps_inadmissible_alternated']}",
        "",
        "## Causas de los gaps no alternados",
        "",
    ]
    for cause, count in sorted(result["omission_causes"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {count}: {cause}")
    ahead = result["route_changes_ahead"]
    lines += [
        "",
        "## Estabilidad de la ruta por delante de la herramienta",
        "",
        f"- Pares de cuadros vigentes consecutivos: {ahead['consecutive_pairs']}",
        f"- Pares en que la ruta ya dibujada cambia mas de {ahead['threshold_m']} m: "
        f"{ahead['pairs_over_threshold']} (maximo {ahead['max_change_m']:.3f} m)",
        f"- De ellos, pares en que cambia la parte ya anudada a plantas (sin contar el cierre "
        f"recto provisional): {ahead['knotted_pairs_over_threshold']} de "
        f"{ahead['knotted_pairs']} (maximo {ahead['knotted_max_change_m']:.3f} m)",
    ]
    setpoint = result.get("setpoint")
    if setpoint:
        step, rate = setpoint["lateral_step_m"], setpoint["lateral_rate_m_s"]
        lines += [
            "",
            "## Consigna en la linea de herramienta",
            "",
            f"- Estados: {setpoint['states']}",
            f"- Pares de cuadros vigentes consecutivos: {setpoint['consecutive_pairs']}",
            f"- Paso lateral entre cuadros (m): mediana {step['median']:.4f}, "
            f"p95 {step['p95']:.4f}, max {step['max']:.4f}",
            f"- Tasa lateral exigida (m/s supuestos): mediana {rate['median']:.3f}, "
            f"p95 {rate['p95']:.3f}, max {rate['max']:.3f}",
            f"- Desvio respecto del eje (m): {setpoint['offset_from_row_axis_m']['min']:.3f} a "
            f"{setpoint['offset_from_row_axis_m']['max']:.3f}",
        ]
    lines += [
        "",
        "## Gaps",
        "",
        "| Par | Ultimo cuadro | Separacion m | Eje libre m | Admisible | Alternado | Causa |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in result["gaps"]:
        lines.append(
            f"| {row['pair']} | {row['last_frame']} | {row['spacing_m']:.3f} | "
            f"{row['free_axis_m']:.3f} | {'si' if row['admissible'] else 'no'} | "
            f"{'si' if row['alternated'] else 'no'} | {row.get('cause', '')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = evaluate(args.sidecar)
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_report(result, args.output_dir / "evaluation.md")
    print(f"evaluacion escrita en {args.output_dir}")


if __name__ == "__main__":
    main()
