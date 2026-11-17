#!/usr/bin/env python3
"""Demo/CLI del planificador geometrico E2 (orden A1/E2 §6, E2-10).

Capa delgada sobre la API publica: construye el mapa del caso, pide una ruta al
planificador, y exporta mapa, perfil efectivo, solicitud, configuracion resuelta,
resultado (validacion, cobertura, cota de despeje, candidatos rechazados), curva y
grafico cenital. No duplica algoritmos del paquete.

Ejemplos:

    python scripts/demo_planner.py                              # caso demo de planner.yaml
    python scripts/demo_planner.py --case straight_lane_free
    python scripts/demo_planner.py --scene five_plants --start 0 0.1 --goal 0 4.9
    python scripts/demo_planner.py --random-seed 7 --no-plot

Codigos de salida:

- `0`: ruta aceptada **geometricamente** (`validation_scope=geometric`,
  `mechanical_feasibility=not_evaluated`); no es una ejecucion aprobada.
- `2`: entrada invalida o peticion no soportada. Se valida todo antes de escribir:
  no se crean artefactos.
- `3`: peticion valida sin ruta aceptada (abstencion) o no certificada. Se exporta
  el diagnostico sin ninguna ruta marcada como aceptada.

`result.json` contiene solo campos deterministas; la fecha, el tiempo de pared y el
comando van en `run.json`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon

from precision_weeder.core.config import (
    PlannerFile,
    SafetyFile,
    ToolFile,
    ValidationFile,
    load_yaml_mapping,
    validate_config_payload,
)
from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.core.trajectory import OrientationRequirement
from precision_weeder.core.types import Frame, PlanStatus, Point2
from precision_weeder.geometry.tool_profile import ToolGeometry, tool_geometry_from_config
from precision_weeder.mapping.local_map import LocalMap
from precision_weeder.mapping.synthetic_adapter import local_map_from_scene
from precision_weeder.planning import GeometricPlanner, PlanRequest, PlanResult
from precision_weeder.scenarios.fixtures import build_fixture
from precision_weeder.scenarios.planning_cases import (
    build_planning_case,
    planning_case_names,
    random_planning_case,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
DEFAULT_OUTPUT = Path("runs/demo_planner")
EXIT_ACCEPTED, EXIT_INVALID, EXIT_NO_ROUTE = 0, 2, 3


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Planificador geometrico E2 (demo).")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--case", choices=planning_case_names(), help="Caso de aceptacion E2.")
    source.add_argument("--scene", help="Escena E1 del catalogo (con --start/--goal).")
    source.add_argument("--random-seed", type=int, help="Escenario aleatorio congelado (E2-08).")
    parser.add_argument("--start", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--goal", type=float, nargs=2, metavar=("X", "Y"))
    parser.add_argument("--theta", type=float, help="Orientacion fija de B en W [rad].")
    parser.add_argument("--planner-config", type=Path, default=CONFIGS / "planner.yaml")
    parser.add_argument("--validation-config", type=Path, default=CONFIGS / "validation.yaml")
    parser.add_argument("--safety-config", type=Path, default=CONFIGS / "safety.yaml")
    parser.add_argument("--tool-config", type=Path, default=CONFIGS / "tool.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args(argv)


@dataclass
class _Prepared:
    name: str
    local_map: LocalMap
    profile: ToolGeometry
    request: PlanRequest
    planner: GeometricPlanner
    resolved: dict[str, Any]


def _prepare(args: argparse.Namespace) -> _Prepared:
    """Valida configuracion, caso y peticion completos. No escribe nada."""
    planner_file = validate_config_payload(
        PlannerFile, load_yaml_mapping(args.planner_config), str(args.planner_config)
    )
    validation = validate_config_payload(
        ValidationFile, load_yaml_mapping(args.validation_config), str(args.validation_config)
    ).validation
    safety = validate_config_payload(
        SafetyFile, load_yaml_mapping(args.safety_config), str(args.safety_config)
    ).safety
    tool_config = validate_config_payload(
        ToolFile, load_yaml_mapping(args.tool_config), str(args.tool_config)
    ).tool
    profile = tool_geometry_from_config(tool_config)
    demo = planner_file.demo
    if args.case is not None:
        case = build_planning_case(args.case, uncertainty=safety.uncertainty)
        name, local_map = case.name, case.local_map
        start, goal, theta = case.start_xy_m, case.goal_xy_m, case.theta_rad
    elif args.random_seed is not None:
        case = random_planning_case(
            args.random_seed, planner_file.random_scenarios, uncertainty=safety.uncertainty
        )
        name, local_map = case.name, case.local_map
        start, goal, theta = case.start_xy_m, case.goal_xy_m, case.theta_rad
    else:
        scene_name = args.scene or demo.scene
        local_map = local_map_from_scene(build_fixture(scene_name, uncertainty=safety.uncertainty))
        name = scene_name
        start, goal, theta = demo.start_xy_m, demo.goal_xy_m, demo.theta_rad
    if args.start is not None:
        start = tuple(args.start)
    if args.goal is not None:
        goal = tuple(args.goal)
    if args.theta is not None:
        theta = args.theta
    domain = local_map.domain
    request = PlanRequest(
        request_id=f"demo:{name}",
        map_id=local_map.map_id,
        map_hash=local_map.content_hash(),
        profile_id=profile.tool_id,
        profile_hash=profile.content_hash(),
        start=Point2(x=start[0], y=start[1], frame=Frame.WORLD, domain=domain),
        goal=Point2(x=goal[0], y=goal[1], frame=Frame.WORLD, domain=domain),
        orientation=OrientationRequirement.fixed(theta),
    )
    planner = GeometricPlanner(planner_file.planner, validation, safety)
    resolved = {
        "planner": planner_file.model_dump(mode="json"),
        "validation": validation.model_dump(mode="json"),
        "safety": safety.model_dump(mode="json"),
        "tool": tool_config.model_dump(mode="json"),
        "case": name,
        "config_hash": planner.config_hash,
    }
    return _Prepared(name, local_map, profile, request, planner, resolved)


def _draw(prepared: _Prepared, result: PlanResult, path: Path, dpi: int) -> None:
    local_map = prepared.local_map
    corridor = local_map.corridor
    figure, axis = plt.subplots(figsize=(5.5, 9.0))
    axis.add_patch(
        MplPolygon(
            corridor.polygon_points(), closed=True, fill=False, ec="#444", lw=1.2, label="corredor"
        )
    )
    unknown = local_map.unknown_union()
    for part in getattr(unknown, "geoms", [unknown]):
        if not part.is_empty and part.area > 0:
            axis.add_patch(
                MplPolygon(
                    np.asarray(part.exterior.coords),
                    closed=True,
                    fc="#999",
                    alpha=0.45,
                    label="desconocido",
                )
            )
    for polygon in local_map.exclusion_polygons():
        axis.add_patch(
            MplPolygon(
                np.asarray(polygon.exterior.coords),
                closed=True,
                fc="#d9534f",
                alpha=0.35,
                ec="#a33",
            )
        )
    for exclusion in local_map.exclusions:
        axis.plot(exclusion.center.x, exclusion.center.y, "x", color="#7a1f1c", ms=5)
    for record in result.rejected:
        pts = np.asarray(record.waypoints)
        axis.plot(pts[:, 0], pts[:, 1], "--", color="#888", lw=0.8)
    validation = result.validation or {}
    if result.path is not None:
        samples = np.linspace(0.0, 1.0, 400)
        curve = np.array([result.path.evaluate(s) for s in samples])
        tool = np.asarray(validation["effective_geometry"]["tool_collision_polygon_W_orientation"])
        for s in np.linspace(0.0, 1.0, 25):
            x, y = result.path.evaluate(s)
            axis.add_patch(
                MplPolygon(
                    tool + np.array([x, y]),
                    closed=True,
                    fill=False,
                    ec="#1f5fa6",
                    lw=0.6,
                    alpha=0.7,
                )
            )
        axis.plot(
            curve[:, 0],
            curve[:, 1],
            color="#1f5fa6",
            lw=1.8,
            label="TaskPath (propuesta geometrica)",
        )
    start, goal = prepared.request.start, prepared.request.goal
    axis.plot(start.x, start.y, "o", color="#2c7a3f", label="inicio (origen B)")
    axis.plot(goal.x, goal.y, "s", color="#2c7a3f", label="meta (origen B)")
    clearance = validation.get("clearance_lower_bound_m")
    coverage = (validation.get("coverage") or {}).get("covered_fraction")
    title = f"{prepared.name}: {result.status.value}"
    if result.reason is not None:
        title += f" ({result.reason.value})"
    axis.set_title(title, fontsize=10)
    figure.text(
        0.02,
        0.005,
        "alcance: geometrico; viabilidad mecanica: no evaluada\n"
        + (f"cota inferior de despeje: {clearance:.4f} m; " if clearance is not None else "")
        + (f"cobertura objetivo: {coverage:.3f}" if coverage is not None else ""),
        fontsize=7,
        va="bottom",
    )
    axis.set_xlabel("x lateral [m] (W)")
    axis.set_ylabel("y longitudinal [m] (W)")
    axis.set_aspect("equal")
    axis.set_xlim(corridor.x_min - 0.15, corridor.x_max + 0.15)
    axis.set_ylim(corridor.y_min - 0.15, corridor.y_max + 0.15)
    axis.legend(loc="upper right", fontsize=6)
    figure.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        prepared = _prepare(args)
        result = prepared.planner.plan(prepared.request, prepared.local_map, prepared.profile)
    except PrecisionWeederError as exc:
        print(f"error: entrada invalida: {exc}", file=sys.stderr)
        return EXIT_INVALID
    if result.status in (PlanStatus.UNSUPPORTED, PlanStatus.INVALID_INPUT):
        print(f"error: {result.status.value}: {result.detail}", file=sys.stderr)
        return EXIT_INVALID

    out = args.output / prepared.name
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("path.json", "plot.png"):
        (out / stale).unlink(missing_ok=True)
    _write_json(out / "map.json", prepared.local_map.to_dict())
    _write_json(out / "profile.json", prepared.profile.to_dict())
    _write_json(out / "request.json", prepared.request.to_dict())
    _write_json(out / "config_resolved.json", prepared.resolved)
    payload = result.to_dict(include_timing=False)
    _write_json(out / "result.json", payload)
    _write_json(
        out / "candidates.json",
        {"chosen": payload["chosen"], "rejected": payload["rejected_candidates"]},
    )
    if result.path is not None:
        _write_json(out / "path.json", result.path.to_dict())
    _write_json(
        out / "run.json",
        {
            "executed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            "command": [sys.executable, *sys.argv] if argv is None else ["main", *argv],
            "python": platform.python_version(),
            "timing": result.to_dict()["timing"],
        },
    )
    if not args.no_plot:
        _draw(prepared, result, out / "plot.png", args.dpi)
    validation = result.validation or {}
    print(
        f"{prepared.name}: {result.status.value}"
        + (f" ({result.reason.value})" if result.reason else "")
        + f"; despeje>= {validation.get('clearance_lower_bound_m')}"
        + f"; salida {out}"
    )
    return EXIT_ACCEPTED if result.accepted else EXIT_NO_ROUTE


if __name__ == "__main__":
    sys.exit(main())
