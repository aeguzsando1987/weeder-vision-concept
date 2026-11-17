#!/usr/bin/env python3
"""Demo delgada: dibuja y exporta escenas sinteticas (E1).

El script no duplica algoritmos del paquete (§4): solo construye escenas mediante
la API publica, las serializa a JSON y las dibuja con unidades explicitas.

Ejemplos:

    python scripts/demo_scene.py --scene nominal_three_plants --output runs/demo_scene
    python scripts/demo_scene.py --all --output runs/demo_scene
    python scripts/demo_scene.py --config configs/synthetic.yaml --output runs/demo_scene

Salidas por escena: `<scene_id>.json` y `<scene_id>.png`.

Validacion (R8): toda la configuracion (documento completo, claves desconocidas
incluidas) y la coherencia escena/herramienta se comprueban **antes** de escribir
cualquier artefacto, con o sin `--no-plot`. Un archivo de configuracion pedido y
ausente es un error; ante entrada invalida el script sale con codigo 2, explica la
causa y no escribe JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # sin interfaz grafica: el reporte debe ser exportable (§4)

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

from precision_weeder.core.config import (
    SCHEMA_VERSION,
    KinematicsFile,
    ProjectConfig,
    SafetyFile,
    ToolFile,
    load_project_config,
    load_yaml_mapping,
    validate_config_payload,
)
from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.core.types import Frame
from precision_weeder.core.units import SpatialDomain, require_same_domain
from precision_weeder.geometry.frames import Transform2D
from precision_weeder.scenarios.fixtures import (
    build_fixture,
    fixture_names,
    svg_tool_geometry,
    svg_tool_transform,
)
from precision_weeder.scenarios.synthetic import (
    SyntheticScene,
    build_scene,
    require_discretization_budget,
)
from precision_weeder.simulation.tool import ToolGeometry, tool_geometry_from_config

DEFAULT_OUTPUT = Path("runs/demo_scene")
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dibuja y exporta escenas sinteticas del banco offline (E1).",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--scene",
        choices=fixture_names(),
        default="nominal_three_plants",
        help="Escena registrada a exportar (por defecto: nominal_three_plants).",
    )
    source.add_argument("--all", action="store_true", help="Exporta todas las escenas registradas.")
    source.add_argument(
        "--config",
        type=Path,
        help="Construye la escena desde configs/synthetic.yaml en lugar del catalogo.",
    )
    # Defaults explicitos: los YAML versionados del repositorio. Una ruta pedida que no
    # exista es un error; nunca se sustituye en silencio por valores por defecto.
    parser.add_argument("--tool-config", type=Path, default=CONFIGS_DIR / "tool.yaml")
    parser.add_argument("--safety-config", type=Path, default=CONFIGS_DIR / "safety.yaml")
    parser.add_argument(
        "--kinematics-config", type=Path, default=CONFIGS_DIR / "kinematics.yaml"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Directorio de salida.")
    parser.add_argument("--no-plot", action="store_true", help="Exporta solo el JSON.")
    parser.add_argument("--dpi", type=int, default=150)
    return parser.parse_args(argv)


@dataclass(frozen=True)
class _PreparedScene:
    """Escena validada y lista para exportar; no se escribe nada hasta tenerlas todas."""

    scene: SyntheticScene
    geometry: ToolGeometry
    transform: Transform2D
    payload: dict[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sources(paths: dict[str, Path]) -> dict[str, dict[str, str]]:
    """Procedencia de la configuracion usada: ruta y hash de cada archivo."""
    return {
        name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
    }


def _tool_for_scene(
    scene: SyntheticScene, metric_tool: ToolGeometry
) -> tuple[ToolGeometry, Transform2D]:
    """Geometria de herramienta y pose de dibujo, coherentes con el dominio de la escena.

    Raises:
        UnitMismatchError: si la herramienta configurada no comparte dominio con la
            escena; se comprueba aqui, no al dibujar.
    """
    if scene.domain is SpatialDomain.GRAPHIC_UNSCALED:
        geometry, transform = svg_tool_geometry(), svg_tool_transform()
    else:
        geometry = metric_tool
        # Pose de dibujo: pivote en el centro lateral del corredor, al inicio del tramo.
        transform = Transform2D(
            parent=Frame.WORLD,
            child=Frame.TOOL,
            tx=scene.corridor.center_x,
            ty=scene.corridor.y_min + 0.5 * geometry.extent_y,
            theta=0.0,
            domain=scene.domain,
        )
    require_same_domain(
        scene.domain,
        geometry.domain,
        transform.domain,
        context=f"escena '{scene.scene_id}' / herramienta '{geometry.tool_id}'",
    )
    return geometry, transform


def _draw_scene(
    scene: SyntheticScene,
    geometry: ToolGeometry,
    transform: Transform2D,
    output_path: Path,
    dpi: int,
) -> None:
    """Dibuja corredor, plantas, exclusiones, huellas, pivote, ejes y unidad."""
    unit = scene.unit.value
    figure, axes = plt.subplots(figsize=(6.0, 8.0))

    corridor = scene.corridor
    axes.add_patch(
        MplPolygon(
            corridor.polygon_points(),
            closed=True,
            facecolor="#f1ece4",
            edgecolor="#8a7355",
            linewidth=1.5,
            label=f"corredor ({corridor.width:g} x {corridor.length:g} {unit})",
            zorder=1,
        )
    )

    for index, exclusion in enumerate(scene.exclusions):
        axes.add_patch(
            MplPolygon(
                exclusion.polygon_points(),
                closed=True,
                facecolor="#cde8c4",
                edgecolor="#4f9b4a",
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
                label=(
                    f"exclusion r={exclusion.total_radius:g} {unit} "
                    f"(P={exclusion.protection_radius:g} + m={exclusion.uncertainty_margin:g})"
                )
                if index == 0
                else None,
                zorder=2,
            )
        )

    if scene.plants:
        axes.scatter(
            [plant.position.x for plant in scene.plants],
            [plant.position.y for plant in scene.plants],
            s=28,
            color="#1f6b2a",
            zorder=5,
            label=f"base de tallo (verdad de referencia, r={scene.plants[0].stem_radius:g} {unit})",
        )
        for plant in scene.plants:
            axes.annotate(
                plant.plant_id.rsplit("_", maxsplit=1)[-1],
                (plant.position.x, plant.position.y),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=7,
                color="#1f6b2a",
            )

    collision = geometry.collision_polygon_in(transform)
    cutting = geometry.cutting_polygon_in(transform)
    axes.add_patch(
        MplPolygon(
            list(collision.exterior.coords)[:-1],
            closed=True,
            facecolor="#ffd166",
            edgecolor="#8a6400",
            linewidth=1.4,
            alpha=0.9,
            label=(
                f"huella de colision {geometry.extent_x:g} x {geometry.extent_y:g} {unit}"
            ),
            zorder=6,
        )
    )
    axes.add_patch(
        MplPolygon(
            list(cutting.exterior.coords)[:-1],
            closed=True,
            facecolor="none",
            edgecolor="#b3261e",
            linewidth=1.2,
            linestyle=":",
            label="huella activa de corte",
            zorder=7,
        )
    )
    pivot = transform.apply_point(geometry.pivot)
    axes.scatter(
        [pivot.x],
        [pivot.y],
        marker="x",
        s=55,
        color="#3d2b14",
        zorder=8,
        label="pivote (origen de B)",
    )

    # Ejes del marco W con su orientacion declarada.
    axis_length = 0.12 * max(corridor.width, corridor.length)
    origin = (corridor.x_min + 0.04 * corridor.width, corridor.y_min + 0.04 * corridor.length)
    axes.annotate(
        "",
        xy=(origin[0] + axis_length, origin[1]),
        xytext=origin,
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.4},
    )
    axes.annotate(
        "",
        xy=(origin[0], origin[1] + axis_length),
        xytext=origin,
        arrowprops={"arrowstyle": "->", "color": "#333333", "linewidth": 1.4},
    )
    y_label = (
        " +y (eje del SVG, sin sentido de avance validado)"
        if scene.domain is SpatialDomain.GRAPHIC_UNSCALED
        else " +y (avance)"
    )
    axes.text(origin[0] + axis_length, origin[1], " +x (lateral)", fontsize=8, va="center")
    axes.text(origin[0], origin[1] + axis_length, y_label, fontsize=8, ha="left", va="bottom")

    gaps = scene.lateral_gaps()
    axes.set_xlabel(f"x [{unit}]  —  marco {scene.frame.value}")
    axes.set_ylabel(f"y [{unit}]  —  marco {scene.frame.value}")
    axes.set_title(
        f"{scene.scene_id}  ·  dominio {scene.domain.value}\n"
        f"TODOS LOS VALORES SON SINTETICOS (procedencia: {scene.provenance.value})",
        fontsize=10,
    )
    subtitle = (
        f"avance {scene.advance_speed:g} {unit}/s"
        f"  ·  area objetivo {scene.target_area():.4g} {unit}^2"
        f"  ·  hueco lateral mas ancho {gaps['widest']:.4g} {unit}"
        f"  ·  escala metrica verificada: no"
    )
    axes.text(
        0.5,
        -0.08,
        subtitle,
        transform=axes.transAxes,
        ha="center",
        va="top",
        fontsize=8,
        color="#444444",
    )
    if scene.domain is SpatialDomain.GRAPHIC_UNSCALED:
        axes.invert_yaxis()  # reproduce el sentido de pantalla del SVG sin alterar los datos
        axes.set_ylabel(f"y [{unit}] (crece hacia abajo, como en el SVG)")

    axes.set_aspect("equal", adjustable="box")
    axes.grid(visible=True, linewidth=0.3, alpha=0.5)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=7, framealpha=0.9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _prepare(
    scene: SyntheticScene,
    metric_tool: ToolGeometry,
    sources: dict[str, dict[str, str]],
    resolved: ProjectConfig | None,
) -> _PreparedScene:
    geometry, transform = _tool_for_scene(scene, metric_tool)
    payload = scene.to_dict()
    payload["demo"] = {
        "schema_version": SCHEMA_VERSION,
        "tool": geometry.to_dict(),
        "tool_pose": transform.to_dict(),
        "physical_values_are_measurements": False,
        "config_sources": sources,
        "scene_source": "configuration" if resolved is not None else "fixture_catalog",
        "note": (
            "Escena y herramienta sinteticas o graficas; ningun valor es una medicion "
            "(docs/assumptions.md)."
        ),
    }
    return _PreparedScene(scene, geometry, transform, payload)


def _load(args: argparse.Namespace) -> tuple[list[_PreparedScene], ProjectConfig | None]:
    """Valida toda la entrada y construye las escenas; no escribe ningun archivo."""
    paths = {
        "tool": args.tool_config,
        "safety": args.safety_config,
        "kinematics": args.kinematics_config,
    }
    if args.config is not None:
        paths = {"synthetic": args.config, **paths}
        project = load_project_config(
            args.config, args.tool_config, args.safety_config, args.kinematics_config
        )
        tool_config, safety = project.tool, project.safety
        uncertainty = safety.uncertainty
        scenes = [
            build_scene(
                project.scene,
                uncertainty=uncertainty.to_policy(project.scene.domain),
                notes=("Escena construida desde configuracion.",),
            )
        ]
    else:
        project = None
        # Documentos completos: una clave desconocida en cualquier nivel es error.
        tool_config = validate_config_payload(
            ToolFile, load_yaml_mapping(args.tool_config), str(args.tool_config)
        ).tool
        # Se conserva SafetyConfig completo: la incertidumbre y el presupuesto de
        # discretizacion (R12) se aplican igual que en la ruta --config.
        safety = validate_config_payload(
            SafetyFile, load_yaml_mapping(args.safety_config), str(args.safety_config)
        ).safety
        uncertainty = safety.uncertainty
        validate_config_payload(
            KinematicsFile, load_yaml_mapping(args.kinematics_config), str(args.kinematics_config)
        )
        names = list(fixture_names()) if args.all else [args.scene]
        scenes = [
            build_fixture(name)
            if name == "svg_graphic"
            else build_fixture(name, uncertainty=uncertainty)
            for name in names
        ]
    for scene in scenes:  # todas las escenas antes de escribir cualquier artefacto
        require_discretization_budget(scene, safety)
    metric_tool = tool_geometry_from_config(tool_config)
    sources = _config_sources(paths)
    return [_prepare(scene, metric_tool, sources, project) for scene in scenes], project


def _write(prepared: _PreparedScene, output_dir: Path, plot: bool, dpi: int) -> dict[str, str]:
    """Escribe la figura (si procede) y despues el JSON, que marca la exportacion completa."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_id = prepared.scene.scene_id
    artifacts: dict[str, str] = {}
    if plot:
        png_path = output_dir / f"{scene_id}.png"
        _draw_scene(prepared.scene, prepared.geometry, prepared.transform, png_path, dpi)
        artifacts["png"] = str(png_path)
    json_path = output_dir / f"{scene_id}.json"
    json_path.write_text(
        json.dumps(prepared.payload, indent=2, sort_keys=True, ensure_ascii=True),
        encoding="utf-8",
    )
    artifacts["json"] = str(json_path)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        prepared, project = _load(args)
    except PrecisionWeederError as exc:
        print(f"error: entrada invalida, no se exporto ningun artefacto\n{exc}", file=sys.stderr)
        return 2

    if project is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "resolved_config.yaml").write_text(project.resolved_yaml(), encoding="utf-8")
    for item in prepared:
        artifacts = _write(item, args.output, not args.no_plot, args.dpi)
        scene = item.scene
        gaps = scene.lateral_gaps()
        print(
            f"{scene.scene_id}: plantas={len(scene.plants)} "
            f"dominio={scene.domain.value} unidad={scene.unit.value} "
            f"area_objetivo={scene.target_area():.4f} {scene.unit.value}^2 "
            f"hueco_mas_ancho={gaps['widest']:.4f} {scene.unit.value} -> "
            + ", ".join(f"{key}={value}" for key, value in artifacts.items())
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
