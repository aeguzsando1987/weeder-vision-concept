"""R1/R8: la demo valida toda la entrada antes de escribir, con o sin graficos.

Se ejecuta el script como proceso, igual que un usuario, para cubrir la ruta CLI.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "demo_scene.py"
CONFIGS = ROOT / "configs"


def _run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=tmp_path,  # no depende del directorio de trabajo
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
        text=True,
        capture_output=True,
        check=False,
    )


def _yaml(path: Path, payload: dict) -> Path:
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _corrupt(tmp_path: Path, name: str, corruption: str) -> Path:
    payload = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text())
    if corruption == "unknown_root_key":
        payload["clave_desconocida"] = 1
    elif corruption == "unknown_nested_key":
        section = next(key for key in payload if isinstance(payload[key], dict))
        payload[section]["clave_desconocida"] = 1
    return _yaml(tmp_path / f"{name}.yaml", payload)


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("name", ["tool", "safety", "kinematics"])
@pytest.mark.parametrize("corruption", ["missing_file", "unknown_root_key", "unknown_nested_key"])
def test_invalid_configuration_writes_nothing(
    tmp_path: Path, plot: bool, name: str, corruption: str
) -> None:
    path = (
        tmp_path / f"{name}.yaml"
        if corruption == "missing_file"
        else _corrupt(tmp_path, name, corruption)
    )
    output = tmp_path / "out"
    args = ["--scene", "nominal_three_plants", f"--{name}-config", str(path)]
    args += ["--output", str(output)]
    if not plot:
        args.append("--no-plot")
    result = _run(tmp_path, *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "no se exporto ningun artefacto" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize("mode", ["--scene", "--config"])
def test_mixed_scene_and_tool_domains_are_rejected(tmp_path: Path, plot: bool, mode: str) -> None:
    payload = yaml.safe_load((CONFIGS / "tool.yaml").read_text())
    payload["tool"]["domain"] = "image_px"
    tool = _yaml(tmp_path / "tool.yaml", payload)
    output = tmp_path / "out"
    source = (
        ["--scene", "nominal_three_plants"]
        if mode == "--scene"
        else ["--config", str(CONFIGS / "synthetic.yaml")]
    )
    args = [*source, "--tool-config", str(tool), "--output", str(output)]
    if not plot:
        args.append("--no-plot")
    result = _run(tmp_path, *args)
    assert result.returncode == 2
    assert "image_px" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("mode", ["--scene", "--config"])
def test_underprotecting_safety_is_rejected_in_both_modes(tmp_path: Path, mode: str) -> None:
    payload = yaml.safe_load((CONFIGS / "safety.yaml").read_text())
    payload["safety"]["uncertainty"].update(
        localization_bound_m=0.02,
        calibration_bound_m=0.02,
        tracking_bound_m=0.02,
        bound_budget_m=0.06,
    )
    safety = _yaml(tmp_path / "safety.yaml", payload)
    output = tmp_path / "out"
    source = (
        ["--scene", "nominal_three_plants"]
        if mode == "--scene"
        else ["--config", str(CONFIGS / "synthetic.yaml")]
    )
    result = _run(
        tmp_path, *source, "--safety-config", str(safety), "--output", str(output), "--no-plot"
    )
    assert result.returncode == 2
    assert "incertidumbre" in result.stderr.lower()
    assert not output.exists()


def test_coherent_raised_uncertainty_is_exported_as_applied(tmp_path: Path) -> None:
    """Positivo R1: cotas y margen elevados de forma coherente llegan al JSON."""
    safety = yaml.safe_load((CONFIGS / "safety.yaml").read_text())
    safety["safety"]["uncertainty"].update(
        localization_bound_m=0.02,
        calibration_bound_m=0.02,
        tracking_bound_m=0.02,
        bound_budget_m=0.06,
    )
    synthetic = yaml.safe_load((CONFIGS / "synthetic.yaml").read_text())
    synthetic["scene"]["exclusion"]["uncertainty_margin_m"] = 0.06
    output = tmp_path / "out"
    result = _run(
        tmp_path,
        "--config",
        str(_yaml(tmp_path / "synthetic.yaml", synthetic)),
        "--safety-config",
        str(_yaml(tmp_path / "safety.yaml", safety)),
        "--output",
        str(output),
        "--no-plot",
    )
    assert result.returncode == 0, result.stderr
    scene = json.loads((output / "nominal_three_plants.json").read_text())
    assert scene["uncertainty"]["total"] == pytest.approx(0.06, abs=1e-12, rel=0)
    assert scene["uncertainty"]["budget"] == 0.06
    assert all(e["uncertainty_margin"] == 0.06 for e in scene["exclusions"])
    resolved = yaml.safe_load((output / "resolved_config.yaml").read_text())
    assert resolved["safety"]["uncertainty"]["bound_budget_m"] == 0.06
    assert resolved["scene"]["exclusion"]["uncertainty_margin_m"] == 0.06


def test_catalog_scenes_use_the_given_safety_policy(tmp_path: Path) -> None:
    """La escena del catalogo aplica la politica del archivo pedido, no un default oculto."""
    safety = yaml.safe_load((CONFIGS / "safety.yaml").read_text())
    safety["safety"]["uncertainty"].update(
        localization_bound_m=0.005,
        calibration_bound_m=0.005,
        tracking_bound_m=0.005,
        bound_budget_m=0.02,
    )
    output = tmp_path / "out"
    result = _run(
        tmp_path,
        "--scene",
        "nominal_three_plants",
        "--safety-config",
        str(_yaml(tmp_path / "safety.yaml", safety)),
        "--output",
        str(output),
        "--no-plot",
    )
    assert result.returncode == 0, result.stderr
    scene = json.loads((output / "nominal_three_plants.json").read_text())
    assert scene["uncertainty"]["total"] == pytest.approx(0.015, abs=1e-12, rel=0)
    assert scene["uncertainty"]["budget"] == 0.02


def test_export_records_configuration_provenance(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = _run(tmp_path, "--scene", "nominal_three_plants", "--output", str(output), "--no-plot")
    assert result.returncode == 0, result.stderr
    demo = json.loads((output / "nominal_three_plants.json").read_text())["demo"]
    assert set(demo["config_sources"]) == {"tool", "safety", "kinematics"}
    assert all(len(entry["sha256"]) == 64 for entry in demo["config_sources"].values())
    assert demo["scene_source"] == "fixture_catalog"


def _safety_with_limit(tmp_path: Path, limit: float) -> Path:
    payload = yaml.safe_load((CONFIGS / "safety.yaml").read_text())
    payload["safety"]["max_conservative_discretization_excess_m"] = limit
    return _yaml(tmp_path / "safety.yaml", payload)


@pytest.mark.parametrize("plot", [True, False])
@pytest.mark.parametrize(
    "source",
    [
        ["--scene", "nominal_three_plants"],
        ["--all"],
        ["--config", str(CONFIGS / "synthetic.yaml")],
    ],
    ids=["scene", "all", "config"],
)
def test_r12_insufficient_discretization_budget_is_rejected_on_every_path(
    tmp_path: Path, plot: bool, source: list[str]
) -> None:
    output = tmp_path / "out"
    args = [*source, "--safety-config", str(_safety_with_limit(tmp_path, 1e-5))]
    args += ["--output", str(output)]
    if not plot:
        args.append("--no-plot")
    result = _run(tmp_path, *args)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "discretizacion" in result.stderr
    assert not output.exists()


@pytest.mark.parametrize("source", [["--scene", "nominal_three_plants"], ["--all"]])
def test_r12_sufficient_budget_exports_the_real_excess(tmp_path: Path, source: list[str]) -> None:
    output = tmp_path / "out"
    limit = 2.2e-4  # justo por encima de 2.17e-4 (r = 0.18 m, 64 lados)
    result = _run(
        tmp_path,
        *source,
        "--safety-config",
        str(_safety_with_limit(tmp_path, limit)),
        "--output",
        str(output),
        "--no-plot",
    )
    assert result.returncode == 0, result.stderr
    scene = json.loads((output / "nominal_three_plants.json").read_text())
    for exclusion in scene["exclusions"]:
        radius, sides = exclusion["total_radius"], exclusion["polygon_segments"]
        expected = radius * (1.0 / math.cos(math.pi / sides) - 1.0)
        assert exclusion["discretization_excess"] == pytest.approx(expected, abs=1e-15, rel=0)
        assert exclusion["discretization_excess"] <= limit
