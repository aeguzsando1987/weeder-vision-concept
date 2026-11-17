"""Regresion del demo: exportacion determinista y artefactos con unidades."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import demo_scene  # noqa: E402


def test_demo_exports_json_and_figure(tmp_path: Path) -> None:
    code = demo_scene.main(["--scene", "nominal_three_plants", "--output", str(tmp_path)])
    assert code == 0
    json_path = tmp_path / "nominal_three_plants.json"
    png_path = tmp_path / "nominal_three_plants.png"
    assert json_path.is_file()
    assert png_path.is_file()
    assert png_path.stat().st_size > 1024


def test_demo_json_is_byte_identical_across_runs(tmp_path: Path) -> None:
    """Misma semilla, misma exportacion: comparacion de datos, no de imagenes (§7)."""
    first, second = tmp_path / "a", tmp_path / "b"
    demo_scene.main(["--scene", "irregular_spacing", "--output", str(first), "--no-plot"])
    demo_scene.main(["--scene", "irregular_spacing", "--output", str(second), "--no-plot"])
    name = "irregular_spacing.json"
    assert (first / name).read_bytes() == (second / name).read_bytes()


def test_demo_json_declares_units_frames_and_provenance(tmp_path: Path) -> None:
    demo_scene.main(["--scene", "nominal_three_plants", "--output", str(tmp_path), "--no-plot"])
    payload = json.loads((tmp_path / "nominal_three_plants.json").read_text(encoding="utf-8"))
    assert payload["unit"] == "m"
    assert payload["frame"] == "W"
    assert payload["domain"] == "ground_metric_synthetic"
    assert payload["metric_scale_verified"] is False
    assert payload["provenance"] == "synthetic"
    assert payload["demo"]["physical_values_are_measurements"] is False
    tool = payload["demo"]["tool"]
    assert tool["unit"] == "m"
    assert tool["pivot"]["frame"] == "B"
    assert pytest.approx(0.20, abs=1e-12, rel=0) == tool["extent_x"]
    assert tool["cutting_area"] < tool["collision_area"]
    pose = payload["demo"]["tool_pose"]
    assert pose["name"] == "T_WB"


def test_demo_from_config_writes_resolved_config(tmp_path: Path) -> None:
    code = demo_scene.main(
        [
            "--config",
            str(PROJECT_ROOT / "configs" / "synthetic.yaml"),
            "--output",
            str(tmp_path),
            "--no-plot",
        ]
    )
    assert code == 0
    resolved = (tmp_path / "resolved_config.yaml").read_text(encoding="utf-8")
    assert "schema_version" in resolved
    assert "ground_metric_synthetic" in resolved
    assert (tmp_path / "nominal_three_plants.json").is_file()


def test_demo_exports_every_scene(tmp_path: Path) -> None:
    from precision_weeder.scenarios.fixtures import fixture_names

    assert demo_scene.main(["--all", "--output", str(tmp_path), "--no-plot"]) == 0
    for name in fixture_names():
        assert (tmp_path / f"{name}.json").is_file()


def test_demo_svg_scene_keeps_graphic_units(tmp_path: Path) -> None:
    demo_scene.main(["--scene", "svg_graphic", "--output", str(tmp_path), "--no-plot"])
    payload = json.loads((tmp_path / "svg_graphic.json").read_text(encoding="utf-8"))
    assert payload["unit"] == "svg_unit"
    assert payload["is_metric"] is False
    assert payload["demo"]["tool"]["unit"] == "svg_unit"
