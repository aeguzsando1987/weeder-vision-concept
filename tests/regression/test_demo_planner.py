"""E2-10: demo/CLI del planificador, codigos de salida y artefactos."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import demo_planner  # noqa: E402
from precision_weeder.core.trajectory import TaskPath  # noqa: E402
from precision_weeder.mapping.local_map import LocalMap  # noqa: E402

SUCCESS_FILES = {
    "map.json",
    "profile.json",
    "request.json",
    "config_resolved.json",
    "result.json",
    "candidates.json",
    "path.json",
    "run.json",
    "plot.png",
}


def _run(tmp_path: Path, *args: str) -> int:
    return demo_planner.main([*args, "--output", str(tmp_path / "out")])


def test_accepted_route_exports_everything(tmp_path: Path) -> None:
    assert _run(tmp_path) == 0
    out = tmp_path / "out" / "nominal_three_plants"
    assert {p.name for p in out.iterdir()} == SUCCESS_FILES
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "ACCEPTED"
    assert result["validation_scope"] == "geometric"
    assert result["mechanical_feasibility"] == "not_evaluated"
    validation = result["validation"]
    assert validation["clearance_lower_bound_m"] > 1e-4
    assert validation["coverage"]["covered_target_area_m2"] > 0
    assert "tool_collision_polygon_W_orientation" in validation["effective_geometry"]
    path = TaskPath.from_dict(json.loads((out / "path.json").read_text()))
    local_map = LocalMap.from_dict(json.loads((out / "map.json").read_text()))
    assert path.map_hash == local_map.content_hash() == result["map_hash"]
    resolved = json.loads((out / "config_resolved.json").read_text())
    assert resolved["config_hash"] == result["config_hash"]
    assert "timing" not in result and "timing" in json.loads((out / "run.json").read_text())


def test_result_is_byte_identical_across_runs(tmp_path: Path) -> None:
    assert _run(tmp_path, "--no-plot") == 0
    first = (tmp_path / "out/nominal_three_plants/result.json").read_bytes()
    assert _run(tmp_path, "--no-plot") == 0
    assert (tmp_path / "out/nominal_three_plants/result.json").read_bytes() == first


@pytest.mark.parametrize(
    ("case", "reason"),
    [("blocked_corridor", "NO_PATH_FOUND"), ("unknown_band", "UNKNOWN_SPACE")],
)
def test_valid_request_without_route_exports_diagnostics_only(tmp_path: Path, case, reason) -> None:
    assert _run(tmp_path, "--case", case) == 3
    out = tmp_path / "out" / case
    names = {p.name for p in out.iterdir()}
    assert "path.json" not in names and "result.json" in names and "plot.png" in names
    result = json.loads((out / "result.json").read_text())
    assert result["status"] == "ABSTAIN" and result["reason"] == reason
    assert result["path"] is None and result["chosen"] is None


def test_stale_route_is_removed_when_a_later_run_fails(tmp_path: Path) -> None:
    assert _run(tmp_path, "--scene", "nominal_three_plants") == 0
    out = tmp_path / "out" / "nominal_three_plants"
    assert (out / "path.json").exists()
    # Mismo escenario, inicio en colision: no debe quedar una ruta aceptada anterior.
    assert _run(tmp_path, "--scene", "nominal_three_plants", "--start", "0", "1.0") == 3
    assert not (out / "path.json").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("--scene", "svg_graphic"),  # dominio grafico: no se planifica en metros
        ("--scene", "no_existe"),
        ("--goal", "0", "0.05"),  # meta por detras del inicio (y=0.10)
        ("--start", "nan", "0.1"),
    ],
)
def test_invalid_input_exits_2_without_artifacts(tmp_path: Path, args) -> None:
    assert _run(tmp_path, *args) == 2
    assert not (tmp_path / "out").exists()


def test_invalid_or_unknown_config_key_exits_2(tmp_path: Path) -> None:
    payload = yaml.safe_load((PROJECT_ROOT / "configs/planner.yaml").read_text())
    payload["planner"]["clave_inventada"] = 1
    bad = tmp_path / "planner.yaml"
    bad.write_text(yaml.safe_dump(payload))
    assert _run(tmp_path, "--planner-config", str(bad)) == 2
    assert not (tmp_path / "out").exists()


def test_unsupported_request_exits_2(tmp_path: Path, monkeypatch) -> None:
    from precision_weeder.core.trajectory import OrientationPolicy, OrientationRequirement

    monkeypatch.setattr(
        demo_planner.OrientationRequirement,
        "fixed",
        classmethod(lambda cls, theta: OrientationRequirement(policy=OrientationPolicy.FREE)),
    )
    assert _run(tmp_path) == 2
    assert not (tmp_path / "out").exists()


def test_random_seed_case_runs(tmp_path: Path) -> None:
    code = _run(tmp_path, "--random-seed", "3", "--no-plot")
    assert code in (0, 3)
    assert (tmp_path / "out" / "random_003" / "result.json").exists()


@pytest.mark.parametrize(
    "selector",
    [
        ("--scene", "nominal_three_plants"),
        ("--case", "straight_lane_free"),
        ("--random-seed", "7"),
    ],
)
def test_discretization_budget_exceeded_exits_2_without_artifacts(tmp_path: Path, selector) -> None:
    """A1E2-R4: presupuesto de exceso de discretizacion vinculante en el CLI."""
    payload = yaml.safe_load((PROJECT_ROOT / "configs/safety.yaml").read_text())
    payload["safety"]["max_conservative_discretization_excess_m"] = 1.0e-8
    payload["safety"]["clearance_epsilon_m"] = 1.0e-4
    payload["safety"]["geometric_approximation_tolerance_m"] = 1.0e-5
    payload["safety"]["limit_comparison_tolerance"] = 1.0e-9
    strict = tmp_path / "safety_strict.yaml"
    strict.write_text(yaml.safe_dump(payload))
    assert _run(tmp_path, *selector, "--safety-config", str(strict), "--no-plot") == 2
    assert not (tmp_path / "out").exists()
    # Control: el mismo selector con el presupuesto por defecto si exporta.
    assert _run(tmp_path, *selector, "--no-plot") in (0, 3)
