"""Sondas de cierre E1 del revisor tecnico, integradas en la suite.

Origen: la evidencia de la revision (no versionada) (R12 catalogo,
N1), conservado sin modificar. Unica adaptacion: `ROOT` se calcula desde la nueva
ubicacion de este archivo (el orden de imports lo normaliza Ruff).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.simulation.tool import ToolGeometry

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("plot", [False, True])
def test_catalog_honors_discretization_budget(tmp_path, plot):
    config = yaml.safe_load((ROOT / "configs/safety.yaml").read_text())
    config["safety"]["max_conservative_discretization_excess_m"] = 1e-5
    safety = tmp_path / "safety.yaml"
    safety.write_text(yaml.safe_dump(config))
    output = tmp_path / "output"
    args = [
        sys.executable,
        str(ROOT / "scripts/demo_scene.py"),
        "--scene",
        "nominal_three_plants",
        "--safety-config",
        str(safety),
        "--output",
        str(output),
    ]
    if not plot:
        args.append("--no-plot")
    result = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert not output.exists()


@pytest.mark.parametrize("offset", [(0.0,), (0.0, 0.0, float("nan"))])
def test_rectangular_offset_requires_exact_dimension(offset):
    with pytest.raises((PrecisionWeederError, ValueError, TypeError)):
        ToolGeometry.rectangular("blade", 0.2, 0.08, pivot_offset=offset)
