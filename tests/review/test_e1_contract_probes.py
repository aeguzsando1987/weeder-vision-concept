"""Sondas de contrato E1 del revisor tecnico, integradas en la suite permanente.

Origen: `docs/reviews/evidence/E1/test_contract_probes.py` (9 casos, R1-R6), que se
conserva sin modificar como evidencia historica. Adaptaciones de esta copia, sin
cambiar ningun invariante comprobado:

- `Path.cwd()` se sustituye por la raiz del repositorio derivada de `__file__`, para
  que la prueba no dependa del directorio de ejecucion.
- `try/except ValueError: pass` se expresa con `contextlib.suppress` (lint SIM105).
"""

import contextlib
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from precision_weeder.core.config import CalibrationConfig, ProjectConfig, SceneConfig, ToolConfig
from precision_weeder.core.errors import (
    FrameMismatchError,
    MissingCalibrationError,
    SchemaValidationError,
)
from precision_weeder.core.types import CalibrationStatus, Corridor, Frame, Point2, UncertaintyBound
from precision_weeder.core.units import SpatialDomain
from precision_weeder.scenarios.synthetic import build_scene
from precision_weeder.simulation.tool import ToolGeometry

D = SpatialDomain.GROUND_METRIC_SYNTHETIC
ROOT = Path(__file__).resolve().parents[2]


def test_verified_configuration_requires_scale_evidence():
    verified = SpatialDomain.GROUND_METRIC_VERIFIED
    with pytest.raises((ValueError, MissingCalibrationError)):
        ProjectConfig(
            mode="calibrated",
            scene=SceneConfig(domain=verified),
            tool=ToolConfig(domain=verified),
            calibration=CalibrationConfig(status=CalibrationStatus.VERIFIED),
        )


def test_corridor_rejects_point_in_different_frame():
    corridor = Corridor.centered(1, 5, frame=Frame.WORLD, domain=D)
    with pytest.raises((FrameMismatchError, SchemaValidationError)):
        corridor.contains_point(Point2(0, 1, Frame.VEHICLE, D))


def test_tool_contract_rejects_non_origin_pivot():
    polygon = np.array([[0.9, -0.04], [1.1, -0.04], [1.1, 0.04], [0.9, 0.04]])
    with pytest.raises(SchemaValidationError):
        ToolGeometry("shifted", polygon, polygon.copy(), Point2(1, 0, Frame.TOOL, D))


def test_tool_owns_validated_polygon_data():
    polygon = np.array([[-0.1, -0.04], [0.1, -0.04], [0.1, 0.04], [-0.1, 0.04]])
    tool = ToolGeometry("owned", polygon, polygon.copy(), Point2(0, 0, Frame.TOOL, D))
    original = tool.collision_polygon.copy()
    with contextlib.suppress(ValueError):
        polygon[:] = np.nan
    assert np.array_equal(tool.collision_polygon, original), "external mutation invalidated tool"


def test_tool_polygon_is_immutable():
    tool = ToolGeometry.rectangular("immutable", 0.2, 0.08)
    with pytest.raises((ValueError, TypeError)):
        tool.collision_polygon[:] = np.nan


def test_uncertainty_override_cannot_underprotect_scene():
    bound = UncertaintyBound(0.1, 0.1, 0.1, D)
    try:
        scene = build_scene(SceneConfig(), uncertainty=bound)
    except (SchemaValidationError, ValueError):
        return  # Rejecting inconsistent input is also a valid contract.
    assert all(e.uncertainty_margin >= scene.uncertainty.total - 1e-12 for e in scene.exclusions)


def test_scene_rejects_nonfinite_advance():
    scene = build_scene(SceneConfig())
    with pytest.raises((SchemaValidationError, ValueError)):
        replace(scene, advance_speed=math.nan)


def test_scene_rejects_missing_exclusions_for_reference_plants():
    from precision_weeder.core.errors import PrecisionWeederError

    scene = build_scene(SceneConfig())
    with pytest.raises((PrecisionWeederError, ValueError)):
        replace(scene, exclusions=())


def test_demo_respects_configured_uncertainty(tmp_path):
    import json
    import os
    import subprocess
    import sys

    import yaml

    root = ROOT
    payload = yaml.safe_load((root / "configs/safety.yaml").read_text())
    payload["safety"]["uncertainty"].update(
        localization_bound_m=0.02,
        calibration_bound_m=0.02,
        tracking_bound_m=0.02,
        bound_budget_m=0.06,
    )
    safety_path = tmp_path / "safety.yaml"
    safety_path.write_text(yaml.safe_dump(payload))
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/demo_scene.py"),
            "--config",
            str(root / "configs/synthetic.yaml"),
            "--safety-config",
            str(safety_path),
            "--output",
            str(output),
            "--no-plot",
        ],
        text=True,
        capture_output=True,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
    )
    if result.returncode != 0:
        assert any(
            word in (result.stdout + result.stderr).lower()
            for word in ["uncertainty", "incertidumbre", "margin", "margen"]
        )
        return  # Explicit rejection of inconsistent configuration is acceptable.
    scene = json.loads((output / "nominal_three_plants.json").read_text())
    assert scene["uncertainty"]["total"] >= 0.06 - 1e-12
    assert all(e["uncertainty_margin"] >= 0.06 - 1e-12 for e in scene["exclusions"])
