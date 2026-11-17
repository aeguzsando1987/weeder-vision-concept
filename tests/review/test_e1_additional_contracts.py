"""Casos adversos adicionales E1 del revisor tecnico, integrados en la suite.

Origen: la evidencia de la revision (no versionada)
(20 casos, R2-R4 y R8-R11), conservado sin modificar como evidencia historica.
Unica adaptacion: `ROOT` se calcula desde la nueva ubicacion de este archivo.
"""

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from precision_weeder.core.config import SceneConfig
from precision_weeder.core.errors import PrecisionWeederError
from precision_weeder.core.types import CalibrationReference, Frame, Point2, validate_polygon_points
from precision_weeder.core.units import SpatialDomain
from precision_weeder.geometry.frames import Transform2D
from precision_weeder.scenarios.synthetic import build_scene
from precision_weeder.simulation.tool import ToolGeometry, ToolState

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = SpatialDomain.GROUND_METRIC_SYNTHETIC
REJECTION = (PrecisionWeederError, ValueError, TypeError)


def test_synthetic_builder_cannot_certify_measured_scale():
    with pytest.raises(REJECTION):
        build_scene(SceneConfig(domain=SpatialDomain.GROUND_METRIC_VERIFIED))


@pytest.mark.parametrize(
    ("evidence", "domain"),
    [("   ", SpatialDomain.GROUND_METRIC_VERIFIED), ("reference", SpatialDomain.IMAGE_PX)],
)
def test_verified_reference_rejects_invalid_evidence_or_domain(evidence, domain):
    with pytest.raises(REJECTION):
        CalibrationReference("cam", status="verified", scale_evidence=evidence, domain=domain)


def test_scene_requires_consistent_frames():
    scene = build_scene(SceneConfig())
    with pytest.raises(REJECTION):
        replace(scene, frame=Frame.VEHICLE)


def test_uncertified_empty_scene_has_no_known_work_area():
    scene = build_scene(SceneConfig())
    try:
        unknown = replace(scene, plants=(), exclusions=(), free_space_certified=False)
    except REJECTION:
        return
    assert unknown.target_area() == 0.0


@pytest.mark.parametrize("corruption", ["duplicate", "displaced", "smaller_than_stem"])
def test_exclusion_identity_alone_does_not_protect_reference(corruption):
    scene = build_scene(SceneConfig())
    first = scene.exclusions[0]
    if corruption == "duplicate":
        exclusions = (*scene.exclusions, first)
    elif corruption == "displaced":
        exclusions = (replace(first, center=replace(first.center, x=0.4)), *scene.exclusions[1:])
    else:
        exclusions = (
            replace(first, protection_radius=0.001, uncertainty_margin=0.0),
            *scene.exclusions[1:],
        )
    with pytest.raises(REJECTION):
        replace(scene, exclusions=exclusions)


@pytest.mark.parametrize("field", ["velocity", "acceleration"])
@pytest.mark.parametrize("value", [(1.0,), (1.0, 2.0, math.nan)])
def test_state_vectors_validate_dimension_before_indexing(field, value):
    with pytest.raises(REJECTION):
        ToolState(0.0, Point2(0, 0, Frame.VEHICLE, DOMAIN), **{field: value})


def test_contact_false_string_is_not_truthy_contact():
    try:
        state = ToolState(0.0, Point2(0, 0, Frame.VEHICLE, DOMAIN), contact="false")
    except REJECTION:
        return
    assert state.contact is False


def test_transform_comparison_checks_domain():
    first = Transform2D.identity(Frame.WORLD, DOMAIN)
    second = Transform2D.identity(Frame.WORLD, SpatialDomain.IMAGE_PX)
    try:
        equivalent = first.allclose(second)
    except REJECTION:
        return
    assert not equivalent


def test_polygon_validity_is_translation_invariant():
    triangle = np.array([[0.0, 0.0], [0.02, 0.0], [0.0, 0.02]])
    validate_polygon_points(triangle, "origin")
    validate_polygon_points(triangle + 10000.0, "translated")


def test_cutting_footprint_is_covered_by_actual_collision_footprint():
    tool = ToolGeometry.rectangular("blade", 0.2, 0.08)
    shifted_cut = tool.cutting_polygon.copy()
    shifted_cut[:, 0] += 5e-6
    try:
        changed = replace(tool, cutting_polygon=shifted_cut)
    except REJECTION:
        return
    assert changed.collision_shape.covers(changed.cutting_shape)


def test_scene_does_not_alias_mutable_input_plants():
    scene = build_scene(SceneConfig())
    external_plants = list(scene.plants)
    try:
        changed = replace(scene, plants=external_plants)
    except REJECTION:
        return
    external_plants.clear()
    assert len(changed.plants) == len(scene.plants)


@pytest.mark.parametrize("corruption", ["missing_file", "unknown_root_key", "mixed_domain"])
def test_demo_no_plot_keeps_configuration_guards(tmp_path, corruption):
    tool_path = tmp_path / "tool.yaml"
    payload = yaml.safe_load((ROOT / "configs/tool.yaml").read_text())
    if corruption == "unknown_root_key":
        payload["unknown_configuration_key"] = 123
    elif corruption == "mixed_domain":
        payload["tool"]["domain"] = "image_px"
    if corruption != "missing_file":
        tool_path.write_text(yaml.safe_dump(payload))
    output = tmp_path / "output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/demo_scene.py"),
            "--scene",
            "nominal_three_plants",
            "--tool-config",
            str(tool_path),
            "--output",
            str(output),
            "--no-plot",
        ],
        cwd=ROOT,
        env={**os.environ, "MPLCONFIGDIR": str(tmp_path / "mpl")},
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        artifact = json.loads((output / "nominal_three_plants.json").read_text())
        pytest.fail(
            f"Accepted {corruption}; scene={artifact['domain']}; "
            f"tool={artifact['demo']['tool']['domain']}"
        )
