"""Integracion E1: configuracion -> escena -> herramienta -> huellas en el mundo.

Comprueba la cadena completa que E2 usara como entrada, sin planificador ni
validador (que pertenecen a etapas posteriores).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from precision_weeder.core.config import load_project_config
from precision_weeder.core.types import Frame, Point2
from precision_weeder.core.units import CLEARANCE_EPSILON_M
from precision_weeder.geometry.frames import world_from_vehicle
from precision_weeder.scenarios.synthetic import build_scene
from precision_weeder.simulation.tool import ToolState, tool_geometry_from_config, world_footprints


@pytest.fixture
def pieces(configs_dir: Path):
    project = load_project_config(
        configs_dir / "synthetic.yaml",
        configs_dir / "tool.yaml",
        configs_dir / "safety.yaml",
        configs_dir / "kinematics.yaml",
    )
    return project, build_scene(project.scene), tool_geometry_from_config(project.tool)


def test_configured_scene_and_tool_share_one_domain(pieces) -> None:
    project, scene, blade = pieces
    assert scene.domain is blade.domain is project.scene.domain


def test_blade_at_corridor_center_collides_with_the_first_exclusion(pieces) -> None:
    """Pasar recto por el centro intersecta la exclusion: una linea libre no basta."""
    project, scene, blade = pieces
    first = scene.exclusions[0]
    state = ToolState(
        time_s=0.0,
        position=Point2(x=0.0, y=0.0, frame=Frame.VEHICLE, domain=scene.domain),
    )
    collision, _ = world_footprints(
        blade, state, world_from_vehicle(0.0, first.center.y, 0.0, scene.domain)
    )
    assert collision.intersects(first.as_polygon())
    _ = project


def test_lateral_offset_within_limits_clears_the_exclusion(pieces) -> None:
    """Existe al menos un desvio factible dentro de los limites declarados."""
    project, scene, blade = pieces
    first = scene.exclusions[0]
    offset = first.total_radius + 0.5 * blade.extent_x + 0.01
    assert offset <= project.kinematics.position_limits_m.x_max
    state = ToolState(
        time_s=0.0,
        position=Point2(x=offset, y=0.0, frame=Frame.VEHICLE, domain=scene.domain),
    )
    collision, _ = world_footprints(
        blade, state, world_from_vehicle(0.0, first.center.y, 0.0, scene.domain)
    )
    assert not collision.intersects(first.as_polygon())
    assert collision.distance(first.as_polygon()) > CLEARANCE_EPSILON_M
    assert scene.corridor.as_polygon().covers(collision)


def test_blocked_scenario_has_no_clearing_offset(pieces) -> None:
    """En el corredor bloqueado ningun desplazamiento lateral deja pasar la cuchilla."""
    from precision_weeder.scenarios.fixtures import build_fixture

    project, _, blade = pieces
    scene = build_fixture("blocked_corridor")
    limits = project.kinematics.position_limits_m
    offsets = [limits.x_min + step * (limits.x_max - limits.x_min) / 200.0 for step in range(201)]
    exclusions = [exclusion.as_polygon() for exclusion in scene.exclusions]
    band_y = scene.exclusions[0].center.y
    cleared = []
    for offset in offsets:
        state = ToolState(
            time_s=0.0,
            position=Point2(x=offset, y=0.0, frame=Frame.VEHICLE, domain=scene.domain),
        )
        collision, _ = world_footprints(
            blade, state, world_from_vehicle(0.0, band_y, 0.0, scene.domain)
        )
        if all(
            collision.distance(exclusion) > CLEARANCE_EPSILON_M for exclusion in exclusions
        ) and scene.corridor.as_polygon().covers(collision):
            cleared.append(offset)
    assert cleared == []


def test_narrow_passage_has_a_clearing_offset_only_near_the_center(pieces) -> None:
    from precision_weeder.scenarios.fixtures import build_fixture

    project, _, blade = pieces
    scene = build_fixture("narrow_passage")
    limits = project.kinematics.position_limits_m
    exclusions = [exclusion.as_polygon() for exclusion in scene.exclusions]
    band_y = scene.exclusions[0].center.y
    cleared = []
    for step in range(401):
        offset = limits.x_min + step * (limits.x_max - limits.x_min) / 400.0
        state = ToolState(
            time_s=0.0,
            position=Point2(x=offset, y=0.0, frame=Frame.VEHICLE, domain=scene.domain),
        )
        collision, _ = world_footprints(
            blade, state, world_from_vehicle(0.0, band_y, 0.0, scene.domain)
        )
        if all(
            collision.distance(exclusion) > CLEARANCE_EPSILON_M for exclusion in exclusions
        ) and scene.corridor.as_polygon().covers(collision):
            cleared.append(offset)
    assert cleared
    assert max(abs(offset) for offset in cleared) < 0.011
