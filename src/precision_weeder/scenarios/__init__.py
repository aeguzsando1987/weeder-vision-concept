"""Escenarios sinteticos y casos adversos."""

from __future__ import annotations

from .fixtures import SCENE_BUILDERS, build_fixture, default_tool_geometry, fixture_names
from .synthetic import SyntheticScene, build_scene

__all__ = [
    "SCENE_BUILDERS",
    "SyntheticScene",
    "build_fixture",
    "build_scene",
    "default_tool_geometry",
    "fixture_names",
]
