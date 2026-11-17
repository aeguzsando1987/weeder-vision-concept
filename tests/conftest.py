"""Fixtures compartidas de pytest."""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Raiz del repositorio."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """Directorio `configs/` con los YAML de E1."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def invalid_configs_dir() -> Path:
    """Configuraciones invalidas usadas como casos adversos."""
    return Path(__file__).resolve().parent / "fixtures" / "configs"


@pytest.fixture(scope="session")
def video_fixture_module():
    """Modulo generador de videos sinteticos (requiere el extra `video`)."""
    pytest.importorskip("av")
    pytest.importorskip("cv2")
    import sys

    fixtures_dir = str(Path(__file__).resolve().parent / "fixtures")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    import video_fixture

    return video_fixture


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory, video_fixture_module) -> Path:
    """MP4 H.264 de 24 cuadros a 15 fps, suelo desplazandose 4 px/cuadro hacia arriba."""
    path = tmp_path_factory.mktemp("video") / "synthetic_clip.mp4"
    return video_fixture_module.write_video(path, frames=24, fps=15, shift_px=4)


@pytest.fixture(scope="session")
def planning():
    """Utilidades de prueba de A1/E2: configuraciones, mapas, rutas y peticiones."""
    import sys

    fixtures_dir = str(Path(__file__).resolve().parent / "fixtures")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    import planning_helpers

    return planning_helpers


@pytest.fixture(scope="session")
def mechanics():
    """Utilidades de prueba de A3/E3: modelo 1R, doble de mecanismo y movimientos."""
    import sys

    fixtures_dir = str(Path(__file__).resolve().parent / "fixtures")
    if fixtures_dir not in sys.path:
        sys.path.insert(0, fixtures_dir)
    import mechanism_helpers

    return mechanism_helpers
