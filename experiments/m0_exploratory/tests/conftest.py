"""Fixtures de pruebas M0-P: videos sinteticos y exportaciones exactas, sin GPU ni pesos."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from m0p_testing import PROFILES, ROOT, load, read_sidecar

pytest.importorskip("av")
pytest.importorskip("cv2")

from m0p.fixtures import fixture_spec
from m0p.observations import ObservationSet
from m0p.pipeline import run_pipeline
from m0p.synthetic import write_fixture


@pytest.fixture(scope="session")
def fixture_files(tmp_path_factory) -> dict[str, tuple[Path, Path]]:
    """Video y exportacion de cada fixture del catalogo congelado."""
    base = tmp_path_factory.mktemp("m0p_fixtures")
    files = {}
    for name in PROFILES:
        video = base / f"{name}.mp4"
        export = write_fixture(fixture_spec(name), video)
        export_path = base / f"{name}.json"
        export_path.write_text(json.dumps(export))
        files[name] = (video, export_path)
    return files


@pytest.fixture(scope="session")
def runs(fixture_files, tmp_path_factory) -> dict[str, tuple]:
    """Ejecucion de referencia por fixture: (resumen, cabecera, registros, salida)."""
    out_dir = tmp_path_factory.mktemp("m0p_runs")
    results = {}
    for name, (video, export) in fixture_files.items():
        profile, resolved = load(name)
        observations = ObservationSet.from_payload(json.loads(export.read_text()))
        output = out_dir / f"{name}_route.mp4"
        summary = run_pipeline(video, profile, resolved, observations, output, ROOT)
        header, records = read_sidecar(Path(summary.sidecar))
        results[name] = (summary, header, records, output)
    return results
