"""Rutas y utilidades compartidas por las pruebas M0-P.

Modulo con nombre propio (no `conftest`): `tests/conftest.py` del proyecto principal
tambien se llama `conftest` y, segun el orden de coleccion, ocuparia ese nombre.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

M0_DIR = Path(__file__).resolve().parents[1]
ROOT = M0_DIR.parents[1]
if str(M0_DIR) not in sys.path:
    sys.path.insert(0, str(M0_DIR))

PROFILES = {
    "cfr_640": M0_DIR / "profiles/fixture_cfr_640.yaml",
    "cfr_640_blocked": M0_DIR / "profiles/fixture_cfr_640.yaml",
    "vfr_960": M0_DIR / "profiles/fixture_vfr_960.yaml",
    "vfr_960_blocked": M0_DIR / "profiles/fixture_vfr_960.yaml",
}


def load(name: str):
    from m0p.profile import load_profile

    return load_profile(PROFILES[name], ROOT)


def read_sidecar(path: Path) -> tuple[dict, list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows[0], rows[1:]
