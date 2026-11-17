"""Sondas de auditoria A1/E2 del revisor tecnico, integradas en la suite.

Origen: la evidencia de la revision (no versionada) (A1E2-R1
contratos, R2 corte concavo, R3 roles, R4 presupuesto de discretizacion), conservado
sin modificar. Adaptaciones: `ROOT` se calcula desde la nueva ubicacion de este
archivo; el formato lo normaliza Ruff y se marcan con `noqa` las importaciones
tardias y sin uso del original. No cambia ninguna asercion. Las causas concretas y los
controles positivos estan en `tests/unit/test_a1e2_review_fixes.py`.
"""

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests/fixtures"))
import planning_helpers as h  # noqa: E402

from precision_weeder.core.errors import PrecisionWeederError  # noqa: E402
from precision_weeder.geometry.tool_profile import ToolGeometry  # noqa: E402
from precision_weeder.planning.base import PlanResult  # noqa: E402, F401
from precision_weeder.validation.swept_volume import (  # noqa: E402
    ValidationReport,  # noqa: F401
    validate_task_path,
)


def accepted():
    m = h.custom_map()
    return h.planner().plan(h.request(m, (0, 0.3), (0, 0.8)), m, h.TOOL)


@pytest.mark.parametrize("kind", ["profile", "result", "report"])
def test_missing_round_trip(kind):
    m = h.custom_map()
    path = h.linear_path([(0, 0.3), (0, 0.8)], m)
    obj = {
        "profile": h.TOOL,
        "result": accepted(),
        "report": validate_task_path(path, m, h.TOOL, h.SAFETY, h.VALIDATION),
    }[kind]
    cls = type(obj)
    assert callable(getattr(cls, "from_dict", None)), f"{cls.__name__}: falta round-trip validado"
    payload = json.loads(json.dumps(obj.to_dict()))
    assert cls.from_dict(payload).to_dict() == payload


def test_result_rejects_report_for_other_path():
    result = accepted()
    m = h.custom_map()
    other = h.linear_path([(0.1, 0.3), (0.1, 0.8)], m)
    assert other.content_hash() != result.validation["path_hash"]
    with pytest.raises(PrecisionWeederError):
        replace(result, path=other)


def test_export_does_not_mutate_accepted_result():
    result = accepted()
    exported = result.to_dict()
    exported["validation"]["status"] = "REJECTED"
    assert result.validation["status"] == "ACCEPTED"


def test_report_export_has_owned_collections():
    m = h.custom_map()
    path = h.linear_path([(0, 0.3), (0, 0.8)], m)
    report = validate_task_path(path, m, h.TOOL, h.SAFETY, h.VALIDATION)
    report.to_dict()["violations"].append({"kind": "COLLISION"})
    assert not report.violations


def test_nonconvex_cut_is_not_claimed_exact_by_convex_hull():
    # L = unión de dos rectángulos. Barrido vertical 0.5 m:
    # 0.05*(0.20+0.50) + 0.15*(0.05+0.50) = 0.1175 m².
    m = h.custom_map()
    profile = ToolGeometry(
        tool_id="concave_cut",
        collision_polygon=[[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]],
        cutting_polygon=[[0, 0], [0.2, 0], [0.2, 0.05], [0.05, 0.05], [0.05, 0.2], [0, 0.2]],
        pivot=h.point(0, 0, frame=h.Frame.TOOL),
    )
    path = h.linear_path([(0, 0.3), (0, 0.8)], m, profile)
    report = validate_task_path(path, m, profile, h.SAFETY, h.VALIDATION)
    # Alternativa mínima admitida: rechazo explícito de la capacidad no soportada.
    if not report.accepted:
        assert any(v["kind"] == "UNSUPPORTED_FEATURE" for v in report.violations)
        return
    assert report.coverage["swept_cut_area_m2"] == pytest.approx(0.1175, abs=1e-12)


def test_unsupported_role_is_explicit():
    m = h.custom_map()
    req = replace(h.request(m, (0, 0.3), (0, 0.8)), role="elevacion_auxiliar")
    result = h.planner().plan(req, m, h.TOOL)
    assert result.status.value == "UNSUPPORTED"
    assert result.reason.value == "UNSUPPORTED_FEATURE"
    assert result.path is None


def test_discretization_cap_is_enforced():
    m = h.custom_map([h.exclusion("P", 0.3, 2.0, 0.1)])
    safety = h.SAFETY.model_copy(update={"max_conservative_discretization_excess_m": 1e-8})
    assert m.exclusions[0].discretization_excess > safety.max_conservative_discretization_excess_m
    path = h.linear_path([(-0.2, 0.3), (-0.2, 0.8)], m)
    with pytest.raises(PrecisionWeederError):
        validate_task_path(path, m, h.TOOL, safety, h.VALIDATION)
