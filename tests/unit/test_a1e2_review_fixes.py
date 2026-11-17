"""Regresiones de A1E2-R1..R4 (dictamen A1/E2 rev. 1), con causas y controles positivos."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import pytest

from precision_weeder.core.config import ConfigError, UncertaintyConfig
from precision_weeder.core.errors import PrecisionWeederError, SchemaValidationError
from precision_weeder.core.trajectory import TaskPath
from precision_weeder.core.types import AbstentionReason, PlanStatus
from precision_weeder.geometry.tool_profile import ToolGeometry
from precision_weeder.mapping.local_map import LocalMap
from precision_weeder.mapping.synthetic_adapter import local_map_from_scene
from precision_weeder.planning.base import CandidateRecord, PlanRequest, PlanResult
from precision_weeder.scenarios import build_scene
from precision_weeder.scenarios.planning_cases import build_planning_case, scene_config
from precision_weeder.validation.swept_volume import (
    ValidationReport,
    ValidationStatus,
    validate_task_path,
)


def _json(payload):
    return json.loads(json.dumps(payload))


def _validate(planning, path, local_map, profile=None, safety=None, config=None):
    return validate_task_path(
        path,
        local_map,
        profile or planning.TOOL,
        safety or planning.SAFETY,
        config or planning.VALIDATION,
    )


@pytest.fixture
def accepted(planning) -> PlanResult:
    case = build_planning_case("nominal_three_plants")
    result = planning.run_case(case)
    assert result.accepted
    return result


@pytest.fixture
def abstained(planning) -> PlanResult:
    result = planning.run_case(build_planning_case("blocked_corridor"))
    assert result.status is PlanStatus.ABSTAIN
    return result


@pytest.fixture
def uncertified(planning) -> PlanResult:
    from precision_weeder.core.config import ValidationConfig

    case = build_planning_case("nominal_three_plants")
    result = planning.run_case(case, validation=ValidationConfig(max_subdivision_depth=1))
    assert result.status is PlanStatus.UNCERTIFIED
    return result


# --- R1: ida y vuelta de los seis contratos -------------------------------------------


def test_six_contracts_round_trip_through_json(planning, accepted, abstained, uncertified) -> None:
    case = build_planning_case("nominal_three_plants")
    request = planning.request(case.local_map, case.start_xy_m, case.goal_xy_m)
    report = ValidationReport.from_dict(accepted.to_dict()["validation"])
    contracts = [
        (LocalMap, case.local_map.to_dict()),
        (ToolGeometry, planning.TOOL.to_dict()),
        (TaskPath, accepted.path.to_dict()),
        (PlanRequest, request.to_dict()),
        (ValidationReport, report.to_dict()),
        (PlanResult, accepted.to_dict()),
        (PlanResult, abstained.to_dict()),
        (PlanResult, uncertified.to_dict()),
    ]
    for cls, payload in contracts:
        payload = _json(payload)
        rebuilt = cls.from_dict(payload)
        assert rebuilt.to_dict() == payload, cls.__name__
        assert rebuilt.content_hash() == payload["content_hash"], cls.__name__


def _corrupt(payload, path, value):
    data = copy.deepcopy(payload)
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return data


@pytest.mark.parametrize(
    ("kind", "path", "value", "message"),
    [
        ("profile", ("content_hash",), "0" * 64, "content_hash"),
        ("profile", ("schema_version",), 99, "schema_version"),
        ("profile", ("unit",), "px", "unidad"),
        ("profile", ("extent_x",), 0.3, "derivados"),
        ("profile", ("collision_polygon", 0, 0), math.inf, "no finitos"),
        ("report", ("content_hash",), "0" * 64, "content_hash"),
        ("report", ("schema_version",), 1, "schema_version"),
        ("report", ("validation_scope",), "mechanical", "alcance"),
        ("report", ("envelope_error_max_m",), -1.0, "envelope_error_max_m"),
        ("report", ("clearance_lower_bound_m",), "big", "clearance_lower_bound_m"),
        ("report", ("status",), "REJECTED", "incoherente"),
        ("result", ("content_hash",), "0" * 64, "content_hash"),
        ("result", ("schema_version",), 1, "schema_version"),
        ("result", ("mechanical_feasibility",), "approved", "alcance"),
        ("result", ("chosen", "cost", "total"), "barato", "CandidateRecord.cost"),
        ("result", ("path", "segments", -1, "y_coefficients", 1), 0.5, "content_hash"),
        ("result", ("path", "segments", 0, "x_coefficients", 0), 0.5, "discontinuidad"),
    ],
)
def test_individual_corruption_is_rejected_by_cause(
    planning, accepted, kind, path, value, message
) -> None:
    payloads = {
        "profile": planning.TOOL.to_dict(),
        "report": dict(accepted.to_dict()["validation"]),
        "result": accepted.to_dict(),
    }
    cls = {"profile": ToolGeometry, "report": ValidationReport, "result": PlanResult}[kind]
    corrupted = _corrupt(_json(payloads[kind]), path, value)
    with pytest.raises(PrecisionWeederError, match=message):
        cls.from_dict(corrupted)


def test_accepted_result_rejects_report_for_other_path_map_or_profile(planning, accepted) -> None:
    case = build_planning_case("nominal_three_plants")
    other_path = planning.smooth_path([(0.0, 0.1), (0.35, 0.6), (0.35, 4.9)], case.local_map)
    with pytest.raises(SchemaValidationError, match="path_hash"):
        replace(accepted, path=other_path)
    # Informe valido, pero de otra ruta: no justifica ACCEPTED.
    other_report = _validate(planning, other_path, case.local_map)
    assert other_report.accepted
    with pytest.raises(SchemaValidationError, match="path_hash"):
        replace(accepted, validation=other_report)
    with pytest.raises(SchemaValidationError, match="map_hash"):
        replace(accepted, map_hash="a" * 64)
    with pytest.raises(SchemaValidationError, match="profile_hash"):
        replace(accepted, profile_hash="b" * 64)


@pytest.mark.parametrize("status", ["REJECTED", "UNCERTIFIED"])
def test_non_accepted_or_incomplete_report_cannot_justify_acceptance(
    planning, accepted, status
) -> None:
    payload = accepted.to_dict()["validation"]
    payload["status"] = status
    if status == "REJECTED":
        payload["violations"] = [{"kind": "COLLISION"}]
    else:
        payload["uncertified_intervals"] = [{"segment": 0, "cause": "max_subdivision_depth"}]
    payload.pop("content_hash")
    report = ValidationReport.from_dict({**payload, "content_hash": _report_hash(payload)})
    with pytest.raises(SchemaValidationError, match="ACCEPTED"):
        replace(accepted, validation=report)
    incomplete = accepted.to_dict()["validation"]
    del incomplete["violations"]
    with pytest.raises(SchemaValidationError, match="violations"):
        replace(accepted, validation=incomplete)


def _report_hash(payload):
    from precision_weeder.core.identity import content_hash

    return content_hash(payload)


def test_exports_and_inputs_cannot_mutate_built_artifacts(planning, accepted) -> None:
    exported = accepted.to_dict()
    exported["validation"]["status"] = "REJECTED"
    exported["validation"]["violations"].append({"kind": "COLLISION"})
    exported["search"]["levels"] = -1
    exported["chosen"]["cost"]["total"] = -1.0
    exported["rejected_candidates"].append({})
    assert accepted.validation["status"] == "ACCEPTED"
    assert not accepted.validation["violations"]
    assert accepted.search["levels"] > 0
    assert accepted.chosen.cost["total"] >= 0.0
    for collection in (accepted.validation, accepted.search, accepted.chosen.cost):
        with pytest.raises(TypeError):
            collection["new"] = 1  # type: ignore[index]
    with pytest.raises(AttributeError):
        accepted.validation["violations"].append({})  # type: ignore[union-attr]
    # Entradas mutables del constructor tampoco alcanzan al artefacto construido.
    cost = {"length": 0.1, "total": 0.1}
    record = CandidateRecord(0, ((0.0, 0.0), (0.0, 1.0)), cost, "accepted", {"a": [1]})
    cost["total"] = 99.0
    assert record.cost["total"] == 0.1
    report = ValidationReport.from_dict(accepted.to_dict()["validation"])
    exported_report = report.to_dict()
    exported_report["effective_geometry"]["tool_inflation_m"] = 9.0
    assert report.effective_geometry["tool_inflation_m"] == 0.0


def test_hash_scopes_are_documented_and_distinct(accepted) -> None:
    payload = accepted.to_dict()
    assert payload["config_hash_scope"] == "planner + validation + safety"
    assert payload["validation"]["config_hash_scope"].startswith("validation + safety")
    assert payload["config_hash"] != payload["validation"]["config_hash"]


def test_result_content_hash_excludes_wall_time(planning) -> None:
    case = build_planning_case("nominal_three_plants")
    a, b = planning.run_case(case), planning.run_case(case)
    assert a.content_hash() == b.content_hash()
    assert a.to_dict(include_timing=False) == b.to_dict(include_timing=False)


# --- R2: corte concavo -------------------------------------------------------------------


def _l_cut_profile(planning) -> ToolGeometry:
    return ToolGeometry(
        tool_id="concave_cut",
        collision_polygon=[[0, 0], [0.2, 0], [0.2, 0.2], [0, 0.2]],
        cutting_polygon=[[0, 0], [0.2, 0], [0.2, 0.05], [0.05, 0.05], [0.05, 0.2], [0, 0.2]],
        pivot=planning.point(0, 0, frame=planning.Frame.TOOL),
    )


def test_concave_cut_is_unsupported_in_planner_and_validator(planning) -> None:
    """La L sobreestimaba la cobertura (0.12875 frente a 0.1175 m2 reales)."""
    profile = _l_cut_profile(planning)
    assert profile.is_convex() and not profile.is_cut_convex()
    local_map = planning.custom_map()
    report = _validate(planning, planning.linear_path([(0, 0.3), (0, 0.8)], local_map, profile),
                       local_map, profile)
    assert report.status is ValidationStatus.REJECTED
    assert [v["kind"] for v in report.violations] == ["UNSUPPORTED_FEATURE"]
    assert "corte no convexa" in report.violations[0]["detail"]
    assert not report.coverage
    request = planning.request(local_map, (0.0, 0.3), (0.0, 0.8), profile=profile)
    result = planning.planner().plan(request, local_map, profile)
    assert result.status is PlanStatus.UNSUPPORTED
    assert result.reason is AbstentionReason.UNSUPPORTED_FEATURE and result.path is None


def test_convex_rectangle_coverage_regression_stays_exact(planning) -> None:
    local_map = planning.custom_map()
    report = _validate(planning, planning.linear_path([(0, 0.3), (0, 0.8)], local_map), local_map)
    assert report.coverage["exact"] is True
    assert report.coverage["swept_cut_area_m2"] == pytest.approx(0.19 * (0.5 + 0.07), abs=1e-12)


def test_e1_geometry_with_concave_cut_is_still_constructible(planning) -> None:
    """El rechazo es de capacidad del planner/validador, no del constructor E1."""
    assert _l_cut_profile(planning).cutting_area == pytest.approx(0.0175, abs=1e-15)


# --- R3: roles, contacto y tolerancia ------------------------------------------------------


@pytest.mark.parametrize("role", ["elevacion_auxiliar", "franja_izquierda", "Pasada_Principal"])
def test_unsupported_role_is_explicit_in_planner(planning, role) -> None:
    local_map = planning.custom_map()
    request = replace(planning.request(local_map, (0, 0.3), (0, 0.8)), role=role)
    result = planning.planner().plan(request, local_map, planning.TOOL)
    assert result.status is PlanStatus.UNSUPPORTED
    assert result.reason is AbstentionReason.UNSUPPORTED_FEATURE
    assert result.path is None and "rol" in result.detail


def test_supported_role_still_plans(planning) -> None:
    local_map = planning.custom_map()
    result = planning.planner().plan(
        planning.request(local_map, (0, 0.3), (0, 0.8)), local_map, planning.TOOL
    )
    assert result.accepted and result.path.role == "pasada_principal"


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"role": "elevacion_auxiliar"}, "rol"),
        ({"working_contact": False}, "working_contact"),
        ({"task_tolerance_m": 0.01}, "task_tolerance_m"),
    ],
)
def test_external_path_outside_e2_profile_gets_no_work_certificate(
    planning, changes, fragment
) -> None:
    local_map = planning.custom_map()
    base = planning.linear_path([(0, 0.3), (0, 0.8)], local_map)
    path = replace(base, **changes)
    report = _validate(planning, path, local_map)
    assert report.status is ValidationStatus.REJECTED
    assert report.violations[0]["kind"] == "UNSUPPORTED_FEATURE"
    assert fragment in report.violations[0]["detail"]
    assert _validate(planning, base, local_map).accepted


# --- R4: presupuesto de discretizacion ---------------------------------------------------------


def test_discretization_budget_is_enforced_in_validator_and_planner(planning) -> None:
    local_map = planning.custom_map([planning.exclusion("P", 0.3, 2.0, 0.1)])
    strict = planning.SAFETY.model_copy(update={"max_conservative_discretization_excess_m": 1e-8})
    path = planning.linear_path([(-0.2, 0.3), (-0.2, 0.8)], local_map)
    with pytest.raises(ConfigError, match="max_conservative_discretization_excess_m"):
        _validate(planning, path, local_map, safety=strict)
    request = planning.request(local_map, (-0.2, 0.3), (-0.2, 0.8))
    with pytest.raises(ConfigError, match="entrada incompatible"):
        planning.planner(safety=strict).plan(request, local_map, planning.TOOL)
    # Bajo el limite configurado por defecto: se valida normalmente.
    assert _validate(planning, path, local_map).accepted


def test_discretization_budget_boundary_follows_e1_semantics(planning) -> None:
    """E1-R12: se rechaza solo si el exceso supera el limite (`excess <= limit` pasa)."""
    local_map = planning.custom_map([planning.exclusion("P", 0.3, 2.0, 0.1)])
    excess = local_map.exclusions[0].discretization_excess
    path = planning.linear_path([(-0.2, 0.3), (-0.2, 0.8)], local_map)
    at_limit = planning.SAFETY.model_copy(
        update={"max_conservative_discretization_excess_m": excess}
    )
    assert _validate(planning, path, local_map, safety=at_limit).accepted
    below = planning.SAFETY.model_copy(
        update={"max_conservative_discretization_excess_m": math.nextafter(excess, 0.0)}
    )
    with pytest.raises(ConfigError):
        _validate(planning, path, local_map, safety=below)


@pytest.mark.parametrize("inflate", ["exclusions", "tool"])
def test_budget_checks_effective_exclusions_without_double_margin(planning, inflate) -> None:
    """Con inflate=tool la exclusion efectiva mide 0.15 m (no 0.18): el presupuesto se
    evalua sobre la geometria que llega al validador."""
    margin = 0.03 if inflate == "exclusions" else 0.0
    config = scene_config("budget", (2.0,), (0.0,), margin_m=margin)
    local_map = local_map_from_scene(
        build_scene(config, uncertainty=UncertaintyConfig(inflate=inflate))
    )
    radius = local_map.exclusions[0].total_radius
    assert radius == pytest.approx(0.18 if inflate == "exclusions" else 0.15)
    excess = local_map.exclusions[0].discretization_excess
    path = planning.linear_path([(0.35, 0.1), (0.35, 4.9)], local_map)
    exact = planning.SAFETY.model_copy(update={"max_conservative_discretization_excess_m": excess})
    assert _validate(planning, path, local_map, safety=exact).accepted
