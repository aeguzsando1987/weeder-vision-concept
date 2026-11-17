"""A1-01..A1-04: contratos de tarea, identidad, rechazos y desacoplamiento."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from precision_weeder.core.errors import (
    FrameMismatchError,
    InvalidPolygonError,
    NonFiniteValueError,
    PrecisionWeederError,
    ScenarioError,
    SchemaValidationError,
    UncertaintyBudgetError,
    UnitMismatchError,
)
from precision_weeder.core.trajectory import (
    OrientationPolicy,
    OrientationRequirement,
    PathSegment,
    TaskPath,
)
from precision_weeder.core.types import (
    AbstentionReason,
    CircularExclusion,
    Frame,
    InflationTarget,
    PlanStatus,
    UncertaintyBound,
    UncertaintyPolicy,
)
from precision_weeder.core.units import Provenance, SpatialDomain
from precision_weeder.geometry import tool_profile
from precision_weeder.mapping.local_map import LocalMap
from precision_weeder.planning import PlanRequest, PlanResult
from precision_weeder.simulation import tool as simulation_tool

# --- A1-02: TaskPath ------------------------------------------------------------


def _segment(**overrides):
    values = {
        "s_start": 0.0,
        "s_end": 1.0,
        "x_coefficients": [0.0, 0.1, 0, 0, 0, 0],
        "y_coefficients": [0.1, 4.8, 0, 0, 0, 0],
    }
    values.update(overrides)
    return PathSegment(**values)


def test_task_path_round_trip_identity_and_units(planning) -> None:
    local_map = planning.custom_map()
    path = planning.smooth_path([(0.0, 0.1), (0.2, 1.0), (0.2, 4.9)], local_map)
    payload = json.loads(json.dumps(path.to_dict()))
    again = TaskPath.from_dict(payload)
    assert again.content_hash() == path.content_hash() == payload["content_hash"]
    assert payload["unit"] == "m" and payload["frame"] == "W"
    assert "no es tiempo" in payload["parameter"]
    # Ni configuraciones articulares ni tiempos inventados en el artefacto de tarea.
    flat = json.dumps(payload)
    assert '"q"' not in flat and "time" not in flat.lower().replace("no es tiempo", "")
    for s in (0.0, 0.3, 0.5, 1.0):
        assert again.evaluate(s) == path.evaluate(s)


def test_task_path_hash_changes_with_geometry_and_tampering_is_rejected(planning) -> None:
    local_map = planning.custom_map()
    a = planning.linear_path([(0.0, 0.1), (0.0, 4.9)], local_map)
    b = planning.linear_path([(0.0, 0.1), (0.0, 4.8)], local_map)
    assert a.content_hash() != b.content_hash()
    payload = a.to_dict()
    payload["segments"][0]["x_coefficients"][0] = 0.001
    with pytest.raises(SchemaValidationError, match="content_hash"):
        TaskPath.from_dict(payload)


def test_path_segment_owns_immutable_coefficients() -> None:
    source = np.array([0.0, 1.0, 0, 0, 0, 0])
    segment = _segment(x_coefficients=source)
    source[0] = 99.0
    assert segment.x_coefficients[0] == 0.0
    with pytest.raises(ValueError):
        segment.x_coefficients[0] = 5.0
    with pytest.raises(ValueError):
        segment.x_coefficients.setflags(write=True)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"x_coefficients": [0.0, math.nan, 0, 0, 0, 0]}, NonFiniteValueError),
        ({"y_coefficients": [0.0, math.inf, 0, 0, 0, 0]}, NonFiniteValueError),
        ({"x_coefficients": [0.0, 1.0]}, SchemaValidationError),
        ({"x_coefficients": [[0.0] * 6]}, SchemaValidationError),
        ({"s_end": 0.0}, SchemaValidationError),
        ({"s_start": 0.6, "s_end": 0.5}, SchemaValidationError),
        ({"s_end": math.nan}, NonFiniteValueError),
    ],
)
def test_path_segment_rejects_invalid_values(overrides, error) -> None:
    with pytest.raises(error):
        _segment(**overrides)


def _path_fields(planning, **overrides):
    local_map = planning.custom_map()
    fields = {
        "path_id": "p",
        "domain": SpatialDomain.GROUND_METRIC_SYNTHETIC,
        "segments": (_segment(),),
        "orientation": OrientationRequirement.fixed(0.0),
        "profile_id": planning.TOOL.tool_id,
        "profile_hash": planning.TOOL.content_hash(),
        "map_id": local_map.map_id,
        "map_hash": local_map.content_hash(),
    }
    fields.update(overrides)
    return fields


@pytest.mark.parametrize("domain", [SpatialDomain.IMAGE_PX, SpatialDomain.GRAPHIC_UNSCALED])
def test_task_path_rejects_pixels_and_graphic_units(planning, domain) -> None:
    with pytest.raises(UnitMismatchError):
        TaskPath(**_path_fields(planning, domain=domain))


def test_task_path_accepts_assumed_and_verified_ground_as_separate_modes(planning) -> None:
    for domain in (SpatialDomain.GROUND_ASSUMED, SpatialDomain.GROUND_METRIC_VERIFIED):
        assert TaskPath(**_path_fields(planning, domain=domain)).domain is domain


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"frame": Frame.VEHICLE}, FrameMismatchError),
        ({"frame": Frame.IMAGE}, FrameMismatchError),
        ({"segments": ()}, SchemaValidationError),
        ({"profile_hash": "abc"}, SchemaValidationError),
        ({"map_hash": "G" * 64}, SchemaValidationError),
        ({"working_contact": 1}, SchemaValidationError),
        ({"task_tolerance_m": -0.1}, SchemaValidationError),
        ({"continuity_declared": "C3"}, SchemaValidationError),
        ({"schema_version": 2}, SchemaValidationError),
    ],
)
def test_task_path_rejects_invalid_contract_fields(planning, overrides, error) -> None:
    with pytest.raises(error):
        TaskPath(**_path_fields(planning, **overrides))


@pytest.mark.parametrize(
    "segments",
    [
        # Hueco en s.
        (
            _segment(s_end=0.5),
            _segment(s_start=0.6, x_coefficients=[0.05, 0, 0, 0, 0, 0]),
        ),
        # No empieza en 0.
        (_segment(s_start=0.1),),
        # No termina en 1.
        (_segment(s_end=0.9),),
        # Discontinuidad de posicion en la union.
        (
            _segment(s_end=0.5),
            _segment(s_start=0.5, x_coefficients=[0.5, 0, 0, 0, 0, 0]),
        ),
    ],
)
def test_task_path_rejects_invalid_s_partition(planning, segments) -> None:
    with pytest.raises(SchemaValidationError):
        TaskPath(**_path_fields(planning, segments=segments))


def test_derivatives_are_with_respect_to_global_s(planning) -> None:
    """Oraculo: tramo lineal en s∈[0.25, 0.75] con Δy=1 => dy/ds = 1/0.5 = 2."""
    first = _segment(
        s_end=0.25, x_coefficients=[0, 0, 0, 0, 0, 0], y_coefficients=[0, 0.5, 0, 0, 0, 0]
    )
    middle = PathSegment(0.25, 0.75, [0, 0, 0, 0, 0, 0], [0.5, 1.0, 0, 0, 0, 0])
    last = PathSegment(0.75, 1.0, [0, 0, 0, 0, 0, 0], [1.5, 0.5, 0, 0, 0, 0])
    path = TaskPath(**_path_fields(planning, segments=(first, middle, last)))
    assert path.evaluate(0.5, 1) == pytest.approx((0.0, 2.0), abs=1e-15)
    assert path.evaluate(0.5) == pytest.approx((0.0, 1.0), abs=1e-15)
    with pytest.raises(SchemaValidationError):
        path.evaluate(1.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "FIXED"},
        {"policy": "FIXED", "theta_rad": math.nan},
        {"policy": "INTERVAL", "interval_rad": (0.5, 0.1)},
        {"policy": "INTERVAL", "theta_rad": 0.0, "interval_rad": (0.0, 0.1)},
        {"policy": "FREE", "theta_rad": 0.0},
        {"policy": "FIXED", "theta_rad": 0.0, "frame": Frame.VEHICLE},
        {"policy": "SOMETIMES"},
    ],
)
def test_orientation_requirement_is_explicit(kwargs) -> None:
    with pytest.raises((SchemaValidationError, ValueError)):
        OrientationRequirement(**kwargs)


def test_free_orientation_is_not_zero() -> None:
    free = OrientationRequirement(policy=OrientationPolicy.FREE)
    assert free.theta_rad is None and free.to_dict()["theta_rad"] is None


# --- A1-02: LocalMap -----------------------------------------------------------------


def test_local_map_round_trip_and_owned_arrays(planning) -> None:
    exclusions = [planning.exclusion("P1", 0.0, 2.0, 0.18)]
    ring = [[-0.2, 1.0], [0.2, 1.0], [0.2, 1.2], [-0.2, 1.2]]
    local_map = planning.custom_map(exclusions, unknown=[ring])
    payload = json.loads(json.dumps(local_map.to_dict()))
    assert LocalMap.from_dict(payload).content_hash() == local_map.content_hash()
    ring[0][0] = 9.0
    assert local_map.unknown_regions[0][0, 0] == -0.2
    with pytest.raises(ValueError):
        local_map.known_region[0, 0] = 1.0
    payload["corridor"]["x_max"] = 0.6
    with pytest.raises(SchemaValidationError, match="content_hash"):
        LocalMap.from_dict(payload)


def test_local_map_hash_changes_with_geometry_or_tolerance(planning) -> None:
    a = planning.custom_map([planning.exclusion("P1", 0.0, 2.0, 0.18)])
    b = planning.custom_map([planning.exclusion("P1", 0.0, 2.0, 0.181)])
    c = planning.custom_map([planning.exclusion("P1", 0.0, 2.0, 0.18, segments=65)])
    assert len({a.content_hash(), b.content_hash(), c.content_hash()}) == 3


def _map_kwargs(planning, **overrides):
    base = planning.custom_map([planning.exclusion("P1", 0.0, 2.0, 0.18)])
    fields = {
        "map_id": base.map_id,
        "segment_id": base.segment_id,
        "domain": base.domain,
        "corridor": base.corridor,
        "known_region": base.known_region,
        "known_region_basis": base.known_region_basis,
        "exclusions": base.exclusions,
        "uncertainty": base.uncertainty,
    }
    fields.update(overrides)
    return fields


def _policy(total: float, inflate: InflationTarget) -> UncertaintyPolicy:
    bound = UncertaintyBound(total, 0.0, 0.0, SpatialDomain.GROUND_METRIC_SYNTHETIC)
    return UncertaintyPolicy(bound=bound, budget=total, inflate=inflate)


@pytest.mark.parametrize(
    ("builder", "error"),
    [
        (lambda h: {"domain": SpatialDomain.IMAGE_PX}, UnitMismatchError),
        (lambda h: {"frame": Frame.VEHICLE}, FrameMismatchError),
        (lambda h: {"exclusions": ()}, ScenarioError),
        (
            lambda h: {
                "exclusions": (
                    CircularExclusion("P1", h.point(0.0, 2.0, frame=Frame.VEHICLE), 0.15, 0.0),
                )
            },
            FrameMismatchError,
        ),
        (lambda h: {"known_region": [[0, 0], [1, 1], [2, 2]]}, InvalidPolygonError),
        (lambda h: {"known_region": [[0, 0], [1, 0], [math.nan, 1]]}, NonFiniteValueError),
        (lambda h: {"static": False}, SchemaValidationError),
        (lambda h: {"valid_from_s": 0.0}, SchemaValidationError),
        (lambda h: {"version": 0}, SchemaValidationError),
        (lambda h: {"known_region_basis": " "}, SchemaValidationError),
        # Incertidumbre aplicada dos veces o sin cubrir la cota.
        (
            lambda h: {"uncertainty": _policy(0.03, InflationTarget.EXCLUSIONS)},
            UncertaintyBudgetError,
        ),
        (
            lambda h: {
                "uncertainty": _policy(0.0, InflationTarget.EXCLUSIONS),
                "tool_inflation": 0.03,
            },
            UncertaintyBudgetError,
        ),
        (lambda h: {"uncertainty": _policy(0.03, InflationTarget.TOOL)}, UncertaintyBudgetError),
    ],
)
def test_local_map_rejects_invalid_inputs(planning, builder, error) -> None:
    with pytest.raises(error):
        LocalMap(**_map_kwargs(planning, **builder(planning)))


def test_empty_map_requires_explicit_certification(planning) -> None:
    with pytest.raises(ScenarioError, match="no certifica"):
        LocalMap(**_map_kwargs(planning, exclusions=()))
    assert LocalMap(**_map_kwargs(planning, exclusions=(), free_space_certified=True))


def test_unknown_region_is_forbidden_space(planning) -> None:
    band = [[-0.5, 2.0], [0.5, 2.0], [0.5, 2.3], [-0.5, 2.3]]
    local_map = planning.custom_map(unknown=[band])
    assert local_map.allowed_region().area == pytest.approx(5.0 - 0.3, abs=1e-12)
    assert local_map.unknown_union().area == pytest.approx(0.3, abs=1e-12)


# --- A1-02: PlanRequest / PlanResult --------------------------------------------------


def test_plan_request_round_trip_and_rejections(planning) -> None:
    local_map = planning.custom_map()
    request = planning.request(local_map, (0.0, 0.1), (0.0, 4.9))
    payload = json.loads(json.dumps(request.to_dict()))
    assert PlanRequest.from_dict(payload).content_hash() == request.content_hash()
    with pytest.raises(SchemaValidationError, match="por delante"):
        planning.request(local_map, (0.0, 2.0), (0.0, 1.0))
    with pytest.raises(SchemaValidationError, match="por delante"):
        planning.request(local_map, (0.0, 2.0), (0.3, 2.0))
    fields = {k: getattr(request, k) for k in request.__dataclass_fields__}
    with pytest.raises(FrameMismatchError):
        PlanRequest(**{**fields, "start": planning.point(0, 0.1, frame=Frame.VEHICLE)})
    with pytest.raises(UnitMismatchError):
        PlanRequest(**{**fields, "start": planning.point(0, 0.1, domain=SpatialDomain.IMAGE_PX)})
    with pytest.raises(SchemaValidationError):
        PlanRequest(**{**fields, "map_hash": "nothash"})
    # La peticion no admite estado articular ni mecanismo.
    assert not {"q", "tool_state", "mechanism", "video", "calibration"} & set(fields)


def test_plan_result_never_carries_an_implicit_route(planning) -> None:
    local_map = planning.custom_map()
    path = planning.linear_path([(0.0, 0.1), (0.0, 4.9)], local_map)
    common = {
        "request_hash": None,
        "map_hash": "a" * 64,
        "profile_hash": "b" * 64,
        "config_hash": "c" * 64,
    }
    with pytest.raises(SchemaValidationError):
        PlanResult(PlanStatus.ABSTAIN, AbstentionReason.NO_PATH_FOUND, "x", path=path, **common)
    with pytest.raises(SchemaValidationError):
        PlanResult(PlanStatus.ACCEPTED, None, "x", **common)
    with pytest.raises(SchemaValidationError):
        PlanResult(
            PlanStatus.ACCEPTED, None, "x", path=path, validation={"status": "REJECTED"}, **common
        )
    with pytest.raises(SchemaValidationError):
        PlanResult(PlanStatus.ABSTAIN, None, "x", **common)


# --- perfil de herramienta -----------------------------------------------------------------


def test_tool_profile_is_a_single_implementation_with_identity(planning) -> None:
    assert simulation_tool.ToolGeometry is tool_profile.ToolGeometry is tool_profile.ToolProfile
    assert simulation_tool.tool_geometry_from_config is tool_profile.tool_geometry_from_config
    base = planning.TOOL
    other = tool_profile.ToolGeometry.rectangular(
        "blade_rect_200x80", 0.20, 0.081, cutting_inset=0.005
    )
    assert base.content_hash() != other.content_hash()
    assert base.to_dict()["content_hash"] == base.content_hash()
    assert base.is_convex()
    l_shape = tool_profile.ToolGeometry(
        tool_id="l_blade",
        collision_polygon=[[0, 0], [0.2, 0], [0.2, 0.05], [0.05, 0.05], [0.05, 0.2], [0, 0.2]],
        cutting_polygon=[[0.01, 0.01], [0.04, 0.01], [0.04, 0.04], [0.01, 0.04]],
        pivot=planning.point(0.0, 0.0, frame=Frame.TOOL),
    )
    assert not l_shape.is_convex()
    assert "q" not in base.to_dict() and "joint" not in json.dumps(base.to_dict())


# --- A1-03: desacoplamiento ---------------------------------------------------------------


def test_planner_import_does_not_load_mechanics_simulation_or_video() -> None:
    code = (
        "import sys\n"
        "import precision_weeder.planning.geometric\n"
        "import precision_weeder.validation.swept_volume\n"
        "import precision_weeder.mapping.local_map\n"
        "import precision_weeder.core.trajectory\n"
        "banned = ['cv2', 'av', 'precision_weeder.kinematics', 'precision_weeder.simulation',\n"
        "          'precision_weeder.video', 'precision_weeder.scenarios', 'matplotlib']\n"
        "loaded = [m for m in banned\n"
        "          if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules)]\n"
        "print(','.join(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == ""


class _SwingArmDouble:
    """Doble de prueba con q de dimension 1; NO es un modelo mecanico."""

    q_dimension = 1

    def consume(self, payload: dict) -> dict:
        path = TaskPath.from_dict(payload)
        return {"q_dimension": self.q_dimension, "segments": len(path.segments)}


class _ThreeJointDouble:
    """Doble de prueba con q de dimension 3; NO es un modelo mecanico."""

    q_dimension = 3

    def consume(self, payload: dict) -> dict:
        path = TaskPath.from_dict(payload)
        return {"q_dimension": self.q_dimension, "end": path.end_point}


def test_two_mechanism_doubles_consume_the_same_task_artifact_unchanged(planning) -> None:
    case_map = planning.custom_map([planning.exclusion("P1", 0.0, 2.0, 0.18)])
    result = planning.planner().plan(
        planning.request(case_map, (0.0, 0.1), (0.0, 4.9)), case_map, planning.TOOL
    )
    assert result.accepted
    serialized = json.dumps(result.path.to_dict(), sort_keys=True)
    outputs = [
        double.consume(json.loads(serialized))
        for double in (_SwingArmDouble(), _ThreeJointDouble())
    ]
    assert [o["q_dimension"] for o in outputs] == [1, 3]
    assert json.dumps(result.path.to_dict(), sort_keys=True) == serialized


# --- A1-04: compatibilidad y alcance -----------------------------------------------------


def test_unsupported_capabilities_are_explicit_not_geometric_impossibility(planning) -> None:
    local_map = planning.custom_map()
    for orientation in (
        OrientationRequirement(policy=OrientationPolicy.FREE),
        OrientationRequirement(policy=OrientationPolicy.INTERVAL, interval_rad=(-0.1, 0.1)),
    ):
        result = planning.planner().plan(
            planning.request(local_map, (0.0, 0.1), (0.0, 4.9), orientation=orientation),
            local_map,
            planning.TOOL,
        )
        assert result.status is PlanStatus.UNSUPPORTED
        assert result.reason is AbstentionReason.UNSUPPORTED_FEATURE
        assert result.path is None and "No es una imposibilidad" in result.detail


def test_non_convex_profile_is_unsupported(planning) -> None:
    local_map = planning.custom_map()
    l_shape = tool_profile.ToolGeometry(
        tool_id="l_blade",
        collision_polygon=[[0, 0], [0.2, 0], [0.2, 0.05], [0.05, 0.05], [0.05, 0.2], [0, 0.2]],
        cutting_polygon=[[0.01, 0.01], [0.04, 0.01], [0.04, 0.04], [0.01, 0.04]],
        pivot=planning.point(0.0, 0.0, frame=Frame.TOOL),
    )
    result = planning.planner().plan(
        planning.request(local_map, (0.0, 0.3), (0.0, 4.5), profile=l_shape), local_map, l_shape
    )
    assert result.status is PlanStatus.UNSUPPORTED


def test_accepted_result_is_geometric_only(planning) -> None:
    local_map = planning.custom_map()
    result = planning.planner().plan(
        planning.request(local_map, (0.0, 0.1), (0.0, 4.9)), local_map, planning.TOOL
    )
    payload = result.to_dict()
    assert payload["status"] == "ACCEPTED"
    assert payload["validation_scope"] == "geometric"
    assert payload["mechanical_feasibility"] == "not_evaluated"
    assert payload["validation"]["mechanical_feasibility"] == "not_evaluated"
    assert "no es trayectoria ejecutable" in payload["output_kind"]


def test_mismatched_identities_are_invalid_input(planning) -> None:
    local_map = planning.custom_map()
    other = planning.custom_map([planning.exclusion("P1", 0.0, 2.0, 0.18)])
    request = planning.request(local_map, (0.0, 0.1), (0.0, 4.9))
    with pytest.raises(PrecisionWeederError, match="otro mapa"):
        planning.planner().plan(request, other, planning.TOOL)
    other_tool = tool_profile.ToolGeometry.rectangular("blade_rect_200x80", 0.2, 0.09)
    with pytest.raises(PrecisionWeederError, match="otro perfil"):
        planning.planner().plan(request, local_map, other_tool)


def test_e1_scene_and_configs_are_consumed_without_modification(project_root) -> None:
    """La escena nominal E1 se usa tal cual; sus YAML no cambian por A1/E2."""
    from precision_weeder.mapping.synthetic_adapter import local_map_from_scene
    from precision_weeder.scenarios import build_fixture

    scene = build_fixture("nominal_three_plants")
    local_map = local_map_from_scene(scene)
    assert local_map.exclusions == scene.exclusions
    assert local_map.corridor == scene.corridor
    assert local_map.uncertainty == scene.uncertainty
    assert local_map.known_region_basis == "synthetic_ground_truth"
    assert scene.provenance is Provenance.SYNTHETIC
