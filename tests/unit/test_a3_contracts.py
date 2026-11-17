"""AE3-01 — contratos temporales, de vinculacion y de mecanismo; separacion de E2."""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from precision_weeder.core.compilation import CompileStatus
from precision_weeder.core.config import MechanismFile, validate_config_payload
from precision_weeder.core.errors import (
    ConfigError,
    NonFiniteValueError,
    SchemaValidationError,
    TemporalOrderError,
    UnitMismatchError,
    UnknownConfigKeyError,
)
from precision_weeder.core.temporal import (
    RoleBinding,
    TaskBinding,
    TaskTrajectory,
    TimePolynomial,
    VehicleMotion,
    WorkRequirement,
)
from precision_weeder.core.units import SpatialDomain


def linear_timing(t0: float, t1: float, s0: float = 0.0, s1: float = 1.0) -> TimePolynomial:
    return TimePolynomial(np.array([t0, t1]), np.array([[[s0, s1 - s0, 0, 0, 0, 0]]]))


@pytest.fixture
def gap_path(planning):
    local_map = planning.custom_map(map_id="a3_contracts")
    return planning.linear_path([(0.0, 1.49), (0.0, 1.75)], local_map)


@pytest.fixture
def trajectory(gap_path) -> TaskTrajectory:
    work = WorkRequirement(0.10, 0.001, ((-0.1, 1.0), (0.1, 1.0), (0.1, 2.0), (-0.1, 2.0)))
    return TaskTrajectory("gap", gap_path, linear_timing(0.0, 1.3), "scenario", work)


# --- TimePolynomial ------------------------------------------------------------------


def test_time_polynomial_rejects_bad_clocks_and_values() -> None:
    coefficients = np.zeros((1, 1, 6))
    with pytest.raises(TemporalOrderError):
        TimePolynomial(np.array([1.0, 1.0]), coefficients)
    with pytest.raises(TemporalOrderError):
        TimePolynomial(np.array([1.0, 0.0]), coefficients)
    with pytest.raises(NonFiniteValueError):
        TimePolynomial(np.array([0.0, math.nan]), coefficients)
    with pytest.raises(NonFiniteValueError):
        TimePolynomial(np.array([0.0, 1.0]), np.full((1, 1, 6), math.inf))
    with pytest.raises(SchemaValidationError):
        TimePolynomial(np.array([0.0, 1.0, 2.0]), coefficients)  # 2 tramos, 1 bloque
    with pytest.raises(SchemaValidationError):
        TimePolynomial(np.array([0.0, 1.0]), np.zeros((1, 1, 4)))


def test_time_polynomial_checks_declared_continuity() -> None:
    times = np.array([0.0, 1.0, 2.0])
    kink = np.array([[[0, 1, 0, 0, 0, 0]], [[1, 2, 0, 0, 0, 0]]], dtype=float)  # C0, no C1
    with pytest.raises(SchemaValidationError):
        TimePolynomial(times, kink, "C2")
    assert TimePolynomial(times, kink, "C0").evaluate(1.5, 1)[0] == pytest.approx(2.0)
    jump = np.array([[[0, 1, 0, 0, 0, 0]], [[1.1, 1, 0, 0, 0, 0]]], dtype=float)
    with pytest.raises(SchemaValidationError):
        TimePolynomial(times, jump, "C0")


def test_time_polynomial_does_not_extrapolate() -> None:
    timing = linear_timing(0.0, 2.0)
    assert timing.evaluate(2.0)[0] == pytest.approx(1.0)
    with pytest.raises(TemporalOrderError):
        timing.evaluate(2.0 + 1e-9)
    with pytest.raises(TemporalOrderError):
        timing.evaluate(-1e-9)


def test_bounds_are_exact_between_nodes() -> None:
    """Un maximo interior entre nodos se detecta sin muestrear (u - u^2, max 0.25)."""
    poly = TimePolynomial(np.array([0.0, 2.0]), np.array([[[0, 1, -1, 0, 0, 0]]]))
    low, high = poly.bounds(0)
    assert (low, high) == (pytest.approx(0.0), pytest.approx(0.25, abs=1e-15))
    rate_low, rate_high = poly.bounds(0, order=1)  # (1 - 2u)/2
    assert (rate_low, rate_high) == (pytest.approx(-0.5), pytest.approx(0.5))


def test_from_states_interpolates_value_and_derivatives() -> None:
    times = np.array([0.0, 0.5, 1.3])
    states = np.array(
        [[[0.0], [0.0], [0.0]], [[0.4], [1.0], [-2.0]], [[1.0], [0.0], [0.0]]], dtype=float
    )
    poly = TimePolynomial.from_states(times, states)
    for k, t in enumerate(times):
        for order in range(3):
            assert poly.evaluate(t, order)[0] == pytest.approx(states[k, order, 0], abs=1e-12)


def test_time_polynomial_owns_immutable_arrays() -> None:
    times = np.array([0.0, 1.0])
    coefficients = np.zeros((1, 1, 6))
    poly = TimePolynomial(times, coefficients)
    times[1] = 5.0
    coefficients[0, 0, 0] = 9.0
    assert poly.t_end == 1.0 and poly.evaluate(0.5)[0] == 0.0
    with pytest.raises(ValueError):
        poly.coefficients[0, 0, 0] = 1.0


# --- TaskTrajectory ------------------------------------------------------------------


def test_trajectory_evaluates_path_with_explicit_clock(trajectory) -> None:
    for t in (0.0, 0.4, 1.3):
        p, v, a = trajectory.position(t)
        assert np.allclose(p, (0.0, 1.49 + 0.2 * t), atol=1e-12)
        assert np.allclose(v, (0.0, 0.2), atol=1e-12) and np.allclose(a, 0.0, atol=1e-12)
    assert trajectory.domain is SpatialDomain.GROUND_METRIC_SYNTHETIC


def test_trajectory_rejects_invalid_laws(gap_path) -> None:
    with pytest.raises(SchemaValidationError):  # s sale de [0, 1] entre nodos
        TaskTrajectory("x", gap_path, linear_timing(0.0, 1.0, 0.0, 1.2), "c")
    decreasing = TimePolynomial(np.array([0.0, 1.0]), np.array([[[0, 2, -2, 0, 0, 0]]]))
    with pytest.raises(SchemaValidationError):  # s(u)=2u-2u^2 sube y retrocede
        TaskTrajectory("x", gap_path, decreasing, "c")
    with pytest.raises(SchemaValidationError):
        TaskTrajectory("x", gap_path, linear_timing(0, 1), "c", mode="PATH_WITH_WINDOWS")
    two = TimePolynomial(np.array([0.0, 1.0]), np.zeros((1, 2, 6)))
    with pytest.raises(SchemaValidationError):
        TaskTrajectory("x", gap_path, two, "c")
    with pytest.raises(SchemaValidationError):
        TaskTrajectory("x", gap_path, linear_timing(0, 1), " ")
    with pytest.raises(NonFiniteValueError):
        TaskTrajectory("x", gap_path, linear_timing(0, 1), "c", tolerance_m=math.nan)


def test_trajectory_roundtrip_and_identity(trajectory, gap_path, planning) -> None:
    payload = json.loads(json.dumps(trajectory.to_dict()))
    again = TaskTrajectory.from_dict(payload)
    assert again.content_hash() == trajectory.content_hash()
    assert payload["task_path_hash"] == gap_path.content_hash()
    tampered = json.loads(json.dumps(payload))
    tampered["task_path_hash"] = "0" * 64
    with pytest.raises(SchemaValidationError):
        TaskTrajectory.from_dict(tampered)
    tampered = json.loads(json.dumps(payload))
    tampered["tolerance_m"] = 1e-5
    with pytest.raises(SchemaValidationError):
        TaskTrajectory.from_dict(tampered)
    other_map = planning.custom_map(map_id="otro_mapa")
    variants = [
        TaskTrajectory(
            "gap",
            planning.linear_path([(0.0, 1.49), (0.0, 1.75)], other_map),
            trajectory.timing,
            "scenario",
            trajectory.work,
        ),
        TaskTrajectory("gap", gap_path, linear_timing(0.0, 1.4), "scenario", trajectory.work),
        TaskTrajectory("gap", gap_path, trajectory.timing, "scenario", WorkRequirement(0.2)),
        TaskTrajectory("gap", gap_path, trajectory.timing, "otro_reloj", trajectory.work),
    ]
    hashes = {v.content_hash() for v in variants} | {trajectory.content_hash()}
    assert len(hashes) == len(variants) + 1


# --- VehicleMotion -------------------------------------------------------------------


def test_vehicle_motion_contract(mechanics) -> None:
    motion = mechanics.constant_motion(velocity=(0.0, 0.2), start=(-0.4, 0.5), duration=5.0)
    pose, rate, accel = motion.state(2.0)
    assert np.allclose(pose, (-0.4, 0.9, 0.0)) and np.allclose(rate, (0, 0.2, 0))
    assert np.allclose(accel, 0.0)
    with pytest.raises(TemporalOrderError):
        motion.state(5.0 + 1e-9)
    again = VehicleMotion.from_dict(json.loads(json.dumps(motion.to_dict())))
    assert again.content_hash() == motion.content_hash()
    moved = mechanics.constant_motion(velocity=(0.0, 0.21), start=(-0.4, 0.5), duration=5.0)
    assert moved.content_hash() != motion.content_hash()
    with pytest.raises(UnitMismatchError):
        VehicleMotion("m", "c", motion.pose, SpatialDomain.IMAGE_PX)
    with pytest.raises(SchemaValidationError):
        VehicleMotion("m", "c", TimePolynomial(np.array([0.0, 1.0]), np.zeros((1, 2, 6))))
    with pytest.raises(NonFiniteValueError):
        VehicleMotion("m", "c", motion.pose, position_bound_m=math.inf)


def test_world_vehicle_round_trip_with_yaw(mechanics) -> None:
    motion = mechanics.constant_motion(velocity=(0.1, 0.2), start=(0.3, -0.2), yaw=0.4)
    for t in (0.0, 1.7, 9.5):
        w = np.array([0.31, -0.12])
        p, theta = motion.to_world(t, w, 0.25)
        back, _, _ = motion.to_vehicle(t, p, np.zeros(2), np.zeros(2))
        assert np.allclose(back, w, rtol=0.0, atol=1e-12) and theta == pytest.approx(0.65)


# --- TaskBinding ---------------------------------------------------------------------


def test_task_binding(mechanics) -> None:
    profile_hash = mechanics.BLADE.content_hash()
    single = TaskBinding((RoleBinding("pasada_principal", "blade_rect_200x80", profile_hash),))
    assert single.is_single_exclusive and single.for_role("pasada_principal").tool_id
    assert single.for_role("franja_izquierda") is None
    double = TaskBinding(
        (
            RoleBinding("pasada_principal", "blade_rect_200x80", profile_hash),
            RoleBinding("franja_izquierda", "otra", profile_hash),
        ),
        "simultaneous",
    )
    assert not double.is_single_exclusive and double.content_hash() != single.content_hash()
    with pytest.raises(SchemaValidationError):
        RoleBinding("pasada_principal", "blade", "abc")
    with pytest.raises(SchemaValidationError):
        TaskBinding((), "exclusive")


# --- MechanismModel ------------------------------------------------------------------


def test_model_identity_tracks_every_parameter(mechanics) -> None:
    base = mechanics.swing()
    assert base.content_hash() == mechanics.swing().content_hash()
    joint = mechanics.MECHANISM.joint
    changes = [
        {"length_m": 0.41},
        {"pivot_m": (0.0, 0.01)},
        {"beta_rad": 0.1},
        {"arm_width_m": 0.03},
        {"version": 2},
        {"joint": joint.model_copy(update={"upper_rad": 1.1})},
        {"joint": joint.model_copy(update={"velocity_limit_rad_s": 0.9})},
    ]
    hashes = {
        mechanics.swing(mechanics.MECHANISM.model_copy(update=c)).content_hash() for c in changes
    }
    assert len(hashes) == len(changes) and base.content_hash() not in hashes


def test_decoupled_double_shares_the_task_contract(mechanics, gap_path) -> None:
    """El mismo `TaskPath`/hash lo consume cualquier modelo: el 1R y un doble con dim(q)=2
    y una sola herramienta. Sin inversa, el doble responde `UNSUPPORTED`, no imposible."""
    swing, double = mechanics.swing(), mechanics.CartesianGantryDouble()
    assert swing.configuration_schema.dimension == 1 and len(swing.tool_ids) == 1
    assert double.configuration_schema.dimension == 2 and len(double.tool_ids) == 1
    for model in (swing, double):
        assert model.tool_body("blade_rect_200x80").tool_id == "blade_rect_200x80"
        assert gap_path.profile_hash == mechanics.BLADE.content_hash()
    assert double.inverse(np.array([0.1, 0.0]), np.zeros(2)).status is CompileStatus.UNSUPPORTED
    assert double.validate_configuration([0.6, 0.0]).violations[0][1] == "x"
    with pytest.raises(SchemaValidationError):
        double.forward([0.0])
    assert double.allowed_pairs == frozenset()
    assert swing.allowed_pairs == {frozenset({"arm", "blade"})}


def test_planner_does_not_import_mechanics() -> None:
    code = (
        "import sys, precision_weeder.planning, precision_weeder.validation.swept_volume\n"
        "bad = [m for m in sys.modules if m.startswith('precision_weeder.kinematics')]\n"
        "assert not bad, bad"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_mechanism_config_is_strict(mechanics) -> None:
    with pytest.raises(UnknownConfigKeyError):
        validate_config_payload(MechanismFile, {"mechanism": {"torque": 3}}, "x")
    bad_joint = {"mechanism": {"joint": {"lower_rad": 1.0, "upper_rad": 1.0}}}
    with pytest.raises(ConfigError):
        validate_config_payload(MechanismFile, bad_joint, "x")
    with pytest.raises(ConfigError):
        validate_config_payload(MechanismFile, {"mechanism": {"length_m": math.nan}}, "x")
    assert mechanics.MECHANISM.length_m == 0.4 and mechanics.MECHANISM.joint.upper_rad == 1.2
