"""AE3-02 — geometria del brazo oscilante 1R contra oraculos analiticos independientes.

Los oraculos se escriben aqui con las ecuaciones de la orden §4, sin llamar a
`forward`/`tool_point` del modelo bajo prueba (manifiesto F-FK, F-IK, F-IK-neg, F-WRAP).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from precision_weeder.core.compilation import CompileStatus, InfeasibilityCode
from precision_weeder.core.config import MechanismFile, validate_config_payload
from precision_weeder.core.errors import ConfigError, InvalidDimensionError, SchemaValidationError

TOL = 1e-9
QS = (-1.2, -0.5, 0.0, 0.7, 1.2)
YAWS = (0.0, 0.4, -2.5)


def oracle_blade(q, pivot=(0.0, 0.0), length=0.4, beta=0.0, base=(0.0, 0.0, 0.0)):
    """`p_B^W = p_V + R(psi)(a + L(cos q, sin q))`, `theta = psi + q + beta`."""
    x_v, y_v, psi = base
    local = (pivot[0] + length * math.cos(q), pivot[1] + length * math.sin(q))
    c, s = math.cos(psi), math.sin(psi)
    return (x_v + c * local[0] - s * local[1], y_v + s * local[0] + c * local[1], psi + q + beta)


def to_vehicle_oracle(point, base):
    x_v, y_v, psi = base
    dx, dy = point[0] - x_v, point[1] - y_v
    c, s = math.cos(psi), math.sin(psi)
    return np.array([c * dx + s * dy, -s * dx + c * dy])


@pytest.mark.parametrize("yaw", YAWS)
@pytest.mark.parametrize("q", QS)
def test_forward_matches_oracle_with_translated_and_rotated_base(mechanics, q, yaw) -> None:
    model = mechanics.swing()
    base = np.array([0.3, -0.2, yaw])
    poses = model.forward([q])
    blade = poses["blade"].in_world(base)
    assert np.allclose(blade, oracle_blade(q, base=tuple(base)), rtol=0.0, atol=TOL)
    arm = poses["arm"].in_world(base)
    assert np.allclose(arm, (0.3, -0.2, yaw + q), rtol=0.0, atol=TOL)
    assert poses["blade"].tool_id == "blade_rect_200x80" and poses["arm"].tool_id is None


def test_pivot_and_blade_offsets_are_applied_once(mechanics) -> None:
    config = mechanics.MECHANISM.model_copy(update={"pivot_m": (0.1, -0.05), "beta_rad": 0.3})
    model = mechanics.swing(config)
    base = np.array([1.0, 2.0, 0.4])
    for q in QS:
        expected = oracle_blade(q, (0.1, -0.05), 0.4, 0.3, tuple(base))
        assert np.allclose(model.forward([q])["blade"].in_world(base), expected, rtol=0.0, atol=TOL)


@pytest.mark.parametrize("yaw", YAWS)
@pytest.mark.parametrize("q", QS)
def test_inverse_recovers_known_angle_from_world_target(mechanics, q, yaw) -> None:
    model = mechanics.swing()
    base = (0.3, -0.2, yaw)
    x, y, _ = oracle_blade(q, base=base)
    result = model.inverse(to_vehicle_oracle((x, y), base), [0.0])
    assert result.status is CompileStatus.FEASIBLE_CANDIDATE
    assert abs(result.q[0] - q) <= TOL


def test_inverse_with_fixed_orientation(mechanics) -> None:
    model = mechanics.swing()
    target = np.array([0.4 * math.cos(0.7), 0.4 * math.sin(0.7)])
    assert model.inverse(target, [0.0], theta_v=0.7).ok
    conflict = model.inverse(target, [0.0], theta_v=0.0)
    assert conflict.status is CompileStatus.INFEASIBLE_PROVEN
    assert conflict.code is InfeasibilityCode.ORIENTATION_CONFLICT
    assert abs(conflict.residual - 0.7) <= TOL


@pytest.mark.parametrize(
    ("radius", "where"), [(0.2, "interior del disco"), (0.6, "exterior"), (0.0, "interior")]
)
def test_points_off_the_arc_are_proven_out_of_reach(mechanics, radius, where) -> None:
    result = mechanics.swing().inverse([radius * math.cos(0.3), radius * math.sin(0.3)], [0.0])
    assert result.status is CompileStatus.INFEASIBLE_PROVEN
    assert result.code is InfeasibilityCode.RADIAL_REACH
    assert abs(result.residual - (radius - 0.4)) <= TOL and where in result.detail


def test_reach_tolerance_boundary(mechanics) -> None:
    model = mechanics.swing()
    direction = np.array([math.cos(0.2), math.sin(0.2)])
    assert model.inverse(direction * (0.4 + 0.5e-9), [0.0]).ok
    assert model.inverse(direction * (0.4 + 1e-8), [0.0]).code is InfeasibilityCode.RADIAL_REACH


def test_point_on_the_arc_beyond_the_stop(mechanics) -> None:
    result = mechanics.swing().inverse([0.4 * math.cos(1.5), 0.4 * math.sin(1.5)], [1.0])
    assert result.status is CompileStatus.INFEASIBLE_PROVEN
    assert result.code is InfeasibilityCode.JOINT_LIMIT
    assert result.limit == 1.2 and abs(result.value - 1.5) <= TOL


def test_zero_length_is_an_invalid_model(mechanics) -> None:
    with pytest.raises(InvalidDimensionError):
        mechanics.SinglePivotSwing.from_length(mechanics.MECHANISM, mechanics.BLADE, 0.0)
    payload = {"mechanism": {"length_m": 0.0}}
    with pytest.raises(ConfigError):
        validate_config_payload(MechanismFile, payload, "L=0")


def test_wrong_tool_or_dimension_is_rejected(mechanics) -> None:
    other = mechanics.MECHANISM.model_copy(update={"tool_id": "otra"})
    with pytest.raises(SchemaValidationError):
        mechanics.swing(other)
    with pytest.raises(SchemaValidationError):
        mechanics.swing().forward([0.0, 0.1])


def test_inverse_crosses_pi_without_jump_inside_wrap_variant(mechanics) -> None:
    """Variante q en [2.0, 4.3]: el objetivo cruza el corte de atan2 en +-pi."""
    model = mechanics.swing(mechanics.WRAP)
    q_prev = math.pi + 0.5 * math.sin(0.0)
    for t in np.linspace(0.0, 8.0, 801):
        q_true = math.pi + 0.5 * math.sin(1.5 * t)
        target = (0.4 * math.cos(q_true), 0.4 * math.sin(q_true))
        result = model.inverse(target, [q_prev])
        assert result.ok, result
        assert abs(result.q[0] - q_true) <= TOL
        assert abs(result.q[0] - q_prev) < 0.1  # sin salto de 2 pi
        q_prev = result.q[0]
    # El mismo punto en el modelo nominal esta fuera de topes, no "alcanzable por otra rama".
    beyond = mechanics.swing().inverse([-0.4, 0.0], [1.2])
    assert beyond.code is InfeasibilityCode.JOINT_LIMIT


def test_validate_configuration_reports_stops_exactly(mechanics) -> None:
    model = mechanics.swing()
    assert model.validate_configuration([1.2]).valid
    assert model.validate_configuration([-1.2]).valid
    check = model.validate_configuration([1.2 + 1e-12])
    assert not check.valid and check.violations[0][:2] == ("JOINT_LIMIT", "swing")


def _arc_target_derivatives(q, q_d, q_dd, length=0.4):
    """`rho`, `rho'`, `rho''` del punto del arco para q(t) dado (en `V`)."""
    c, s = math.cos(q), math.sin(q)
    rho = length * np.array([c, s])
    rho_d = length * q_d * np.array([-s, c])
    rho_dd = length * (q_dd * np.array([-s, c]) - q_d**2 * np.array([c, s]))
    return rho, rho_d, rho_dd


def test_joint_rates_with_translating_and_yawing_base(mechanics) -> None:
    """Objetivo en `W` desde `q = 0.5 sin(1.5 t)` y base con yaw variable: el compilador
    debe recuperar q' y q'' incluyendo la rotacion del vehiculo (F-SIN)."""
    model = mechanics.swing()
    motion = mechanics.sampled_motion(mechanics.yawing_pose, duration=8.0, nodes=161)
    j = np.array([[0.0, -1.0], [1.0, 0.0]])
    for t in motion.pose.breakpoints[::7]:
        q, q_d, q_dd = 0.5 * math.sin(1.5 * t), 0.75 * math.cos(1.5 * t), -1.125 * math.sin(1.5 * t)
        (x, y, psi), (vx, vy, w), (ax, ay, al) = mechanics.yawing_pose(t)
        rot = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
        rho, rho_d, rho_dd = _arc_target_derivatives(q, q_d, q_dd)
        p = np.array([x, y]) + rot @ rho
        p_d = np.array([vx, vy]) + w * j @ rot @ rho + rot @ rho_d
        p_dd = (
            np.array([ax, ay])
            + (al * j @ rot - w**2 * rot) @ rho
            + 2 * w * j @ rot @ rho_d
            + rot @ rho_dd
        )
        r, r_d, r_dd = motion.to_vehicle(float(t), p, p_d, p_dd)
        assert np.allclose(r, rho, rtol=0.0, atol=TOL)
        assert model.inverse(r, [0.0]).q[0] == pytest.approx(q, rel=0.0, abs=TOL)
        rate, accel = model.joint_rates(r, r_d, r_dd)
        assert abs(rate - q_d) <= 1e-6 and abs(accel - q_dd) <= 1e-6
