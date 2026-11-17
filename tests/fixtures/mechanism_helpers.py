"""Utilidades de prueba A3/E3: modelo 1R del repositorio, doble de mecanismo y
movimientos de vehiculo. Valores del manifiesto congelado
`docs/reviews/evidence/A3_E3/fixture_manifest.md`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from planning_helpers import ROOT

from precision_weeder.core.config import (
    MechanismFile,
    ToolFile,
    load_yaml_mapping,
    validate_config_payload,
)
from precision_weeder.core.temporal import TimePolynomial, VehicleMotion
from precision_weeder.geometry.tool_profile import tool_geometry_from_config
from precision_weeder.kinematics import (
    BodyPose,
    BodySpec,
    ConfigurationSchema,
    JointKind,
    JointSpec,
    MechanismModel,
    SinglePivotSwing,
)

MECHANISM = validate_config_payload(
    MechanismFile, load_yaml_mapping(ROOT / "configs/mechanism.yaml"), "mechanism.yaml"
).mechanism
BLADE = tool_geometry_from_config(
    validate_config_payload(ToolFile, load_yaml_mapping(ROOT / "configs/tool.yaml"), "tool").tool
)
WRAP = MECHANISM.model_copy(
    update={
        "model_id": "single_pivot_swing_wrap",
        "joint": MECHANISM.joint.model_copy(update={"lower_rad": 2.0, "upper_rad": 4.3}),
    }
)


def swing(config=MECHANISM) -> SinglePivotSwing:
    return SinglePivotSwing(config, BLADE)


def constant_motion(
    velocity=(0.0, 0.2), start=(0.0, 0.0), yaw=0.0, duration=10.0, motion_id="const"
) -> VehicleMotion:
    """Traslacion de velocidad constante y yaw fijo, exacta como polinomio."""
    vx, vy = velocity
    coefficients = np.zeros((1, 3, 6))
    coefficients[0, :, 0] = (start[0], start[1], yaw)
    coefficients[0, :2, 1] = (vx * duration, vy * duration)
    return VehicleMotion(
        motion_id, "scenario", TimePolynomial(np.array([0.0, duration]), coefficients)
    )


def sampled_motion(pose_fn, duration=10.0, nodes=401, motion_id="sampled") -> VehicleMotion:
    """`T_WV(t)` por quinticos de Hermite con valor y derivadas exactas en los nodos.

    `pose_fn(t)` devuelve `(3, 3)`: filas valor, d/dt, d2/dt2 de `(x, y, psi)`.
    """
    times = np.linspace(0.0, duration, nodes)
    states = np.stack([np.asarray(pose_fn(t)) for t in times])
    return VehicleMotion(motion_id, "scenario", TimePolynomial.from_states(times, states))


def yawing_pose(t: float) -> np.ndarray:
    """Avance 0.2 m/s con yaw `0.1 sin(0.5 t)`."""
    psi = 0.1 * math.sin(0.5 * t)
    return np.array(
        [
            [0.3, -0.2 + 0.2 * t, psi],
            [0.0, 0.2, 0.05 * math.cos(0.5 * t)],
            [0.0, 0.0, -0.025 * math.sin(0.5 * t)],
        ]
    )


class CartesianGantryDouble(MechanismModel):
    """Doble de desacoplamiento: dos juntas prismaticas `(x, y)` en `V`, una herramienta.

    `dim(q) = 2` con una sola herramienta; no es un mecanismo del proyecto.
    """

    def __init__(self, tool=BLADE) -> None:
        self.tool = tool
        self.model_id = "cartesian_gantry_double"
        self.version = 1
        self._schema = ConfigurationSchema(
            (
                JointSpec("x", JointKind.PRISMATIC, -0.5, 0.5, 0.6, 3.0),
                JointSpec("y", JointKind.PRISMATIC, -0.3, 0.3, 0.4, 2.0),
            )
        )
        carriage = np.array([[-0.03, -0.03], [0.03, -0.03], [0.03, 0.03], [-0.03, 0.03]])
        self._bodies = (
            BodySpec("carriage", carriage),
            BodySpec("blade", tool.collision_polygon, tool.tool_id, tool.cutting_polygon),
        )

    @property
    def configuration_schema(self) -> ConfigurationSchema:
        return self._schema

    @property
    def bodies(self) -> tuple[BodySpec, ...]:
        return self._bodies

    def parameters(self) -> dict[str, Any]:
        return {"tool_profile_hash": self.tool.content_hash()}

    def forward(self, q, assembly_mode: str = "unico") -> dict[str, BodyPose]:
        x, y = self.as_configuration(q)
        return {
            "carriage": BodyPose("carriage", float(x), float(y), 0.0),
            "blade": BodyPose("blade", float(x), float(y), 0.0, self.tool.tool_id),
        }
