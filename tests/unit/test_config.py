"""Configuracion: carga estricta, claves desconocidas, modos y calibracion."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from precision_weeder.core.config import (
    SCHEMA_VERSION,
    CalibrationConfig,
    KinematicsConfig,
    ProjectConfig,
    RunMode,
    SafetyConfig,
    SceneConfig,
    SyntheticFile,
    ToolConfig,
    UncertaintyConfig,
    load_project_config,
    load_yaml_mapping,
    validate_config_payload,
)
from precision_weeder.core.errors import ConfigError, MissingCalibrationError, UnknownConfigKeyError
from precision_weeder.core.types import CalibrationStatus
from precision_weeder.core.units import SpatialDomain


@pytest.fixture
def project(configs_dir: Path) -> ProjectConfig:
    return load_project_config(
        configs_dir / "synthetic.yaml",
        configs_dir / "tool.yaml",
        configs_dir / "safety.yaml",
        configs_dir / "kinematics.yaml",
    )


# --- carga de los YAML del repositorio ---------------------------------------


def test_repository_configs_load(project: ProjectConfig) -> None:
    assert project.schema_version == SCHEMA_VERSION
    assert project.mode is RunMode.SYNTHETIC
    assert project.scene.scene_id == "nominal_three_plants"
    assert project.scene.domain is SpatialDomain.GROUND_METRIC_SYNTHETIC
    assert project.tool.length_x_m == pytest.approx(0.20)
    assert project.safety.clearance_epsilon_m == pytest.approx(1e-4)
    assert project.kinematics.horizon_s == pytest.approx(2.0)


def test_resolved_config_materializes_every_default(project: ProjectConfig) -> None:
    """Todo valor por defecto aparece explicito en la configuracion resuelta (§7)."""
    resolved = project.resolved_dict()
    assert resolved["calibration"]["status"] == CalibrationStatus.ABSENT.value
    assert resolved["scene"]["advance"]["heading_rad"] == 0.0
    assert resolved["kinematics"]["latencies"]["planning_s"] == pytest.approx(0.02)
    assert resolved["tool"]["depth_z_m"] is None
    round_trip = yaml.safe_load(project.resolved_yaml())
    assert round_trip == resolved


def test_configs_declare_feasible_start_state(project: ProjectConfig) -> None:
    """Debe existir al menos un caso factible sin tocar los casos adversos (§5 E1).

    El pivote puede alcanzar lateralmente el centro de un pasillo libre junto a una
    exclusion de 0.18 m con una cuchilla de 0.20 m de extension lateral.
    """
    required_offset = (
        project.scene.exclusion.protection_radius_m
        + project.scene.exclusion.uncertainty_margin_m
        + 0.5 * project.tool.length_x_m
    )
    assert project.kinematics.position_limits_m.x_max >= required_offset
    corridor_half_width = 0.5 * project.scene.corridor.width_m
    assert required_offset <= corridor_half_width


def test_latency_budget_fits_before_encounter(project: ProjectConfig) -> None:
    """La latencia total declarada cabe dentro del horizonte de planificacion."""
    assert project.kinematics.latencies.total_s < project.kinematics.horizon_s


# --- rechazo de configuraciones invalidas ------------------------------------


def test_unknown_key_is_an_error(invalid_configs_dir: Path) -> None:
    payload = load_yaml_mapping(invalid_configs_dir / "unknown_key.yaml")
    with pytest.raises(UnknownConfigKeyError, match="anchura_m"):
        validate_config_payload(SyntheticFile, payload, "unknown_key.yaml")


@pytest.mark.parametrize("name", ["non_finite.yaml", "negative_dimension.yaml"])
def test_invalid_dimensions_are_rejected(invalid_configs_dir: Path, name: str) -> None:
    payload = load_yaml_mapping(invalid_configs_dir / name)
    with pytest.raises(ValueError):
        SyntheticFile.model_validate(payload)


def test_calibrated_mode_without_calibration_fails_explicitly(
    invalid_configs_dir: Path, configs_dir: Path
) -> None:
    """Pedir dominio verificado sin calibracion no degrada: falla (§5 E1, §7).

    Escena y herramienta comparten dominio verificado, de modo que la unica causa
    posible del rechazo es la calibracion ausente (R2).
    """
    with pytest.raises(MissingCalibrationError, match="calibracion"):
        load_project_config(
            invalid_configs_dir / "calibrated_without_calibration.yaml",
            invalid_configs_dir / "tool_verified.yaml",
            configs_dir / "safety.yaml",
            configs_dir / "kinematics.yaml",
        )


def test_empty_plant_list_requires_explicit_certification(invalid_configs_dir: Path) -> None:
    """Ausencia de plantas no prueba ausencia de cultivo (§3.2)."""
    payload = load_yaml_mapping(invalid_configs_dir / "empty_plants_uncertified.yaml")
    with pytest.raises(ValueError, match="free_space_certified"):
        SyntheticFile.model_validate(payload)


def test_certified_free_space_rejects_declared_plants() -> None:
    with pytest.raises(ValueError, match="free_space_certified"):
        SceneConfig(free_space_certified=True)


def test_unknown_key_in_every_file_is_rejected(configs_dir: Path, tmp_path: Path) -> None:
    for name in ("synthetic", "tool", "safety", "kinematics"):
        payload = load_yaml_mapping(configs_dir / f"{name}.yaml")
        payload["clave_inesperada"] = 1
        target = tmp_path / f"{name}.yaml"
        target.write_text(yaml.safe_dump(payload), encoding="utf-8")
        paths = {
            key: (tmp_path / f"{key}.yaml" if key == name else configs_dir / f"{key}.yaml")
            for key in ("synthetic", "tool", "safety", "kinematics")
        }
        with pytest.raises(UnknownConfigKeyError):
            load_project_config(
                paths["synthetic"], paths["tool"], paths["safety"], paths["kinematics"]
            )


def test_missing_file_is_a_config_error(configs_dir: Path) -> None:
    with pytest.raises(ConfigError):
        load_project_config(
            configs_dir / "no_existe.yaml",
            configs_dir / "tool.yaml",
            configs_dir / "safety.yaml",
            configs_dir / "kinematics.yaml",
        )


# --- reglas de dominio y de modo ---------------------------------------------


def test_mode_synthetic_rejects_verified_domain() -> None:
    with pytest.raises(ValueError, match="no es licito"):
        ProjectConfig(
            mode=RunMode.SYNTHETIC,
            scene=SceneConfig(domain=SpatialDomain.GROUND_METRIC_VERIFIED),
            tool=ToolConfig(domain=SpatialDomain.GROUND_METRIC_VERIFIED),
        )


def test_scene_and_tool_domains_must_match() -> None:
    with pytest.raises(ValueError, match="deben coincidir"):
        ProjectConfig(
            scene=SceneConfig(domain=SpatialDomain.GROUND_METRIC_SYNTHETIC),
            tool=ToolConfig(domain=SpatialDomain.GRAPHIC_UNSCALED),
        )


def test_calibrated_mode_accepts_verified_calibration() -> None:
    config = ProjectConfig(
        mode=RunMode.CALIBRATED,
        scene=SceneConfig(domain=SpatialDomain.GROUND_METRIC_VERIFIED),
        tool=ToolConfig(domain=SpatialDomain.GROUND_METRIC_VERIFIED),
        calibration=CalibrationConfig(
            calibration_id="cam0",
            status=CalibrationStatus.VERIFIED,
            scale_evidence="patron de 1.000 m medido de forma independiente",
            independent_point_error=0.004,
            domain=SpatialDomain.GROUND_METRIC_VERIFIED,
        ),
    )
    assert config.calibration_reference.is_verified


def test_exploratory_mode_rejects_metric_domains() -> None:
    with pytest.raises(ValueError, match="no es licito"):
        ProjectConfig(
            mode=RunMode.EXPLORATORY,
            scene=SceneConfig(domain=SpatialDomain.GROUND_METRIC_SYNTHETIC),
            tool=ToolConfig(domain=SpatialDomain.GROUND_METRIC_SYNTHETIC),
        )


# --- reglas propias de cada seccion -------------------------------------------


def test_uncertainty_budget_must_cover_the_sum_of_bounds() -> None:
    with pytest.raises(ValueError, match=r"presupuesto|excede"):
        UncertaintyConfig(
            localization_bound_m=0.02,
            calibration_bound_m=0.02,
            tracking_bound_m=0.02,
            bound_budget_m=0.03,
        )


def test_safety_inflates_exclusions_or_tool_but_not_both() -> None:
    assert SafetyConfig().uncertainty.inflate == "exclusions"
    with pytest.raises(ValueError):
        UncertaintyConfig(inflate="ambos")  # type: ignore[arg-type]


def test_tool_rejects_variable_orientation_in_h1() -> None:
    with pytest.raises(ValueError, match="orientacion fija"):
        ToolConfig(orientation_mode="variable")


def test_tool_rejects_non_null_depth_in_h1() -> None:
    with pytest.raises(ValueError, match="depth_z_m"):
        ToolConfig(depth_z_m=-0.03)


def test_kinematics_rejects_start_state_outside_limits() -> None:
    with pytest.raises(ValueError, match="start_state"):
        KinematicsConfig(start_state={"x_m": 10.0})  # type: ignore[arg-type]


def test_plants_lists_must_have_matching_lengths() -> None:
    with pytest.raises(ValueError, match="misma longitud"):
        SceneConfig(plants={"positions_y_m": [1.0, 2.0], "offsets_x_m": [0.0]})  # type: ignore[arg-type]


def test_config_objects_are_immutable() -> None:
    config = ToolConfig()
    with pytest.raises(ValueError):
        config.length_x_m = 0.5  # type: ignore[misc]
