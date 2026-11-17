"""E2-08: 100 escenarios congelados (semillas 0-99, `configs/planner.yaml`).

No se impone un porcentaje de exito. Para cada ruta aceptada, un validador
independiente (llamada nueva) y comprobaciones de propiedades verifican los
invariantes; ademas, una comprobacion densa por muestras, solo como contraste
secundario (no como prueba de seguridad).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from shapely.geometry import Polygon

from precision_weeder.core.types import PlanStatus
from precision_weeder.scenarios.planning_cases import random_planning_case
from precision_weeder.validation.swept_volume import ValidationStatus, validate_task_path


@pytest.fixture(scope="module")
def outcomes(planning):
    config = planning.RANDOM
    assert (config.seed_start, config.count) == (0, 100)
    results = []
    for seed in range(config.seed_start, config.seed_start + config.count):
        case = random_planning_case(seed, config)
        results.append((seed, case, planning.run_case(case)))
    return results


def test_generator_is_frozen_and_reproducible(planning) -> None:
    a = random_planning_case(17, planning.RANDOM)
    b = random_planning_case(17, planning.RANDOM)
    assert a.local_map.content_hash() == b.local_map.content_hash()
    assert (
        random_planning_case(18, planning.RANDOM).local_map.content_hash()
        != a.local_map.content_hash()
    )


def test_every_accepted_route_satisfies_independent_invariants(planning, outcomes) -> None:
    tool = np.asarray(planning.TOOL.collision_polygon)
    accepted = 0
    for seed, case, result in outcomes:
        assert result.status in (PlanStatus.ACCEPTED, PlanStatus.ABSTAIN, PlanStatus.UNCERTIFIED), (
            seed
        )
        if not result.accepted:
            assert result.path is None and result.reason is not None
            continue
        accepted += 1
        path = result.path
        report = validate_task_path(
            path,
            case.local_map,
            planning.TOOL,
            planning.SAFETY,
            planning.VALIDATION,
            expected_start=case.start_xy_m,
            expected_goal=case.goal_xy_m,
        )
        assert report.status is ValidationStatus.ACCEPTED, seed
        assert report.continuity["max_jump_second_derivative"] <= 1e-8
        assert report.progress["non_decreasing"] and report.progress["advances"]
        assert report.envelope_error_max_m <= report.envelope_error_budget_m
        # Contraste secundario por muestras densas (no sustituye al certificado).
        allowed = case.local_map.allowed_region()
        exclusions = case.local_map.exclusion_union()
        for s in np.linspace(0.0, 1.0, 101):
            footprint = Polygon(tool + path.evaluate(s))
            assert allowed.covers(footprint), seed
            if not exclusions.is_empty:
                assert footprint.distance(exclusions) > planning.SAFETY.clearance_epsilon_m, seed
    assert accepted > 0


def test_outcome_summary_is_recorded(outcomes, tmp_path) -> None:
    counts = Counter(
        (result.status.value, result.reason.value if result.reason else None)
        for _, _, result in outcomes
    )
    assert sum(counts.values()) == 100
    # Los positivos obligatorios de E2-01 impiden aprobar rechazando todo; aqui solo se
    # registra el reparto, sin porcentaje exigido.
    assert counts[("ACCEPTED", None)] >= 1
