"""Senal para el brazo V1 (orden V1, criterio V1-03): consigna en la linea de herramienta."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from m0p.projection import SegmentProjection
from m0p.route import Reference, chain_hash, chain_x_at
from m0p.setpoint import FIELDS, SetpointLog
from test_v1_pattern import AXIS, BOX, planner_for, scene


def projection(frame: int) -> SegmentProjection:
    return SegmentProjection("W", 0, frame, 0.001, 1, (0, 0), (0, -8.0 * frame), (1000, 1000))


def test_setpoint_follows_the_route_at_the_tool_line() -> None:
    planner, _ = planner_for("alternating")
    local_map = scene((0.25, 0.50, 0.75), length=2.0)
    log = SetpointLog()
    reference = None
    for frame in range(80):
        proj = projection(frame)
        attempt = planner.plan(local_map, proj, BOX, AXIS, reference)
        assert attempt.paths
        route_hash = chain_hash(attempt.paths)
        row = log.update(
            frame, frame / 30, proj, planner.profile, attempt.paths, attempt.axis_x, route_hash
        )
        y_tool = proj.row_v_to_ground_y(planner.profile.tool_line.v_norm)
        x_tool = chain_x_at(attempt.paths, y_tool)
        assert row["state"] == "trabajando" and row["route_hash"] == route_hash
        assert row["offset_from_row_axis_m"] == pytest.approx(x_tool - AXIS, abs=1e-12)
        assert row["lateral_camera_m"] == pytest.approx(x_tool - 0.5, abs=1e-12)
        assert row["tool_u_px"] == pytest.approx(x_tool / 0.001, abs=1e-6)
        reference = Reference(attempt.legs, attempt.axis_x, frame, route_hash)
    offsets = [row["offset_from_row_axis_m"] for row in log.rows]
    # La herramienta se aparta del eje hacia ambos costados: la consigna zigzaguea.
    assert max(offsets) > 0.05 and min(offsets) < -0.05
    steps = [abs(row["lateral_step_m"]) for row in log.rows[1:]]
    assert max(steps) == pytest.approx(log.max_step_m)
    # 8 px por cuadro a 1 mm/px: la consigna no salta entre cuadros consecutivos.
    assert log.max_step_m < 0.03
    assert log.summary()["travel_m"] == pytest.approx(79 * 0.008)
    assert log.max_rate_m_s == pytest.approx(log.max_step_m * 30, rel=1e-6)


def test_no_route_gives_no_setpoint_and_declares_the_state(tmp_path) -> None:
    planner, _ = planner_for("alternating")
    local_map = scene((0.25, 0.50, 0.75))
    proj = projection(0)
    attempt = planner.plan(local_map, proj, BOX, AXIS)
    log = SetpointLog()
    log.update(0, 0.0, proj, planner.profile, attempt.paths, AXIS, "h")
    lost = log.update(1, 0.1, projection(1), planner.profile, None, AXIS, None)
    assert lost["state"] == "retirada"
    assert lost["lateral_camera_m"] is None and lost["tool_u_px"] is None
    still = log.update(2, 0.2, None, planner.profile, None, None, None)
    assert still["state"] == "sin_ruta" and still["segment_id"] is None
    # Tras la interrupcion no se calcula paso contra la ultima consigna valida.
    back = log.update(3, 0.3, proj, planner.profile, attempt.paths, AXIS, "h")
    assert back["state"] == "trabajando" and back["lateral_step_m"] is None
    assert log.working_frames == 2 and log.max_step_m == 0.0
    path = tmp_path / "setpoints.csv"
    log.write_csv(path)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert tuple(rows[0].keys()) == FIELDS and len(rows) == 4
    assert rows[1]["lateral_camera_m"] == "" and rows[1]["state"] == "retirada"


def test_pipeline_publishes_one_setpoint_row_per_frame(runs) -> None:
    summary, _, records, output = runs["cfr_640"]
    path = Path(summary.setpoints)
    assert path == output.with_suffix(".setpoints.csv") and path.is_file()
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(records) == summary.frames
    for row, record in zip(rows, records, strict=True):
        assert int(row["frame_index"]) == record["frame_index"]
        working = row["state"] == "trabajando"
        assert working == (record["status"] == "vigente")
        assert (row["route_hash"] != "") == working
        assert record["setpoint"]["state"] == row["state"]
    assert summary.setpoint["working_frames"] == len(summary.valid_frames)
