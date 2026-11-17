"""Regresiones de E0-R1 (dictamen E0 rev. 1): el diagnostico temporal no se pierde.

Cada caso fija que los contadores, las filas de anomalias, la fuente temporal y las
metricas certificadas concuerdan, y que lo que no se puede tratar se rechaza con un
error explicito en lugar de un exito vacio.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from precision_weeder.core.errors import (
    PrecisionWeederError,
    SchemaValidationError,
    TemporalOrderError,
)
from precision_weeder.video.timing import TimestampReport, analyze_timestamps


def _consistent(report: TimestampReport) -> None:
    """Las filas de anomalias y los contadores dicen lo mismo."""
    kinds = Counter(row["kind"] for row in report.anomalies)
    assert kinds["duplicate"] == report.duplicates
    assert kinds["regression"] == report.regressions
    assert kinds["gap"] == report.gaps
    assert kinds["missing_pts"] == report.pts_missing
    assert report.pts_observed + report.pts_missing == report.frame_count
    assert len(report.timeline_s) == report.frame_count
    if report.monotonic_increasing is not None:
        assert report.monotonic_increasing == (report.duplicates == report.regressions == 0)


def test_complete_regular_pts_are_certified() -> None:
    report = analyze_timestamps([i / 30 for i in range(90)])
    _consistent(report)
    assert report.source == "pts" and report.pts_missing == 0
    assert report.monotonic_increasing is True and not report.anomalies
    assert report.fps_effective == pytest.approx(30.0, abs=1e-9, rel=0)
    assert report.timeline_s[5] == pytest.approx(5 / 30, abs=1e-15, rel=0)


@pytest.mark.parametrize(
    ("values", "kind", "count"),
    [
        ([0.0, 0.0, 0.0], "duplicate", 2),
        ([2.0, 1.0, 0.0], "regression", 2),
        ([0.0, 0.0], "duplicate", 1),
        ([1.0, 0.5], "regression", 1),
    ],
)
def test_nonpositive_intervals_produce_rows_and_null_metrics(
    values: list[float], kind: str, count: int
) -> None:
    """Sin intervalos positivos: filas presentes, metricas nulas y no certificadas."""
    report = analyze_timestamps(values)
    _consistent(report)
    assert [row["kind"] for row in report.anomalies] == [kind] * count
    assert report.monotonic_increasing is False
    assert report.interval_stats_s == {} and report.gap_threshold_s is None
    assert report.fps_effective is None and report.duration_observed_s is None
    assert "no certificado" in report.fps_effective_method


def test_regression_with_positive_span_does_not_certify_fps() -> None:
    """[0, 1, 0.5, 2]: el tramo total es positivo, pero los PTS no son monotonos."""
    report = analyze_timestamps([0.0, 1.0, 0.5, 2.0])
    _consistent(report)
    assert report.regressions == 1 and report.fps_effective is None


def test_missing_pts_keep_observed_regression_and_label_estimate() -> None:
    """Sonda del Director: la estimacion por indice no borra la regresion observada."""
    report = analyze_timestamps([0.0, 0.1, 0.05, None, 0.2], fps_nominal=10.0)
    _consistent(report)
    assert report.source == "index_fps_estimate"
    assert report.pts_missing == 1 and report.pts_observed == 4
    (regression,) = [row for row in report.anomalies if row["kind"] == "regression"]
    assert (regression["after_index"], regression["before_index"]) == (1, 2)
    assert {"kind": "missing_pts", "frame_index": 3} in report.anomalies
    assert report.monotonic_increasing is False
    # La linea temporal estimada no se presenta como medida ni certifica FPS/duracion.
    assert report.timeline_s == pytest.approx((0.0, 0.1, 0.2, 0.3, 0.4), abs=1e-12, rel=0)
    assert report.fps_effective is None and report.duration_observed_s is None
    assert "index_fps_estimate" in report.fps_effective_method


def test_regression_across_a_missing_frame_is_still_reported() -> None:
    report = analyze_timestamps([0.0, 0.2, None, 0.1], fps_nominal=10.0)
    _consistent(report)
    (regression,) = [row for row in report.anomalies if row["kind"] == "regression"]
    assert (regression["after_index"], regression["before_index"]) == (1, 3)


def test_gaps_are_only_measured_between_adjacent_frames() -> None:
    """Un hueco por PTS ausente no se confunde con un salto temporal."""
    report = analyze_timestamps([0.0, 0.1, None, 0.3, 0.4, 0.9], fps_nominal=10.0)
    _consistent(report)
    gaps = [row for row in report.anomalies if row["kind"] == "gap"]
    assert [(row["after_index"], row["before_index"]) for row in gaps] == [(4, 5)]


def test_all_pts_missing() -> None:
    report = analyze_timestamps([None, None, None], fps_nominal=10.0)
    _consistent(report)
    assert report.pts_observed == 0 and report.monotonic_increasing is None
    assert report.first_s is None and report.last_s is None
    assert [row["kind"] for row in report.anomalies] == ["missing_pts"] * 3
    with pytest.raises(TemporalOrderError):
        analyze_timestamps([None, None], fps_nominal=10.0, allow_index_estimate=False)


@pytest.mark.parametrize(
    "values",
    [
        [0.0, math.nan, None, 0.3],
        [0.0, math.inf, None, 0.3],
        [None, -math.inf],
        [0.0, math.nan],
        [0.0, "0.1"],
        [0.0, True],
    ],
)
def test_nonfinite_or_nonnumeric_pts_are_rejected_even_with_missing(values: list) -> None:
    with pytest.raises(TemporalOrderError):
        analyze_timestamps(values, fps_nominal=10.0)


@pytest.mark.parametrize("fps", [0.0, -30.0, math.nan, math.inf, True])
def test_invalid_nominal_fps_is_rejected(fps: float) -> None:
    with pytest.raises(SchemaValidationError, match="fps_nominal"):
        analyze_timestamps([0.0, None, 0.2], fps_nominal=fps)


@pytest.mark.parametrize("factor", [1.0, 0.5, math.nan, math.inf])
def test_invalid_gap_factor_is_rejected(factor: float) -> None:
    with pytest.raises(SchemaValidationError, match="gap_factor"):
        analyze_timestamps([0.0, 0.1], gap_factor=factor)


def test_errors_are_explicit_not_empty_success() -> None:
    for call in (
        lambda: analyze_timestamps([]),
        lambda: analyze_timestamps([0.0, None], fps_nominal=None),
    ):
        with pytest.raises(PrecisionWeederError):
            call()


def test_report_dict_omits_full_timeline() -> None:
    report = analyze_timestamps([0.0, 0.1])
    assert "timeline_s" not in report.to_dict()
    assert report.to_dict()["pts_observed"] == 2
