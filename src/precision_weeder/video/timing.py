"""Analisis de marcas temporales por cuadro (E0, orden §2).

El tiempo se toma de PTS por `time_base`. El diagnostico (duplicados, regresiones,
saltos y ausencias) se calcula **siempre sobre los PTS observados**, sin
normalizarlos. Si faltan PTS y la configuracion lo permite, la linea temporal que se
usa aguas abajo es `indice / fps_nominal`, etiquetada `index_fps_estimate`; esa
estimacion nunca se presenta como PTS medido ni borra el diagnostico de los PTS
presentes, y con ella no se certifican FPS efectivo ni duracion observada.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core.errors import SchemaValidationError, TemporalOrderError

__all__ = ["TimestampReport", "analyze_timestamps"]


@dataclass(frozen=True)
class TimestampReport:
    """Diagnostico de PTS observados y linea temporal resultante, con su metodo.

    `source` es el origen de `timeline_s` (la linea usada aguas abajo); los contadores,
    `monotonic_increasing`, las estadisticas y `anomalies` describen solo los PTS
    observados. `monotonic_increasing` es `None` si hay menos de dos PTS observados.
    """

    source: str
    unit: str
    frame_count: int
    pts_observed: int
    pts_missing: int
    first_s: float | None
    last_s: float | None
    monotonic_increasing: bool | None
    duplicates: int
    regressions: int
    gaps: int
    gap_threshold_s: float | None
    interval_stats_s: dict[str, float]
    fps_effective: float | None
    fps_effective_method: str
    duration_observed_s: float | None
    duration_observed_method: str
    anomalies: list[dict[str, Any]] = field(default_factory=list)
    timeline_s: tuple[float, ...] = field(default=(), repr=False)

    def to_dict(self) -> dict[str, Any]:
        """Resumen serializable; la linea temporal completa va en `timestamps.csv`."""
        return {key: value for key, value in self.__dict__.items() if key != "timeline_s"}


def _observed_values(timestamps_s: list[float | None]) -> list[tuple[int, float]]:
    observed = []
    for index, value in enumerate(timestamps_s):
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            raise TemporalOrderError(f"marca temporal no numerica en el cuadro {index}: {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise TemporalOrderError(f"marca temporal no finita en el cuadro {index}: {value!r}")
        observed.append((index, number))
    return observed


def analyze_timestamps(
    timestamps_s: list[float | None],
    fps_nominal: float | None = None,
    gap_factor: float = 1.5,
    allow_index_estimate: bool = True,
) -> TimestampReport:
    """Analiza las marcas temporales de los cuadros en orden de decodificacion.

    Duplicados y regresiones se buscan entre PTS observados consecutivos (aunque
    medie un cuadro sin PTS); los saltos y las estadisticas de intervalo, solo entre
    cuadros adyacentes con PTS, que son los pares comparables. Cada ausencia de PTS
    genera una fila `missing_pts`.

    Raises:
        TemporalOrderError: si no hay cuadros, hay marcas no finitas o no numericas, o
            faltan PTS y no se admite la estimacion por indice (o no hay FPS valido).
        SchemaValidationError: si `gap_factor` o `fps_nominal` no son validos.
    """
    if not timestamps_s:
        raise TemporalOrderError("sin cuadros decodificados: no hay marcas temporales")
    if isinstance(gap_factor, bool) or not (math.isfinite(gap_factor) and gap_factor > 1.0):
        raise SchemaValidationError(f"gap_factor debe ser finito y > 1; recibido {gap_factor!r}")
    if fps_nominal is not None and (
        isinstance(fps_nominal, bool) or not (math.isfinite(fps_nominal) and fps_nominal > 0)
    ):
        raise SchemaValidationError(f"fps_nominal debe ser finito y > 0; recibido {fps_nominal!r}")

    observed = _observed_values(timestamps_s)
    observed_indices = {index for index, _ in observed}
    missing = [index for index in range(len(timestamps_s)) if index not in observed_indices]
    if missing:
        if not allow_index_estimate or fps_nominal is None:
            raise TemporalOrderError(
                f"{len(missing)} cuadros sin PTS y sin estimacion autorizada por indice/FPS"
            )
        source = "index_fps_estimate"
        timeline = tuple(index / fps_nominal for index in range(len(timestamps_s)))
    else:
        source = "pts"
        timeline = tuple(value for _, value in observed)

    anomalies: list[dict[str, Any]] = [
        {"kind": "missing_pts", "frame_index": index} for index in missing
    ]
    duplicates = regressions = 0
    comparable: list[tuple[int, float, float]] = []  # (indice previo, intervalo, t previo)
    for (index_a, value_a), (index_b, value_b) in itertools.pairwise(observed):
        interval = value_b - value_a
        if interval <= 0.0:
            kind = "duplicate" if interval == 0.0 else "regression"
            duplicates += kind == "duplicate"
            regressions += kind == "regression"
            anomalies.append(
                {
                    "kind": kind,
                    "after_index": index_a,
                    "before_index": index_b,
                    "interval_s": interval,
                    "at_s": value_a,
                }
            )
        if index_b == index_a + 1:
            comparable.append((index_a, interval, value_a))

    intervals = np.array([interval for _, interval, _ in comparable], dtype=np.float64)
    positive = intervals[intervals > 0.0]
    stats: dict[str, float] = {}
    gap_threshold = None
    gaps = 0
    if positive.size:
        stats = {
            "min": float(np.min(intervals)),
            "p05": float(np.percentile(intervals, 5)),
            "median": float(np.median(intervals)),
            "mean": float(np.mean(intervals)),
            "p95": float(np.percentile(intervals, 95)),
            "max": float(np.max(intervals)),
            "std": float(np.std(intervals)),
            "comparable_pairs": float(intervals.size),
        }
        gap_threshold = gap_factor * float(np.median(positive))
        for index_a, interval, value_a in comparable:
            if interval > gap_threshold:
                gaps += 1
                anomalies.append(
                    {
                        "kind": "gap",
                        "after_index": index_a,
                        "before_index": index_a + 1,
                        "interval_s": interval,
                        "at_s": value_a,
                    }
                )
    anomalies.sort(key=lambda row: (row.get("before_index", row.get("frame_index")), row["kind"]))

    monotonic = (duplicates == 0 and regressions == 0) if len(observed) >= 2 else None
    # FPS efectivo y duracion solo se certifican con PTS completos, monotonos y con
    # intervalos positivos; en otro caso quedan nulos con el motivo declarado.
    certified = source == "pts" and monotonic is True and bool(stats)
    if certified:
        span = observed[-1][1] - observed[0][1]
        fps_effective = (len(observed) - 1) / span
        duration = span + stats["median"]
        fps_method = "(pts) (N-1) / (t_last - t_first)"
        duration_method = "(pts) t_last - t_first + mediana del intervalo"
    else:
        fps_effective = duration = None
        reason = (
            "linea temporal estimada por indice/FPS"
            if source != "pts"
            else "PTS no monotonos o sin intervalos positivos"
        )
        fps_method = duration_method = f"({source}) no certificado: {reason}"
    return TimestampReport(
        source=source,
        unit="s",
        frame_count=len(timestamps_s),
        pts_observed=len(observed),
        pts_missing=len(missing),
        first_s=observed[0][1] if observed else None,
        last_s=observed[-1][1] if observed else None,
        monotonic_increasing=monotonic,
        duplicates=duplicates,
        regressions=regressions,
        gaps=gaps,
        gap_threshold_s=gap_threshold,
        interval_stats_s=stats,
        fps_effective=fps_effective,
        fps_effective_method=fps_method,
        duration_observed_s=duration,
        duration_observed_method=duration_method,
        anomalies=anomalies,
        timeline_s=timeline,
    )
