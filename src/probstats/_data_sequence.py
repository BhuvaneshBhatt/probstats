"""Internal implementation for data sequence."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ._data_core import _finite_array, _integer_parameter


@dataclass(frozen=True, slots=True)
class SequenceSummary:
    count: int
    minimum: float
    maximum: float
    mean: float
    variance: float
    first: float
    last: float
    change: float
    lag1_autocorrelation: float
    runs: int
    turning_points: int
    longest_increasing_run: int
    longest_decreasing_run: int


def run_length_encode(sequence: Iterable) -> tuple[tuple[object, int], ...]:
    values = list(sequence)
    if not values:
        return ()
    runs = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            runs.append((current, count))
            current, count = value, 1
    runs.append((current, count))
    return tuple(runs)


def differences(data, lag=1):
    x = _finite_array(data, ndim=1)
    lag = _integer_parameter(lag, name="lag", minimum=1)
    if lag >= x.size:
        raise ValueError("lag must be smaller than the data length")
    return x[lag:] - x[:-lag]


def _longest_monotone(diff: np.ndarray, positive: bool) -> int:
    best = current = 1
    for delta in diff:
        if (delta > 0) if positive else (delta < 0):
            current += 1
            best = max(best, current)
        else:
            current = 1
    return best


def sequence_summary(data) -> SequenceSummary:
    x = _finite_array(data, ndim=1)
    d = np.diff(x)
    if x.size > 1 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0:
        lag1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    else:
        lag1 = float("nan")
    signs = np.sign(x - np.median(x))
    signs = signs[signs != 0]
    runs = 0 if signs.size == 0 else int(1 + np.count_nonzero(signs[1:] != signs[:-1]))
    turns = int(np.count_nonzero(d[:-1] * d[1:] < 0)) if d.size > 1 else 0
    return SequenceSummary(
        int(x.size),
        float(x.min()),
        float(x.max()),
        float(x.mean()),
        float(np.var(x, ddof=1)) if x.size > 1 else float("nan"),
        float(x[0]),
        float(x[-1]),
        float(x[-1] - x[0]),
        lag1,
        runs,
        turns,
        _longest_monotone(d, True),
        _longest_monotone(d, False),
    )
