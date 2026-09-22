"""Internal implementation for data rolling."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ._data_core import (
    _finite_array,
    _integer_parameter,
    _validated_ddof,
)


def _window_bounds(n: int, window: int, center: bool):
    window = _integer_parameter(window, name="window", minimum=1)
    for i in range(n):
        if center:
            left = (window - 1) // 2
            right = window - left
            yield max(0, i - left), min(n, i + right)
        else:
            yield max(0, i - window + 1), i + 1


def rolling_apply(
    data,
    window: int,
    func: Callable[[np.ndarray], float],
    *,
    min_periods=None,
    center=False,
):
    """Apply ``func`` to rolling windows, returning NaN until ``min_periods``."""
    x = _finite_array(data, ndim=1)
    if min_periods is None:
        min_periods = window
    if (
        not isinstance(min_periods, (int, np.integer))
        or min_periods <= 0
        or min_periods > window
    ):
        raise ValueError("min_periods must be in [1, window]")
    out = np.full(x.size, np.nan)
    for i, (lo, hi) in enumerate(_window_bounds(x.size, window, center)):
        if hi - lo >= min_periods:
            out[i] = float(func(x[lo:hi]))
    return out


def _rolling_reduction(data, window, reducer, min_periods, center):
    x = _finite_array(data, ndim=1)
    window = _integer_parameter(window, name="window", minimum=1)
    if min_periods is None:
        min_periods = window
    min_periods = _integer_parameter(min_periods, name="min_periods", minimum=1)
    if min_periods > window:
        raise ValueError("min_periods must be in [1, window]")

    out = np.full(x.size, np.nan)
    if window <= x.size:
        windows = np.lib.stride_tricks.sliding_window_view(x, window)
        reduced = reducer(windows, axis=-1)
        if center:
            left = (window - 1) // 2
            out[left : left + reduced.size] = reduced
        else:
            out[window - 1 :] = reduced

    if center:
        left = (window - 1) // 2
        right = window - left
        edge_indices = list(range(min(left, x.size)))
        edge_indices.extend(range(max(left, x.size - right + 1), x.size))
    else:
        edge_indices = range(min(window - 1, x.size))

    for i in edge_indices:
        if center:
            lo, hi = max(0, i - left), min(x.size, i + right)
        else:
            lo, hi = 0, i + 1
        if hi - lo >= min_periods:
            out[i] = reducer(x[lo:hi])
    return out


def rolling_sum(data, window, *, min_periods=None, center=False):
    return _rolling_reduction(data, window, np.sum, min_periods, center)


def rolling_mean(data, window, *, min_periods=None, center=False):
    return _rolling_reduction(data, window, np.mean, min_periods, center)


def rolling_variance(data, window, *, ddof=1, min_periods=None, center=False):
    ddof = _validated_ddof(ddof)
    return rolling_apply(
        data,
        window,
        lambda z: np.var(z, ddof=ddof) if z.size > ddof else np.nan,
        min_periods=min_periods,
        center=center,
    )


def rolling_standard_deviation(data, window, *, ddof=1, min_periods=None, center=False):
    return np.sqrt(
        rolling_variance(
            data, window, ddof=ddof, min_periods=min_periods, center=center
        )
    )


def rolling_min(data, window, *, min_periods=None, center=False):
    return rolling_apply(data, window, np.min, min_periods=min_periods, center=center)


def rolling_max(data, window, *, min_periods=None, center=False):
    return rolling_apply(data, window, np.max, min_periods=min_periods, center=center)


def _paired_arrays(x, y):
    xx = _finite_array(x, ndim=1)
    yy = _finite_array(y, ndim=1)
    if xx.size != yy.size:
        raise ValueError("x and y must have the same length")
    return xx, yy


def rolling_covariance(x, y, window, *, ddof=1, min_periods=None, center=False):
    ddof = _validated_ddof(ddof)
    xx, yy = _paired_arrays(x, y)
    if min_periods is None:
        min_periods = window
    if (
        not isinstance(min_periods, (int, np.integer))
        or min_periods <= 0
        or min_periods > window
    ):
        raise ValueError("min_periods must be in [1, window]")
    out = np.full(xx.size, np.nan)
    for i, (lo, hi) in enumerate(_window_bounds(xx.size, window, center)):
        if hi - lo >= min_periods and hi - lo > ddof:
            out[i] = np.cov(xx[lo:hi], yy[lo:hi], ddof=ddof)[0, 1]
    return out


def rolling_correlation(x, y, window, *, min_periods=None, center=False):
    xx, yy = _paired_arrays(x, y)
    cov = rolling_covariance(
        xx, yy, window, ddof=1, min_periods=min_periods, center=center
    )
    sx = rolling_standard_deviation(
        xx, window, ddof=1, min_periods=min_periods, center=center
    )
    sy = rolling_standard_deviation(
        yy, window, ddof=1, min_periods=min_periods, center=center
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        result = cov / (sx * sy)
    return result


def rolling_quantile(
    data, window, q, *, min_periods=None, center=False, method="linear"
):
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    return rolling_apply(
        data,
        window,
        lambda z: np.quantile(z, q, method=method),
        min_periods=min_periods,
        center=center,
    )


def running_sum(data):
    return np.cumsum(_finite_array(data, ndim=1))


def running_mean(data):
    x = _finite_array(data, ndim=1)
    return np.cumsum(x) / np.arange(1, x.size + 1)


def running_variance(data, *, ddof=1):
    """Numerically stable expanding variance using Welford's algorithm."""
    x = _finite_array(data, ndim=1)
    ddof = _validated_ddof(ddof)
    out = np.full(x.size, np.nan)
    mean = 0.0
    m2 = 0.0
    for i, value in enumerate(x, start=1):
        delta = value - mean
        mean += delta / i
        m2 += delta * (value - mean)
        if i > ddof:
            out[i - 1] = m2 / (i - ddof)
    return out


def running_standard_deviation(data, *, ddof=1):
    return np.sqrt(running_variance(data, ddof=ddof))


class _FenwickCounts:
    """Prefix-count tree supporting incremental order-statistic queries."""

    def __init__(self, size: int):
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index: int) -> None:
        index += 1
        while index < self.tree.size:
            self.tree[index] += 1
            index += index & -index

    def select(self, rank: int) -> int:
        """Return the zero-based value index containing zero-based order ``rank``."""
        index = 0
        bit = 1 << (self.tree.size.bit_length() - 1)
        target = rank + 1
        while bit:
            candidate = index + bit
            if candidate < self.tree.size and self.tree[candidate] < target:
                index = candidate
                target -= int(self.tree[candidate])
            bit >>= 1
        return index


def running_quantile(data, q, *, method="linear"):
    """Expanding quantiles with logarithmic-time order-statistic updates.

    The common NumPy methods ``linear``, ``lower``, ``higher``, ``midpoint``,
    and ``nearest`` use a coordinate-compressed Fenwick tree. Other NumPy
    quantile methods use direct NumPy prefix calculations to preserve their
    exact interpolation conventions.
    """
    x = _finite_array(data, ndim=1)
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    fast_methods = {"linear", "lower", "higher", "midpoint", "nearest"}
    if method not in fast_methods:
        return np.asarray(
            [np.quantile(x[: i + 1], q, method=method) for i in range(x.size)],
            dtype=float,
        )

    values, encoded = np.unique(x, return_inverse=True)
    counts = _FenwickCounts(values.size)
    out = np.empty(x.size, dtype=float)
    for i, value_index in enumerate(encoded):
        counts.add(int(value_index))
        position = i * q
        lower = int(np.floor(position))
        upper = int(np.ceil(position))
        if method == "lower":
            out[i] = values[counts.select(lower)]
        elif method == "higher":
            out[i] = values[counts.select(upper)]
        elif method == "nearest":
            nearest = int(np.rint(position))
            out[i] = values[counts.select(nearest)]
        else:
            lo = values[counts.select(lower)]
            hi = values[counts.select(upper)]
            if method == "midpoint":
                out[i] = (lo + hi) / 2
            else:
                out[i] = lo + (position - lower) * (hi - lo)
    return out
