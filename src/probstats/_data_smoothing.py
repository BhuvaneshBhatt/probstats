"""Internal implementation for data smoothing."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._data_core import (
    WeightedData,
    _finite_array,
    _integer_parameter,
    _validated_weights,
)


def _kernel_weights(distance: np.ndarray, kernel: str) -> np.ndarray:
    u = np.asarray(distance, dtype=float)
    key = kernel.lower().replace("-", "_")
    if key == "gaussian":
        return np.exp(-0.5 * u * u)
    a = np.abs(u)
    if key == "epanechnikov":
        return np.where(a < 1, 0.75 * (1 - u * u), 0.0)
    if key == "tricube":
        return np.where(a < 1, (1 - a**3) ** 3, 0.0)
    if key in {"uniform", "box"}:
        return np.where(a <= 1, 1.0, 0.0)
    raise ValueError(f"unknown kernel: {kernel!r}")


@dataclass(frozen=True, slots=True)
class SmootherResult:
    """Smoothed values plus residuals on the original observation coordinates."""

    x: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    method: str
    bandwidth: float | None = None
    span: float | None = None
    residual_x: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("x", "fitted", "residuals"):
            value = np.asarray(getattr(self, name), dtype=float).copy()
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if self.x.shape != self.fitted.shape:
            raise ValueError("x and fitted must have the same shape")
        residual_x = self.x if self.residual_x is None else self.residual_x
        residual_x = np.asarray(residual_x, dtype=float).copy()
        if residual_x.shape != self.residuals.shape:
            raise ValueError("residual_x and residuals must have the same shape")
        if not np.all(np.isfinite(residual_x)):
            raise ValueError("residual_x must contain only finite values")
        residual_x.setflags(write=False)
        object.__setattr__(self, "residual_x", residual_x)


def _xyw(x, y=None, weights=None):
    if isinstance(y, WeightedData):
        raise TypeError("pass WeightedData as the first argument, not y")
    if isinstance(x, WeightedData):
        if y is None:
            arr = np.asarray(x.values, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError(
                    "WeightedData without y must contain two columns: x and y"
                )
            return arr[:, 0], arr[:, 1], x.weights
        if weights is not None:
            raise ValueError("weights must not be supplied with WeightedData")
        return _finite_array(x.values, ndim=1), _finite_array(y, ndim=1), x.weights
    xx = _finite_array(x, ndim=1)
    if y is None:
        raise ValueError("y is required unless x is two-column WeightedData")
    yy = _finite_array(y, ndim=1)
    if xx.size != yy.size:
        raise ValueError("x and y must have the same length")
    ww = np.ones(xx.size) if weights is None else _validated_weights(weights, xx.size)
    return xx, yy, ww


def _fit_local_subset(offsets, yy, weights, degree):
    positive = weights > 0
    if np.count_nonzero(positive) < degree + 1:
        raise ValueError("at least degree + 1 positive local weights are required")
    local_x = offsets[positive]
    local_y = yy[positive]
    local_w = weights[positive]
    if degree == 0:
        return float(np.average(local_y, weights=local_w))
    design = np.vander(local_x, N=degree + 1, increasing=True)
    sw = np.sqrt(local_w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], local_y * sw, rcond=None)
    return float(coef[0])


def _local_fit(xx, yy, obs_w, x_eval, bandwidth, degree, kernel):
    """Fit local polynomials while avoiding rows outside compact kernels."""
    if bandwidth <= 0 or not np.isfinite(bandwidth):
        raise ValueError("bandwidth must be positive")
    order = np.argsort(xx, kind="stable")
    sx, sy, sw = xx[order], yy[order], obs_w[order]
    compact = kernel.lower().replace("-", "_") != "gaussian"
    targets = np.asarray(x_eval, dtype=float)
    fitted = np.full(targets.shape, np.nan, dtype=float)
    for index, target in np.ndenumerate(targets):
        if compact:
            left = np.searchsorted(sx, target - bandwidth, side="left")
            right = np.searchsorted(sx, target + bandwidth, side="right")
        else:
            left, right = 0, sx.size
        local_x = sx[left:right]
        local_y = sy[left:right]
        local_obs_w = sw[left:right]
        offsets = local_x - target
        weights = local_obs_w * _kernel_weights(offsets / bandwidth, kernel)
        if np.count_nonzero(weights > 0) < degree + 1:
            eligible = np.flatnonzero(sw > 0)
            if eligible.size < degree + 1:
                raise ValueError(
                    "at least degree + 1 positive-weight observations are required"
                )
            nearest = eligible[np.argsort(np.abs(sx[eligible] - target))[: degree + 1]]
            local_x = sx[nearest]
            local_y = sy[nearest]
            offsets = local_x - target
            weights = np.maximum(sw[nearest], np.finfo(float).tiny)
        fitted[index] = _fit_local_subset(offsets, local_y, weights, degree)
    return fitted


def local_polynomial_smooth(
    x,
    y=None,
    *,
    x_eval=None,
    bandwidth=None,
    degree=1,
    kernel="gaussian",
    weights=None,
) -> SmootherResult:
    """Local polynomial regression, including Nadaraya-Watson at degree 0."""
    xx, yy, ww = _xyw(x, y, weights)
    degree = _integer_parameter(degree, name="degree")
    if bandwidth is None:
        scale = np.std(xx, ddof=1) if xx.size > 1 else 0.0
        bandwidth = 1.06 * scale * max(xx.size, 2) ** (-1 / 5)
        if bandwidth <= 0:
            raise ValueError("bandwidth is undefined for constant x")
    xe = xx if x_eval is None else _finite_array(x_eval, ndim=1)
    fitted_training = _local_fit(xx, yy, ww, xx, float(bandwidth), degree, kernel)
    fitted = (
        fitted_training
        if x_eval is None
        else _local_fit(xx, yy, ww, xe, float(bandwidth), degree, kernel)
    )
    residuals = yy - fitted_training
    return SmootherResult(
        xe.copy(),
        fitted,
        residuals,
        f"local_polynomial_{degree}",
        float(bandwidth),
        None,
        xx.copy(),
    )


def kernel_smooth(x, y=None, **kwargs) -> SmootherResult:
    """Nadaraya-Watson kernel smoother."""
    kwargs.pop("degree", None)
    result = local_polynomial_smooth(x, y, degree=0, **kwargs)
    return SmootherResult(
        result.x,
        result.fitted,
        result.residuals,
        "kernel",
        result.bandwidth,
        None,
        result.residual_x,
    )


def _nearest_sorted_window(values, target, size):
    """Return bounds of the contiguous ``size`` nearest sorted observations."""
    n = values.size
    if size >= n:
        return 0, n
    pos = int(np.searchsorted(values, target))
    left = max(0, pos - size)
    right = min(n, pos + size)
    candidates = values[left:right]
    if candidates.size <= size:
        return left, right
    # For sorted one-dimensional data, the nearest set is contiguous. Shift a
    # size-wide window until dropping either edge can no longer improve it.
    start = max(0, min(pos - size // 2, n - size))
    while start > 0 and abs(values[start - 1] - target) < abs(
        values[start + size - 1] - target
    ):
        start -= 1
    while start + size < n and abs(values[start + size] - target) < abs(
        values[start] - target
    ):
        start += 1
    return start, start + size


def lowess(
    x, y=None, *, x_eval=None, span=2 / 3, degree=1, robust_iterations=2, weights=None
) -> SmootherResult:
    """Locally weighted regression with sorted-neighborhood reuse."""
    xx, yy, obs_w = _xyw(x, y, weights)
    if not 0 < span <= 1:
        raise ValueError("span must be in (0, 1]")
    if degree not in (0, 1, 2):
        raise ValueError("LOWESS degree must be 0, 1, or 2")
    robust_iterations = _integer_parameter(robust_iterations, name="robust_iterations")
    k = max(degree + 1, int(np.ceil(span * xx.size)))
    order = np.argsort(xx, kind="stable")
    sx, sy, sw = xx[order], yy[order], obs_w[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    robust = np.ones(sx.size)

    def fit_targets(targets, robust_weights):
        targets = np.asarray(targets, dtype=float)
        out = np.empty(targets.shape, dtype=float)
        active_w = sw * robust_weights
        eligible = np.flatnonzero(active_w > 0)
        if eligible.size < degree + 1:
            raise ValueError(
                "LOWESS needs at least degree + 1 observations with positive combined weight"
            )
        for index, target in np.ndenumerate(targets):
            left, right = _nearest_sorted_window(sx, target, k)
            local_x = sx[left:right]
            local_y = sy[left:right]
            local_active = active_w[left:right]
            distances = np.abs(local_x - target)
            h = float(distances.max()) if distances.size else 0.0
            if h <= 0:
                positive_dist = np.abs(sx[eligible] - target)
                positive_dist = positive_dist[positive_dist > 0]
                h = float(positive_dist.min()) if positive_dist.size else 1.0
            weights = local_active * _kernel_weights(distances / h, "tricube")
            if np.count_nonzero(weights > 0) < degree + 1:
                nearest = eligible[
                    np.argsort(np.abs(sx[eligible] - target))[: degree + 1]
                ]
                local_x = sx[nearest]
                local_y = sy[nearest]
                weights = active_w[nearest]
            out[index] = _fit_local_subset(local_x - target, local_y, weights, degree)
        return out

    fitted_sorted = fit_targets(sx, robust)
    for _ in range(robust_iterations):
        residual = sy - fitted_sorted
        mad = np.median(np.abs(residual))
        if mad <= np.finfo(float).eps:
            break
        u = residual / (6.0 * mad)
        robust = np.where(np.abs(u) < 1, (1 - u * u) ** 2, 0.0)
        fitted_sorted = fit_targets(sx, robust)

    fitted_training = fitted_sorted[inverse]
    xe = xx if x_eval is None else _finite_array(x_eval, ndim=1)
    fitted = fitted_training if x_eval is None else fit_targets(xe, robust)
    return SmootherResult(
        xe.copy(),
        fitted,
        yy - fitted_training,
        "lowess",
        None,
        float(span),
        xx.copy(),
    )
