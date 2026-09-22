"""Internal implementation for data exponential."""

from __future__ import annotations

import numpy as np

from ._data_core import _finite_array
from ._data_rolling import _paired_arrays


def _ew_alpha(*, alpha=None, span=None, com=None, halflife=None) -> float:
    provided = sum(v is not None for v in (alpha, span, com, halflife))
    if provided != 1:
        raise ValueError("specify exactly one of alpha, span, com, or halflife")
    if alpha is not None:
        a = float(alpha)
    elif span is not None:
        if span <= 1:
            raise ValueError("span must be greater than 1")
        a = 2.0 / (float(span) + 1.0)
    elif com is not None:
        if com < 0:
            raise ValueError("com must be nonnegative")
        a = 1.0 / (1.0 + float(com))
    else:
        if halflife <= 0:
            raise ValueError("halflife must be positive")
        a = 1.0 - np.exp(np.log(0.5) / float(halflife))
    if not 0 < a <= 1:
        raise ValueError("alpha must be in (0, 1]")
    return a


def exponentially_weighted_mean(
    data, *, alpha=None, span=None, com=None, halflife=None, adjust=False
):
    x = _finite_array(data, ndim=1)
    a = _ew_alpha(alpha=alpha, span=span, com=com, halflife=halflife)
    out = np.empty_like(x)
    if not adjust:
        out[0] = x[0]
        for i in range(1, x.size):
            out[i] = a * x[i] + (1.0 - a) * out[i - 1]
        return out
    numerator = 0.0
    denominator = 0.0
    decay = 1.0 - a
    for i, value in enumerate(x):
        numerator = value + decay * numerator
        denominator = 1.0 + decay * denominator
        out[i] = numerator / denominator
    return out


def exponentially_weighted_variance(
    data, *, alpha=None, span=None, com=None, halflife=None, bias=False
):
    """Online exponentially weighted variance with stable weighted moments."""
    x = _finite_array(data, ndim=1)
    a = _ew_alpha(alpha=alpha, span=span, com=com, halflife=halflife)
    decay = 1.0 - a
    out = np.full_like(x, np.nan)
    mean = x[0]
    second = x[0] ** 2
    weight_sq = 1.0
    out[0] = 0.0 if bias else np.nan
    for i in range(1, x.size):
        mean = a * x[i] + decay * mean
        second = a * x[i] ** 2 + decay * second
        v = max(second - mean * mean, 0.0)
        weight_sq = a * a + decay * decay * weight_sq
        if not bias:
            correction = 1.0 - weight_sq
            v = v / correction if correction > np.finfo(float).eps else np.nan
        out[i] = v
    return out


def exponentially_weighted_standard_deviation(data, **kwargs):
    return np.sqrt(exponentially_weighted_variance(data, **kwargs))


def exponentially_weighted_covariance(
    x, y, *, alpha=None, span=None, com=None, halflife=None, bias=False
):
    xx, yy = _paired_arrays(x, y)
    a = _ew_alpha(alpha=alpha, span=span, com=com, halflife=halflife)
    decay = 1.0 - a
    out = np.full_like(xx, np.nan)
    mean_x, mean_y = xx[0], yy[0]
    cross = xx[0] * yy[0]
    weight_sq = 1.0
    out[0] = 0.0 if bias else np.nan
    for i in range(1, xx.size):
        mean_x = a * xx[i] + decay * mean_x
        mean_y = a * yy[i] + decay * mean_y
        cross = a * xx[i] * yy[i] + decay * cross
        cov = cross - mean_x * mean_y
        weight_sq = a * a + decay * decay * weight_sq
        if not bias:
            correction = 1.0 - weight_sq
            cov = cov / correction if correction > np.finfo(float).eps else np.nan
        out[i] = cov
    return out


def exponentially_weighted_correlation(
    x, y, *, alpha=None, span=None, com=None, halflife=None
):
    kwargs = {
        "alpha": alpha,
        "span": span,
        "com": com,
        "halflife": halflife,
        "bias": False,
    }
    cov = exponentially_weighted_covariance(x, y, **kwargs)
    sx = exponentially_weighted_standard_deviation(x, **kwargs)
    sy = exponentially_weighted_standard_deviation(y, **kwargs)
    with np.errstate(divide="ignore", invalid="ignore"):
        return cov / (sx * sy)
