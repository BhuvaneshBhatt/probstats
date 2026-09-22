"""Descriptive and empirical statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import WeightedData


def _array(data) -> np.ndarray:
    if isinstance(data, WeightedData):
        raise TypeError(
            "this statistic does not define weighted semantics; pass plain observations "
            "or use a WeightedData method that explicitly supports the statistic"
        )
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        raise ValueError("data must not be empty")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must contain only finite values")
    return x


def mean(data, weights=None):
    """Return the arithmetic mean, with optional explicit weights."""
    if isinstance(data, WeightedData):
        if weights is not None:
            raise ValueError("weights must not be supplied with WeightedData")
        value = data.mean()
    elif weights is None:
        value = np.mean(_array(data))
    else:
        value = WeightedData(_array(data), weights).mean()
    return float(value) if np.ndim(value) == 0 else value


def median(data):
    """Return the sample median."""
    if isinstance(data, WeightedData):
        return data.quantile(0.5)
    return float(np.median(_array(data)))


def quantile(data, q, method="linear"):
    """Return empirical quantiles using the requested interpolation method."""
    if isinstance(data, WeightedData):
        if method != "linear":
            raise ValueError(
                "weighted quantiles use the linear weighted-CDF convention"
            )
        return data.quantile(q)
    return np.quantile(_array(data), q, method=method)


def variance(data, ddof=1):
    """Return the sample variance using the requested degrees-of-freedom correction."""
    if isinstance(data, WeightedData):
        value = data.variance(ddof=ddof)
        return float(value) if np.ndim(value) == 0 else value
    return float(np.var(_array(data), ddof=ddof))


def standard_deviation(data, ddof=1):
    """Return the sample standard deviation."""
    if isinstance(data, WeightedData):
        value = data.standard_deviation(ddof=ddof)
        return float(value) if np.ndim(value) == 0 else value
    return float(np.std(_array(data), ddof=ddof))


def _weighted_pair(x: WeightedData, y):
    if x.values.ndim != 1:
        raise ValueError("paired weighted statistics require scalar observations")
    if isinstance(y, WeightedData):
        if y.values.ndim != 1 or len(y) != len(x):
            raise ValueError("weighted series must have matching lengths")
        if not np.array_equal(x.weights, y.weights):
            raise ValueError("weighted series must use the same weights")
        y_values = y.values
    else:
        y_values = _array(y)
        if y_values.ndim != 1 or y_values.size != len(x):
            raise ValueError("x and y must have the same length")
    return WeightedData(np.column_stack((x.values, y_values)), x.weights)


def covariance(x, y=None, ddof=1):
    """Return covariance for paired observations or weighted data."""
    if isinstance(x, WeightedData):
        data = x if y is None else _weighted_pair(x, y)
        value = data.covariance(ddof=ddof)
        if y is not None:
            value = value[0, 1]
        return float(value) if np.ndim(value) == 0 else value
    if y is None:
        raise ValueError("y is required unless x is WeightedData")
    return float(np.cov(_array(x), _array(y), ddof=ddof)[0, 1])


def correlation(x, y=None):
    """Return the Pearson correlation for paired observations or weighted data."""
    if isinstance(x, WeightedData):
        data = x if y is None else _weighted_pair(x, y)
        value = data.correlation()
        if y is not None:
            value = value[0, 1]
        return float(value) if np.ndim(value) == 0 else value
    if y is None:
        raise ValueError("y is required unless x is WeightedData")
    return float(np.corrcoef(_array(x), _array(y))[0, 1])


def skewness(data, bias=False):
    """Return the standardized third central moment with optional bias correction."""
    x = _array(data)
    c = x - x.mean()
    m2 = np.mean(c * c)
    m3 = np.mean(c**3)
    if m2 == 0:
        return 0.0
    g = m3 / m2**1.5
    n = x.size
    if bias or n < 3:
        return float(g)
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g)


def kurtosis(data, fisher=True, bias=False):
    """Return sample kurtosis, optionally on the Fisher excess-kurtosis scale."""
    x = _array(data)
    c = x - x.mean()
    m2 = np.mean(c * c)
    m4 = np.mean(c**4)
    if m2 == 0:
        return -3.0 if fisher else 0.0
    g = m4 / m2**2
    n = x.size
    if not bias and n > 3:
        g = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g - 3 * (n - 1)) + 3
    return float(g - 3 if fisher else g)


def trimmed_mean(data, proportion=0.1):
    """Return the mean after symmetrically trimming each tail."""
    x = np.sort(_array(data))
    if proportion < 0 or proportion >= 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    k = int(np.floor(proportion * x.size))
    return float(np.mean(x[k : x.size - k] if k else x))


def winsorize(data, proportion=0.1):
    """Return observations with each tail winsorized to boundary order statistics."""
    x = np.sort(_array(data).copy())
    if proportion < 0 or proportion >= 0.5:
        raise ValueError("proportion must be in [0, 0.5)")
    k = int(np.floor(proportion * x.size))
    if k:
        x[:k] = x[k]
        x[-k:] = x[-k - 1]
    return x


def median_absolute_deviation(data, scale=1.0):
    """Return the median absolute deviation about the sample median."""
    x = _array(data)
    m = np.median(x)
    return float(scale * np.median(np.abs(x - m)))


def interquartile_range(data):
    """Return the difference between the empirical 75th and 25th percentiles."""
    q = np.quantile(_array(data), [0.25, 0.75])
    return float(q[1] - q[0])


@dataclass(frozen=True, slots=True)
class EmpiricalDistribution:
    data: np.ndarray

    def __init__(self, data):
        x = np.sort(_array(data)).copy()
        x.setflags(write=False)
        object.__setattr__(self, "data", x)

    @property
    def n(self):
        return int(self.data.size)

    def cdf(self, x):
        return np.searchsorted(self.data, x, side="right") / self.n

    def quantile(self, q, method="linear"):
        """Return empirical quantiles using the requested interpolation method."""
        return np.quantile(self.data, q, method=method)

    def sample(self, size=None, rng=None):
        g = np.random.default_rng(rng)
        return g.choice(self.data, size=size, replace=True)

    @property
    def mean(self):
        """Return the arithmetic mean, with optional explicit weights."""
        return float(np.mean(self.data))

    @property
    def variance(self):
        """Return the sample variance using the requested degrees-of-freedom correction."""
        return float(np.var(self.data, ddof=1))


@dataclass(frozen=True, slots=True)
class DescriptiveSummary:
    count: int
    mean: float
    std: float
    minimum: float
    q1: float
    median: float
    q3: float
    maximum: float
    iqr: float
    mad: float
    skewness: float
    kurtosis: float


def describe(data):
    """Return a compact descriptive summary of a one-dimensional sample."""
    x = _array(data)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return DescriptiveSummary(
        x.size,
        float(x.mean()),
        float(x.std(ddof=1)) if x.size > 1 else float("nan"),
        float(x.min()),
        float(q1),
        float(med),
        float(q3),
        float(x.max()),
        float(q3 - q1),
        median_absolute_deviation(x),
        skewness(x),
        kurtosis(x),
    )


__all__ = [
    "DescriptiveSummary",
    "EmpiricalDistribution",
    "correlation",
    "covariance",
    "describe",
    "interquartile_range",
    "kurtosis",
    "mean",
    "median",
    "median_absolute_deviation",
    "quantile",
    "skewness",
    "standard_deviation",
    "trimmed_mean",
    "variance",
    "winsorize",
]
