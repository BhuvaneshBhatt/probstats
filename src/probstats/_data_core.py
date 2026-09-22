"""Internal implementation for data core."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._validation import integer


def _finite_array(data, *, ndim: int | None = None) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        raise ValueError("data must not be empty")
    if ndim is not None and x.ndim != ndim:
        raise ValueError(f"data must be {ndim}-dimensional")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must contain only finite values")
    return x


def _validated_weights(weights, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != n:
        raise ValueError("weights must have one entry per observation")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite, nonnegative, and not all zero")
    return w


def _validated_ddof(ddof) -> float:
    value = float(ddof)
    if not np.isfinite(value) or value < 0:
        raise ValueError("ddof must be finite and nonnegative")
    return value


def _integer_parameter(value, *, name: str, minimum: int = 0) -> int:
    return integer(value, name=name, minimum=minimum)


def _weighted_quantile_1d(x: np.ndarray, w: np.ndarray, q) -> np.ndarray | float:
    qs = np.asarray(q, dtype=float)
    if np.any((qs < 0) | (qs > 1)):
        raise ValueError("q must be in [0, 1]")
    order = np.argsort(x, kind="stable")
    xs = x[order]
    ws = w[order]
    positive = ws > 0
    xs = xs[positive]
    ws = ws[positive]
    cumulative = np.cumsum(ws) - 0.5 * ws
    cumulative /= ws.sum()
    values = np.interp(qs, cumulative, xs, left=xs[0], right=xs[-1])
    return float(values) if qs.ndim == 0 else values


@dataclass(frozen=True, slots=True)
class WeightedData:
    """Observations paired with nonnegative observation weights.

    The first axis indexes observations; trailing axes are event dimensions.
    Weights are stored on their original scale and normalized only when a
    calculation requires probabilities. ``effective_n`` is Kish's effective
    sample size, ``(sum w)^2 / sum(w^2)``. Weighted quantiles use midpoint
    weighted-CDF interpolation rather than frequency-replication semantics.
    """

    values: np.ndarray
    weights: np.ndarray

    def __init__(self, values, weights=None):
        x = _finite_array(values)
        if x.ndim == 0:
            x = x.reshape(1)
        n = x.shape[0]
        w = (
            np.ones(n, dtype=float)
            if weights is None
            else _validated_weights(weights, n)
        )
        values_array = x.copy()
        weights_array = w.copy()
        values_array.setflags(write=False)
        weights_array.setflags(write=False)
        object.__setattr__(self, "values", values_array)
        object.__setattr__(self, "weights", weights_array)

    def __len__(self) -> int:
        return int(self.values.shape[0])

    @property
    def ndim(self) -> int:
        return self.values.ndim

    @property
    def total_weight(self) -> float:
        return float(self.weights.sum())

    @property
    def normalized_weights(self) -> np.ndarray:
        return self.weights / self.total_weight

    @property
    def effective_n(self) -> float:
        s = self.total_weight
        return float(s * s / np.dot(self.weights, self.weights))

    @property
    def positive(self) -> WeightedData:
        mask = self.weights > 0
        return WeightedData(self.values[mask], self.weights[mask])

    def normalized(self) -> WeightedData:
        return WeightedData(self.values, self.normalized_weights)

    def reweight(self, weights) -> WeightedData:
        return WeightedData(self.values, weights)

    def subset(self, index) -> WeightedData:
        values = self.values[index]
        weights = self.weights[index]
        if np.asarray(values).ndim == self.values.ndim - 1:
            values = np.expand_dims(values, axis=0)
            weights = np.atleast_1d(weights)
        return WeightedData(values, weights)

    def mean(self):
        return np.average(self.values, axis=0, weights=self.weights)

    def variance(self, *, ddof: float = 0.0):
        ddof = _validated_ddof(ddof)
        w = self.normalized_weights
        mu = self.mean()
        centered = self.values - mu
        var = np.sum(
            w.reshape((-1,) + (1,) * (self.values.ndim - 1)) * centered**2, axis=0
        )
        if ddof:
            n_eff = self.effective_n
            if n_eff <= ddof:
                return np.full_like(np.asarray(var, dtype=float), np.nan)
            var = var * n_eff / (n_eff - ddof)
        return var

    def standard_deviation(self, *, ddof: float = 0.0):
        return np.sqrt(self.variance(ddof=ddof))

    def covariance(self, *, ddof: float = 0.0) -> np.ndarray:
        ddof = _validated_ddof(ddof)
        x = np.asarray(self.values, dtype=float)
        if x.ndim == 1:
            return np.asarray(self.variance(ddof=ddof))
        if x.ndim != 2:
            raise ValueError("covariance requires scalar or vector observations")
        w = self.normalized_weights
        centered = x - self.mean()
        cov = (centered * w[:, None]).T @ centered
        if ddof:
            n_eff = self.effective_n
            if n_eff <= ddof:
                return np.full((x.shape[1], x.shape[1]), np.nan)
            cov *= n_eff / (n_eff - ddof)
        return cov

    def correlation(self) -> np.ndarray:
        """Weighted correlation matrix for vector observations."""
        cov = np.asarray(self.covariance(), dtype=float)
        if cov.ndim == 0:
            return np.asarray(1.0 if cov > 0 else np.nan)
        scales = np.sqrt(np.diag(cov))
        denom = np.outer(scales, scales)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / denom
        corr[denom == 0] = np.nan
        np.fill_diagonal(corr, np.where(scales > 0, 1.0, np.nan))
        return corr

    def quantile(self, q):
        x = self.values
        if x.ndim == 1:
            return _weighted_quantile_1d(x, self.weights, q)
        if x.ndim != 2:
            raise ValueError("quantile requires scalar or vector observations")
        columns = [
            _weighted_quantile_1d(x[:, j], self.weights, q) for j in range(x.shape[1])
        ]
        return np.asarray(columns).T

    def sample(self, size=None, rng=None):
        g = np.random.default_rng(rng)
        indices = g.choice(len(self), size=size, p=self.normalized_weights)
        return self.values[indices]

    def histogram(self, bins="auto", *, range=None) -> BinnedData:
        if self.values.ndim != 1:
            raise ValueError("histogram requires one-dimensional observations")
        return bin_data(self, bins=bins, range=range)

    def kde(self, bandwidth="scott"):
        if self.values.ndim != 1:
            raise ValueError("kde requires one-dimensional observations")
        from .distributions.nonparametric import KernelDensityDistribution

        return KernelDensityDistribution(
            self.values, bandwidth=bandwidth, weights=self.weights
        )


@dataclass(frozen=True, slots=True)
class BinnedData:
    """One-dimensional histogram/bin summary."""

    edges: np.ndarray
    counts: np.ndarray
    total_weight: float

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges, dtype=float).copy()
        counts = np.asarray(self.counts, dtype=float).copy()
        if edges.ndim != 1 or counts.ndim != 1 or edges.size != counts.size + 1:
            raise ValueError("edges must have exactly one more entry than counts")
        if not np.all(np.isfinite(edges)) or not np.all(np.diff(edges) > 0):
            raise ValueError("edges must be finite and strictly increasing")
        if not np.all(np.isfinite(counts)) or np.any(counts < 0):
            raise ValueError("counts must be finite and nonnegative")
        total = float(self.total_weight)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("total_weight must be finite and positive")
        if not np.isclose(counts.sum(), total):
            raise ValueError("counts must sum to total_weight")
        edges.setflags(write=False)
        counts.setflags(write=False)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "total_weight", total)

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    @property
    def midpoints(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def frequencies(self) -> np.ndarray:
        return self.counts / self.total_weight

    @property
    def density(self) -> np.ndarray:
        return self.frequencies / self.widths

    @property
    def cumulative(self) -> np.ndarray:
        return np.cumsum(self.frequencies)

    def to_distribution(self):
        from .distributions.nonparametric import HistogramDistribution

        return HistogramDistribution(
            bin_edges=self.edges, probabilities=self.frequencies
        )


def bin_data(data, bins="auto", *, weights=None, range=None) -> BinnedData:
    """Bin scalar observations with optional observation weights."""
    if isinstance(data, WeightedData):
        if weights is not None:
            raise ValueError("weights must not be supplied with WeightedData")
        if data.values.ndim != 1:
            raise ValueError("binning requires one-dimensional observations")
        x, w = data.values, data.weights
    else:
        x = _finite_array(data, ndim=1)
        w = None if weights is None else _validated_weights(weights, x.size)
    if w is not None and isinstance(bins, str):
        edges = np.histogram_bin_edges(x, bins=bins, range=range)
        counts, edges = np.histogram(x, bins=edges, range=range, weights=w)
    else:
        counts, edges = np.histogram(x, bins=bins, range=range, weights=w)
    total = float(np.sum(counts))
    if total <= 0:
        raise ValueError("no observations fall inside the requested histogram range")
    return BinnedData(edges.astype(float), counts.astype(float), total)
