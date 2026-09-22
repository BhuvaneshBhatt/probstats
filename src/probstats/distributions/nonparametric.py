"""Nonparametric probability distributions and bandwidth selection.

The implementations in this module are NumPy/SymPy only.  Gaussian KDEs use
closed-form Gaussian convolution identities for bandwidth scoring so SciPy is
not required by the core package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import sympy as sp

from .._validation import integer
from ..data import WeightedData
from ..spaces import MeasureType, VectorEventSpace
from .base import Distribution, scalar_batch

_SQRT_2PI = float(np.sqrt(2.0 * np.pi))


def _data1d(data: Any) -> np.ndarray:
    if isinstance(data, WeightedData):
        data = data.values
    x = np.asarray(data, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("data must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must contain only finite values")
    return x.copy()


def _data2d(data: Any) -> np.ndarray:
    if isinstance(data, WeightedData):
        data = data.values
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("data must have shape (n_samples, n_dimensions)")
    if not np.all(np.isfinite(x)):
        raise ValueError("data must contain only finite values")
    return x.copy()


def _normalized_weights(weights: Any | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size != n:
        raise ValueError("weights must have one entry per observation")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or not np.any(w > 0):
        raise ValueError("weights must be finite, nonegative, and not all zero")
    return w / w.sum()


def _data_and_weights(data: Any, weights: Any | None):
    if isinstance(data, WeightedData):
        if weights is not None:
            raise ValueError("weights must not be supplied with WeightedData")
        return data.values, data.weights
    return data, weights


def _effective_n(weights: np.ndarray) -> float:
    return float(1.0 / np.dot(weights, weights))


def _weighted_std(x: np.ndarray, w: np.ndarray) -> float:
    m = float(np.dot(w, x))
    v = float(np.dot(w, (x - m) ** 2))
    return float(np.sqrt(max(v, 0.0)))


def _weighted_quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    c = np.cumsum(ws)
    return float(np.interp(q, c, xs))


def _scale_1d(x: np.ndarray, w: np.ndarray, *, robust: bool) -> float:
    std = _weighted_std(x, w)
    if not robust:
        return std
    iqr = _weighted_quantile(x, w, 0.75) - _weighted_quantile(x, w, 0.25)
    candidate = iqr / 1.349 if iqr > 0 else std
    return min(std, candidate) if std > 0 and candidate > 0 else max(std, candidate)


def scott_bandwidth(data: Any, weights: Any | None = None) -> float:
    """Scott's rule-of-thumb bandwidth for a univariate Gaussian KDE."""
    data, weights = _data_and_weights(data, weights)
    x = _data1d(data)
    w = _normalized_weights(weights, x.size)
    scale = _scale_1d(x, w, robust=False)
    if scale <= 0:
        raise ValueError("bandwidth is undefined for zero-variance data")
    return scale * _effective_n(w) ** (-1.0 / 5.0)


def silverman_bandwidth(data: Any, weights: Any | None = None) -> float:
    """Silverman's robust normal-reference bandwidth."""
    data, weights = _data_and_weights(data, weights)
    x = _data1d(data)
    w = _normalized_weights(weights, x.size)
    scale = _scale_1d(x, w, robust=True)
    if scale <= 0:
        raise ValueError("bandwidth is undefined for zero-variance data")
    return 0.9 * scale * _effective_n(w) ** (-1.0 / 5.0)


def plugin_bandwidth(data: Any, weights: Any | None = None) -> float:
    """One-stage Gaussian plug-in bandwidth using a pilot curvature estimate.

    The pilot is Silverman's bandwidth.  ``R(f'')`` is estimated exactly for
    the Gaussian pilot mixture using the fourth derivative of the convolution
    kernel, then inserted into the AMISE-optimal Gaussian-kernel bandwidth.
    """
    data, weights = _data_and_weights(data, weights)
    x = _data1d(data)
    w = _normalized_weights(weights, x.size)
    pilot = silverman_bandwidth(x, w)
    d = x[:, None] - x[None, :]
    s = np.sqrt(2.0) * pilot
    phi = np.exp(-0.5 * (d / s) ** 2) / (_SQRT_2PI * s)
    fourth = (d**4 / s**8 - 6.0 * d**2 / s**6 + 3.0 / s**4) * phi
    r_f2 = float(w @ fourth @ w)
    if not np.isfinite(r_f2) or r_f2 <= 0:
        return scott_bandwidth(x, w)
    r_kernel = 1.0 / (2.0 * np.sqrt(np.pi))
    h = (r_kernel / (_effective_n(w) * r_f2)) ** 0.2
    return float(h) if np.isfinite(h) and h > 0 else scott_bandwidth(x, w)


def _lscv_score(x: np.ndarray, w: np.ndarray, h: float) -> float:
    if h <= 0 or not np.isfinite(h):
        return float("inf")
    d = x[:, None] - x[None, :]
    first_kernel = np.exp(-0.25 * (d / h) ** 2) / (2.0 * np.sqrt(np.pi) * h)
    integrated_square = float(w @ first_kernel @ w)
    kernel = np.exp(-0.5 * (d / h) ** 2) / (_SQRT_2PI * h)
    np.fill_diagonal(kernel, 0.0)
    denom = 1.0 - w
    valid = denom > 1e-15
    loo = np.zeros_like(w)
    loo[valid] = (kernel[valid] @ w) / denom[valid]
    return integrated_square - 2.0 * float(np.dot(w[valid], loo[valid]))


def cv_bandwidth(
    data: Any,
    weights: Any | None = None,
    *,
    candidates: Any | None = None,
    grid_size: int = 31,
) -> float:
    """Least-squares cross-validation bandwidth for a Gaussian KDE."""
    data, weights = _data_and_weights(data, weights)
    x = _data1d(data)
    if x.size < 2:
        raise ValueError("cross-validation requires at least two observations")
    w = _normalized_weights(weights, x.size)
    if candidates is None:
        center = scott_bandwidth(x, w)
        grid_size = integer(grid_size, name="grid_size", minimum=1)
        hs = center * np.exp(np.linspace(-1.5, 1.5, grid_size))
    else:
        hs = np.asarray(candidates, dtype=float)
        if hs.ndim != 1 or hs.size == 0 or np.any(hs <= 0):
            raise ValueError("candidates must be a nonempty vector of positive values")
    scores = np.asarray([_lscv_score(x, w, float(h)) for h in hs])
    return float(hs[int(np.argmin(scores))])


def select_bandwidth(
    data: Any, method: str = "scott", weights: Any | None = None
) -> float:
    """Select a univariate KDE bandwidth by name."""
    key = method.lower().replace("-", "_")
    methods = {
        "scott": scott_bandwidth,
        "silverman": silverman_bandwidth,
        "plugin": plugin_bandwidth,
        "plug_in": plugin_bandwidth,
        "cv": cv_bandwidth,
        "lscv": cv_bandwidth,
    }
    try:
        return methods[key](data, weights)
    except KeyError as exc:
        raise ValueError(f"unknown bandwidth method: {method!r}") from exc


@dataclass(frozen=True, slots=True)
class HistogramDistribution(Distribution):
    """Piecewise-uniform histogram probability distribution."""

    bin_edges: tuple[sp.Expr, ...]
    probabilities: tuple[sp.Expr, ...]

    def __init__(
        self,
        data=None,
        bins="auto",
        *,
        weights=None,
        bin_edges=None,
        probabilities=None,
    ):
        if isinstance(data, WeightedData):
            data, weights = _data_and_weights(data, weights)
        if bin_edges is not None:
            edges = np.asarray(bin_edges, dtype=float)
            if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
                raise ValueError("bin_edges must be strictly increasing")
            if probabilities is None:
                if data is None:
                    raise ValueError("probabilities or data is required with bin_edges")
                counts, _ = np.histogram(_data1d(data), bins=edges, weights=weights)
                probs = counts.astype(float)
            else:
                probs = np.asarray(probabilities, dtype=float)
        else:
            if data is None:
                raise ValueError("data is required when bin_edges is not supplied")
            x = _data1d(data)
            if weights is not None and isinstance(bins, str):
                edges = np.histogram_bin_edges(x, bins=bins)
                counts, edges = np.histogram(x, bins=edges, weights=weights)
            else:
                counts, edges = np.histogram(x, bins=bins, weights=weights)
            probs = counts.astype(float)
        if probs.ndim != 1 or probs.size != edges.size - 1:
            raise ValueError("probabilities must have one value per bin")
        if np.any(probs < 0) or not np.any(probs > 0):
            raise ValueError("probabilities must be nonnegative and not all zero")
        probs = probs / probs.sum()
        object.__setattr__(self, "bin_edges", tuple(sp.Float(v) for v in edges))
        object.__setattr__(self, "probabilities", tuple(sp.Float(v) for v in probs))

    @property
    def support(self):
        return sp.Interval(self.bin_edges[0], self.bin_edges[-1])

    @property
    def parameters(self):
        return (*self.bin_edges, *self.probabilities)

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        pieces = []
        for i, p in enumerate(self.probabilities):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            cond = (
                sp.And(x >= lo, x <= hi)
                if i == len(self.probabilities) - 1
                else sp.And(x >= lo, x < hi)
            )
            pieces.append((p / (hi - lo), cond))
        return sp.Piecewise(*pieces, (0, True))

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, value):
        x = sp.sympify(value)
        cumulative = sp.S.Zero
        pieces = [(sp.S.Zero, x < self.bin_edges[0])]
        for i, p in enumerate(self.probabilities):
            lo, hi = self.bin_edges[i], self.bin_edges[i + 1]
            pieces.append((sp.simplify(cumulative + p * (x - lo) / (hi - lo)), x < hi))
            cumulative += p
        pieces.append((sp.S.One, True))
        return sp.Piecewise(*pieces)

    def _mean(self):
        mids = [
            (self.bin_edges[i] + self.bin_edges[i + 1]) / 2
            for i in range(len(self.probabilities))
        ]
        return sp.simplify(sum(p * m for p, m in zip(self.probabilities, mids)))

    def _raw_moment(self, order: int):
        if order < 0:
            raise ValueError("moment order must be nonnegative")
        total = sp.S.Zero
        for i, p in enumerate(self.probabilities):
            a, b = self.bin_edges[i], self.bin_edges[i + 1]
            total += p * (b ** (order + 1) - a ** (order + 1)) / ((order + 1) * (b - a))
        return sp.simplify(total)

    def _variance(self):
        mean = self._mean()
        return sp.simplify(self._raw_moment(2) - mean**2)


@dataclass(frozen=True, slots=True)
class KernelMixtureDistribution(Distribution):
    """Finite univariate Gaussian-kernel mixture."""

    locations: tuple[sp.Expr, ...]
    bandwidths: tuple[sp.Expr, ...]
    weights: tuple[sp.Expr, ...]

    def __init__(self, locations, bandwidths, weights=None):
        loc = _data1d(locations)
        bw = np.asarray(bandwidths, dtype=float)
        if bw.ndim == 0:
            bw = np.full(loc.size, float(bw))
        if (
            bw.ndim != 1
            or bw.size != loc.size
            or np.any(bw <= 0)
            or not np.all(np.isfinite(bw))
        ):
            raise ValueError("bandwidths must be positive with one value per location")
        w = _normalized_weights(weights, loc.size)
        object.__setattr__(self, "locations", tuple(sp.Float(v) for v in loc))
        object.__setattr__(self, "bandwidths", tuple(sp.Float(v) for v in bw))
        object.__setattr__(self, "weights", tuple(sp.Float(v) for v in w))

    @property
    def support(self):
        return sp.S.Reals

    @property
    def parameters(self):
        return (*self.locations, *self.bandwidths, *self.weights)

    @scalar_batch
    def pdf(self, value):
        x = sp.sympify(value)
        root = sp.sqrt(2 * sp.pi)
        return sp.simplify(
            sum(
                w * sp.exp(-((x - m) ** 2) / (2 * h**2)) / (root * h)
                for m, h, w in zip(self.locations, self.bandwidths, self.weights)
            )
        )

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _cdf(self, value):
        x = sp.sympify(value)
        return sp.simplify(
            sum(
                w * (1 + sp.erf((x - m) / (sp.sqrt(2) * h))) / 2
                for m, h, w in zip(self.locations, self.bandwidths, self.weights)
            )
        )

    def _mean(self):
        return sp.simplify(sum(w * m for m, w in zip(self.locations, self.weights)))

    def _raw_moment(self, order: int):
        if order < 0:
            raise ValueError("moment order must be nonnegative")
        total = sp.S.Zero
        for m, h, w in zip(self.locations, self.bandwidths, self.weights):
            component = sp.S.Zero
            for j in range(order // 2 + 1):
                k = 2 * j
                component += (
                    sp.binomial(order, k)
                    * m ** (order - k)
                    * h**k
                    * sp.factorial2(k - 1)
                )
            total += w * component
        return sp.simplify(total)

    def _variance(self):
        mean = self._mean()
        return sp.simplify(self._raw_moment(2) - mean**2)


@dataclass(frozen=True, slots=True, init=False)
class KernelDensityDistribution(KernelMixtureDistribution):
    """Weighted univariate Gaussian kernel density estimate."""

    bandwidth_method: str

    def __init__(self, data, bandwidth="scott", *, weights=None):
        data, weights = _data_and_weights(data, weights)
        x = _data1d(data)
        if isinstance(bandwidth, str):
            h = select_bandwidth(x, bandwidth, weights)
            method = bandwidth
        else:
            h = float(bandwidth)
            if not np.isfinite(h) or h <= 0:
                raise ValueError("bandwidth must be positive")
            method = "fixed"
        KernelMixtureDistribution.__init__(self, x, h, weights)
        object.__setattr__(self, "bandwidth_method", method)

    @property
    def bandwidth(self):
        return self.bandwidths[0]


@dataclass(frozen=True, slots=True)
class MultivariateKernelDensityDistribution(Distribution):
    """Multivariate Gaussian KDE with a full positive-definite bandwidth matrix."""

    data: tuple[tuple[sp.Expr, ...], ...]
    bandwidth_matrix: sp.ImmutableDenseMatrix
    weights: tuple[sp.Expr, ...]

    def __init__(self, data, bandwidth="scott", *, weights=None):
        data, weights = _data_and_weights(data, weights)
        x = _data2d(data)
        w = _normalized_weights(weights, x.shape[0])
        d = x.shape[1]
        if isinstance(bandwidth, str):
            key = bandwidth.lower()
            if key not in {"scott", "silverman"}:
                raise ValueError(
                    "multivariate bandwidth method must be 'scott' or 'silverman'"
                )
            neff = _effective_n(w)
            factor = (
                neff ** (-1.0 / (d + 4.0))
                if key == "scott"
                else (neff * (d + 2.0) / 4.0) ** (-1.0 / (d + 4.0))
            )
            mean = np.average(x, axis=0, weights=w)
            centered = x - mean
            cov = (centered * w[:, None]).T @ centered
            # Effective Bessel-style correction; regularization handles collinear samples.
            correction = 1.0 - float(np.dot(w, w))
            if correction > 0:
                cov /= correction
            hmat = factor**2 * cov
            scale = max(float(np.trace(hmat)) / d, 1.0)
            hmat = hmat + np.eye(d) * np.finfo(float).eps * scale * 100.0
        else:
            hmat = np.asarray(bandwidth, dtype=float)
            if hmat.shape != (d, d):
                raise ValueError(
                    "bandwidth matrix must have shape (dimension, dimension)"
                )
        if not np.all(np.isfinite(hmat)) or not np.allclose(hmat, hmat.T):
            raise ValueError("bandwidth matrix must be finite and symmetric")
        if np.linalg.eigvalsh(hmat).min() <= 0:
            raise ValueError("bandwidth matrix must be positive definite")
        object.__setattr__(
            self, "data", tuple(tuple(sp.Float(v) for v in row) for row in x)
        )
        object.__setattr__(self, "bandwidth_matrix", sp.ImmutableMatrix(hmat))
        object.__setattr__(self, "weights", tuple(sp.Float(v) for v in w))

    @property
    def dimension(self):
        return len(self.data[0])

    @property
    def support(self):
        return sp.ProductSet(*([sp.S.Reals] * self.dimension))

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def event_space(self):
        return VectorEventSpace(self.dimension)

    @property
    def parameters(self):
        return (
            tuple(v for row in self.data for v in row)
            + tuple(self.bandwidth_matrix)
            + self.weights
        )

    @scalar_batch
    def pdf(self, value):
        v = sp.ImmutableMatrix(value)
        if v.cols != 1:
            if v.rows == 1:
                v = v.T
            else:
                raise ValueError("value must be a vector")
        if v.rows != self.dimension:
            raise ValueError("value dimension does not match KDE dimension")
        h = self.bandwidth_matrix
        inv = h.inv()
        norm = sp.sqrt((2 * sp.pi) ** self.dimension * h.det())
        total = sp.S.Zero
        for row, weight in zip(self.data, self.weights):
            delta = v - sp.ImmutableMatrix(row)
            total += weight * sp.exp(-(delta.T * inv * delta)[0] / 2) / norm
        return sp.simplify(total)

    @scalar_batch
    def logpdf(self, value):
        return sp.log(self.pdf(value))

    def _mean(self):
        weights = sp.ImmutableMatrix(self.weights)
        points = sp.ImmutableMatrix(self.data)
        return sp.ImmutableMatrix(points.T * weights)

    def _variance(self):
        mean = self._mean()
        cov = self.bandwidth_matrix
        for row, weight in zip(self.data, self.weights):
            delta = sp.ImmutableMatrix(row) - mean
            cov += weight * (delta * delta.T)
        return sp.simplify(cov)


__all__ = [
    "HistogramDistribution",
    "KernelDensityDistribution",
    "KernelMixtureDistribution",
    "MultivariateKernelDensityDistribution",
    "cv_bandwidth",
    "plugin_bandwidth",
    "scott_bandwidth",
    "select_bandwidth",
    "silverman_bandwidth",
]
