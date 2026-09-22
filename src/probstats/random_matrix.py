"""Random-matrix ensembles and spectral statistics.

This module separates finite matrix ensembles and spectral-limit
laws from matrix-valued probability distributions such as Wishart or LKJ.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from math import pi, sqrt
from typing import Protocol

import numpy as np
import sympy as sp

from ._validation import integer, sample_shape
from .distributions.exponential_family import Gamma, Normal

Array = np.ndarray


def _rng(rng=None) -> np.random.Generator:
    return rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)


def _validate_positive_integer(value, *, name: str) -> int:
    return integer(value, name=name, minimum=1)


def _validate_dimension(n: int) -> int:
    return _validate_positive_integer(n, name="dimension")


def _validate_beta(beta: int) -> int:
    if isinstance(beta, (bool, np.bool_)) or not isinstance(beta, (int, np.integer)):
        raise TypeError("beta must be an integer")
    beta = int(beta)
    if beta not in (1, 2, 4):
        raise ValueError("beta must be one of 1 (GOE), 2 (GUE), or 4 (GSE)")
    return beta


def _sample_shape(size) -> tuple[int, ...] | None:
    return sample_shape(size, none_as_empty=False)


def _hermitian_matrix(matrix) -> Array:
    value = np.asarray(matrix)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("matrix must be square")
    if not np.all(np.isfinite(value)):
        raise ValueError("matrix must contain only finite values")
    if not np.allclose(value, value.conj().T):
        raise ValueError("matrix must be Hermitian")
    return value


@dataclass(frozen=True, slots=True)
class GaussianEnsemble:
    """Gaussian beta ensemble with Wigner scaling.

    ``beta=1,2,4`` correspond to GOE, GUE, and GSE eigenvalue laws.  Matrix
    samples for beta 1 and 2 use the invariant dense ensembles.  For beta 4,
    ``sample_matrix`` returns the real symmetric Dumitriu--Edelman tridiagonal
    spectral representative, whose eigenvalues have the GSE joint law.
    """

    dimension: int
    beta: int = 1

    def __post_init__(self):
        object.__setattr__(self, "dimension", _validate_dimension(self.dimension))
        object.__setattr__(self, "beta", _validate_beta(self.beta))

    @property
    def name(self) -> str:
        return {1: "GOE", 2: "GUE", 4: "GSE"}[self.beta]

    def sample_matrix(self, rng=None, *, size=None) -> Array:
        rng = _rng(rng)
        sample_shape = _sample_shape(size) or ()
        n = self.dimension
        if self.beta == 1:
            x = rng.normal(size=sample_shape + (n, n))
            return (x + np.swapaxes(x, -1, -2)) / np.sqrt(2.0 * n)
        if self.beta == 2:
            shape = sample_shape + (n, n)
            x = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
            return (x + np.swapaxes(x.conj(), -1, -2)) / np.sqrt(2.0 * n)
        return self._sample_tridiagonal(rng, sample_shape)

    def _sample_tridiagonal(self, rng: np.random.Generator, sample_shape=()) -> Array:
        n, beta = self.dimension, self.beta
        diagonal = rng.normal(scale=np.sqrt(2.0), size=sample_shape + (n,)) / np.sqrt(
            beta * n
        )
        dfs = beta * np.arange(n - 1, 0, -1)
        off = np.sqrt(rng.chisquare(dfs, size=sample_shape + (n - 1,))) / np.sqrt(
            beta * n
        )
        result = np.zeros(sample_shape + (n, n), dtype=float)
        indices = np.arange(n)
        result[..., indices, indices] = diagonal
        off_indices = np.arange(n - 1)
        result[..., off_indices, off_indices + 1] = off
        result[..., off_indices + 1, off_indices] = off
        return result

    def sample_eigenvalues(self, size=None, rng=None) -> Array:
        shape = _sample_shape(size) or ()
        matrices = self._sample_tridiagonal(_rng(rng), shape)
        return np.linalg.eigvalsh(matrices)

    def largest_eigenvalue(self, size=None, rng=None) -> Array | float:
        values = self.sample_eigenvalues(size=size, rng=rng)
        return np.max(values, axis=-1)

    def trace_distribution(self) -> Normal:
        # Under Wigner scaling, Var(trace H) = 2/beta.
        return Normal(0, sp.sqrt(sp.Rational(2, self.beta)))


class GaussianOrthogonalEnsemble(GaussianEnsemble):
    def __init__(self, dimension: int):
        super().__init__(dimension, 1)


class GaussianUnitaryEnsemble(GaussianEnsemble):
    def __init__(self, dimension: int):
        super().__init__(dimension, 2)


class GaussianSymplecticEnsemble(GaussianEnsemble):
    def __init__(self, dimension: int):
        super().__init__(dimension, 4)


GOE = GaussianOrthogonalEnsemble
GUE = GaussianUnitaryEnsemble
GSE = GaussianSymplecticEnsemble


@dataclass(frozen=True, slots=True)
class WishartEnsemble:
    """Real/complex Wishart ensemble ``X* X`` or normalized ``X*X / df``."""

    dimension: int
    df: int
    beta: int = 1
    scale: Array | None = None
    normalized: bool = True

    def __post_init__(self):
        p = _validate_dimension(self.dimension)
        df = _validate_positive_integer(self.df, name="df")
        beta = _validate_beta(self.beta)
        if df < p:
            raise ValueError("df must be an integer >= dimension")
        if beta not in (1, 2):
            raise ValueError(
                "dense Wishart matrix sampling currently supports beta=1 or 2"
            )
        object.__setattr__(self, "dimension", p)
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "beta", beta)
        if self.scale is not None:
            scale = np.asarray(self.scale, dtype=complex if beta == 2 else float)
            if scale.shape != (p, p):
                raise ValueError("scale must be dimension x dimension")
            if not np.all(np.isfinite(scale)):
                raise ValueError("scale must contain only finite values")
            if not np.allclose(scale, scale.conj().T):
                raise ValueError("scale must be Hermitian")
            if np.min(np.linalg.eigvalsh(scale)) <= 0:
                raise ValueError("scale must be positive definite")
            scale = scale.copy()
            scale.setflags(write=False)
            object.__setattr__(self, "scale", scale)

    def _data_matrix(self, rng: np.random.Generator, sample_shape=()) -> Array:
        shape = sample_shape + (self.df, self.dimension)
        if self.beta == 1:
            x = rng.normal(size=shape)
        else:
            x = (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2)
        if self.scale is not None:
            chol = np.linalg.cholesky(self.scale)
            x = x @ chol.conj().T
        return x

    def sample_matrix(self, rng=None, *, size=None) -> Array:
        sample_shape = _sample_shape(size) or ()
        x = self._data_matrix(_rng(rng), sample_shape)
        matrix = np.swapaxes(x.conj(), -1, -2) @ x
        return matrix / self.df if self.normalized else matrix

    def sample_eigenvalues(self, size=None, rng=None) -> Array:
        shape = _sample_shape(size)
        if shape is None:
            return np.linalg.eigvalsh(self.sample_matrix(rng))
        if self.dimension <= 16:
            return np.linalg.eigvalsh(self.sample_matrix(rng, size=shape))
        generator = _rng(rng)
        values = np.empty(shape + (self.dimension,), dtype=float)
        for index in np.ndindex(shape):
            values[index] = np.linalg.eigvalsh(self.sample_matrix(generator))
        return values

    def largest_eigenvalue(self, size=None, rng=None):
        return np.max(self.sample_eigenvalues(size=size, rng=rng), axis=-1)

    def trace_distribution(self):
        if self.beta != 1:
            raise NotImplementedError(
                "closed-form trace law is currently exposed for real Wishart"
            )
        if self.scale is None or np.allclose(
            self.scale, np.eye(self.dimension) * self.scale[0, 0]
        ):
            scalar = 1.0 if self.scale is None else float(np.real(self.scale[0, 0]))
            factor = scalar / self.df if self.normalized else scalar
            return Gamma(sp.Rational(self.df * self.dimension, 2), 2 * factor)
        raise NotImplementedError(
            "trace is gamma only for a scalar multiple of identity scale"
        )

    def determinant_law(self) -> WishartDeterminantLaw:
        if self.beta != 1:
            raise NotImplementedError(
                "Bartlett determinant law currently implemented for real Wishart"
            )
        determinant_scale = (
            1.0 if self.scale is None else float(np.linalg.det(self.scale).real)
        )
        if self.normalized:
            determinant_scale /= self.df**self.dimension
        return WishartDeterminantLaw(self.dimension, self.df, determinant_scale)


@dataclass(frozen=True, slots=True)
class WishartDeterminantLaw:
    """Exact Bartlett-product representation of a real Wishart determinant."""

    dimension: int
    df: int
    scale_factor: float = 1.0

    def __post_init__(self) -> None:
        dimension = _validate_dimension(self.dimension)
        df = _validate_positive_integer(self.df, name="df")
        scale_factor = float(self.scale_factor)
        if df < dimension:
            raise ValueError("df must be an integer >= dimension")
        if not np.isfinite(scale_factor) or scale_factor <= 0:
            raise ValueError("scale_factor must be finite and positive")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "df", df)
        object.__setattr__(self, "scale_factor", scale_factor)

    @property
    def chi_square_dfs(self) -> tuple[int, ...]:
        return tuple(self.df - i for i in range(self.dimension))

    def sample(self, size=None, rng=None):
        rng = _rng(rng)
        sample_shape = _sample_shape(size)
        shape = () if sample_shape is None else sample_shape
        result = np.full(shape, self.scale_factor, dtype=float)
        for df in self.chi_square_dfs:
            result *= rng.chisquare(df, size=shape)
        return float(result) if size is None else result

    @property
    def mean(self) -> float:
        return self.scale_factor * float(np.prod(self.chi_square_dfs))

    @property
    def mean_log(self) -> float:
        from math import log

        return log(self.scale_factor) + sum(
            float(sp.digamma(sp.Rational(df, 2))) + log(2) for df in self.chi_square_dfs
        )

    @property
    def variance_log(self) -> float:
        return float(
            sum(sp.polygamma(1, sp.Rational(df, 2)) for df in self.chi_square_dfs)
        )


@dataclass(frozen=True, slots=True)
class WignerSemicircle:
    """Wigner semicircle law with center ``center`` and radius ``radius``."""

    radius: float = 2.0
    center: float = 0.0

    def __post_init__(self):
        radius = float(self.radius)
        center = float(self.center)
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("radius must be finite and positive")
        if not np.isfinite(center):
            raise ValueError("center must be finite")
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "center", center)

    @property
    def support(self) -> tuple[float, float]:
        return self.center - self.radius, self.center + self.radius

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        z = x - self.center
        value = (
            2.0
            * np.sqrt(np.maximum(self.radius**2 - z**2, 0.0))
            / (pi * self.radius**2)
        )
        value = np.where(np.abs(z) <= self.radius, value, 0.0)
        return float(value) if value.ndim == 0 else value

    def cdf(self, x):
        x = np.asarray(x, dtype=float)
        z = np.clip((x - self.center) / self.radius, -1.0, 1.0)
        value = 0.5 + (np.arcsin(z) + z * np.sqrt(np.maximum(1 - z**2, 0.0))) / pi
        value = np.where(x <= self.center - self.radius, 0.0, value)
        value = np.where(x >= self.center + self.radius, 1.0, value)
        return float(value) if value.ndim == 0 else value

    @property
    def mean(self) -> float:
        return self.center

    @property
    def variance(self) -> float:
        return self.radius**2 / 4

    def sample(self, size=None, rng=None):
        rng = _rng(rng)
        sample_shape = _sample_shape(size)
        shape = () if sample_shape is None else sample_shape
        r = self.radius * np.sqrt(rng.random(shape))
        theta = 2 * pi * rng.random(shape)
        value = self.center + r * np.cos(theta)
        return float(value) if size is None else value


@lru_cache(maxsize=16)
def _legendre_rule(points: int):
    """Cache Gauss-Legendre rules shared by repeated spectral CDF calls."""
    nodes, weights = np.polynomial.legendre.leggauss(points)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


@dataclass(frozen=True, slots=True)
class MarchenkoPastur:
    """Marchenko--Pastur law for aspect ratio ``p/n`` and covariance scale."""

    ratio: float
    scale: float = 1.0

    def __post_init__(self):
        ratio = float(self.ratio)
        scale = float(self.scale)
        if not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("ratio must be finite and positive")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "ratio", ratio)
        object.__setattr__(self, "scale", scale)

    @property
    def lower_edge(self) -> float:
        return self.scale * (1 - sqrt(self.ratio)) ** 2

    @property
    def upper_edge(self) -> float:
        return self.scale * (1 + sqrt(self.ratio)) ** 2

    @property
    def atom_at_zero(self) -> float:
        return max(1.0 - 1.0 / self.ratio, 0.0)

    @property
    def continuous_mass(self) -> float:
        return 1.0 - self.atom_at_zero

    @property
    def support(self) -> tuple[float, float]:
        return self.lower_edge, self.upper_edge

    def pdf(self, x):
        x = np.asarray(x, dtype=float)
        a, b = self.lower_edge, self.upper_edge
        core = np.sqrt(np.maximum((b - x) * (x - a), 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            value = core / (2 * pi * self.ratio * self.scale * x)
        value = np.where((x >= a) & (x <= b) & (x > 0), value, 0.0)
        return float(value) if value.ndim == 0 else value

    def cdf(self, x, *, quadrature_points: int = 256):
        """Evaluate the CDF using sorted soft-edge angular quadrature.

        The change of variables
        ``x = scale * (1 + ratio - 2*sqrt(ratio)*cos(theta))`` removes the
        square-root edge factors from the Marchenko--Pastur density. Array
        inputs are sorted once and only the intervals between successive
        requested points are integrated, instead of reintegrating from the
        lower edge for every value.
        """
        points = _validate_positive_integer(quadrature_points, name="quadrature_points")
        arr = np.asarray(x, dtype=float)
        flat = arr.ravel()
        result = np.empty(flat.size, dtype=float)
        a, b = self.lower_edge, self.upper_edge
        base = self.atom_at_zero
        result[flat < 0] = 0.0
        result[(flat >= 0) & (flat <= a)] = base
        result[flat >= b] = 1.0
        interior = (flat > a) & (flat < b)
        if np.any(interior):
            values = flat[interior]
            root = sqrt(self.ratio)
            cosine = (1 + self.ratio - values / self.scale) / (2 * root)
            theta = np.arccos(np.clip(cosine, -1.0, 1.0))
            unique_theta, inverse = np.unique(theta, return_inverse=True)
            bounds = np.concatenate(([0.0], unique_theta))
            lo = bounds[:-1]
            hi = bounds[1:]
            nodes, weights = _legendre_rule(points)
            increments = np.empty(unique_theta.size, dtype=float)
            chunk = max(1, 131072 // points)
            for first in range(0, unique_theta.size, chunk):
                last = min(first + chunk, unique_theta.size)
                half = (hi[first:last] - lo[first:last])[:, None] / 2
                mid = (hi[first:last] + lo[first:last])[:, None] / 2
                angle = mid + half * nodes[None, :]
                denominator = 1 + self.ratio - 2 * root * np.cos(angle)
                density = 2 * np.sin(angle) ** 2 / (pi * denominator)
                increments[first:last] = half[:, 0] * (density @ weights)
            cumulative = base + np.cumsum(increments)
            result[interior] = cumulative[inverse]
        result = np.clip(result.reshape(arr.shape), 0.0, 1.0)
        return float(result) if result.ndim == 0 else result

    @property
    def mean(self) -> float:
        return self.scale

    @property
    def variance(self) -> float:
        return self.ratio * self.scale**2


class TracyWidomBackend(Protocol):
    def pdf(self, x: Array | float, beta: int) -> Array | float: ...
    def cdf(self, x: Array | float, beta: int) -> Array | float: ...
    def ppf(self, p: Array | float, beta: int) -> Array | float: ...


_TRACY_WIDOM_BACKENDS: dict[str, TracyWidomBackend] = {}


def register_tracy_widom_backend(
    name: str, backend: TracyWidomBackend, *, replace: bool = False
) -> None:
    """Register a numerical Tracy--Widom backend by name."""
    if not isinstance(name, str):
        raise TypeError("backend name must be a string")
    name = name.strip()
    if not name:
        raise ValueError("backend name must be nonempty")
    for method in ("pdf", "cdf", "ppf"):
        if not callable(getattr(backend, method, None)):
            raise TypeError(f"Tracy-Widom backend must provide {method}()")
    if name in _TRACY_WIDOM_BACKENDS and not replace:
        raise ValueError(f"Tracy-Widom backend {name!r} is already registered")
    _TRACY_WIDOM_BACKENDS[name] = backend


def unregister_tracy_widom_backend(name: str) -> TracyWidomBackend:
    """Remove and return a registered Tracy--Widom backend."""
    if not isinstance(name, str):
        raise TypeError("backend name must be a string")
    name = name.strip()
    try:
        return _TRACY_WIDOM_BACKENDS.pop(name)
    except KeyError as exc:
        raise ValueError(f"unknown Tracy-Widom backend {name!r}") from exc


def available_tracy_widom_backends() -> tuple[str, ...]:
    return tuple(sorted(_TRACY_WIDOM_BACKENDS))


@dataclass(frozen=True, slots=True)
class TracyWidom:
    """Tracy--Widom law interface for beta=1,2,4.

    Numerical PDF/CDF/PPF evaluation is delegated to a registered backend.
    This keeps a stable public interface without silently shipping a low-quality
    approximation to the Painleve-II law.
    """

    beta: int = 2
    backend: str | TracyWidomBackend | None = None

    def __post_init__(self):
        _validate_beta(self.beta)

    def _backend(self) -> TracyWidomBackend:
        if self.backend is not None and not isinstance(self.backend, str):
            backend = self.backend
            for method in ("pdf", "cdf", "ppf"):
                if not callable(getattr(backend, method, None)):
                    raise TypeError(f"Tracy-Widom backend must provide {method}()")
            return backend
        name = self.backend
        if name is None:
            if len(_TRACY_WIDOM_BACKENDS) == 1:
                return next(iter(_TRACY_WIDOM_BACKENDS.values()))
            raise RuntimeError(
                "no Tracy-Widom numerical backend is selected; register one with "
                "register_tracy_widom_backend() or pass backend=..."
            )
        try:
            return _TRACY_WIDOM_BACKENDS[name]
        except KeyError as exc:
            raise ValueError(f"unknown Tracy-Widom backend {name!r}") from exc

    def pdf(self, x):
        return self._backend().pdf(x, self.beta)

    def cdf(self, x):
        return self._backend().cdf(x, self.beta)

    def ppf(self, p):
        return self._backend().ppf(p, self.beta)

    def sample(self, size=None, rng=None):
        rng = _rng(rng)
        return self.ppf(rng.random(size))


@dataclass(frozen=True, slots=True)
class LargestEigenvalueScaling:
    center: float
    scale: float
    beta: int

    def __post_init__(self) -> None:
        center = float(self.center)
        scale = float(self.scale)
        if not np.isfinite(center):
            raise ValueError("center must be finite")
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "beta", _validate_beta(self.beta))

    def standardize(self, value):
        return (np.asarray(value) - self.center) / self.scale

    def unstandardize(self, value):
        return self.center + self.scale * np.asarray(value)


def gaussian_largest_eigenvalue_scaling(
    dimension: int, beta: int = 2
) -> LargestEigenvalueScaling:
    """Soft-edge scaling for Wigner-normalized Gaussian beta ensembles."""
    n = _validate_dimension(dimension)
    _validate_beta(beta)
    return LargestEigenvalueScaling(2.0, n ** (-2.0 / 3.0), beta)


def wishart_largest_eigenvalue_scaling(
    dimension: int,
    df: int,
    beta: int = 1,
    *,
    normalized: bool = True,
) -> LargestEigenvalueScaling:
    """Asymptotic soft-edge scaling for an identity-covariance Wishart ensemble."""
    p = _validate_dimension(dimension)
    df = _validate_positive_integer(df, name="df")
    if df < p:
        raise ValueError("df must be an integer >= dimension")
    beta = _validate_beta(beta)
    c = p / df
    center = (1 + sqrt(c)) ** 2
    scale = df ** (-2 / 3) * (1 + sqrt(c)) ** (4 / 3) * c ** (-1 / 6)
    if not normalized:
        center *= df
        scale *= df
    return LargestEigenvalueScaling(center, scale, beta)


@dataclass(frozen=True, slots=True)
class LargestEigenvalueApproximation:
    """Tracy--Widom soft-edge approximation for a largest eigenvalue."""

    scaling: LargestEigenvalueScaling
    tracy_widom: TracyWidom

    def cdf(self, value):
        return self.tracy_widom.cdf(self.scaling.standardize(value))

    def pdf(self, value):
        return (
            self.tracy_widom.pdf(self.scaling.standardize(value)) / self.scaling.scale
        )

    def ppf(self, probability):
        return self.scaling.unstandardize(self.tracy_widom.ppf(probability))

    def sample(self, size=None, rng=None):
        return self.scaling.unstandardize(self.tracy_widom.sample(size=size, rng=rng))


def gaussian_largest_eigenvalue_approximation(
    dimension: int,
    beta: int = 2,
    *,
    backend: str | TracyWidomBackend | None = None,
) -> LargestEigenvalueApproximation:
    return LargestEigenvalueApproximation(
        gaussian_largest_eigenvalue_scaling(dimension, beta),
        TracyWidom(beta, backend=backend),
    )


def wishart_largest_eigenvalue_approximation(
    dimension: int,
    df: int,
    beta: int = 1,
    *,
    normalized: bool = True,
    backend: str | TracyWidomBackend | None = None,
) -> LargestEigenvalueApproximation:
    return LargestEigenvalueApproximation(
        wishart_largest_eigenvalue_scaling(dimension, df, beta, normalized=normalized),
        TracyWidom(beta, backend=backend),
    )


def eigenvalue_statistics(matrix: Array) -> dict[str, float]:
    values = np.linalg.eigvalsh(_hermitian_matrix(matrix))
    abs_values = np.abs(values)
    smallest = float(np.min(abs_values))
    return {
        "minimum": float(values[0]),
        "maximum": float(values[-1]),
        "trace": float(np.sum(values).real),
        "determinant": float(np.prod(values).real),
        "spectral_radius": float(np.max(abs_values)),
        "condition_number": float(
            np.inf if smallest == 0 else np.max(abs_values) / smallest
        ),
    }


def sample_spectral_statistic(
    ensemble: GaussianEnsemble | WishartEnsemble,
    statistic: str | Callable[[Array], float],
    *,
    size: int = 1_000,
    rng=None,
) -> Array:
    """Monte-Carlo distribution sample of a spectral matrix statistic."""
    size = _validate_positive_integer(size, name="size")
    if isinstance(statistic, str):
        allowed = {"largest_eigenvalue", "trace", "determinant", "condition_number"}
        if statistic not in allowed:
            raise ValueError(f"unknown spectral statistic {statistic!r}")

        def evaluate(matrix):
            if statistic == "largest_eigenvalue":
                return float(np.linalg.eigvalsh(matrix)[-1])
            if statistic == "trace":
                return float(np.trace(matrix).real)
            if statistic == "determinant":
                return float(np.linalg.det(matrix).real)
            return float(np.linalg.cond(matrix))
    else:
        evaluate = statistic
    rng = _rng(rng)
    if isinstance(statistic, str):
        matrices = ensemble.sample_matrix(rng, size=size)
        if statistic == "largest_eigenvalue":
            return np.linalg.eigvalsh(matrices)[..., -1]
        if statistic == "trace":
            return np.trace(matrices, axis1=-2, axis2=-1).real
        if statistic == "determinant":
            return np.linalg.det(matrices).real
        return np.linalg.cond(matrices)
    return np.asarray(
        [evaluate(ensemble.sample_matrix(rng)) for _ in range(size)], dtype=float
    )


__all__ = [
    "GOE",
    "GSE",
    "GUE",
    "GaussianEnsemble",
    "GaussianOrthogonalEnsemble",
    "GaussianSymplecticEnsemble",
    "GaussianUnitaryEnsemble",
    "LargestEigenvalueApproximation",
    "LargestEigenvalueScaling",
    "MarchenkoPastur",
    "TracyWidom",
    "TracyWidomBackend",
    "WignerSemicircle",
    "WishartDeterminantLaw",
    "WishartEnsemble",
    "available_tracy_widom_backends",
    "eigenvalue_statistics",
    "gaussian_largest_eigenvalue_approximation",
    "gaussian_largest_eigenvalue_scaling",
    "register_tracy_widom_backend",
    "sample_spectral_statistic",
    "unregister_tracy_widom_backend",
    "wishart_largest_eigenvalue_approximation",
    "wishart_largest_eigenvalue_scaling",
]
