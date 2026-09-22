"""Random sampling protocol for probstats distributions.

Sampling is NumPy-backed and accepts an explicit ``Generator`` or
seed.  Symbolic parameters are rejected: symbolic laws remain valid probability
objects, but random variate generation requires concrete numeric parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from ._validation import sample_shape
from .distributions.base import Distribution, SymbolicDistribution
from .distributions.basic import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    InverseWishart,
    Multinomial,
    MultivariateNormal,
    Wishart,
)
from .distributions.exponential_family import (
    Exponential,
    Gamma,
    InverseGamma,
    LogNormal,
    Normal,
    Poisson,
)
from .distributions.matrix import LKJ, LKJCholesky, MatrixNormal
from .distributions.nonparametric import (
    HistogramDistribution,
    KernelMixtureDistribution,
    MultivariateKernelDensityDistribution,
)
from .distributions.student_t import MultivariateStudentT, StudentT


class SamplingError(RuntimeError):
    """Raised when an exact distribution object cannot be sampled numerically."""


def as_rng(rng: Any = None) -> np.random.Generator:
    """Return a NumPy Generator from ``None``, a seed, or an existing Generator."""
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def _size_tuple(size) -> tuple[int, ...]:
    return sample_shape(size, none_as_empty=True)


def _float(value: Any, *, name: str) -> float:
    value = sp.sympify(value)
    if value.free_symbols:
        raise SamplingError(f"{name} must be numeric for sampling; got {value}")
    try:
        result = float(value.evalf())
    except (TypeError, ValueError, OverflowError) as exc:
        raise SamplingError(f"{name} must be a finite real numeric value") from exc
    if not np.isfinite(result):
        raise SamplingError(f"{name} must be finite for sampling")
    return result


def _int(value: Any, *, name: str) -> int:
    v = sp.sympify(value)
    if v.free_symbols or v.is_integer is not True:
        raise SamplingError(f"{name} must be a concrete integer for sampling; got {v}")
    return int(v)


def _array(values: Any, *, name: str) -> np.ndarray:
    matrix = sp.Matrix(values)
    if matrix.free_symbols:
        raise SamplingError(f"{name} must be numeric for sampling")
    try:
        arr = np.asarray(matrix.tolist(), dtype=float)
    except (TypeError, ValueError) as exc:
        raise SamplingError(f"{name} must be numeric for sampling") from exc
    if not np.all(np.isfinite(arr)):
        raise SamplingError(f"{name} must be finite for sampling")
    return arr


def _weights(values: Sequence[Any], *, name: str = "probabilities") -> np.ndarray:
    result = np.asarray([_float(v, name=name) for v in values], dtype=float)
    if np.any(result < 0) or not np.isclose(result.sum(), 1.0, rtol=1e-12, atol=1e-12):
        raise SamplingError(f"{name} must be nonnegative and sum to one")
    return result


def _wishart(
    rng: np.random.Generator, df: float, scale: np.ndarray, size: tuple[int, ...]
) -> np.ndarray:
    p = scale.shape[0]
    if df <= p - 1:
        raise SamplingError("Wishart df must exceed dimension - 1")
    chol = np.linalg.cholesky(scale)

    def one():
        a = np.zeros((p, p), dtype=float)
        for i in range(p):
            a[i, i] = np.sqrt(rng.chisquare(df - i))
            if i:
                a[i, :i] = rng.normal(size=i)
        b = chol @ a
        return b @ b.T

    if not size:
        return one()
    out = np.empty(size + (p, p), dtype=float)
    for idx in np.ndindex(size):
        out[idx] = one()
    return out


def sample(
    distribution: Distribution, size=None, rng=None, *, max_attempts: int = 100_000
):
    """Draw random variates from ``distribution``.

    ``size`` describes only the sample/batch dimensions.  Event dimensions are
    appended automatically for vector and matrix distributions.
    """
    if not isinstance(distribution, Distribution):
        raise TypeError("distribution must be a Distribution")
    r = as_rng(rng)
    shape = _size_tuple(size)

    if isinstance(distribution, HistogramDistribution):
        probs = np.asarray([float(p) for p in distribution.probabilities], dtype=float)
        edges = np.asarray([float(e) for e in distribution.bin_edges], dtype=float)
        indices = r.choice(probs.size, size=shape or None, p=probs)
        u = r.random(size=shape or None)
        return edges[indices] + u * (edges[indices + 1] - edges[indices])
    if isinstance(distribution, KernelMixtureDistribution):
        probs = np.asarray([float(w) for w in distribution.weights], dtype=float)
        loc = np.asarray([float(v) for v in distribution.locations], dtype=float)
        bw = np.asarray([float(v) for v in distribution.bandwidths], dtype=float)
        indices = r.choice(probs.size, size=shape or None, p=probs)
        return r.normal(loc[indices], bw[indices])
    if isinstance(distribution, MultivariateKernelDensityDistribution):
        probs = np.asarray([float(w) for w in distribution.weights], dtype=float)
        data = np.asarray(distribution.data, dtype=float)
        hmat = np.asarray(distribution.bandwidth_matrix.tolist(), dtype=float)
        if not shape:
            index = int(r.choice(data.shape[0], p=probs))
            return r.multivariate_normal(data[index], hmat)
        indices = r.choice(data.shape[0], size=shape, p=probs)
        noise = r.multivariate_normal(np.zeros(data.shape[1]), hmat, size=shape)
        return data[indices] + noise

    if isinstance(distribution, Bernoulli):
        return r.binomial(1, _float(distribution.p, name="p"), size=shape or None)
    if isinstance(distribution, Binomial):
        return r.binomial(
            _int(distribution.n, name="n"),
            _float(distribution.p, name="p"),
            size=shape or None,
        )
    if isinstance(distribution, Beta):
        return r.beta(
            _float(distribution.alpha, name="alpha"),
            _float(distribution.beta, name="beta"),
            size=shape or None,
        )
    if isinstance(distribution, Categorical):
        probs = _weights(distribution.probabilities)
        return r.choice(len(probs), size=shape or None, p=probs)
    if isinstance(distribution, Multinomial):
        probs = _weights(distribution.probabilities)
        return r.multinomial(_int(distribution.n, name="n"), probs, size=shape or None)
    if isinstance(distribution, Dirichlet):
        alpha = np.asarray(
            [_float(a, name="alpha") for a in distribution.concentration], dtype=float
        )
        return r.dirichlet(alpha, size=shape or None)
    if isinstance(distribution, Exponential):
        return r.exponential(
            1.0 / _float(distribution.rate, name="rate"), size=shape or None
        )
    if isinstance(distribution, Normal):
        return r.normal(
            _float(distribution.mean, name="mean"),
            _float(distribution.sigma, name="sigma"),
            size=shape or None,
        )
    if isinstance(distribution, Poisson):
        return r.poisson(_float(distribution.rate, name="rate"), size=shape or None)
    if isinstance(distribution, LogNormal):
        return r.lognormal(
            _float(distribution.mean, name="mean"),
            _float(distribution.sigma, name="sigma"),
            size=shape or None,
        )
    if isinstance(distribution, Gamma):
        return r.gamma(
            _float(distribution.shape, name="shape"),
            _float(distribution.scale, name="scale"),
            size=shape or None,
        )
    if isinstance(distribution, InverseGamma):
        g = r.gamma(
            _float(distribution.shape, name="shape"),
            1.0 / _float(distribution.scale, name="scale"),
            size=shape or None,
        )
        return 1.0 / g
    if isinstance(distribution, StudentT):
        return _float(distribution.location, name="location") + _float(
            distribution.scale, name="scale"
        ) * r.standard_t(_float(distribution.df, name="df"), size=shape or None)
    if isinstance(distribution, MultivariateNormal):
        mean = _array(distribution.mean, name="mean").reshape(-1)
        cov = _array(distribution.covariance, name="covariance")
        return r.multivariate_normal(mean, cov, size=shape or None)
    if isinstance(distribution, MultivariateStudentT):
        loc = _array(distribution.location, name="location").reshape(-1)
        scale = _array(distribution.scale, name="scale")
        df = _float(distribution.df, name="df")
        z = r.multivariate_normal(
            np.zeros(distribution.dimension), scale, size=shape or None
        )
        u = r.chisquare(df, size=shape or None) / df
        if shape:
            return loc + z / np.sqrt(u)[..., None]
        return loc + z / np.sqrt(u)
    if isinstance(distribution, Wishart):
        return _wishart(
            r,
            _float(distribution.df, name="df"),
            _array(distribution.scale, name="scale"),
            shape,
        )
    if isinstance(distribution, InverseWishart):
        scale = _array(distribution.scale, name="scale")
        w = _wishart(r, _float(distribution.df, name="df"), np.linalg.inv(scale), shape)
        return np.linalg.inv(w)
    if isinstance(distribution, MatrixNormal):
        mean = _array(distribution.mean, name="mean")
        row_cov = _array(distribution.row_covariance, name="row_covariance")
        col_cov = _array(distribution.column_covariance, name="column_covariance")
        lu = np.linalg.cholesky(row_cov)
        lv = np.linalg.cholesky(col_cov)

        def one_matrix_normal():
            z = r.normal(size=mean.shape)
            return mean + lu @ z @ lv.T

        if not shape:
            return one_matrix_normal()
        out = np.empty(shape + mean.shape, dtype=float)
        for idx in np.ndindex(shape):
            out[idx] = one_matrix_normal()
        return out
    if isinstance(distribution, (LKJ, LKJCholesky)):
        d = distribution.dimension
        eta = _float(distribution.eta, name="eta")

        def one_lkj_cholesky():
            l = np.zeros((d, d), dtype=float)
            l[0, 0] = 1.0
            for i in range(1, d):
                remaining = 1.0
                for j in range(i):
                    alpha = eta + 0.5 * (d - (j + 1) - 1)
                    z = 2.0 * r.beta(alpha, alpha) - 1.0
                    l[i, j] = z * np.sqrt(remaining)
                    remaining *= 1.0 - z * z
                l[i, i] = np.sqrt(max(remaining, 0.0))
            return l

        def one_lkj():
            l = one_lkj_cholesky()
            return l if isinstance(distribution, LKJCholesky) else l @ l.T

        if not shape:
            return one_lkj()
        out = np.empty(shape + (d, d), dtype=float)
        for idx in np.ndindex(shape):
            out[idx] = one_lkj()
        return out

    # Joint/coupled distributions are imported locally to avoid cycles.
    from .joint import CopulaDistribution, PushforwardDistribution

    if isinstance(distribution, PushforwardDistribution):
        raw = np.asarray(
            sample(
                distribution.source,
                size=shape or None,
                rng=r,
                max_attempts=max_attempts,
            ),
            dtype=float,
        )
        if raw.shape == () or raw.shape[-1] != len(distribution.variables):
            raise SamplingError(
                "pushforward source sampler must return its vector event on the final axis"
            )
        funcs = [
            sp.lambdify(distribution.variables, expr, modules="numpy")
            for expr in distribution.expressions
        ]
        coords = [raw[..., i] for i in range(len(distribution.variables))]
        outputs = [np.asarray(func(*coords), dtype=float) for func in funcs]
        if len(outputs) == 1:
            return outputs[0]
        return np.stack(outputs, axis=-1)

    if isinstance(distribution, CopulaDistribution):
        uniforms = np.asarray(
            distribution.copula.sample_uniform(size=shape or None, rng=r), dtype=float
        )
        if not shape:
            uniforms = uniforms.reshape((distribution.dimension,))
            return np.asarray(
                [
                    float(sp.N(m.quantile(sp.Float(u))))
                    for m, u in zip(distribution.marginals, uniforms)
                ]
            )
        flat = uniforms.reshape((-1, distribution.dimension))
        out = np.empty_like(flat, dtype=float)
        for row_i, row in enumerate(flat):
            for j, (marginal, u) in enumerate(zip(distribution.marginals, row)):
                q = marginal.quantile(sp.Float(u))
                if getattr(q, "free_symbols", set()):
                    raise SamplingError(
                        "copula marginal quantile must evaluate numerically"
                    )
                out[row_i, j] = float(sp.N(q))
        return out.reshape(shape + (distribution.dimension,))

    # Composition imports stay local to avoid an import cycle.
    from .composition import (
        ConditionalDistribution,
        IndependentDistribution,
        MarginalDistribution,
        MixtureDistribution,
        ProductDistribution,
        TruncatedDistribution,
    )
    from .transforms import MultivariateTransform, Transform, TransformedDistribution

    if isinstance(distribution, ProductDistribution):
        # Preserve heterogeneous event types: one result per component.
        return tuple(
            sample(d, size=shape or None, rng=r, max_attempts=max_attempts)
            for d in distribution.components
        )
    if isinstance(distribution, IndependentDistribution):
        # Draw count as an extra sample dimension, then move it after user batch axes.
        raw = sample(
            distribution.base,
            size=shape + (distribution.count,),
            rng=r,
            max_attempts=max_attempts,
        )
        return raw
    if isinstance(distribution, MixtureDistribution):
        weights = _weights(distribution.weights, name="mixture weights")
        choices = r.choice(len(distribution.components), size=shape or None, p=weights)
        if not shape:
            return sample(
                distribution.components[int(choices)], rng=r, max_attempts=max_attempts
            )
        # Sample by component masks while preserving event dimensions.
        first = sample(
            distribution.components[0], size=1, rng=r, max_attempts=max_attempts
        )
        event_shape = np.asarray(first).shape[1:]
        out = np.empty(shape + event_shape, dtype=np.asarray(first).dtype)
        for i, comp in enumerate(distribution.components):
            mask = choices == i
            count = int(np.count_nonzero(mask))
            if count:
                draws = np.asarray(
                    sample(comp, size=count, rng=r, max_attempts=max_attempts)
                )
                out[mask] = draws
        return out
    if isinstance(distribution, TruncatedDistribution):
        # Generic truncations use exact rejection sampling from the base law.
        need = int(np.prod(shape)) if shape else 1
        accepted = []
        attempts = 0
        while len(accepted) < need and attempts < max_attempts:
            batch = min(max(32, 2 * (need - len(accepted))), max_attempts - attempts)
            candidates = np.asarray(
                sample(distribution.base, size=batch, rng=r, max_attempts=max_attempts)
            ).reshape(-1)
            attempts += batch
            for value in candidates:
                contains = distribution.support.contains(sp.sympify(value))
                if contains is sp.true or contains is True:
                    accepted.append(value)
                    if len(accepted) == need:
                        break
        if len(accepted) < need:
            raise SamplingError("truncated rejection sampler exceeded max_attempts")
        arr = np.asarray(accepted)
        return arr.reshape(shape) if shape else arr[0]
    if isinstance(distribution, TransformedDistribution):
        base_draws = sample(
            distribution.base, size=shape or None, rng=r, max_attempts=max_attempts
        )
        t = distribution.transform
        if isinstance(t, Transform):
            fn = sp.lambdify(t.variable, t.forward, modules="numpy")
            return fn(base_draws)
        if isinstance(t, MultivariateTransform):
            fn = sp.lambdify(t.variables, list(t.forward), modules="numpy")
            arr = np.asarray(base_draws)
            if not shape:
                return np.asarray(fn(*arr), dtype=float).reshape(-1)
            flat = arr.reshape((-1, len(t.variables)))
            mapped = np.asarray(
                [np.asarray(fn(*row), dtype=float).reshape(-1) for row in flat]
            )
            return mapped.reshape(shape + (len(t.variables),))
    if isinstance(distribution, (ConditionalDistribution, MarginalDistribution)):
        return sample(
            distribution.distribution,
            size=shape or None,
            rng=r,
            max_attempts=max_attempts,
        )
    if isinstance(distribution, SymbolicDistribution):
        raise SamplingError("SymbolicDistribution has no generic numerical sampler")

    hook = getattr(distribution, "_sample", None)
    if hook is not None:
        return hook(size=shape or None, rng=r)
    raise SamplingError(f"No sampling implementation for {type(distribution).__name__}")


__all__ = ["SamplingError", "as_rng", "sample"]
