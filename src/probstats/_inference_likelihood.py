"""Internal implementation for inference likelihood."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from ._special import chi2_sf
from .distributions import Distribution
from .spaces import (
    IntegerSpace,
    NaturalNumberSpace,
    NonnegativeRealSpace,
    PositiveRealSpace,
    RealSpace,
    UnitIntervalSpace,
)
from .testing import HypothesisTestResult


def _as1d(data, *, name="data"):
    x = np.asarray(data, dtype=float).reshape(-1)
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must contain finite observations")
    return x


@dataclass(frozen=True)
class Likelihood:
    factory: Callable[..., Distribution]
    data: np.ndarray
    parameter_names: tuple[str, ...]
    fixed: Mapping[str, Any] = field(default_factory=dict)

    def distribution(self, theta):
        vals = np.asarray(theta, dtype=float).reshape(-1)
        if vals.size != len(self.parameter_names):
            raise ValueError("theta has wrong dimension")
        kwargs = dict(self.fixed)
        kwargs.update(dict(zip(self.parameter_names, vals, strict=True)))
        return self.factory(**kwargs)

    def log_likelihood(self, theta):
        try:
            dist = self.distribution(theta)
            values = [float(dist.logpdf(v).evalf()) for v in self.data]
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return -math.inf
        total = float(np.sum(values))
        return total if math.isfinite(total) else -math.inf

    __call__ = log_likelihood


def likelihood(factory, data, parameter_names=None, *, fixed=None):
    """Construct a parametric likelihood or evaluate an iid sample likelihood.

    Passing a distribution class plus ``parameter_names`` returns a
    :class:`Likelihood` for numerical inference. Passing an already constructed
    distribution and omitting ``parameter_names`` returns the exact iid sample
    likelihood value.
    """
    if isinstance(factory, Distribution):
        if parameter_names is not None or fixed is not None:
            raise TypeError(
                "parameter_names and fixed are not used when evaluating a distribution"
            )
        from .functionals import likelihood_value

        return likelihood_value(factory, data)
    if parameter_names is None:
        raise TypeError("parameter_names are required for a parametric likelihood")
    return Likelihood(factory, _as1d(data), tuple(parameter_names), fixed or {})


@dataclass(frozen=True)
class OptimizationResult:
    parameters: dict[str, float]
    estimate: np.ndarray
    objective: float
    log_likelihood: float
    converged: bool
    iterations: int
    method: str
    message: str
    covariance: np.ndarray | None = None
    standard_errors: dict[str, float] | None = None
    information: np.ndarray | None = None
    log_prior: float = 0.0


def _clip(x, bounds):
    if bounds is None:
        return x
    y = x.copy()
    for i, bound in enumerate(bounds):
        if bound is None:
            continue
        lo, hi = bound
        if lo is not None:
            y[i] = max(y[i], lo)
        if hi is not None:
            y[i] = min(y[i], hi)
    return y


def _gradient(func, x):
    out = np.empty_like(x, dtype=float)
    f0 = func(x)
    for i in range(x.size):
        h = 1e-5 * max(1.0, abs(x[i]))
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fp, fm = func(xp), func(xm)
        if math.isfinite(fp) and math.isfinite(fm):
            out[i] = (fp - fm) / (2 * h)
        elif math.isfinite(fp) and math.isfinite(f0):
            out[i] = (fp - f0) / h
        elif math.isfinite(fm) and math.isfinite(f0):
            out[i] = (f0 - fm) / h
        else:
            out[i] = math.nan
    return out


def _hessian(func, x):
    n = x.size
    out = np.empty((n, n), dtype=float)
    fx = func(x)
    hs = np.array([1e-4 * max(1.0, abs(v)) for v in x])
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = hs[i]
        out[i, i] = (func(x + ei) - 2 * fx + func(x - ei)) / hs[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = hs[j]
            value = (
                func(x + ei + ej)
                - func(x + ei - ej)
                - func(x - ei + ej)
                + func(x - ei - ej)
            ) / (4 * hs[i] * hs[j])
            out[i, j] = out[j, i] = value
    return out


def _bfgs_minimize(func, initial, *, bounds=None, max_iter=500, tol=1e-8):
    x = _clip(np.asarray(initial, dtype=float).copy(), bounds)
    inv_hessian = np.eye(x.size)
    fx = func(x)
    if not math.isfinite(fx):
        raise ValueError("initial parameters have non-finite objective")
    for iteration in range(1, max_iter + 1):
        grad = _gradient(func, x)
        if np.linalg.norm(grad, np.inf) < tol:
            return x, fx, True, iteration, "gradient tolerance reached"
        direction = -inv_hessian @ grad
        if float(grad @ direction) >= 0:
            direction = -grad
            inv_hessian = np.eye(x.size)
        step = 1.0
        while step > 1e-10:
            candidate = _clip(x + step * direction, bounds)
            f_candidate = func(candidate)
            if math.isfinite(f_candidate) and f_candidate <= fx + 1e-4 * step * float(
                grad @ direction
            ):
                break
            step *= 0.5
        else:
            return x, fx, False, iteration, "line search failed"
        new_grad = _gradient(func, candidate)
        s = candidate - x
        y = new_grad - grad
        ys = float(y @ s)
        if ys > 1e-12:
            rho = 1 / ys
            eye = np.eye(x.size)
            inv_hessian = (eye - rho * np.outer(s, y)) @ inv_hessian @ (
                eye - rho * np.outer(y, s)
            ) + rho * np.outer(s, s)
        if np.linalg.norm(s) < tol * (1 + np.linalg.norm(x)):
            return candidate, f_candidate, True, iteration, "step tolerance reached"
        x, fx = candidate, f_candidate
    return x, fx, False, max_iter, "maximum iterations reached"


def observed_fisher_information(lik, theta):
    return -_hessian(lik.log_likelihood, np.asarray(theta, dtype=float))


def estimator_covariance(lik, theta):
    info = observed_fisher_information(lik, theta)
    try:
        return np.linalg.inv(info)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(info)


def expected_fisher_information(lik, theta, *, samples=2000, rng=None):
    """Monte Carlo expected Fisher information for the full sample size."""
    theta = np.asarray(theta, dtype=float)
    dist = lik.distribution(theta)
    draws = np.asarray(dist.sample(size=samples, rng=rng), dtype=float).reshape(-1)
    total = np.zeros((theta.size, theta.size), dtype=float)
    one = likelihood(lik.factory, [0.0], lik.parameter_names, fixed=lik.fixed)
    for value in draws:
        one_data = Likelihood(
            one.factory, np.asarray([value]), one.parameter_names, one.fixed
        )
        total += -_hessian(one_data.log_likelihood, theta)
    return total / samples * lik.data.size


def _optimization_result(
    lik,
    x,
    objective,
    converged,
    iterations,
    method,
    message,
    log_prior=0.0,
    posterior_hessian=None,
):
    ll = lik.log_likelihood(x)
    info = (
        observed_fisher_information(lik, x)
        if posterior_hessian is None
        else posterior_hessian
    )
    try:
        covariance = np.linalg.inv(info)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(info)
    se = np.sqrt(np.maximum(np.diag(covariance), 0))
    return OptimizationResult(
        dict(zip(lik.parameter_names, map(float, x), strict=True)),
        x,
        float(objective),
        float(ll),
        converged,
        iterations,
        method,
        message,
        covariance,
        dict(zip(lik.parameter_names, map(float, se), strict=True)),
        info,
        float(log_prior),
    )


@dataclass(frozen=True)
class ParameterTransformPlan:
    """Unconstrained coordinate map derived from a distribution ParameterSpace."""

    parameter_names: tuple[str, ...]
    spaces: tuple[Any, ...]
    relation_count: int

    def to_unconstrained(self, theta):
        values = np.asarray(theta, dtype=float).reshape(-1)
        if values.size != len(self.spaces):
            raise ValueError("theta has wrong dimension")
        return np.array(
            [
                _space_inverse(space, value)
                for space, value in zip(self.spaces, values, strict=True)
            ]
        )

    def from_unconstrained(self, unconstrained):
        values = np.asarray(unconstrained, dtype=float).reshape(-1)
        if values.size != len(self.spaces):
            raise ValueError("unconstrained vector has wrong dimension")
        return np.array(
            [
                _space_forward(space, value)
                for space, value in zip(self.spaces, values, strict=True)
            ]
        )


def _space_forward(space, value):
    if isinstance(space, RealSpace):
        return float(value)
    if isinstance(space, PositiveRealSpace):
        return float(math.exp(np.clip(value, -700, 700)))
    if isinstance(space, NonnegativeRealSpace):
        # Interior parameterization. Boundary optima are approached as u -> -inf.
        return float(math.exp(np.clip(value, -700, 700)))
    if isinstance(space, UnitIntervalSpace):
        if value >= 0:
            z = math.exp(-min(float(value), 700.0))
            return 1.0 / (1.0 + z)
        z = math.exp(max(float(value), -700.0))
        return z / (1.0 + z)
    if isinstance(space, (IntegerSpace, NaturalNumberSpace)):
        raise NotImplementedError(
            "integer-valued parameters cannot be optimized with continuous BFGS"
        )
    raise NotImplementedError(
        f"automatic unconstraining is not implemented for {type(space).__name__}"
    )


def _space_inverse(space, value):
    x = float(value)
    if isinstance(space, RealSpace):
        return x
    if isinstance(space, (PositiveRealSpace, NonnegativeRealSpace)):
        if x <= 0:
            raise ValueError(
                f"initial value {x} must be positive for automatic unconstraining"
            )
        return math.log(x)
    if isinstance(space, UnitIntervalSpace):
        if not 0 < x < 1:
            raise ValueError(
                "automatic unit-interval unconstraining requires an interior initial value"
            )
        return math.log(x / (1 - x))
    if isinstance(space, (IntegerSpace, NaturalNumberSpace)):
        raise NotImplementedError(
            "integer-valued parameters cannot be optimized with continuous BFGS"
        )
    raise NotImplementedError(
        f"automatic unconstraining is not implemented for {type(space).__name__}"
    )


def parameter_transform_plan(lik, initial):
    """Derive an optimization-coordinate plan from the distribution ParameterSpace."""
    dist = lik.distribution(initial)
    space = dist.parameter_space
    by_name = {spec.name: spec.space for spec in space.specs}
    try:
        spaces = tuple(by_name[name] for name in lik.parameter_names)
    except KeyError as exc:
        raise ValueError(
            f"likelihood parameter {exc.args[0]!r} is absent from distribution ParameterSpace"
        ) from exc
    return ParameterTransformPlan(lik.parameter_names, spaces, len(space.relations))


def _parameter_space_valid(lik, theta):
    try:
        dist = lik.distribution(theta)
        condition = dist.parameter_space_constraints
        simplified = sp.simplify(condition)
        return simplified is not sp.false and simplified is not False
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        return False


def _auto_unconstrained_minimize(
    lik, initial, objective_theta, *, max_iter=500, tol=1e-8
):
    plan = parameter_transform_plan(lik, initial)
    u0 = plan.to_unconstrained(initial)

    def objective_u(u):
        theta = plan.from_unconstrained(u)
        if not _parameter_space_valid(lik, theta):
            return math.inf
        return objective_theta(theta)

    u, value, conv, iterations, message = _bfgs_minimize(
        objective_u, u0, max_iter=max_iter, tol=tol
    )
    return plan.from_unconstrained(u), value, conv, iterations, message


def numerical_mle(
    lik, initial, *, bounds=None, constraints=None, max_iter=500, tol=1e-8
):
    def objective(z):
        return -lik.log_likelihood(z)

    use_auto = constraints == "auto" or (constraints is None and bounds is None)
    if use_auto:
        try:
            x, value, conv, iterations, message = _auto_unconstrained_minimize(
                lik, initial, objective, max_iter=max_iter, tol=tol
            )
            method = "BFGS finite-difference MLE (ParameterSpace unconstrained)"
        except (NotImplementedError, ValueError):
            if constraints == "auto":
                raise
            x, value, conv, iterations, message = _bfgs_minimize(
                objective, initial, bounds=bounds, max_iter=max_iter, tol=tol
            )
            method = "BFGS finite-difference MLE"
    else:
        x, value, conv, iterations, message = _bfgs_minimize(
            objective, initial, bounds=bounds, max_iter=max_iter, tol=tol
        )
        method = "BFGS finite-difference MLE"
    return _optimization_result(lik, x, value, conv, iterations, method, message)


def numerical_map(
    lik,
    initial,
    *,
    log_prior=None,
    priors=None,
    bounds=None,
    constraints=None,
    max_iter=500,
    tol=1e-8,
):
    """Compute a MAP estimate with either a joint log prior or named priors.

    The optimizer and constraint behavior match :func:`numerical_mle`. Exactly
    one prior representation must be supplied.
    """
    if (log_prior is None) == (priors is None):
        raise ValueError("provide exactly one of log_prior or priors")

    def prior_value(z):
        if log_prior is not None:
            return float(log_prior(dict(zip(lik.parameter_names, z, strict=True))))
        return sum(
            float(priors[name].logpdf(float(value)).evalf())
            for name, value in zip(lik.parameter_names, z, strict=True)
        )

    def objective(z):
        ll, lp = lik.log_likelihood(z), prior_value(z)
        return -(ll + lp) if math.isfinite(ll) and math.isfinite(lp) else math.inf

    use_auto = constraints == "auto" or (constraints is None and bounds is None)
    if use_auto:
        try:
            x, value, conv, iterations, message = _auto_unconstrained_minimize(
                lik, initial, objective, max_iter=max_iter, tol=tol
            )
            method = "BFGS finite-difference MAP (ParameterSpace unconstrained)"
        except (NotImplementedError, ValueError):
            if constraints == "auto":
                raise
            x, value, conv, iterations, message = _bfgs_minimize(
                objective, initial, bounds=bounds, max_iter=max_iter, tol=tol
            )
            method = "BFGS finite-difference MAP"
    else:
        x, value, conv, iterations, message = _bfgs_minimize(
            objective, initial, bounds=bounds, max_iter=max_iter, tol=tol
        )
        method = "BFGS finite-difference MAP"
    posterior_info = _hessian(objective, x)
    return _optimization_result(
        lik, x, value, conv, iterations, method, message, prior_value(x), posterior_info
    )


def likelihood_ratio_test(full, restricted, df):
    stat = max(0.0, 2 * (full.log_likelihood - restricted.log_likelihood))
    return HypothesisTestResult(
        stat, chi2_sf(stat, df), "greater", "likelihood-ratio test", (), 0.0, float(df)
    )


def wald_test(estimate, null, covariance, *, contrast=None):
    beta = np.atleast_1d(np.asarray(estimate, dtype=float))
    null_value = np.atleast_1d(np.asarray(null, dtype=float))
    cov = np.asarray(covariance, dtype=float)
    matrix = (
        np.eye(beta.size)
        if contrast is None
        else np.atleast_2d(np.asarray(contrast, dtype=float))
    )
    delta = matrix @ (beta - null_value)
    projected = matrix @ cov @ matrix.T
    stat = float(delta.T @ np.linalg.pinv(projected) @ delta)
    df = matrix.shape[0]
    return HypothesisTestResult(
        stat, chi2_sf(stat, df), "greater", "Wald test", (), 0.0, float(df)
    )


def score_test(lik, null_theta, *, tested_indices=None):
    theta = np.asarray(null_theta, dtype=float)
    gradient = _gradient(lik.log_likelihood, theta)
    info = observed_fisher_information(lik, theta)
    idx = (
        np.arange(theta.size)
        if tested_indices is None
        else np.asarray(tested_indices, dtype=int)
    )
    g = gradient[idx]
    block = info[np.ix_(idx, idx)]
    stat = float(g.T @ np.linalg.pinv(block) @ g)
    return HypothesisTestResult(
        stat,
        chi2_sf(stat, len(idx)),
        "greater",
        "score (Lagrange multiplier) test",
        (),
        0.0,
        float(len(idx)),
    )
