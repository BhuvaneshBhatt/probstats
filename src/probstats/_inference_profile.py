"""Internal implementation for inference profile."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._inference_likelihood import (
    Likelihood,
    OptimizationResult,
    _bfgs_minimize,
    _parameter_space_valid,
)
from ._special import chi2_sf


@dataclass(frozen=True)
class ProfileLikelihoodResult:
    parameter: str
    values: np.ndarray
    log_likelihoods: np.ndarray
    mle: float
    maximum_log_likelihood: float
    confidence: float
    cutoff: float
    interval: tuple[float, float] | None
    converged: np.ndarray

    def statistic(self):
        return 2 * (self.maximum_log_likelihood - self.log_likelihoods)


@dataclass(frozen=True)
class LikelihoodConfidenceRegion:
    likelihood: Likelihood
    fit: OptimizationResult
    parameter_names: tuple[str, ...]
    confidence: float
    cutoff: float
    max_iter: int = 300
    tol: float = 1e-8

    def statistic(self, values):
        vals = np.asarray(values, dtype=float).reshape(-1)
        if vals.size != len(self.parameter_names):
            raise ValueError("values have wrong dimension")
        indices = tuple(
            self.likelihood.parameter_names.index(name) for name in self.parameter_names
        )
        profiled_ll, _ = _conditional_max_log_likelihood(
            self.likelihood,
            self.fit,
            indices,
            vals,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        if not math.isfinite(profiled_ll):
            return math.inf
        return max(0.0, 2 * (self.fit.log_likelihood - profiled_ll))

    def contains(self, values):
        return self.statistic(values) <= self.cutoff


def _chi2_ppf(probability, df):
    if not 0 < probability < 1:
        raise ValueError("probability must lie in (0, 1)")
    lo, hi = 0.0, max(1.0, float(df))
    while 1 - chi2_sf(hi, df) < probability:
        hi *= 2
    for _ in range(100):
        mid = (lo + hi) / 2
        if 1 - chi2_sf(mid, df) < probability:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def likelihood_confidence_region(
    lik,
    fit,
    parameters=None,
    *,
    confidence=0.95,
    max_iter=300,
    tol=1e-8,
):
    names = tuple(lik.parameter_names if parameters is None else parameters)
    if not names or any(name not in lik.parameter_names for name in names):
        raise ValueError("unknown or empty parameter selection")
    cutoff = _chi2_ppf(confidence, len(names))
    return LikelihoodConfidenceRegion(
        lik, fit, names, confidence, cutoff, max_iter=max_iter, tol=tol
    )


def _conditional_max_log_likelihood(
    lik,
    fit,
    fixed_indices,
    fixed_values,
    *,
    max_iter=300,
    tol=1e-8,
):
    base = fit.estimate.copy()
    fixed_indices = tuple(fixed_indices)
    fixed_values = np.asarray(fixed_values, dtype=float).reshape(-1)
    if len(fixed_indices) != fixed_values.size:
        raise ValueError("fixed indices/values mismatch")
    base[list(fixed_indices)] = fixed_values
    nuisance = [i for i in range(base.size) if i not in fixed_indices]
    if not _parameter_space_valid(lik, base) and not nuisance:
        # With nuisance coordinates, profiling may restore cross-parameter validity.
        return -math.inf, False
    if not nuisance:
        return lik.log_likelihood(base), True
    initial = base[nuisance]

    def objective(z):
        theta = base.copy()
        theta[nuisance] = z
        if not _parameter_space_valid(lik, theta):
            return math.inf
        ll = lik.log_likelihood(theta)
        return -ll if math.isfinite(ll) else math.inf

    try:
        z, _, conv, _, _ = _bfgs_minimize(
            objective, initial, max_iter=max_iter, tol=tol
        )
    except ValueError:
        return -math.inf, False
    theta = base.copy()
    theta[nuisance] = z
    return lik.log_likelihood(theta), conv


def _profile_optimize_at(lik, fit, index, value, *, max_iter=300, tol=1e-8):
    return _conditional_max_log_likelihood(
        lik, fit, (index,), (value,), max_iter=max_iter, tol=tol
    )


def profile_likelihood(
    lik,
    fit,
    parameter,
    *,
    values=None,
    confidence=0.95,
    points=81,
    span=4.0,
    max_iter=300,
    tol=1e-8,
):
    """Profile one likelihood parameter while optimizing all nuisance values.

    The returned profile includes likelihood-ratio statistics and an interval
    obtained from the asymptotic chi-square cutoff.
    """
    if parameter not in lik.parameter_names:
        raise ValueError(f"unknown parameter {parameter!r}")
    index = lik.parameter_names.index(parameter)
    mle = float(fit.estimate[index])
    cutoff = _chi2_ppf(confidence, 1)
    if values is None:
        if fit.covariance is None:
            raise ValueError("automatic profile grid requires estimator covariance")
        se = math.sqrt(max(float(fit.covariance[index, index]), 0.0))
        if se <= 0:
            raise ValueError("automatic profile grid requires positive standard error")
        values = np.linspace(mle - span * se, mle + span * se, points)
    grid = np.asarray(values, dtype=float).reshape(-1)
    if grid.size < 2:
        raise ValueError("profile requires at least two parameter values")
    grid = np.unique(np.r_[grid, mle])
    grid.sort()
    lls = np.empty(grid.size, dtype=float)
    converged = np.zeros(grid.size, dtype=bool)
    for i, value in enumerate(grid):
        lls[i], converged[i] = _profile_optimize_at(
            lik, fit, index, float(value), max_iter=max_iter, tol=tol
        )
    statistics = 2 * (fit.log_likelihood - lls)
    accepted = statistics <= cutoff
    interval = None
    if np.any(accepted):
        accepted_indices = np.flatnonzero(accepted)
        left_i, right_i = int(accepted_indices[0]), int(accepted_indices[-1])
        left, right = float(grid[left_i]), float(grid[right_i])
        if left_i > 0 and np.isfinite(statistics[left_i - 1]):
            x0, x1 = grid[left_i - 1], grid[left_i]
            y0, y1 = statistics[left_i - 1], statistics[left_i]
            if y1 != y0:
                left = float(x0 + (cutoff - y0) * (x1 - x0) / (y1 - y0))
        if right_i + 1 < grid.size and np.isfinite(statistics[right_i + 1]):
            x0, x1 = grid[right_i], grid[right_i + 1]
            y0, y1 = statistics[right_i], statistics[right_i + 1]
            if y1 != y0:
                right = float(x0 + (cutoff - y0) * (x1 - x0) / (y1 - y0))
        interval = (left, right)
    return ProfileLikelihoodResult(
        parameter,
        grid,
        lls,
        mle,
        fit.log_likelihood,
        confidence,
        cutoff,
        interval,
        converged,
    )
