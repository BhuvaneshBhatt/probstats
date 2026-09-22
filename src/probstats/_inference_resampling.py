"""Internal implementation for inference resampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ._inference_likelihood import _as1d
from ._special import normal_cdf, normal_ppf


@dataclass(frozen=True)
class ResamplingResult:
    observed: float
    replicates: np.ndarray
    estimate: float
    standard_error: float
    confidence_interval: tuple[float, float] | None
    pvalue: float | None
    method: str
    iterations: int


def _jackknife_values(x, statistic):
    if x.size < 3:
        raise ValueError("jackknife requires at least three observations")
    return np.array(
        [float(statistic(np.delete(x, i))) for i in range(x.size)], dtype=float
    )


def _bootstrap_standard_error(sample, statistic, iterations, gen):
    reps = np.array(
        [
            float(statistic(gen.choice(sample, size=sample.size, replace=True)))
            for _ in range(iterations)
        ]
    )
    return float(reps.std(ddof=1))


def bootstrap(
    data,
    statistic=np.mean,
    *,
    iterations=2000,
    confidence=0.95,
    rng=None,
    method="percentile",
    standard_error=None,
    studentized_iterations=200,
):
    """Bootstrap a statistic and optionally construct a confidence interval.

    ``method`` may be ``"percentile"``, ``"bca"``, or ``"bootstrap-t"``.
    Bootstrap-t uses ``standard_error`` when provided; otherwise it estimates
    replicate standard errors with an inner bootstrap.
    """
    x = _as1d(data)
    if iterations < 2 or not 0 < confidence < 1:
        raise ValueError("invalid iterations or confidence")
    method_key = method.lower().replace("_", "-")
    if method_key not in {"percentile", "bca", "bootstrap-t", "studentized"}:
        raise ValueError("method must be percentile, bca, or bootstrap-t")
    gen = np.random.default_rng(rng)
    observed = float(statistic(x))
    samples = [gen.choice(x, size=x.size, replace=True) for _ in range(iterations)]
    reps = np.array([float(statistic(sample)) for sample in samples])
    alpha = 1 - confidence
    if method_key == "percentile":
        ci = tuple(map(float, np.quantile(reps, [alpha / 2, 1 - alpha / 2])))
        label = "nonparametric bootstrap percentile"
    elif method_key == "bca":
        proportion_less = np.mean(reps < observed)
        eps = 0.5 / iterations
        z0 = normal_ppf(float(np.clip(proportion_less, eps, 1 - eps)))
        jack = _jackknife_values(x, statistic)
        jack_mean = float(jack.mean())
        diffs = jack_mean - jack
        denominator = 6 * float(np.sum(diffs**2)) ** 1.5
        acceleration = float(np.sum(diffs**3)) / denominator if denominator > 0 else 0.0
        z_low, z_high = normal_ppf(alpha / 2), normal_ppf(1 - alpha / 2)
        adjusted = []
        for z in (z_low, z_high):
            denom = 1 - acceleration * (z0 + z)
            prob = (
                normal_cdf(z0 + (z0 + z) / denom)
                if denom != 0
                else (0.0 if z < 0 else 1.0)
            )
            adjusted.append(float(np.clip(prob, 0, 1)))
        ci = tuple(map(float, np.quantile(reps, adjusted)))
        label = "nonparametric bootstrap BCa"
    else:
        if studentized_iterations < 10:
            raise ValueError("studentized_iterations must be at least 10")
        se_func = standard_error
        if se_func is None:
            se_observed = _bootstrap_standard_error(
                x, statistic, studentized_iterations, gen
            )
            se_values = np.array(
                [
                    _bootstrap_standard_error(
                        sample, statistic, studentized_iterations, gen
                    )
                    for sample in samples
                ]
            )
        else:
            se_observed = float(se_func(x))
            se_values = np.array([float(se_func(sample)) for sample in samples])
        valid = np.isfinite(se_values) & (se_values > 0)
        if se_observed <= 0 or not np.isfinite(se_observed) or valid.sum() < 2:
            raise ValueError("bootstrap-t requires positive finite standard errors")
        pivots = (reps[valid] - observed) / se_values[valid]
        q_low, q_high = np.quantile(pivots, [alpha / 2, 1 - alpha / 2])
        ci = (
            float(observed - q_high * se_observed),
            float(observed - q_low * se_observed),
        )
        label = "nonparametric bootstrap-t"
    return ResamplingResult(
        observed,
        reps,
        float(reps.mean()),
        float(reps.std(ddof=1)),
        ci,
        None,
        label,
        iterations,
    )


def permutation_test(
    x,
    y,
    statistic=None,
    *,
    alternative="two-sided",
    iterations=5000,
    rng=None,
    exact_threshold=100_000,
):
    """Test exchangeability of two samples by label permutation.

    Small assignment spaces are enumerated exactly. Larger spaces use Monte
    Carlo permutations with a finite-simulation correction.
    """
    a, b = _as1d(x, name="x"), _as1d(y, name="y")
    statistic = statistic or (lambda u, v: np.mean(u) - np.mean(v))
    observed = float(statistic(a, b))
    pool = np.r_[a, b]
    total_assignments = math.comb(pool.size, a.size)
    if total_assignments <= exact_threshold:
        reps = np.empty(total_assignments)
        all_indices = np.arange(pool.size)
        for i, chosen in enumerate(combinations(range(pool.size), a.size)):
            mask = np.ones(pool.size, dtype=bool)
            mask[list(chosen)] = False
            reps[i] = float(statistic(pool[list(chosen)], pool[all_indices[mask]]))
        method = "exact permutation test"
        correction = 0
    else:
        gen = np.random.default_rng(rng)
        reps = np.empty(iterations)
        for i in range(iterations):
            shuffled = gen.permutation(pool)
            reps[i] = float(statistic(shuffled[: a.size], shuffled[a.size :]))
        method = "Monte Carlo permutation test"
        correction = 1
    if alternative == "two-sided":
        extreme = np.abs(reps) >= abs(observed) - 1e-15
    elif alternative == "greater":
        extreme = reps >= observed - 1e-15
    elif alternative == "less":
        extreme = reps <= observed + 1e-15
    else:
        raise ValueError("invalid alternative")
    pvalue = (correction + int(extreme.sum())) / (correction + reps.size)
    return ResamplingResult(
        observed,
        reps,
        float(reps.mean()),
        float(reps.std(ddof=1)),
        None,
        pvalue,
        method,
        reps.size,
    )
