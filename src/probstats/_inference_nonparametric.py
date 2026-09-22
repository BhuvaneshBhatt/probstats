"""Internal implementation for inference nonparametric."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from ._inference_likelihood import _as1d
from ._special import chi2_sf, normal_cdf
from .distributions import Exponential, Normal, Uniform
from .estimation import maximum_likelihood
from .testing import HypothesisTestResult


def _rank_average(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = np.arange(1, values.size + 1)
    for value in np.unique(values):
        mask = values == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def mann_whitney_u_test(
    x, y, alternative="two-sided", *, exact="auto", exact_threshold=200_000
):
    """Compare two independent samples using the Mann-Whitney rank statistic.

    Exact conditional enumeration is used when requested and computationally
    feasible; otherwise the tie-corrected normal approximation is used.
    """
    a, b = _as1d(x), _as1d(y)
    values = np.r_[a, b]
    ranks = _rank_average(values)
    statistic = float(ranks[: a.size].sum() - a.size * (a.size + 1) / 2)
    mean = a.size * b.size / 2
    total_assignments = math.comb(values.size, a.size)
    use_exact = exact is True or (
        exact == "auto" and total_assignments <= exact_threshold
    )
    if use_exact:
        if total_assignments > exact_threshold and exact is True:
            raise ValueError("exact Mann-Whitney enumeration exceeds exact_threshold")
        stats = np.empty(total_assignments, dtype=float)
        offset = a.size * (a.size + 1) / 2
        for i, chosen in enumerate(combinations(range(values.size), a.size)):
            stats[i] = float(ranks[list(chosen)].sum() - offset)
        if alternative == "two-sided":
            extreme = np.abs(stats - mean) >= abs(statistic - mean) - 1e-15
        elif alternative == "less":
            extreme = stats <= statistic + 1e-15
        elif alternative == "greater":
            extreme = stats >= statistic - 1e-15
        else:
            raise ValueError("invalid alternative")
        pvalue = float(np.mean(extreme))
        method = "Mann-Whitney U test (exact conditional permutation distribution)"
    else:
        _, counts = np.unique(values, return_counts=True)
        ties = sum(c**3 - c for c in counts)
        variance = (
            a.size
            * b.size
            / 12
            * ((values.size + 1) - ties / (values.size * (values.size - 1)))
        )
        z = (statistic - mean) / math.sqrt(variance) if variance > 0 else 0.0
        if alternative == "two-sided":
            pvalue = 2 * min(normal_cdf(z), 1 - normal_cdf(z))
        elif alternative == "less":
            pvalue = normal_cdf(z)
        elif alternative == "greater":
            pvalue = 1 - normal_cdf(z)
        else:
            raise ValueError("invalid alternative")
        method = "Mann-Whitney U test (normal approximation)"
    return HypothesisTestResult(
        statistic, min(1.0, pvalue), alternative, method, (a.size, b.size)
    )


def wilcoxon_signed_rank_test(
    x,
    y=None,
    alternative="two-sided",
    *,
    exact="auto",
    exact_threshold=200_000,
):
    """Test paired or one-sample location differences with signed ranks.

    Zero differences are removed. Exact sign enumeration is used for small
    state spaces; larger problems use the tie-corrected normal approximation.
    """
    differences = _as1d(x) if y is None else _as1d(x) - _as1d(y)
    differences = differences[differences != 0]
    if differences.size == 0:
        return HypothesisTestResult(
            0.0, 1.0, alternative, "Wilcoxon signed-rank test", (0,), 0.0
        )
    absolute = np.abs(differences)
    ranks = _rank_average(absolute)
    statistic = float(ranks[differences > 0].sum())
    n = differences.size
    mean = n * (n + 1) / 4
    total_assignments = 2**n
    use_exact = exact is True or (
        exact == "auto" and total_assignments <= exact_threshold
    )
    if use_exact:
        if total_assignments > exact_threshold and exact is True:
            raise ValueError("exact Wilcoxon enumeration exceeds exact_threshold")
        stats = np.zeros(total_assignments, dtype=float)
        for mask in range(total_assignments):
            stats[mask] = sum(ranks[i] for i in range(n) if mask & (1 << i))
        if alternative == "two-sided":
            extreme = np.abs(stats - mean) >= abs(statistic - mean) - 1e-15
        elif alternative == "less":
            extreme = stats <= statistic + 1e-15
        elif alternative == "greater":
            extreme = stats >= statistic - 1e-15
        else:
            raise ValueError("invalid alternative")
        pvalue = float(np.mean(extreme))
        method = "Wilcoxon signed-rank test (exact sign-randomization distribution)"
    else:
        _, counts = np.unique(absolute, return_counts=True)
        variance = n * (n + 1) * (2 * n + 1) / 24
        variance -= sum(c * (c - 1) * (2 * c + 5) for c in counts if c > 1) / 48
        z = (statistic - mean) / math.sqrt(variance) if variance > 0 else 0.0
        if alternative == "two-sided":
            pvalue = 2 * min(normal_cdf(z), 1 - normal_cdf(z))
        elif alternative == "less":
            pvalue = normal_cdf(z)
        elif alternative == "greater":
            pvalue = 1 - normal_cdf(z)
        else:
            raise ValueError("invalid alternative")
        method = "Wilcoxon signed-rank test (normal approximation)"
    return HypothesisTestResult(
        statistic, min(1.0, pvalue), alternative, method, (n,), 0.0
    )


def kruskal_wallis_test(*groups):
    """Perform the Kruskal–Wallis rank test for independent groups."""
    values = [_as1d(group) for group in groups]
    if len(values) < 2:
        raise ValueError("Kruskal-Wallis requires at least two groups")
    pooled = np.concatenate(values)
    labels = np.concatenate([np.full(len(group), i) for i, group in enumerate(values)])
    ranks = _rank_average(pooled)
    n = pooled.size
    statistic = 12 / (n * (n + 1)) * sum(
        ranks[labels == i].sum() ** 2 / len(group) for i, group in enumerate(values)
    ) - 3 * (n + 1)
    _, counts = np.unique(pooled, return_counts=True)
    correction = 1 - sum(c**3 - c for c in counts) / (n**3 - n)
    statistic = statistic / correction if correction > 0 else 0.0
    df = len(values) - 1
    return HypothesisTestResult(
        float(statistic),
        chi2_sf(statistic, df),
        "greater",
        "Kruskal-Wallis test",
        tuple(map(len, values)),
        None,
        float(df),
    )


def _fast_gof_fit(family, data):
    cls = family if isinstance(family, type) else type(family)
    x = np.asarray(data, dtype=float)
    if cls is Normal:
        sigma = float(np.sqrt(np.mean((x - x.mean()) ** 2)))
        if sigma <= 0:
            raise ValueError("Normal fit requires nonzero variance")
        return Normal(float(x.mean()), sigma)
    if cls is Exponential:
        return Exponential(1 / float(x.mean()))
    if cls is Uniform:
        return Uniform(float(x.min()), float(x.max()))
    return maximum_likelihood(family, x).distribution


def _numeric_cdf_values(distribution, x):
    if isinstance(distribution, Normal):
        mean, sigma = map(float, distribution.parameters)
        return np.array([normal_cdf((value - mean) / sigma) for value in x])
    if isinstance(distribution, Exponential):
        rate = float(distribution.rate)
        return np.where(x < 0, 0.0, 1 - np.exp(-rate * x))
    if isinstance(distribution, Uniform):
        low, high = map(float, distribution.parameters)
        return np.clip((x - low) / (high - low), 0.0, 1.0)
    return np.array([float(distribution.cdf(value).evalf()) for value in x])


def _ks_statistic(data, distribution):
    x = np.sort(_as1d(data))
    cdf_values = _numeric_cdf_values(distribution, x)
    n = x.size
    d_plus = np.max(np.arange(1, n + 1) / n - cdf_values)
    d_minus = np.max(cdf_values - np.arange(0, n) / n)
    return float(max(d_plus, d_minus))


def kolmogorov_smirnov_test(
    data,
    distribution=None,
    *,
    family=None,
    fit=False,
    bootstrap_iterations=500,
    rng=None,
):
    x = np.sort(_as1d(data))
    n = x.size
    if distribution is None:
        if family is None or not fit:
            raise ValueError("provide distribution or family with fit=True")
        distribution = _fast_gof_fit(family, x)
        statistic = _ks_statistic(x, distribution)
        gen = np.random.default_rng(rng)
        extreme = 0
        for _ in range(bootstrap_iterations):
            sample = np.asarray(distribution.sample(size=n, rng=gen), dtype=float)
            refit = _fast_gof_fit(family, sample)
            extreme += _ks_statistic(sample, refit) >= statistic - 1e-15
        pvalue = (extreme + 1) / (bootstrap_iterations + 1)
        method = "Kolmogorov-Smirnov goodness-of-fit with parametric-bootstrap fitted-parameter calibration"
        return HypothesisTestResult(statistic, pvalue, "greater", method, (n,))
    statistic = _ks_statistic(x, distribution)
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * statistic
    pvalue = min(
        1.0,
        max(
            0.0,
            2
            * sum(
                (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam)
                for k in range(1, 101)
            ),
        ),
    )
    return HypothesisTestResult(
        statistic, pvalue, "greater", "one-sample Kolmogorov-Smirnov test", (n,)
    )


@dataclass(frozen=True)
class MultipleTestingResult:
    pvalues: np.ndarray
    adjusted_pvalues: np.ndarray
    rejected: np.ndarray
    alpha: float
    method: str


def adjust_pvalues(pvalues, *, method="holm", alpha=0.05):
    """Adjust a family of p-values for FWER or FDR control."""
    p = np.asarray(pvalues, dtype=float).reshape(-1)
    if p.size == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be finite values in [0, 1]")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1)")
    key = method.lower()
    m = p.size
    order = np.argsort(p, kind="mergesort")
    ranked = p[order]
    adjusted_ranked = np.empty(m, dtype=float)
    if key == "bonferroni":
        adjusted = np.minimum(1.0, m * p)
    elif key == "sidak":
        adjusted = np.minimum(1.0, -np.expm1(m * np.log1p(-p)))
    elif key == "holm":
        raw = (m - np.arange(m)) * ranked
        adjusted_ranked = np.maximum.accumulate(raw)
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(1.0, adjusted_ranked)
    elif key == "holm-sidak":
        raw = -np.expm1((m - np.arange(m)) * np.log1p(-ranked))
        adjusted_ranked = np.maximum.accumulate(raw)
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(1.0, adjusted_ranked)
    elif key == "hochberg":
        raw = (m - np.arange(m)) * ranked
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(1.0, adjusted_ranked)
    elif key == "benjamini-hochberg":
        raw = ranked * m / np.arange(1, m + 1)
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(1.0, adjusted_ranked)
    elif key == "benjamini-yekutieli":
        harmonic = float(np.sum(1 / np.arange(1, m + 1)))
        raw = ranked * m * harmonic / np.arange(1, m + 1)
        adjusted_ranked = np.minimum.accumulate(raw[::-1])[::-1]
        adjusted = np.empty(m)
        adjusted[order] = np.minimum(1.0, adjusted_ranked)
    else:
        raise ValueError(
            "method must be bonferroni, sidak, holm, holm-sidak, hochberg, "
            "benjamini-hochberg, or benjamini-yekutieli"
        )
    return MultipleTestingResult(p, adjusted, adjusted <= alpha, alpha, key)
