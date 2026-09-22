"""Internal implementation for testing dependence."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._special import normal_cdf, t_cdf
from ._testing_core import HypothesisTestResult, _tail, _x
from .results import HypothesisResult


@dataclass(frozen=True)
class MultivariateNormalityResult(HypothesisResult):
    """Mardia multivariate-normality diagnostics."""

    skewness: float
    skewness_statistic: float
    skewness_df: int
    skewness_pvalue: float
    kurtosis: float
    kurtosis_z: float
    kurtosis_pvalue: float
    sample_size: int
    dimension: int

    @property
    def reject_05(self):
        return self.skewness_pvalue < 0.05 or self.kurtosis_pvalue < 0.05


def _paired_vectors(x, y):
    a = _x(x).reshape(-1)
    b = _x(y).reshape(-1)
    if a.size != b.size or a.size < 2:
        raise ValueError("paired samples must have equal lengths of at least 2")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("paired samples must contain finite observations")
    return a, b


def _average_ranks(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(x, y):
    """Spearman rank correlation with average ranks for ties."""
    a, b = _paired_vectors(x, y)
    ra = _average_ranks(a)
    rb = _average_ranks(b)
    sa = ra.std()
    sb = rb.std()
    if sa == 0 or sb == 0:
        raise ValueError("Spearman correlation is undefined for a constant sample")
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_correlation_test(x, y, alternative="two-sided"):
    """Test zero Spearman rank correlation using the usual t approximation."""
    a, b = _paired_vectors(x, y)
    if a.size < 3:
        raise ValueError("Spearman correlation test requires at least 3 pairs")
    rho = spearman_correlation(a, b)
    df = a.size - 2
    if abs(rho) >= 1:
        stat = math.copysign(math.inf, rho)
        pvalue = 0.0
    else:
        stat = rho * math.sqrt(df / max(1e-300, 1 - rho * rho))
        pvalue = _tail(lambda z: t_cdf(z, df), stat, alternative)
    return HypothesisTestResult(
        stat,
        pvalue,
        alternative,
        "Spearman rank-correlation test (t approximation)",
        (a.size,),
        0.0,
        df,
        rho,
        None,
    )


def _tie_group_sizes(values):
    _, counts = np.unique(np.asarray(values), return_counts=True)
    return counts[counts > 1].astype(float)


class _FenwickOrder:
    """Fenwick tree for rank counts used by Kendall statistics."""

    def __init__(self, size):
        self.tree = np.zeros(size + 1, dtype=np.int64)

    def add(self, index):
        index += 1
        while index < self.tree.size:
            self.tree[index] += 1
            index += index & -index

    def prefix(self, end):
        total = 0
        while end > 0:
            total += int(self.tree[end])
            end -= end & -end
        return total


def _kendall_score(a, b):
    """Return Kendall's concordance score in O(n log n), excluding x ties."""
    order = np.lexsort((b, a))
    ax = a[order]
    by = b[order]
    _, y_rank = np.unique(by, return_inverse=True)
    tree = _FenwickOrder(int(y_rank.max()) + 1)
    score = 0
    prior = 0
    start = 0
    n = ax.size
    while start < n:
        stop = start + 1
        while stop < n and ax[stop] == ax[start]:
            stop += 1
        for rank in y_rank[start:stop]:
            rank = int(rank)
            less = tree.prefix(rank)
            less_or_equal = tree.prefix(rank + 1)
            greater = prior - less_or_equal
            score += less - greater
        for rank in y_rank[start:stop]:
            tree.add(int(rank))
        prior += stop - start
        start = stop
    return float(score)


def kendall_tau(x, y):
    """Kendall's tau-b using an O(n log n) inversion-count algorithm."""
    a, b = _paired_vectors(x, y)
    score = _kendall_score(a, b)
    n0 = a.size * (a.size - 1) / 2
    tx = sum(t * (t - 1) / 2 for t in _tie_group_sizes(a))
    ty = sum(t * (t - 1) / 2 for t in _tie_group_sizes(b))
    denominator = math.sqrt((n0 - tx) * (n0 - ty))
    if denominator == 0:
        raise ValueError("Kendall tau is undefined when either sample is constant")
    return float(score / denominator)


def kendall_tau_test(x, y, alternative="two-sided"):
    """Kendall tau-b test using the large-sample tie-corrected Normal approximation."""
    a, b = _paired_vectors(x, y)
    n = a.size
    if n < 3:
        raise ValueError("Kendall test requires at least 3 pairs")
    score = _kendall_score(a, b)
    tx = _tie_group_sizes(a)
    ty = _tie_group_sizes(b)
    var_s = n * (n - 1) * (2 * n + 5)
    var_s -= float(np.sum(tx * (tx - 1) * (2 * tx + 5)))
    var_s -= float(np.sum(ty * (ty - 1) * (2 * ty + 5)))
    var_s /= 18.0
    if n > 2:
        var_s += (
            float(np.sum(tx * (tx - 1) * (tx - 2)))
            * float(np.sum(ty * (ty - 1) * (ty - 2)))
            / (9.0 * n * (n - 1) * (n - 2))
        )
    var_s += (
        float(np.sum(tx * (tx - 1)))
        * float(np.sum(ty * (ty - 1)))
        / (2.0 * n * (n - 1))
    )
    if var_s <= 0:
        raise ValueError("Kendall test variance is zero")
    z = score / math.sqrt(var_s)
    pvalue = _tail(normal_cdf, z, alternative)
    return HypothesisTestResult(
        z,
        pvalue,
        alternative,
        "Kendall tau-b test (tie-corrected Normal approximation)",
        (n,),
        0.0,
        None,
        kendall_tau(a, b),
        math.sqrt(var_s),
    )


def _centered_distances(values):
    data = np.asarray(values, dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError("data must contain at least two scalar or vector observations")
    if not np.all(np.isfinite(data)):
        raise ValueError("data must contain only finite observations")
    diff = data[:, None, :] - data[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=2))
    return (
        distances
        - distances.mean(axis=0)[None, :]
        - distances.mean(axis=1)[:, None]
        + distances.mean()
    )


def _distance_correlation_from_centered(ac, bc, *, squared=False):
    dcov2 = max(0.0, float(np.mean(ac * bc)))
    dvarx2 = max(0.0, float(np.mean(ac * ac)))
    dvary2 = max(0.0, float(np.mean(bc * bc)))
    if dvarx2 == 0 or dvary2 == 0:
        return 0.0
    dcor2 = max(0.0, min(1.0, dcov2 / math.sqrt(dvarx2 * dvary2)))
    return dcor2 if squared else math.sqrt(dcor2)


def distance_correlation(x, y, *, squared=False):
    """Biased sample distance correlation for scalar or vector observations."""
    ac = _centered_distances(x)
    bc = _centered_distances(y)
    if ac.shape != bc.shape:
        raise ValueError("x and y must contain the same number of observations")
    return _distance_correlation_from_centered(ac, bc, squared=squared)


def distance_correlation_test(x, y, *, permutations=999, rng=None):
    """Permutation test of independence using distance correlation."""
    if isinstance(permutations, (bool, np.bool_)) or not isinstance(
        permutations, (int, np.integer)
    ):
        raise TypeError("permutations must be an integer")
    permutations = int(permutations)
    if permutations < 1:
        raise ValueError("permutations must be positive")
    ac = _centered_distances(x)
    bc = _centered_distances(y)
    if ac.shape != bc.shape:
        raise ValueError("x and y must contain the same number of observations")
    observed = _distance_correlation_from_centered(ac, bc)
    generator = np.random.default_rng(rng)
    exceed = 0
    n = ac.shape[0]
    for _ in range(permutations):
        order = generator.permutation(n)
        permuted = bc[np.ix_(order, order)]
        statistic = _distance_correlation_from_centered(ac, permuted)
        if statistic >= observed - 1e-15:
            exceed += 1
    pvalue = (exceed + 1) / (permutations + 1)
    return HypothesisTestResult(
        observed,
        float(pvalue),
        "greater",
        "distance-correlation permutation test",
        (n,),
        0.0,
        None,
        observed,
        None,
    )


def cramers_v(table, *, bias_corrected=True):
    """Cramer's V for a contingency table, optionally using the bias correction."""
    observed = np.asarray(table, dtype=float)
    if (
        observed.ndim != 2
        or min(observed.shape) < 2
        or not np.all(np.isfinite(observed))
        or np.any(observed < 0)
    ):
        raise ValueError(
            "table must be a finite nonnegative 2D array with at least 2x2 cells"
        )
    n = float(observed.sum())
    if n <= 1:
        raise ValueError("contingency table must contain more than one observation")
    row = observed.sum(axis=1, keepdims=True)
    col = observed.sum(axis=0, keepdims=True)
    expected = row @ col / n
    if np.any(expected <= 0):
        raise ValueError("all expected cell counts must be positive")
    phi2 = float(np.sum((observed - expected) ** 2 / expected)) / n
    r, k = observed.shape
    if not bias_corrected:
        return math.sqrt(phi2 / min(k - 1, r - 1))
    phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denominator = min(kc - 1, rc - 1)
    return 0.0 if denominator <= 0 else math.sqrt(phi2 / denominator)
