"""Internal implementation for testing diagnostics."""

from __future__ import annotations

import math

import numpy as np

from ._special import chi2_sf, f_cdf, normal_cdf
from ._testing_core import HypothesisTestResult, _x
from ._testing_dependence import MultivariateNormalityResult
from ._validation import integer


def autocorrelation(data, lag=1, *, demean=True):
    """Sample autocorrelation at a nonnegative lag."""
    x = _x(data).reshape(-1)
    if not isinstance(lag, (int, np.integer)) or lag < 0 or lag >= x.size:
        raise ValueError("lag must be an integer in [0, n-1]")
    z = x - x.mean() if demean else x.copy()
    denominator = float(np.dot(z, z))
    if denominator == 0:
        raise ValueError("autocorrelation is undefined for a constant series")
    if lag == 0:
        return 1.0
    return float(np.dot(z[lag:], z[:-lag]) / denominator)


def _portmanteau_test(data, lags, *, ljung_box):
    x = _x(data).reshape(-1)
    if isinstance(lags, (int, np.integer)) and not isinstance(lags, (bool, np.bool_)):
        maximum_lag = integer(lags, name="lags", minimum=1)
        lag_values = np.arange(1, maximum_lag + 1)
    else:
        raw_lags = tuple(lags)
        if not raw_lags:
            raise ValueError("lags must contain positive integers")
        lag_values = np.asarray(
            [integer(lag, name="lag", minimum=1) for lag in raw_lags], dtype=int
        )
    if np.max(lag_values) >= x.size:
        raise ValueError("all lags must be smaller than the sample size")
    rhos = np.array([autocorrelation(x, int(k)) for k in lag_values])
    if ljung_box:
        statistic = float(
            x.size * (x.size + 2) * np.sum(rhos**2 / (x.size - lag_values))
        )
        method = "Ljung-Box portmanteau test"
    else:
        statistic = float(x.size * np.sum(rhos**2))
        method = "Box-Pierce portmanteau test"
    df = int(lag_values.size)
    return HypothesisTestResult(
        statistic,
        chi2_sf(statistic, df),
        "greater",
        method,
        (x.size,),
        0.0,
        df,
        float(np.max(np.abs(rhos))),
        None,
    )


def ljung_box_test(data, lags=10):
    return _portmanteau_test(data, lags, ljung_box=True)


def box_pierce_test(data, lags=10):
    return _portmanteau_test(data, lags, ljung_box=False)


def durbin_watson(residuals):
    """Durbin-Watson statistic for first-order residual autocorrelation."""
    e = _x(residuals).reshape(-1)
    if e.size < 2:
        raise ValueError("Durbin-Watson requires at least two residuals")
    denominator = float(np.dot(e, e))
    if denominator == 0:
        raise ValueError("Durbin-Watson is undefined for all-zero residuals")
    return float(np.sum(np.diff(e) ** 2) / denominator)


def _matrix_sample(data, *, name="data"):
    x = np.asarray(data, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a finite n-by-p matrix with n >= 2")
    return x


def hotelling_t2_test(data, mean=None):
    """One-sample Hotelling T-squared test for a multivariate mean vector."""
    x = _matrix_sample(data)
    n, p = x.shape
    if n <= p:
        raise ValueError("one-sample Hotelling test requires n > p")
    target = np.zeros(p) if mean is None else np.asarray(mean, dtype=float).reshape(-1)
    if target.size != p:
        raise ValueError("mean has wrong dimension")
    delta = x.mean(axis=0) - target
    covariance = np.cov(x, rowvar=False, ddof=1)
    covariance = np.atleast_2d(covariance)
    try:
        precision_delta = np.linalg.solve(covariance, delta)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sample covariance matrix must be nonsingular") from exc
    t2 = float(n * delta @ precision_delta)
    fstat = (n - p) * t2 / (p * (n - 1))
    pvalue = 1 - f_cdf(fstat, p, n - p)
    return HypothesisTestResult(
        t2,
        pvalue,
        "greater",
        "one-sample Hotelling T-squared test",
        (n,),
        0.0,
        p,
        float(np.linalg.norm(delta)),
        None,
    )


def two_sample_hotelling_t2_test(x, y):
    """Two-sample Hotelling T-squared test assuming equal covariance matrices."""
    a = _matrix_sample(x, name="x")
    b = _matrix_sample(y, name="y")
    if a.shape[1] != b.shape[1]:
        raise ValueError("x and y must have the same number of variables")
    n1, p = a.shape
    n2 = b.shape[0]
    if n1 + n2 <= p + 1:
        raise ValueError("two-sample Hotelling test requires n1+n2 > p+1")
    s1 = np.atleast_2d(np.cov(a, rowvar=False, ddof=1))
    s2 = np.atleast_2d(np.cov(b, rowvar=False, ddof=1))
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    delta = a.mean(axis=0) - b.mean(axis=0)
    try:
        precision_delta = np.linalg.solve(pooled, delta)
    except np.linalg.LinAlgError as exc:
        raise ValueError("pooled covariance matrix must be nonsingular") from exc
    t2 = float((n1 * n2 / (n1 + n2)) * delta @ precision_delta)
    df2 = n1 + n2 - p - 1
    fstat = df2 * t2 / (p * (n1 + n2 - 2))
    pvalue = 1 - f_cdf(fstat, p, df2)
    return HypothesisTestResult(
        t2,
        pvalue,
        "greater",
        "two-sample Hotelling T-squared test",
        (n1, n2),
        0.0,
        p,
        float(np.linalg.norm(delta)),
        None,
    )


def box_m_test(*samples):
    """Box's M test for equality of covariance matrices across groups."""
    groups = [_matrix_sample(sample) for sample in samples]
    if len(groups) < 2:
        raise ValueError("Box's M test requires at least two groups")
    p = groups[0].shape[1]
    if any(group.shape[1] != p for group in groups):
        raise ValueError("all groups must have the same number of variables")
    if any(group.shape[0] <= p for group in groups):
        raise ValueError("each group needs more observations than variables")
    ns = np.array([group.shape[0] for group in groups], dtype=float)
    covariances = [
        np.atleast_2d(np.cov(group, rowvar=False, ddof=1)) for group in groups
    ]
    dfs = ns - 1
    pooled = sum(df * cov for df, cov in zip(dfs, covariances, strict=True)) / dfs.sum()
    sign_p, logdet_p = np.linalg.slogdet(pooled)
    logdets = [np.linalg.slogdet(cov) for cov in covariances]
    if sign_p <= 0 or any(sign <= 0 for sign, _ in logdets):
        raise ValueError("covariance matrices must be positive definite")
    m = float(
        dfs.sum() * logdet_p
        - sum(df * ld for df, (_, ld) in zip(dfs, logdets, strict=True))
    )
    g = len(groups)
    correction = 1 - (
        (2 * p * p + 3 * p - 1)
        / (6 * (p + 1) * (g - 1))
        * (float(np.sum(1 / dfs)) - 1 / float(dfs.sum()))
    )
    statistic = max(0.0, correction * m)
    df = int((g - 1) * p * (p + 1) / 2)
    return HypothesisTestResult(
        statistic,
        chi2_sf(statistic, df),
        "greater",
        "Box's M covariance-equality test",
        tuple(int(n) for n in ns),
        None,
        df,
    )


def mardia_normality_test(data):
    """Mardia multivariate skewness and kurtosis tests of normality."""
    x = _matrix_sample(data)
    n, p = x.shape
    if n <= p:
        raise ValueError("Mardia test requires more observations than variables")
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / n
    try:
        solved = np.linalg.solve(covariance, centered.T)
    except np.linalg.LinAlgError as exc:
        raise ValueError("sample covariance matrix must be nonsingular") from exc
    inner = centered @ solved
    skewness = float(np.mean(inner**3))
    skew_stat = n * skewness / 6
    skew_df = int(p * (p + 1) * (p + 2) / 6)
    skew_p = chi2_sf(skew_stat, skew_df)
    mahalanobis2 = np.diag(inner)
    kurtosis = float(np.mean(mahalanobis2**2))
    expected = p * (p + 2)
    kurtosis_z = (kurtosis - expected) / math.sqrt(8 * p * (p + 2) / n)
    kurtosis_p = 2 * min(normal_cdf(kurtosis_z), 1 - normal_cdf(kurtosis_z))
    return MultivariateNormalityResult(
        skewness,
        skew_stat,
        skew_df,
        skew_p,
        kurtosis,
        kurtosis_z,
        kurtosis_p,
        n,
        p,
    )
