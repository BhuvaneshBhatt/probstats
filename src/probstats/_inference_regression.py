"""Internal implementation for inference regression."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ._inference_likelihood import _as1d
from ._special import f_cdf, t_cdf
from ._tabular import array_metadata, labeled_vector
from .results import ModelFitResult, coefficient_frame


@dataclass(frozen=True)
class ANOVAResult:
    statistic: float
    pvalue: float
    df_between: int
    df_within: int
    ss_between: float
    ss_within: float
    ms_between: float
    ms_within: float
    group_means: tuple[float, ...]


def one_way_anova(*groups):
    values = [_as1d(group) for group in groups]
    if len(values) < 2:
        raise ValueError("ANOVA requires at least two groups")
    n, k = sum(map(len, values)), len(values)
    grand = float(np.concatenate(values).mean())
    means = tuple(float(group.mean()) for group in values)
    ss_between = sum(
        len(group) * (mean - grand) ** 2
        for group, mean in zip(values, means, strict=True)
    )
    ss_within = sum(
        float(np.sum((group - mean) ** 2))
        for group, mean in zip(values, means, strict=True)
    )
    df_between, df_within = k - 1, n - k
    ms_between, ms_within = ss_between / df_between, ss_within / df_within
    statistic = (
        math.inf
        if ms_within == 0 and ms_between > 0
        else (0.0 if ms_within == 0 else ms_between / ms_within)
    )
    pvalue = (
        0.0 if math.isinf(statistic) else 1 - f_cdf(statistic, df_between, df_within)
    )
    return ANOVAResult(
        statistic,
        pvalue,
        df_between,
        df_within,
        ss_between,
        ss_within,
        ms_between,
        ms_within,
        means,
    )


@dataclass(frozen=True)
class RegressionResult(ModelFitResult):
    coefficients: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    pvalues: np.ndarray
    covariance: np.ndarray
    fitted: np.ndarray
    residuals: np.ndarray
    r_squared: float
    adjusted_r_squared: float
    sigma2: float
    df_resid: int
    f_statistic: float
    f_pvalue: float
    intercept: bool
    design: np.ndarray | None = field(default=None, repr=False, compare=False)
    response: np.ndarray | None = field(default=None, repr=False, compare=False)
    covariance_type: str = "classical"
    feature_names: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )
    response_name: str | None = field(default=None, repr=False, compare=False)
    row_index: Any = field(default=None, repr=False, compare=False)

    def predict(self, x):
        raw, index, _, _ = array_metadata(x)
        matrix = np.asarray(raw, dtype=float)
        matrix = matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
        if self.intercept:
            matrix = np.column_stack([np.ones(matrix.shape[0]), matrix])
        values = matrix @ self.coefficients
        return labeled_vector(values, index, name=self.response_name)

    def with_covariance(self, covariance="HC3"):
        return regression_with_covariance(self, covariance)

    def coefficient_table(self, names=None):
        """Return coefficient estimates as a labeled pandas table."""
        labels = names or self.feature_names
        if labels is None:
            labels = tuple(f"x{index}" for index in range(len(self.coefficients)))
        return coefficient_frame(
            labels,
            self.coefficients,
            self.standard_errors,
            self.t_statistics,
            self.pvalues,
        )

    def to_frame(self):
        return self.coefficient_table()


def robust_regression_covariance(result, kind="HC3"):
    """White/MacKinnon heteroskedasticity-consistent OLS covariance."""
    if result.design is None:
        raise ValueError(
            "robust covariance requires a RegressionResult produced by linear_regression"
        )
    design = np.asarray(result.design, dtype=float)
    residuals = np.asarray(result.residuals, dtype=float)
    n, p = design.shape
    bread = np.linalg.pinv(design.T @ design)
    leverage = np.clip(
        np.einsum("ij,jk,ik->i", design, bread, design), 0.0, 1.0 - 1e-12
    )
    key = kind.upper()
    squared = residuals**2
    if key == "HC0":
        omega = squared
    elif key == "HC1":
        omega = squared * n / (n - p)
    elif key == "HC2":
        omega = squared / (1 - leverage)
    elif key == "HC3":
        omega = squared / (1 - leverage) ** 2
    elif key == "HC4":
        delta = np.minimum(4.0, n * leverage / p)
        omega = squared / (1 - leverage) ** delta
    else:
        raise ValueError("robust covariance must be one of HC0, HC1, HC2, HC3, HC4")
    meat = design.T @ (omega[:, None] * design)
    return bread @ meat @ bread


def regression_with_covariance(result, covariance="HC3"):
    key = covariance.upper()
    cov = robust_regression_covariance(result, key)
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    t_stats = np.divide(
        result.coefficients,
        se,
        out=np.full_like(result.coefficients, math.inf),
        where=se > 0,
    )
    pvalues = np.array(
        [
            2
            * min(
                t_cdf(float(value), result.df_resid),
                1 - t_cdf(float(value), result.df_resid),
            )
            for value in t_stats
        ]
    )
    return replace(
        result,
        covariance=cov,
        standard_errors=se,
        t_statistics=t_stats,
        pvalues=pvalues,
        covariance_type=key,
    )


def linear_regression(x, y, *, intercept=True, covariance="classical"):
    """Fit ordinary least squares with coefficient and model-level inference.

    ``covariance`` may request the classical covariance or one of the supported
    heteroskedasticity-consistent estimators. Residual degrees of freedom are
    required because this routine always returns inferential statistics.
    """
    covariance_option = covariance
    x_raw, row_index, feature_names, _ = array_metadata(x)
    _, response_index, _, response_name = array_metadata(y)
    response = _as1d(y, name="y")
    design = np.asarray(x_raw, dtype=float)
    design = design.reshape(-1, 1) if design.ndim == 1 else design
    if design.shape[0] != response.size:
        raise ValueError("X and y row counts differ")
    if intercept:
        design = np.column_stack([np.ones(response.size), design])
    n, p = design.shape
    if n <= p:
        raise ValueError("regression requires residual degrees of freedom")
    beta = np.linalg.lstsq(design, response, rcond=None)[0]
    fitted = design @ beta
    residuals = response - fitted
    df = n - p
    sse = float(residuals @ residuals)
    sigma2 = sse / df
    classical_covariance = sigma2 * np.linalg.pinv(design.T @ design)
    se = np.sqrt(np.maximum(np.diag(classical_covariance), 0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, math.inf), where=se > 0)
    pvalues = np.array(
        [2 * min(t_cdf(float(v), df), 1 - t_cdf(float(v), df)) for v in t_stats]
    )
    sst = float(np.sum((response - response.mean()) ** 2))
    r2 = 1 - sse / sst if sst > 0 else 1.0
    adjusted = 1 - (1 - r2) * (n - 1) / df
    numerator_df = p - 1 if intercept else p
    ssr = max(0.0, sst - sse)
    if numerator_df == 0:
        f_stat, f_pvalue = 0.0, 1.0
    elif sse == 0:
        f_stat, f_pvalue = math.inf, 0.0
    else:
        f_stat = (ssr / numerator_df) / (sse / df)
        f_pvalue = 1 - f_cdf(f_stat, numerator_df, df)
    result = RegressionResult(
        beta,
        se,
        t_stats,
        pvalues,
        classical_covariance,
        fitted,
        residuals,
        r2,
        adjusted,
        sigma2,
        df,
        f_stat,
        f_pvalue,
        intercept,
        design,
        response,
        "classical",
        (("Intercept",) if intercept else ())
        + (
            feature_names
            or tuple(f"x{index}" for index in range(design.shape[1] - int(intercept)))
        ),
        response_name,
        row_index if row_index is not None else response_index,
    )
    return (
        result
        if covariance_option.lower() == "classical"
        else regression_with_covariance(result, covariance_option)
    )
