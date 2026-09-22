"""Formula-based ordinary and weighted linear regression."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._model_design import DesignMatrix, design_matrix
from ._model_glm import _observation_vector
from ._special import f_cdf, t_cdf
from ._tabular import column_mapping, labeled_vector
from .results import ModelFitResult


@dataclass(frozen=True)
class FormulaRegressionResult(ModelFitResult):
    design_matrix: DesignMatrix
    regression: Any

    def __getattr__(self, name):
        """Delegate fitted-model attributes without recursing during initialization."""
        if name == "regression":
            raise AttributeError(name)
        regression = object.__getattribute__(self, "regression")
        return getattr(regression, name)

    @property
    def coefficient_names(self):
        return self.design_matrix.column_names

    def predict(self, data: Mapping[str, Any]):
        values = self.design_matrix.transform(data) @ self.regression.coefficients
        _, index = column_mapping(data)
        return labeled_vector(values, index, name=self.design_matrix.response_name)

    def summary_data(self):
        return {
            "formula": self.design_matrix.formula,
            "n_observations": len(self.design_matrix.response),
            "r_squared": getattr(self.regression, "r_squared", None),
            "adjusted_r_squared": getattr(self.regression, "adjusted_r_squared", None),
        }

    def coefficient_table(self):
        return self.regression.coefficient_table(self.coefficient_names)

    def to_frame(self):
        return self.coefficient_table()


def _weighted_ols(dm: DesignMatrix, weights, covariance="classical"):
    """Fit weighted least squares for an already compiled design matrix."""
    from .inference import RegressionResult

    if str(covariance).lower() != "classical":
        raise ValueError("weighted formula OLS supports classical covariance only")
    x, y = dm.matrix, dm.response
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != len(y) or np.any(~np.isfinite(w)) or np.any(w <= 0):
        raise ValueError(
            "weights must contain one positive finite value per observation"
        )
    n, p = x.shape
    if n <= p or np.linalg.matrix_rank(x) < p:
        raise ValueError(
            "weighted regression requires a full-rank design with residual degrees of freedom"
        )
    sqrt_w = np.sqrt(w)
    beta = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)[0]
    fitted = x @ beta
    residuals = y - fitted
    df = n - p
    sse = float(np.sum(w * residuals**2))
    sigma2 = sse / df
    covariance_matrix = sigma2 * np.linalg.pinv(x.T @ (w[:, None] * x))
    se = np.sqrt(np.maximum(np.diag(covariance_matrix), 0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, math.inf), where=se > 0)
    pvalues = np.array(
        [2 * min(t_cdf(float(v), df), 1 - t_cdf(float(v), df)) for v in t_stats]
    )
    mean = float(np.average(y, weights=w))
    sst = float(np.sum(w * (y - mean) ** 2))
    r2 = 1 - sse / sst if sst > 0 else 1.0
    adjusted = 1 - (1 - r2) * (n - 1) / df
    numerator_df = p - 1 if dm.intercept else p
    ssr = max(0.0, sst - sse)
    if numerator_df == 0:
        f_stat, f_pvalue = 0.0, 1.0
    elif sse == 0:
        f_stat, f_pvalue = math.inf, 0.0
    else:
        f_stat = (ssr / numerator_df) / (sse / df)
        f_pvalue = 1 - f_cdf(f_stat, numerator_df, df)
    return RegressionResult(
        beta,
        se,
        t_stats,
        pvalues,
        covariance_matrix,
        fitted,
        residuals,
        r2,
        adjusted,
        sigma2,
        df,
        f_stat,
        f_pvalue,
        dm.intercept,
        x,
        y,
        "classical",
    )


def linear_model(
    formula: str,
    data: Mapping[str, Any],
    *,
    contrasts=None,
    covariance="classical",
    weights=None,
) -> FormulaRegressionResult:
    """Fit OLS/WLS from a formula using the shared design-matrix compiler."""
    from .inference import linear_regression

    dm = design_matrix(formula, data, contrasts=contrasts)
    if weights is not None:
        w = _observation_vector(
            weights, data, len(dm.response), name="weights", default=1.0
        )
        return FormulaRegressionResult(dm, _weighted_ols(dm, w, covariance))
    predictors = dm.matrix[:, 1:] if dm.intercept else dm.matrix
    fit = linear_regression(
        predictors, dm.response, intercept=dm.intercept, covariance=covariance
    )
    return FormulaRegressionResult(dm, fit)


__all__ = ["FormulaRegressionResult", "linear_model"]
