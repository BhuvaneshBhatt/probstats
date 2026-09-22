"""Contrasts, marginal means, ANCOVA, and model comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ._model_anova import FactorialANOVAResult, factorial_anova
from ._model_design import DesignMatrix, design_matrix
from ._model_glm import GLMResult, resolve_family
from ._model_linear import FormulaRegressionResult
from ._special import chi2_cdf, f_cdf, normal_cdf, t_cdf
from .results import StatisticalResult


@dataclass(frozen=True)
class CoefficientContrastResult(StatisticalResult):
    contrast: np.ndarray
    estimate: float
    standard_error: float
    statistic: float
    pvalue: float
    null_value: float
    degrees_of_freedom: float | None
    method: str


@dataclass(frozen=True)
class JointWaldResult(StatisticalResult):
    statistic: float
    pvalue: float
    df: int
    method: str
    contrast_matrix: np.ndarray
    null_value: np.ndarray


def _model_components(model):
    if isinstance(model, FormulaRegressionResult):
        return (
            np.asarray(model.coefficients, dtype=float),
            np.asarray(model.covariance, dtype=float),
            model.coefficient_names,
            float(model.df_resid),
            "t",
        )
    if isinstance(model, GLMResult):
        return (
            np.asarray(model.coefficients, dtype=float),
            np.asarray(model.covariance, dtype=float),
            model.coefficient_names,
            None,
            "normal",
        )
    raise TypeError("model must be a FormulaRegressionResult or GLMResult")


def _contrast_vector(names: tuple[str, ...], contrast) -> np.ndarray:
    if isinstance(contrast, Mapping):
        vector = np.zeros(len(names), dtype=float)
        for name, coefficient in contrast.items():
            if name not in names:
                raise KeyError(f"unknown coefficient {name!r}")
            vector[names.index(name)] = float(coefficient)
        return vector
    vector = np.asarray(contrast, dtype=float).reshape(-1)
    if len(vector) != len(names):
        raise ValueError("contrast length must equal the number of coefficients")
    return vector


def coefficient_contrast(model, contrast, *, value=0.0) -> CoefficientContrastResult:
    """Test a scalar linear coefficient contrast c' beta = value."""
    beta, covariance, names, df, reference = _model_components(model)
    c = _contrast_vector(names, contrast)
    estimate = float(c @ beta)
    variance = float(c @ covariance @ c)
    se = math.sqrt(max(variance, 0.0))
    delta = estimate - value
    if abs(delta) <= 1e-12 * (1 + abs(estimate) + abs(value)):
        statistic = 0.0
    else:
        statistic = delta / se if se > 0 else math.copysign(math.inf, delta)
    if math.isinf(statistic):
        pvalue = 0.0
    elif reference == "t":
        pvalue = 2 * min(t_cdf(statistic, float(df)), 1 - t_cdf(statistic, float(df)))
    else:
        pvalue = 2 * min(normal_cdf(statistic), 1 - normal_cdf(statistic))
    return CoefficientContrastResult(
        c,
        estimate,
        se,
        float(statistic),
        float(pvalue),
        float(value),
        df,
        f"{reference} coefficient contrast",
    )


def joint_wald_test(model, contrast_matrix, *, value=None) -> JointWaldResult:
    """Joint Wald chi-square test R beta = q."""
    beta, covariance, names, _, _ = _model_components(model)
    if isinstance(contrast_matrix, Mapping):
        rows = [
            _contrast_vector(names, contrast) for contrast in contrast_matrix.values()
        ]
        r = np.vstack(rows)
    else:
        r = np.asarray(contrast_matrix, dtype=float)
        if r.ndim == 1:
            r = r.reshape(1, -1)
    if r.ndim != 2 or r.shape[1] != len(beta):
        raise ValueError("contrast_matrix must have one column per model coefficient")
    q = (
        np.zeros(r.shape[0])
        if value is None
        else np.asarray(value, dtype=float).reshape(-1)
    )
    if len(q) != r.shape[0]:
        raise ValueError("value must have one entry per contrast row")
    delta = r @ beta - q
    middle = r @ covariance @ r.T
    statistic = float(delta @ np.linalg.pinv(middle) @ delta)
    df = int(np.linalg.matrix_rank(r))
    pvalue = 1 - chi2_cdf(statistic, df) if df > 0 else 1.0
    return JointWaldResult(
        statistic, float(pvalue), df, "joint Wald chi-square test", r, q
    )


@dataclass(frozen=True)
class MarginalMean:
    level: Any
    estimate: float
    standard_error: float | None = None


@dataclass(frozen=True)
class EstimatedMarginalMeansResult(StatisticalResult):
    factor: str
    means: tuple[MarginalMean, ...]
    scale: str
    weighting: str

    def mean(self, level) -> MarginalMean:
        for result in self.means:
            if result.level == level:
                return result
        raise KeyError(level)


def _reference_grid(
    dm: DesignMatrix, target: str
) -> tuple[list[dict[str, Any]], tuple[Any, ...]]:
    if dm.training_data is None:
        raise ValueError(
            "estimated marginal means require a fitted design matrix with training data"
        )
    # Resolve target by source name, accepting C(factor) or bare factor.
    target_encodings = [
        enc
        for enc in dm.factor_encodings.values()
        if enc.source_name == target and enc.categorical
    ]
    if not target_encodings:
        raise ValueError(f"{target!r} is not a categorical factor in the model")
    levels = target_encodings[0].levels
    categorical_sources: dict[str, tuple[Any, ...]] = {}
    numeric_sources: set[str] = set()
    for enc in dm.factor_encodings.values():
        if enc.categorical:
            categorical_sources.setdefault(enc.source_name, enc.levels)
        else:
            numeric_sources.add(enc.source_name)
    other_factors = [name for name in categorical_sources if name != target]
    combos = (
        list(product(*(categorical_sources[name] for name in other_factors)))
        if other_factors
        else [()]
    )
    numeric_means = {
        name: float(np.mean(np.asarray(dm.training_data[name], dtype=float)))
        for name in numeric_sources
    }
    grid = []
    for level in levels:
        for combo in combos:
            row = dict(numeric_means)
            row[target] = level
            row.update(dict(zip(other_factors, combo, strict=True)))
            grid.append(row)
    return grid, levels


def estimated_marginal_means(
    model, factor: str, *, scale="response"
) -> EstimatedMarginalMeansResult:
    """Balanced reference-grid estimated marginal means for a categorical factor."""
    dm = model.design_matrix
    grid, levels = _reference_grid(dm, factor)
    results = []
    for level in levels:
        rows = [row for row in grid if row[factor] == level]
        batch = {name: np.asarray([row[name] for row in rows]) for name in rows[0]}
        x = dm.transform(batch)
        xbar = np.mean(x, axis=0)
        eta = float(xbar @ model.coefficients)
        var_eta = float(xbar @ model.covariance @ xbar)
        se_eta = math.sqrt(max(var_eta, 0.0))
        if isinstance(model, GLMResult) and scale == "response":
            _, link = resolve_family(model.family, model.link)
            estimate = float(link.inverse(np.asarray([eta]))[0])
            derivative = float(link.dmu_deta(np.asarray([eta]))[0])
            se = abs(derivative) * se_eta
        elif scale in {"linear", "response"}:
            estimate, se = eta, se_eta
        else:
            raise ValueError("scale must be 'response' or 'linear'")
        results.append(MarginalMean(level, estimate, se))
    return EstimatedMarginalMeansResult(
        factor, tuple(results), scale, "equal reference-grid"
    )


def ancova(
    formula: str, data: Mapping[str, Any], *, type=2, contrasts=None
) -> FactorialANOVAResult:
    """Fixed-effects ANCOVA using the common formula/term infrastructure."""
    dm = design_matrix(formula, data, contrasts=contrasts)
    has_cat = any(enc.categorical for enc in dm.factor_encodings.values())
    has_numeric = any(not enc.categorical for enc in dm.factor_encodings.values())
    if not (has_cat and has_numeric):
        raise ValueError(
            "ANCOVA requires at least one categorical factor and one numeric covariate"
        )
    return factorial_anova(formula, data, type=type, contrasts=contrasts)


@dataclass(frozen=True)
class ModelComparisonRow:
    model: str
    df_resid: int
    residual_deviance: float
    df_difference: int | None
    statistic: float | None
    pvalue: float | None
    criterion: str


@dataclass(frozen=True)
class ModelComparisonResult(StatisticalResult):
    rows: tuple[ModelComparisonRow, ...]
    method: str


def compare_models(*models, names=None) -> ModelComparisonResult:
    """Compare nested OLS models by F tests or GLMs by deviance tests."""
    if len(models) < 2:
        raise ValueError("at least two fitted models are required")
    if names is None:
        names = tuple(f"model_{i + 1}" for i in range(len(models)))
    if len(names) != len(models):
        raise ValueError("names length must match models")
    regression = all(isinstance(model, FormulaRegressionResult) for model in models)
    glm_models = all(isinstance(model, GLMResult) for model in models)
    if not (regression or glm_models):
        raise TypeError("all models must be formula OLS models or all must be GLMs")
    responses = [np.asarray(model.design_matrix.response) for model in models]
    response_names = {model.design_matrix.response_name for model in models}
    if len(response_names) != 1 or any(
        not np.array_equal(response, responses[0]) for response in responses[1:]
    ):
        raise ValueError("model comparison requires the same response data")
    rows = []
    previous = None
    for name, model in zip(names, models, strict=True):
        if regression:
            dev = float(model.sigma2 * model.df_resid)
            df_resid = int(model.df_resid)
            criterion = "SSE"
        else:
            dev = float(model.deviance)
            df_resid = int(model.df_resid)
            criterion = "deviance"
        if previous is None:
            rows.append(
                ModelComparisonRow(
                    str(name), df_resid, dev, None, None, None, criterion
                )
            )
        else:
            prev_dev, prev_df, _ = previous
            df_diff = prev_df - df_resid
            if df_diff <= 0:
                raise ValueError("models must be supplied from reduced to fuller model")
            improvement = prev_dev - dev
            if improvement < -1e-8:
                raise ValueError("models do not appear nested in the supplied order")
            improvement = max(0.0, improvement)
            if regression:
                stat = (improvement / df_diff) / (dev / df_resid)
                pvalue = 1 - f_cdf(stat, df_diff, df_resid)
                method = "nested OLS F test"
            else:
                if model.family.startswith("quasi"):
                    stat = (improvement / df_diff) / max(model.dispersion, 1e-15)
                    pvalue = 1 - f_cdf(stat, df_diff, df_resid)
                    method = "quasi-GLM scaled-deviance F test"
                else:
                    stat = improvement
                    pvalue = 1 - chi2_cdf(stat, df_diff)
                    method = "nested GLM deviance chi-square test"
            rows.append(
                ModelComparisonRow(
                    str(name),
                    df_resid,
                    dev,
                    df_diff,
                    float(stat),
                    float(pvalue),
                    criterion,
                )
            )
        previous = (dev, df_resid, model)
    return ModelComparisonResult(
        tuple(rows), method if len(models) > 1 else "model comparison"
    )


__all__ = [
    "CoefficientContrastResult",
    "EstimatedMarginalMeansResult",
    "JointWaldResult",
    "MarginalMean",
    "ModelComparisonResult",
    "ModelComparisonRow",
    "ancova",
    "coefficient_contrast",
    "compare_models",
    "estimated_marginal_means",
    "joint_wald_test",
]
