"""Factorial ANOVA for formula-compiled linear models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._model_linear import FormulaRegressionResult, linear_model
from ._special import f_cdf
from .results import StatisticalResult


@dataclass(frozen=True)
class ANOVATermResult(StatisticalResult):
    term: str
    df: int
    sum_squares: float
    mean_square: float
    statistic: float
    pvalue: float


@dataclass(frozen=True)
class FactorialANOVAResult(StatisticalResult):
    formula: str
    type: int
    terms: tuple[ANOVATermResult, ...]
    residual_df: int
    residual_sum_squares: float
    residual_mean_square: float
    model: FormulaRegressionResult

    def term(self, name: str) -> ANOVATermResult:
        for result in self.terms:
            if result.term == name:
                return result
        raise KeyError(name)


def _sse_for_columns(
    x: np.ndarray, y: np.ndarray, columns: list[int]
) -> tuple[float, int]:
    matrix = x[:, columns] if columns else np.empty((len(y), 0))
    if matrix.shape[1] == 0:
        fitted = np.zeros_like(y)
        rank = 0
    else:
        beta = np.linalg.lstsq(matrix, y, rcond=None)[0]
        fitted = matrix @ beta
        rank = np.linalg.matrix_rank(matrix)
    residual = y - fitted
    return float(residual @ residual), int(rank)


def factorial_anova(
    formula: str, data: Mapping[str, Any], *, type=2, contrasts=None
) -> FactorialANOVAResult:
    """Multi-factor fixed-effects ANOVA using the shared formula/design infrastructure.

    Type I is sequential. Type II tests each term after terms that do not contain it,
    respecting hierarchy. Type III compares the full model with the model dropping
    only the requested term. Sum contrasts are recommended for Type III inference.
    """
    if type not in {1, 2, 3}:
        raise ValueError("ANOVA type must be 1, 2, or 3")
    model = linear_model(formula, data, contrasts=contrasts)
    dm, fit = model.design_matrix, model.regression
    x, y = dm.matrix, dm.response
    all_columns = list(range(x.shape[1]))
    intercept_cols = list(dm.term_slices.get("Intercept", ()))
    term_names = [term.name for term in dm.terms if term.name in dm.term_slices]
    full_sse = float(fit.residuals @ fit.residuals)
    full_rank = int(np.linalg.matrix_rank(x))
    residual_df = len(y) - full_rank
    if residual_df <= 0:
        raise ValueError("ANOVA requires residual degrees of freedom")
    mse = full_sse / residual_df
    results = []
    previous_cols = intercept_cols.copy()
    for name in term_names:
        term_cols = list(dm.term_slices[name])
        if type == 1:
            reduced_cols = previous_cols
            full_cols = previous_cols + term_cols
            previous_cols = full_cols
        elif type == 3:
            reduced_cols = [col for col in all_columns if col not in term_cols]
            full_cols = all_columns
        else:
            factors = set(name.split(":"))
            # Type II: condition on terms not containing this term as a subset;
            # higher-order interactions containing the tested term are excluded.
            conditioning = intercept_cols.copy()
            for other in term_names:
                if other == name:
                    continue
                other_factors = set(other.split(":"))
                if factors.issubset(other_factors):
                    continue
                conditioning.extend(dm.term_slices[other])
            reduced_cols = sorted(set(conditioning))
            full_cols = sorted(set(reduced_cols + term_cols))
        sse_reduced, rank_reduced = _sse_for_columns(x, y, reduced_cols)
        sse_full, rank_full = _sse_for_columns(x, y, full_cols)
        ss = max(0.0, sse_reduced - sse_full)
        df = rank_full - rank_reduced
        ms = ss / df if df > 0 else 0.0
        statistic = ms / mse if mse > 0 and df > 0 else (math.inf if ms > 0 else 0.0)
        pvalue = (
            0.0
            if math.isinf(statistic)
            else 1 - f_cdf(statistic, df, residual_df)
            if df > 0
            else 1.0
        )
        results.append(ANOVATermResult(name, df, ss, ms, statistic, pvalue))
    return FactorialANOVAResult(
        formula, type, tuple(results), residual_df, full_sse, mse, model
    )


__all__ = ["ANOVATermResult", "FactorialANOVAResult", "factorial_anova"]
