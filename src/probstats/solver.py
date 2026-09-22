"""Probability/expectation solver results and numerical fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import sympy as sp

from .events import ProbabilityEvent, ProductEvent, as_event
from .results import StatisticalResult
from .sampling import as_rng, sample


class EvaluationMethod(str, Enum):
    CLOSED_FORM = "closed_form"
    STRUCTURAL = "structural"
    SYMBOLIC = "symbolic"
    MONTE_CARLO = "monte_carlo"


@dataclass(frozen=True, slots=True)
class FunctionalResult(StatisticalResult):
    """A value plus provenance for a probability/statistics functional."""

    value: Any
    method: EvaluationMethod
    exact: bool
    standard_error: float | None = None
    samples: int | None = None


class NumericalEvaluationError(RuntimeError):
    """Raised when an explicitly numerical fallback cannot produce a value."""


def _truth(value: Any) -> bool:
    value = sp.sympify(value)
    if value is sp.true or value is True:
        return True
    if value is sp.false or value is False:
        return False
    try:
        return bool(value)
    except TypeError as exc:
        raise NumericalEvaluationError(
            f"event membership remained symbolic: {value}"
        ) from exc


def _event_mask(event: ProbabilityEvent, draws: Any) -> np.ndarray:
    if isinstance(event, ProductEvent):
        values = tuple(draws)
        if len(values) != len(event.events):
            raise NumericalEvaluationError("product draw and event dimensions differ")
        masks = [_event_mask(e, x) for e, x in zip(event.events, values)]
        return np.logical_and.reduce(masks)
    arr = np.asarray(draws)
    flat = arr.reshape(-1)
    mask = np.asarray(
        [
            _truth(event.contains(sp.sympify(x.item() if hasattr(x, "item") else x)))
            for x in flat
        ],
        dtype=bool,
    )
    return mask.reshape(arr.shape)


def monte_carlo_probability(
    distribution, event, *, variable=None, samples: int = 100_000, rng=None
) -> FunctionalResult:
    """Estimate an expectation from explicit Monte Carlo draws.

    Symbolic expressions are evaluated with NumPy lambdification. Product-law
    expressions may name one symbolic variable per sampled component.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    ev = as_event(event, variable=variable)
    draws = sample(distribution, size=samples, rng=as_rng(rng))
    mask = _event_mask(ev, draws)
    # Product events yield one Boolean per sample.  Scalar event shapes also do.
    if mask.ndim > 1:
        axes = tuple(range(1, mask.ndim))
        mask = np.all(mask, axis=axes)
    values = mask.astype(float)
    estimate = float(values.mean())
    se = float(np.sqrt(max(estimate * (1.0 - estimate), 0.0) / samples))
    return FunctionalResult(estimate, EvaluationMethod.MONTE_CARLO, False, se, samples)


def monte_carlo_expectation(
    distribution,
    expression=None,
    *,
    variable=None,
    variables=None,
    samples: int = 100_000,
    rng=None,
) -> FunctionalResult:
    """Estimate an expectation from explicit Monte Carlo draws.

    Symbolic expressions are evaluated with NumPy lambdification. Product-law
    expressions may name one symbolic variable per sampled component.
    """
    if samples <= 0:
        raise ValueError("samples must be positive")
    draws = sample(distribution, size=samples, rng=as_rng(rng))
    if expression is None:
        vals = np.asarray(draws, dtype=float)
    elif callable(expression):
        vals = np.asarray(expression(draws), dtype=float)
    else:
        expr = sp.sympify(expression)
        if variables is not None:
            vars_ = tuple(map(sp.sympify, variables))
            if not isinstance(draws, tuple) or len(draws) != len(vars_):
                raise NumericalEvaluationError(
                    "variables must match a product-distribution draw"
                )
            fn = sp.lambdify(vars_, expr, "numpy")
            vals = np.asarray(fn(*draws), dtype=float)
        else:
            if variable is None:
                symbols = sorted(
                    expr.free_symbols - distribution.free_symbols,
                    key=sp.default_sort_key,
                )
                if len(symbols) != 1:
                    raise NumericalEvaluationError(
                        "variable is required for numerical expectation"
                    )
                variable = symbols[0]
            fn = sp.lambdify(sp.sympify(variable), expr, "numpy")
            vals = np.asarray(fn(draws), dtype=float)
    if vals.shape == ():
        vals = np.full(samples, float(vals))
    if vals.shape[0] != samples:
        raise NumericalEvaluationError(
            "expectation expression did not preserve the sample axis"
        )
    estimate = np.mean(vals, axis=0)
    se = (
        np.std(vals, axis=0, ddof=1) / np.sqrt(samples)
        if samples > 1
        else np.zeros_like(estimate, dtype=float)
    )
    value = float(estimate) if np.ndim(estimate) == 0 else estimate
    standard_error = float(se) if np.ndim(se) == 0 else se
    return FunctionalResult(
        value, EvaluationMethod.MONTE_CARLO, False, standard_error, samples
    )


__all__ = [
    "EvaluationMethod",
    "FunctionalResult",
    "NumericalEvaluationError",
    "monte_carlo_expectation",
    "monte_carlo_probability",
]
