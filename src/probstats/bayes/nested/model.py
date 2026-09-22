"""Adapters between the symbolic Model IR and nested sampling."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp

from ..core import Model, Variable
from .core import ConstrainedSampler, PriorTransform, infer_nested


def model_log_likelihood(
    model: Model, variables: Sequence[str | Variable] | None = None
):
    """Compile observed-target factors into a numerical log-likelihood callable.

    Prior factors on latent variables are excluded: nested sampling
    receives the prior separately through ``prior_sampler`` or ``prior_transform``.
    """
    selected = (
        model.latent_variables
        if variables is None
        else tuple(
            model.variable_map[v if isinstance(v, str) else v.name] for v in variables
        )
    )
    symbols = tuple(v.symbol for v in selected)
    observed_names = set(model.observations)
    likelihood_terms = [
        factor.expression
        for factor in model.factors
        if factor.target.name in observed_names
    ]
    if not likelihood_terms:
        raise ValueError("Model has no factors targeting observed variables")
    expr = sp.Add(*likelihood_terms)
    expr = expr.subs(
        {
            model.variable_map[name].symbol: sp.sympify(value)
            for name, value in model.observations.items()
        }
    )
    extra = expr.free_symbols - set(symbols)
    if extra:
        raise ValueError(
            f"Log likelihood contains unresolved symbols: {sorted(map(str, extra))}"
        )
    func = sp.lambdify(symbols, expr, modules="numpy")

    def evaluate(point: np.ndarray) -> float:
        values = np.asarray(point, dtype=float).reshape(-1)
        if len(values) != len(symbols):
            raise ValueError("Point dimension does not match selected model variables")
        return float(func(*values))

    return evaluate


def infer_nested_model(
    model: Model,
    prior_sampler: Callable[[np.random.Generator], np.ndarray] | None = None,
    *,
    prior_transform: PriorTransform | None = None,
    variables: Sequence[str | Variable] | None = None,
    n_live: int = 100,
    max_iterations: int = 10_000,
    min_iterations: int = 100,
    termination_fraction: float = 0.01,
    constrained_sampler: ConstrainedSampler | None = None,
    rng=None,
):
    selected = (
        model.latent_variables
        if variables is None
        else tuple(
            model.variable_map[v if isinstance(v, str) else v.name] for v in variables
        )
    )
    return infer_nested(
        model_log_likelihood(model, selected),
        prior_sampler,
        prior_transform=prior_transform,
        ndim=len(selected),
        n_live=n_live,
        max_iterations=max_iterations,
        min_iterations=min_iterations,
        termination_fraction=termination_fraction,
        constrained_sampler=constrained_sampler,
        rng=rng,
        parameter_names=[v.name for v in selected],
    )
