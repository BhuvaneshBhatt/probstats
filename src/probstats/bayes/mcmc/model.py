"""Adapters from the symbolic Bayesian model IR to numerical MCMC targets."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import sympy as sp

from ..core import InferenceKind, InferenceResult, InferenceStep, Model, Variable
from ..diagnostics import PosteriorData
from .core import AdaptiveMetropolis, NoUTurnSampler, Sampler, run_chains
from .diagnostics import diagnose_chains


def model_log_posterior(
    model: Model, variables: Sequence[str | Variable] | None = None
):
    selected = (
        model.latent_variables
        if variables is None
        else tuple(
            model.variable_map[v if isinstance(v, str) else v.name] for v in variables
        )
    )
    symbols = tuple(v.symbol for v in selected)
    expr = model.joint_log_density(substitute_observations=True)
    extra = expr.free_symbols - set(symbols)
    if extra:
        raise ValueError(
            f"Log posterior contains unresolved symbols: {sorted(map(str, extra))}"
        )
    func = sp.lambdify(symbols, expr, modules="numpy")

    def evaluate(point: np.ndarray) -> float:
        values = np.asarray(point, dtype=float).reshape(-1)
        if len(values) != len(symbols):
            raise ValueError("Point dimension does not match model variables")
        try:
            value = float(func(*values))
        except (
            TypeError,
            ValueError,
            OverflowError,
            ZeroDivisionError,
            FloatingPointError,
        ):
            return -np.inf
        return value if np.isfinite(value) else -np.inf

    return evaluate


def model_log_posterior_gradient(
    model: Model, variables: Sequence[str | Variable] | None = None
):
    selected = (
        model.latent_variables
        if variables is None
        else tuple(
            model.variable_map[v if isinstance(v, str) else v.name] for v in variables
        )
    )
    symbols = tuple(v.symbol for v in selected)
    expr = model.joint_log_density(substitute_observations=True)
    gradients = tuple(sp.diff(expr, s) for s in symbols)
    if any(g.has(sp.Derivative) for g in gradients):
        return None
    func = sp.lambdify(symbols, gradients, modules="numpy")

    def evaluate(point: np.ndarray) -> np.ndarray:
        values = np.asarray(point, dtype=float).reshape(-1)
        out = np.asarray(func(*values), dtype=float).reshape(-1)
        if out.shape != (len(symbols),) or not np.all(np.isfinite(out)):
            raise ValueError("Non-finite model gradient")
        return out

    return evaluate


def infer_mcmc_model(
    model: Model,
    *,
    initial_positions: Sequence[np.ndarray],
    sampler: Sampler | None = None,
    variables: Sequence[str | Variable] | None = None,
    draws: int = 1000,
    warmup: int = 500,
    thin: int = 1,
    rng: Any = None,
) -> InferenceResult:
    selected = (
        model.latent_variables
        if variables is None
        else tuple(
            model.variable_map[v if isinstance(v, str) else v.name] for v in variables
        )
    )
    logp = model_log_posterior(model, selected)
    gradient = model_log_posterior_gradient(model, selected)
    chosen = sampler or (
        NoUTurnSampler(gradient=gradient)
        if gradient is not None
        else AdaptiveMetropolis()
    )
    chains = run_chains(
        logp,
        initial_positions,
        sampler_factory=chosen,
        draws=draws,
        warmup=warmup,
        thin=thin,
        rng=rng,
        parameter_names=[v.name for v in selected],
    )
    diagnostics = diagnose_chains(chains)
    arrays = {
        v.name: np.stack([c.samples[:, i] for c in chains])
        for i, v in enumerate(selected)
    }
    pdata = PosteriorData(
        arrays,
        sample_stats={"log_prob": np.stack([c.log_prob for c in chains])},
        attrs={"source": "mcmc", "sampler": chosen.name},
    )
    return InferenceResult(
        posterior=pdata,
        kind=InferenceKind.SAMPLED,
        steps=(
            InferenceStep(
                "mcmc",
                f"Sampled {len(chains)} chains with {chosen.name}.",
                False,
                {"draws": draws, "warmup": warmup, "thin": thin},
            ),
        ),
        diagnostics={
            "mcmc": diagnostics,
            "rhat": diagnostics.rank_rhat,
            "ess": diagnostics.bulk_ess,
            "mcse": diagnostics.mcse,
            "divergences": diagnostics.divergence_count,
            "acceptance_rate": diagnostics.acceptance_rate,
        },
        metadata={"chains": chains, "sampler": chosen.name},
    )
