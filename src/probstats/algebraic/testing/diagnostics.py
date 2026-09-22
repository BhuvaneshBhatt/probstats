"""Diagnostics and simulation calibration for SDL semialgebraic tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from numbers import Real

import numpy as np

from ..._arrays import readonly_array
from ._validation import (
    generator_from_seed_or_rng,
    positive_integer,
    spawn_named_generators,
)
from .hypotheses import SemialgebraicHypothesis
from .sdl import SDLTestResult, sdl_test


@dataclass(frozen=True, slots=True)
class SDLDiagnostics:
    """Numerical diagnostics for one completed basic-null SDL test."""

    realized_budget: int
    requested_budget: int
    realized_budget_ratio: float
    alpha_n: float
    projection_size: int
    blocks_per_observation: int
    kernel_order: int
    constraint_count: int
    hajek_variance_share: np.ndarray
    kernel_variance_share: np.ndarray
    zero_standard_error: np.ndarray
    zero_over_zero: np.ndarray
    nonzero_over_zero: np.ndarray
    bootstrap_standard_error: float
    bootstrap_critical_values: Mapping[float, float]

    def __post_init__(self):
        for name in (
            "hajek_variance_share",
            "kernel_variance_share",
            "zero_standard_error",
            "zero_over_zero",
            "nonzero_over_zero",
        ):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class SDLCalibrationResult:
    """Monte Carlo null calibration summary for a fixed SDL configuration."""

    alpha: float
    repetitions: int
    rejection_rate: float
    monte_carlo_standard_error: float
    p_values: np.ndarray
    statistics: np.ndarray
    realized_budgets: np.ndarray
    requested_budget: int
    kernel_orders: np.ndarray
    projection_sizes: np.ndarray
    seed: int | None

    def __post_init__(self):
        for name in (
            "p_values",
            "statistics",
            "realized_budgets",
            "kernel_orders",
            "projection_sizes",
        ):
            object.__setattr__(self, name, readonly_array(getattr(self, name)))


def _probability(value: float, *, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not 0 < value < 1:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return value


def sdl_diagnostics(
    result: SDLTestResult,
    *,
    critical_levels: Sequence[float] = (0.90, 0.95, 0.99),
) -> SDLDiagnostics:
    """Summarize the numerical regime of a completed SDL test.

    The variance shares expose how much of each coordinate's estimated variance
    comes from the first-order Hájek term versus incomplete-kernel sampling.
    Bootstrap critical values are empirical quantiles of the already-generated
    multiplier distribution; no new randomness is introduced.
    """
    if not isinstance(result, SDLTestResult):
        raise TypeError("result must be an SDLTestResult")
    levels = tuple(
        _probability(level, name="critical level") for level in critical_levels
    )
    st = result.studentization
    m = result.kernel_order
    hajek_term = (m * m) * np.asarray(st.hajek.variance, dtype=float)
    kernel_term = st.alpha * np.asarray(st.kernel_variance, dtype=float)
    total = np.asarray(st.variance, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hajek_share = hajek_term / total
        kernel_share = kernel_term / total
    hajek_share = np.where(total == 0, np.nan, hajek_share)
    kernel_share = np.where(total == 0, np.nan, kernel_share)
    boot = np.asarray(result.bootstrap.statistics, dtype=float)
    critical = {level: float(np.quantile(boot, level)) for level in levels}
    p = result.p_value
    bootstrap_se = sqrt(p * (1 - p) / result.bootstrap_replicates)
    realized = result.realized_subsets
    return SDLDiagnostics(
        realized_budget=realized,
        requested_budget=result.budget,
        realized_budget_ratio=realized / result.budget,
        alpha_n=st.alpha,
        projection_size=result.projection_size,
        blocks_per_observation=st.hajek.blocks_per_observation,
        kernel_order=m,
        constraint_count=result.constraint_count,
        hajek_variance_share=hajek_share,
        kernel_variance_share=kernel_share,
        zero_standard_error=np.asarray(st.standard_error == 0, dtype=bool),
        zero_over_zero=np.asarray(
            (st.standard_error == 0) & (np.asarray(st.u_statistic.value) == 0),
            dtype=bool,
        ),
        nonzero_over_zero=np.asarray(
            (st.standard_error == 0) & (np.asarray(st.u_statistic.value) != 0),
            dtype=bool,
        ),
        bootstrap_standard_error=bootstrap_se,
        bootstrap_critical_values=critical,
    )


def sdl_calibrate(
    sampler: Callable[[np.random.Generator], Sequence[object]],
    hypothesis: SemialgebraicHypothesis,
    *,
    repetitions: int,
    alpha: float = 0.05,
    budget: int,
    bootstrap_replicates: int = 1000,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    **test_options,
) -> SDLCalibrationResult:
    """Estimate null rejection frequency for one SDL configuration by simulation.

    ``sampler(generator)`` must generate one complete i.i.d. sample under the
    null configuration the caller wants to study.  The function by design
    does not infer or simulate from a symbolic hypothesis: calibration is only
    meaningful relative to an explicit data-generating distribution.
    """
    R = positive_integer(repetitions, name="repetitions")
    level = _probability(alpha, name="alpha")
    if not callable(sampler):
        raise TypeError("sampler must be callable")
    generator, seed = generator_from_seed_or_rng(seed, rng)
    streams, _ = spawn_named_generators(generator, ("simulation", "tests"))
    simulation_rng = streams["simulation"]
    test_rng = streams["tests"]

    if "seed" in test_options or "rng" in test_options:
        raise ValueError("seed and rng are controlled by sdl_calibrate")
    p_values = np.empty(R, dtype=float)
    statistics = np.empty(R, dtype=float)
    realized = np.empty(R, dtype=int)
    orders = np.empty(R, dtype=int)
    projection_sizes = np.empty(R, dtype=int)
    for index in range(R):
        sample = tuple(sampler(simulation_rng))
        if not sample:
            raise ValueError("sampler returned an empty sample")
        result = sdl_test(
            sample,
            hypothesis,
            budget=budget,
            bootstrap_replicates=bootstrap_replicates,
            rng=spawn_named_generators(test_rng, ("replicate",))[0]["replicate"],
            **test_options,
        )
        p_values[index] = result.p_value
        statistics[index] = result.statistic
        realized[index] = result.realized_subsets
        orders[index] = result.kernel_order
        projection_sizes[index] = result.projection_size
    rejection_rate = float(np.mean(p_values <= level))
    mcse = sqrt(rejection_rate * (1 - rejection_rate) / R)
    return SDLCalibrationResult(
        alpha=level,
        repetitions=R,
        rejection_rate=rejection_rate,
        monte_carlo_standard_error=mcse,
        p_values=p_values,
        statistics=statistics,
        realized_budgets=realized,
        requested_budget=int(budget),
        kernel_orders=orders,
        projection_sizes=projection_sizes,
        seed=seed,
    )
