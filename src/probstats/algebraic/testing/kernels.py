"""Unbiased polynomial kernels for semialgebraic hypothesis constraints."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import permutations
from math import factorial

import numpy as np
import sympy as sp

from ._validation import as_generator
from .hypotheses import BasicSemialgebraicNull, SemialgebraicHypothesis


@dataclass(frozen=True, slots=True)
class UnbiasedParameterEstimator:
    """An unbiased estimator of one parameter from ``arity`` i.i.d. observations."""

    function: Callable[..., object]
    arity: int = 1

    def __post_init__(self) -> None:
        if not callable(self.function):
            raise TypeError("estimator function must be callable")
        if not isinstance(self.arity, int) or isinstance(self.arity, bool):
            raise TypeError("estimator arity must be an integer")
        if self.arity < 1:
            raise ValueError("estimator arity must be positive")

    def __call__(self, *observations):
        if len(observations) != self.arity:
            raise ValueError(f"estimator requires exactly {self.arity} observations")
        return self.function(*observations)


@dataclass(frozen=True, slots=True)
class Kernel:
    """A scalar or vector-valued kernel with declared order and symmetrization metadata.

    Exact permutation averaging produces a symmetric kernel. Random permutation
    averaging is an approximate computational construction and does not guarantee
    exact permutation invariance.
    """

    order: int
    function: Callable[..., object]
    dimension: int = 1
    symmetrization: str = "exact"
    permutation_count: int | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.order, int)
            or isinstance(self.order, bool)
            or self.order < 1
        ):
            raise ValueError("kernel order must be a positive integer")
        if not isinstance(self.dimension, int) or self.dimension < 1:
            raise ValueError("kernel dimension must be a positive integer")
        if not callable(self.function):
            raise TypeError("kernel function must be callable")
        if self.symmetrization not in {"exact", "random"}:
            raise ValueError("symmetrization must be 'exact' or 'random'")
        if self.permutation_count is not None and (
            not isinstance(self.permutation_count, int)
            or isinstance(self.permutation_count, bool)
            or self.permutation_count < 1
        ):
            raise ValueError("permutation_count must be a positive integer")

    def __call__(self, *observations):
        if len(observations) != self.order:
            raise ValueError(f"kernel requires exactly {self.order} observations")
        return self.function(*observations)


def _estimator(value) -> UnbiasedParameterEstimator:
    if isinstance(value, UnbiasedParameterEstimator):
        return value
    if callable(value):
        return UnbiasedParameterEstimator(value)
    raise TypeError(
        "parameter estimators must be callables or UnbiasedParameterEstimator objects"
    )


def _constraint_degree(expr: sp.Expr, parameters: tuple[sp.Symbol, ...]) -> int:
    return int(sp.Poly(expr, *parameters).total_degree())


def _unsymmetrized_constraint(
    polynomial: sp.Expr,
    parameters: tuple[sp.Symbol, ...],
    estimators: Mapping[sp.Symbol, UnbiasedParameterEstimator],
    eta: int,
    order: int,
):
    poly = sp.Poly(polynomial, *parameters)
    terms = poly.terms()

    def evaluate(observations: Sequence[object]):
        total = 0
        for powers, coefficient in terms:
            block = 0
            value = coefficient
            for parameter, power in zip(parameters, powers, strict=True):
                estimator = estimators[parameter]
                for _ in range(power):
                    start = block * eta
                    value *= estimator(*observations[start : start + eta])
                    block += 1
            total += value
        return total

    return evaluate


def polynomial_constraint_kernel(
    component: BasicSemialgebraicNull,
    estimators: Mapping[sp.Symbol, object],
    *,
    order: int | None = None,
    symmetrization: str = "exact",
    permutation_count: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> Kernel:
    """Construct an unbiased SDL polynomial kernel for a basic null.

    Exact symmetrization is the default and iterates over permutations lazily, so
    construction does not allocate ``order!`` permutation tuples. Random
    symmetrization fixes a Monte Carlo sample of permutations at construction.

    For estimator arity ``eta``, the default order is
    ``eta * max(deg(f_j))`` as in the SDL kernel construction.  All parameter
    estimators must have the same arity, matching that construction.  A larger
    explicit order is allowed when it is a multiple of ``eta``.
    """
    constraints = component.sdl_constraints
    if not constraints:
        raise ValueError("at least one SDL constraint is required")
    needed = set().union(*(expr.free_symbols for expr in constraints))
    normalized = {
        parameter: _estimator(value) for parameter, value in estimators.items()
    }
    missing = needed.difference(normalized)
    if missing:
        names = ", ".join(sorted(str(symbol) for symbol in missing))
        raise ValueError(f"missing unbiased estimators for parameters: {names}")
    arities = {normalized[parameter].arity for parameter in needed}
    if len(arities) > 1:
        raise ValueError(
            "SDL polynomial-kernel construction requires a common estimator arity"
        )
    eta = next(iter(arities), 1)
    max_degree = max(
        _constraint_degree(expr, component.parameters) for expr in constraints
    )
    minimum_order = max(1, eta * max_degree)
    if order is None:
        order = minimum_order
    if not isinstance(order, int) or isinstance(order, bool):
        raise TypeError("kernel order must be an integer")
    if order < minimum_order:
        raise ValueError(f"kernel order must be at least {minimum_order}")
    if order % eta:
        raise ValueError("kernel order must be a multiple of the estimator arity")

    raw = [
        _unsymmetrized_constraint(expr, component.parameters, normalized, eta, order)
        for expr in constraints
    ]
    full_count = factorial(order)
    if symmetrization == "exact":
        if permutation_count is not None:
            raise ValueError(
                "permutation_count is only valid for random symmetrization"
            )
        selected_permutations = None
        declared_count = full_count
    elif symmetrization == "random":
        if permutation_count is None:
            raise ValueError("random symmetrization requires permutation_count")
        if (
            not isinstance(permutation_count, int)
            or isinstance(permutation_count, bool)
            or permutation_count < 1
        ):
            raise ValueError("permutation_count must be a positive integer")
        generator = as_generator(rng)
        # Draw independent uniform permutations once at construction time.  The
        # resulting approximate kernel is then deterministic for every U-statistic
        # evaluation in that run. Repeated permutations are allowed, matching
        # Monte Carlo averaging over S_m without enumerating the group.
        selected_permutations = tuple(
            tuple(int(i) for i in generator.permutation(order))
            for _ in range(permutation_count)
        )
        declared_count = permutation_count
    else:
        raise ValueError("symmetrization must be 'exact' or 'random'")

    def symmetric(*observations):
        totals = [0] * len(raw)
        permutation_iter = (
            permutations(range(order))
            if selected_permutations is None
            else iter(selected_permutations)
        )
        for permutation in permutation_iter:
            permuted = tuple(observations[index] for index in permutation)
            for index, function in enumerate(raw):
                totals[index] += function(permuted)
        values = tuple(value / declared_count for value in totals)
        return values[0] if len(values) == 1 else values

    return Kernel(order, symmetric, len(raw), symmetrization, declared_count)


def hypothesis_kernel(
    hypothesis: SemialgebraicHypothesis,
    *,
    order: int | None = None,
    symmetrization: str = "exact",
    permutation_count: int | None = None,
    rng: np.random.Generator | int | None = None,
) -> Kernel:
    """Construct the core SDL kernel for a basic semialgebraic hypothesis."""
    return polynomial_constraint_kernel(
        hypothesis.basic_null,
        hypothesis.estimators,
        order=order,
        symmetrization=symmetrization,
        permutation_count=permutation_count,
        rng=rng,
    )
