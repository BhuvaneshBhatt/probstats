"""Reference trinomial models used in methodological SDL validation studies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import sympy as sp

from .hypotheses import BasicSemialgebraicNull, SemialgebraicHypothesis


@dataclass(frozen=True, slots=True)
class SDLReferenceConfiguration:
    """Published SDL configuration attached to a reference experiment."""

    sample_size: int
    budget: int
    bootstrap_replicates: int
    projection_size: int


TRINOMIAL_PAPER_CONFIGURATION = SDLReferenceConfiguration(
    sample_size=300,
    budget=1000,
    bootstrap_replicates=1000,
    projection_size=300,
)


def _trinomial_symbols() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    return sp.symbols("x y z")


def _coordinate_estimators(parameters: Sequence[sp.Symbol]):
    def coordinate(index: int):
        def estimate(observation):
            return observation[index]

        return estimate

    return {parameter: coordinate(index) for index, parameter in enumerate(parameters)}


def _simplex_ambient(x: sp.Symbol, y: sp.Symbol, z: sp.Symbol) -> sp.Expr:
    return sp.And(sp.Eq(x + y + z, 1), x >= 0, y >= 0, z >= 0)


def trinomial_reference_hypothesis(model: int) -> SemialgebraicHypothesis:
    """Return one of the paper's four trinomial semialgebraic null models.

    The observations are standard basis vectors in R^3, so the coordinate
    functions are unbiased estimators of the trinomial probabilities x, y, z.
    The simplex is represented as the ambient parameter space rather than as
    additional SDL test coordinates.
    """
    if not isinstance(model, Integral) or isinstance(model, bool):
        raise TypeError("model must be an integer from 1 through 4")
    model = int(model)
    if model not in {1, 2, 3, 4}:
        raise ValueError("model must be one of 1, 2, 3, or 4")
    x, y, z = _trinomial_symbols()
    parameters = (x, y, z)
    equalities: tuple[sp.Expr, ...]
    inequalities: tuple[sp.Expr, ...]
    if model == 1:
        equalities = (y - z,)
        inequalities = ()
    elif model == 2:
        equalities = (y - z,)
        inequalities = (sp.Rational(1, 3) - x,)
    elif model == 3:
        equalities = ((y - z) * (x - y) * (x - z),)
        inequalities = ()
    else:
        equalities = ((x - y) * (x - z) * (y - z),)
        inequalities = (
            (x - z) ** 2 * (y - z) ** 2 * (sp.Rational(1, 3) - x),
            (x - y) ** 2 * (y - z) ** 2 * (sp.Rational(1, 3) - y),
            (x - y) ** 2 * (x - z) ** 2 * (sp.Rational(1, 3) - z),
        )
    component = BasicSemialgebraicNull(
        parameters,
        equalities=equalities,
        inequalities=inequalities,
        label=f"trinomial model {model}",
        metadata={"reference": "Barnhill et al. (2025)", "model": model},
    )
    return SemialgebraicHypothesis(
        parameters,
        (component,),
        ambient=_simplex_ambient(x, y, z),
        estimators=_coordinate_estimators(parameters),
        sample_space="trinomial standard-basis observations",
        metadata={"reference": "Barnhill et al. (2025)", "model": model},
    )


def trinomial_model4_components() -> SemialgebraicHypothesis:
    """Return Model 4 decomposed into its three Model-2-like components."""
    x, y, z = _trinomial_symbols()
    parameters = (x, y, z)
    components = (
        BasicSemialgebraicNull(
            parameters,
            equalities=(y - z,),
            inequalities=(sp.Rational(1, 3) - x,),
            label="y = z, x >= 1/3",
        ),
        BasicSemialgebraicNull(
            parameters,
            equalities=(x - z,),
            inequalities=(sp.Rational(1, 3) - y,),
            label="x = z, y >= 1/3",
        ),
        BasicSemialgebraicNull(
            parameters,
            equalities=(x - y,),
            inequalities=(sp.Rational(1, 3) - z,),
            label="x = y, z >= 1/3",
        ),
    )
    return SemialgebraicHypothesis(
        parameters,
        components,
        ambient=_simplex_ambient(x, y, z),
        estimators=_coordinate_estimators(parameters),
        sample_space="trinomial standard-basis observations",
        metadata={
            "reference": "Barnhill et al. (2025)",
            "model": 4,
            "decomposed": True,
        },
    )


def trinomial_sampler(
    probabilities: Sequence[float],
    sample_size: int,
):
    """Return a sampler producing standard-basis trinomial observations."""
    probabilities = np.asarray(tuple(probabilities), dtype=float)
    if probabilities.shape != (3,):
        raise ValueError("probabilities must contain exactly three values")
    if np.any(~np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    if not np.isclose(probabilities.sum(), 1.0):
        raise ValueError("probabilities must sum to one")
    if not isinstance(sample_size, Integral) or isinstance(sample_size, bool):
        raise TypeError("sample_size must be an integer")
    sample_size = int(sample_size)
    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    basis = np.eye(3)

    def sample(generator: np.random.Generator):
        indices = generator.choice(3, size=sample_size, p=probabilities)
        return tuple(basis[indices])

    return sample


def multinomial_point_on_model(model: int, probabilities: Sequence[float]) -> bool:
    """Numerically check whether a trinomial point satisfies a reference null."""
    hypothesis = trinomial_reference_hypothesis(model)
    values = tuple(float(value) for value in probabilities)
    if len(values) != 3:
        return False
    if any(value < -1e-12 for value in values) or not np.isclose(sum(values), 1.0):
        return False
    substitutions = dict(zip(hypothesis.parameters, values, strict=True))
    component = hypothesis.basic_null
    equalities_hold = all(
        abs(float(expr.subs(substitutions))) <= 1e-12 for expr in component.equalities
    )
    inequalities_hold = all(
        float(expr.subs(substitutions)) <= 1e-12 for expr in component.inequalities
    )
    return equalities_hold and inequalities_hold
