"""Exact likelihood geometry for algebraic probability models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import sympy as sp

from ._dependencies import require_likelihood_semialg
from ._validation import count_vector, exact_integer
from .models import AlgebraicModel
from .tables import ContingencyTable


def _likelihood(probabilities: Sequence[sp.Symbol], counts: Sequence[int]) -> sp.Expr:
    return sp.prod(
        probability**count
        for probability, count in zip(probabilities, counts, strict=True)
        if count
    )


def _eq_generators(model: AlgebraicModel) -> tuple[sp.Expr, ...]:
    return (*model.equations, model.normalization)


def _projected_critical_equations(
    model: AlgebraicModel,
    counts: Sequence[int],
    *,
    zero_indices: tuple[int, ...] = (),
) -> tuple[tuple[sp.Symbol, ...], tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    zero_set = set(zero_indices)
    active = tuple(
        probability
        for index, probability in enumerate(model.probabilities)
        if index not in zero_set
    )
    equations = (
        *_eq_generators(model),
        *(model.probabilities[i] for i in zero_indices),
    )
    multipliers = sp.symbols(f"lambda0:{len(equations)}")
    stationarity = []
    for index, probability in enumerate(model.probabilities):
        if index in zero_set:
            continue
        gradient = sum(
            multiplier * sp.diff(equation, probability)
            for multiplier, equation in zip(multipliers, equations, strict=True)
        )
        stationarity.append(sp.expand(counts[index] - probability * gradient))
    lagrange = (*equations, *stationarity)
    backend = require_likelihood_semialg()
    variables = (*multipliers, *model.probabilities)
    projected = backend.elimination_ideal_qq(lagrange, variables, multipliers)
    factor = sp.prod(active) if active else sp.Integer(1)
    critical = tuple(backend.saturate_ideal_qq(projected, factor, model.probabilities))
    return tuple(multipliers), tuple(lagrange), critical


@dataclass(frozen=True, slots=True)
class LikelihoodEquationSystem:
    """Polynomial likelihood critical equations in probability coordinates."""

    probabilities: tuple[sp.Symbol, ...]
    counts: tuple[int, ...]
    likelihood: sp.Expr
    equations: tuple[sp.Expr, ...]
    multipliers: tuple[sp.Symbol, ...]
    lagrange_equations: tuple[sp.Expr, ...]
    saturated: bool = True


@dataclass(frozen=True, slots=True)
class CriticalPointResult:
    """Exact finite critical points of a likelihood function."""

    points: tuple[Mapping[sp.Symbol, sp.Expr], ...]
    likelihood_values: tuple[sp.Expr, ...]
    equations: LikelihoodEquationSystem
    real: bool
    positive: bool
    exact: bool
    complete: bool
    backend: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlgebraicMLEResult:
    """Exact global maximum-likelihood result on a semialgebraic model."""

    mle: Mapping[sp.Symbol, sp.Expr] | None
    mle_points: tuple[Mapping[sp.Symbol, sp.Expr], ...]
    likelihood_value: sp.Expr
    critical_points: CriticalPointResult | None
    exact: bool
    complete: bool
    attained: bool
    method: str
    certificate: object | None = None


@dataclass(frozen=True, slots=True)
class MLDegreeResult:
    """Likelihood critical-point count for generic-data witness samples.

    ``generic`` is true only when all requested witness samples give the same
    finite complex critical-point count. This is exact evidence at those
    samples; ``generic_certified`` remains false until a parameter-space
    exceptional-locus certificate is available.
    """

    degree: int
    generic: bool
    generic_certified: bool
    witness_counts: tuple[tuple[int, ...], ...]
    witness_degrees: tuple[int, ...]
    exact: bool
    complete: bool


def likelihood_equations(model: AlgebraicModel, data) -> LikelihoodEquationSystem:
    """Return the multiplier-eliminated, very-affine likelihood equations.

    The logarithmic score equations are polynomialized with Lagrange
    multipliers, the multipliers are eliminated, and the resulting ideal is
    saturated by the product of probability coordinates. Saturation removes
    coordinate-hyperplane components introduced by polynomialization.
    """
    counts = count_vector(
        data.flat_counts if isinstance(data, ContingencyTable) else data,
        width=len(model.probabilities),
        name="data",
    )
    if not any(counts):
        raise ValueError("likelihood data must contain at least one observation")
    multipliers, lagrange, equations = _projected_critical_equations(model, counts)
    return LikelihoodEquationSystem(
        probabilities=model.probabilities,
        counts=counts,
        likelihood=_likelihood(model.probabilities, counts),
        equations=equations,
        multipliers=multipliers,
        lagrange_equations=tuple(lagrange),
    )


def _positive_formula(probabilities: Sequence[sp.Symbol]) -> sp.Expr:
    return sp.And(*(probability > 0 for probability in probabilities))


def critical_points(
    model: AlgebraicModel,
    data,
    *,
    real: bool = True,
    positive: bool = True,
) -> CriticalPointResult:
    """Solve the finite likelihood critical-point system exactly with semialg."""
    system = likelihood_equations(model, data)
    backend = require_likelihood_semialg()
    constraints = _positive_formula(model.probabilities) if real and positive else None
    result = backend.solve_zero_dimensional_system(
        system.equations,
        inequalities=constraints,
        variables=model.probabilities,
        real=real,
        return_result=True,
    )
    points = tuple(result.assignments)
    values = tuple(sp.simplify(system.likelihood.subs(point)) for point in points)
    return CriticalPointResult(
        points=points,
        likelihood_values=values,
        equations=system,
        real=real,
        positive=positive,
        exact=True,
        complete=True,
        backend=result.backend,
        notes=tuple(result.notes),
    )


def _face_constraints(model: AlgebraicModel, zero_indices: tuple[int, ...]) -> sp.Expr:
    zero_set = set(zero_indices)
    atoms: list[sp.Expr] = list(model.inequalities)
    for index, probability in enumerate(model.probabilities):
        atoms.append(sp.Eq(probability, 0) if index in zero_set else probability > 0)
    return sp.And(*atoms) if atoms else sp.true


def _compare_exact(left: sp.Expr, right: sp.Expr) -> int:
    difference = sp.simplify(left - right)
    if difference == 0:
        return 0
    if difference.is_positive is True:
        return 1
    if difference.is_negative is True:
        return -1
    relation = sp.simplify(left > right)
    if relation is sp.true:
        return 1
    if relation is sp.false:
        return -1
    raise NotImplementedError(
        "semialg returned exact likelihood values whose ordering is undecidable"
    )


def algebraic_mle(
    model: AlgebraicModel,
    data,
    *,
    include_critical_points: bool = True,
    max_faces: int = 64,
) -> AlgebraicMLEResult:
    """Return the exact global MLE by solving every relevant simplex face.

    Only coordinate faces associated with zero observed counts can support a
    positive likelihood maximum. Each face critical system is solved and
    feasibility-filtered exactly by semialg. Additional model inequalities are
    supported as filters, but active non-coordinate inequality boundaries are
    outside this solver's certified scope and raise when no certified candidate is
    found.
    """
    counts = count_vector(
        data.flat_counts if isinstance(data, ContingencyTable) else data,
        width=len(model.probabilities),
        name="data",
    )
    if not any(counts):
        raise ValueError("likelihood data must contain at least one observation")
    likelihood = _likelihood(model.probabilities, counts)
    zeroable = tuple(index for index, count in enumerate(counts) if count == 0)
    face_count = 1 << len(zeroable)
    face_limit = exact_integer(max_faces, name="max_faces", minimum=1)
    if face_count > face_limit:
        raise ValueError(
            "too many zero-count coordinate faces for exact MLE enumeration"
        )
    backend = require_likelihood_semialg()
    candidates: list[tuple[Mapping[sp.Symbol, sp.Expr], sp.Expr]] = []
    from itertools import combinations

    for size in range(len(zeroable) + 1):
        for subset in combinations(zeroable, size):
            _, _, equations = _projected_critical_equations(
                model, counts, zero_indices=tuple(subset)
            )
            try:
                result = backend.solve_zero_dimensional_system(
                    equations,
                    inequalities=_face_constraints(model, tuple(subset)),
                    variables=model.probabilities,
                    real=True,
                    return_result=True,
                )
            except (ValueError, NotImplementedError) as exc:
                raise NotImplementedError(
                    "exact MLE requires finite critical sets on the relevant simplex faces"
                ) from exc
            for point in result.assignments:
                value = sp.simplify(likelihood.subs(point))
                candidates.append((point, value))
    if not candidates:
        raise NotImplementedError(
            "no finite feasible likelihood candidates were certified; "
            "models with active non-coordinate inequality boundaries require a broader optimizer"
        )
    best_value = candidates[0][1]
    best_points = [candidates[0][0]]
    for point, value in candidates[1:]:
        comparison = _compare_exact(value, best_value)
        if comparison > 0:
            best_value = value
            best_points = [point]
        elif comparison == 0 and point not in best_points:
            best_points.append(point)
    critical = (
        critical_points(model, counts, real=True, positive=True)
        if include_critical_points
        else None
    )
    points = tuple(best_points)
    return AlgebraicMLEResult(
        mle=points[0] if points else None,
        mle_points=points,
        likelihood_value=best_value,
        critical_points=critical,
        exact=True,
        complete=True,
        attained=True,
        method="semialg_face_critical_enumeration",
        certificate={"faces_checked": face_count, "candidate_count": len(candidates)},
    )


def _generic_witnesses(width: int, samples: int) -> tuple[tuple[int, ...], ...]:
    samples = exact_integer(samples, name="samples", minimum=1)
    if samples < 1:
        raise ValueError("samples must be positive")
    primes = tuple(int(p) for p in list(sp.primerange(2, 1000)))
    needed = width * samples
    if needed > len(primes):
        raise ValueError("too many ML-degree witness samples requested")
    return tuple(
        tuple(primes[sample * width + index] for index in range(width))
        for sample in range(samples)
    )


def maximum_likelihood_degree(
    model: AlgebraicModel,
    *,
    samples: int = 2,
) -> MLDegreeResult:
    """Return the stable exact complex critical-point count at generic witnesses.

    The current implementation uses deterministic positive prime-count witness
    data. Agreement across witnesses supplies reproducible generic evidence but
    is not reported as a symbolic genericity certificate.
    """
    witnesses = _generic_witnesses(len(model.probabilities), samples)
    degrees = []
    for counts in witnesses:
        result = critical_points(model, counts, real=False, positive=False)
        degrees.append(len(result.points))
    stable = len(set(degrees)) == 1
    if not stable:
        raise ArithmeticError(
            "ML-degree witness counts disagree; increase samples or inspect the model"
        )
    return MLDegreeResult(
        degree=degrees[0],
        generic=True,
        generic_certified=False,
        witness_counts=witnesses,
        witness_degrees=tuple(degrees),
        exact=True,
        complete=True,
    )


__all__ = [
    "AlgebraicMLEResult",
    "CriticalPointResult",
    "LikelihoodEquationSystem",
    "MLDegreeResult",
    "algebraic_mle",
    "critical_points",
    "likelihood_equations",
    "maximum_likelihood_degree",
]
