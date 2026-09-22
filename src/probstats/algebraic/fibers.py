"""Sufficient-statistic fibers and Markov-basis sampling."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import exp, lgamma, prod

import numpy as np

from ._dependencies import require_toric_semialg
from ._validation import count_vector, exact_integer


def _matrix(matrix) -> tuple[tuple[int, ...], ...]:
    rows = tuple(
        tuple(exact_integer(value, name="design_matrix value") for value in row)
        for row in matrix
    )
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("design_matrix must be a nonempty rectangular integer matrix")
    return rows


def _log_conditional_weight(table: tuple[int, ...]) -> float:
    return -sum(lgamma(count + 1) for count in table)


@dataclass(frozen=True, slots=True)
class Fiber:
    """Nonnegative integer solutions of ``A x = b``."""

    design_matrix: tuple[tuple[int, ...], ...]
    statistic: tuple[int, ...]

    def __post_init__(self) -> None:
        matrix = _matrix(self.design_matrix)
        statistic = tuple(
            exact_integer(value, name="statistic value") for value in self.statistic
        )
        if len(statistic) != len(matrix):
            raise ValueError("statistic dimension must match design_matrix rows")
        object.__setattr__(self, "design_matrix", matrix)
        object.__setattr__(self, "statistic", statistic)

    @property
    def width(self) -> int:
        return len(self.design_matrix[0])

    def contains(self, counts) -> bool:
        """Return whether ``counts`` is a nonnegative integer point in the fiber."""
        try:
            vector = count_vector(counts, width=self.width)
        except (TypeError, ValueError):
            return False
        return all(
            sum(
                coefficient * count
                for coefficient, count in zip(row, vector, strict=True)
            )
            == target
            for row, target in zip(self.design_matrix, self.statistic, strict=True)
        )

    def coordinate_bounds(self) -> tuple[int, ...]:
        """Return finite coordinate bounds for a nonnegative design matrix.

        Exact enumeration is defined for ordinary nonnegative log-linear design
        matrices. Negative coefficients or an unconstrained column do not give
        safe coordinatewise bounds and are rejected instead of guessing.
        """
        if any(value < 0 for row in self.design_matrix for value in row):
            raise ValueError("fiber enumeration requires a nonnegative design matrix")
        if any(value < 0 for value in self.statistic):
            raise ValueError("fiber enumeration requires a nonnegative statistic")
        bounds = []
        for column in range(self.width):
            candidates = [
                target // row[column]
                for row, target in zip(self.design_matrix, self.statistic, strict=True)
                if row[column] > 0
            ]
            if not candidates:
                raise ValueError("fiber has no finite coordinate bound")
            bounds.append(min(candidates))
        return tuple(bounds)

    def enumerate(
        self, *, max_candidates: int = 1_000_000
    ) -> tuple[tuple[int, ...], ...]:
        """Enumerate the fiber exactly when finite coordinate bounds are available."""
        bounds = self.coordinate_bounds()
        candidate_limit = exact_integer(
            max_candidates, name="max_candidates", minimum=1
        )
        candidate_count = prod(bound + 1 for bound in bounds)
        if candidate_count > candidate_limit:
            raise ValueError(
                f"fiber enumeration would inspect {candidate_count} candidates; "
                f"increase max_candidates to proceed"
            )
        return tuple(
            vector
            for vector in product(*(range(bound + 1) for bound in bounds))
            if self.contains(vector)
        )

    def markov_basis(self) -> tuple[tuple[int, ...], ...]:
        """Return a Markov basis connecting every fiber of the design matrix."""
        backend = require_toric_semialg()
        return tuple(backend.markov_basis(self.design_matrix))


@dataclass(frozen=True, slots=True)
class FiberSample:
    """Samples from a Markov-basis random walk on a fiber."""

    samples: tuple[tuple[int, ...], ...]
    accepted: int
    proposed: int

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.proposed if self.proposed else 0.0


def sample_fiber(
    observed,
    *,
    model=None,
    fiber: Fiber | None = None,
    size: int = 1000,
    burnin: int = 100,
    thin: int = 1,
    rng=None,
    measure: str = "conditional",
) -> FiberSample:
    """Sample a fiber with a Markov-basis Metropolis walk.

    ``measure="conditional"`` targets the exact log-linear conditional mass
    proportional to ``1 / prod(x_i!)``. ``measure="uniform"`` samples the
    fiber uniformly.
    """
    if (model is None) == (fiber is None):
        raise ValueError("provide exactly one of model or fiber")
    size = exact_integer(size, name="size", minimum=1)
    burnin = exact_integer(burnin, name="burnin", minimum=0)
    thin = exact_integer(thin, name="thin", minimum=1)
    if measure not in {"conditional", "uniform"}:
        raise ValueError("measure must be 'conditional' or 'uniform'")
    width = len(model.probabilities) if model is not None else fiber.width
    current = count_vector(observed, width=width)
    active_fiber = model.fiber(current) if fiber is None else fiber
    if not active_fiber.contains(current):
        raise ValueError("observed counts are not in the requested fiber")
    moves = active_fiber.markov_basis()
    if not moves:
        return FiberSample(tuple(current for _ in range(size)), 0, 0)
    generator = np.random.default_rng(rng)
    samples = []
    accepted = 0
    proposed = 0
    steps = burnin + size * thin
    for step in range(steps):
        move = moves[int(generator.integers(len(moves)))]
        sign = 1 if int(generator.integers(2)) else -1
        candidate = tuple(
            value + sign * delta for value, delta in zip(current, move, strict=True)
        )
        proposed += 1
        if all(value >= 0 for value in candidate):
            accept = True
            if measure == "conditional":
                log_ratio = _log_conditional_weight(
                    candidate
                ) - _log_conditional_weight(current)
                accept = log_ratio >= 0 or generator.random() < exp(log_ratio)
            if accept:
                current = candidate
                accepted += 1
        if step >= burnin and (step - burnin) % thin == 0:
            samples.append(current)
    return FiberSample(tuple(samples), accepted, proposed)


__all__ = ["Fiber", "FiberSample", "sample_fiber"]
