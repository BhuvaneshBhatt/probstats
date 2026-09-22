"""Constraint-representation utilities for semialgebraic SDL tests."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import sympy as sp

from ._validation import generator_from_seed_or_rng
from .hypotheses import BasicSemialgebraicNull, SemialgebraicHypothesis


@dataclass(frozen=True, slots=True)
class ConstraintAugmentation:
    """A basic null together with generated redundant SDL constraints."""

    component: BasicSemialgebraicNull
    original_constraints: tuple[sp.Expr, ...]
    added_constraints: tuple[sp.Expr, ...]
    weights: np.ndarray
    method: str
    seed: int | None

    @property
    def constraints(self) -> tuple[sp.Expr, ...]:
        return (*self.original_constraints, *self.added_constraints)


def _count(value: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError("count must be an integer")
    value = int(value)
    if value < 0:
        raise ValueError("count must be nonnegative")
    return value


def augment_constraints(
    component: BasicSemialgebraicNull,
    *,
    count: int,
    method: str = "convex",
    concentration: float = 1.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> ConstraintAugmentation:
    """Add redundant random convex combinations of canonical SDL constraints.

    For original constraints ``f_j <= 0``, every generated
    ``sum_j w_j f_j <= 0`` with nonnegative weights summing to one is implied by
    the original system.  Because the originals are retained, augmentation
    leaves the represented null set unchanged by construction.

    The random weights are Dirichlet distributed.  ``concentration`` is the
    common positive Dirichlet concentration parameter.
    """
    if not isinstance(component, BasicSemialgebraicNull):
        raise TypeError("component must be a BasicSemialgebraicNull")
    r = _count(count)
    if method != "convex":
        raise ValueError("method must be 'convex'")
    if not np.isscalar(concentration) or isinstance(concentration, (str, bytes)):
        raise TypeError("concentration must be a positive real number")
    concentration = float(concentration)
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("concentration must be finite and positive")
    generator, seed = generator_from_seed_or_rng(seed, rng)

    constraints = component.sdl_constraints
    if not constraints and r:
        raise ValueError("cannot augment an empty SDL constraint system")
    if r and len(constraints) == 1:
        # The only convex combination of one coordinate is the coordinate itself.
        weights = np.empty((0, 1), dtype=float)
        added = ()
    elif r:
        weights = generator.dirichlet(
            np.full(len(constraints), concentration, dtype=float), size=r
        )
        candidates = tuple(
            sp.expand(
                sum(
                    sp.Float(float(w)) * f
                    for w, f in zip(row, constraints, strict=True)
                )
            )
            for row in weights
        )
        added_list: list[sp.Expr] = []
        kept_rows: list[np.ndarray] = []
        for row, expression in zip(weights, candidates, strict=True):
            existing = (*constraints, *added_list)
            duplicate = any(
                expression == other or sp.expand(expression - other) == 0
                for other in existing
            )
            if not duplicate:
                added_list.append(expression)
                kept_rows.append(row)
        added = tuple(added_list)
        weights = (
            np.stack(kept_rows)
            if kept_rows
            else np.empty((0, len(constraints)), dtype=float)
        )
    else:
        weights = np.empty((0, len(constraints)), dtype=float)
        added = ()

    # Preserve equality provenance while adding the redundant all-inequality
    # coordinates.  The original native inequalities are retained exactly.
    augmented = BasicSemialgebraicNull(
        component.parameters,
        equalities=component.equalities,
        inequalities=(*component.inequalities, *added),
        label=component.label,
        metadata={
            **dict(component.metadata),
            "constraint_augmentation": {
                "method": method,
                "requested_count": r,
                "count": len(added),
                "concentration": concentration,
                "seed": seed,
            },
        },
    )
    return ConstraintAugmentation(augmented, constraints, added, weights, method, seed)


def augment_hypothesis_constraints(
    hypothesis: SemialgebraicHypothesis,
    *,
    count: int,
    method: str = "convex",
    concentration: float = 1.0,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[SemialgebraicHypothesis, ConstraintAugmentation]:
    """Augment the single basic component of a hypothesis."""
    augmentation = augment_constraints(
        hypothesis.basic_null,
        count=count,
        method=method,
        concentration=concentration,
        seed=seed,
        rng=rng,
    )
    augmented = SemialgebraicHypothesis(
        hypothesis.parameters,
        (augmentation.component,),
        ambient=hypothesis.ambient,
        estimators=hypothesis.estimators,
        sample_space=hypothesis.sample_space,
        metadata={
            **dict(hypothesis.metadata),
            "constraint_augmentation": dict(augmentation.component.metadata)[
                "constraint_augmentation"
            ],
        },
    )
    return augmented, augmentation
