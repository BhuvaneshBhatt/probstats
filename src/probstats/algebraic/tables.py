"""Finite contingency tables for algebraic statistics."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp


def _array(value) -> sp.ImmutableDenseNDimArray:
    arr = sp.Array(value)
    if not arr.shape:
        raise ValueError("a contingency table requires at least one axis")
    return sp.ImmutableDenseNDimArray(arr)


@dataclass(frozen=True, slots=True)
class ContingencyTable:
    """Immutable finite table of nonnegative integer counts."""

    counts: object
    variable_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        counts = _array(self.counts)
        values = tuple(
            counts[index] for index in product(*(range(dim) for dim in counts.shape))
        )
        if any(not value.is_Integer or value < 0 for value in values):
            raise ValueError("contingency-table counts must be nonnegative integers")
        names = tuple(self.variable_names) or tuple(
            f"X{i}" for i in range(len(counts.shape))
        )
        if len(names) != len(counts.shape) or len(set(names)) != len(names):
            raise ValueError("variable_names must be distinct and match table rank")
        object.__setattr__(self, "counts", counts)
        object.__setattr__(self, "variable_names", names)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.counts.shape)

    @property
    def flat_counts(self) -> tuple[int, ...]:
        """Return counts in the canonical row-major probability-coordinate order."""
        return tuple(
            int(self.counts[index])
            for index in product(*(range(dim) for dim in self.shape))
        )

    @property
    def total(self) -> sp.Integer:
        return sp.Integer(
            sum(
                self.counts[index]
                for index in product(*(range(dim) for dim in self.shape))
            )
        )

    def marginal(self, *variables: str):
        """Return counts marginalized to the requested variables."""
        if not variables:
            return self.total
        keep = tuple(self.variable_names.index(name) for name in variables)
        if len(set(keep)) != len(keep):
            raise ValueError("marginal variables must be distinct")
        shape = tuple(self.shape[axis] for axis in keep)
        out = []
        for target in product(*(range(dim) for dim in shape)):
            total = 0
            for source in product(*(range(dim) for dim in self.shape)):
                if all(
                    source[axis] == value
                    for axis, value in zip(keep, target, strict=True)
                ):
                    total += self.counts[source]
            out.append(total)
        return sp.ImmutableDenseNDimArray(out, shape)

    def marginals(self) -> dict[str, sp.ImmutableDenseNDimArray]:
        """Return all one-variable marginal count arrays."""
        return {name: self.marginal(name) for name in self.variable_names}
