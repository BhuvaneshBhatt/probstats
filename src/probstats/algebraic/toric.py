"""Toric statistical models and their sufficient statistics."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import sympy as sp

from ._dependencies import require_toric_semialg
from ._validation import count_vector
from .models import AlgebraicModel
from .tables import ContingencyTable


def _design_matrix(matrix) -> tuple[tuple[int, ...], ...]:
    mat = sp.Matrix(matrix)
    if mat.rows == 0 or mat.cols == 0:
        raise ValueError("design_matrix must have at least one row and one column")
    rows: list[tuple[int, ...]] = []
    for row in range(mat.rows):
        values = []
        for column in range(mat.cols):
            value = sp.sympify(mat[row, column])
            if value.is_Integer is not True:
                raise ValueError("design_matrix entries must be integers")
            values.append(int(value))
        rows.append(tuple(values))
    return tuple(rows)


def _is_homogeneous(matrix: tuple[tuple[int, ...], ...]) -> bool:
    mat = sp.Matrix(matrix)
    ones = sp.ones(1, mat.cols)
    return mat.rank() == mat.col_join(ones).rank()


@dataclass(frozen=True, slots=True, init=False)
class ToricModel(AlgebraicModel):
    """Normalized toric probability model defined by an integer design matrix.

    The columns of the design matrix must lie on an affine hyperplane. This
    homogeneity condition makes the toric parameterization compatible with a
    fixed sample size and ensures Markov moves preserve total count.
    """

    design_matrix: tuple[tuple[int, ...], ...]

    def __init__(
        self,
        design_matrix,
        probabilities: Sequence[sp.Symbol] | None = None,
        *,
        inequalities: Iterable[object] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        matrix = _design_matrix(design_matrix)
        width = len(matrix[0])
        if not _is_homogeneous(matrix):
            raise ValueError(
                "a toric probability model requires a homogeneous design matrix"
            )
        probs = (
            tuple(probabilities)
            if probabilities is not None
            else sp.symbols(f"p0:{width}")
        )
        if len(probs) != width:
            raise ValueError("probabilities must match the design matrix width")
        backend = require_toric_semialg()
        equations = tuple(backend.toric_ideal(matrix, probs))
        params = sp.symbols(f"theta0:{len(matrix)}", positive=True)
        monomials = []
        for column in range(width):
            value = sp.Integer(1)
            for row, parameter in enumerate(params):
                value *= parameter ** matrix[row][column]
            monomials.append(value)
        normalizer = sp.Add(*monomials)
        parameterization = {
            probability: sp.cancel(monomial / normalizer)
            for probability, monomial in zip(probs, monomials, strict=True)
        }
        model_metadata = dict(metadata or {})
        model_metadata.setdefault("family", "toric")
        AlgebraicModel.__init__(
            self,
            probabilities=probs,
            equations=equations,
            inequalities=tuple(inequalities),
            parameters=tuple(params),
            parameterization=parameterization,
            metadata=model_metadata,
        )
        object.__setattr__(self, "design_matrix", matrix)

    def toric_ideal(self) -> tuple[sp.Expr, ...]:
        """Return the homogeneous toric equations, excluding normalization."""
        return self.equations

    def sufficient_statistic(self, counts) -> tuple[int, ...]:
        """Return ``A x`` for a nonnegative count vector ``x``."""
        vector = count_vector(
            counts.flat_counts if isinstance(counts, ContingencyTable) else counts,
            width=len(self.probabilities),
        )
        return tuple(
            sum(
                coefficient * count
                for coefficient, count in zip(row, vector, strict=True)
            )
            for row in self.design_matrix
        )

    def fiber(self, counts):
        """Return the conditional fiber containing ``counts``."""
        from .fibers import Fiber

        vector = count_vector(
            counts.flat_counts if isinstance(counts, ContingencyTable) else counts,
            width=len(self.probabilities),
        )
        return Fiber(self.design_matrix, self.sufficient_statistic(vector))

    def markov_basis(self) -> tuple[tuple[int, ...], ...]:
        """Return an exact Markov basis for the sufficient-statistic fibers."""
        backend = require_toric_semialg()
        return tuple(backend.markov_basis(self.design_matrix))

    def log_linear_parameterization(self) -> Mapping[sp.Symbol, sp.Expr]:
        """Return the normalized monomial probability parameterization."""
        return self.parameterization or {}


def toric_model(design_matrix, probabilities=None, **kwargs) -> ToricModel:
    """Construct a :class:`ToricModel`."""
    return ToricModel(design_matrix, probabilities, **kwargs)


__all__ = ["ToricModel", "toric_model"]
