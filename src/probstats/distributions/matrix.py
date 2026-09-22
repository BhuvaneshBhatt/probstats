"""Correlation and matrix-valued probability distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import sympy as sp

from .._exact_linear_algebra import exact_determinant, exact_solve
from ..spaces import (
    CorrelationCholeskySpace,
    CorrelationMatrixSpace,
    MatrixEventSpace,
    MeasureType,
    NaturalNumberSpace,
    ParameterSpace,
    PositiveDefiniteMatrixSpace,
    PositiveRealSpace,
)
from .base import Distribution, scalar_batch


def _matrix(value: Any, *, name: str) -> sp.ImmutableDenseMatrix:
    try:
        result = sp.ImmutableMatrix(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be matrix-like") from exc
    if result.rows <= 0 or result.cols <= 0:
        raise ValueError(f"{name} must be nonempty")
    return result


def _validate_pd(value: sp.MatrixBase, *, name: str) -> None:
    if value.rows != value.cols:
        raise ValueError(f"{name} must be square")
    status = value.is_positive_definite
    if status is False:
        raise ValueError(f"{name} must be positive definite")
    if value != value.T and not isinstance(value, sp.MatrixExpr):
        raise ValueError(f"{name} must be symmetric")


def _lkj_log_normalizer(dimension: int, eta: sp.Expr) -> sp.Expr:
    """Log integral of det(R)**(eta-1) over d-dimensional correlation matrices."""
    terms = []
    for k in range(1, dimension):
        m = dimension - k
        a = eta + sp.Rational(m - 1, 2)
        # Duplication-form of the LKJ normalizer.  Keeping beta(1/2, a)
        # explicit is both simpler and much easier for SymPy to recognize.
        terms.append(m * sp.log(sp.beta(sp.Rational(1, 2), a)))
    return sp.simplify(sp.Add(*terms))


@dataclass(frozen=True, slots=True)
class MatrixNormal(Distribution):
    """Matrix normal ``MN(mean, row_covariance, column_covariance)``.

    The covariance of ``vec(X)`` (column-major convention) is
    ``KroneckerProduct(column_covariance, row_covariance)``.
    """

    mean: sp.ImmutableDenseMatrix
    row_covariance: sp.ImmutableDenseMatrix
    column_covariance: sp.ImmutableDenseMatrix

    def __init__(self, mean: Any, row_covariance: Any, column_covariance: Any):
        m = _matrix(mean, name="mean")
        u = _matrix(row_covariance, name="row_covariance")
        v = _matrix(column_covariance, name="column_covariance")
        if u.shape != (m.rows, m.rows):
            raise ValueError("row_covariance dimension must match mean rows")
        if v.shape != (m.cols, m.cols):
            raise ValueError("column_covariance dimension must match mean columns")
        _validate_pd(u, name="row_covariance")
        _validate_pd(v, name="column_covariance")
        object.__setattr__(self, "mean", m)
        object.__setattr__(self, "row_covariance", u)
        object.__setattr__(self, "column_covariance", v)

    @property
    def rows(self) -> int:
        return self.mean.rows

    @property
    def cols(self) -> int:
        return self.mean.cols

    @property
    def support(self):
        return sp.S.UniversalSet

    @property
    def event_space(self):
        return MatrixEventSpace(self.rows, self.cols)

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def parameters(self):
        return (self.mean, self.row_covariance, self.column_covariance)

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("mean", MatrixEventSpace(self.rows, self.cols)),
                ("row_covariance", PositiveDefiniteMatrixSpace(self.rows)),
                ("column_covariance", PositiveDefiniteMatrixSpace(self.cols)),
            )
        )

    @property
    def parameter_space_constraints(self):
        return self.parameter_space.constraints(self.parameters)

    @scalar_batch
    def logpdf(self, value):
        x = _matrix(value, name="value")
        if x.shape != self.mean.shape:
            raise ValueError("matrix value has wrong shape")
        n, p = self.rows, self.cols
        delta = x - self.mean
        row_solved = exact_solve(self.row_covariance, delta)
        q = sp.trace(exact_solve(self.column_covariance, delta.T * row_solved))
        return sp.simplify(
            -sp.Rational(n * p, 2) * sp.log(2 * sp.pi)
            - sp.Rational(p, 2) * sp.log(exact_determinant(self.row_covariance))
            - sp.Rational(n, 2) * sp.log(exact_determinant(self.column_covariance))
            - q / 2
        )

    def _mean(self):
        return self.mean

    def _variance(self):
        return sp.KroneckerProduct(self.column_covariance, self.row_covariance)

    def _entropy(self):
        n, p = self.rows, self.cols
        return sp.simplify(
            sp.Rational(n * p, 2) * (1 + sp.log(2 * sp.pi))
            + sp.Rational(p, 2) * sp.log(exact_determinant(self.row_covariance))
            + sp.Rational(n, 2) * sp.log(exact_determinant(self.column_covariance))
        )


@dataclass(frozen=True, slots=True)
class LKJ(Distribution):
    """LKJ prior over ``dimension`` x ``dimension`` correlation matrices."""

    dimension: int
    eta: sp.Expr

    def __init__(self, dimension: int, eta: Any = 1):
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension < 2
        ):
            raise ValueError("dimension must be an integer >= 2")
        concentration = sp.sympify(eta)
        if concentration.is_positive is False:
            raise ValueError("eta must be positive")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "eta", concentration)

    @property
    def support(self):
        return sp.S.UniversalSet

    @property
    def event_space(self):
        return CorrelationMatrixSpace(self.dimension)

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def parameters(self):
        return (sp.Integer(self.dimension), self.eta)

    @property
    def parameter_constraints(self):
        return self.eta > 0

    @property
    def parameter_space(self):
        return ParameterSpace(
            (
                ("dimension", NaturalNumberSpace(include_zero=False)),
                ("eta", PositiveRealSpace()),
            ),
            relations=(lambda values: sp.sympify(values["dimension"]) >= 2,),
        )

    @property
    def log_normalizing_constant(self):
        return _lkj_log_normalizer(self.dimension, self.eta)

    @scalar_batch
    def logpdf(self, value):
        r = _matrix(value, name="value")
        if r.shape != (self.dimension, self.dimension):
            raise ValueError("correlation matrix has wrong shape")
        return sp.simplify(
            (self.eta - 1) * sp.log(r.det()) - self.log_normalizing_constant
        )

    def _mean(self):
        return sp.eye(self.dimension)

    def _variance(self):
        # Elementwise variances: diagonal entries are fixed at one, while each
        # pairwise correlation has the same symmetric marginal variance.
        offdiag = sp.simplify(1 / (2 * self.eta + self.dimension - 1))
        return sp.ImmutableMatrix(
            self.dimension,
            self.dimension,
            lambda i, j: 0 if i == j else offdiag,
        )

    def _expected_log_determinant(self):
        total = sp.S.Zero
        d = self.dimension
        for i in range(1, d):
            for j in range(i):
                alpha = self.eta + sp.Rational(d - (j + 1) - 1, 2)
                total += sp.digamma(alpha) - sp.digamma(alpha + sp.Rational(1, 2))
        return sp.simplify(total)

    def _entropy(self):
        return sp.simplify(
            self.log_normalizing_constant
            - (self.eta - 1) * self._expected_log_determinant()
        )


@dataclass(frozen=True, slots=True)
class LKJCholesky(Distribution):
    """LKJ law expressed over correlation-matrix Cholesky factors."""

    dimension: int
    eta: sp.Expr

    def __init__(self, dimension: int, eta: Any = 1):
        base = LKJ(dimension, eta)
        object.__setattr__(self, "dimension", base.dimension)
        object.__setattr__(self, "eta", base.eta)

    @property
    def support(self):
        return sp.S.UniversalSet

    @property
    def event_space(self):
        return CorrelationCholeskySpace(self.dimension)

    @property
    def measure_type(self):
        return MeasureType.CONTINUOUS

    @property
    def parameters(self):
        return (sp.Integer(self.dimension), self.eta)

    @property
    def parameter_constraints(self):
        return self.eta > 0

    @property
    def parameter_space(self):
        return LKJ(self.dimension, self.eta).parameter_space

    @property
    def log_normalizing_constant(self):
        return _lkj_log_normalizer(self.dimension, self.eta)

    @scalar_batch
    def logpdf(self, value):
        chol = _matrix(value, name="value")
        if chol.shape != (self.dimension, self.dimension):
            raise ValueError("Cholesky factor has wrong shape")
        powers = sp.Add(
            *(
                (self.dimension - (i + 1) + 2 * self.eta - 2) * sp.log(chol[i, i])
                for i in range(1, self.dimension)
            )
        )
        return sp.simplify(powers - self.log_normalizing_constant)

    def _partial_alpha(self, column: int):
        # column is zero-based and indexes a partial correlation.
        return self.eta + sp.Rational(self.dimension - column - 2, 2)

    def _mean(self):
        d = self.dimension
        out = sp.zeros(d)
        out[0, 0] = 1
        for i in range(1, d):
            diagonal_mean = sp.S.One
            for j in range(i):
                a = self._partial_alpha(j)
                diagonal_mean *= sp.beta(
                    sp.Rational(1, 2), a + sp.Rational(1, 2)
                ) / sp.beta(sp.Rational(1, 2), a)
            out[i, i] = sp.simplify(diagonal_mean)
        return sp.ImmutableMatrix(out)

    def _variance(self):
        d = self.dimension
        means = self._mean()
        out = sp.zeros(d)
        for i in range(1, d):
            prefix_second = sp.S.One
            for j in range(i):
                a = self._partial_alpha(j)
                z2 = sp.simplify(1 / (2 * a + 1))
                out[i, j] = sp.simplify(z2 * prefix_second)
                prefix_second *= sp.simplify(2 * a / (2 * a + 1))
            out[i, i] = sp.simplify(prefix_second - means[i, i] ** 2)
        return sp.ImmutableMatrix(out)

    def _entropy(self):
        d = self.dimension
        log_density_term = sp.S.Zero
        for i in range(1, d):
            power = d - (i + 1) + 2 * self.eta - 2
            expected_log_diag = sp.S.Zero
            for j in range(i):
                a = self._partial_alpha(j)
                expected_log_diag += sp.Rational(1, 2) * (
                    sp.digamma(a) - sp.digamma(a + sp.Rational(1, 2))
                )
            log_density_term += power * expected_log_diag
        return sp.simplify(self.log_normalizing_constant - log_density_term)

    def correlation_distribution(self) -> LKJ:
        return LKJ(self.dimension, self.eta)


__all__ = ["LKJ", "LKJCholesky", "MatrixNormal"]
