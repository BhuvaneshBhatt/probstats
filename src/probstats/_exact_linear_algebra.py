"""Exact linear-algebra helpers for symbolic probability calculations."""

from __future__ import annotations

import sympy as sp
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.exceptions import DMNonInvertibleMatrixError, DMShapeError


def exact_determinant(matrix) -> sp.Expr:
    """Return an exact determinant using polynomial-domain arithmetic when useful."""
    m = sp.Matrix(matrix)
    if m.rows != m.cols:
        raise ValueError("determinant requires a square matrix")
    if m.rows <= 2:
        return sp.expand(m.det())
    domain_matrix = DomainMatrix.from_Matrix(m)
    if domain_matrix.domain.is_EX:
        return sp.expand(m.det())
    return sp.expand(domain_matrix.domain.to_sympy(domain_matrix.det()))


def exact_solve(matrix, rhs) -> sp.ImmutableDenseMatrix:
    """Solve an exact symbolic linear system without forming an inverse when possible.

    Polynomial and rational systems use ``DomainMatrix.solve_den`` to keep the
    elimination in the exact coefficient domain. General symbolic expressions
    fall back to SymPy's fraction-free Gaussian solver.
    """
    a = sp.Matrix(matrix)
    b = sp.Matrix(rhs)
    if a.rows != a.cols:
        raise ValueError("linear solve requires a square coefficient matrix")
    if b.rows != a.rows:
        raise ValueError("right-hand side has incompatible row dimension")
    domain_a = DomainMatrix.from_Matrix(a)
    domain_b = DomainMatrix.from_Matrix(b)
    domain_a, domain_b = domain_a.unify(domain_b)
    if not domain_a.domain.is_EX:
        try:
            numerator, denominator = domain_a.solve_den(domain_b)
        except (
            DMNonInvertibleMatrixError,
            DMShapeError,
            ValueError,
            ZeroDivisionError,
        ):
            pass
        else:
            den = domain_a.domain.to_sympy(denominator)
            result = numerator.to_Matrix().applyfunc(
                lambda value: sp.cancel(value / den)
            )
            return sp.ImmutableMatrix(result)
    solution, parameters = a.gauss_jordan_solve(b)
    if parameters.rows:
        raise ValueError("linear system does not have a unique solution")
    return sp.ImmutableMatrix(solution)


def quadratic_form(matrix, vector) -> sp.Expr:
    """Return ``vector.T * matrix**-1 * vector`` without materializing the inverse."""
    v = sp.Matrix(vector)
    solved = exact_solve(matrix, v)
    return sp.simplify((v.T * solved)[0])
