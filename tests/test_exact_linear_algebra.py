import pytest
import sympy as sp

from probstats._exact_linear_algebra import exact_determinant


def test_exact_determinant_matches_sympy_for_polynomial_matrix():
    x, y = sp.symbols("x y")
    matrix = sp.Matrix([[1 + x, y, 2], [x, 2 - y, 3], [4, x + y, 5]])
    assert sp.expand(exact_determinant(matrix) - matrix.det()) == 0


def test_exact_determinant_falls_back_for_expression_domain():
    x = sp.symbols("x")
    matrix = sp.Matrix([[sp.sin(x), 1], [1, x]])
    assert sp.expand(exact_determinant(matrix) - matrix.det()) == 0


def test_exact_determinant_rejects_rectangular_matrix():
    with pytest.raises(ValueError, match="square"):
        exact_determinant([[1, 2, 3], [4, 5, 6]])


def test_exact_solve_matches_symbolic_system_without_forming_inverse():
    from probstats._exact_linear_algebra import exact_solve

    x = sp.symbols("x")
    matrix = sp.Matrix([[x + 2, 1], [1, x + 3]])
    rhs = sp.Matrix([x, 2])
    solution = exact_solve(matrix, rhs)
    assert sp.simplify(matrix * solution - rhs) == sp.zeros(2, 1)


def test_exact_solve_supports_multiple_right_hand_sides():
    from probstats._exact_linear_algebra import exact_solve

    a, b = sp.symbols("a b", nonzero=True)
    matrix = sp.diag(a, b)
    rhs = sp.eye(2)
    solution = exact_solve(matrix, rhs)
    assert sp.simplify(matrix * solution - rhs) == sp.zeros(2)


def test_exact_solve_rejects_nonunique_system():
    from probstats._exact_linear_algebra import exact_solve

    with pytest.raises((ValueError, ZeroDivisionError)):
        exact_solve([[1, 1], [2, 2]], [1, 2])
