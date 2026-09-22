import sympy as sp

from probstats.algebraic import (
    AlgebraicModel,
    algebraic_mle,
    critical_points,
    independent_model,
    likelihood_equations,
    maximum_likelihood_degree,
)


def test_full_simplex_likelihood_geometry():
    p = sp.symbols("p0:3")
    model = AlgebraicModel(p)
    system = likelihood_equations(model, (2, 3, 5))
    assert system.likelihood == p[0] ** 2 * p[1] ** 3 * p[2] ** 5
    result = critical_points(model, (2, 3, 5))
    assert result.points == (
        {p[0]: sp.Rational(1, 5), p[1]: sp.Rational(3, 10), p[2]: sp.Rational(1, 2)},
    )


def test_independence_critical_point_and_ml_degree():
    model = independent_model((2, 2))
    point = critical_points(model, (2, 3, 5, 7)).points[0]
    expected = (
        sp.Rational(35, 289),
        sp.Rational(50, 289),
        sp.Rational(84, 289),
        sp.Rational(120, 289),
    )
    assert tuple(point[p] for p in model.probabilities) == expected
    degree = maximum_likelihood_degree(model)
    assert degree.degree == 1
    assert degree.witness_degrees == (1, 1)
    assert degree.generic is True
    assert degree.generic_certified is False


def test_algebraic_mle_handles_zero_count_boundary():
    p = sp.symbols("p0:3")
    model = AlgebraicModel(p)
    result = algebraic_mle(model, (2, 3, 0), include_critical_points=False)
    assert result.mle == {p[0]: sp.Rational(2, 5), p[1]: sp.Rational(3, 5), p[2]: 0}
    assert result.likelihood_value == sp.Rational(108, 3125)
    assert result.complete is True


def test_likelihood_rejects_nonintegral_counts():
    p = sp.symbols("p0:2")
    model = AlgebraicModel(p)
    try:
        likelihood_equations(model, (1, sp.Rational(3, 2)))
    except ValueError as exc:
        assert "nonnegative integer" in str(exc)
    else:
        raise AssertionError("nonintegral likelihood counts must be rejected")
