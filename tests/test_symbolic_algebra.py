import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    Exponential,
    MultivariateNormal,
    Normal,
    Poisson,
    Uniform,
)
from probstats.symbolic import (
    characteristic_function,
    convolution,
    distribution_of,
    moment_generating_function,
    order_statistic,
    probability_generating_function,
    recognize_affine_matrix,
)


def test_transform_functions_closed_forms():
    t, z = sp.symbols("t z", real=True)
    assert (
        sp.simplify(
            moment_generating_function(Normal(2, 3), t)
            - sp.exp(2 * t + sp.Rational(9, 2) * t**2)
        )
        == 0
    )
    assert (
        sp.simplify(
            characteristic_function(Poisson(4), t) - sp.exp(4 * (sp.exp(sp.I * t) - 1))
        )
        == 0
    )
    assert (
        sp.expand(probability_generating_function(Bernoulli(sp.Rational(1, 3)), z))
        == sp.Rational(2, 3) + z / 3
    )


def test_convolution_prefers_closed_family():
    d = convolution(Poisson(2), Poisson(5))
    assert isinstance(d, Poisson) and d.rate == 7


def test_order_statistic_standard_uniform_beta():
    d = order_statistic(Uniform(0, 1), 5, 2)
    assert isinstance(d, Beta) and d.alpha == 2 and d.beta == 4


def test_generic_order_statistic_cdf():
    d = order_statistic(Exponential(2), 3, 1)
    x = sp.symbols("x", positive=True)
    assert sp.simplify(d.cdf(x) - (1 - sp.exp(-6 * x))) == 0


def test_affine_matrix_mvn():
    d = MultivariateNormal([1, 2], [[2, 1], [1, 3]])
    out = recognize_affine_matrix(d, [[1, 1]], [4])
    assert isinstance(out, MultivariateNormal)
    assert out.mean == sp.ImmutableMatrix([7])
    assert out.covariance == sp.ImmutableMatrix([[7]])


def test_distribution_of_expression_rules():
    x, y = sp.symbols("x y")
    d = distribution_of(2 * x + 3, {x: Normal(1, 4)})
    assert d.equivalent_to(Normal(5, 8))
    s = distribution_of(x + y, {x: Poisson(2), y: Poisson(3)})
    assert s.equivalent_to(Poisson(5))
    e = distribution_of(sp.exp(x), {x: Normal(2, 3)})
    from probstats.distributions import LogNormal

    assert e.equivalent_to(LogNormal(2, 3))


def test_uniform_characteristic_function_uses_sympy_sinc_convention():
    from probstats import Uniform
    from probstats.symbolic import characteristic_function

    t = sp.symbols("t", real=True)
    cf = characteristic_function(Uniform(-2, 4), t)
    expected = sp.exp(sp.I * t) * sp.sinc(3 * t)
    assert sp.simplify(cf - expected) == 0
    assert cf.subs(t, 0) == 1


def test_logistic_characteristic_function_is_one_at_zero():
    from probstats.distributions import Logistic
    from probstats.symbolic import characteristic_function

    t = sp.symbols("t", real=True)
    cf = characteristic_function(Logistic(2, 3), t)
    assert cf.subs(t, 0) == 1


def test_order_statistic_rejects_fractional_integer_parameters():
    import pytest

    from probstats import Normal
    from probstats.symbolic import order_statistic

    with pytest.raises(TypeError, match="sample_size"):
        order_statistic(Normal(0, 1), 3.5, 1)
    with pytest.raises(TypeError, match="rank"):
        order_statistic(Normal(0, 1), 3, 1.5)


def test_multivariate_transforms_sum_only_finite_support_points():
    import sympy as sp

    from probstats import JointDistribution
    from probstats.symbolic import (
        multivariate_characteristic_function,
        multivariate_moment_generating_function,
    )

    x, y = sp.symbols("x y", integer=True)
    joint = JointDistribution(
        (x, y),
        sp.Rational(1, 4),
        (sp.FiniteSet(0, 2), sp.FiniteSet(0, 1)),
        discrete=True,
    )
    assert multivariate_characteristic_function(joint, (0, 0)) == 1
    assert multivariate_moment_generating_function(joint, (0, 0)) == 1
