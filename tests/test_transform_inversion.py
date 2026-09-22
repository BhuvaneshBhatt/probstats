import sympy as sp

from probstats import (
    Bernoulli,
    MultivariateNormal,
    Normal,
    Poisson,
)
from probstats.symbolic import (
    DistributionContext,
    JointBlock,
    TransformInversionDistribution,
    distribution_from_characteristic_function,
    distribution_from_mgf,
    distribution_of,
)
from probstats.transforms import (
    TransformedDistribution,
    invert_transform,
    transformed_distribution,
)


def test_generic_transform_inversion_monotone_exp():
    x = sp.symbols("x", real=True)
    tr = invert_transform(sp.exp(x), x)
    assert (
        sp.simplify(
            tr.invert(sp.Symbol("y", positive=True))
            - sp.log(sp.Symbol("y", positive=True))
        )
        == 0
    )


def test_generic_transform_inversion_multiple_branches_square():
    x = sp.symbols("x", real=True)
    tr = invert_transform(x**2, x)
    assert tr.orientation == 0
    assert len(tr.branches) == 2
    assert tr.codomain == sp.Interval(0, sp.oo)


def test_transformed_distribution_auto_inverse_square_normal():
    x = sp.symbols("x", real=True)
    d = transformed_distribution(Normal(0, 1), x**2, x)
    assert isinstance(d, TransformedDistribution)
    y = sp.symbols("y", positive=True)
    expected = sp.exp(-y / 2) / sp.sqrt(2 * sp.pi * y)
    assert sp.simplify(d.pdf(y) - expected) == 0


def test_dependent_gaussian_linear_combination_includes_covariance():
    x, y = sp.symbols("x y", real=True)
    joint = MultivariateNormal([1, 2], [[4, 3], [3, 9]])
    ctx = DistributionContext.from_joint((x, y), joint)
    d = distribution_of(x + 2 * y + 5, ctx)
    assert isinstance(d, Normal)
    assert sp.simplify(d.mean - 10) == 0
    # var = 4 + 4*9 + 4*3 = 52
    assert sp.simplify(d.sigma**2 - 52) == 0


def test_independent_context_keeps_independent_sum_rule():
    x, y = sp.symbols("x y", real=True)
    ctx = DistributionContext.from_independent({x: Poisson(2), y: Poisson(3)})
    d = distribution_of(x + y, ctx)
    assert d.equivalent_to(Poisson(5))


def test_context_rejects_duplicate_dependency_assignment():
    x, y = sp.symbols("x y")
    joint = JointBlock((x, y), MultivariateNormal([0, 0], [[1, 0], [0, 1]]))
    try:
        DistributionContext({x: Normal(0, 1)}, (joint,))
    except ValueError:
        pass
    else:
        raise AssertionError("expected duplicate-variable rejection")


def test_mgf_inversion_normal_and_poisson():
    t = sp.symbols("t", real=True)
    n = distribution_from_mgf(sp.exp(2 * t + sp.Rational(9, 2) * t**2), t)
    assert n.equivalent_to(Normal(2, 3))
    p = distribution_from_mgf(sp.exp(4 * (sp.exp(t) - 1)), t)
    assert p.equivalent_to(Poisson(4))


def test_cf_inversion_normal_and_bernoulli():
    t = sp.symbols("t", real=True)
    n = distribution_from_characteristic_function(sp.exp(2 * sp.I * t - 8 * t**2), t)
    assert n.equivalent_to(Normal(2, 4))
    b = distribution_from_characteristic_function(
        sp.Rational(2, 3) + sp.exp(sp.I * t) / 3, t
    )
    assert b.equivalent_to(Bernoulli(sp.Rational(1, 3)))


def test_unrecognized_cf_retains_exact_inverse_fourier_law():
    t, x = sp.symbols("t x", real=True)
    phi = 1 / (1 + t**4)
    d = distribution_from_characteristic_function(phi, t)
    assert isinstance(d, TransformInversionDistribution)
    assert d.pdf(x).has(sp.Integral)


def test_mixed_dependent_and_independent_blocks_group_before_convolution():
    x, y, z = sp.symbols("x y z", real=True)
    joint = JointBlock((x, y), MultivariateNormal([1, 2], [[4, 3], [3, 9]]))
    ctx = DistributionContext({z: Normal(10, 5)}, (joint,))
    d = distribution_of(x + y + z, ctx)
    assert isinstance(d, Normal)
    assert sp.simplify(d.mean - 13) == 0
    # Var(x+y)=4+9+2*3=19, plus independent Var(z)=25.
    assert sp.simplify(d.sigma**2 - 44) == 0


def test_nonlinear_joint_coordinate_uses_exact_marginal_then_transform():
    x, y = sp.symbols("x y", real=True)
    ctx = DistributionContext.from_joint(
        (x, y), MultivariateNormal([2, 0], [[9, 2], [2, 4]])
    )
    d = distribution_of(sp.exp(x), ctx)
    from probstats.distributions import LogNormal

    assert d.equivalent_to(LogNormal(2, 3))


def test_mgf_inversion_laplace():
    t = sp.symbols("t", real=True)
    mgf = sp.exp(2 * t) / (1 - 9 * t**2)
    from probstats.distributions import Laplace

    d = distribution_from_mgf(mgf, t)
    assert d.equivalent_to(Laplace(2, 3))


def test_cf_inversion_cauchy():
    t = sp.symbols("t", real=True)
    from probstats.distributions import Cauchy

    d = distribution_from_characteristic_function(
        sp.exp(2 * sp.I * t - 3 * sp.Abs(t)), t
    )
    assert d.equivalent_to(Cauchy(2, 3))
