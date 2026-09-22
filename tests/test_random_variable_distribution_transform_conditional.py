import sympy as sp

from probstats import (
    Binomial,
    Normal,
    Poisson,
    RandomExpressionLaw,
    RandomVariable,
    StatisticalAssumptions,
    characteristic_function,
    conditional_expectation,
    conditional_variance,
    cumulant_generating_function,
    distribution,
    expectation,
    independent,
    moment_generating_function,
    probability_generating_function,
    total_expectation,
    total_variance,
    variance,
)
from probstats.random_variable_conditional import ConditionalExpectation
from probstats.random_variable_transforms import MomentGeneratingFunction


def test_random_variable_law_metadata_is_instance_local():
    x = RandomVariable("X", distribution=Normal(0, 1))
    y = RandomVariable("X")
    assert x != y
    assert x is not y
    assert isinstance(x.distribution, Normal)
    assert y.distribution is None


def test_distribution_affine_normal_and_independent_normal_sum():
    x = RandomVariable("X", distribution=Normal(2, 3))
    y = RandomVariable("Y", distribution=Normal(5, 4))
    assert distribution(2 * x + 1) == Normal(5, 6)
    ctx = StatisticalAssumptions(independent(x, y))
    assert distribution(x + y, assumptions=ctx) == Normal(7, 5)


def test_distribution_named_discrete_closure_and_symbolic_fallback():
    x = RandomVariable("X", distribution=Poisson(2))
    y = RandomVariable("Y", distribution=Poisson(3))
    ctx = StatisticalAssumptions(independent(x, y))
    assert distribution(x + y, assumptions=ctx) == Poisson(5)

    a = RandomVariable("A", distribution=Binomial(2, sp.Rational(1, 3)))
    b = RandomVariable("B", distribution=Binomial(4, sp.Rational(1, 3)))
    ctx2 = StatisticalAssumptions(independent(a, b))
    assert distribution(a + b, assumptions=ctx2) == Binomial(6, sp.Rational(1, 3))
    assert isinstance(distribution(a * b, assumptions=ctx2), RandomExpressionLaw)


def test_generating_function_affine_and_independent_sum_rules():
    t = sp.Symbol("t", real=True)
    x = RandomVariable("X")
    y = RandomVariable("Y")
    assert moment_generating_function(2 * x + 3, t) == sp.exp(
        3 * t
    ) * MomentGeneratingFunction(x, 2 * t)

    ctx = StatisticalAssumptions(independent(x, y))
    assert moment_generating_function(
        x + y, t, assumptions=ctx
    ) == MomentGeneratingFunction(x, t) * MomentGeneratingFunction(y, t)
    cgf = cumulant_generating_function(x + y, t, assumptions=ctx)
    assert cgf == cumulant_generating_function(x, t) + cumulant_generating_function(
        y, t
    )


def test_generating_functions_use_attached_distribution():
    t = sp.Symbol("t", real=True)
    x = RandomVariable("X", distribution=Normal(1, 2))
    assert sp.simplify(moment_generating_function(x, t) - sp.exp(t + 2 * t**2)) == 0
    assert sp.simplify(characteristic_function(x, t) - sp.exp(sp.I * t - 2 * t**2)) == 0

    n = RandomVariable("N", distribution=Poisson(3))
    z = sp.Symbol("z")
    assert probability_generating_function(n, z) == sp.exp(3 * (z - 1))


def test_conditional_expectation_basic_rules_and_independent_reduction():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    a = sp.Symbol("a", real=True)
    assert conditional_expectation(x, given=x) == x
    assert conditional_expectation(a, given=y) == a
    assert (
        conditional_expectation(a * x + 2, given=y)
        == a * ConditionalExpectation(x, sp.Tuple(y)) + 2
    )
    ctx = StatisticalAssumptions(independent(x, y))
    assert conditional_expectation(x, given=y, assumptions=ctx) == expectation(x)


def test_conditional_variance_and_total_identities():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    a = sp.Symbol("a", real=True)
    ctx = StatisticalAssumptions(independent(x, y))
    assert conditional_variance(x, given=x) == 0
    assert conditional_variance(a * x, given=y, assumptions=ctx) == a**2 * variance(x)
    assert total_expectation(x, given=y) == expectation(x)
    assert total_variance(x, given=y) == variance(x)


def test_unknown_independent_relationship_does_not_factor():
    x = RandomVariable("X")
    y = RandomVariable("Y")
    t = sp.Symbol("t")
    assert moment_generating_function(x + y, t) != moment_generating_function(
        x, t
    ) * moment_generating_function(y, t)
    assert conditional_expectation(x, given=y) == ConditionalExpectation(x, sp.Tuple(y))


def test_conditional_independent_reduces_conditioning_set_and_tower_rule():
    from probstats import conditionally_independent

    x = RandomVariable("X")
    y = RandomVariable("Y")
    z = RandomVariable("Z")
    ctx = StatisticalAssumptions(conditionally_independent(x, y, given=z))
    assert conditional_expectation(
        x, given=(y, z), assumptions=ctx
    ) == ConditionalExpectation(x, sp.Tuple(z))
    inner = conditional_expectation(x, given=y)
    assert expectation(inner) == expectation(x)


def test_total_variance_expanded_identity_shape():
    from probstats.random_variable_algebra import Expectation, Variance
    from probstats.random_variable_conditional import ConditionalVariance

    x = RandomVariable("X")
    y = RandomVariable("Y")
    result = total_variance(x, given=y, expanded=True)
    assert result == Expectation(ConditionalVariance(x, sp.Tuple(y))) + Variance(
        ConditionalExpectation(x, sp.Tuple(y))
    )


def test_transform_factorization_requires_mutual_independence():
    from probstats import mutually_independent, pairwise_independent

    x, y, z = (RandomVariable(name) for name in ("X", "Y", "Z"))
    t = sp.Symbol("t")
    pairwise = StatisticalAssumptions(pairwise_independent(x, y, z))
    mutual = StatisticalAssumptions(mutually_independent(x, y, z))
    assert (
        moment_generating_function(x + y + z, t, assumptions=pairwise).func
        is MomentGeneratingFunction
    )
    assert moment_generating_function(x + y + z, t, assumptions=mutual) == sp.prod(
        MomentGeneratingFunction(v, t) for v in (x, y, z)
    )
