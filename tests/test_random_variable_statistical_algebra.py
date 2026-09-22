import pytest
import sympy as sp

from probstats import (
    RandomVariable,
    StatisticalAssumptions,
    central_moment,
    covariance,
    cumulant,
    expectation,
    independent,
    mean,
    mixed_moment,
    moment,
    mutually_independent,
    raw_moment,
    uncorrelated,
    variance,
)
from probstats.random_variable_algebra import (
    CentralMoment,
    Covariance,
    Cumulant,
    Expectation,
    RawMoment,
    Variance,
)


def variables():
    return tuple(RandomVariable(name) for name in "XYZ")


def test_expectation_linearity_and_deterministic_terms():
    x, y, _ = variables()
    a, b, c = sp.symbols("a b c")

    assert expectation(c) == c
    assert expectation(a * x + b * y + c) == a * Expectation(x) + b * Expectation(y) + c
    assert mean(2 * x + 3) == 2 * Expectation(x) + 3


def test_expectation_requires_certified_independence():
    x, y, _ = variables()

    assert expectation(x * y) == Expectation(x * y)

    context = StatisticalAssumptions(independent(x, y))
    assert expectation(x * y, assumptions=context) == Expectation(x) * Expectation(y)
    assert mixed_moment(x, y, assumptions=context) == Expectation(x) * Expectation(y)


def test_independent_product_factorization_extends_to_functions_and_powers():
    x, y, _ = variables()
    f = sp.Function("f")
    context = StatisticalAssumptions(independent(x, y))

    result = expectation(x**2 * f(y), assumptions=context)
    assert result == RawMoment(x, 2) * Expectation(f(y))


def test_three_factor_product_requires_mutual_independence():
    x, y, z = variables()
    pairwise = StatisticalAssumptions(
        independent(x, y), independent(x, z), independent(y, z)
    )
    mutual = StatisticalAssumptions(mutually_independent(x, y, z))

    assert expectation(x * y * z, assumptions=pairwise) == Expectation(x * y * z)
    assert expectation(x * y * z, assumptions=mutual) == Expectation(x) * Expectation(
        y
    ) * Expectation(z)


def test_variance_affine_rule_and_sum_covariance_expansion():
    x, y, _ = variables()
    a, b = sp.symbols("a b", real=True)

    assert variance(a * x + b) == a**2 * Variance(x)
    assert variance(x + y) == Variance(x) + Variance(y) + 2 * Covariance(x, y)


def test_variance_of_independent_sum_drops_covariance():
    x, y, _ = variables()
    context = StatisticalAssumptions(independent(x, y))

    assert variance(x + y, assumptions=context) == Variance(x) + Variance(y)


def test_covariance_is_bilinear_symmetric_and_relationship_aware():
    x, y, _ = variables()
    a, b, c, d = sp.symbols("a b c d", real=True)

    assert covariance(a * x + b, c * y + d) == a * c * Covariance(x, y)
    assert covariance(y, x) == covariance(x, y)
    assert covariance(x, x) == Variance(x)

    assert covariance(x, y, assumptions=independent(x, y)) == 0
    assert covariance(x, y, assumptions=uncorrelated(x, y)) == 0


def test_independent_function_covariance_zero():
    x, y, _ = variables()
    f = sp.Function("f")
    g = sp.Function("g")
    context = StatisticalAssumptions(independent(x, y))

    assert covariance(f(x), g(y), assumptions=context) == 0


def test_raw_moment_expands_affine_expression_exactly():
    x, _, _ = variables()
    a, b = sp.symbols("a b", real=True)

    assert raw_moment(x, 0) == 1
    assert raw_moment(x, 1) == Expectation(x)
    assert raw_moment(x, 3) == RawMoment(x, 3)
    assert sp.expand(moment(a * x + b, 2)) == (
        a**2 * RawMoment(x, 2) + 2 * a * b * Expectation(x) + b**2
    )


def test_central_moment_is_translation_invariant_and_scales_homogeneously():
    x, _, _ = variables()
    a, b = sp.symbols("a b", real=True)

    assert central_moment(x, 0) == 1
    assert central_moment(x, 1) == 0
    assert central_moment(x + b, 4) == CentralMoment(x, 4)
    assert central_moment(a * x + b, 3) == a**3 * CentralMoment(x, 3)
    assert moment(a * x + b, 3, central=True) == a**3 * CentralMoment(x, 3)


def test_mixed_moment_orders_and_validation():
    x, y, _ = variables()
    context = StatisticalAssumptions(independent(x, y))

    assert mixed_moment(x, y, orders=(2, 3), assumptions=context) == RawMoment(
        x, 2
    ) * RawMoment(y, 3)
    with pytest.raises(ValueError, match="orders"):
        mixed_moment(x, y, orders=(2,))
    with pytest.raises(ValueError, match="at least one"):
        mixed_moment()


def test_cumulants_affine_transform_and_add_for_independent_sums():
    x, y, _ = variables()
    a, b = sp.symbols("a b", real=True)
    context = StatisticalAssumptions(independent(x, y))

    assert cumulant(x, 0) == 0
    assert cumulant(x, 2) == Variance(x)
    assert central_moment(x, 2) == Variance(x)
    assert cumulant(a * x + b, 1) == a * Expectation(x) + b
    assert cumulant(a * x + b, 4) == a**4 * Cumulant(x, 4)
    assert cumulant(x + y, 3, assumptions=context) == Cumulant(x, 3) + Cumulant(y, 3)


def test_cumulants_do_not_add_without_required_independent_evidence():
    x, y, z = variables()
    pairwise = StatisticalAssumptions(
        independent(x, y), independent(x, z), independent(y, z)
    )

    assert cumulant(x + y, 3) == Cumulant(x + y, 3)
    assert cumulant(x + y + z, 3, assumptions=pairwise) == Cumulant(x + y + z, 3)


def test_symbolic_orders_are_exact_and_nonnegative():
    x, _, _ = variables()

    assert raw_moment(x, sp.Integer(2)) == RawMoment(x, 2)
    with pytest.raises(TypeError):
        raw_moment(x, sp.Rational(3, 2))
    with pytest.raises(ValueError):
        raw_moment(x, -1)


def test_distribution_and_sample_dispatch_remain_unchanged():
    from probstats import Normal

    normal = Normal(0, 2)
    assert variance(normal) == 4
    assert covariance([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)


def test_assumption_keywords_rejected_for_non_random_inputs():
    with pytest.raises(TypeError, match="assumptions"):
        variance([1, 2, 3], assumptions=StatisticalAssumptions())
