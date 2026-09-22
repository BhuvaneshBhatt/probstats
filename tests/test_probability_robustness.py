import pytest
import sympy as sp

from probstats import (
    Binomial,
    Exponential,
    Gamma,
    Normal,
    Poisson,
    RandomVariable,
    StatisticalAssumptions,
    characteristic_function,
    conditionally_independent,
    conditionally_independent_collections,
    cumulant,
    cumulant_generating_function,
    expectation,
    finite_moment,
    independent,
    mean,
    moment_generating_function,
    mutual_information,
    probability_generating_function,
    raw_moment,
    variance,
)
from probstats._transformation_certification import (
    certify_random_expression_bijection,
    transformed_entropy_value,
)
from probstats.functionals import raw_moment as distribution_raw_moment
from probstats.information import information_entropy, kl_divergence
from probstats.random_variable_algebra import Cumulant
from probstats.random_variable_information import MutualInformation
from probstats.transforms import Transform, TransformedDistribution


def _variables():
    return tuple(RandomVariable(name) for name in "XYZW")


def test_conditional_independence_collection_decomposition_and_symmetry():
    x, y, z, w = _variables()
    relation = conditionally_independent_collections(x, (y, w), given=z)
    context = StatisticalAssumptions(relation)
    assert conditionally_independent(x, y, given=z, assumptions=context) is True
    assert conditionally_independent(y, x, given=z, assumptions=context) is True


def test_conditional_independence_weak_union():
    x, y, z, w = _variables()
    context = StatisticalAssumptions(
        conditionally_independent_collections(x, (y, w), given=z)
    )
    assert conditionally_independent(x, y, given=(z, w), assumptions=context) is True


def test_conditional_independence_contraction():
    x, y, z, w = _variables()
    context = StatisticalAssumptions(
        conditionally_independent_collections(x, y, given=z),
        conditionally_independent_collections(x, w, given=(z, y)),
    )
    assert (
        conditionally_independent_collections(x, (y, w), given=z, assumptions=context)
        is True
    )


def test_no_unjustified_intersection_inference():
    x, y, z, w = _variables()
    context = StatisticalAssumptions(
        conditionally_independent(x, y, given=(z, w)),
        conditionally_independent(x, w, given=(z, y)),
    )
    # Intersection requires additional positivity assumptions and is absent.
    assert (
        conditionally_independent_collections(x, (y, w), given=z, assumptions=context)
        is None
    )


@pytest.mark.parametrize(
    "law",
    [Normal(2, 3), Poisson(4), Binomial(7, sp.Rational(2, 5)), Gamma(3, 2)],
)
def test_attached_law_statistics_match_distribution_functionals(law):
    x = RandomVariable("X", distribution=law)
    assert sp.simplify(expectation(x) - mean(law)) == 0
    assert sp.simplify(variance(x) - law.variance_value) == 0
    for order in (1, 2, 3):
        assert (
            sp.simplify(raw_moment(x, order) - distribution_raw_moment(law, order)) == 0
        )
        assert sp.simplify(cumulant(x, order) - law.cumulant(order)) == 0


def test_known_law_transform_round_trip_normal():
    t = sp.symbols("t", real=True)
    x = RandomVariable("X", distribution=Normal(2, 3))
    assert (
        sp.simplify(
            moment_generating_function(x, t)
            - x.distribution.moment_generating_function(t)
        )
        == 0
    )
    assert (
        sp.simplify(
            cumulant_generating_function(x, t)
            - x.distribution.cumulant_generating_function(t)
        )
        == 0
    )
    assert (
        sp.simplify(
            characteristic_function(x, t) - x.distribution.characteristic_function(t)
        )
        == 0
    )


def test_known_law_pgf_round_trip_poisson():
    z = sp.symbols("z")
    x = RandomVariable("X", distribution=Poisson(3))
    assert (
        sp.simplify(
            probability_generating_function(x, z)
            - x.distribution.probability_generating_function(z)
        )
        == 0
    )


def test_finite_moment_known_law_and_unknown_law_semantics():
    known = RandomVariable("K", distribution=Normal(0, 1))
    unknown = RandomVariable("U")
    assert finite_moment(known, 8, assumptions=StatisticalAssumptions()) is True
    assert finite_moment(unknown, 8, assumptions=StatisticalAssumptions()) is None
    assert isinstance(cumulant(unknown, 8), Cumulant)


def _affine_transform(scale=2, shift=3):
    x, y = sp.symbols("x y", real=True)
    return Transform.from_expressions(
        scale * x + shift,
        x,
        (y - shift) / scale,
        y,
        domain=sp.S.Reals,
        codomain=sp.S.Reals,
    )


def test_kl_affine_bijection_metamorphic_oracle():
    transform = _affine_transform(-2, 3)
    p = Normal(-1, 2)
    q = Normal(3, 4)
    pushed_p = TransformedDistribution(p, transform)
    pushed_q = TransformedDistribution(q, transform)
    assert sp.simplify(kl_divergence(pushed_p, pushed_q) - kl_divergence(p, q)) == 0


def test_affine_entropy_change_of_variables():
    transform = _affine_transform(-3, 4)
    p = Normal(2, 5)
    pushed = TransformedDistribution(p, transform)
    assert (
        sp.simplify(information_entropy(pushed) - information_entropy(p) - sp.log(3))
        == 0
    )


def test_mutual_information_accepts_support_restricted_square_bijection():
    x = RandomVariable("X", distribution=Exponential(2))
    y = RandomVariable("Y", distribution=Normal(0, 1))
    result = mutual_information(x**2, sp.exp(y))
    assert result == MutualInformation(x, y)


def test_funcprops_unavailable_falls_back_without_theorem_application(monkeypatch):
    import probstats._transformation_certification as certification

    x = RandomVariable("X", distribution=Normal(0, 1))
    y = RandomVariable("Y", distribution=Normal(0, 1))
    monkeypatch.setattr(certification, "_load_funcprops", lambda: None)
    assert certify_random_expression_bijection(sp.exp(x)) is None
    assert mutual_information(sp.exp(x), sp.exp(y)) == MutualInformation(
        sp.exp(x), sp.exp(y)
    )

    transform = _affine_transform(2, 1)
    pushed = TransformedDistribution(Normal(0, 1), transform)
    assert transformed_entropy_value(pushed) is None


def test_external_closure_registration():
    from dataclasses import dataclass

    from probstats.distributions.base import Distribution
    from probstats.random_variable_distributions import (
        distribution,
        register_affine_closure,
    )

    @dataclass(frozen=True, slots=True)
    class ToyLocationLaw(Distribution):
        location: sp.Expr

        @property
        def support(self):
            return sp.S.Reals

        def logpdf(self, value):
            value = sp.sympify(value)
            return -((value - self.location) ** 2) / 2 - sp.log(sp.sqrt(2 * sp.pi))

    @register_affine_closure(ToyLocationLaw)
    def toy_affine(law, coefficient, offset):
        if coefficient != 1:
            return None
        return ToyLocationLaw(sp.simplify(law.location + offset))

    x = RandomVariable("ToyX", distribution=ToyLocationLaw(sp.Integer(4)))
    assert distribution(x + 7) == ToyLocationLaw(11)


def test_known_nonfinite_moments_propagate_to_standardized_statistics():
    from probstats import coefficient_of_variation, correlation, kurtosis, skewness
    from probstats.distributions import Cauchy

    x = RandomVariable("C", distribution=Cauchy(0, 1))
    assert finite_moment(x, 1, assumptions=StatisticalAssumptions()) is False
    assert finite_moment(x, 2, assumptions=StatisticalAssumptions()) is False
    assert correlation(x, x) is sp.nan
    assert skewness(x) is sp.nan
    assert kurtosis(x) is sp.nan
    assert coefficient_of_variation(x) is sp.nan


@pytest.mark.parametrize("scale", [sp.Integer(3), sp.Integer(-4)])
def test_mutual_information_affine_bijection_invariance(scale):
    x = RandomVariable("MI_X", distribution=Normal(0, 1))
    y = RandomVariable("MI_Y", distribution=Normal(1, 2))
    assert mutual_information(scale * x + 2, sp.exp(y)) == MutualInformation(x, y)


def test_funcprops_unknown_falls_back_without_theorem_application(monkeypatch):
    import probstats._transformation_certification as certification

    class FakeTruthValue:
        TRUE = object()
        UNKNOWN = object()

    class FakeFuncprops:
        TruthValue = FakeTruthValue

        @staticmethod
        def function_range(expression, variable, image_variable, constraints=True):
            return sp.S.Reals

        @staticmethod
        def function_bijective(expression, variable, *, domain, codomain):
            return FakeTruthValue.UNKNOWN

    x = RandomVariable("UnknownX", distribution=Normal(0, 1))
    monkeypatch.setattr(certification, "_load_funcprops", lambda: FakeFuncprops)
    assert certify_random_expression_bijection(2 * x + 1) is None


def test_sum_closure_registration_is_idempotent_and_rule_errors_propagate(monkeypatch):
    import probstats.random_variable_distributions as registry
    from probstats import register_sum_closure

    original = list(registry._SUM_CLOSURES)
    monkeypatch.setattr(registry, "_SUM_CLOSURES", original.copy())

    def rule(laws, coefficients, deterministic):
        raise RuntimeError("rule bug")

    register_sum_closure(rule)
    register_sum_closure(rule)
    assert registry._SUM_CLOSURES.count(rule) == 1

    x = RandomVariable("RegistryX", distribution=Normal(0, 1))
    y = RandomVariable("RegistryY", distribution=Normal(0, 1))
    assumptions = StatisticalAssumptions(independent(x, y))
    # Put the failing external rule first to prove implementation errors are not
    # silently interpreted as an unsupported closure.
    registry._SUM_CLOSURES.remove(rule)
    registry._SUM_CLOSURES.insert(0, rule)
    with pytest.raises(RuntimeError, match="rule bug"):
        registry.distribution(x + y, assumptions=assumptions)


def test_affine_closure_registration_is_idempotent(monkeypatch):
    import probstats.random_variable_distributions as registry
    from probstats import register_affine_closure

    monkeypatch.setattr(registry, "_AFFINE_CLOSURES", registry._AFFINE_CLOSURES.copy())

    class Family:
        pass

    def rule(law, coefficient, offset):
        return None

    decorator = register_affine_closure(Family)
    decorator(rule)
    decorator(rule)
    assert registry._AFFINE_CLOSURES.count((Family, rule)) == 1
