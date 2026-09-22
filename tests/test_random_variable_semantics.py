import pytest
import sympy as sp

from probstats import (
    Gamma,
    Normal,
    Poisson,
    RandomVariable,
    StatisticalAssumptions,
    bayes_probability,
    coefficient_of_variation,
    conditional_covariance,
    conditional_entropy,
    conditional_expectation,
    conditional_moment,
    conditional_probability,
    conditional_variance,
    correlation,
    cumulant,
    delta_method,
    delta_variance,
    distribution,
    entropy_chain_rule,
    event_complement,
    event_intersection,
    event_union,
    events_independent,
    factorial_moment,
    finite_moment,
    identically_distributed,
    iid,
    inclusion_exclusion,
    independent,
    independent_collections,
    joint_entropy,
    kurtosis,
    mean,
    moment_generating_function,
    mutual_information,
    mutually_independent,
    probability_generating_function,
    raw_moment,
    skewness,
    standardized_moment,
    taylor_expectation,
    total_covariance,
    total_probability,
    variance,
)
from probstats.events import PredicateEvent
from probstats.random_variable_algebra import Covariance, Variance
from probstats.random_variable_complex import ComplexCovariance, complex_covariance
from probstats.random_variable_conditional import ConditionalExpectation
from probstats.random_variable_information import InformationEntropy
from probstats.random_variable_transforms import ProbabilityGeneratingFunction


def test_pairwise_independence_not_joint():
    x, y, z = map(RandomVariable, "XYZ")
    ctx = StatisticalAssumptions(
        independent(x, y), independent(x, z), independent(y, z)
    )
    assert conditional_expectation(
        x, given=(y, z), assumptions=ctx
    ) == ConditionalExpectation(x, sp.Tuple(y, z))
    strong = StatisticalAssumptions(mutually_independent(x, y, z))
    assert conditional_expectation(x, given=(y, z), assumptions=strong) == mean(x)


def test_collection_independence_is_stronger_than_pairwise_shortcut():
    x, y, z = map(RandomVariable, "XYZ")
    relation = independent_collections((x,), (y, z))
    ctx = StatisticalAssumptions(relation)
    assert independent_collections((x,), (y, z), assumptions=ctx) is True
    assert independent(x, y, assumptions=ctx) is True
    assert independent(x, z, assumptions=ctx) is True


def test_real_covariance_and_variance_do_not_apply_complex_scaling():
    x, y = map(RandomVariable, "XY")
    assert variance(sp.I * x) == Variance(sp.I * x)
    expected = Covariance(y, sp.I * x)
    from probstats import covariance as cov

    assert cov(sp.I * x, y) == expected
    a = sp.Symbol("a", real=True)
    assert variance(a * x) == a**2 * Variance(x)


def test_hermitian_covariance_canonicalizes_reverse_order():
    x, y = map(RandomVariable, "XY")
    assert complex_covariance(y, x) == sp.conjugate(ComplexCovariance(x, y))


def test_attached_law_is_authoritative_for_core_statistics():
    x = RandomVariable("X", Normal(2, 3))
    assert mean(x) == 2
    assert variance(x) == 9
    assert raw_moment(x, 2) == 13
    assert cumulant(x, 3) == 0


def test_iid_validates_attached_laws_and_entails_identical_distribution():
    x = RandomVariable("X", Normal(0, 1))
    y = RandomVariable("Y", Normal(0, 1))
    ctx = StatisticalAssumptions(iid(x, y))
    assert identically_distributed(x, y, assumptions=ctx) is True
    assert independent(x, y, assumptions=ctx) is True
    z = RandomVariable("Z", Poisson(1))
    with pytest.raises(ValueError):
        iid(x, z)


def test_finite_moment_predicate_has_higher_to_lower_entailment():
    x = RandomVariable("X")
    ctx = StatisticalAssumptions(finite_moment(x, 4))
    assert finite_moment(x, 2, assumptions=ctx) is True


def test_conditional_measurable_factor_rules_and_moments():
    x, y = map(RandomVariable, "XY")
    a = sp.Symbol("a", real=True)
    assert conditional_expectation(y * x, given=y) == y * conditional_expectation(
        x, given=y
    )
    assert conditional_variance(y * x, given=y) == y**2 * conditional_variance(
        x, given=y
    )
    assert conditional_moment(x, 2, given=y, central=True) == conditional_variance(
        x, given=y
    )
    assert conditional_variance(a * x, given=y) == a**2 * conditional_variance(
        x, given=y
    )


def test_conditional_covariance_and_total_covariance_identity():
    x, y, z = map(RandomVariable, "XYZ")
    expanded = total_covariance(x, y, given=z, expanded=True)
    assert expanded == conditional_expectation(
        conditional_covariance(x, y, given=z), given=z
    ) or expanded.has(conditional_covariance(x, y, given=z))
    assert total_covariance(x, y, given=z) == Covariance(x, y)


def test_standardized_statistics_affine_identities():
    x = RandomVariable("X")
    a = sp.Symbol("a", positive=True, real=True)
    b = sp.Symbol("b", real=True)
    assert sp.simplify(correlation(a * x + b, x) - 1) == 0
    assert standardized_moment(a * x + b, 3) == standardized_moment(x, 3)
    assert skewness(a * x + b) == skewness(x)
    assert kurtosis(a * x + b) == kurtosis(x)
    assert coefficient_of_variation(x) == sp.sqrt(Variance(x)) / mean(x)


def test_factorial_moment_uses_known_discrete_law():
    x = RandomVariable("X", Poisson(3))
    assert factorial_moment(x, 2) == 9


def test_deterministic_transform_rules():
    t, z = sp.symbols("t z", real=True)
    assert moment_generating_function(3, t) == sp.exp(3 * t)
    assert probability_generating_function(3, z) == z**3
    assert probability_generating_function(
        sp.Rational(1, 2), z
    ) == ProbabilityGeneratingFunction(sp.Rational(1, 2), z)


def test_distribution_registry_gamma_reproductive_closure():
    x = RandomVariable("X", Gamma(2, 3))
    y = RandomVariable("Y", Gamma(4, 3))
    ctx = StatisticalAssumptions(independent(x, y))
    assert distribution(x + y, assumptions=ctx) == Gamma(6, 3)


def test_information_algebra_independence_and_chain_rule():
    x, y = map(RandomVariable, "XY")
    ctx = StatisticalAssumptions(independent(x, y))
    assert mutual_information(x, y, assumptions=ctx) == 0
    assert joint_entropy(x, y, assumptions=ctx) == InformationEntropy(
        x
    ) + InformationEntropy(y)
    assert conditional_entropy(x, given=y, assumptions=ctx) == InformationEntropy(x)
    assert entropy_chain_rule(x, y) == InformationEntropy(x) + conditional_entropy(
        y, given=x
    )


def test_delta_method_structured_semantics():
    x = RandomVariable("X")
    assert delta_variance(sp.exp(x)) == sp.exp(2 * mean(x)) * variance(x)
    result = delta_method(sp.exp(x))
    assert result.derivative == sp.exp(mean(x))
    assert result.asymptotic_variance == sp.exp(2 * mean(x)) * variance(x)
    approx = taylor_expectation(sp.exp(x), order=2, return_result=True)
    assert approx.order == 2
    assert approx.required_moments == (2,)


def test_event_algebra_and_probability_identities_on_uniform():
    from probstats import Uniform

    x = sp.Symbol("x", real=True)
    dist = Uniform(0, 1)
    a = PredicateEvent(x < sp.Rational(1, 2), x)
    b = PredicateEvent(x > sp.Rational(1, 4), x)
    assert event_intersection(a, b).predicate == sp.And(
        x > sp.Rational(1, 4), x < sp.Rational(1, 2)
    )
    assert event_union(a, b).predicate == sp.Or(
        x > sp.Rational(1, 4), x < sp.Rational(1, 2)
    )
    assert event_complement(a).predicate == (x >= sp.Rational(1, 2))
    assert conditional_probability(dist, a, given=b, variable=x) == sp.Rational(1, 3)
    assert bayes_probability(dist, a, given=b, variable=x) == sp.Rational(1, 3)
    partition = (
        PredicateEvent(x < sp.Rational(1, 2), x),
        PredicateEvent(x >= sp.Rational(1, 2), x),
    )
    assert total_probability(dist, b, partition=partition, variable=x) == sp.Rational(
        3, 4
    )
    assert inclusion_exclusion(dist, a, b, variable=x) == 1
    assert events_independent(dist, a, b, variable=x) is False


def test_mutual_information_uses_funcprops_bijectivity_certification(monkeypatch):
    import sys
    import types

    import sympy as sp

    from probstats import Normal, RandomVariable, mutual_information

    class TruthValue:
        TRUE = object()
        FALSE = object()
        UNKNOWN = object()

    calls = []
    module = types.ModuleType("funcprops")
    module.TruthValue = TruthValue

    def function_range(expr, var, image, *, constraints=True, **kwargs):
        calls.append(("range", expr, var, constraints))
        return image > 0 if expr == sp.exp(var) else sp.S.Reals

    def function_bijective(expr, var, *, domain=True, codomain=sp.S.Reals, **kwargs):
        calls.append(("bijective", expr, var, domain, codomain))
        return TruthValue.TRUE if expr == sp.exp(var) else TruthValue.FALSE

    module.function_range = function_range
    module.function_bijective = function_bijective
    monkeypatch.setitem(sys.modules, "funcprops", module)

    x = RandomVariable("X_bij", Normal(0, 1))
    y = RandomVariable("Y_bij", Normal(0, 1))

    assert mutual_information(sp.exp(x), sp.exp(y)) == mutual_information(x, y)
    assert any(call[0] == "bijective" for call in calls)


def test_mutual_information_keeps_uncertified_transform_formal(monkeypatch):
    import sys
    import types

    from probstats import Normal, RandomVariable, mutual_information
    from probstats.random_variable_information import MutualInformation

    class TruthValue:
        TRUE = object()
        FALSE = object()
        UNKNOWN = object()

    module = types.ModuleType("funcprops")
    module.TruthValue = TruthValue
    module.function_range = lambda expr, var, image, **kwargs: image >= 0
    module.function_bijective = lambda *args, **kwargs: TruthValue.FALSE
    monkeypatch.setitem(sys.modules, "funcprops", module)

    x = RandomVariable("X_nonbij", Normal(0, 1))
    y = RandomVariable("Y_nonbij", Normal(0, 1))
    result = mutual_information(x**2, y)
    assert isinstance(result, MutualInformation)
    assert result.has(x**2)
