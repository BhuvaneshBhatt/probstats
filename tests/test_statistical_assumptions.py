import pytest

from probstats import (
    RandomVariable,
    StatisticalAssumptions,
    conditionally_independent,
    iid,
    independent,
    mutually_independent,
    pairwise_independent,
    uncorrelated,
)


def variables():
    return tuple(RandomVariable(name) for name in "XYZW")


def test_independent_is_symmetric_and_queries_are_three_valued():
    x, y, z, _ = variables()
    relation = independent(x, y)
    context = StatisticalAssumptions(relation)

    assert relation == independent(y, x)
    assert independent(x, y, assumptions=context) is True
    assert independent(x, z, assumptions=context) is None


def test_explicit_negative_relation_resolves_false():
    x, y, _, _ = variables()
    context = StatisticalAssumptions(~independent(x, y))

    assert independent(x, y, assumptions=context) is False
    assert independent(y, x, assumptions=context) is False


def test_pairwise_mutual_and_iid_implications_are_one_way():
    x, y, z, _ = variables()

    pairwise = StatisticalAssumptions(pairwise_independent(x, y, z))
    assert independent(x, z, assumptions=pairwise) is True
    assert mutually_independent(x, y, z, assumptions=pairwise) is None

    mutual = StatisticalAssumptions(mutually_independent(x, y, z))
    assert pairwise_independent(x, y, z, assumptions=mutual) is True
    assert independent(y, z, assumptions=mutual) is True
    assert iid(x, y, z, assumptions=mutual) is None

    iid_context = StatisticalAssumptions(iid(x, y, z))
    assert mutually_independent(x, y, z, assumptions=iid_context) is True
    assert pairwise_independent(x, y, z, assumptions=iid_context) is True


def test_individual_pair_relations_can_certify_pairwise_not_mutual():
    x, y, z, _ = variables()
    context = StatisticalAssumptions(
        independent(x, y), independent(x, z), independent(y, z)
    )

    assert pairwise_independent(x, y, z, assumptions=context) is True
    assert mutually_independent(x, y, z, assumptions=context) is None


def test_conditional_independence_symmetry():
    x, y, z, w = variables()
    relation = conditionally_independent(x, y, given=(z, w))
    context = StatisticalAssumptions(relation)

    assert relation == conditionally_independent(y, x, given=(w, z))
    assert conditionally_independent(x, y, given=(z, w), assumptions=context) is True
    assert conditionally_independent(x, z, given=y, assumptions=context) is None


def test_uncorrelated_is_not_independent():
    x, y, _, _ = variables()

    uncorrelated_context = StatisticalAssumptions(uncorrelated(x, y))
    assert independent(x, y, assumptions=uncorrelated_context) is None

    independent_context = StatisticalAssumptions(independent(x, y))
    assert uncorrelated(x, y, assumptions=independent_context) is None


def test_context_is_immutable_and_assume_returns_new_context():
    x, y, z, _ = variables()
    first = StatisticalAssumptions(independent(x, y))
    second = first.assume(independent(x, z))

    assert independent(x, z, assumptions=first) is None
    assert independent(x, z, assumptions=second) is True


def test_contradictory_assumptions_rejected():
    x, y, z, _ = variables()

    with pytest.raises(ValueError, match="contradictory"):
        StatisticalAssumptions(independent(x, y), ~independent(y, x))

    with pytest.raises(ValueError, match="contradictory"):
        StatisticalAssumptions(
            mutually_independent(x, y, z),
            ~independent(x, y),
        )


def test_relations_reject_non_random_variables_and_duplicates():
    x, y, _, _ = variables()

    with pytest.raises(TypeError):
        independent(x, "Y")
    with pytest.raises(ValueError):
        independent(x, x)
    with pytest.raises(ValueError):
        pairwise_independent(x, y, x)
    with pytest.raises(ValueError):
        conditionally_independent(x, y, given=x)


def test_context_storage_is_frozen():
    x, y, _, _ = variables()
    context = StatisticalAssumptions(independent(x, y))

    with pytest.raises(AttributeError):
        context._positive = frozenset()


def test_obsolete_algebraic_independence_names_are_absent():
    from probstats import algebraic

    assert not hasattr(algebraic, "independence_model")
    assert not hasattr(algebraic, "conditional_independence_model")
    assert not hasattr(algebraic, "tensor_independence")
    assert not hasattr(algebraic, "TensorIndependenceResult")
