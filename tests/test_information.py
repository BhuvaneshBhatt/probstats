import math

import pytest
import sympy as sp

from probstats import (
    Bernoulli,
    Normal,
    Uniform,
)
from probstats.information import (
    InformationMethod,
    SupportRelation,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    cross_entropy,
    hellinger_distance,
    hellinger_squared,
    information_entropy,
    jensen_shannon_divergence,
    kl_divergence,
    renyi_divergence,
    support_subset,
)


def test_kl_identical_distribution_is_exact_zero():
    p = Normal(2, 3)
    result = kl_divergence(p, p, return_result=True)
    assert result.value == 0
    assert result.method is InformationMethod.CLOSED_FORM
    assert result.support_relation is SupportRelation.PROVEN


def test_normal_kl_matches_closed_form():
    p = Normal(0, 1)
    q = Normal(1, 2)
    expected = sp.log(2) + (1 + 1) / sp.Integer(8) - sp.Rational(1, 2)
    assert sp.simplify(kl_divergence(p, q) - expected) == 0


def test_bernoulli_kl_and_cross_entropy_relationship():
    p = Bernoulli(sp.Rational(1, 4))
    q = Bernoulli(sp.Rational(3, 4))
    kl = kl_divergence(p, q)
    ce = cross_entropy(p, q)
    h = information_entropy(p)
    assert sp.simplify(ce - h - kl) == 0
    assert sp.simplify(p.cross_entropy(q) - ce) == 0


def test_support_mismatch_makes_relative_information_infinite():
    p = Uniform(0, 2)
    q = Uniform(0, 1)
    assert support_subset(p, q) is SupportRelation.DISPROVEN
    assert kl_divergence(p, q) is sp.oo
    assert cross_entropy(p, q) is sp.oo
    assert renyi_divergence(p, q, 2) is sp.oo


def test_support_containment_allows_uniform_kl():
    p = Uniform(0, 1)
    q = Uniform(0, 2)
    assert support_subset(p, q) is SupportRelation.PROVEN
    assert kl_divergence(p, q) == sp.log(2)


def test_renyi_order_one_is_kl_and_order_two_matches_bernoulli_identity():
    p = Bernoulli(sp.Rational(1, 4))
    q = Bernoulli(sp.Rational(3, 4))
    assert renyi_divergence(p, q, 1) == kl_divergence(p, q)
    expected = sp.log(sp.Rational(7, 3))
    assert sp.simplify(renyi_divergence(p, q, 2) - expected) == 0


def test_bhattacharyya_hellinger_relationships():
    p = Bernoulli(sp.Rational(1, 4))
    q = Bernoulli(sp.Rational(3, 4))
    bc = bhattacharyya_coefficient(p, q)
    hs = hellinger_squared(p, q)
    h = hellinger_distance(p, q)
    bd = bhattacharyya_distance(p, q)
    assert bc == sp.sqrt(3) / 2
    assert sp.simplify(hs - (1 - bc)) == 0
    assert sp.simplify(h**2 - hs) == 0
    assert sp.simplify(bd + sp.log(bc)) == 0


def test_bhattacharyya_uses_overlap_not_nominal_pdf_outside_support():
    p = Uniform(0, 2)
    q = Uniform(1, 3)
    assert bhattacharyya_coefficient(p, q) == sp.Rational(1, 2)


def test_jensen_shannon_is_symmetric_at_half_weight_and_bounded():
    p = Bernoulli(sp.Rational(1, 4))
    q = Bernoulli(sp.Rational(3, 4))
    pq = jensen_shannon_divergence(p, q)
    qp = jensen_shannon_divergence(q, p)
    assert sp.simplify(pq - qp) == 0
    assert pq.is_nonnegative is True
    assert sp.simplify(sp.log(2) - pq).is_nonnegative is True


def test_numerical_fallback_for_normal_js_is_reproducible_and_reasonable():
    p = Normal(0, 1)
    q = Normal(1, 1)
    result = jensen_shannon_divergence(
        p, q, numerical_fallback=True, samples=6000, rng=1234, return_result=True
    )
    assert result.method is InformationMethod.MONTE_CARLO
    assert 0 < result.value < math.log(2)
    again = jensen_shannon_divergence(
        p, q, numerical_fallback=True, samples=6000, rng=1234
    )
    assert result.value == again


def test_distribution_information_measure_methods_delegate_to_public_api():
    p = Normal(0, 1)
    q = Normal(1, 2)
    assert p.kl_divergence(q) == kl_divergence(p, q)
    assert p.renyi_divergence(q, 2) == renyi_divergence(p, q, 2)
    assert p.hellinger_distance(q) == hellinger_distance(p, q)
    assert p.bhattacharyya_distance(q) == bhattacharyya_distance(p, q)


def test_renyi_half_is_bhattacharyya_distance_relationship():
    p = Bernoulli(sp.Rational(1, 5))
    q = Bernoulli(sp.Rational(3, 5))
    rhalf = renyi_divergence(p, q, sp.Rational(1, 2))
    bd = bhattacharyya_distance(p, q)
    assert sp.simplify(sp.exp(rhalf - 2 * bd)) == 1


def test_entropy_api_agrees_with_existing_distribution_entropy():
    p = Normal(0, 2)
    assert sp.simplify(information_entropy(p) - p.entropy()) == 0


def test_numerical_fallback_propagates_runtime_errors(
    monkeypatch,
):
    from probstats import Normal, information

    def broken(*args, **kwargs):
        raise RuntimeError("implementation bug")

    monkeypatch.setattr(information, "_symbolic_expectation_under", broken)
    monkeypatch.setattr(information, "get_information_formula", lambda *args: None)
    with pytest.raises(RuntimeError, match="implementation bug"):
        information.kl_divergence(
            Normal(0, 1), Normal(1, 1), numerical_fallback=True, samples=10
        )
