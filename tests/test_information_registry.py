import pytest
import sympy as sp

from probstats import (
    Beta,
    Gamma,
    MultivariateNormal,
    Normal,
)
from probstats.distributions import Dirichlet
from probstats.information import (
    InformationMethod,
    bhattacharyya_coefficient,
    cross_entropy,
    information_entropy,
    kl_divergence,
    renyi_divergence,
)
from probstats.information_registry import registered_information_formulas


def _assert_same(a, b):
    assert sp.simplify(sp.expand_func(a - b)) == 0


def test_registry_contains_common_pair_formulas():
    keys = {
        (m, p.__name__, q.__name__) for m, p, q in registered_information_formulas()
    }
    for name in ("Normal", "MultivariateNormal", "Gamma", "Beta", "Dirichlet"):
        assert ("kl", name, name) in keys
        assert ("renyi", name, name) in keys
        assert ("entropy", name, name) in keys


def test_normal_kl_and_renyi_use_closed_form_registry():
    p, q = Normal(0, 2), Normal(1, 3)
    kl = kl_divergence(p, q, return_result=True)
    r = renyi_divergence(p, q, sp.Rational(1, 2), return_result=True)
    assert kl.method is InformationMethod.CLOSED_FORM
    assert r.method is InformationMethod.CLOSED_FORM
    expected = sp.log(sp.Rational(3, 2)) + sp.Rational(5, 18) - sp.Rational(1, 2)
    _assert_same(kl.value, expected)


def test_multivariate_normal_kl_closed_form():
    p = MultivariateNormal([0, 0], [[1, 0], [0, 2]])
    q = MultivariateNormal([1, -1], [[2, 0], [0, 3]])
    result = kl_divergence(p, q, return_result=True)
    assert result.method is InformationMethod.CLOSED_FORM
    expected = (
        sp.trace(q.covariance.inv() * p.covariance)
        + ((q.mean - p.mean).T * q.covariance.inv() * (q.mean - p.mean))[0]
        - 2
        + sp.log(q.covariance.det() / p.covariance.det())
    ) / 2
    _assert_same(result.value, expected)


def test_gamma_beta_dirichlet_kl_closed_forms_match_generic_identities():
    pairs = [
        (Gamma(2, 3), Gamma(4, 5)),
        (Beta(2, 3), Beta(4, 5)),
        (Dirichlet([2, 3, 4]), Dirichlet([3, 5, 7])),
    ]
    for p, q in pairs:
        result = kl_divergence(p, q, return_result=True)
        assert result.method is InformationMethod.CLOSED_FORM
        assert sp.N(result.value) >= 0
        _assert_same(cross_entropy(p, q), information_entropy(p) + result.value)


def test_beta_and_dirichlet_renyi_closed_form_at_half_order():
    for p, q in [
        (Beta(2, 5), Beta(3, 4)),
        (Dirichlet([2, 3, 5]), Dirichlet([3, 4, 6])),
    ]:
        r = renyi_divergence(p, q, sp.Rational(1, 2), return_result=True)
        bc = bhattacharyya_coefficient(p, q, return_result=True)
        assert r.method is InformationMethod.CLOSED_FORM
        assert bc.method is InformationMethod.CLOSED_FORM
        _assert_same(sp.exp(-r.value / 2), bc.value)


def test_cross_entropy_uses_registered_entropy_plus_kl():
    p, q = Gamma(3, 2), Gamma(5, 4)
    result = cross_entropy(p, q, return_result=True)
    assert result.method is InformationMethod.CLOSED_FORM
    _assert_same(result.value, information_entropy(p) + kl_divergence(p, q))


def test_binomial_closed_form_requires_certified_matching_counts():
    from probstats.distributions import Binomial
    from probstats.information_registry import _binomial_kl

    n = sp.Symbol("n", integer=True, nonnegative=True)
    m = sp.Symbol("m", integer=True, nonnegative=True)
    p = Binomial(n, sp.Rational(1, 3))
    q = Binomial(m, sp.Rational(1, 2))
    with pytest.raises(NotImplementedError, match="matching trial counts"):
        _binomial_kl(p, q)

    equal = _binomial_kl(p, Binomial(n, sp.Rational(1, 2)))
    assert equal.has(n)
