import sympy as sp

from probstats import (
    Bernoulli,
    Binomial,
)
from probstats.bayes import NormalInverseWishart
from probstats.distributions import (
    Categorical,
    InverseWishart,
    Multinomial,
)
from probstats.information import (
    InformationMethod,
    cross_entropy,
    information_entropy,
    kl_divergence,
    renyi_divergence,
)


def test_binomial_kl_and_renyi_reduce_to_bernoulli():
    p = Binomial(7, sp.Rational(1, 4))
    q = Binomial(7, sp.Rational(2, 5))
    bp = Bernoulli(p.p)
    bq = Bernoulli(q.p)
    kl = kl_divergence(p, q, return_result=True)
    r = renyi_divergence(p, q, sp.Rational(1, 2), return_result=True)
    assert kl.method is InformationMethod.CLOSED_FORM
    assert r.method is InformationMethod.CLOSED_FORM
    assert sp.simplify(kl.value - 7 * kl_divergence(bp, bq)) == 0
    assert sp.simplify(r.value - 7 * renyi_divergence(bp, bq, sp.Rational(1, 2))) == 0


def test_multinomial_kl_and_renyi_reduce_to_categorical():
    p = Multinomial(5, [sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 6)])
    q = Multinomial(5, [sp.Rational(1, 3), sp.Rational(1, 2), sp.Rational(1, 6)])
    cp = Categorical(p.probabilities)
    cq = Categorical(q.probabilities)
    kl = kl_divergence(p, q, return_result=True)
    r = renyi_divergence(p, q, sp.Rational(2, 3), return_result=True)
    assert kl.method is InformationMethod.CLOSED_FORM
    assert r.method is InformationMethod.CLOSED_FORM
    assert sp.simplify(kl.value - 5 * kl_divergence(cp, cq)) == 0
    assert sp.simplify(r.value - 5 * renyi_divergence(cp, cq, sp.Rational(2, 3))) == 0


def test_niw_kl_reduces_to_inverse_wishart_when_conditionals_match():
    p = NormalInverseWishart([0, 0], 3, sp.diag(2, 3), 6)
    q = NormalInverseWishart([0, 0], 3, sp.diag(4, 5), 8)
    result = kl_divergence(p, q, return_result=True)
    expected = kl_divergence(InverseWishart(p.nu, p.psi), InverseWishart(q.nu, q.psi))
    assert result.method is InformationMethod.CLOSED_FORM
    assert sp.simplify(result.value - expected) == 0


def test_niw_entropy_cross_entropy_identity_and_renyi_symmetry_at_half():
    p = NormalInverseWishart([0, 1], 2, sp.diag(3, 4), 7)
    q = NormalInverseWishart([1, -1], 5, sp.diag(5, 6), 9)
    h = information_entropy(p, return_result=True)
    ce = cross_entropy(p, q, return_result=True)
    kl = kl_divergence(p, q, return_result=True)
    r_pq = renyi_divergence(p, q, sp.Rational(1, 2), return_result=True)
    r_qp = renyi_divergence(q, p, sp.Rational(1, 2), return_result=True)
    assert h.method is InformationMethod.CLOSED_FORM
    assert ce.method is InformationMethod.CLOSED_FORM
    assert kl.method is InformationMethod.CLOSED_FORM
    assert r_pq.method is InformationMethod.CLOSED_FORM
    assert sp.simplify(ce.value - h.value - kl.value) == 0
    assert sp.simplify(r_pq.value - r_qp.value) == 0
