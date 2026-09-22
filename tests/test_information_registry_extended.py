import sympy as sp

from probstats import (
    Bernoulli,
    Exponential,
    Gamma,
    MultivariateNormal,
    Poisson,
    StudentT,
)
from probstats.distributions import (
    Categorical,
    InverseWishart,
    LogNormal,
    MatrixNormal,
    Wishart,
)
from probstats.information import (
    InformationMethod,
    information_entropy,
    kl_divergence,
    renyi_divergence,
)


def _closed(formula, *args):
    result = formula(*args, return_result=True)
    assert result.method is InformationMethod.CLOSED_FORM
    return result.value


def test_discrete_and_positive_family_closed_forms():
    p = Bernoulli(sp.Rational(1, 4))
    q = Bernoulli(sp.Rational(2, 3))
    assert _closed(kl_divergence, p, q) == sp.simplify(
        sp.Rational(1, 4) * sp.log(sp.Rational(3, 8))
        + sp.Rational(3, 4) * sp.log(sp.Rational(9, 4))
    )

    cp = Categorical([sp.Rational(1, 2), sp.Rational(1, 3), sp.Rational(1, 6)])
    cq = Categorical([sp.Rational(1, 4), sp.Rational(1, 2), sp.Rational(1, 4)])
    assert (
        sp.simplify(
            _closed(kl_divergence, cp, cq)
            - sum(a * sp.log(a / b) for a, b in zip(cp.probabilities, cq.probabilities))
        )
        == 0
    )

    pp, pq = Poisson(2), Poisson(5)
    assert (
        sp.simplify(
            _closed(kl_divergence, pp, pq) - (2 * sp.log(sp.Rational(2, 5)) + 3)
        )
        == 0
    )

    ep, eq = Exponential(2), Exponential(5)
    assert (
        sp.simplify(
            _closed(kl_divergence, ep, eq)
            - (sp.log(sp.Rational(2, 5)) + sp.Rational(3, 2))
        )
        == 0
    )


def test_lognormal_inherits_normal_divergences():
    p = LogNormal(1, 2)
    q = LogNormal(-1, 3)
    np = MultivariateNormal([1], [[4]])
    nq = MultivariateNormal([-1], [[9]])
    assert (
        sp.simplify(_closed(kl_divergence, p, q) - _closed(kl_divergence, np, nq)) == 0
    )
    assert (
        sp.simplify(
            _closed(renyi_divergence, p, q, sp.Rational(1, 2))
            - _closed(renyi_divergence, np, nq, sp.Rational(1, 2))
        )
        == 0
    )


def test_student_t_entropy_is_registered_but_pair_kl_is_not_forced_closed():
    p = StudentT(0, 2, 7)
    result = information_entropy(p, return_result=True)
    assert result.method is InformationMethod.CLOSED_FORM
    assert sp.simplify(result.value - p._entropy()) == 0


def test_wishart_one_dimensional_kl_and_renyi_agree_with_gamma():
    wp = Wishart(7, [[sp.Rational(3, 2)]])
    wq = Wishart(9, [[sp.Rational(4, 3)]])
    # W_1(nu, V) == Gamma(nu/2, scale=2 V).
    gp = Gamma(sp.Rational(7, 2), 3)
    gq = Gamma(sp.Rational(9, 2), sp.Rational(8, 3))
    assert (
        sp.simplify(_closed(kl_divergence, wp, wq) - _closed(kl_divergence, gp, gq))
        == 0
    )
    assert (
        sp.simplify(
            _closed(renyi_divergence, wp, wq, sp.Rational(1, 2))
            - _closed(renyi_divergence, gp, gq, sp.Rational(1, 2))
        )
        == 0
    )


def test_inverse_wishart_divergences_are_zero_for_identical_laws():
    p = InverseWishart(6, [[2, 0], [0, 3]])
    assert sp.simplify(_closed(kl_divergence, p, p)) == 0
    assert sp.simplify(_closed(renyi_divergence, p, p, sp.Rational(1, 2))) == 0


def test_matrix_normal_matches_vectorized_multivariate_normal():
    mp = sp.Matrix([[0, 1], [2, -1]])
    mq = sp.Matrix([[1, 0], [1, 2]])
    up = sp.diag(2, 3)
    uq = sp.diag(4, 5)
    vp = sp.diag(6, 7)
    vq = sp.diag(8, 9)
    p = MatrixNormal(mp, up, vp)
    q = MatrixNormal(mq, uq, vq)

    # SymPy's list(Matrix) order matches the representation used by the registry.
    pv = MultivariateNormal(list(mp), sp.kronecker_product(vp, up))
    qv = MultivariateNormal(list(mq), sp.kronecker_product(vq, uq))
    assert (
        sp.simplify(_closed(kl_divergence, p, q) - _closed(kl_divergence, pv, qv)) == 0
    )
    assert (
        sp.simplify(
            _closed(renyi_divergence, p, q, sp.Rational(1, 2))
            - _closed(renyi_divergence, pv, qv, sp.Rational(1, 2))
        )
        == 0
    )


def test_wishart_and_inverse_wishart_entropy_are_closed_form():
    w = Wishart(8, [[2, 0], [0, 3]])
    iw = InverseWishart(8, [[2, 0], [0, 3]])
    assert (
        information_entropy(w, return_result=True).method
        is InformationMethod.CLOSED_FORM
    )
    assert (
        information_entropy(iw, return_result=True).method
        is InformationMethod.CLOSED_FORM
    )
