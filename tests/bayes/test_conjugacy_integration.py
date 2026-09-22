import sympy as sp

from probstats import Normal as ProbstatsNormal
from probstats.bayes.conjugacy import conjugate_update, infer_conjugate
from probstats.bayes.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Categorical,
    Dirichlet,
    Gamma,
    Multinomial,
    MultivariateNormal,
    Normal,
    Poisson,
)


def test_bayesian_namespace_reuses_root_distribution_class():
    assert Normal is ProbstatsNormal


def test_beta_bernoulli_update():
    out = conjugate_update(Bernoulli(sp.Symbol("p")), [1, 0, 1], Beta(2, 3))
    assert out["posterior"] == Beta(4, 4)
    assert out["sufficient_statistics"] == {"successes": 2, "failures": 1}


def test_beta_binomial_update():
    out = conjugate_update(Binomial(5, sp.Symbol("p")), [2, 4], Beta(1, 1))
    assert out["posterior"] == Beta(7, 5)


def test_gamma_poisson_update():
    out = conjugate_update(
        Poisson(sp.Symbol("lam", positive=True)), [2, 3, 1], Gamma(2, 4)
    )
    assert out["posterior"].shape == 8
    assert sp.simplify(out["posterior"].scale - sp.Rational(4, 13)) == 0


def test_dirichlet_categorical_update():
    like = Categorical([sp.Rational(1, 3)] * 3)
    out = conjugate_update(like, [0, 2, 2, 1], Dirichlet([1, 1, 1]))
    assert out["posterior"] == Dirichlet([2, 2, 3])


def test_dirichlet_multinomial_update_and_inference_result():
    like = Multinomial(3, [sp.Rational(1, 2), sp.Rational(1, 2)])
    res = infer_conjugate(like, [[1, 2], [2, 1]], Dirichlet([2, 3]))
    assert res.posterior == Dirichlet([5, 6])
    assert res.metadata["n_observations"] == 2


def test_multivariate_normal_is_available_in_bayesian_namespace():
    d = MultivariateNormal([0, 0], sp.eye(2))
    assert d.dimension == 2


def test_multivariate_normal_niw_is_registered():
    from probstats.bayes.conjugacy import NormalInverseWishart

    like = MultivariateNormal([0, 0], sp.eye(2))
    prior = NormalInverseWishart([0, 0], 1, sp.eye(2), 3)
    out = conjugate_update(like, [[1, 0], [0, 1]], prior)
    assert isinstance(out["posterior"], NormalInverseWishart)
    assert out["posterior"].nu == 5
