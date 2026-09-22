import pytest
import sympy as sp

from probstats import Bernoulli, Exponential, Gamma, Normal, Poisson, Uniform


@pytest.mark.parametrize(
    "constructor,args",
    [
        (Bernoulli, (sp.nan,)),
        (Bernoulli, (sp.oo,)),
        (Exponential, (sp.oo,)),
        (Gamma, (2, sp.oo)),
        (Normal, (sp.oo, 1)),
        (Normal, (0, sp.oo)),
        (Poisson, (sp.oo,)),
        (Uniform, (-sp.oo, 1)),
        (Uniform, (0, sp.oo)),
    ],
)
def test_concrete_nonfinite_distribution_parameters_are_rejected(constructor, args):
    with pytest.raises((TypeError, ValueError)):
        constructor(*args)


def test_unknown_symbolic_parameters_allowed():
    sigma = sp.symbols("sigma", real=True)
    distribution = Normal(0, sigma)
    assert distribution.sigma == sigma
