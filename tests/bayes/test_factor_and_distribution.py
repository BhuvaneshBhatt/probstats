import pytest
import sympy as sp

from probstats.bayes import (
    Factor,
    Parameter,
)
from probstats.distributions import SymbolicDistribution


def test_symbolic_distribution_substitutes_value_and_exposes_parameters():
    x, mu, sigma = sp.symbols("x mu sigma", real=True)
    dist = SymbolicDistribution(x, -((x - mu) ** 2) / (2 * sigma**2))
    assert sp.simplify(dist.logpdf(3) + (3 - mu) ** 2 / (2 * sigma**2)) == 0
    assert dist.parameters == (mu, sigma)


def test_factor_requires_exactly_one_representation():
    x = Parameter("x")
    with pytest.raises(ValueError, match="exactly one"):
        Factor(target=x)
    with pytest.raises(ValueError, match="exactly one"):
        Factor(
            target=x,
            distribution=SymbolicDistribution(sp.Symbol("z"), 0),
            log_density=0,
        )


def test_custom_log_density_factor_tracks_parent_symbols():
    x = Parameter("x")
    mu = sp.Symbol("mu")
    factor = Factor.from_log_density(x, -((x.symbol - mu) ** 2))
    assert factor.parent_symbols == {mu}
