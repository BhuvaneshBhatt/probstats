"""Central contract metadata for concrete scalar distributions.

This registry stays in the test suite.  It records one or more safe numeric
instances for every concrete scalar law that is part of the distribution
catalogue, together with optional SciPy oracles and contract exceptions.
Adding a new scalar distribution should normally require adding an entry here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import sympy as sp

from probstats.distributions import (
    Bernoulli,
    Beta,
    BetaBinomial,
    BetaPrime,
    Binomial,
    Cauchy,
    ChiSquared,
    DiscreteUniform,
    Exponential,
    FDistribution,
    Frechet,
    Gamma,
    Geometric,
    Gumbel,
    HalfCauchy,
    HalfNormal,
    Hypergeometric,
    InverseGamma,
    InverseGaussian,
    Kumaraswamy,
    Laplace,
    Logistic,
    LogNormal,
    Maxwell,
    Nakagami,
    NegativeBinomial,
    NoncentralChiSquared,
    NoncentralF,
    NoncentralT,
    Normal,
    Pareto,
    Poisson,
    PowerDistribution,
    Rayleigh,
    Rice,
    Skellam,
    StudentT,
    Triangular,
    Uniform,
    VonMises,
    Weibull,
    Zipf,
)


@dataclass(frozen=True)
class ScalarDistributionSpec:
    name: str
    factory: Callable[[], object]
    probes: tuple[float, ...]
    probabilities: tuple[float, ...] = (0.05, 0.25, 0.5, 0.8, 0.95)
    scipy_name: str | None = None
    scipy_args: tuple = ()
    scipy_kwds: tuple[tuple[str, float], ...] = ()
    density_contract: bool = True
    quantile_contract: bool = True
    cdf_contract: bool = True
    normalization_contract: bool = True

    @property
    def scipy_kwargs(self):
        return dict(self.scipy_kwds)


SCALAR_SPECS = (
    ScalarDistributionSpec(
        "Bernoulli",
        lambda: Bernoulli(0.3),
        (-1, 0, 1, 2),
        scipy_name="bernoulli",
        scipy_args=(0.3,),
    ),
    ScalarDistributionSpec(
        "Beta",
        lambda: Beta(2, 5),
        (-0.2, 0.1, 0.5, 0.9, 1.2),
        scipy_name="beta",
        scipy_args=(2, 5),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "BetaBinomial",
        lambda: BetaBinomial(8, 2, 3),
        (-1, 0, 2, 5, 8, 9),
        scipy_name="betabinom",
        scipy_args=(8, 2, 3),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "BetaPrime",
        lambda: BetaPrime(3, 4, 1.2),
        (-1, 0.1, 0.5, 2, 10),
        scipy_name="betaprime",
        scipy_args=(3, 4),
        scipy_kwds=(("scale", 1.2),),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Binomial",
        lambda: Binomial(8, 0.35),
        (-1, 0, 2, 5, 8, 9),
        scipy_name="binom",
        scipy_args=(8, 0.35),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Cauchy",
        lambda: Cauchy(0.4, 1.3),
        (-8, -1, 0.4, 2, 8),
        scipy_name="cauchy",
        scipy_kwds=(("loc", 0.4), ("scale", 1.3)),
    ),
    ScalarDistributionSpec(
        "ChiSquared",
        lambda: ChiSquared(5),
        (-1, 0.2, 2, 8, 20),
        scipy_name="chi2",
        scipy_args=(5,),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "DiscreteUniform",
        lambda: DiscreteUniform(2, 7),
        (0, 2, 4, 7, 9),
        scipy_name="randint",
        scipy_args=(2, 8),
    ),
    ScalarDistributionSpec(
        "Exponential",
        lambda: Exponential(1.4),
        (-1, 0, 0.5, 2, 8),
        scipy_name="expon",
        scipy_kwds=(("scale", 1 / 1.4),),
    ),
    ScalarDistributionSpec(
        "FDistribution",
        lambda: FDistribution(5, 9),
        (-1, 0.1, 1, 4, 15),
        scipy_name="f",
        scipy_args=(5, 9),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Frechet",
        lambda: Frechet(3, 1.2, 0.4),
        (0.2, 0.5, 1, 3, 10),
        scipy_name="invweibull",
        scipy_args=(3,),
        scipy_kwds=(("loc", 0.4), ("scale", 1.2)),
    ),
    ScalarDistributionSpec(
        "Gamma",
        lambda: Gamma(2.5, 1.2),
        (-1, 0.2, 1, 5, 15),
        scipy_name="gamma",
        scipy_args=(2.5,),
        scipy_kwds=(("scale", 1.2),),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Geometric",
        lambda: Geometric(0.3),
        (0, 1, 2, 5, 12),
        scipy_name="geom",
        scipy_args=(0.3,),
    ),
    ScalarDistributionSpec(
        "Gumbel",
        lambda: Gumbel(0.4, 1.3),
        (-8, -1, 0.4, 2, 8),
        scipy_name="gumbel_r",
        scipy_kwds=(("loc", 0.4), ("scale", 1.3)),
    ),
    ScalarDistributionSpec(
        "HalfCauchy",
        lambda: HalfCauchy(1.3),
        (-1, 0.1, 1, 4, 12),
        scipy_name="halfcauchy",
        scipy_kwds=(("scale", 1.3),),
    ),
    ScalarDistributionSpec(
        "HalfNormal",
        lambda: HalfNormal(1.3),
        (-1, 0.1, 1, 3, 8),
        scipy_name="halfnorm",
        scipy_kwds=(("scale", 1.3),),
    ),
    ScalarDistributionSpec(
        "Hypergeometric",
        lambda: Hypergeometric(12, 8, 6),
        (-1, 0, 2, 5, 7),
        scipy_name="hypergeom",
        scipy_args=(20, 12, 6),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "InverseGamma",
        lambda: InverseGamma(4, 2),
        (-1, 0.1, 0.5, 2, 8),
        scipy_name="invgamma",
        scipy_args=(4,),
        scipy_kwds=(("scale", 2),),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "InverseGaussian",
        lambda: InverseGaussian(2, 3),
        (-1, 0.1, 1, 3, 10),
        scipy_name=None,
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Kumaraswamy",
        lambda: Kumaraswamy(2, 4),
        (-0.2, 0.1, 0.5, 0.9, 1.2),
        scipy_name=None,
    ),
    ScalarDistributionSpec(
        "Laplace",
        lambda: Laplace(0.4, 1.3),
        (-8, -1, 0.4, 2, 8),
        scipy_name="laplace",
        scipy_kwds=(("loc", 0.4), ("scale", 1.3)),
    ),
    ScalarDistributionSpec(
        "LogNormal",
        lambda: LogNormal(0.4, 0.7),
        (-1, 0.1, 1, 3, 10),
        scipy_name="lognorm",
        scipy_args=(0.7,),
        scipy_kwds=(("scale", float(sp.exp(sp.Rational(2, 5)))),),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Logistic",
        lambda: Logistic(0.4, 1.3),
        (-8, -1, 0.4, 2, 8),
        scipy_name="logistic",
        scipy_kwds=(("loc", 0.4), ("scale", 1.3)),
    ),
    ScalarDistributionSpec(
        "Maxwell",
        lambda: Maxwell(1.3),
        (-1, 0.1, 1, 3, 8),
        scipy_name="maxwell",
        scipy_kwds=(("scale", 1.3),),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Nakagami",
        lambda: Nakagami(2, 1.5),
        (-1, 0.1, 1, 2, 5),
        scipy_name="nakagami",
        scipy_args=(2,),
        scipy_kwds=(("scale", float(sp.sqrt(sp.Rational(3, 2)))),),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "NegativeBinomial",
        lambda: NegativeBinomial(4, 0.35),
        (-1, 0, 2, 8, 20),
        scipy_name="nbinom",
        scipy_args=(4, 0.35),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "NoncentralChiSquared",
        lambda: NoncentralChiSquared(5, 1.2),
        (-1, 0.1, 2, 8, 20),
        scipy_name="ncx2",
        scipy_args=(5, 1.2),
        density_contract=False,
        quantile_contract=False,
        cdf_contract=False,
    ),
    ScalarDistributionSpec(
        "NoncentralF",
        lambda: NoncentralF(5, 9, 1.2),
        (-1, 0.1, 1, 4, 15),
        scipy_name="ncf",
        scipy_args=(5, 9, 1.2),
        density_contract=False,
        quantile_contract=False,
        cdf_contract=False,
    ),
    ScalarDistributionSpec(
        "NoncentralT",
        lambda: NoncentralT(8, 0.7),
        (-8, -1, 0, 2, 8),
        scipy_name="nct",
        scipy_args=(8, 0.7),
        density_contract=False,
        quantile_contract=False,
        cdf_contract=False,
    ),
    ScalarDistributionSpec(
        "Normal",
        lambda: Normal(0.3, 1.7),
        (-8, -1, 0.3, 2, 9),
        scipy_name="norm",
        scipy_kwds=(("loc", 0.3), ("scale", 1.7)),
    ),
    ScalarDistributionSpec(
        "Pareto",
        lambda: Pareto(3, 1.2),
        (0.5, 1.2, 2, 5, 12),
        scipy_name="pareto",
        scipy_args=(3,),
        scipy_kwds=(("scale", 1.2),),
    ),
    ScalarDistributionSpec(
        "Poisson",
        lambda: Poisson(2.5),
        (-1, 0, 2, 5, 12),
        scipy_name="poisson",
        scipy_args=(2.5,),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "PowerDistribution",
        lambda: PowerDistribution(2.5),
        (-0.2, 0.1, 0.5, 0.9, 1.2),
        scipy_name="powerlaw",
        scipy_args=(2.5,),
    ),
    ScalarDistributionSpec(
        "Rayleigh",
        lambda: Rayleigh(1.3),
        (-1, 0.1, 1, 3, 8),
        scipy_name="rayleigh",
        scipy_kwds=(("scale", 1.3),),
    ),
    ScalarDistributionSpec(
        "Rice",
        lambda: Rice(1.2, 0.8),
        (-1, 0.1, 1, 3, 8),
        scipy_name="rice",
        scipy_args=(1.2 / 0.8,),
        scipy_kwds=(("scale", 0.8),),
        quantile_contract=False,
        cdf_contract=False,
    ),
    ScalarDistributionSpec(
        "Skellam",
        lambda: Skellam(3, 2),
        (-8, -2, 0, 3, 8),
        scipy_name="skellam",
        scipy_args=(3, 2),
        quantile_contract=False,
        cdf_contract=False,
    ),
    ScalarDistributionSpec(
        "StudentT",
        lambda: StudentT(0.4, 1.3, 7),
        (-8, -1, 0.4, 2, 8),
        scipy_name="t",
        scipy_args=(7,),
        scipy_kwds=(("loc", 0.4), ("scale", 1.3)),
    ),
    ScalarDistributionSpec(
        "Triangular",
        lambda: Triangular(-2, 0.5, 4),
        (-3, -2, 0, 2, 4, 5),
        scipy_name="triang",
        scipy_args=((0.5 + 2) / 6,),
        scipy_kwds=(("loc", -2), ("scale", 6)),
        quantile_contract=False,
    ),
    ScalarDistributionSpec(
        "Uniform",
        lambda: Uniform(-2, 3),
        (-3, -2, 0, 3, 4),
        scipy_name="uniform",
        scipy_kwds=(("loc", -2), ("scale", 5)),
    ),
    ScalarDistributionSpec(
        "VonMises",
        lambda: VonMises(0.4, 2),
        (-2.5, -1, 0.4, 2, 3),
        scipy_name="vonmises",
        scipy_args=(2,),
        scipy_kwds=(("loc", 0.4),),
        quantile_contract=False,
        cdf_contract=False,
    ),
    ScalarDistributionSpec(
        "Weibull",
        lambda: Weibull(2.5, 1.3),
        (-1, 0.1, 1, 3, 8),
        scipy_name="weibull_min",
        scipy_args=(2.5,),
        scipy_kwds=(("scale", 1.3),),
    ),
    ScalarDistributionSpec(
        "Zipf",
        lambda: Zipf(2.5),
        (0, 1, 2, 5, 20),
        scipy_name="zipf",
        scipy_args=(2.5,),
        quantile_contract=False,
    ),
)

SCALAR_SPEC_BY_NAME = {spec.name: spec for spec in SCALAR_SPECS}
