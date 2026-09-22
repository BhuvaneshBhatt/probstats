"""Differential checks between exact symbolic formulas and numerical oracles."""

from __future__ import annotations

import math

import pytest
import sympy as sp

from probstats import Beta, Gamma, Normal, Poisson, StudentT
from probstats.functionals import mean, variance


def _float(expr):
    return float(sp.N(expr, 50))


@pytest.mark.parametrize(
    "dist, scipy_name, args, kwargs, points",
    [
        (
            Normal(sp.Rational(1, 3), sp.Rational(5, 4)),
            "norm",
            (),
            {"loc": 1 / 3, "scale": 5 / 4},
            (-2, 0, 1, 3),
        ),
        (
            Gamma(sp.Rational(5, 2), sp.Rational(3, 4)),
            "gamma",
            (2.5,),
            {"scale": 0.75},
            (0.1, 0.5, 2, 6),
        ),
        (
            Beta(sp.Rational(3, 2), sp.Rational(7, 3)),
            "beta",
            (1.5, 7 / 3),
            {},
            (0.1, 0.3, 0.7, 0.95),
        ),
        (Poisson(sp.Rational(7, 3)), "poisson", (7 / 3,), {}, (0, 1, 3, 7)),
        (
            StudentT(sp.Rational(1, 4), sp.Rational(3, 2), sp.Integer(9)),
            "t",
            (9,),
            {"loc": 0.25, "scale": 1.5},
            (-3, 0, 1, 4),
        ),
    ],
)
def test_symbolic_density_cdf_numeric_oracle(dist, scipy_name, args, kwargs, points):
    scipy_stats = pytest.importorskip("scipy.stats")
    ref = getattr(scipy_stats, scipy_name)(*args, **kwargs)
    discrete = scipy_name == "poisson"
    for x in points:
        ours_density = _float(dist.pdf(sp.Rational(str(x))))
        ref_density = float(ref.pmf(x) if discrete else ref.pdf(x))
        assert math.isclose(ours_density, ref_density, rel_tol=2e-10, abs_tol=2e-12)
        assert math.isclose(
            _float(dist.cdf(sp.Rational(str(x)))),
            float(ref.cdf(x)),
            rel_tol=2e-10,
            abs_tol=2e-12,
        )


def test_symbolic_moments_preserve_exactness_before_numeric_evaluation():
    dist = Gamma(sp.Rational(5, 2), sp.Rational(3, 4))
    assert sp.simplify(mean(dist) - sp.Rational(15, 8)) == 0
    assert sp.simplify(variance(dist) - sp.Rational(45, 32)) == 0
    assert math.isclose(_float(mean(dist)), 15 / 8, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(_float(variance(dist)), 45 / 32, rel_tol=0, abs_tol=1e-15)
