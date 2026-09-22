"""Metamorphic identities spanning distribution algebra and information measures."""

from __future__ import annotations

import math

import numpy as np
import sympy as sp

from probstats import Gamma, Normal, Poisson
from probstats.algebra import recognize_affine
from probstats.information import kl_divergence
from probstats.symbolic import convolution


def _float(value):
    return float(sp.N(value, 40))


def test_affine_normal_preserves_cdf_coordinates_and_moments():
    base = Normal(sp.Rational(2, 3), sp.Rational(4, 5))
    transformed = recognize_affine(
        base, sp.Rational(5, 2), sp.Rational(-7, 4)
    ).distribution
    for x in (-1.5, 0.0, 1.25, 3.0):
        y = 2.5 * x - 1.75
        assert math.isclose(
            _float(transformed.cdf(y)),
            _float(base.cdf(x)),
            rel_tol=1e-11,
            abs_tol=1e-11,
        )
    assert (
        sp.simplify(
            transformed.mean - (sp.Rational(5, 2) * base.mean - sp.Rational(7, 4))
        )
        == 0
    )
    assert sp.simplify(transformed.sigma - sp.Rational(5, 2) * base.sigma) == 0


def test_closed_family_convolution_identities():
    n = convolution(Normal(1, 2), Normal(3, 4))
    assert isinstance(n, Normal)
    assert sp.simplify(n.mean - 4) == 0
    assert sp.simplify(n.sigma**2 - 20) == 0

    p = convolution(Poisson(2), Poisson(5))
    assert isinstance(p, Poisson)
    assert sp.simplify(p.rate - 7) == 0

    g = convolution(Gamma(2, 3), Gamma(5, 3))
    assert isinstance(g, Gamma)
    assert sp.simplify(g.shape - 7) == 0
    assert sp.simplify(g.scale - 3) == 0


def test_kl_identity_and_nonnegativity_for_normal_family():
    a = Normal(0, 1)
    b = Normal(1, 2)
    assert sp.simplify(kl_divergence(a, a)) == 0
    assert _float(kl_divergence(a, b)) >= 0
    assert _float(kl_divergence(b, a)) >= 0


def test_sample_covariance_is_symmetric_and_psd():
    rng = np.random.default_rng(1234)
    x = rng.normal(size=(500, 4))
    covariance = np.cov(x, rowvar=False)
    assert np.allclose(covariance, covariance.T)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-12
