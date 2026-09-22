import numpy as np
import pytest
import sympy as sp

from probstats.distributions import (
    HistogramDistribution,
    KernelDensityDistribution,
    KernelMixtureDistribution,
    MultivariateKernelDensityDistribution,
    cv_bandwidth,
    plugin_bandwidth,
    scott_bandwidth,
    silverman_bandwidth,
)
from probstats.functionals import (
    mean,
    variance,
)


def test_histogram_distribution_normalizes_and_has_exact_moments():
    dist = HistogramDistribution(bin_edges=[0, 1, 3], probabilities=[1, 3])
    x = sp.symbols("x", real=True)
    assert sp.simplify(sp.integrate(dist.pdf(x), (x, -sp.oo, sp.oo)) - 1) == 0
    assert sp.simplify(dist.cdf(1) - sp.Rational(1, 4)) == 0
    assert float(mean(dist)) == pytest.approx(1.625)
    assert float(variance(dist)) > 0


def test_histogram_sampling_matches_bin_probabilities():
    dist = HistogramDistribution(bin_edges=[0, 1, 2], probabilities=[1, 3])
    draws = dist.sample(20_000, rng=123)
    assert np.mean(draws >= 1) == pytest.approx(0.75, abs=0.02)


def test_kernel_mixture_cdf_and_moments():
    dist = KernelMixtureDistribution([-1, 1], [0.5, 0.5], weights=[1, 1])
    assert sp.simplify(dist.cdf(0) - sp.Rational(1, 2)) == 0
    assert float(mean(dist)) == pytest.approx(0.0, abs=1e-12)
    assert float(variance(dist)) == pytest.approx(1.25)


def test_kde_weighted_mean_and_sampling():
    dist = KernelDensityDistribution([0, 2], bandwidth=0.25, weights=[1, 3])
    assert float(mean(dist)) == pytest.approx(1.5)
    draws = dist.sample(10_000, rng=42)
    assert draws.mean() == pytest.approx(1.5, abs=0.05)


def test_bandwidth_rules_are_positive_and_distinct():
    data = np.array([-2.0, -1.1, -0.2, 0.0, 0.3, 1.2, 2.8])
    values = [
        scott_bandwidth(data),
        silverman_bandwidth(data),
        plugin_bandwidth(data),
        cv_bandwidth(data, grid_size=11),
    ]
    assert all(np.isfinite(h) and h > 0 for h in values)
    assert len({round(h, 8) for h in values}) >= 2


def test_cv_bandwidth_uses_candidate_grid():
    data = [-1.0, -0.5, 0.2, 1.0]
    candidates = np.array([0.2, 0.4, 0.8])
    result = cv_bandwidth(data, candidates=candidates)
    assert result in candidates


def test_multivariate_kde_pdf_and_sampling_shape():
    data = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    dist = MultivariateKernelDensityDistribution(data, bandwidth="scott")
    value = dist.pdf([0.5, 0.5])
    assert float(value.evalf()) > 0
    draws = dist.sample((5, 3), rng=7)
    assert draws.shape == (5, 3, 2)


def test_multivariate_kde_weighted_center_approximately_preserved():
    data = np.array([[0.0, 0.0], [3.0, 1.0], [2.0, 4.0]])
    weights = np.array([1.0, 2.0, 5.0])
    dist = MultivariateKernelDensityDistribution(data, weights=weights)
    expected = np.average(data, axis=0, weights=weights)
    draws = dist.sample(30_000, rng=13)
    assert np.allclose(draws.mean(axis=0), expected, atol=0.06)


def test_bandwidth_validation():
    with pytest.raises(ValueError):
        KernelDensityDistribution([1, 2, 3], bandwidth=0)
    with pytest.raises(ValueError):
        MultivariateKernelDensityDistribution(
            [[0, 0], [1, 1]], bandwidth=[[1, 2], [0, 1]]
        )


def test_cv_bandwidth_rejects_fractional_grid_size():
    with pytest.raises(TypeError, match="grid_size"):
        cv_bandwidth([0.0, 1.0, 2.0], grid_size=5.5)
