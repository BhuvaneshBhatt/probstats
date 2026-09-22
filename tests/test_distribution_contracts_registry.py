"""Registry-driven contracts for the concrete scalar distribution catalogue."""

from __future__ import annotations

import inspect
import itertools
import math

import numpy as np
import pytest
import sympy as sp

from probstats import distributions
from probstats.distributions import Distribution
from probstats.spaces import MeasureType, ScalarEventSpace
from tests.distribution_contract_registry import SCALAR_SPEC_BY_NAME, SCALAR_SPECS


def _float(value):
    if hasattr(value, "doit"):
        value = value.doit()
    if hasattr(value, "evalf"):
        value = value.evalf(40)
    return float(value)


def _concrete_exported_scalar_names():
    excluded = {
        "Distribution",
        "SymbolicDistribution",
        "ExponentialFamily",
        "ProbabilityDistribution",
        "ParameterMixtureDistribution",
        "SplicedDistribution",
        "CensoredDistribution",
        "HistogramDistribution",
        "KernelDensityDistribution",
        "KernelMixtureDistribution",
        "MultivariateKernelDensityDistribution",
    }
    result = set()
    for name in distributions.__all__:
        obj = getattr(distributions, name, None)
        if name in excluded or not inspect.isclass(obj):
            continue
        try:
            if not issubclass(obj, Distribution):
                continue
        except TypeError:
            continue
        # Matrix/vector-valued catalogue entries are tested in their dedicated suites.
        if name in {
            "Categorical",
            "Dirichlet",
            "DirichletMultinomial",
            "InverseWishart",
            "LKJ",
            "LKJCholesky",
            "MatrixNormal",
            "Multinomial",
            "MultivariateNormal",
            "MultivariateStudentT",
            "Wishart",
        }:
            continue
        result.add(name)
    return result


def test_registry_covers_every_concrete_scalar_catalogue_distribution():
    assert set(SCALAR_SPEC_BY_NAME) == _concrete_exported_scalar_names()


@pytest.mark.parametrize("spec", SCALAR_SPECS, ids=lambda s: s.name)
def test_scalar_distribution_support_density_and_sample_contracts(spec):
    dist = spec.factory()
    assert isinstance(dist.event_space, ScalarEventSpace)
    if spec.density_contract:
        for x in spec.probes:
            density = _float(dist.pdf(x))
            assert math.isfinite(density) or density == math.inf
            assert density >= -1e-12
    draws = np.asarray(dist.sample(size=(2, 3), rng=1729))
    assert draws.shape == (2, 3)


@pytest.mark.parametrize(
    "spec", [s for s in SCALAR_SPECS if s.cdf_contract], ids=lambda s: s.name
)
def test_scalar_cdf_monotonicity_and_complement_identity(spec):
    dist = spec.factory()
    values = sorted(spec.probes)
    cdfs = [_float(dist.cdf(x)) for x in values]
    assert all(-1e-10 <= v <= 1 + 1e-10 for v in cdfs)
    assert all(left <= right + 1e-10 for left, right in itertools.pairwise(cdfs))
    for x, cdf_value in zip(values, cdfs):
        survival = _float(dist.survival(x))
        assert math.isclose(cdf_value + survival, 1.0, rel_tol=2e-8, abs_tol=2e-8)


@pytest.mark.parametrize(
    "spec", [s for s in SCALAR_SPECS if s.quantile_contract], ids=lambda s: s.name
)
def test_scalar_generalized_inverse_quantile_contract(spec):
    dist = spec.factory()
    for p in spec.probabilities:
        q = _float(dist.quantile(p))
        fq = _float(dist.cdf(q))
        assert fq + 2e-8 >= p
        if dist.measure_type is MeasureType.CONTINUOUS:
            assert math.isclose(fq, p, rel_tol=3e-6, abs_tol=3e-6)
        else:
            previous = _float(dist.cdf(math.floor(q) - 1))
            assert previous < p + 2e-8


@pytest.mark.parametrize(
    "spec", [s for s in SCALAR_SPECS if s.scipy_name], ids=lambda s: s.name
)
def test_scipy_density_cdf_and_quantile_differential(spec):
    scipy_stats = pytest.importorskip("scipy.stats")
    ref = getattr(scipy_stats, spec.scipy_name)(*spec.scipy_args, **spec.scipy_kwargs)
    ours = spec.factory()
    for x in spec.probes:
        if spec.density_contract:
            ours_density = _float(ours.pdf(x))
            ref_density = float(
                ref.pmf(x) if ours.measure_type is MeasureType.DISCRETE else ref.pdf(x)
            )
            assert math.isclose(ours_density, ref_density, rel_tol=2e-6, abs_tol=2e-8)
        if spec.cdf_contract:
            assert math.isclose(
                _float(ours.cdf(x)), float(ref.cdf(x)), rel_tol=2e-6, abs_tol=2e-8
            )
    if spec.quantile_contract:
        for p in (0.1, 0.5, 0.9):
            assert math.isclose(
                _float(ours.quantile(p)), float(ref.ppf(p)), rel_tol=4e-6, abs_tol=4e-6
            )


@pytest.mark.parametrize(
    "spec", [s for s in SCALAR_SPECS if s.normalization_contract], ids=lambda s: s.name
)
def test_numerically_tractable_normalization(spec):
    scipy_integrate = pytest.importorskip("scipy.integrate")
    dist = spec.factory()
    support = dist.support
    if dist.measure_type is MeasureType.DISCRETE:
        # Infinite discrete supports are checked by their SciPy differential CDF/PMF contracts;
        # exact finite supports can be summed directly.
        if isinstance(support, sp.FiniteSet):
            total = sum(_float(dist.pmf(k)) for k in support)
            assert math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9)
        return
    if isinstance(support, sp.Interval):
        lo = -np.inf if support.start is -sp.oo else float(support.start)
        hi = np.inf if support.end is sp.oo else float(support.end)
    elif support == sp.S.Reals:
        lo, hi = -np.inf, np.inf
    else:
        return
    # Symbolic integral representations are too expensive for this numerical contract.
    if spec.name in {"NoncentralT", "NoncentralF", "NoncentralChiSquared", "VonMises"}:
        return
    value, _ = scipy_integrate.quad(
        lambda z: max(0.0, _float(dist.pdf(z))), lo, hi, limit=100
    )
    assert math.isclose(value, 1.0, rel_tol=3e-5, abs_tol=3e-5)
