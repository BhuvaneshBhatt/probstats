import numpy as np
import pytest
import sympy as sp

from probstats.algebraic.testing import (
    SemialgebraicHypothesis,
    sdl_calibrate,
    sdl_diagnostics,
    sdl_test,
)


def hypothesis():
    theta = sp.Symbol("theta")
    return SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta,), estimators={theta: lambda x: x}
    )


def result():
    return sdl_test(
        [-2.0, -1.0, 0.0, 1.0],
        hypothesis(),
        budget=4,
        bootstrap_replicates=101,
        seed=14,
    )


def test_diagnostics_reuse_bootstrap():
    r = result()
    d = sdl_diagnostics(r)
    assert d.realized_budget == r.realized_subsets
    assert d.requested_budget == 4
    assert d.realized_budget_ratio == pytest.approx(r.realized_subsets / 4)
    assert d.alpha_n == r.studentization.alpha
    assert d.kernel_order == 1
    assert d.projection_size == 4
    assert set(d.bootstrap_critical_values) == {0.90, 0.95, 0.99}
    assert d.bootstrap_critical_values[0.95] == pytest.approx(
        np.quantile(r.bootstrap.statistics, 0.95)
    )


def test_variance_shares():
    d = sdl_diagnostics(result())
    mask = ~np.isnan(d.hajek_variance_share)
    np.testing.assert_allclose(
        d.hajek_variance_share[mask] + d.kernel_variance_share[mask], 1.0
    )


def test_bootstrap_pvalue_standard_error_formula():
    r = result()
    d = sdl_diagnostics(r)
    expected = np.sqrt(r.p_value * (1 - r.p_value) / r.bootstrap_replicates)
    assert d.bootstrap_standard_error == pytest.approx(expected)


def test_custom_critical_levels_validated():
    d = sdl_diagnostics(result(), critical_levels=(0.8,))
    assert tuple(d.bootstrap_critical_values) == (0.8,)
    with pytest.raises(ValueError):
        sdl_diagnostics(result(), critical_levels=(1.0,))


def test_calibration_reproducibility():
    def sampler(rng):
        return rng.normal(loc=-0.5, scale=1.0, size=8)

    kwargs = {
        "repetitions": 12,
        "alpha": 0.1,
        "budget": 8,
        "bootstrap_replicates": 17,
        "seed": 91,
    }
    a = sdl_calibrate(sampler, hypothesis(), **kwargs)
    b = sdl_calibrate(sampler, hypothesis(), **kwargs)
    np.testing.assert_array_equal(a.p_values, b.p_values)
    np.testing.assert_array_equal(a.statistics, b.statistics)
    np.testing.assert_array_equal(a.realized_budgets, b.realized_budgets)
    assert a.rejection_rate == np.mean(a.p_values <= 0.1)
    assert a.requested_budget == 8
    np.testing.assert_array_equal(a.kernel_orders, np.ones(12, dtype=int))
    np.testing.assert_array_equal(a.projection_sizes, np.full(12, 8))


def test_calibration_mcse_formula():
    def sampler(rng):
        return rng.normal(loc=-1.0, size=6)

    r = sdl_calibrate(
        sampler,
        hypothesis(),
        repetitions=10,
        budget=6,
        bootstrap_replicates=11,
        seed=3,
    )
    expected = np.sqrt(r.rejection_rate * (1 - r.rejection_rate) / 10)
    assert r.monte_carlo_standard_error == pytest.approx(expected)


def test_calibration_rejects_empty_sample():
    with pytest.raises(ValueError):
        sdl_calibrate(lambda rng: (), hypothesis(), repetitions=1, budget=1, seed=1)


def test_calibration_forwards_kernel_controls():
    theta = sp.Symbol("theta")
    h = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta**2 - 1,), estimators={theta: lambda x: x}
    )
    r = sdl_calibrate(
        lambda rng: rng.normal(size=5),
        h,
        repetitions=3,
        budget=5,
        bootstrap_replicates=7,
        seed=8,
        augment_constraints_count=1,
        kernel_symmetrization="random",
        symmetrization_permutations=3,
    )
    np.testing.assert_array_equal(r.kernel_orders, np.full(3, 2))


def test_zero_standard_error_diagnostics_distinguish_degeneracy():
    hypothesis = globals()["hypothesis"]()
    result = sdl_test(
        [0.0, 0.0, 0.0, 0.0],
        hypothesis,
        budget=4,
        bootstrap_replicates=5,
        seed=12,
    )
    diagnostics = sdl_diagnostics(result)
    assert np.array_equal(diagnostics.zero_over_zero, diagnostics.zero_standard_error)
    assert not np.any(diagnostics.nonzero_over_zero)


def test_calibration_data_stream_is_independent_of_test_randomness():
    samples_a = []
    samples_b = []

    def sampler_a(rng):
        sample = rng.normal(size=6)
        samples_a.append(sample.copy())
        return sample

    def sampler_b(rng):
        sample = rng.normal(size=6)
        samples_b.append(sample.copy())
        return sample

    sdl_calibrate(
        sampler_a,
        hypothesis(),
        repetitions=4,
        budget=6,
        bootstrap_replicates=5,
        seed=44,
    )
    sdl_calibrate(
        sampler_b,
        hypothesis(),
        repetitions=4,
        budget=6,
        bootstrap_replicates=19,
        seed=44,
    )
    for left, right in zip(samples_a, samples_b, strict=True):
        np.testing.assert_array_equal(left, right)
