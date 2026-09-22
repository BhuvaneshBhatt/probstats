import numpy as np

from probstats.algebraic.testing import sdl_multiplier_bootstrap, sdl_studentize


def pair_kernel(x, y):
    return np.array([(x + y) / 2, x * y], dtype=float)


pair_kernel.order = 2


def test_studentization_variance_decomposition():
    sample = np.arange(1.0, 9.0)
    result = sdl_studentize(sample, pair_kernel, budget=8, rng=17)
    expected = (
        result.u_statistic.order**2 * result.hajek.variance
        + result.alpha * result.kernel_variance
    )
    np.testing.assert_allclose(result.variance, expected)
    np.testing.assert_allclose(result.standard_error, np.sqrt(expected))
    np.testing.assert_allclose(
        result.coordinate_statistics,
        np.sqrt(len(sample))
        * np.asarray(result.u_statistic.value)
        / result.standard_error,
    )
    assert result.statistic == np.max(result.coordinate_statistics)


def test_bootstrap_combines_hajek_and_kernel_processes():
    result = sdl_studentize(np.arange(1.0, 9.0), pair_kernel, budget=8, rng=21)
    boot = sdl_multiplier_bootstrap(
        result, replicates=23, rng=4, retain_components=True
    )
    expected = (
        result.u_statistic.order * boot.g_component
        + np.sqrt(result.alpha) * boot.h_component
    )
    np.testing.assert_allclose(boot.combined, expected)
    np.testing.assert_allclose(
        boot.statistics, np.max(expected / result.standard_error, axis=1)
    )
