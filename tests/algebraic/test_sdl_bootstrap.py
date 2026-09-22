import numpy as np
import pytest

from probstats.algebraic.testing import sdl_multiplier_bootstrap, sdl_studentize


def mean_kernel(x):
    return np.array([x], dtype=float)


mean_kernel.order = 1


def pair_kernel(x, y):
    return np.array([(x + y) / 2, x * y], dtype=float)


pair_kernel.order = 2


def test_bootstrap_order_one_matches_manual_multiplier_formula():
    sample = [1.0, 2.0, 4.0, 8.0]
    st = sdl_studentize(sample, mean_kernel, budget=4, rng=11)
    result = sdl_multiplier_bootstrap(st, replicates=7, rng=123, retain_components=True)
    gen = np.random.default_rng(123)
    xi_h = gen.standard_normal((7, 4))
    xi_g = gen.standard_normal((7, 4))
    h_centered = st.kernel_values - np.asarray(st.u_statistic.value).reshape(-1)
    g_centered = st.hajek.values - st.hajek.mean
    uh = xi_h @ h_centered / 2
    ug = xi_g @ g_centered / 2
    expected = ug + uh  # alpha=n/N=1, m=1
    np.testing.assert_allclose(result.h_component, uh)
    np.testing.assert_allclose(result.g_component, ug)
    np.testing.assert_allclose(result.combined, expected)
    np.testing.assert_allclose(
        result.statistics, np.max(expected / st.standard_error, axis=1)
    )


def test_bootstrap_is_reproducible():
    sample = np.arange(1.0, 9.0)
    st = sdl_studentize(sample, pair_kernel, budget=8, rng=42)
    a = sdl_multiplier_bootstrap(st, replicates=31, rng=7)
    b = sdl_multiplier_bootstrap(st, replicates=31, rng=7)
    np.testing.assert_array_equal(a.statistics, b.statistics)
    assert a.p_value == b.p_value


def test_bootstrap_batch_reproducibility():
    sample = np.arange(1.0, 9.0)
    st = sdl_studentize(sample, pair_kernel, budget=8, rng=42)
    a = sdl_multiplier_bootstrap(st, replicates=17, rng=9, batch_size=4)
    b = sdl_multiplier_bootstrap(st, replicates=17, rng=9, batch_size=4)
    np.testing.assert_array_equal(a.statistics, b.statistics)


def test_p_value_is_empirical_exceedance_fraction():
    sample = np.arange(1.0, 9.0)
    st = sdl_studentize(sample, pair_kernel, budget=8, rng=42)
    result = sdl_multiplier_bootstrap(st, replicates=53, rng=8)
    assert result.p_value == np.mean(result.statistics >= st.statistic)
    assert 0 <= result.p_value <= 1


@pytest.mark.parametrize("replicates", [0, -1])
def test_replicates_must_be_positive(replicates):
    sample = [1.0, 2.0, 3.0]
    st = sdl_studentize(sample, mean_kernel, budget=3, rng=1)
    with pytest.raises(ValueError):
        sdl_multiplier_bootstrap(st, replicates=replicates)


def test_components_not_retained_by_default():
    sample = [1.0, 2.0, 3.0]
    st = sdl_studentize(sample, mean_kernel, budget=3, rng=1)
    result = sdl_multiplier_bootstrap(st, replicates=5, rng=2)
    assert result.h_component is None
    assert result.g_component is None
    assert result.combined is None


def test_batch_size_is_not_an_exact_stream_invariance_contract():
    sample = np.arange(1.0, 9.0)
    st = sdl_studentize(sample, pair_kernel, budget=8, rng=42)
    unbatched = sdl_multiplier_bootstrap(st, replicates=19, rng=13)
    batched = sdl_multiplier_bootstrap(st, replicates=19, rng=13, batch_size=4)
    assert not np.array_equal(unbatched.statistics, batched.statistics)
