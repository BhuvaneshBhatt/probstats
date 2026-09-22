import numpy as np
import pytest

from probstats.random_matrix import (
    GOE,
    GSE,
    GUE,
    MarchenkoPastur,
    TracyWidom,
    WignerSemicircle,
    WishartEnsemble,
    available_tracy_widom_backends,
    eigenvalue_statistics,
    gaussian_largest_eigenvalue_scaling,
    register_tracy_widom_backend,
    sample_spectral_statistic,
    unregister_tracy_widom_backend,
)


def test_gaussian_ensemble_matrix_symmetry_and_eigenvalue_shapes():
    for cls, complex_expected in [(GOE, False), (GUE, True), (GSE, False)]:
        ensemble = cls(6)
        matrix = ensemble.sample_matrix(123)
        assert matrix.shape == (6, 6)
        assert np.allclose(matrix, matrix.conj().T)
        if complex_expected:
            assert np.iscomplexobj(matrix)
        values = ensemble.sample_eigenvalues(size=4, rng=321)
        assert values.shape == (4, 6)
        assert np.all(np.diff(values, axis=-1) >= 0)


def test_random_matrix_validation_rejects_ambiguous_or_nonfinite_inputs():
    assert GOE(np.int64(3)).dimension == 3
    with pytest.raises(TypeError):
        GOE(3.0)
    with pytest.raises(TypeError):
        GUE(3).sample_eigenvalues(size=True)
    with pytest.raises(ValueError):
        MarchenkoPastur(np.nan)
    with pytest.raises(ValueError):
        MarchenkoPastur(1.0).cdf(1.0, quadrature_points=0)
    with pytest.raises(ValueError):
        WishartEnsemble(2, 3, scale=[[1.0, np.nan], [np.nan, 1.0]])


def test_eigenvalue_statistics_require_a_finite_hermitian_matrix():
    stats = eigenvalue_statistics([[2.0, 1.0], [1.0, 3.0]])
    assert stats["trace"] == pytest.approx(5.0)
    with pytest.raises(ValueError, match="Hermitian"):
        eigenvalue_statistics([[1.0, 10.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="finite"):
        eigenvalue_statistics([[1.0, np.nan], [np.nan, 1.0]])


def test_wigner_semicircle_normalization_cdf_and_moments():
    law = WignerSemicircle()
    grid = np.linspace(-2, 2, 10001)
    assert np.trapezoid(law.pdf(grid), grid) == pytest.approx(1, rel=2e-4)
    assert law.cdf(-2) == 0
    assert law.cdf(0) == pytest.approx(0.5)
    assert law.cdf(2) == 1
    draws = law.sample(50_000, rng=10)
    assert draws.mean() == pytest.approx(0, abs=0.02)
    assert draws.var() == pytest.approx(1, abs=0.03)


def test_gaussian_ensemble_converges_toward_semicircle_second_moment():
    ensemble = GUE(100)
    values = ensemble.sample_eigenvalues(size=100, rng=11)
    assert np.mean(values) == pytest.approx(0, abs=0.03)
    assert np.mean(values**2) == pytest.approx(1, abs=0.06)


def test_marchenko_pastur_density_mass_edges_and_atom():
    law = MarchenkoPastur(0.5)
    grid = np.linspace(law.lower_edge, law.upper_edge, 8000)
    assert np.trapezoid(law.pdf(grid), grid) == pytest.approx(1, abs=2e-3)
    assert law.cdf(law.upper_edge) == pytest.approx(1)
    assert law.atom_at_zero == 0

    singular = MarchenkoPastur(2.0)
    assert singular.atom_at_zero == pytest.approx(0.5)
    grid = np.linspace(singular.lower_edge, singular.upper_edge, 8000)
    assert np.trapezoid(singular.pdf(grid), grid) == pytest.approx(0.5, abs=2e-3)
    assert singular.cdf(0) == pytest.approx(0.5)


def test_wishart_empirical_spectrum_matches_mp_first_two_moments():
    p, n = 50, 100
    ensemble = WishartEnsemble(p, n, normalized=True)
    values = ensemble.sample_eigenvalues(size=80, rng=12)
    mp = MarchenkoPastur(p / n)
    assert values.mean() == pytest.approx(mp.mean, abs=0.03)
    assert values.var() == pytest.approx(mp.variance, abs=0.07)


def test_wishart_trace_and_bartlett_determinant_laws():
    ensemble = WishartEnsemble(3, 7, normalized=False)
    trace = ensemble.trace_distribution()
    assert float(trace.mean_value) == pytest.approx(21)
    assert float(trace.variance_value) == pytest.approx(42)

    determinant = ensemble.determinant_law()
    assert determinant.chi_square_dfs == (7, 6, 5)
    assert determinant.mean == pytest.approx(210)
    draws = determinant.sample(30_000, rng=20)
    direct = sample_spectral_statistic(ensemble, "determinant", size=8_000, rng=21)
    assert draws.mean() == pytest.approx(210, rel=0.04)
    assert direct.mean() == pytest.approx(210, rel=0.08)


def test_trace_distribution_matches_goe_sampling():
    ensemble = GOE(30)
    expected = ensemble.trace_distribution()
    draws = sample_spectral_statistic(ensemble, "trace", size=5000, rng=33)
    assert draws.mean() == pytest.approx(0, abs=0.05)
    assert draws.var() == pytest.approx(float(expected.variance_value), rel=0.06)


def test_soft_edge_scaling_and_largest_eigenvalues():
    scaling = gaussian_largest_eigenvalue_scaling(100, beta=2)
    assert scaling.center == 2
    assert scaling.scale == pytest.approx(100 ** (-2 / 3))
    values = GUE(100).largest_eigenvalue(size=100, rng=8)
    standardized = scaling.standardize(values)
    assert np.isfinite(standardized).all()
    assert np.median(values) < 2.1


def test_tracy_widom_backend_protocol():
    class LogisticBackend:
        def cdf(self, x, beta):
            x = np.asarray(x)
            return 1 / (1 + np.exp(-x))

        def pdf(self, x, beta):
            c = self.cdf(x, beta)
            return c * (1 - c)

        def ppf(self, p, beta):
            p = np.asarray(p)
            return np.log(p / (1 - p))

    register_tracy_widom_backend("test-logistic", LogisticBackend())
    try:
        assert "test-logistic" in available_tracy_widom_backends()
        tw = TracyWidom(2, backend="test-logistic")
        assert tw.cdf(0) == pytest.approx(0.5)
        assert tw.ppf(0.5) == pytest.approx(0)
        with pytest.raises(ValueError, match="already registered"):
            register_tracy_widom_backend("test-logistic", LogisticBackend())
    finally:
        unregister_tracy_widom_backend("test-logistic")


def test_tracy_widom_without_backend_fails_explicitly():
    with pytest.raises(RuntimeError, match="no Tracy-Widom numerical backend"):
        TracyWidom(2).cdf(0)


def test_largest_eigenvalue_approximation_transforms_backend_values():
    from probstats.random_matrix import gaussian_largest_eigenvalue_approximation

    class StandardNormalLike:
        def cdf(self, x, beta):
            return np.asarray(x) * 0 + 0.25

        def pdf(self, x, beta):
            return np.asarray(x) * 0 + 2.0

        def ppf(self, p, beta):
            return np.asarray(p) * 0 + 1.5

    approximation = gaussian_largest_eigenvalue_approximation(
        8, beta=2, backend=StandardNormalLike()
    )
    assert approximation.cdf(2) == pytest.approx(0.25)
    assert approximation.ppf(0.5) == pytest.approx(2 + 1.5 * 8 ** (-2 / 3))


def test_ensemble_matrix_sampling_supports_batch_shapes():
    goe = GOE(4)
    matrices = goe.sample_matrix(rng=123, size=(2, 3))
    assert matrices.shape == (2, 3, 4, 4)
    assert np.allclose(matrices, np.swapaxes(matrices, -1, -2))

    wishart = WishartEnsemble(3, 6)
    samples = wishart.sample_matrix(rng=123, size=5)
    assert samples.shape == (5, 3, 3)
    assert np.allclose(samples, np.swapaxes(samples, -1, -2))


def test_marchenko_pastur_array_cdf_matches_scalar_path():
    for ratio in (0.25, 1.0, 2.0):
        law = MarchenkoPastur(ratio)
        x = np.linspace(law.lower_edge, law.upper_edge, 41)
        batched = law.cdf(x)
        scalar = np.array([law.cdf(float(value)) for value in x])
        assert np.all(np.diff(batched) >= 0)
        assert batched == pytest.approx(scalar, abs=3e-10)
