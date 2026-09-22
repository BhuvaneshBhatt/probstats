import numpy as np
import sympy as sp

from probstats.distributions import (
    BetaPrime,
    DiscreteUniform,
    FDistribution,
    Frechet,
    Gumbel,
    Hypergeometric,
    InverseGaussian,
    Kumaraswamy,
    Maxwell,
    Nakagami,
    NoncentralChiSquared,
    NoncentralF,
    NoncentralT,
    PowerDistribution,
    Rayleigh,
    Rice,
    Skellam,
    Triangular,
    VonMises,
    Zipf,
)


def test_f_distribution_matches_beta_prime_canonical_form():
    f = FDistribution(6, 10)
    canonical = f.canonicalize()
    assert canonical == BetaPrime(3, 5, sp.Rational(5, 3))
    assert sp.simplify(f.pdf(1) - canonical.pdf(1)) == 0
    assert f.mean_value == sp.Rational(5, 4)


def test_noncentral_chi_squared_moments_and_central_limit():
    d = NoncentralChiSquared(4, 3)
    assert d.mean_value == 7
    assert d.variance_value == 20
    x = sp.Symbol("x", positive=True)
    central = NoncentralChiSquared(4, 0)
    expected = x * sp.exp(-x / 2) / 4
    assert sp.simplify(central.pdf(x).subs(x, 2) - expected.subs(x, 2)) == 0


def test_noncentral_f_and_t_sampling_conventions():
    f = NoncentralF(5, 20, 2)
    draws = f.sample(20_000, rng=123)
    assert abs(float(np.mean(draws)) - float(f.mean_value)) < 0.08
    t = NoncentralT(20, 1.5)
    assert NoncentralT(20, 0).canonicalize().df == 20
    assert NoncentralF(5, 20, 0).canonicalize() == FDistribution(5, 20).canonicalize()
    assert NoncentralChiSquared(4, 0).canonicalize().shape == 2
    assert t.moment_generating_function(0) == 1
    assert t.moment_generating_function(1) == sp.oo
    draws_t = t.sample(20_000, rng=123)
    assert abs(float(np.mean(draws_t)) - float(sp.N(t.mean_value))) < 0.06


def test_discrete_uniform_exact_functions_and_sampling():
    d = DiscreteUniform(2, 5)
    assert d.pmf(3) == sp.Rational(1, 4)
    assert d.cdf(3) == sp.Rational(1, 2)
    assert d.quantile(sp.Rational(3, 4)) == 4
    assert d.mean_value == sp.Rational(7, 2)
    assert set(np.unique(d.sample(500, rng=4))) <= {2, 3, 4, 5}


def test_hypergeometric_mass_moments_and_factorial_moment():
    d = Hypergeometric(10, 20, 5)
    assert sp.simplify(sum(d.pmf(k) for k in d.support) - 1) == 0
    assert d.mean_value == sp.Rational(5, 3)
    assert d.factorial_moment(2) == sp.Rational(60, 29)


def test_skellam_mass_symmetry_and_characteristic_function():
    symmetric = Skellam(2, 2)
    assert sp.simplify(symmetric.pmf(-3) - symmetric.pmf(3)) == 0
    d = Skellam(2, 3)
    assert d.mean_value == -1
    assert d.variance_value == 5
    assert d.characteristic_function(0) == 1


def test_zipf_mass_cdf_and_moments():
    d = Zipf(4)
    assert d.pmf(1) == 1 / sp.zeta(4)
    assert d.cdf(2) == sp.harmonic(2, 4) / sp.zeta(4)
    assert d.mean_value == sp.zeta(3) / sp.zeta(4)
    assert d.probability_generating_function(sp.Symbol("z")) == sp.polylog(
        4, sp.Symbol("z")
    ) / sp.zeta(4)


def test_inverse_gaussian_functions_and_sampling():
    d = InverseGaussian(2, 3)
    assert d.mean_value == 2
    assert d.variance_value == sp.Rational(8, 3)
    assert d.cdf(0) == 0
    assert d.moment_generating_function(0) == 1
    draws = d.sample(30_000, rng=1)
    assert abs(float(np.mean(draws)) - 2) < 0.05


def test_beta_prime_density_and_moments():
    d = BetaPrime(3, 5, 2)
    assert d.cdf(0) == 0
    assert d.mean_value == sp.Rational(3, 2)
    assert d.variance_value == sp.Rational(7, 4)
    draws = d.sample(20_000, rng=2)
    assert abs(float(np.mean(draws)) - 1.5) < 0.06


def test_rayleigh_and_maxwell_moments():
    r = Rayleigh(2)
    assert sp.simplify(r.moment(2) - 8) == 0
    assert sp.simplify(r.quantile(sp.Rational(1, 2)) - 2 * sp.sqrt(2 * sp.log(2))) == 0
    m = Maxwell(2)
    assert sp.simplify(m.moment(2) - 12) == 0
    assert m.cdf(0) == 0


def test_rice_reduces_to_rayleigh_at_zero_noncentrality():
    rice = Rice(0, 2)
    rayleigh = Rayleigh(2)
    x = sp.Symbol("x", positive=True)
    assert sp.simplify(rice.pdf(x) - rayleigh.pdf(x)) == 0
    assert sp.simplify(rice.mean_value - rayleigh.mean_value) == 0


def test_nakagami_squared_mean_is_spread():
    d = Nakagami(2, 3)
    assert sp.simplify(d.moment(2) - 3) == 0
    draws = d.sample(20_000, rng=3)
    assert abs(float(np.mean(draws**2)) - 3) < 0.06


def test_gumbel_and_frechet_quantile_inverse_cdf():
    p = sp.Rational(2, 5)
    g = Gumbel(1, 2)
    assert sp.simplify(g.cdf(g.quantile(p)) - p) == 0
    assert g.moment_generating_function(0) == 1
    f = Frechet(3, 2, 1)
    assert sp.simplify(f.cdf(f.quantile(p)) - p) == 0


def test_kumaraswamy_power_and_triangular_functions():
    p = sp.Rational(3, 7)
    k = Kumaraswamy(2, 3)
    assert sp.simplify(k.cdf(k.quantile(p)) - p) == 0
    power = PowerDistribution(3)
    assert power.mean_value == sp.Rational(3, 4)
    tri = Triangular(0, 1, 2)
    assert tri.mean_value == 1
    assert (
        sp.simplify(sp.integrate(tri.pdf(sp.Symbol("x")), (sp.Symbol("x"), 0, 2)) - 1)
        == 0
    )


def test_triangular_endpoint_modes_are_well_defined():
    left = Triangular(0, 0, 2)
    right = Triangular(0, 2, 2)
    assert left.pdf(1) == sp.Rational(1, 2)
    assert right.pdf(1) == sp.Rational(1, 2)
    assert left.cdf(1) == sp.Rational(3, 4)
    assert right.cdf(1) == sp.Rational(1, 4)


def test_von_mises_density_normalizes_on_principal_interval():
    d = VonMises(0, 2)
    grid = np.linspace(-np.pi, np.pi, 20_001)
    vals = np.exp(2 * np.cos(grid)) / (2 * np.pi * float(sp.besseli(0, 2)))
    assert abs(float(np.trapezoid(vals, grid)) - 1) < 1e-8
    draws = d.sample(1000, rng=8)
    assert np.all(draws >= -np.pi) and np.all(draws <= np.pi)
