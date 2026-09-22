import numpy as np
import sympy as sp

from probstats.distributions import (
    LKJ,
    LKJCholesky,
    MatrixNormal,
)
from probstats.spaces import (
    CorrelationCholeskySpace,
    CorrelationMatrixSpace,
)
from probstats.transforms import (
    CorrelationCholeskyTransform,
    CorrelationMatrixTransform,
)


def test_matrix_normal_density_functionals_and_sampling():
    d = MatrixNormal([[1, 2], [3, 4]], sp.eye(2), 2 * sp.eye(2))
    assert d.event_space.shape == (2, 2)
    assert d.mean_value == sp.ImmutableMatrix([[1, 2], [3, 4]])
    assert d.variance_value == sp.KroneckerProduct(2 * sp.eye(2), sp.eye(2))
    assert sp.simplify(d.entropy() - (2 * (1 + sp.log(2 * sp.pi)) + 2 * sp.log(2))) == 0
    assert d.logpdf(d.mean) == sp.simplify(-2 * sp.log(2 * sp.pi) - sp.log(4))
    x = d.sample((3, 4), rng=7)
    assert x.shape == (3, 4, 2, 2)


def test_lkj_dimension_two_normalization_and_functionals():
    eta, rho = sp.symbols("eta rho", positive=True)
    d = LKJ(2, eta)
    r = sp.ImmutableMatrix([[1, rho], [rho, 1]])
    expected_logz = sp.log(sp.beta(sp.Rational(1, 2), eta))
    assert sp.simplify(d.log_normalizing_constant - expected_logz) == 0
    assert (
        sp.simplify(d.logpdf(r) - ((eta - 1) * sp.log(1 - rho**2) - expected_logz)) == 0
    )
    assert d.mean_value == sp.eye(2)
    assert d.variance_value[0, 0] == 0
    assert d.variance_value[0, 1] == 1 / (2 * eta + 1)


def test_lkj_eta_one_dimension_two_is_uniform_correlation():
    rho = sp.symbols("rho", real=True)
    d = LKJ(2, 1)
    r = sp.ImmutableMatrix([[1, rho], [rho, 1]])
    assert sp.simplify(d.pdf(r) - sp.Rational(1, 2)) == 0


def test_lkj_cholesky_density_matches_dimension_two_parameterization():
    eta, rho = sp.symbols("eta rho", positive=True)
    l = sp.ImmutableMatrix([[1, 0], [rho, sp.sqrt(1 - rho**2)]])
    d = LKJCholesky(2, eta)
    expected = (eta - 1) * sp.log(1 - rho**2) - sp.log(sp.beta(sp.Rational(1, 2), eta))
    assert sp.simplify(d.logpdf(l) - expected) == 0


def test_correlation_cholesky_space_membership():
    l = sp.ImmutableMatrix([[1, 0], [sp.Rational(3, 5), sp.Rational(4, 5)]])
    assert sp.simplify(CorrelationCholeskySpace(2).contains(l)) is sp.true
    bad = sp.ImmutableMatrix([[1, 1], [0, 1]])
    assert sp.simplify(CorrelationCholeskySpace(2).contains(bad)) is sp.false


def test_correlation_transforms_symbolic_dimension_two():
    y = sp.symbols("y", real=True)
    ct = CorrelationCholeskyTransform(2)
    l = ct.apply([y])
    assert l[1, 0] == sp.tanh(y)
    assert sp.simplify(l[1, 1] ** 2 - (1 - sp.tanh(y) ** 2)) == 0
    assert sp.simplify(ct.invert(l)[0] - sp.atanh(sp.tanh(y))) == 0

    rt = CorrelationMatrixTransform(2)
    r = rt.apply([y])
    assert sp.simplify(r[0, 1] - sp.tanh(y)) == 0
    assert sp.simplify(r.det() - (1 - sp.tanh(y) ** 2)) == 0


def test_correlation_transform_numeric_round_trip():
    transform = CorrelationMatrixTransform(4)
    y = np.array([-0.7, 0.2, 1.0, -0.4, 0.8, -1.1])
    r = transform.apply_numeric(y)
    assert r.shape == (4, 4)
    assert np.allclose(np.diag(r), 1.0)
    assert np.all(np.linalg.eigvalsh(r) > 0)
    assert np.allclose(transform.invert_numeric(r), y, atol=1e-10)


def test_lkj_sampling_shapes_and_support():
    d = LKJ(4, sp.Rational(3, 2))
    draws = d.sample((5, 3), rng=123)
    assert draws.shape == (5, 3, 4, 4)
    for r in draws.reshape(-1, 4, 4):
        assert np.allclose(r, r.T)
        assert np.allclose(np.diag(r), 1.0)
        assert np.all(np.linalg.eigvalsh(r) > 0)

    chol = LKJCholesky(4, 1).sample(7, rng=123)
    assert chol.shape == (7, 4, 4)
    assert np.allclose(np.triu(chol, 1), 0)
    assert np.all(np.diagonal(chol, axis1=-2, axis2=-1) > 0)
    assert np.allclose(np.sum(chol * chol, axis=-1), 1.0)


def test_lkj_parameter_and_event_spaces():
    eta = sp.symbols("eta", positive=True)
    d = LKJ(3, eta)
    assert d.parameter_space.names == ("dimension", "eta")
    assert sp.simplify(d.parameter_space_constraints) is sp.true
    assert isinstance(d.event_space, CorrelationMatrixSpace)
    assert isinstance(LKJCholesky(3, eta).event_space, CorrelationCholeskySpace)


def test_correlation_transform_jacobians_dimension_two():
    y = sp.symbols("y", real=True)
    expected = sp.log(1 - sp.tanh(y) ** 2)
    assert (
        sp.simplify(
            CorrelationCholeskyTransform(2).log_abs_det_jacobian([y]) - expected
        )
        == 0
    )
    assert (
        sp.simplify(CorrelationMatrixTransform(2).log_abs_det_jacobian([y]) - expected)
        == 0
    )


def test_lkj_entropy_and_cholesky_functionals():
    eta = sp.symbols("eta", positive=True)
    d = LKJ(2, eta)
    # At eta=1 the 2D LKJ is Uniform(-1, 1) in rho, hence entropy log(2).
    assert sp.simplify(d.entropy().subs(eta, 1) - sp.log(2)) == 0

    c = LKJCholesky(2, 1)
    assert c.mean_value[1, 0] == 0
    assert sp.simplify(c.variance_value[1, 0] - sp.Rational(1, 3)) == 0
    assert sp.simplify(c.entropy() - sp.log(2)) == 0
