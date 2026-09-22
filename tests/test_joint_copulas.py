import numpy as np
import pytest
import sympy as sp

from probstats import (
    JointDistribution,
    MultivariateNormal,
    Normal,
    Uniform,
)
from probstats.joint import (
    CopulaDistribution,
    GaussianCopula,
    IndependenceCopula,
    dependent_convolution,
    dependent_linear_combination,
)
from probstats.symbolic import (
    ProofStatus,
    multivariate_characteristic_function,
    multivariate_moment_generating_function,
    prove_equivalent_by_transform,
)


def test_joint_distribution_marginal_and_conditional():
    x, y = sp.symbols("x y", real=True)
    density = sp.Rational(1, 2) * sp.exp(-x) * sp.exp(-y / 2)
    joint = JointDistribution(
        (x, y), density, (sp.Interval(0, sp.oo), sp.Interval(0, sp.oo))
    )
    mx = joint.marginal(0)
    assert sp.simplify(mx.pdf(x) - sp.exp(-x)) == 0
    cond = joint.conditional((0,), {1: 3})
    assert sp.simplify(cond.pdf(x) - sp.exp(-x)) == 0


def test_independence_copula_recovers_product_density():
    x, y = sp.symbols("x y", real=True)
    d = CopulaDistribution((Normal(0, 1), Normal(2, 3)), IndependenceCopula(2))
    assert sp.simplify(d.pdf((x, y)) - Normal(0, 1).pdf(x) * Normal(2, 3).pdf(y)) == 0


def test_gaussian_copula_density_dimension_two():
    u, v, rho = sp.symbols("u v rho", real=True)
    copula = GaussianCopula(sp.ImmutableMatrix([[1, rho], [rho, 1]]))
    density = copula.density((u, v))
    assert density.has(sp.erfinv)
    assert sp.simplify(density.subs(rho, 0) - 1) == 0


def test_gaussian_copula_marginal_submatrix():
    r = sp.ImmutableMatrix(
        [
            [1, sp.Rational(1, 4), 0],
            [sp.Rational(1, 4), 1, sp.Rational(1, 3)],
            [0, sp.Rational(1, 3), 1],
        ]
    )
    d = CopulaDistribution(
        (Normal(0, 1), Uniform(0, 1), Normal(1, 2)), GaussianCopula(r)
    )
    m = d.marginal((0, 2))
    assert isinstance(m.copula, GaussianCopula)
    assert m.copula.correlation == sp.eye(2)


def test_dependent_linear_combination_mvn_includes_covariance():
    d = MultivariateNormal([1, 2], [[4, 3], [3, 9]])
    out = dependent_linear_combination(d, (1, 2), 5)
    assert isinstance(out, Normal)
    assert sp.simplify(out.mean - 10) == 0
    assert sp.simplify(out.sigma**2 - 52) == 0


def test_dependent_convolution_from_exact_joint_density():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        (x, y),
        sp.Rational(1, 2) * sp.exp(-x) * sp.exp(-y / 2),
        (sp.Interval(0, sp.oo), sp.Interval(0, sp.oo)),
    )
    summed = dependent_convolution(joint)
    z = summed.value_symbol
    # Hypoexponential sum: exp(-z/2) - exp(-z), z >= 0.
    assert sp.simplify(summed.pdf(z) - (sp.exp(-z / 2) - sp.exp(-z))) == 0
    assert summed.support == sp.Interval(0, sp.oo)


def test_multivariate_normal_characteristic_and_mgf():
    t1, t2 = sp.symbols("t1 t2", real=True)
    d = MultivariateNormal([1, 2], [[2, 1], [1, 3]])
    cf = multivariate_characteristic_function(d, (t1, t2))
    mgf = multivariate_moment_generating_function(d, (t1, t2))
    assert sp.simplify(cf.subs({t1: 0, t2: 0}) - 1) == 0
    assert sp.simplify(mgf.subs({t1: 0, t2: 0}) - 1) == 0
    assert sp.simplify(sp.diff(sp.log(mgf), t1).subs({t1: 0, t2: 0}) - 1) == 0
    assert sp.simplify(sp.diff(sp.log(mgf), t2).subs({t1: 0, t2: 0}) - 2) == 0


def test_transform_uniqueness_proves_equal_normals_and_disproves_unequal():
    same = prove_equivalent_by_transform(Normal(0, 1), Normal(0, 1))
    assert same.status is ProofStatus.PROVED
    assert same.transform_kind == "characteristic"
    different = prove_equivalent_by_transform(Normal(0, 1), Normal(1, 1))
    assert different.status is ProofStatus.DISPROVED


def test_multivariate_transform_uniqueness():
    a = MultivariateNormal([0, 0], [[1, 0], [0, 1]])
    b = MultivariateNormal([0, 0], sp.eye(2))
    proof = prove_equivalent_by_transform(a, b)
    assert proof.status is ProofStatus.PROVED


def test_dependent_discrete_convolution_finite_joint():
    x, y = sp.symbols("x y", integer=True)
    # Perfect dependence on {0,1}: P(0,0)=P(1,1)=1/2.
    pmf = sp.Piecewise(
        (
            sp.Rational(1, 2),
            sp.Or(sp.And(sp.Eq(x, 0), sp.Eq(y, 0)), sp.And(sp.Eq(x, 1), sp.Eq(y, 1))),
        ),
        (0, True),
    )
    joint = JointDistribution(
        (x, y), pmf, (sp.FiniteSet(0, 1), sp.FiniteSet(0, 1)), discrete=True
    )
    summed = dependent_convolution(joint)
    assert sp.simplify(summed.pmf(0) - sp.Rational(1, 2)) == 0
    assert sp.simplify(summed.pmf(1)) == 0
    assert sp.simplify(summed.pmf(2) - sp.Rational(1, 2)) == 0


def test_copula_sampling_shape_and_marginal_scale():
    d = CopulaDistribution((Normal(0, 1), Uniform(-2, 4)), IndependenceCopula(2))
    draws = d.sample(size=20, rng=123)
    assert draws.shape == (20, 2)
    assert ((draws[:, 1] >= -2) & (draws[:, 1] <= 4)).all()


def test_joint_indices_reject_fractional_values():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        (x, y),
        sp.Rational(1, 4),
        (sp.Interval(-1, 1), sp.Interval(-1, 1)),
    )
    with pytest.raises(TypeError, match="index must be an integer"):
        joint.marginal(0.5)


def test_conditional_rejects_value_outside_support():
    x, y = sp.symbols("x y", real=True)
    joint = JointDistribution(
        (x, y),
        sp.Rational(1, 4),
        (sp.Interval(-1, 1), sp.Interval(-1, 1)),
    )
    with pytest.raises(ValueError, match="outside support"):
        joint.conditional(0, {1: 2})


def test_copula_sampling_uses_strict_numpy_style_sample_shapes():
    copula = IndependenceCopula(np.int64(2))
    assert copula.sample_uniform(size=np.int64(3), rng=1).shape == (3, 2)
    with pytest.raises(TypeError, match="size"):
        copula.sample_uniform(size=2.5, rng=1)


def test_gaussian_copula_rejects_indefinite_correlation():
    with pytest.raises(ValueError, match="positive definite"):
        GaussianCopula([[1, 2], [2, 1]])
