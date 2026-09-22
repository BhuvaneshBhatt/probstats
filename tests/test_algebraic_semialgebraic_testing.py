import pytest
import sympy as sp

from probstats.algebraic import (
    AlgebraicModel,
    BasicSemialgebraicNull,
    SemialgebraicHypothesis,
)


def test_basic_null_preserves_equalities_and_builds_sdl_constraints():
    x, y = sp.symbols("x y")
    null = BasicSemialgebraicNull(
        (x, y),
        equalities=(sp.Eq(x - y, 0),),
        inequalities=(sp.Ge(x, sp.Rational(1, 3)),),
    )
    assert null.equalities == (x - y,)
    assert null.inequalities == (sp.Rational(1, 3) - x,)
    assert null.sdl_constraints == (x - y, -x + y, sp.Rational(1, 3) - x)
    assert null.formula == sp.And(sp.Eq(x - y, 0), sp.Le(sp.Rational(1, 3) - x, 0))


def test_basic_hypothesis_has_ambient_relative_alternative():
    x, y = sp.symbols("x y")
    simplex = sp.And(x >= 0, y >= 0, x + y <= 1)
    hypothesis = SemialgebraicHypothesis.basic(
        (x, y), equalities=(x - y,), ambient=simplex
    )
    assert hypothesis.is_basic
    assert hypothesis.basic_null.equalities == (x - y,)
    assert hypothesis.null_formula == sp.And(simplex, sp.Eq(x - y, 0))
    assert hypothesis.alternative_formula == sp.And(simplex, sp.Ne(x - y, 0))


def test_finite_union_retains_component_structure():
    x = sp.symbols("x")
    left = BasicSemialgebraicNull((x,), equalities=(x + 1,), label="left")
    right = BasicSemialgebraicNull((x,), equalities=(x - 1,), label="right")
    hypothesis = SemialgebraicHypothesis((x,), (left, right))
    assert not hypothesis.is_basic
    assert hypothesis.null_formula == sp.Or(sp.Eq(x + 1, 0), sp.Eq(x - 1, 0))
    with pytest.raises(ValueError, match="union"):
        _ = hypothesis.basic_null


def test_from_algebraic_model_uses_probability_region_semantics():
    p0, p1 = sp.symbols("p0 p1")
    model = AlgebraicModel((p0, p1), inequalities=(sp.Ge(p0 - p1, 0),))
    hypothesis = SemialgebraicHypothesis.from_model(model)
    null = hypothesis.basic_null
    assert null.equalities == (p0 + p1 - 1,)
    assert null.inequalities == (-p0, -p1, -p0 + p1)
    assert hypothesis.basic_null.formula == sp.And(
        sp.Eq(p0 + p1 - 1, 0),
        sp.Le(-p0, 0),
        sp.Le(-p1, 0),
        sp.Le(-p0 + p1, 0),
    )


def test_estimator_keys_must_be_parameters_and_mappings_are_immutable():
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError, match="estimator keys"):
        SemialgebraicHypothesis.basic((x,), estimators={y: lambda data: data})
    hypothesis = SemialgebraicHypothesis.basic((x,), estimators={x: "mean"})
    with pytest.raises(TypeError):
        hypothesis.estimators[x] = "other"


def test_constraints_reject_nonpolynomial_and_strict_relations():
    x = sp.symbols("x")
    with pytest.raises(ValueError, match="polynomial"):
        BasicSemialgebraicNull((x,), equalities=(sp.sin(x),))
    with pytest.raises(TypeError, match="non-strict"):
        BasicSemialgebraicNull((x,), inequalities=(x < 0,))


def test_ambient_must_be_semialgebraic_in_declared_parameters():
    x, y = sp.symbols("x y")
    with pytest.raises(ValueError, match="undeclared symbols"):
        SemialgebraicHypothesis.basic((x,), ambient=y >= 0)
    with pytest.raises(ValueError, match="polynomial"):
        SemialgebraicHypothesis.basic((x,), ambient=sp.sin(x) >= 0)


def test_components_must_share_parameter_order():
    x, y = sp.symbols("x y")
    component = BasicSemialgebraicNull((y, x), equalities=(x - y,))
    with pytest.raises(ValueError, match="parameters in order"):
        SemialgebraicHypothesis((x, y), (component,))


def test_semialgebraic_equivalence_uses_exact_backend():
    pytest.importorskip("exprtest")
    x = sp.symbols("x")
    left = SemialgebraicHypothesis.basic((x,), inequalities=(x <= 0,))
    right = SemialgebraicHypothesis.basic((x,), inequalities=(-x >= 0,))
    different = SemialgebraicHypothesis.basic((x,), inequalities=(x <= 1,))
    assert left.semialgebraically_equivalent(right)
    assert not left.semialgebraically_equivalent(different)


def test_polynomial_kernel_symmetry():
    from probstats.algebraic.testing import polynomial_constraint_kernel

    theta = sp.symbols("theta")
    component = BasicSemialgebraicNull((theta,), inequalities=(theta**2 - theta,))
    kernel = polynomial_constraint_kernel(component, {theta: lambda x: x})
    assert kernel.order == 2
    assert kernel.dimension == 1
    assert sp.simplify(kernel(2, 5) - kernel(5, 2)) == 0
    assert sp.simplify(kernel(2, 5) - (2 * 5 - sp.Rational(2 + 5, 2))) == 0


def test_equality_produces_two_kernel_coordinates():
    from probstats.algebraic.testing import polynomial_constraint_kernel

    theta = sp.symbols("theta")
    component = BasicSemialgebraicNull((theta,), equalities=(theta**2 - 1,))
    kernel = polynomial_constraint_kernel(component, {theta: lambda x: x})
    assert kernel.dimension == 2
    first, second = kernel(2, 3)
    assert first == -second


def test_hypothesis_kernel_uses_estimators():
    from probstats.algebraic.testing import hypothesis_kernel

    theta = sp.symbols("theta")
    hypothesis = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta**2 - 1,), estimators={theta: lambda x: x}
    )
    kernel = hypothesis_kernel(hypothesis)
    assert kernel.order == 2
    assert kernel(2, 3) == 5


def test_higher_arity_estimators():
    from probstats.algebraic.testing import (
        UnbiasedParameterEstimator,
        polynomial_constraint_kernel,
    )

    theta = sp.symbols("theta")
    component = BasicSemialgebraicNull((theta,), inequalities=(theta**2,))
    estimator = UnbiasedParameterEstimator(lambda x, y: (x + y) / 2, arity=2)
    kernel = polynomial_constraint_kernel(component, {theta: estimator})
    assert kernel.order == 4
    assert kernel(1, 3, 5, 7) == kernel(7, 5, 3, 1)


def test_estimator_validation():
    from probstats.algebraic.testing import (
        UnbiasedParameterEstimator,
        polynomial_constraint_kernel,
    )

    x, y = sp.symbols("x y")
    component = BasicSemialgebraicNull((x, y), inequalities=(x * y,))
    with pytest.raises(ValueError, match="missing unbiased"):
        polynomial_constraint_kernel(component, {x: lambda z: z})
    with pytest.raises(ValueError, match="common estimator arity"):
        polynomial_constraint_kernel(
            component,
            {
                x: lambda z: z,
                y: UnbiasedParameterEstimator(lambda a, b: a + b, arity=2),
            },
        )


def test_union_kernel_rejected():
    from probstats.algebraic.testing import hypothesis_kernel

    x = sp.symbols("x")
    left = BasicSemialgebraicNull((x,), inequalities=(x,))
    right = BasicSemialgebraicNull((x,), inequalities=(-x,))
    hypothesis = SemialgebraicHypothesis(
        (x,), (left, right), estimators={x: lambda z: z}
    )
    with pytest.raises(ValueError, match="union"):
        hypothesis_kernel(hypothesis)


def test_polynomial_kernel_u_statistic():
    from probstats.algebraic.testing import polynomial_constraint_kernel
    from probstats.u_statistics import u_statistic

    theta = sp.symbols("theta")
    component = BasicSemialgebraicNull((theta,), inequalities=(theta**2,))
    kernel = polynomial_constraint_kernel(component, {theta: lambda x: x})
    assert u_statistic([1, 2, 3], kernel) == sp.Rational(11, 3)
