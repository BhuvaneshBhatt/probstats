import pytest
import sympy as sp

from probstats import (
    Bernoulli,
    Beta,
    Binomial,
    Gamma,
    MultivariateNormal,
    Normal,
    StudentT,
)
from probstats.distributions import (
    Categorical,
    Dirichlet,
    InverseWishart,
    Multinomial,
    Wishart,
)
from probstats.spaces import (
    CorrelationMatrixSpace,
    CountVectorEventSpace,
    IntegerSpace,
    NaturalNumberSpace,
    NonnegativeRealSpace,
    OrderedVectorSpace,
    ParameterSpace,
    PositiveDefiniteMatrixSpace,
    PositiveRealSpace,
    PositiveSemidefiniteMatrixSpace,
    PositiveVectorSpace,
    RealSpace,
    SimplexEventSpace,
    UnitIntervalSpace,
)


def test_scalar_spaces_emit_semialgebraic_constraints():
    x = sp.symbols("x", real=True)
    assert PositiveRealSpace().constraint(x) == (x > 0)
    assert NonnegativeRealSpace().constraint(x) == (x >= 0)
    assert UnitIntervalSpace().constraint(x) == sp.And(x >= 0, x <= 1)
    assert UnitIntervalSpace(open_left=True, open_right=True).constraint(x) == sp.And(
        x > 0, x < 1
    )
    assert IntegerSpace(2, 5).contains(3) is sp.true
    assert IntegerSpace(2, 5).contains(6) is sp.false
    assert NaturalNumberSpace().contains(-1) is sp.false


def test_vector_spaces_encode_relations():
    a, b, c = sp.symbols("a b c", real=True)
    assert OrderedVectorSpace(3).constraint([a, b, c]) == sp.And(a < b, b < c)
    assert OrderedVectorSpace(3, strict=False).constraint([a, b, c]) == sp.And(
        a <= b, b <= c
    )
    simplex = SimplexEventSpace(3).constraint([a, b, c])
    assert simplex.has(sp.Eq(a + b + c, 1))
    assert (
        SimplexEventSpace(3).contains([sp.Rational(1, 2), sp.Rational(1, 2), 0])
        is sp.true
    )
    assert (
        SimplexEventSpace(3).contains([sp.Rational(1, 2), sp.Rational(1, 2), 1])
        is sp.false
    )
    assert CountVectorEventSpace(3, 4).contains([1, 2, 1]) is sp.true
    assert CountVectorEventSpace(3, 4).contains([1, 2, 2]) is sp.false
    assert PositiveVectorSpace(2).contains([1, 2]) is sp.true


def test_matrix_spaces_numeric_and_symbolic_constraints():
    pd = PositiveDefiniteMatrixSpace(2)
    psd = PositiveSemidefiniteMatrixSpace(2)
    corr = CorrelationMatrixSpace(2)
    assert pd.contains(sp.eye(2)) is sp.true
    assert pd.contains([[1, 2], [2, 1]]) is sp.false
    assert psd.contains([[1, 1], [1, 1]]) is sp.true
    assert corr.contains(sp.eye(2)) is sp.true
    assert corr.contains([[2, 0], [0, 1]]) is sp.false

    A = sp.MatrixSymbol("A", 2, 2)
    condition = pd.constraint(A)
    assert condition.has(sp.Q.positive_definite(A))
    assert condition.has(sp.Q.symmetric(A))


def test_parameter_space_supports_cross_parameter_relations():
    x, y = sp.symbols("x y", real=True)
    space = ParameterSpace(
        (("x", PositiveRealSpace()), ("y", RealSpace())),
        relations=(lambda values: values["y"] < values["x"],),
    )
    assert space.constraints({"x": 2, "y": 1}) is sp.true
    assert space.constraints((2, 3)) is sp.false
    symbolic = space.constraints((x, y))
    assert symbolic == sp.And(x > 0, y < x)
    with pytest.raises(ValueError):
        space.validate((2, 3))


def test_distribution_parameter_spaces_are_structured():
    assert Normal(0, 2).parameter_space.names == ("mean", "sigma")
    assert Normal(0, 2).parameter_space_constraints is sp.true
    p = sp.symbols("p", real=True)
    assert Bernoulli(p).parameter_space_constraints == sp.And(p >= 0, p <= 1)
    assert Binomial(4, p).parameter_space.names == ("n", "p")
    assert Beta(2, 3).parameter_space.names == ("alpha", "beta")
    assert Gamma(2, 3).parameter_space.names == ("shape", "scale")
    assert StudentT(0, 1, 4).parameter_space.names == ("location", "scale", "df")


def test_structured_distribution_parameter_spaces():
    categorical = Categorical([sp.Rational(1, 4), sp.Rational(3, 4)])
    assert categorical.parameter_space.names == ("probabilities",)
    assert categorical.parameter_space_constraints is sp.true

    multinomial = Multinomial(4, [sp.Rational(1, 4), sp.Rational(3, 4)])
    assert multinomial.parameter_space.names == ("n", "probabilities")
    assert multinomial.parameter_space_constraints is sp.true

    dirichlet = Dirichlet([2, 3, 4])
    assert dirichlet.parameter_space.names == ("concentration",)
    assert dirichlet.parameter_space_constraints is sp.true

    mvn = MultivariateNormal([0, 0], sp.eye(2))
    assert mvn.parameter_space.names == ("mean", "covariance")
    assert mvn.parameter_space_constraints is sp.true


def test_wishart_parameter_space_keeps_cross_parameter_df_constraint():
    df = sp.symbols("nu", positive=True)
    w = Wishart(df, sp.eye(3))
    iw = InverseWishart(df, sp.eye(3))
    assert sp.simplify(w.parameter_space_constraints) == (df > 2)
    assert sp.simplify(iw.parameter_space_constraints) == (df > 2)
    with pytest.raises(ValueError):
        w.parameter_space.validate((2, sp.eye(3)))
