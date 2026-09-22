import sympy as sp

from probstats.algebraic import (
    ProbabilityTensor,
    independent_model,
    tensor_independent,
)


def test_independent_model_is_probability_segre_slice():
    model = independent_model((2, 3), variables=("X", "Y"))
    assert model.cardinalities == (2, 3)
    assert model.variable_names == ("X", "Y")
    assert len(model.probabilities) == 6
    assert len(model.equations) == 3
    assert model.metadata["model"] == "complete_independent"
    assert all(
        sp.Poly(eq, *model.probabilities).total_degree() == 2 for eq in model.equations
    )


def test_probability_tensor_flatten_and_independence():
    tensor = ProbabilityTensor(
        [
            [sp.Rational(1, 6), sp.Rational(1, 3)],
            [sp.Rational(1, 6), sp.Rational(1, 3)],
        ],
        ("X", "Y"),
    )
    assert tensor.is_normalized() is True
    assert tensor.is_nonnegative() is True
    assert tensor.flatten(("X",), ("Y",)) == sp.Matrix(
        [[sp.Rational(1, 6), sp.Rational(1, 3)], [sp.Rational(1, 6), sp.Rational(1, 3)]]
    )
    assert tensor_independent(tensor).independent is True


def test_tensor_independent_rejects_dependent_table():
    tensor = ProbabilityTensor(
        [[sp.Rational(1, 2), 0], [0, sp.Rational(1, 2)]],
        ("X", "Y"),
    )
    result = tensor_independent(tensor)
    assert result.independent is False
    assert any(value != 0 for value in result.residuals)
