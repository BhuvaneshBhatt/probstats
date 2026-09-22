"""Equivalent statistical representations must produce identical algebraic results."""

import sympy as sp
from hypothesis import given, settings
from hypothesis import strategies as st

from probstats.algebraic import (
    ContingencyTable,
    ProbabilityTensor,
    ToricModel,
    conditional_test,
    likelihood_equations,
    tensor_independent,
)

A_2X2 = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)


@st.composite
def _nonzero_2x2_counts(draw):
    flat = draw(
        st.tuples(*(st.integers(min_value=0, max_value=4) for _ in range(4))).filter(
            any
        )
    )
    return flat


@settings(max_examples=20, deadline=None)
@given(_nonzero_2x2_counts())
def test_table_vector_representation_equivalence(flat):
    table = ContingencyTable((flat[:2], flat[2:]), variable_names=("X", "Y"))
    model = ToricModel(A_2X2)

    assert table.flat_counts == flat
    assert model.sufficient_statistic(table) == model.sufficient_statistic(flat)
    assert model.fiber(table) == model.fiber(flat)

    table_system = likelihood_equations(model, table)
    vector_system = likelihood_equations(model, flat)
    assert table_system.counts == vector_system.counts
    assert table_system.likelihood == vector_system.likelihood
    assert table_system.equations == vector_system.equations


@settings(max_examples=12, deadline=None)
@given(_nonzero_2x2_counts())
def test_table_and_flat_vector_have_identical_exact_conditional_test(flat):
    table = ContingencyTable((flat[:2], flat[2:]))
    model = ToricModel(A_2X2)

    table_result = conditional_test(table, model)
    vector_result = conditional_test(flat, model)
    assert table_result == vector_result


@settings(max_examples=20, deadline=None)
@given(
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
    st.integers(min_value=1, max_value=5),
)
def test_probability_tensor_representation_equivalence(a, b, c, d):
    total = a + b + c + d
    values = [
        [sp.Rational(a, total), sp.Rational(b, total)],
        [sp.Rational(c, total), sp.Rational(d, total)],
    ]
    tensor = ProbabilityTensor(values, variable_names=("X", "Y"))
    copied = ProbabilityTensor(tensor.values.tolist(), variable_names=("X", "Y"))

    assert tensor.cardinalities == copied.cardinalities
    assert tensor.is_normalized() is True
    assert copied.is_normalized() is True
    assert tensor_independent(tensor) == tensor_independent(copied)
