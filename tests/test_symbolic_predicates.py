import sympy as sp

from probstats._symbolic_predicates import TruthValue, certified_equal, certified_zero


def test_certified_predicates_distinguish_unknown_from_inequality():
    x = sp.Symbol("x")
    assert certified_equal((x + 1) ** 2, x**2 + 2 * x + 1) is TruthValue.TRUE
    assert certified_equal(sp.Integer(2), sp.Integer(3)) is TruthValue.FALSE
    assert certified_zero(sp.sin(sp.pi * x)) is TruthValue.UNKNOWN


def test_truth_value_rejects_implicit_boolean_collapse():
    import pytest

    with pytest.raises(TypeError, match="no implicit Boolean"):
        bool(TruthValue.UNKNOWN)
