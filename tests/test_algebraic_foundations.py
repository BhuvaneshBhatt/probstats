import json
import subprocess
import sys

import sympy as sp

from probstats.algebraic import (
    AlgebraicModel,
    ContingencyTable,
    conditionally_independent_model,
    model_ideal,
)


def test_algebraic_model_separates_variety_and_probability_region():
    p0, p1 = sp.symbols("p0 p1", real=True)
    model = AlgebraicModel((p0, p1), equations=(p0 - p1,))
    ideal = model_ideal(model)
    assert ideal.variables == (p0, p1)
    assert ideal.generators == (p0 - p1, p0 + p1 - 1)
    assert model.variety.has(sp.Eq(p0 - p1, 0))
    assert model.region.has(p0 >= 0)
    assert model.region.has(p1 >= 0)


def test_contingency_table_marginals_and_total():
    table = ContingencyTable([[12, 8, 4], [5, 9, 7]], ("X", "Y"))
    assert table.shape == (2, 3)
    assert table.total == 45
    assert table.marginal("X") == sp.ImmutableDenseNDimArray([24, 21], (2,))
    assert table.marginal("Y") == sp.ImmutableDenseNDimArray([17, 17, 11], (3,))


def test_contingency_table_rejects_negative_counts():
    try:
        ContingencyTable([[1, -1], [0, 2]])
    except ValueError as exc:
        assert "nonnegative" in str(exc)
    else:
        raise AssertionError("negative counts must be rejected")


def test_conditionally_independent_binary_slice_equations():
    model = conditionally_independent_model(
        {"X": 2, "Y": 2, "Z": 2},
        [("X", "Y", ("Z",))],
    )
    assert model.cardinalities == (2, 2, 2)
    assert len(model.equations) == 2
    p000, p010, p100, p110 = sp.symbols("p_0_0_0 p_0_1_0 p_1_0_0 p_1_1_0")
    assert sp.expand(p000 * p110 - p010 * p100) in model.equations


def test_conditionally_independent_requires_partition():
    try:
        conditionally_independent_model(
            {"X": 2, "Y": 2, "Z": 2, "W": 2},
            [("X", "Y", ("Z",))],
        )
    except ValueError as exc:
        assert "partition" in str(exc)
    else:
        raise AssertionError("incomplete conditional-independence statement must fail")


def test_algebraic_namespace_keeps_optional_backends_lazy():
    code = """
import json
import sys
import probstats
print(json.dumps({
    "semialg": "semialg" in sys.modules,
    "tensoratlas": "tensoratlas" in sys.modules,
    "algebraic": hasattr(probstats, "algebraic"),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    state = json.loads(result.stdout)
    assert state == {"semialg": False, "tensoratlas": False, "algebraic": True}
