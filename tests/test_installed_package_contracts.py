"""Permanent release contracts intended to run against installed wheels.

Set ``PROBSTATS_INSTALLED_CONTRACTS=1`` in a clean wheel environment. The
suite is skipped during ordinary source-tree development so it cannot
accidentally validate repository-relative imports.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("PROBSTATS_INSTALLED_CONTRACTS") != "1",
    reason="installed-wheel release contracts are opt-in",
)


def test_installed_stack_public_imports_and_dependency_ownership():
    import probstats
    from probstats import algebraic

    assert algebraic.__name__ == "probstats.algebraic"
    assert "algebraic" in probstats.__all__


def test_installed_binary_independence_theorem_chain():
    import sympy as sp

    from probstats.algebraic import independent_model, likelihood_equations

    model = independent_model((2, 2))
    system = likelihood_equations(model, (2, 3, 5, 7))
    assert model.dimension() == 2
    assert model.degree() == 2
    assert system.counts == (2, 3, 5, 7)
    assert all(isinstance(eq, sp.Expr) for eq in system.equations)


def test_installed_table_vector_coordinate_contract():
    from probstats.algebraic import ContingencyTable, ToricModel

    design = (
        (1, 1, 1, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
    )
    model = ToricModel(design)
    table = ContingencyTable([[1, 3], [3, 1]], ("X", "Y"))
    assert table.flat_counts == (1, 3, 3, 1)
    assert model.sufficient_statistic(table) == model.sufficient_statistic(
        table.flat_counts
    )


def test_installed_certificate_semantics_remain_three_valued():
    from tensoratlas.algebraic import generic_cp_identifiability

    inconclusive = generic_cp_identifiability((2, 2), 2)
    assert inconclusive.identifiable is None
    assert inconclusive.certified is False


def test_installed_moment_and_latent_smoke():
    from probstats.algebraic import (
        generic_identifiability,
        latent_class_model,
        moment_tensor,
    )

    moment = moment_tensor(((1, 2), (3, 4)), 2)
    assert moment.values.tolist() == [[5, 7], [7, 10]]
    result = generic_identifiability(latent_class_model((2, 2, 2), 2))
    assert result.identifiable is True
    assert result.up_to_label_swapping is True


def test_installed_random_variable_contracts():
    import sympy as sp

    from probstats import (
        Normal,
        RandomVariable,
        StatisticalAssumptions,
        conditional_expectation,
        distribution,
        expectation,
        independent,
        moment_generating_function,
    )

    x = RandomVariable("X", distribution=Normal(1, 2))
    y = RandomVariable("Y", distribution=Normal(3, 4))
    context = StatisticalAssumptions(independent(x, y))
    assert distribution(x + y, assumptions=context) == Normal(4, 2 * sp.sqrt(5))
    assert conditional_expectation(x, given=y, assumptions=context) == expectation(x)
    t = sp.Symbol("t", real=True)
    assert sp.simplify(moment_generating_function(x, t) - sp.exp(t + 2 * t**2)) == 0
