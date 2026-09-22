import math

import numpy as np
import pytest
import sympy as sp

from probstats.algebraic.testing import (
    BasicSemialgebraicNull,
    SemialgebraicHypothesis,
    polynomial_constraint_kernel,
    sdl_test,
)


def setup_quadratic():
    theta = sp.Symbol("theta")
    c = BasicSemialgebraicNull((theta,), inequalities=(theta**2,))
    # An arity-one unbiased estimator of theta. The polynomial construction
    # makes the unsymmetrized theta**2 term x1*x2.
    return theta, c, {theta: lambda x: x}


def test_exact_remains_default_and_matches_manual_symmetrization():
    _, c, estimators = setup_quadratic()
    k = polynomial_constraint_kernel(c, estimators)
    assert k.symmetrization == "exact"
    assert k.permutation_count == 2
    assert k(2.0, 5.0) == 10.0


def test_random_symmetrization_is_reproducible_at_kernel_construction():
    _, c, estimators = setup_quadratic()
    a = polynomial_constraint_kernel(
        c, estimators, symmetrization="random", permutation_count=7, rng=12
    )
    b = polynomial_constraint_kernel(
        c, estimators, symmetrization="random", permutation_count=7, rng=12
    )
    assert a.symmetrization == "random"
    assert a.permutation_count == 7
    for observations in [(2.0, 5.0), (3.0, -4.0)]:
        assert a(*observations) == b(*observations)


def test_random_symmetrization_approximates_exact_for_asymmetric_raw_terms():
    x, y = sp.symbols("x y")
    c = BasicSemialgebraicNull((x, y), inequalities=(x * y,))
    estimators = {x: lambda z: z[0], y: lambda z: z[1]}
    exact = polynomial_constraint_kernel(c, estimators)
    approx = polynomial_constraint_kernel(
        c, estimators, symmetrization="random", permutation_count=20000, rng=5
    )
    observations = ((1.0, 10.0), (4.0, 2.0))
    assert approx(*observations) == pytest.approx(exact(*observations), rel=0.03)


def test_random_mode_requires_positive_sample_count():
    _, c, estimators = setup_quadratic()
    with pytest.raises(ValueError):
        polynomial_constraint_kernel(c, estimators, symmetrization="random")
    with pytest.raises(ValueError):
        polynomial_constraint_kernel(
            c, estimators, symmetrization="random", permutation_count=0
        )


def test_exact_mode_rejects_random_only_argument():
    _, c, estimators = setup_quadratic()
    with pytest.raises(ValueError):
        polynomial_constraint_kernel(c, estimators, permutation_count=3)


def test_unknown_mode_rejected():
    _, c, estimators = setup_quadratic()
    with pytest.raises(ValueError):
        polynomial_constraint_kernel(c, estimators, symmetrization="partial")


def test_sdl_test_random_symmetrization_uses_whole_pipeline_rng():
    theta = sp.Symbol("theta")
    h = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta**2 - 1,), estimators={theta: lambda x: x}
    )
    kwargs = {
        "budget": 6,
        "bootstrap_replicates": 13,
        "seed": 77,
        "kernel_symmetrization": "random",
        "symmetrization_permutations": 5,
    }
    a = sdl_test([-2.0, -1.0, 0.0, 1.0], h, **kwargs)
    b = sdl_test([-2.0, -1.0, 0.0, 1.0], h, **kwargs)
    assert a.symmetrization == "random"
    assert a.symmetrization_permutations == 5
    assert a.p_value == b.p_value
    np.testing.assert_array_equal(a.bootstrap.statistics, b.bootstrap.statistics)
    assert (
        a.studentization.u_statistic.subset_indices
        == b.studentization.u_statistic.subset_indices
    )


def test_exact_kernel_does_not_store_factorial_permutations():
    theta = sp.Symbol("theta")
    c = BasicSemialgebraicNull((theta,), equalities=(theta**10,))
    kernel = polynomial_constraint_kernel(c, {theta: lambda x: x})
    assert kernel.order == 10
    assert kernel.permutation_count == math.factorial(10)
    closure = dict(
        zip(
            kernel.function.__code__.co_freevars,
            (cell.cell_contents for cell in kernel.function.__closure__),
        )
    )
    assert closure["selected_permutations"] is None


def test_exact_kernel_construction_memory_is_not_factorial():
    import tracemalloc

    theta = sp.Symbol("theta")
    component = BasicSemialgebraicNull((theta,), equalities=(theta**10,))
    tracemalloc.start()
    kernel = polynomial_constraint_kernel(component, {theta: lambda x: x})
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert kernel.order == 10
    assert peak < 2_000_000
