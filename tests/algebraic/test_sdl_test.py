import numpy as np
import pytest
import sympy as sp

from probstats.algebraic.testing import (
    BasicSemialgebraicNull,
    SemialgebraicHypothesis,
    UnbiasedParameterEstimator,
    sdl_test,
)


def mean(x):
    return x


def basic_hypothesis():
    theta = sp.Symbol("theta")
    return SemialgebraicHypothesis.basic(
        (theta,),
        inequalities=(theta,),
        estimators={theta: UnbiasedParameterEstimator(mean)},
    )


def test_sdl_test_composes_full_pipeline():
    result = sdl_test(
        [-2.0, -1.0, 0.0, 1.0],
        basic_hypothesis(),
        budget=4,
        bootstrap_replicates=31,
        seed=123,
    )
    assert result.sample_size == 4
    assert result.kernel_order == 1
    assert result.budget == 4
    assert result.bootstrap_replicates == 31
    assert result.constraint_count == 1
    assert result.statistic == result.studentization.statistic
    assert result.p_value == result.bootstrap.p_value
    assert result.realized_subsets == len(
        result.studentization.u_statistic.subset_indices
    )
    assert result.coordinate_statistics.shape == (1,)
    assert 0 <= result.p_value <= 1


def test_sdl_test_seed_reproduces_entire_stochastic_pipeline():
    kwargs = {"budget": 4, "bootstrap_replicates": 17, "seed": 91}
    a = sdl_test([-2.0, -1.0, 0.0, 1.0], basic_hypothesis(), **kwargs)
    b = sdl_test([-2.0, -1.0, 0.0, 1.0], basic_hypothesis(), **kwargs)
    assert a.statistic == b.statistic
    assert a.p_value == b.p_value
    assert (
        a.studentization.u_statistic.subset_indices
        == b.studentization.u_statistic.subset_indices
    )
    np.testing.assert_array_equal(a.bootstrap.statistics, b.bootstrap.statistics)


def test_sdl_test_accepts_generator():
    result = sdl_test(
        [-1.0, 0.0, 1.0],
        basic_hypothesis(),
        budget=3,
        bootstrap_replicates=5,
        rng=np.random.default_rng(4),
    )
    assert result.seed is None


def test_seed_and_rng_are_mutually_exclusive():
    with pytest.raises(ValueError):
        sdl_test(
            [-1.0, 0.0, 1.0],
            basic_hypothesis(),
            budget=3,
            seed=1,
            rng=np.random.default_rng(1),
        )


def test_union_null_is_rejected_by_core_sdl_test():
    theta = sp.Symbol("theta")
    a = BasicSemialgebraicNull((theta,), inequalities=(theta,))
    b = BasicSemialgebraicNull((theta,), inequalities=(-theta,))
    hypothesis = SemialgebraicHypothesis((theta,), (a, b), estimators={theta: mean})
    with pytest.raises(ValueError):
        sdl_test([0.0, 1.0], hypothesis, budget=2, seed=1)


def test_result_retains_bootstrap_components_on_request():
    result = sdl_test(
        [-2.0, -1.0, 0.0, 1.0],
        basic_hypothesis(),
        budget=4,
        bootstrap_replicates=7,
        seed=3,
        retain_bootstrap_components=True,
    )
    assert result.bootstrap.h_component.shape == (7, 1)
    assert result.bootstrap.g_component.shape == (7, 1)
    assert result.bootstrap.combined.shape == (7, 1)


def test_random_streams_isolate_symmetrization_from_subset_selection():
    theta = sp.Symbol("theta")
    h = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta**2 - 1,), estimators={theta: mean}
    )
    common = {
        "sample": [-1.0, -0.5, 0.25, 1.0, 1.5],
        "hypothesis": h,
        "budget": 6,
        "bootstrap_replicates": 7,
        "seed": 123,
        "kernel_symmetrization": "random",
    }
    a = sdl_test(**common, symmetrization_permutations=2)
    b = sdl_test(**common, symmetrization_permutations=9)
    assert (
        a.studentization.u_statistic.subset_indices
        == b.studentization.u_statistic.subset_indices
    )
