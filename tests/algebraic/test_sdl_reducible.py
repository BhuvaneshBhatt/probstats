import numpy as np
import pytest
import sympy as sp

from probstats.algebraic.testing import (
    BasicSemialgebraicNull,
    SemialgebraicHypothesis,
    sdl_test,
    sdl_union_test,
)


def union_hypothesis():
    theta = sp.Symbol("theta")
    left = BasicSemialgebraicNull(
        (theta,), inequalities=(theta + 1,), label="theta <= -1"
    )
    right = BasicSemialgebraicNull(
        (theta,), inequalities=(1 - theta,), label="theta >= 1"
    )
    return SemialgebraicHypothesis(
        (theta,), (left, right), estimators={theta: lambda x: x}
    )


def test_union_test_runs_every_component_and_uses_maximum_p_value():
    result = sdl_union_test(
        [-0.5, 0.0, 0.5, 0.25],
        union_hypothesis(),
        budget=4,
        bootstrap_replicates=31,
        seed=10,
    )
    assert result.component_count == 2
    assert result.p_value == max(r.p_value for r in result.component_results)
    np.testing.assert_array_equal(
        result.component_p_values,
        [r.p_value for r in result.component_results],
    )
    assert result.component_results[0].hypothesis.basic_null.label == "theta <= -1"
    assert result.component_results[1].hypothesis.basic_null.label == "theta >= 1"


def test_seeded_union_test_is_reproducible_and_records_child_seeds():
    kwargs = {"budget": 4, "bootstrap_replicates": 17, "seed": 1234}
    a = sdl_union_test([-0.5, 0.0, 0.5, 0.25], union_hypothesis(), **kwargs)
    b = sdl_union_test([-0.5, 0.0, 0.5, 0.25], union_hypothesis(), **kwargs)
    assert a.component_seeds == b.component_seeds
    assert len(set(a.component_seeds)) == 2
    np.testing.assert_array_equal(a.component_p_values, b.component_p_values)
    for ar, br in zip(a.component_results, b.component_results, strict=True):
        np.testing.assert_array_equal(ar.bootstrap.statistics, br.bootstrap.statistics)


def test_recorded_child_seed_reproduces_component_independently():
    sample = [-0.5, 0.0, 0.5, 0.25]
    h = union_hypothesis()
    result = sdl_union_test(sample, h, budget=4, bootstrap_replicates=13, seed=77)
    first = SemialgebraicHypothesis(
        h.parameters, (h.components[0],), estimators=h.estimators
    )
    replay = sdl_test(
        sample,
        first,
        budget=4,
        bootstrap_replicates=13,
        seed=result.component_seeds[0],
    )
    assert replay.p_value == result.component_results[0].p_value
    np.testing.assert_array_equal(
        replay.bootstrap.statistics, result.component_results[0].bootstrap.statistics
    )


def test_generator_mode_derives_replayable_component_streams():
    result = sdl_union_test(
        [-0.5, 0.0, 0.5, 0.25],
        union_hypothesis(),
        budget=4,
        bootstrap_replicates=7,
        rng=np.random.default_rng(3),
    )
    assert result.component_seeds is not None
    assert len(result.component_seeds) == result.component_count


def test_basic_hypothesis_is_valid_one_component_union_test():
    theta = sp.Symbol("theta")
    h = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta,), estimators={theta: lambda x: x}
    )
    result = sdl_union_test(
        [-1.0, 0.0, 1.0], h, budget=3, bootstrap_replicates=7, seed=4
    )
    assert result.component_count == 1
    assert result.p_value == result.component_results[0].p_value


def test_seed_and_rng_mutually_exclusive():
    with pytest.raises(ValueError):
        sdl_union_test(
            [-1.0, 0.0],
            union_hypothesis(),
            budget=2,
            seed=1,
            rng=np.random.default_rng(1),
        )
