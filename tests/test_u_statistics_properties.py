from itertools import combinations
from math import comb

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from probstats.u_statistics import (
    EmptySubsetSelectionError,
    incomplete_u_statistic,
    u_statistic,
)


@settings(max_examples=35, deadline=None)
@given(st.lists(st.integers(-8, 8), min_size=2, max_size=7), st.integers(1, 3))
def test_complete_matches_direct_enumeration(sample, order):
    if order > len(sample):
        return

    def kernel(*xs):
        return sum((i + 1) * x for i, x in enumerate(xs))

    expected = np.mean(
        [
            kernel(*(sample[i] for i in idx))
            for idx in combinations(range(len(sample)), order)
        ]
    )
    assert u_statistic(sample, kernel, order=order) == pytest.approx(expected)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.integers(-5, 5), min_size=2, max_size=7), st.integers(1, 3))
def test_vector_complete_matches_direct_enumeration(sample, order):
    if order > len(sample):
        return

    def kernel(*xs):
        return np.array([sum(xs), sum(x * x for x in xs)], dtype=float)

    expected = np.mean(
        [
            kernel(*(sample[i] for i in idx))
            for idx in combinations(range(len(sample)), order)
        ],
        axis=0,
    )
    np.testing.assert_allclose(u_statistic(sample, kernel, order=order), expected)


def test_fixed_budget_full_enumeration_matches_complete():
    sample = tuple(range(6))
    order = 3
    budget = comb(len(sample), order)

    def kernel(*xs):
        return sum(x * x for x in xs)

    result = incomplete_u_statistic(
        sample,
        kernel,
        order=order,
        budget=budget,
        selection="fixed",
        rng=9,
        return_result=True,
    )
    assert result.value == pytest.approx(u_statistic(sample, kernel, order=order))
    assert set(result.subset_indices) == set(combinations(range(len(sample)), order))


def test_bernoulli_subset_count_matches_binomial_moments():
    n, order, budget = 8, 2, 8
    total = comb(n, order)
    p = budget / total
    counts = []
    for seed in range(1200):
        try:
            result = incomplete_u_statistic(
                range(n),
                lambda x, y: x + y,
                order=order,
                budget=budget,
                rng=seed,
                return_result=True,
            )
            counts.append(result.evaluated_subsets)
        except EmptySubsetSelectionError:
            counts.append(0)
    counts = np.asarray(counts)
    assert np.mean(counts) == pytest.approx(total * p, abs=0.22)
    assert np.var(counts) == pytest.approx(total * p * (1 - p), abs=0.65)
