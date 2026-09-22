"""Metamorphic/property contracts for algebraic-statistics semantics."""

from hypothesis import given, settings
from hypothesis import strategies as st

from probstats.algebraic import (
    central_moment_tensor,
    cumulant_tensor,
    generic_identifiability,
    latent_class_model,
    moment_tensor,
)


@st.composite
def _small_vector_samples(draw):
    count = draw(st.integers(min_value=2, max_value=6))
    rows = draw(
        st.lists(
            st.tuples(
                st.integers(min_value=-3, max_value=3),
                st.integers(min_value=-3, max_value=3),
            ),
            min_size=count,
            max_size=count,
        )
    )
    return tuple(rows)


@settings(max_examples=25, deadline=None)
@given(_small_vector_samples(), st.data())
def test_empirical_moment_tensors_are_invariant_to_sample_order(samples, data):
    permutation = data.draw(st.permutations(tuple(range(len(samples)))))
    reordered = tuple(samples[i] for i in permutation)

    assert moment_tensor(samples, 2).values == moment_tensor(reordered, 2).values
    assert (
        central_moment_tensor(samples, 2).values
        == central_moment_tensor(reordered, 2).values
    )
    assert cumulant_tensor(samples, 3).values == cumulant_tensor(reordered, 3).values


@settings(max_examples=12, deadline=None)
@given(st.permutations((2, 2, 3)))
def test_latent_identifiability_axis_invariance(
    shape,
):
    reference = generic_identifiability(latent_class_model((2, 2, 3), 2))
    permuted = generic_identifiability(latent_class_model(shape, 2))

    assert permuted.parameter_dimension == reference.parameter_dimension
    assert permuted.image_dimension == reference.image_dimension
    assert permuted.fiber_dimension == reference.fiber_dimension
    assert permuted.identifiable == reference.identifiable
    assert permuted.certified == reference.certified
    assert permuted.up_to_label_swapping == reference.up_to_label_swapping
    assert permuted.expected_label_orbit == reference.expected_label_orbit
