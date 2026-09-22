import numpy as np
import pytest
import sympy as sp

from probstats.algebraic.testing import (
    BasicSemialgebraicNull,
    SemialgebraicHypothesis,
    augment_constraints,
    augment_hypothesis_constraints,
    sdl_test,
)


def component():
    x, y = sp.symbols("x y")
    return BasicSemialgebraicNull(
        (x, y), equalities=(x + y - 1,), inequalities=(-x, -y)
    )


def test_convex_augmentation_retains_originals_and_adds_r_constraints():
    c = component()
    result = augment_constraints(c, count=5, seed=7)
    assert result.original_constraints == c.sdl_constraints
    assert len(result.added_constraints) == 5
    assert len(result.component.sdl_constraints) == len(c.sdl_constraints) + 5
    np.testing.assert_allclose(result.weights.sum(axis=1), 1)
    assert np.all(result.weights >= 0)


def test_generated_constraints_are_the_recorded_convex_combinations():
    c = component()
    result = augment_constraints(c, count=3, seed=12)
    for row, added in zip(result.weights, result.added_constraints, strict=True):
        expected = sp.expand(
            sum(
                sp.Float(float(w)) * f
                for w, f in zip(row, c.sdl_constraints, strict=True)
            )
        )
        assert sp.expand(added - expected) == 0


def test_augmentation_is_reproducible():
    a = augment_constraints(component(), count=4, seed=19)
    b = augment_constraints(component(), count=4, seed=19)
    np.testing.assert_array_equal(a.weights, b.weights)
    assert a.added_constraints == b.added_constraints


def test_zero_count_is_identity_at_constraint_level():
    c = component()
    result = augment_constraints(c, count=0, seed=1)
    assert result.added_constraints == ()
    assert result.component.sdl_constraints == c.sdl_constraints
    assert result.weights.shape == (0, len(c.sdl_constraints))


def test_hypothesis_wrapper_preserves_context():
    c = component()
    h = SemialgebraicHypothesis(
        c.parameters,
        (c,),
        estimators={c.parameters[0]: lambda z: z[0], c.parameters[1]: lambda z: z[1]},
        sample_space="simplex",
        metadata={"source": "test"},
    )
    augmented, info = augment_hypothesis_constraints(h, count=2, seed=2)
    assert augmented.estimators == h.estimators
    assert augmented.sample_space == "simplex"
    assert augmented.metadata["source"] == "test"
    assert augmented.basic_null == info.component


@pytest.mark.parametrize("concentration", [0, -1, np.inf, np.nan])
def test_invalid_concentration_rejected(concentration):
    with pytest.raises(ValueError):
        augment_constraints(component(), count=1, concentration=concentration)


def test_sdl_test_can_augment_constraints_end_to_end():
    theta = sp.Symbol("theta")
    h = SemialgebraicHypothesis.basic(
        (theta,), inequalities=(theta,), estimators={theta: lambda x: x}
    )
    result = sdl_test(
        [-2.0, -1.0, 0.0, 1.0],
        h,
        budget=4,
        bootstrap_replicates=11,
        seed=42,
        augment_constraints_count=3,
    )
    assert result.augmentation is not None
    assert result.constraint_count == 1
    assert result.coordinate_statistics.shape == (1,)
    assert result.hypothesis is not h


def test_original_constraints_imply_added_constraints_numerically():
    c = component()
    result = augment_constraints(c, count=10, seed=5)
    x, y = c.parameters
    for point in [(0.2, 0.8), (0.0, 1.0), (0.75, 0.25)]:
        subs = dict(zip((x, y), point, strict=True))
        assert all(float(f.subs(subs)) <= 1e-12 for f in result.original_constraints)
        assert all(float(f.subs(subs)) <= 1e-12 for f in result.added_constraints)


def test_augmentation_drops_exact_duplicate_coordinates():
    theta = sp.Symbol("theta")
    component = BasicSemialgebraicNull((theta,), inequalities=(theta,))
    result = augment_constraints(component, count=5, seed=2)
    assert result.added_constraints == ()
    assert result.weights.shape == (0, 1)
