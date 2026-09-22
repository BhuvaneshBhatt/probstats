import pytest
import sympy as sp

from probstats.algebraic import Fiber, ToricModel, toric_model

A_2X2 = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)


def test_toric_model_builds_exact_ideal_and_statistics():
    model = ToricModel(A_2X2)
    p0, p1, p2, p3 = model.probabilities
    assert model.toric_ideal() == (p0 * p3 - p1 * p2,)
    assert model.sufficient_statistic((1, 3, 3, 1)) == (8, 4, 4)
    assert model.markov_basis() == ((1, -1, -1, 1),)
    assert model.dimension() == 2


def test_toric_parameterization_contract():
    model = toric_model(A_2X2)
    mapping = model.log_linear_parameterization()
    assert sp.simplify(sum(mapping.values()) - 1) == 0
    assert all(sp.simplify(eq.subs(mapping)) == 0 for eq in model.equations)


def test_toric_model_requires_homogeneous_design_matrix():
    with pytest.raises(ValueError, match="homogeneous"):
        ToricModel(((1, 0, 2),))


def test_fiber_exact_enumeration_and_markov_basis():
    fiber = Fiber(A_2X2, (8, 4, 4))
    points = fiber.enumerate()
    assert points == (
        (0, 4, 4, 0),
        (1, 3, 3, 1),
        (2, 2, 2, 2),
        (3, 1, 1, 3),
        (4, 0, 0, 4),
    )
    assert fiber.markov_basis() == ((1, -1, -1, 1),)


def test_fiber_refuses_unbounded_coordinate_enumeration():
    fiber = Fiber(((1, 0),), (3,))
    with pytest.raises(ValueError, match="finite coordinate bound"):
        fiber.enumerate()


def test_toric_counts_reject_nonintegral_values():
    model = ToricModel(A_2X2)
    with pytest.raises(ValueError, match="nonnegative integers"):
        model.sufficient_statistic((1, 2.5, 3, 1))
    assert model.fiber((1, 3, 3, 1)).contains((1, 2.5, 3, 1)) is False
