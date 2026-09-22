"""Boundary contracts for public algebraic-statistics complexity guards."""

from types import SimpleNamespace

import pytest
import sympy as sp

import probstats.algebraic._latent_geometry as latent_geometry
from probstats.algebraic import (
    Fiber,
    algebraic_mle,
    cumulant_tensor,
    latent_class_model,
    likelihood,
)

A_2X2 = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)


def test_fiber_candidate_guard_accepts_exact_budget_and_rejects_one_below():
    fiber = Fiber(A_2X2, (4, 2, 2))
    candidate_count = sp.prod(bound + 1 for bound in fiber.coordinate_bounds())
    tables = fiber.enumerate(max_candidates=candidate_count)
    assert tables
    with pytest.raises(ValueError, match="increase max_candidates"):
        fiber.enumerate(max_candidates=candidate_count - 1)


def test_cumulant_order_guard_accepts_exact_order_and_rejects_one_below():
    samples = ((-1,), (1,))
    assert cumulant_tensor(samples, 3, max_order=3).order == 3
    with pytest.raises(ValueError, match="exceeds max_order=2"):
        cumulant_tensor(samples, 3, max_order=2)


def test_latent_elimination_guard_accepts_exact_parameter_budget(monkeypatch):
    model = latent_class_model((2, 2), 1)
    calls = []

    def eliminate(graph, variables, eliminate_variables):
        calls.append((graph, variables, eliminate_variables))
        return (sp.Integer(0),)

    monkeypatch.setattr(
        latent_geometry,
        "require_latent_backends",
        lambda: SimpleNamespace(
            elimination_ideal_qq=eliminate,
            secant_expected_dimension=lambda shape, rank: 2,
        ),
    )
    assert model.image_ideal(max_parameters=len(model.parameters)) == (sp.Integer(0),)
    assert len(calls) == 1


def test_latent_elimination_guard_rejects_one_below_before_backend(monkeypatch):
    model = latent_class_model((2, 2), 1)

    def forbidden_backend():
        raise AssertionError("backend must not load when the guard rejects")

    monkeypatch.setattr(latent_geometry, "require_latent_backends", forbidden_backend)
    with pytest.raises(ValueError, match="increase max_parameters"):
        model.image_ideal(max_parameters=len(model.parameters) - 1)


def test_mle_face_guard_rejects_one_below_before_critical_backend(monkeypatch):
    model = likelihood.AlgebraicModel(sp.symbols("p0:4"))

    def forbidden_projection(*args, **kwargs):
        raise AssertionError(
            "critical equations must not be constructed past the guard"
        )

    monkeypatch.setattr(
        likelihood, "_projected_critical_equations", forbidden_projection
    )
    # Three zero counts imply 2**3 = 8 relevant coordinate faces.
    with pytest.raises(ValueError, match="too many zero-count coordinate faces"):
        algebraic_mle(model, (1, 0, 0, 0), max_faces=7)


def test_mle_face_guard_accepts_exact_budget(monkeypatch):
    model = likelihood.AlgebraicModel(sp.symbols("p0:4"))
    calls = []

    def projected(model, counts, *, zero_indices=()):
        calls.append(tuple(zero_indices))
        return (), (), (sp.Integer(1),)

    class EmptyResult:
        assignments = ()

    monkeypatch.setattr(likelihood, "_projected_critical_equations", projected)
    monkeypatch.setattr(
        likelihood,
        "require_likelihood_semialg",
        lambda: SimpleNamespace(
            solve_zero_dimensional_system=lambda *args, **kwargs: EmptyResult()
        ),
    )
    with pytest.raises(NotImplementedError, match="no finite feasible"):
        algebraic_mle(
            model,
            (1, 0, 0, 0),
            include_critical_points=False,
            max_faces=8,
        )
    assert len(calls) == 8
