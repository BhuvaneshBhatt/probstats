"""Invalid public inputs must fail before optional or expensive backends start."""

import pytest
import sympy as sp

import probstats.algebraic._latent_geometry as latent_geometry
import probstats.algebraic._moment_decomposition as moment_decomposition
import probstats.algebraic.independent as independent_module
from probstats.algebraic import (
    ProbabilityTensor,
    algebraic_mle,
    independent_model,
    latent_class_model,
    likelihood,
    recover_multiview_mixture,
)


def _forbidden_backend():
    raise AssertionError("optional backend loaded before public input validation")


def test_invalid_cardinality_fails_before_tensoratlas_load(monkeypatch):
    monkeypatch.setattr(independent_module, "require_tensoratlas", _forbidden_backend)
    with pytest.raises(ValueError, match="cardinality"):
        independent_model((2.5, 2))


def test_invalid_latent_class_count_fails_before_latent_backend(monkeypatch):
    monkeypatch.setattr(latent_geometry, "require_latent_backends", _forbidden_backend)
    with pytest.raises(ValueError, match="latent_classes"):
        latent_class_model((2, 2, 2), sp.Rational(3, 2))


def test_invalid_mixture_component_count_fails_before_cp_backend(monkeypatch):
    monkeypatch.setattr(
        moment_decomposition, "require_moment_tensoratlas", _forbidden_backend
    )
    tensor = ProbabilityTensor([[sp.Rational(1, 2), 0], [0, sp.Rational(1, 2)]])
    with pytest.raises(ValueError, match="components"):
        recover_multiview_mixture(tensor, sp.Rational(3, 2))


def test_face_guard_fails_before_semialg_load(monkeypatch):
    model = likelihood.AlgebraicModel(sp.symbols("p0:4"))
    monkeypatch.setattr(likelihood, "require_likelihood_semialg", _forbidden_backend)
    with pytest.raises(ValueError, match="too many zero-count coordinate faces"):
        algebraic_mle(model, (1, 0, 0, 0), max_faces=7)
