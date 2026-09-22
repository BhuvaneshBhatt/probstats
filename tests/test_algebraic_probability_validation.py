import inspect

import pytest
import sympy as sp

from probstats.algebraic import ProbabilityTensor, ProbabilityValidationResult
from probstats.algebraic._tolerances import (
    DECOMPOSITION_TOLERANCE,
    FITTING_TOLERANCE,
    STOCHASTIC_TOLERANCE,
)
from probstats.algebraic.latent import fit_latent_class
from probstats.algebraic.moments import (
    decompose_moment_tensor,
    recover_multiview_mixture,
)


def test_probability_tensor_truth_states():
    valid = ProbabilityTensor([[sp.Rational(1, 3), sp.Rational(2, 3)]])
    result = valid.validate()
    assert result == ProbabilityValidationResult(True, True)
    assert result.valid is True
    assert valid.require_probability() is valid

    invalid = ProbabilityTensor([[sp.Integer(2), sp.Integer(-1)]])
    assert invalid.validate().valid is False
    with pytest.raises(ValueError, match="nonnegative and sum to one"):
        invalid.require_probability()

    x = sp.Symbol("x", real=True)
    unknown = ProbabilityTensor([[x, 1 - x]])
    assert unknown.validate() == ProbabilityValidationResult(True, None)
    assert unknown.validate().valid is None
    with pytest.raises(ValueError, match="nonnegativity"):
        unknown.require_probability()


def test_algebraic_numerical_defaults_come_from_one_tolerance_policy():
    assert (
        inspect.signature(decompose_moment_tensor).parameters["tolerance"].default
        == DECOMPOSITION_TOLERANCE
    )
    assert (
        inspect.signature(recover_multiview_mixture).parameters["tolerance"].default
        == STOCHASTIC_TOLERANCE
    )
    assert (
        inspect.signature(fit_latent_class).parameters["tolerance"].default
        == FITTING_TOLERANCE
    )


@pytest.mark.parametrize(
    "bad", [-1e-9, float("inf"), float("nan"), True, sp.Symbol("eps")]
)
def test_algebraic_numerical_tolerances_reject_invalid_values_before_work(bad):
    from probstats.algebraic._tolerances import validate_tolerance

    with pytest.raises((TypeError, ValueError)):
        validate_tolerance(bad)


@pytest.mark.parametrize("method", ["als", "banana", ""])
def test_moment_decomposition_rejects_unknown_method_before_backend(
    method, monkeypatch
):
    import probstats.algebraic._moment_decomposition as module

    monkeypatch.setattr(
        module,
        "require_moment_tensoratlas",
        lambda: (_ for _ in ()).throw(AssertionError("backend should not load")),
    )
    with pytest.raises(ValueError, match="method must be one of"):
        module.decompose_moment_tensor([[1, 0], [0, 1]], method=method)


def test_stochastic_recovery_rejects_unresolved_symbolic_outputs():
    from probstats.algebraic._moment_decomposition import _validate_stochastic_recovery

    x = sp.Symbol("x", real=True)
    with pytest.raises(ValueError, match="unresolved mixture weight"):
        _validate_stochastic_recovery((x,), ((((sp.Integer(1),),),),), tolerance=1e-8)
