"""Registry-driven contracts for integer-valued algebraic-statistics inputs."""

from collections.abc import Callable
from dataclasses import dataclass

import pytest
import sympy as sp

from probstats.algebraic import (
    Fiber,
    ToricModel,
    algebraic_mle,
    categorical_multi_view_moment,
    cumulant_tensor,
    fit_latent_class,
    independent_model,
    latent_class_model,
    maximum_likelihood_degree,
    moment_tensor,
    recover_multiview_mixture,
    sample_fiber,
)

A_2X2 = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)


@dataclass(frozen=True)
class IntegerContract:
    """One public integer-valued input contract."""

    name: str
    invoke: Callable[[object], object]
    message: str


_MODEL = None


def _independent_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = independent_model((2, 2))
    return _MODEL


def _toric_model():
    return ToricModel(A_2X2)


def _fiber():
    return Fiber(A_2X2, (4, 2, 2))


CONTRACTS = (
    IntegerContract(
        "cardinality",
        lambda value: independent_model((value, 2)),
        "cardinality",
    ),
    IntegerContract(
        "latent_classes",
        lambda value: latent_class_model((2, 2, 2), value),
        "latent_classes",
    ),
    IntegerContract(
        "moment_order",
        lambda value: moment_tensor(((1, 2), (3, 4)), value),
        "order",
    ),
    IntegerContract(
        "cumulant_max_order",
        lambda value: cumulant_tensor(((1,), (2,)), 1, max_order=value),
        "max_order",
    ),
    IntegerContract(
        "categorical_cardinality",
        lambda value: categorical_multi_view_moment(
            ((0, 1), (0, 1)), cardinalities=(value, 2)
        ),
        "cardinality",
    ),
    IntegerContract(
        "mixture_components",
        lambda value: recover_multiview_mixture(
            [[sp.Rational(1, 2), 0], [0, sp.Rational(1, 2)]],
            value,
            method="exact",
        ),
        "components",
    ),
    IntegerContract(
        "mixture_restarts",
        lambda value: recover_multiview_mixture(
            [[sp.Rational(1, 2), 0], [0, sp.Rational(1, 2)]],
            1,
            method="numerical",
            restarts=value,
            rng=1,
        ),
        "restarts",
    ),
    IntegerContract(
        "fiber_max_candidates",
        lambda value: _fiber().enumerate(max_candidates=value),
        "max_candidates",
    ),
    IntegerContract(
        "sample_size",
        lambda value: sample_fiber(
            (1, 1, 1, 1), fiber=_fiber(), size=value, burnin=0, thin=1, rng=1
        ),
        "size",
    ),
    IntegerContract(
        "sample_burnin",
        lambda value: sample_fiber(
            (1, 1, 1, 1), fiber=_fiber(), size=1, burnin=value, thin=1, rng=1
        ),
        "burnin",
    ),
    IntegerContract(
        "sample_thin",
        lambda value: sample_fiber(
            (1, 1, 1, 1), fiber=_fiber(), size=1, burnin=0, thin=value, rng=1
        ),
        "thin",
    ),
    IntegerContract(
        "mle_max_faces",
        lambda value: algebraic_mle(
            _independent_model(), (1, 1, 1, 1), max_faces=value
        ),
        "max_faces",
    ),
    IntegerContract(
        "ml_degree_samples",
        lambda value: maximum_likelihood_degree(_independent_model(), samples=value),
        "samples",
    ),
    IntegerContract(
        "em_latent_classes",
        lambda value: fit_latent_class([[1, 0], [0, 1]], value, max_iter=1, rng=1),
        "latent_classes",
    ),
    IntegerContract(
        "em_max_iter",
        lambda value: fit_latent_class([[1, 0], [0, 1]], 1, max_iter=value, rng=1),
        "max_iter",
    ),
)


@pytest.mark.parametrize("contract", CONTRACTS, ids=lambda case: case.name)
@pytest.mark.parametrize(
    "invalid",
    (2.0, 2.5, sp.Rational(5, 2), -1, True, sp.Symbol("n")),
)
def test_public_integer_contracts_reject_lossy_or_undecidable_values(contract, invalid):
    with pytest.raises(ValueError, match=contract.message):
        contract.invoke(invalid)


@pytest.mark.parametrize(
    "value",
    (2, sp.Integer(2), 2.0),
)
def test_exact_integer_contract(value):
    # A floating point representation of an exact integer is not
    # accepted: public exact/algebraic inputs do not infer exactness from 2.0.
    if isinstance(value, float):
        with pytest.raises(ValueError, match="latent_classes"):
            latent_class_model((2, 2, 2), value)
    else:
        assert latent_class_model((2, 2, 2), value).latent_classes == 2


@pytest.mark.parametrize(
    ("invoke", "message"),
    (
        (lambda: latent_class_model((0, 2), 1), "cardinality"),
        (lambda: latent_class_model((2, 2), 0), "latent_classes"),
        (lambda: moment_tensor(((1,), (2,)), 0), "order"),
        (lambda: _fiber().enumerate(max_candidates=0), "max_candidates"),
        (
            lambda: sample_fiber(
                (1, 1, 1, 1), fiber=_fiber(), size=0, burnin=0, thin=1
            ),
            "size",
        ),
    ),
)
def test_public_integer_contracts_enforce_declared_lower_bounds(invoke, message):
    with pytest.raises(ValueError, match=message):
        invoke()
