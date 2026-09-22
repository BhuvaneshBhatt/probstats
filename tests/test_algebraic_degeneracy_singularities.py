"""Degenerate and singular algebraic-statistics configurations."""

import sympy as sp

from probstats.algebraic import (
    algebraic_mle,
    central_moment_tensor,
    generic_identifiability,
    identifiability,
    independent_model,
    latent_class_model,
    moment_tensor,
    recover_multiview_mixture,
)


def test_boundary_mle_with_zero_count_is_exact_and_complete():
    model = independent_model((1, 3), variables=("X", "Y"))
    result = algebraic_mle(model, (2, 3, 0))

    assert result.exact is True
    assert result.complete is True
    assert result.mle is not None
    values = tuple(result.mle[p] for p in model.probabilities)
    assert values == (sp.Rational(2, 5), sp.Rational(3, 5), sp.Integer(0))


def test_coincident_latent_components_are_pointwise_singular():
    model = latent_class_model((2, 2, 2), 2)
    values = {}
    for parameter in model.parameters:
        if parameter.name.startswith("lambda_"):
            values[parameter] = sp.Rational(1, 2)
        else:
            values[parameter] = sp.Rational(1, 2)

    result = identifiability(model, values)
    assert result.generic is False
    assert result.identifiable is False
    assert result.certified is True
    assert result.locally_finite_to_one is False
    assert result.fiber_dimension > 0


def test_zero_latent_weight_is_not_promoted_to_generic_identifiability():
    model = latent_class_model((2, 2, 2), 2)
    values = {}
    for i, parameter in enumerate(model.parameters):
        if parameter.name.startswith("lambda_"):
            values[parameter] = sp.Integer(0)
        else:
            values[parameter] = sp.Rational((i % 3) + 1, 5)

    result = identifiability(model, values)
    assert result.generic is False
    assert result.identifiable is False
    assert result.certified is True
    assert result.fiber_dimension > 0


def test_two_variable_latent_fiber_dimension():
    result = generic_identifiability(latent_class_model((2, 2), 2))
    assert result.identifiable is False
    assert result.certified is True
    assert result.fiber_dimension > 0


def test_constant_samples_have_zero_second_central_moment():
    samples = ((3, -2), (3, -2), (3, -2), (3, -2))
    central = central_moment_tensor(samples, 2)
    assert all(value == 0 for value in central.values._array)

    raw = moment_tensor(samples, 2)
    assert raw.values[0, 0] == 9
    assert raw.values[0, 1] == -6
    assert raw.values[1, 1] == 4


def test_probability_mixture_recovery_rejects_negative_entries():
    tensor = [[[sp.Rational(3, 5), 0], [0, 0]], [[0, 0], [0, sp.Rational(2, 5)]]]
    tensor[0][0][1] = sp.Rational(-1, 10)
    tensor[0][1][0] = sp.Rational(1, 10)

    try:
        recover_multiview_mixture(tensor, 2, method="numerical", rng=1)
    except ValueError as exc:
        assert "nonnegative" in str(exc).lower()
    else:
        raise AssertionError("negative probability tensor must be rejected")
