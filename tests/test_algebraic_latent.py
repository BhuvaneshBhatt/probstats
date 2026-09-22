import sympy as sp

from probstats.algebraic import (
    fit_latent_class,
    generic_identifiability,
    identifiability,
    latent_class_model,
)


def test_latent_class_uses_reduced_stochastic_parameterization():
    model = latent_class_model((2, 2, 2), 2, variables=("X", "Y", "Z"))
    assert model.parameter_dimension == 7
    assert len(model.parameters) == 7
    assert (
        sp.simplify(sum(model.parameterization[p] for p in model.probabilities) - 1)
        == 0
    )
    assert model.image_dimension() == 7
    assert model.dimension() == 7
    assert model.image_ideal() == (sp.expand(sum(model.probabilities) - 1),)


def test_binary_two_class_generic_identifiability():
    result = generic_identifiability(latent_class_model((2, 2, 2), 2))
    assert result.identifiable is True
    assert result.up_to_label_swapping is True
    assert result.expected_label_orbit == 2
    assert result.fiber_dimension == 0
    assert result.certified is True


def test_two_observed_variables_have_positive_dimensional_generic_fiber():
    result = generic_identifiability(latent_class_model((2, 2), 2))
    assert result.identifiable is False
    assert result.fiber_dimension == 2
    assert result.certified is True


def test_degenerate_latent_not_locally_identifiable():
    model = latent_class_model((2, 2, 2), 2)
    values = {parameter: sp.Rational(1, 3) for parameter in model.parameters}
    result = identifiability(model, values)
    assert result.identifiable is False
    assert result.fiber_dimension > 0


def test_rank_one_latent_image_ideal_recovers_independence_determinant():
    model = latent_class_model((2, 2), 1)
    ideal = model.image_ideal(max_parameters=4)
    p00, p01, p10, p11 = model.probabilities
    expected = p00 * p11 - p01 * p10
    # The affine stochastic image includes normalization in the graph, so the
    # determinant must reduce to zero modulo the returned elimination basis.
    assert (
        sp.groebner(ideal, *model.probabilities, order="grevlex").reduce(expected)[1]
        == 0
    )


def test_em_fit_returns_normalized_latent_parameters():
    result = fit_latent_class([[30, 10], [10, 30]], 2, rng=7, max_iter=1000)
    assert abs(sum(result.weights) - 1.0) < 1e-12
    assert all(
        abs(sum(vector) - 1.0) < 1e-10
        for component in result.conditional_probabilities
        for vector in component
    )
    assert result.iterations <= 1000
