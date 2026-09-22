import sympy as sp
from semialg.algebraic import toric_ideal
from tensoratlas.algebraic import (
    generic_cp_identifiability,
    secant_expected_dimension,
    tensor_rank,
)

from probstats.algebraic import (
    ToricModel,
    algebraic_mle,
    categorical_multi_view_moment,
    generic_identifiability,
    independent_model,
    latent_class_model,
    maximum_likelihood_degree,
    recover_multiview_mixture,
)


def _same_ideal(left, right, variables):
    left_basis = sp.groebner(left, *variables, order="grevlex")
    right_basis = sp.groebner(right, *variables, order="grevlex")
    return all(left_basis.reduce(poly)[1] == 0 for poly in right) and all(
        right_basis.reduce(poly)[1] == 0 for poly in left
    )


def test_binary_independence_segre_and_toric_descriptions_agree():
    model = independent_model((2, 2), variables=("X", "Y"))
    design = (
        (1, 1, 1, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
    )
    toric = toric_ideal(design, model.probabilities)
    assert _same_ideal(model.equations, toric, model.probabilities)


def test_binary_independence_toric_fiber_and_likelihood_geometry_cohere():
    design = (
        (1, 1, 1, 1),
        (1, 1, 0, 0),
        (1, 0, 1, 0),
    )
    model = ToricModel(design)
    observed = (1, 3, 3, 1)
    statistic = model.sufficient_statistic(observed)
    move = model.markov_basis()[0]
    moved = tuple(x + u for x, u in zip(observed, move, strict=True))
    assert model.sufficient_statistic(moved) == statistic

    mle = algebraic_mle(model, (2, 3, 5, 7))
    assert mle.exact is True
    assert mle.complete is True
    assert mle.mle is not None
    assert sp.simplify(sum(mle.mle.values()) - 1) == 0

    ml_degree = maximum_likelihood_degree(model)
    assert ml_degree.degree == 1
    assert ml_degree.exact is True
    assert ml_degree.complete is True


def test_latent_class_identifiability_matches_secant_and_kruskal_geometry():
    model = latent_class_model((2, 2, 2), 2, variables=("X", "Y", "Z"))
    assert secant_expected_dimension((2, 2, 2), 2) == 7
    cp = generic_cp_identifiability((2, 2, 2), 2)
    result = generic_identifiability(model)

    assert model.parameter_dimension == 7
    assert model.image_dimension() == 7
    assert cp.identifiable is True and cp.certified is True
    assert result.identifiable is True
    assert result.certified is True
    assert result.up_to_label_swapping is True
    assert result.expected_label_orbit == 2


def test_multiview_recovery_tensor_rank():
    observations = (
        (0, 0, 0, 0),
        (0, 0, 1, 1),
        (0, 1, 0, 1),
    )
    moment = categorical_multi_view_moment(observations, cardinalities=(1, 2, 2))
    rank = tensor_rank(moment.tensor)
    recovered = recover_multiview_mixture(moment.values, 1, method="exact")

    assert rank.rank == 1
    assert rank.exact is True
    assert recovered.exact is True
    assert recovered.converged is True
    assert recovered.weights == (1,)
    assert recovered.residual == 0
