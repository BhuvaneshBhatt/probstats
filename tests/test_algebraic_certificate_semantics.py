import sympy as sp
from tensoratlas.algebraic import generic_cp_identifiability, tensor_rank

from probstats.algebraic import (
    generic_identifiability,
    identifiability,
    independent_model,
    latent_class_model,
    maximum_likelihood_degree,
)


def test_ml_degree_exact_witness_count_is_not_genericity_certificate():
    result = maximum_likelihood_degree(independent_model((2, 2)))
    assert result.degree == 1
    assert result.exact is True
    assert result.complete is True
    assert result.generic is True
    assert result.generic_certified is False


def test_positive_dimensional_latent_fiber_is_certified_nonidentifiability():
    result = generic_identifiability(latent_class_model((2, 2), 2))
    assert result.identifiable is False
    assert result.certified is True
    assert result.fiber_dimension > 0
    assert result.locally_finite_to_one is False


def test_failed_kruskal_sufficient_condition_is_inconclusive_not_negative():
    result = generic_cp_identifiability((2, 2, 2), 3)
    assert result.identifiable is None
    assert result.certified is False
    assert result.lhs < result.threshold


def test_singular_point_not_promoted_to_generic():
    model = latent_class_model((2, 2, 2), 2)
    parameters = {parameter: sp.Rational(1, 3) for parameter in model.parameters}
    result = identifiability(model, parameters)
    assert result.generic is False
    assert result.identifiable is False
    assert result.certified is True
    assert result.fiber_dimension > 0


def test_tensor_rank_retains_unknown_exact_rank_when_bounds_do_not_meet():
    tensor = sp.MutableDenseNDimArray.zeros(2, 2, 2)
    tensor[0, 0, 0] = 1
    tensor[0, 1, 1] = 1
    tensor[1, 0, 1] = 1
    result = tensor_rank(tensor.tolist(), method="exact")
    assert result.lower_bound == 2
    assert result.upper_bound >= result.lower_bound
    if result.lower_bound != result.upper_bound:
        assert result.rank is None
        assert result.exact is False
        assert result.decomposition is None
