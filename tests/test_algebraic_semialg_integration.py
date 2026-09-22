from probstats.algebraic import independent_model, model_invariants


def test_independent_model_exact_invariants():
    model = independent_model((2, 2), variables=("X", "Y"))
    invariants = model_invariants(model)
    assert invariants.dimension == 2
    assert invariants.degree == 2
