import sympy as sp

from probstats.algebraic import ToricModel, conditional_test, sample_fiber

A_2X2 = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)


def test_probability_ordered_exact_conditional_test():
    model = ToricModel(A_2X2)
    result = conditional_test((1, 3, 3, 1), model)
    assert result.exact is True
    assert result.fiber_size == 5
    assert result.pvalue == sp.Rational(17, 35)
    assert result.observed_probability == sp.Rational(8, 35)


def test_callable_exact_conditional_statistic():
    model = ToricModel(A_2X2)
    result = conditional_test(
        (1, 3, 3, 1),
        model,
        statistic=lambda table: abs(table[0] - table[1]),
        alternative="greater",
    )
    assert result.pvalue == sp.Rational(17, 35)
    assert result.observed_score == 2


def test_markov_basis_sampler_stays_in_fiber_and_is_reproducible():
    model = ToricModel(A_2X2)
    first = sample_fiber((1, 3, 3, 1), model=model, size=25, burnin=10, rng=7)
    second = sample_fiber((1, 3, 3, 1), model=model, size=25, burnin=10, rng=7)
    assert first == second
    fiber = model.fiber((1, 3, 3, 1))
    assert all(fiber.contains(sample) for sample in first.samples)
    assert len(first.samples) == 25


def test_uniform_fiber_sampler_is_explicit_option():
    model = ToricModel(A_2X2)
    result = sample_fiber(
        (1, 3, 3, 1), model=model, size=10, burnin=2, rng=11, measure="uniform"
    )
    assert len(result.samples) == 10
    assert all(model.fiber((1, 3, 3, 1)).contains(sample) for sample in result.samples)
