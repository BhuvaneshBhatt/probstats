import numpy as np
import sympy as sp

from probstats.algebraic.testing import (
    TRINOMIAL_PAPER_CONFIGURATION,
    multinomial_point_on_model,
    sdl_calibrate,
    sdl_test,
    sdl_union_test,
    trinomial_model4_components,
    trinomial_reference_hypothesis,
    trinomial_sampler,
)


def test_published_trinomial_configuration_is_recorded():
    cfg = TRINOMIAL_PAPER_CONFIGURATION
    assert (
        cfg.sample_size,
        cfg.budget,
        cfg.bootstrap_replicates,
        cfg.projection_size,
    ) == (300, 1000, 1000, 300)


def test_reference_models_match_published_polynomial_descriptions():
    x, y, z = sp.symbols("x y z")
    expected = {
        1: ((y - z,), ()),
        2: ((y - z,), (sp.Rational(1, 3) - x,)),
        3: (((y - z) * (x - y) * (x - z),), ()),
        4: (
            ((x - y) * (x - z) * (y - z),),
            (
                (x - z) ** 2 * (y - z) ** 2 * (sp.Rational(1, 3) - x),
                (x - y) ** 2 * (y - z) ** 2 * (sp.Rational(1, 3) - y),
                (x - y) ** 2 * (x - z) ** 2 * (sp.Rational(1, 3) - z),
            ),
        ),
    }
    for model, (equalities, inequalities) in expected.items():
        component = trinomial_reference_hypothesis(model).basic_null
        assert component.equalities == tuple(map(sp.expand, equalities))
        assert component.inequalities == tuple(map(sp.expand, inequalities))


def test_reference_ambient_is_the_probability_simplex():
    h = trinomial_reference_hypothesis(1)
    x, y, z = h.parameters
    assert bool(h.ambient.subs({x: 0.2, y: 0.4, z: 0.4}))
    assert not bool(h.ambient.subs({x: -0.1, y: 0.5, z: 0.6}))


def test_known_points_have_expected_model_membership():
    center = (1 / 3, 1 / 3, 1 / 3)
    assert all(multinomial_point_on_model(model, center) for model in range(1, 5))
    assert multinomial_point_on_model(1, (0.6, 0.2, 0.2))
    assert multinomial_point_on_model(2, (0.6, 0.2, 0.2))
    assert not multinomial_point_on_model(2, (0.2, 0.4, 0.4))


def test_model4_decomposition_has_three_model2_like_components():
    h = trinomial_model4_components()
    assert len(h.components) == 3
    assert all(len(component.equalities) == 1 for component in h.components)
    assert all(len(component.inequalities) == 1 for component in h.components)
    center = {parameter: 1 / 3 for parameter in h.parameters}
    assert all(bool(component.formula.subs(center)) for component in h.components)


def test_trinomial_sampler_reproducibility():
    sampler = trinomial_sampler((0.5, 0.25, 0.25), 20)
    a = sampler(np.random.default_rng(12))
    b = sampler(np.random.default_rng(12))
    np.testing.assert_array_equal(a, b)
    assert len(a) == 20
    assert all(np.sum(row) == 1 and set(row).issubset({0.0, 1.0}) for row in a)


def test_model1_reference_end_to_end():
    h = trinomial_reference_hypothesis(1)
    sample = trinomial_sampler((1 / 3, 1 / 3, 1 / 3), 300)(np.random.default_rng(3))
    result = sdl_test(sample, h, budget=60, bootstrap_replicates=25, n1=300, seed=9)
    assert result.sample_size == 300
    assert result.projection_size == 300
    assert result.constraint_count == 2  # equality -> g <= 0 and -g <= 0
    assert 0 <= result.p_value <= 1


def test_reference_calibration_smoke_uses_explicit_null_sampler():
    h = trinomial_reference_hypothesis(1)
    calibration = sdl_calibrate(
        trinomial_sampler((1 / 3, 1 / 3, 1 / 3), 40),
        h,
        repetitions=4,
        alpha=0.05,
        budget=20,
        bootstrap_replicates=9,
        seed=14,
    )
    assert calibration.repetitions == 4
    assert calibration.p_values.shape == (4,)
    assert 0 <= calibration.rejection_rate <= 1


def test_model4_decomposed_reference_runs_intersection_union_path():
    h = trinomial_model4_components()
    sample = trinomial_sampler((0.6, 0.2, 0.2), 50)(np.random.default_rng(5))
    result = sdl_union_test(sample, h, budget=20, bootstrap_replicates=9, seed=4)
    assert result.component_count == 3
    assert result.p_value == max(result.component_p_values)
