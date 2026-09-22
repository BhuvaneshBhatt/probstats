import sympy as sp

from probstats.algebraic import (
    central_moment_tensor,
    cumulant_tensor,
    decompose_moment_tensor,
    moment_tensor,
    multi_view_moment,
    recover_multiview_mixture,
)


def test_raw_and_central_moment_tensors_are_exact():
    data = ((1, 2), (3, 4))
    raw = moment_tensor(data, 2)
    central = central_moment_tensor(data, 2)
    assert raw.values == sp.ImmutableDenseNDimArray([[5, 7], [7, 10]])
    assert central.values == sp.ImmutableDenseNDimArray([[1, 1], [1, 1]])
    assert raw.kind == "raw_moment"
    assert central.centered is True


def test_cumulant_tensor_matches_covariance_at_order_two():
    data = ((1, 0), (3, 2))
    cumulant = cumulant_tensor(data, 2)
    central = central_moment_tensor(data, 2)
    assert cumulant.values == central.values


def test_third_cumulant_of_symmetric_scalar_sample_is_zero():
    result = cumulant_tensor(((-1,), (1,)), 3)
    assert result.values[(0, 0, 0)] == 0


def test_multi_view_moment_is_aligned_cross_moment():
    first = ((1, 0), (0, 1))
    second = ((2, 0), (0, 4))
    result = multi_view_moment((first, second))
    assert result.values == sp.ImmutableDenseNDimArray([[1, 0], [0, 2]])
    assert result.view_dimensions == (2, 2)


def test_exact_rank_one_multiview_recovery_normalizes_cp_scaling():
    tensor = sp.ImmutableDenseNDimArray(
        [
            sp.Rational(1, 16),
            sp.Rational(3, 16),
            sp.Rational(1, 16),
            sp.Rational(3, 16),
            sp.Rational(1, 16),
            sp.Rational(3, 16),
            sp.Rational(1, 16),
            sp.Rational(3, 16),
        ],
        (2, 2, 2),
    )
    result = recover_multiview_mixture(tensor, 1, method="exact")
    assert result.exact is True
    assert result.converged is True
    assert result.weights == (1,)
    for vector in result.component_means[0]:
        assert sp.simplify(sum(vector) - 1) == 0


def test_moment_decomposition_uses_symmetric_rank_one_certificate():
    tensor = sp.ImmutableDenseNDimArray([1, 2, 2, 4, 2, 4, 4, 8], (2, 2, 2))
    result = decompose_moment_tensor(
        tensor, components=1, symmetric=True, method="exact"
    )
    assert result.exact is True
    assert result.complete is True
    assert result.rank.rank == 1


def test_categorical_multi_view_moment_is_joint_probability_tensor():
    from probstats.algebraic import categorical_multi_view_moment

    result = categorical_multi_view_moment(
        ((0, 0, 1, 1), (0, 1, 0, 1)), cardinalities=(2, 2)
    )
    assert result.values == sp.ImmutableDenseNDimArray(
        [[sp.Rational(1, 4), sp.Rational(1, 4)], [sp.Rational(1, 4), sp.Rational(1, 4)]]
    )
    assert sum(result.values[index] for index in ((0, 0), (0, 1), (1, 0), (1, 1))) == 1


def test_multiview_recovery_rejects_non_probability_tensor():
    from probstats.algebraic import recover_multiview_mixture

    tensor = sp.ImmutableDenseNDimArray([1, 2, 2, 4, 2, 4, 4, 8], (2, 2, 2))
    try:
        recover_multiview_mixture(tensor, 1, method="exact")
    except ValueError as exc:
        assert "sum to one" in str(exc)
    else:
        raise AssertionError("expected probability-normalization validation")


def test_two_component_probability_mixture_recovery_uses_nonnegative_cp():
    tensor = sp.ImmutableDenseNDimArray(
        [sp.Rational(2, 5), 0, 0, 0, 0, 0, 0, sp.Rational(3, 5)],
        (2, 2, 2),
    )
    result = recover_multiview_mixture(
        tensor,
        2,
        method="numerical",
        rng=1,
        tolerance=1e-7,
    )
    assert result.converged is True
    assert result.identifiable is True
    assert result.certified_identifiable is True
    assert tuple(round(float(weight), 8) for weight in result.weights) == (0.4, 0.6)
    assert result.component_means[0][0][0] > 0.999999
    assert result.component_means[1][0][1] > 0.999999
