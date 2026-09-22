import pytest
import sympy as sp

from probstats._transformation_certification import (
    certify_common_pushforward,
    certify_scalar_transformation,
    transformed_entropy_value,
)
from probstats.distributions import Normal
from probstats.information import information_entropy, kl_divergence
from probstats.transforms import InverseBranch, Transform, TransformedDistribution

funcprops = pytest.importorskip("funcprops")


def _exp_transform():
    x, y = sp.symbols("x y", real=True)
    return Transform.from_expressions(
        sp.exp(x),
        x,
        sp.log(y),
        y,
        domain=sp.S.Reals,
        codomain=sp.Interval.open(0, sp.oo),
    )


def test_change_certificate_uses_stable_funcprops_api(monkeypatch):
    monkeypatch.delattr(funcprops, "function_change_of_variables", raising=False)
    x = sp.symbols("x", real=True)

    result = certify_scalar_transformation(
        sp.exp(x),
        x,
        domain=sp.S.Reals,
        require_change_of_variables=True,
    )

    assert result is not None
    assert result.change_of_variables.verify()
    assert sp.simplify(result.volume_factor - sp.exp(x)) == 0


def test_change_of_variables_certificate():
    x = sp.symbols("x", real=True)
    result = certify_scalar_transformation(
        sp.exp(x),
        x,
        domain=sp.S.Reals,
        require_change_of_variables=True,
    )

    assert result is not None
    assert result.bijective
    assert result.codomain == sp.Interval.open(0, sp.oo)
    assert result.change_of_variables is not None
    assert result.change_of_variables.verify()
    assert sp.simplify(result.volume_factor - sp.exp(x)) == 0


def test_kl_is_invariant_under_a_certified_common_pushforward():
    transform = _exp_transform()
    p = Normal(0, 1)
    q = Normal(1, 2)
    pushed_p = TransformedDistribution(p, transform)
    pushed_q = TransformedDistribution(q, transform)

    assert certify_common_pushforward(pushed_p, pushed_q) is not None
    assert sp.simplify(kl_divergence(pushed_p, pushed_q) - kl_divergence(p, q)) == 0


def test_common_pushforward_requires_one_map_on_the_union_of_supports():
    x, y = sp.symbols("x y", real=True)
    exp_transform = _exp_transform()
    affine_transform = Transform.from_expressions(
        2 * x,
        x,
        y / 2,
        y,
        domain=sp.S.Reals,
        codomain=sp.S.Reals,
    )

    left = TransformedDistribution(Normal(0, 1), exp_transform)
    right = TransformedDistribution(Normal(1, 2), affine_transform)

    assert certify_common_pushforward(left, right) is None


def test_entropy_uses_certified_jacobian():
    transform = _exp_transform()
    base = Normal(2, 1)
    pushed = TransformedDistribution(base, transform)

    value = transformed_entropy_value(pushed)

    assert value is not None
    assert sp.simplify(value - information_entropy(base) - 2) == 0
    assert sp.simplify(information_entropy(pushed) - value) == 0


def test_nonbijective_transform_does_not_apply_entropy_identity():
    x, y = sp.symbols("x y", real=True)
    transform = Transform.from_branches(
        x**2,
        x,
        y,
        (
            InverseBranch(sp.sqrt(y), y, sp.Interval(0, sp.oo)),
            InverseBranch(-sp.sqrt(y), y, sp.Interval(0, sp.oo)),
        ),
        domain=sp.S.Reals,
        codomain=sp.Interval(0, sp.oo),
    )
    pushed = TransformedDistribution(Normal(0, 1), transform)

    assert transformed_entropy_value(pushed) is None
