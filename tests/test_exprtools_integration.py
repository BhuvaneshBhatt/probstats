"""New backend defaults and end-to-end compatibility for injected loaders."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import sympy as sp

from probstats.bayes import Factor, Model, RandomVariable
from probstats.bayes.certification import certify_equal, certify_zero
from probstats.bayes.exact import infer_exact
from probstats.bayes.laplace import infer_laplace
from probstats.bayes.laplace.backends import OptimizationResult
from probstats.distributions import Normal


def oracle_with(result):
    calls = []

    def zero(expr, **kwargs):
        calls.append((expr, kwargs))
        return result

    return SimpleNamespace(zerotest=zero), calls


def test_default_backend_loads_exprtools():
    result = certify_zero(sp.sqrt(2) - 1)
    assert result.backend == "exprtools" and result.is_zero is False


@pytest.mark.parametrize("answer", [True, False, None])
def test_new_loader_confidence_and_unknown(answer):
    oracle, calls = oracle_with(answer)
    x = sp.Symbol("x")
    result = certify_equal(sp.sin(x), 0, exprtools_loader=lambda: oracle)
    assert result.is_zero is answer
    assert calls == [(sp.sin(x), {"confidence": "certified"})]
    if answer is not None:
        assert result.backend == "exprtools"


@pytest.mark.parametrize("flag", ["use_exprtools", "use_exprtest"])
def test_both_disable_keywords_preserve_native_fallback(flag):
    with patch("probstats.bayes.certification.core.import_module") as load:
        assert certify_zero(0, **{flag: False}).backend == "sympy"
    load.assert_not_called()


def test_missing_backend_falls_back_without_promoting_unknown():
    with patch(
        "probstats.bayes.certification.core.import_module", side_effect=ImportError
    ):
        assert certify_zero(sp.sin(sp.Symbol("x"))).is_zero is None


def test_simultaneous_loaders_are_rejected():
    with pytest.raises(ValueError, match="only one"):
        certify_zero(0, exprtools_loader=lambda: None, exprtest_loader=lambda: None)


def simple_model():
    theta = RandomVariable("theta", support=sp.S.Reals)
    return Model([theta], [Factor.from_distribution(theta, Normal(0, 1), name="prior")])


class FixedBackend:
    name = "fixed"

    def optimize(self, log_density, variables, **kwargs):
        return OptimizationResult(
            point=(0.0,), value=0.0, certified=True, attained=True
        )


def test_new_loader_reaches_exact_normalizer_and_laplace_hessian():
    oracle, calls = oracle_with(False)
    exact = infer_exact(
        simple_model(), backend="sympy", exprtools_loader=lambda: oracle
    )
    assert exact.metadata["normalizer_certificate"].backend == "exprtools"
    laplace = infer_laplace(
        simple_model(), backend=FixedBackend(), exprtools_loader=lambda: oracle
    )
    assert (
        laplace.diagnostics["precision_determinant_certificate"].backend == "exprtools"
    )
    assert len(calls) == 2
