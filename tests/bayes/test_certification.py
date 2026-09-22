import types

import sympy as sp

from probstats.bayes import (
    Factor,
    Model,
    RandomVariable,
)
from probstats.bayes.certification import (
    CertificationStatus,
    ExpressionCertificate,
    certify_equal,
    certify_zero,
)
from probstats.bayes.diagnostics import assess_quality
from probstats.bayes.exact import infer_exact
from probstats.bayes.laplace import (
    LaplaceApproximationError,
    infer_laplace,
)
from probstats.bayes.laplace.backends import OptimizationResult
from probstats.distributions import Normal


def _fake_exprtest(result):
    calls = []
    module = types.ModuleType("exprtest")

    def zerotest(
        expr,
        assumptions=True,
        use_cache=True,
        *,
        rng=None,
        seed=None,
        confidence="probable",
    ):
        calls.append((expr, confidence))
        return result

    module.zerotest = zerotest
    return module, calls


def test_certify_zero_uses_exprtest_certified_mode():
    module, calls = _fake_exprtest(False)
    cert = certify_zero(sp.sqrt(2) - 1, exprtest_loader=lambda: module)
    assert isinstance(cert, ExpressionCertificate)
    assert cert.status is CertificationStatus.NONZERO
    assert cert.proven
    assert cert.backend == "exprtest"
    assert calls == [(sp.sqrt(2) - 1, "certified")]


def test_certify_zero_does_not_promote_unknown():
    module, _ = _fake_exprtest(None)
    x = sp.Symbol("x")
    cert = certify_zero(sp.sin(x), exprtest_loader=lambda: module)
    assert cert.status is CertificationStatus.UNKNOWN
    assert cert.is_zero is None
    assert not cert.proven


def test_certify_equal_reduces_to_zero_test():
    module, calls = _fake_exprtest(True)
    cert = certify_equal(
        sp.sin(sp.pi / 6),
        sp.Rational(1, 2),
        exprtest_loader=lambda: module,
    )
    assert cert.status is CertificationStatus.ZERO
    assert cert.is_zero is True
    assert calls[0][1] == "certified"


def _simple_model():
    theta = RandomVariable("theta", support=sp.S.Reals)
    return Model(
        [theta],
        [Factor.from_distribution(theta, Normal(0, 1), name="prior")],
    )


def test_exact_inference_records_exprtest_normalizer_certificate():
    module, _ = _fake_exprtest(False)
    result = infer_exact(
        _simple_model(), backend="sympy", exprtest_loader=lambda: module
    )
    cert = result.metadata["normalizer_certificate"]
    assert cert.backend == "exprtest"
    assert result.metadata["normalizer_certified_nonzero"] is True
    quality = assess_quality(result)
    assert quality.metrics["normalizer_certified_nonzero"] is True


class _FixedBackend:
    name = "fixed"

    def optimize(self, log_density, variables, **kwargs):
        return OptimizationResult(
            point=(0.0,), value=0.0, certified=True, attained=True
        )


def test_laplace_records_exprtest_precision_certificate():
    module, _ = _fake_exprtest(False)
    result = infer_laplace(
        _simple_model(), backend=_FixedBackend(), exprtest_loader=lambda: module
    )
    cert = result.diagnostics["precision_determinant_certificate"]
    assert cert.backend == "exprtest"
    assert cert.is_zero is False
    quality = assess_quality(result)
    assert quality.metrics["precision_determinant_certified_nonzero"] is True


def test_exprtest_zero_certificate_overrides_numeric_singularity_threshold():
    # A quartic mode has exactly zero Hessian determinant; exprtest proof is used
    # to route into singular-Laplace rather than treating the determinant as a
    # merely small floating value.  Missing corrector then gives the expected
    # singular-specific error.
    module, _ = _fake_exprtest(True)
    theta = RandomVariable("theta", support=sp.S.Reals)
    model = Model(
        [theta], [Factor.from_log_density(theta, -(theta.symbol**4), name="quartic")]
    )
    import pytest

    with pytest.raises(LaplaceApproximationError, match="singular"):
        infer_laplace(model, backend=_FixedBackend(), exprtest_loader=lambda: module)


def test_certification_public_api_is_documented():
    import inspect
    from pathlib import Path

    from probstats.bayes import certification

    text = (Path(__file__).parents[2] / "docs" / "api" / "certification.md").read_text()
    for name in certification.__all__:
        obj = getattr(certification, name)
        assert inspect.getdoc(obj), f"missing docstring for {name}"
        assert f"`{name}`" in text, f"missing API documentation for {name}"
