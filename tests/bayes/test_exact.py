import pytest
import sympy as sp

from probstats.bayes import (
    Factor,
    Model,
    NormalInverseGamma,
    Parameter,
    RandomVariable,
)
from probstats.bayes.exact import (
    ExactIntegrationError,
    evidence,
    infer_exact,
    marginalize,
    normalize_posterior,
)
from probstats.distributions import (
    Normal,
    SymbolicDistribution,
)


def gaussian_location_model():
    theta = Parameter("theta", sp.S.Reals)
    y = RandomVariable("y", sp.S.Reals)
    prior = Factor.from_distribution(theta, Normal(0, 1))
    like = Factor.from_distribution(y, Normal(theta.symbol, 1))
    return Model((theta, y), (prior, like)).observe(y=1), theta


def test_exact_normalization_and_evidence_for_normal_location():
    model, theta = gaussian_location_model()
    posterior, z = normalize_posterior(model)
    expected_z = sp.exp(-sp.Rational(1, 4)) / (2 * sp.sqrt(sp.pi))
    assert sp.simplify(z - expected_z) == 0
    expected = Normal(sp.Rational(1, 2), 1 / sp.sqrt(2)).pdf(theta.symbol)
    assert sp.simplify(posterior.density / expected) == 1
    assert sp.simplify(evidence(model) - expected_z) == 0


def test_infer_exact_records_log_evidence():
    model, _ = gaussian_location_model()
    result = infer_exact(model)
    assert result.is_exact
    assert result.steps[0].method == "symbolic-normalization"
    assert sp.simplify(sp.exp(result.log_evidence) - result.metadata["evidence"]) == 0


def test_symbolic_marginalization_eliminates_nuisance_variable():
    x = Parameter("x", sp.S.Reals)
    z = Parameter("z", sp.S.Reals)
    # Independent standard normals; posterior after normalization is already joint normal.
    fx = Factor.from_distribution(x, Normal(0, 1))
    fz = Factor.from_distribution(z, Normal(0, 1))
    model = Model((x, z), (fx, fz))
    posterior, norm = normalize_posterior(model, targets=["x"])
    assert norm == 1
    assert posterior.symbols == (x.symbol,)
    assert sp.simplify(posterior.density / Normal(0, 1).pdf(x.symbol)) == 1


def test_marginalize_raw_joint():
    x = Parameter("x", sp.S.Reals)
    z = Parameter("z", sp.S.Reals)
    model = Model(
        (x, z),
        (
            Factor.from_distribution(x, Normal(0, 1)),
            Factor.from_distribution(z, Normal(0, 1)),
        ),
    )
    result = marginalize(model, ["z"])
    assert sp.simplify(result / Normal(0, 1).pdf(x.symbol)) == 1


def test_discrete_exact_evidence_uses_summation():
    k = Parameter("k", sp.S.Naturals0)
    # Proper geometric mass 2^-(k+1).
    dist = SymbolicDistribution(
        k.symbol, -(k.symbol + 1) * sp.log(2), sp.S.Naturals0, True
    )
    model = Model((k,), (Factor.from_distribution(k, dist),))
    assert evidence(model) == 1


def test_improper_density_rejected():
    x = Parameter("x", sp.S.Reals)
    model = Model((x,), (Factor.from_log_density(x, 0),))
    with pytest.raises(ExactIntegrationError):
        normalize_posterior(model)


def test_conjugate_fallback_when_direct_model_is_not_integrable():
    # An unsupported domain makes the direct route inapplicable.
    x = Parameter("x", sp.ConditionSet(sp.Symbol("u"), sp.Symbol("u") > 0, sp.S.Reals))
    model = Model((x,), (Factor.from_log_density(x, -(x.symbol**2)),))
    likelihood = Normal(sp.Symbol("mu", real=True), sp.Symbol("sigma", positive=True))
    prior = NormalInverseGamma(0, 1, 1, 2)
    result = infer_exact(model, likelihood=likelihood, data=[1, 2, 3], prior=prior)
    assert result.is_exact
    assert result.steps[0].method == "conjugacy-fallback"
    assert result.metadata["fallback_from"] == "symbolic-normalization"


def test_multiple_integrate_batches_ranges():
    from probstats.bayes.exact import (
        MultipleIntegrateBackend,
        integrate_over,
    )

    x = Parameter("x", sp.S.Reals)
    y = Parameter("y", sp.Interval(0, 1))
    calls = []

    def fake(expr, *ranges, **kwargs):
        calls.append((expr, ranges, kwargs))
        return sp.Integer(7)

    backend = MultipleIntegrateBackend(fake)
    a = sp.symbols("a", positive=True)
    value = integrate_over(
        sp.exp(-a * x.symbol**2) * y.symbol,
        (x, y),
        backend=backend,
        assumptions={a > 0},
    )
    assert value == 7
    assert calls[0][1] == (
        (x.symbol, -sp.oo, sp.oo),
        (y.symbol, 0, 1),
    )
    assert calls[0][2] == {"assumptions": {a > 0}}


def test_multiple_integrate_backend_rejects_discrete_support():
    from probstats.bayes.exact import (
        ExactIntegrationError,
        MultipleIntegrateBackend,
        integrate_over,
    )

    k = Parameter("k", sp.S.Naturals0)
    with pytest.raises(ExactIntegrationError, match="continuous interval support"):
        integrate_over(
            2 ** (-k.symbol), (k,), backend=MultipleIntegrateBackend(lambda *a, **k: 1)
        )


def test_auto_exact_backend_prefers_structured_backend_when_available():
    from probstats.bayes.exact import (
        AutoExactIntegrationBackend,
        MultipleIntegrateBackend,
    )

    model, _ = gaussian_location_model()
    calls = []

    def fake_integrator(expr, *ranges, **kwargs):
        calls.append((expr, ranges, kwargs))
        out = expr
        for symbol, lo, hi in ranges:
            out = sp.integrate(out, (symbol, lo, hi))
        return out

    backend = AutoExactIntegrationBackend(
        structured=MultipleIntegrateBackend(fake_integrator)
    )
    result = infer_exact(model, backend=backend)
    assert calls
    assert result.metadata["engine"] == "multiple-integrate"
    assert result.metadata["integration_attempts"] == ("multiple-integrate",)


def test_auto_exact_falls_back_to_sympy():
    from probstats.bayes.exact import (
        AutoExactIntegrationBackend,
        MultipleIntegrateBackend,
    )

    model, _ = gaussian_location_model()

    def missing_loader():
        from probstats._integration import StructuredIntegrationUnavailable

        raise StructuredIntegrationUnavailable("not installed")

    backend = AutoExactIntegrationBackend(
        structured=MultipleIntegrateBackend(loader=missing_loader)
    )
    result = infer_exact(model, backend=backend)
    assert result.metadata["engine"] == "sympy"
    assert result.metadata["integration_attempts"] == ("multiple-integrate", "sympy")


def test_explicit_multiple_integrate_missing_dependency_is_actionable():
    from probstats._integration import StructuredIntegrationUnavailable
    from probstats.bayes.exact import (
        ExactBackendUnavailableError,
        MultipleIntegrateBackend,
    )

    def missing_loader():
        raise StructuredIntegrationUnavailable(
            "multiple-integrate is not installed; install probstats[exact]."
        )

    x = Parameter("x", sp.S.Reals)
    backend = MultipleIntegrateBackend(loader=missing_loader)
    with pytest.raises(ExactBackendUnavailableError, match=r"probstats\[exact\]"):
        backend.integrate(sp.exp(-(x.symbol**2)), (x,))


def test_get_exact_integration_backend_names():
    from probstats.bayes.exact import (
        AutoExactIntegrationBackend,
        MultipleIntegrateBackend,
        SymPyExactIntegrationBackend,
        get_exact_integration_backend,
    )

    assert isinstance(
        get_exact_integration_backend("auto"), AutoExactIntegrationBackend
    )
    assert isinstance(
        get_exact_integration_backend("sympy"), SymPyExactIntegrationBackend
    )
    assert isinstance(
        get_exact_integration_backend("multiple-integrate"), MultipleIntegrateBackend
    )
    with pytest.raises(ValueError, match="Unknown exact-integration backend"):
        get_exact_integration_backend("multiple_integrate")
    with pytest.raises(ValueError, match="Unknown exact-integration backend"):
        get_exact_integration_backend("bogus")


def test_exact_public_functions_have_docstrings_and_api_docs():
    import inspect
    from pathlib import Path

    from probstats.bayes import exact

    docs = Path(__file__).parents[2] / "docs" / "api" / "exact.md"
    text = docs.read_text()
    for name in exact.__all__:
        obj = getattr(exact, name)
        if inspect.isfunction(obj):
            assert inspect.getdoc(obj), f"missing docstring for {name}"
            assert f"`{name}`" in text, f"missing API documentation for {name}"


def test_exact_public_backend_classes_have_docstrings():
    import inspect

    from probstats.bayes.exact import (
        AutoExactIntegrationBackend,
        ExactIntegrationTrace,
        MultipleIntegrateBackend,
        SymbolicJointDistribution,
        SymPyExactIntegrationBackend,
    )

    for cls in (
        AutoExactIntegrationBackend,
        ExactIntegrationTrace,
        MultipleIntegrateBackend,
        SymPyExactIntegrationBackend,
        SymbolicJointDistribution,
    ):
        assert inspect.getdoc(cls), f"missing docstring for {cls.__name__}"
