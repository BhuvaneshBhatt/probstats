import math
import types
from typing import ClassVar

import numpy as np
import pytest
import sympy as sp

from probstats.bayes import (
    Factor,
    InferenceKind,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.laplace import (
    AsymptoticCorrector,
    AsymptoticIntegrationError,
    GaussianLaplacePosterior,
    LaplaceApproximationError,
    OptimizationError,
    OptimizationResult,
    SymboptOptimizationBackend,
    fit_precision_at_max,
    infer_laplace,
    laplace_log_evidence,
    precision_at,
    symbolic_hessian,
)
from probstats.distributions import Normal


def gaussian_model():
    theta = Parameter("theta")
    y = RandomVariable("y")
    model = Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1)
    return model, theta


def test_symbolic_hessian_and_precision():
    x, y = sp.symbols("x y", real=True)
    expr = -(x**2) - 3 * y**2 - x * y
    assert symbolic_hessian(expr, (x, y)) == sp.Matrix([[-2, -1], [-1, -6]])
    np.testing.assert_allclose(precision_at(expr, (x, y), (0, 0)), [[2, 1], [1, 6]])


def test_laplace_log_evidence_quadratic_is_exact():
    value = -sp.log(2 * sp.pi)
    got = laplace_log_evidence(value, np.array([[2.0]]))
    expected = value + sp.log(2 * sp.pi) / 2 - sp.log(2) / 2
    assert abs(float(sp.N(got - expected))) < 1e-12


def test_sympy_laplace_matches_gaussian_exact_posterior_and_evidence():
    model, theta = gaussian_model()
    result = infer_laplace(model, backend="sympy")
    assert result.kind is InferenceKind.APPROXIMATE
    assert isinstance(result.posterior, GaussianLaplacePosterior)
    assert result.posterior.variables == (theta.symbol,)
    np.testing.assert_allclose(result.posterior.mean, [0.5], atol=1e-12)
    np.testing.assert_allclose(result.posterior.precision, [[2.0]], atol=1e-12)
    np.testing.assert_allclose(result.posterior.covariance, [[0.5]], atol=1e-12)
    exact_log_z = sp.log(sp.exp(-sp.Rational(1, 4)) / (2 * sp.sqrt(sp.pi)))
    assert abs(float(sp.N(result.log_evidence - exact_log_z))) < 1e-12
    assert result.metadata["optimization_backend"] == "sympy"


def test_scipy_backend_matches_sympy_on_gaussian():
    pytest.importorskip("scipy")
    model, _ = gaussian_model()
    result = infer_laplace(model, backend="scipy", initial_guess=[0.0])
    np.testing.assert_allclose(result.posterior.mean, [0.5], atol=1e-8)
    np.testing.assert_allclose(result.posterior.precision, [[2.0]], atol=1e-8)


def test_nonmaximum_stationary_point_is_rejected_by_precision_check():
    theta = Parameter("theta")
    model = Model((theta,), (Factor.from_log_density(theta, theta.symbol**2),))
    with pytest.raises(LaplaceApproximationError, match="positive definite"):
        infer_laplace(model, backend="sympy")


def test_symbopt_backend_accepts_explicit_adapter_callable():
    x = sp.Symbol("x", real=True)

    def optimizer(expr, variables, **kwargs):
        assert expr == -((x - 2) ** 2)
        assert variables == (x,)
        return ([2.0], 0.0)

    backend = SymboptOptimizationBackend()
    result = backend.optimize(-((x - 2) ** 2), (x,), options={"optimizer": optimizer})
    assert result == OptimizationResult((2.0,), 0.0, raw=([2.0], 0.0))


def test_higher_order_corrector_hook_is_additive():
    model, _ = gaussian_model()

    def correction_function(**kwargs):
        assert kwargs["order"] == 4
        return sp.Rational(1, 10)

    base = infer_laplace(model, backend="sympy")
    refined = infer_laplace(
        model,
        backend="sympy",
        order=4,
        corrector=AsymptoticCorrector(correction_function),
    )
    assert (
        abs(float(sp.N(refined.log_evidence - base.log_evidence - sp.Rational(1, 10))))
        < 1e-12
    )
    assert refined.metadata["higher_order_correction"] == sp.Rational(1, 10)
    assert len(refined.steps) == 2


def test_order_above_two_requires_corrector():
    model, _ = gaussian_model()
    with pytest.raises(LaplaceApproximationError, match="corrector"):
        infer_laplace(model, backend="sympy", order=4)


def test_fit_precision_at_max_recovers_quadratic():
    precision = np.array([[3.0, 0.5], [0.5, 2.0]])
    pts = []
    values = []
    for x in [-1.0, 0.0, 1.0]:
        for y in [-1.0, 0.0, 1.0]:
            d = np.array([x, y])
            pts.append(d)
            values.append(5.0 - 0.5 * d @ precision @ d)
    fitted = fit_precision_at_max(pts, values)
    np.testing.assert_allclose(fitted, precision, atol=1e-12)


def test_gaussian_posterior_logpdf_at_mean():
    post = GaussianLaplacePosterior(
        variables=(sp.Symbol("x"),),
        mean=(1.0,),
        covariance=np.array([[0.25]]),
        precision=np.array([[4.0]]),
    )
    got = float(sp.N(post.logpdf([1.0])))
    expected = -0.5 * math.log(2 * math.pi * 0.25)
    assert abs(got - expected) < 1e-12


def test_symbopt_backend_uses_supported_pipeline_api():
    x = sp.Symbol("x", real=True)

    class Candidate:
        original_variable_values: ClassVar[dict] = {x: sp.Integer(2)}
        objective_value = sp.Integer(5)

    class PipelineResult:
        best_candidate = Candidate()
        optimum_value = sp.Integer(5)
        attained = True
        certified = True
        status = "global_optimal"

    seen = {}

    def maximize(objective, constraints, **kwargs):
        seen["objective"] = objective
        seen["constraints"] = constraints
        seen["kwargs"] = kwargs
        return PipelineResult()

    module = types.SimpleNamespace(maximize=maximize)
    backend = SymboptOptimizationBackend(loader=lambda: module)
    result = backend.optimize(
        -((x - 2) ** 2) + 5,
        (x,),
        supports=(sp.Interval.open(0, 3),),
        assumptions=sp.Ge(x, sp.Rational(1, 2)),
        options={"working_precision": 50, "semialg_certification": "complete"},
    )
    assert seen["objective"] == -((x - 2) ** 2) + 5
    assert seen["constraints"] == [
        sp.Gt(x, 0),
        sp.Lt(x, 3),
        sp.Ge(x, sp.Rational(1, 2)),
    ]
    assert seen["kwargs"]["variables"] == [x]
    assert seen["kwargs"]["working_precision"] == 50
    assert result.point == (2.0,)
    assert result.value == 5.0
    assert result.certified is True
    assert result.attained is True
    assert result.status == "global_optimal"
    assert result.global_value == 5


def test_symbopt_backend_rejects_unattained_supremum():
    x = sp.Symbol("x", real=True)

    class PipelineResult:
        best_candidate = None
        optimum_value = sp.Integer(0)
        attained = False
        certified = True
        status = "global_supremum"

    module = types.SimpleNamespace(maximize=lambda *args, **kwargs: PipelineResult())
    backend = SymboptOptimizationBackend(loader=lambda: module)
    with pytest.raises(OptimizationError, match="not attained"):
        backend.optimize(x, (x,), supports=(sp.Interval.open(-1, 0),))


def test_symbopt_backend_rejects_discrete_support():
    module = types.SimpleNamespace(maximize=lambda *args, **kwargs: None)
    x = sp.Symbol("x", integer=True)
    backend = SymboptOptimizationBackend(loader=lambda: module)
    with pytest.raises(OptimizationError, match="continuous"):
        backend.optimize(-(x**2), (x,), supports=(sp.S.Integers,))


def test_laplace_public_functions_have_docstrings_and_api_docs():
    import inspect
    from pathlib import Path

    from probstats.bayes import laplace

    docs = Path(__file__).parents[2] / "docs" / "api" / "laplace.md"
    text = docs.read_text()
    for name in laplace.__all__:
        obj = getattr(laplace, name)
        if inspect.isfunction(obj):
            assert inspect.getdoc(obj), f"missing docstring for {name}"
            assert f"`{name}`" in text, f"missing API documentation for {name}"


def test_infer_laplace_propagates_symbopt_certification():
    model, theta = gaussian_model()

    class Candidate:
        original_variable_values: ClassVar[dict] = {theta.symbol: sp.Rational(1, 2)}
        objective_value = model.joint_log_density(substitute_observations=True).subs(
            theta.symbol, sp.Rational(1, 2)
        )

    class PipelineResult:
        best_candidate = Candidate()
        optimum_value = Candidate.objective_value
        attained = True
        certified = True
        status = "proven_optimal"

    module = types.SimpleNamespace(maximize=lambda *args, **kwargs: PipelineResult())
    backend = SymboptOptimizationBackend(loader=lambda: module)
    result = infer_laplace(model, backend=backend)
    assert result.metadata["optimization_backend"] == "symbopt"
    assert result.metadata["optimization_certified"] is True
    assert result.metadata["optimization_attained"] is True
    assert result.metadata["optimization_status"] == "proven_optimal"
    np.testing.assert_allclose(result.posterior.mean, [0.5], atol=1e-12)


def _fake_asymptotic(expression_builder, *, status="CERTIFIED"):
    calls = []

    class Result:
        def __init__(self, expression):
            self.expression = expression
            self.status = status
            self.certified = status in {"EXACT", "CERTIFIED"}
            self.remainder = "R"
            self.certificate = "C"

    def laplace_asymptotic_integral(integrand, variable, domain, **kwargs):
        calls.append((integrand, variable, domain, kwargs))
        return Result(expression_builder(integrand, variable, domain, kwargs))

    module = types.SimpleNamespace(
        laplace_asymptotic_integral=laplace_asymptotic_integral
    )
    return module, calls


def test_concrete_asymptotic_corrector_refines_regular_gaussian():
    # The auxiliary integral for ell(x)=-x^2 is sqrt(pi/lambda), exactly.
    module, calls = _fake_asymptotic(
        lambda integrand, variable, domain, kwargs: sp.sqrt(
            sp.pi / kwargs["parameter"]
        ),
        status="EXACT",
    )
    x = Parameter("x")
    model = Model((x,), (Factor.from_log_density(x, -(x.symbol**2)),))
    result = infer_laplace(
        model,
        backend="sympy",
        order=4,
        corrector=AsymptoticCorrector(terms=3, loader=lambda: module),
    )
    assert len(calls) == 1
    integrand, variable, domain, kwargs = calls[0]
    assert variable == x.symbol
    assert domain == sp.Interval(-sp.oo, sp.oo)
    assert kwargs["terms"] == 3
    assert kwargs["certify"] is True
    assert integrand.has(kwargs["parameter"])
    assert abs(float(sp.N(result.log_evidence - sp.log(sp.sqrt(sp.pi))))) < 1e-12
    assert result.metadata["asymptotic_status"] == "EXACT"
    assert result.metadata["asymptotic_certified"] is True
    assert result.metadata["singular_laplace"] is False


def test_singular_quartic_laplace_uses_asymptotic_and_builds_posterior():
    from probstats.bayes.laplace import SingularLaplacePosterior

    module, calls = _fake_asymptotic(
        lambda integrand, variable, domain, kwargs: (
            sp.gamma(sp.Rational(1, 4)) / (2 * kwargs["parameter"] ** sp.Rational(1, 4))
        ),
        status="CERTIFIED",
    )
    x = Parameter("x")
    model = Model((x,), (Factor.from_log_density(x, -(x.symbol**4)),))
    result = infer_laplace(
        model,
        backend="sympy",
        corrector=AsymptoticCorrector(terms=2, loader=lambda: module),
        order=4,
    )
    assert len(calls) == 1
    assert isinstance(result.posterior, SingularLaplacePosterior)
    assert result.posterior.local_orders == (4,)
    assert result.posterior.local_coefficients == (sp.Integer(1),)
    expected = sp.log(sp.gamma(sp.Rational(1, 4)) / 2)
    assert abs(float(sp.N(result.log_evidence - expected))) < 1e-12
    assert result.metadata["base_log_evidence"] is None
    assert result.metadata["singular_laplace"] is True
    assert result.metadata["asymptotic_certified"] is True


def test_singular_laplace_posterior_is_normalized_for_quartic_leading_term():
    from probstats.bayes.laplace import SingularLaplacePosterior

    x = sp.Symbol("x", real=True)
    posterior = SingularLaplacePosterior((x,), (0.0,), (4,), (sp.Integer(1),))
    integral = sp.integrate(posterior.pdf([x]), (x, -sp.oo, sp.oo))
    assert sp.simplify(integral - 1) == 0


def test_asymptotic_corrector_factorizes_separable_multivariate_model():
    module, calls = _fake_asymptotic(
        lambda integrand, variable, domain, kwargs: (
            sp.sqrt(sp.pi / kwargs["parameter"])
            if variable.name == "x"
            else sp.gamma(sp.Rational(1, 4))
            / (2 * kwargs["parameter"] ** sp.Rational(1, 4))
        ),
    )
    x = Parameter("x")
    y = Parameter("y")
    model = Model(
        (x, y),
        (
            Factor.from_log_density(x, -(x.symbol**2)),
            Factor.from_log_density(y, -(y.symbol**4)),
        ),
    )
    result = infer_laplace(
        model,
        backend="sympy",
        corrector=AsymptoticCorrector(loader=lambda: module),
        order=4,
    )
    assert len(calls) == 2
    assert result.posterior.local_orders == (2, 4)
    expected_z = sp.sqrt(sp.pi) * sp.gamma(sp.Rational(1, 4)) / 2
    assert abs(float(sp.N(sp.exp(result.log_evidence) - expected_z))) < 1e-12


def test_asymptotic_corrector_rejects_coupled_singular_multivariate_model():
    module, _ = _fake_asymptotic(lambda *args: sp.Integer(1))
    x = Parameter("x")
    y = Parameter("y")
    # Hessian is singular at zero and x^2*y^2 couples the higher-order geometry.
    model = Model(
        (x, y),
        (
            Factor.from_log_density(
                x, -(x.symbol**4 + y.symbol**4 + x.symbol**2 * y.symbol**2)
            ),
        ),
    )
    with pytest.raises(LaplaceApproximationError, match="coordinate-separable"):
        infer_laplace(
            model,
            backend="sympy",
            corrector=AsymptoticCorrector(loader=lambda: module),
            order=4,
        )


def test_asymptotic_optional_dependency_error_is_actionable():
    def missing_loader():
        raise ImportError("missing")

    x = sp.Symbol("x")
    with pytest.raises(AsymptoticIntegrationError, match=r"probstats\[asymptotic\]"):
        AsymptoticCorrector(loader=missing_loader).analyze(
            log_density=-(x**2),
            variables=(x,),
            point=(0.0,),
            supports=(sp.S.Reals,),
            precision=np.array([[2.0]]),
            order=4,
            base_log_evidence=sp.log(sp.sqrt(sp.pi)),
        )


def test_new_asymptotic_public_api_is_documented():
    import inspect
    from pathlib import Path

    from probstats.bayes.laplace import (
        AsymptoticCorrector,
        AsymptoticIntegrationError,
        AsymptoticLaplaceAnalysis,
        SingularLaplacePosterior,
    )

    text = (Path(__file__).parents[2] / "docs" / "api" / "laplace.md").read_text()
    for obj in (
        AsymptoticCorrector,
        AsymptoticIntegrationError,
        AsymptoticLaplaceAnalysis,
        SingularLaplacePosterior,
    ):
        assert inspect.getdoc(obj), f"missing public API docstring for {obj.__name__}"
        assert f"`{obj.__name__}`" in text, (
            f"missing API documentation for {obj.__name__}"
        )
    for cls, methods in (
        (AsymptoticCorrector, ("analyze", "correction")),
        (AsymptoticLaplaceAnalysis, ("singular",)),
        (SingularLaplacePosterior, ("dimension", "logpdf", "pdf")),
    ):
        for method_name in methods:
            assert inspect.getdoc(getattr(cls, method_name)), (
                f"missing docstring for {cls.__name__}.{method_name}"
            )
            assert method_name in text, (
                f"missing API docs for {cls.__name__}.{method_name}"
            )
