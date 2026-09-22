import inspect
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import sympy as sp

from probstats.bayes import (
    Factor,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.diagnostics import assess_quality
from probstats.bayes.exact import infer_exact
from probstats.bayes.laplace import infer_laplace
from probstats.bayes.laplace.backends import OptimizationResult
from probstats.bayes.planner import (
    InferencePlanner,
    PlannerConfig,
)
from probstats.bayes.reasoning import (
    ProofStatus,
    analyze_posterior_geometry,
    certify_positive,
    certify_positive_definite,
)
from probstats.distributions import Normal


def _fake_semialg(*, positive=True, nonpositive=False):
    calls = []
    module = types.ModuleType("semialg")

    def make(name, truth):
        def fn(
            expr, variables=None, *, assumptions=True, return_result=False, **kwargs
        ):
            calls.append((name, sp.sympify(expr), assumptions))
            result = SimpleNamespace(
                proven=bool(truth),
                counterexample=None
                if truth
                else (
                    {sp.Symbol("w"): 0}
                    if name == "nonpositive" and nonpositive
                    else None
                ),
                method="fake-cad",
            )
            return result if return_result else bool(truth)

        return fn

    module.prove_positive = make("positive", positive)
    module.prove_nonnegative = make("nonnegative", positive)
    module.prove_negative = make("negative", False)
    module.prove_nonpositive = make("nonpositive", nonpositive)
    module.prove_zero = make("zero", False)
    module.prove_nonzero = make("nonzero", positive)
    return module, calls


def _fake_funcprops(
    *,
    convexity="strongly_concave",
    continuity="continuous",
    singularities=(),
):
    module = types.ModuleType("funcprops")
    calls = []

    def function_convexity(expr, var, **kwargs):
        calls.append(("convexity", expr, var, kwargs))
        return SimpleNamespace(
            primary=SimpleNamespace(value=convexity), source="fake-funcprops"
        )

    def function_continuous(expr, var, **kwargs):
        calls.append(("continuity", expr, var, kwargs))
        return SimpleNamespace(value=continuity)

    def function_singularities(expr, var, **kwargs):
        calls.append(("singularities", expr, var, kwargs))
        return tuple(singularities)

    module.function_convexity = function_convexity
    module.function_continuous = function_continuous
    module.function_singularities = function_singularities
    return module, calls


def _simple_model():
    theta = Parameter("theta")
    y = RandomVariable("y")
    return Model(
        (theta, y),
        (
            Factor.from_distribution(theta, Normal(0, 1)),
            Factor.from_distribution(y, Normal(theta.symbol, 1)),
        ),
    ).observe(y=1)


def test_semialg_certify_positive_preserves_assumptions():
    module, calls = _fake_semialg(positive=True)
    a = sp.Symbol("a", real=True)
    cert = certify_positive(a, assumptions=a > 0, semialg_loader=lambda: module)
    assert cert.status is ProofStatus.TRUE
    assert cert.backend == "semialg"
    assert calls[0][0] == "positive"
    assert calls[0][2] == (a > 0)


def test_semialg_failed_proof_without_counterexample_is_unknown():
    module, _ = _fake_semialg(positive=False)
    a = sp.Symbol("a")
    cert = certify_positive(a, semialg_loader=lambda: module)
    assert cert.status is ProofStatus.UNKNOWN
    assert cert.value is None


def test_positive_definite_uses_certified_leading_minors():
    module, calls = _fake_semialg(positive=True)
    a, b = sp.symbols("a b", positive=True)
    cert = certify_positive_definite(
        sp.diag(a, b),
        assumptions=sp.And(a > 0, b > 0),
        semialg_loader=lambda: module,
    )
    assert cert.positive_definite is True
    assert cert.leading_principal_minors == (a, a * b)
    assert [c[0] for c in calls].count("positive") == 2


def test_exact_inference_records_semialg_positive_normalizer_certificate():
    module, _ = _fake_semialg(positive=True)
    result = infer_exact(
        _simple_model(), backend="sympy", semialg_loader=lambda: module
    )
    cert = result.metadata["positive_certificate"]
    assert cert.backend == "semialg"
    assert result.metadata["normalizer_certified_positive"] is True
    quality = assess_quality(result)
    assert quality.metrics["normalizer_certified_positive"] is True


class _FixedBackend:
    name = "fixed"

    def optimize(self, log_density, variables, **kwargs):
        return OptimizationResult(
            point=(0.5,),
            value=float(sp.N(log_density.subs(variables[0], sp.Rational(1, 2)))),
        )


def test_laplace_records_semialg_definiteness_and_geometry():
    semialg, _ = _fake_semialg(positive=True)
    funcprops, _ = _fake_funcprops(
        convexity="strongly_concave", continuity="continuous"
    )
    result = infer_laplace(
        _simple_model(),
        backend=_FixedBackend(),
        semialg_loader=lambda: semialg,
        funcprops_loader=lambda: funcprops,
    )
    assert result.diagnostics["precision_certified_positive_definite"] is True
    geometry = result.diagnostics["posterior_geometry"]
    assert geometry.globally_concave is True
    assert geometry.continuity is True
    quality = assess_quality(result)
    assert quality.metrics["precision_certified_positive_definite"] is True
    assert quality.metrics["posterior_global_concavity"] is True


def test_funcprops_geometry_detects_singularities():
    x = sp.Symbol("x", real=True)
    module, _ = _fake_funcprops(
        convexity="unknown",
        continuity="continuous",
        singularities=(sp.Eq(x, 0),),
    )
    geometry = analyze_posterior_geometry(
        sp.log(x**2), (x,), funcprops_loader=lambda: module
    )
    assert geometry.has_singularities is True
    assert geometry.continuity is True


def test_planner_prefers_nested_when_funcprops_finds_singularity():
    module, _ = _fake_funcprops(
        convexity="unknown",
        continuity="continuous",
        singularities=(sp.Symbol("s"),),
    )
    model = _simple_model()
    planner = InferencePlanner(
        config=PlannerConfig(max_exact_operations=0, prefer_exact=False),
        geometry_analyzer=lambda *args, **kwargs: analyze_posterior_geometry(
            *args, **kwargs, funcprops_loader=lambda: module
        ),
    )
    plan = planner.plan(model, prior_sampler=lambda rng: np.array([rng.normal()]))
    assert plan.selected.method == "nested-sampling"
    laplace = next(c for c in plan.candidates if c.method == "laplace")
    assert laplace.metadata["has_singularities"] is True
    assert "singular" in laplace.reason.lower()


def test_reasoning_public_api_is_documented():
    from probstats.bayes import reasoning

    text = (Path(__file__).parents[2] / "docs" / "api" / "reasoning.md").read_text()
    for name in reasoning.__all__:
        obj = getattr(reasoning, name)
        assert inspect.getdoc(obj), f"missing docstring for {name}"
        assert f"`{name}`" in text, f"missing API documentation for {name}"
