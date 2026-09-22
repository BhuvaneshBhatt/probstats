import types

import numpy as np
import pytest
import sympy as sp

from probstats.bayes import (
    BayesianLinearRegression,
    Factor,
    Model,
    Parameter,
    RandomVariable,
)
from probstats.bayes.core import ObservationError
from probstats.bayes.distributions import Exponential
from probstats.bayes.distributions.student_t import MultivariateStudentT
from probstats.bayes.exact import (
    evidence,
    infer_exact,
    normalize_posterior,
)
from probstats.bayes.exact.symbolic import ExactIntegrationError
from probstats.bayes.laplace import (
    GaussianLaplacePosterior,
    LaplaceApproximationError,
    infer_laplace,
)
from probstats.bayes.models import LinearRegressionPrior
from probstats.bayes.reasoning import analyze_posterior_geometry
from probstats.distributions import (
    Normal,
    StudentT,
)


def test_distribution_support_controls_exact_integration():
    x = Parameter("x")  # declared Reals
    model = Model((x,), (Factor.from_distribution(x, Exponential(2)),))
    assert model.effective_support(x) == sp.Interval(0, sp.oo)
    assert model.latent_variables[0].support == sp.Interval(0, sp.oo)
    assert sp.simplify(evidence(model, backend="sympy") - 1) == 0


def test_declared_support_can_further_restrict_distribution_support():
    x = Parameter("x", support=sp.Interval(1, 2))
    model = Model((x,), (Factor.from_distribution(x, Exponential(2)),))
    assert model.effective_support(x) == sp.Interval(1, 2)


def test_observation_uses_effective_distribution_support():
    y = RandomVariable("y")
    model = Model((y,), (Factor.from_distribution(y, Exponential(1)),))
    with pytest.raises(ObservationError, match="outside support"):
        model.observe(y=-1)


def test_sympy_laplace_enforces_simple_target_assumption():
    x = Parameter("x")
    model = Model((x,), (Factor.from_log_density(x, -((x.symbol - 2) ** 2)),))
    result = infer_laplace(model, backend="sympy", assumptions=x.symbol > 1)
    assert result.posterior.mean == pytest.approx((2.0,))


def test_sympy_laplace_rejects_unsupported_coupled_constraint():
    x, y = Parameter("x"), Parameter("y")
    model = Model(
        (x, y),
        (
            Factor.from_log_density(x, -(x.symbol**2) / 2),
            Factor.from_log_density(y, -(y.symbol**2) / 2),
        ),
    )
    with pytest.raises(
        LaplaceApproximationError, match="Coupled optimization constraint"
    ):
        infer_laplace(model, backend="sympy", assumptions=x.symbol + y.symbol > 0)


def test_partial_target_laplace_rejects_unresolved_latent():
    x, y = Parameter("x"), Parameter("y")
    model = Model(
        (x, y),
        (
            Factor.from_log_density(x, -(x.symbol**2) / 2),
            Factor.from_log_density(y, -((y.symbol - x.symbol) ** 2) / 2),
        ),
    )
    with pytest.raises(
        LaplaceApproximationError, match="leave latent variables unresolved"
    ):
        infer_laplace(model, targets=["x"], backend="sympy")


class NegativeNormalizerBackend:
    name = "negative-test"

    def integrate(self, expr, variables, *, assumptions=None):
        return sp.Integer(-1)


@pytest.mark.parametrize("entry", ["evidence", "normalize", "infer"])
def test_all_exact_entry_points_share_nonpositive_normalizer_contract(entry):
    x = Parameter("x")
    model = Model((x,), (Factor.from_log_density(x, -(x.symbol**2)),))
    backend = NegativeNormalizerBackend()
    with pytest.raises(ExactIntegrationError, match="nonpositive|negative"):
        if entry == "evidence":
            evidence(model, backend=backend)
        elif entry == "normalize":
            normalize_posterior(model, backend=backend)
        else:
            infer_exact(model, backend=backend, fallback_to_conjugacy=False)


def test_funcprops_singularities_outside_domain_are_filtered():
    x = sp.Symbol("x", real=True)
    module = types.ModuleType("funcprops")
    module.function_convexity = lambda *a, **k: types.SimpleNamespace(
        primary=types.SimpleNamespace(value="unknown")
    )
    module.function_continuous = lambda *a, **k: types.SimpleNamespace(
        value="continuous"
    )
    module.function_singularities = lambda *a, **k: (sp.Eq(x, -1),)
    geometry = analyze_posterior_geometry(
        sp.log(x + 1), (x,), domain=x > 0, funcprops_loader=lambda: module
    )
    assert geometry.has_singularities is False


def test_invalid_distribution_parameters_fail_early():
    with pytest.raises(ValueError, match="Invalid parameters"):
        Normal(0, -1)
    with pytest.raises(ValueError, match="Invalid parameters"):
        Exponential(0)
    with pytest.raises(ValueError, match="scale"):
        StudentT(0, 0, 3)
    with pytest.raises(ValueError, match="degrees of freedom"):
        StudentT(0, 1, 0)


def test_invalid_matrix_probability_objects_fail_early():
    with pytest.raises(ValueError, match="positive definite"):
        MultivariateStudentT([0, 0], [[1, 2], [2, 1]], 4)
    with pytest.raises(ValueError, match="positive definite"):
        LinearRegressionPrior([0, 0], [[1, 2], [2, 1]], 1, 1)
    with pytest.raises(ValueError, match="scale"):
        LinearRegressionPrior([0], [[1]], 0, 1)


def test_gaussian_laplace_logpdf_uses_stable_logdet():
    cov = np.diag([1e-200, 1e-200])
    precision = np.diag([1e200, 1e200])
    post = GaussianLaplacePosterior(
        variables=(sp.Symbol("x"), sp.Symbol("y")),
        mean=(0.0, 0.0),
        covariance=cov,
        precision=precision,
    )
    value = float(sp.N(post.logpdf([0, 0])))
    assert np.isfinite(value)


def test_regression_exposes_public_design_dimension_for_planner():
    model = BayesianLinearRegression(include_intercept=True)
    assert model.design_dimension([[1, 2], [3, 4]]) == 3
