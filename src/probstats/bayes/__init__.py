"""High-level Bayesian modeling and inference workflows.

The package facade exposes common model, conjugacy, prediction, comparison,
and planning APIs. Method-specific samplers, optimization backends, exact
integration tools, diagnostics, and specialist models live in their owning
subpackages. Facade objects are loaded on first access so importing
``probstats.bayes`` does not initialize every inference backend.
"""

from importlib import import_module as _import_module

_MODULES = {
    "certification": "probstats.bayes.certification",
    "comparison": "probstats.bayes.comparison",
    "conjugacy": "probstats.bayes.conjugacy",
    "diagnostics": "probstats.bayes.diagnostics",
    "empirical_bayes": "probstats.bayes.empirical_bayes",
    "exact": "probstats.bayes.exact",
    "gp": "probstats.bayes.gp",
    "laplace": "probstats.bayes.laplace",
    "mcmc": "probstats.bayes.mcmc",
    "models": "probstats.bayes.models",
    "nested": "probstats.bayes.nested",
    "neural": "probstats.bayes.neural",
    "planner": "probstats.bayes.planner",
    "predictive": "probstats.bayes.predictive",
    "reasoning": "probstats.bayes.reasoning",
}

_OWNERS = {
    "BayesianLinearRegression": "models",
    "BayesianLinearRegressionFit": "models",
    "BayesianModelComparisonResult": "comparison",
    "BayesianMultivariateLinearRegression": "models",
    "BayesianMultivariateLinearRegressionFit": "models",
    "ConjugateUpdateResult": "conjugacy",
    "Factor": "core",
    "GaussianProcessFit": "gp",
    "GaussianProcessRegressor": "gp",
    "InferenceKind": "core",
    "InferencePlan": "planner",
    "InferenceResult": "core",
    "Model": "core",
    "NormalInverseGamma": "conjugacy",
    "NormalInverseWishart": "conjugacy",
    "Observation": "core",
    "Parameter": "core",
    "PosteriorData": "diagnostics",
    "RBFKernel": "gp",
    "RandomVariable": "core",
    "compare_models": "comparison",
    "conjugate_update": "conjugacy",
    "infer": "planner",
    "infer_conjugate": "conjugacy",
    "plan_inference": "planner",
    "posterior_predictive": "predictive",
    "predict": "predictive",
    "predict_distribution": "predictive",
    "predictive_interval": "predictive",
}

__all__ = [
    "BayesianLinearRegression",
    "BayesianLinearRegressionFit",
    "BayesianModelComparisonResult",
    "BayesianMultivariateLinearRegression",
    "BayesianMultivariateLinearRegressionFit",
    "ConjugateUpdateResult",
    "Factor",
    "GaussianProcessFit",
    "GaussianProcessRegressor",
    "InferenceKind",
    "InferencePlan",
    "InferenceResult",
    "Model",
    "NormalInverseGamma",
    "NormalInverseWishart",
    "Observation",
    "Parameter",
    "PosteriorData",
    "RBFKernel",
    "RandomVariable",
    "certification",
    "compare_models",
    "comparison",
    "conjugacy",
    "conjugate_update",
    "diagnostics",
    "empirical_bayes",
    "exact",
    "gp",
    "infer",
    "infer_conjugate",
    "laplace",
    "mcmc",
    "models",
    "nested",
    "neural",
    "plan_inference",
    "planner",
    "posterior_predictive",
    "predict",
    "predict_distribution",
    "predictive",
    "predictive_interval",
    "reasoning",
]


def __getattr__(name):
    """Load one documented facade object from its owning module."""
    module_path = _MODULES.get(name)
    if module_path is not None:
        value = _import_module(module_path)
        globals()[name] = value
        return value
    owner = _OWNERS.get(name)
    if owner is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = _import_module(f"probstats.bayes.{owner}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    """Return the supported Bayesian discovery surface."""
    return sorted(__all__)
