"""Probability and statistics with exact and numerical computation.

The package root exposes a small set of unambiguous probability primitives and
major domain namespaces. Specialized statistical procedures live in their
own namespaces, for example ``probstats.stats`` and ``probstats.survival``.
"""

from . import (
    algebraic,
    bayes,
    distributions,
    functionals,
    information,
    random_matrix,
    smoothing,
    spaces,
    stats,
    survival,
    symbolic,
    u_statistics,
)
from ._generic import covariance, mean, median, quantile, standard_deviation, variance
from ._version import __version__
from .assumptions import (
    StatisticalAssumptions,
    conditionally_independent,
    conditionally_independent_collections,
    finite_mean,
    finite_moment,
    finite_variance,
    identically_distributed,
    iid,
    independent,
    independent_collections,
    mutually_independent,
    pairwise_independent,
    uncorrelated,
)
from .distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Distribution,
    Exponential,
    Gamma,
    MultivariateNormal,
    Normal,
    Poisson,
    StudentT,
    Uniform,
)
from .events import event_complement, event_intersection, event_union
from .functionals import (
    cdf,
    central_moment,
    cumulant,
    density,
    expectation,
    moment,
    probability,
    raw_moment,
)
from .joint import JointDistribution
from .probability_algebra import (
    bayes_probability,
    conditional_probability,
    events_independent,
    inclusion_exclusion,
    total_probability,
    union_probability,
)
from .random_variable_algebra import mixed_moment
from .random_variable_approximations import (
    DeltaMethodResult,
    TaylorApproximationResult,
    delta_method,
    delta_variance,
    taylor_expectation,
    taylor_variance,
)
from .random_variable_complex import (
    complex_covariance,
    complex_variance,
    pseudo_covariance,
)
from .random_variable_conditional import (
    conditional_covariance,
    conditional_expectation,
    conditional_moment,
    conditional_variance,
    total_covariance,
    total_expectation,
    total_variance,
)
from .random_variable_distributions import (
    RandomExpressionLaw,
    distribution,
    register_affine_closure,
    register_sum_closure,
)
from .random_variable_information import (
    conditional_entropy,
    entropy_chain_rule,
    joint_entropy,
    mutual_information,
    mutual_information_chain_rule,
    product_kl_divergence,
    random_entropy,
)
from .random_variable_statistics import (
    coefficient_of_variation,
    correlation,
    factorial_moment,
    kurtosis,
    skewness,
    standardized_moment,
)
from .random_variable_transforms import (
    characteristic_function,
    cumulant_generating_function,
    moment_generating_function,
    probability_generating_function,
)
from .random_variables import RandomVariable, depends_on, random_variables
from .sampling import sample

__all__ = [
    "Bernoulli",
    "Beta",
    "Binomial",
    "DeltaMethodResult",
    "Distribution",
    "Exponential",
    "Gamma",
    "JointDistribution",
    "MultivariateNormal",
    "Normal",
    "Poisson",
    "RandomExpressionLaw",
    "RandomVariable",
    "StatisticalAssumptions",
    "StudentT",
    "TaylorApproximationResult",
    "Uniform",
    "__version__",
    "algebraic",
    "bayes",
    "bayes_probability",
    "cdf",
    "central_moment",
    "characteristic_function",
    "coefficient_of_variation",
    "complex_covariance",
    "complex_variance",
    "conditional_covariance",
    "conditional_entropy",
    "conditional_expectation",
    "conditional_moment",
    "conditional_probability",
    "conditional_variance",
    "conditionally_independent",
    "conditionally_independent_collections",
    "correlation",
    "covariance",
    "cumulant",
    "cumulant_generating_function",
    "delta_method",
    "delta_variance",
    "density",
    "depends_on",
    "distribution",
    "distributions",
    "entropy_chain_rule",
    "event_complement",
    "event_intersection",
    "event_union",
    "events_independent",
    "expectation",
    "factorial_moment",
    "finite_mean",
    "finite_moment",
    "finite_variance",
    "functionals",
    "identically_distributed",
    "iid",
    "inclusion_exclusion",
    "independent",
    "independent_collections",
    "information",
    "joint_entropy",
    "kurtosis",
    "mean",
    "median",
    "mixed_moment",
    "moment",
    "moment_generating_function",
    "mutual_information",
    "mutual_information_chain_rule",
    "mutually_independent",
    "pairwise_independent",
    "probability",
    "probability_generating_function",
    "product_kl_divergence",
    "pseudo_covariance",
    "quantile",
    "random_entropy",
    "random_matrix",
    "random_variables",
    "raw_moment",
    "register_affine_closure",
    "register_sum_closure",
    "sample",
    "skewness",
    "smoothing",
    "spaces",
    "standard_deviation",
    "standardized_moment",
    "stats",
    "survival",
    "symbolic",
    "taylor_expectation",
    "taylor_variance",
    "total_covariance",
    "total_expectation",
    "total_probability",
    "total_variance",
    "u_statistics",
    "uncorrelated",
    "union_probability",
    "variance",
]

# Importing objects from these implementation modules makes Python attach the
# modules to the package object. Remove that known implementation surface
# explicitly rather than deleting arbitrary future globals by convention.
_HIDDEN_ROOT_IMPORTS = (
    "algebra",
    "composition",
    "data",
    "descriptive",
    "estimation",
    "inference",
    "information_registry",
    "modeling",
    "results",
    "solver",
    "testing",
    "transforms",
    "assumptions",
    "events",
    "joint",
    "probability_algebra",
    "random_variable_algebra",
    "random_variable_approximations",
    "random_variable_complex",
    "random_variable_conditional",
    "random_variable_distributions",
    "random_variable_information",
    "random_variable_statistics",
    "random_variable_transforms",
    "sampling",
)
for _name in _HIDDEN_ROOT_IMPORTS:
    globals().pop(_name, None)


def __dir__():
    """Return the compact public discovery surface."""
    return sorted(__all__)
