import inspect
import json
import subprocess
import sys
import types

import probstats

EXPECTED_ROOT_API = {
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
    "u_statistics",
    "taylor_expectation",
    "taylor_variance",
    "total_covariance",
    "total_expectation",
    "total_probability",
    "total_variance",
    "uncorrelated",
    "union_probability",
    "variance",
}


def test_root_api_is_exact_and_discoverable():
    assert set(probstats.__all__) == EXPECTED_ROOT_API
    assert set(dir(probstats)) == EXPECTED_ROOT_API


def test_fresh_root_import_has_no_accidental_public_attributes():
    code = """
import json
import probstats
print(json.dumps(sorted(name for name in vars(probstats) if not name.startswith('_'))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = set(json.loads(result.stdout))
    expected = EXPECTED_ROOT_API - {"__version__"}
    assert actual == expected


def test_root_does_not_flatten_specialized_domains():
    for name in (
        "TracyWidom",
        "kaplan_meier",
        "linear_regression",
        "lowess",
        "kl_divergence",
        "cox_ph",
        "ConjugacySignature",
        "OptimizationResult",
    ):
        assert name not in probstats.__all__
        assert name not in dir(probstats)


def test_major_domains_are_namespaces_not_colliding_functions():
    for name in (
        "algebraic",
        "bayes",
        "distributions",
        "functionals",
        "information",
        "random_matrix",
        "smoothing",
        "spaces",
        "stats",
        "survival",
        "symbolic",
        "u_statistics",
    ):
        assert isinstance(getattr(probstats, name), types.ModuleType)


def test_root_distribution_constructors_match_distribution_namespace():
    for name in (
        "Bernoulli",
        "Beta",
        "Binomial",
        "Distribution",
        "Exponential",
        "Gamma",
        "MultivariateNormal",
        "Normal",
        "Poisson",
        "StudentT",
        "Uniform",
    ):
        assert getattr(probstats, name) is getattr(probstats.distributions, name)


def test_survival_name_is_unambiguously_the_analysis_namespace():
    assert probstats.survival is not probstats.functionals.survival
    assert callable(probstats.functionals.survival)


def test_root_public_objects_have_documentation():
    for name in probstats.__all__:
        value = getattr(probstats, name)
        if (
            inspect.isclass(value)
            or inspect.isfunction(value)
            or inspect.ismodule(value)
        ):
            assert inspect.getdoc(value), f"missing public documentation for {name}"


def test_root_generic_statistics_delegate_to_owning_apis():
    data = [1.0, 2.0, 4.0, 8.0]
    dist = probstats.Normal(2, 3)

    assert probstats.mean(data) == probstats.stats.mean(data)
    assert probstats.median(data) == probstats.stats.median(data)
    assert probstats.quantile(data, 0.25) == probstats.stats.quantile(data, 0.25)
    assert probstats.variance(data) == probstats.stats.variance(data)
    assert probstats.standard_deviation(data) == probstats.stats.standard_deviation(
        data
    )

    assert probstats.mean(dist) == probstats.functionals.mean(dist)
    assert probstats.median(dist) == probstats.functionals.quantile(dist, 0.5)
    assert probstats.quantile(dist, 0.25) == probstats.functionals.quantile(dist, 0.25)
    assert probstats.variance(dist) == probstats.functionals.variance(dist)
    assert probstats.standard_deviation(dist) ** 2 == probstats.variance(dist)


EXPECTED_STATS_API = set(probstats.stats.__all__)
EXPECTED_BAYES_API = set(probstats.bayes.__all__)


def test_domain_discovery_contracts_are_curated():
    assert len(EXPECTED_STATS_API) <= 70
    assert len(EXPECTED_BAYES_API) <= 50
    assert set(dir(probstats.stats)) == EXPECTED_STATS_API
    assert set(dir(probstats.bayes)) == EXPECTED_BAYES_API


def test_bayes_does_not_advertise_probability_distributions_or_backends():
    for name in (
        "Normal",
        "Beta",
        "Gamma",
        "Poisson",
        "Wishart",
        "Dirichlet",
        "DEFAULT_REGISTRY",
        "OptimizationBackend",
        "MultipleIntegrateBackend",
        "SciPyOptimizationBackend",
        "MCMCState",
    ):
        assert name not in probstats.bayes.__all__
        assert name not in dir(probstats.bayes)
    assert not hasattr(probstats.bayes, "Normal")


def test_fresh_bayes_import_does_not_eagerly_load_method_packages():
    code = """
import json
import sys
import probstats.bayes
print(json.dumps(sorted(name for name in sys.modules if name.startswith('probstats.bayes.'))))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []


def test_bayes_exposes_method_namespaces():
    for name in (
        "mcmc",
        "laplace",
        "exact",
        "nested",
        "diagnostics",
        "conjugacy",
        "certification",
        "reasoning",
        "empirical_bayes",
        "neural",
    ):
        assert isinstance(getattr(probstats.bayes, name), types.ModuleType)
        assert name in probstats.bayes.__all__


def test_stats_exposes_layer_namespaces():
    for name in ("descriptive", "estimation", "testing", "inference", "modeling"):
        assert isinstance(getattr(probstats.stats, name), types.ModuleType)
        assert name in probstats.stats.__all__


# Every root export has one explicit testing contract. ``behavior`` entries are
# exercised through the root; ``identity`` entries are constructors/result
# types whose owning-module behavior is tested elsewhere; namespaces have
# discovery contracts above.
ROOT_NAMESPACES = {
    "algebraic",
    "bayes",
    "distributions",
    "functionals",
    "information",
    "random_matrix",
    "smoothing",
    "spaces",
    "stats",
    "survival",
    "symbolic",
    "u_statistics",
}
ROOT_IDENTITY_CONTRACTS = {
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
}
ROOT_BEHAVIOR_CONTRACTS = EXPECTED_ROOT_API - ROOT_NAMESPACES - ROOT_IDENTITY_CONTRACTS


def test_every_root_export_has_an_explicit_contract_category():
    categories = ROOT_NAMESPACES | ROOT_IDENTITY_CONTRACTS | ROOT_BEHAVIOR_CONTRACTS
    assert categories == EXPECTED_ROOT_API
    assert not (ROOT_NAMESPACES & ROOT_IDENTITY_CONTRACTS)
    assert not (ROOT_NAMESPACES & ROOT_BEHAVIOR_CONTRACTS)
    assert not (ROOT_IDENTITY_CONTRACTS & ROOT_BEHAVIOR_CONTRACTS)


def test_root_result_types_are_produced_by_root_approximation_apis():
    x = probstats.RandomVariable("RootApproxX", distribution=probstats.Normal(0, 1))
    taylor = probstats.taylor_expectation(x**2, return_result=True)
    delta = probstats.delta_method(x**2, variable=x)
    assert isinstance(taylor, probstats.TaylorApproximationResult)
    assert isinstance(delta, probstats.DeltaMethodResult)


def test_root_finiteness_helpers_preserve_three_valued_semantics():
    known = probstats.RandomVariable("Known", distribution=probstats.Normal(0, 1))
    unknown = probstats.RandomVariable("Unknown")
    context = probstats.StatisticalAssumptions()
    assert probstats.finite_mean(known, assumptions=context) is True
    assert probstats.finite_variance(known, assumptions=context) is True
    assert probstats.finite_mean(unknown, assumptions=context) is None
    assert probstats.finite_variance(unknown, assumptions=context) is None


def test_root_union_probability_computes_union_event():
    import sympy as sp

    x = sp.symbols("x", real=True)
    value = probstats.union_probability(
        probstats.Uniform(0, 1),
        x < sp.Rational(1, 4),
        x > sp.Rational(3, 4),
        variable=x,
    )
    assert value == sp.Rational(1, 2)


def test_root_information_helpers_have_behavioral_contracts():
    import sympy as sp

    x = probstats.RandomVariable(
        "EntropyX", distribution=probstats.Bernoulli(sp.Rational(1, 2))
    )
    y = probstats.RandomVariable("EntropyY")
    z = probstats.RandomVariable("EntropyZ")
    assert probstats.random_entropy(x) == sp.log(2)
    from probstats.random_variable_information import ConditionalMutualInformation

    chain = probstats.mutual_information_chain_rule(x, y, z)
    assert chain.has(ConditionalMutualInformation)
    assert (
        probstats.product_kl_divergence(
            [(probstats.Normal(0, 1), probstats.Normal(0, 1))]
        )
        == 0
    )


def test_root_behavior_exports_have_direct_call_evidence_in_tests():
    import ast
    from pathlib import Path

    called = set()
    for path in Path(__file__).parent.glob("test_*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "probstats":
                for alias in node.names:
                    imported[alias.asname or alias.name] = alias.name
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id in imported:
                called.add(imported[node.func.id])
            elif (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "probstats"
            ):
                called.add(node.func.attr)
    # __version__ is data rather than a callable. All other behavior-contract
    # exports must be invoked directly somewhere in the root test suite.
    assert (ROOT_BEHAVIOR_CONTRACTS - {"__version__"}) <= called
