import inspect

import probstats

ASSUMPTION_AWARE_ROOT_FUNCTIONS = {
    "characteristic_function",
    "coefficient_of_variation",
    "complex_covariance",
    "complex_variance",
    "conditional_covariance",
    "conditional_entropy",
    "conditional_expectation",
    "conditional_moment",
    "conditional_variance",
    "conditionally_independent",
    "conditionally_independent_collections",
    "correlation",
    "cumulant_generating_function",
    "delta_method",
    "delta_variance",
    "distribution",
    "factorial_moment",
    "finite_mean",
    "finite_moment",
    "finite_variance",
    "identically_distributed",
    "iid",
    "independent",
    "independent_collections",
    "joint_entropy",
    "kurtosis",
    "mixed_moment",
    "moment_generating_function",
    "mutual_information",
    "mutual_information_chain_rule",
    "mutually_independent",
    "pairwise_independent",
    "probability_generating_function",
    "pseudo_covariance",
    "random_entropy",
    "skewness",
    "standardized_moment",
    "taylor_expectation",
    "taylor_variance",
    "total_covariance",
    "total_expectation",
    "total_variance",
    "uncorrelated",
}


def test_random_variable_assumptions_keyword():
    for name in ASSUMPTION_AWARE_ROOT_FUNCTIONS:
        signature = inspect.signature(getattr(probstats, name))
        parameter = signature.parameters.get("assumptions")
        assert parameter is not None, name
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, name
        assert parameter.default is None, name


def test_delta_method_variance_contracts():
    delta_method = inspect.signature(probstats.delta_method)
    delta_variance = inspect.signature(probstats.delta_variance)
    assert "return_result" not in delta_method.parameters
    assert "center" in delta_method.parameters
    assert "asymptotic_variance" in delta_method.parameters
    assert "center" not in delta_variance.parameters


def test_no_low_level_transformation_theorem_helper_is_root_exported():
    forbidden = {
        "bijective_mutual_information",
        "certify_scalar_transformation",
        "certify_common_pushforward",
        "transformed_entropy_value",
    }
    assert forbidden.isdisjoint(probstats.__all__)
