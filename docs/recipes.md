# Task-oriented recipes

These recipes are short and executable. They show the preferred user-facing path rather than backend internals.

## Choose and evaluate a distribution

```python
# probstats: execute
from probstats import Normal, cdf, mean, quantile, variance

dist = Normal(1, 2)
assert mean(dist) == 1
assert variance(dist) == 4
assert float(cdf(dist, 1)) == 0.5
assert abs(float(quantile(dist, 0.5)) - 1.0) < 1e-12
```

Use [Distribution capabilities](distribution-capabilities.md) to distinguish native symbolic CDF/quantile support from generic fallbacks.

## Summarize observed data

```python
# probstats: execute
from probstats import mean, median, quantile, variance

x = [1, 2, 3, 4]
assert mean(x) == 2.5
assert median(x) == 2.5
assert quantile(x, 0.5) == 2.5
assert variance(x) > 0
```

Sample-only options such as `ddof` and empirical interpolation methods belong to observed-data calls and are rejected for theoretical distributions when they would be meaningless.

## Fit and compare likelihood models

Use `probstats.stats.maximum_likelihood` for supported closed-form family fits and `probstats.stats.inference.numerical_mle` for general scalar likelihoods. Inspect convergence, the objective value, information/covariance diagnostics, and parameter constraints rather than treating the estimate alone as the result.

## Perform a hypothesis test

Choose a test from [Statistics](statistics.md), verify the scientific assumptions before calling it, then interpret the returned statistic, p-value, effect estimate, and uncertainty together. Do not use `reject_05` as a substitute for a predeclared significance level or a practical-effect threshold.

## Exact probability versus numerical fallback

```python
# probstats: execute
import sympy as sp
from probstats import Normal, probability

x = sp.Symbol("x", real=True)
dist = Normal(0, 1)
value = probability(dist, x > 0, variable=x)
assert sp.simplify(value - sp.Rational(1, 2)) == 0
```

The probability layer attempts exact symbolic evaluation first. Numerical or Monte Carlo fallback is opt-in where the API exposes it; unresolved exact work should not silently become a numerical claim. See [Failure semantics](failure-semantics.md).

## Build a Bayesian model and choose inference

Start with `probstats.bayes.plan_inference(...)` when more than one engine may apply. Prefer conjugacy or analytic model-specific solvers, then tractable exact normalization, then a geometry-appropriate approximation or sampler. See the [Bayesian method-selection matrix](bayesian-methods.md#method-selection-matrix).

## Diagnose posterior samples

For MCMC, inspect convergence and effective-sample diagnostics before posterior summaries. For nested sampling, inspect evidence uncertainty and weighted-posterior quality. Posterior conversion/diagnostic failures are distinct from model-validation and inference-engine failures; see [Failure semantics](failure-semantics.md).
