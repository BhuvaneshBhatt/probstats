# Public API

`probstats` uses a small package root and explicit domain namespaces. The goal is for a name's import path to communicate what kind of operation it performs and to prevent unrelated statistical concepts from colliding in one flat namespace.

## Find an API by task

| Task | Recommended entry point |
| --- | --- |
| Construct common distributions | `probstats.Normal`, `probstats.Poisson`, or `probstats.distributions` |
| Probability, expectation, CDF, density | package root or `probstats.functionals` |
| Mean, median, quantile, variance, standard deviation | package root; dispatches for data and distributions |
| Descriptive statistics, estimation, tests, regression, GLMs | `probstats.stats` |
| Smoothing and rolling/running statistics | `probstats.smoothing` |
| Survival estimation and tests | `probstats.survival` |
| Entropy and divergences | `probstats.information` |
| Random-matrix models and spectral statistics | `probstats.random_matrix` |
| Symbolic transforms and distribution algebra | `probstats.symbolic` |
| Algebraic statistical models and probability tensors | `probstats.algebraic` |
| Bayesian inference and prediction | `probstats.bayes` |
| Event and parameter spaces | `probstats.spaces` |

## Package root

The root is for core probability objects and the main discovery namespaces:

```python
import probstats
from probstats import Normal, Poisson, expectation, probability, sample

probstats.algebraic
probstats.stats
probstats.smoothing
probstats.survival
probstats.information
probstats.random_matrix
probstats.bayes
```

Core distribution constructors are available both at the root and under `probstats.distributions`. Specialized distributions should be imported from `probstats.distributions`.

The root does not flatten hypothesis tests, regression procedures, smoothing routines, survival estimators, information measures, random-matrix tools, or Bayesian implementation types. For example:

```python
from probstats.stats import linear_regression, welch_t_test
from probstats.smoothing import lowess, rolling_mean
from probstats.survival import kaplan_meier, logrank_test
from probstats.information import kl_divergence
from probstats.random_matrix import GOE, TracyWidom
from probstats.bayes import BayesianLinearRegression
```

## Generic statistics for data and distributions

A small set of root-level statistics answer the same mathematical question for observed data and for probability distributions. They dispatch on the first argument rather than requiring separate names:

```python
from probstats import Normal, mean, median, quantile, standard_deviation, variance

data = [1, 2, 3, 4]
normal = Normal(0, 1)

mean(data)
mean(normal)
quantile(data, 0.5)
quantile(normal, 0.95)
variance(data)
standard_deviation(normal)
```

Sample-specific options remain sample-specific. For example, ``quantile(data, q, method=...)`` controls empirical interpolation, while distribution quantiles use the distribution definition and reject non-default interpolation methods. Likewise, data variance defaults to ``ddof=1`` while a distribution variance is a population quantity.

Distribution methods remain available when they read more naturally:

```python
normal.cdf(0)
normal.quantile(0.95)
```

This dispatch rule is narrow: a root generic exists only when the operations answer essentially the same mathematical question and the first argument identifies the domain unambiguously. Regression, hypothesis tests, smoothing, survival analysis, information measures, and Bayesian procedures remain in their owning namespaces.

## Namespace ownership

- `probstats.distributions`: probability distributions and distribution families.
- `probstats.functionals`: generic probability functionals such as density, CDF, moments, hazard, and likelihood values.
- `probstats.stats`: descriptive statistics, estimation, tests, regression, GLMs, ANOVA, and related inferential tools.
- `probstats.smoothing`: weighted/binned data, rolling and running statistics, LOWESS, and local smoothing.
- `probstats.survival`: nonparametric survival estimation and survival tests. Regression models live in `probstats.survival_regression`.
- `probstats.information`: entropy, divergences, and related information measures.
- `probstats.random_matrix`: random-matrix ensembles and spectral statistics.
- `probstats.symbolic`: symbolic transforms, transform inversion, order statistics, and distribution-of-expression tools.
- `probstats.spaces`: event and parameter-space objects.
- `probstats.algebraic`: polynomial probability models, contingency tables, probability tensors, independence models, and exact model invariants.
- `probstats.bayes`: Bayesian models, inference engines, diagnostics, evidence, prediction, and model comparison.

Supporting implementation modules are not part of the root API. Advanced users can import a documented submodule explicitly when needed, but importing `probstats` exposes only the names listed above.

## Curated subnamespace discovery

`probstats.stats` and `probstats.bayes` use the same discovery discipline as the package root. Their `__all__` values and `dir(...)` results define user-facing facades; implementation classes and backends belong to their owning submodules.

For statistics, the owning layers are also directly discoverable as `probstats.stats.descriptive`, `.estimation`, `.testing`, `.inference`, and `.modeling`. For Bayesian inference, method-specific APIs live under `probstats.bayes.mcmc`, `.laplace`, `.exact`, `.nested`, `.diagnostics`, `.conjugacy`, `.certification`, `.reasoning`, `.empirical_bayes`, and `.neural`.

Ordinary probability distributions are absent from `probstats.bayes.__all__`; import them from `probstats` or `probstats.distributions`.

See the [generated API reference](api-reference.md), [distribution capability matrix](distribution-capabilities.md), [optional dependency guide](optional-dependencies.md), and [testing strategy](testing.md).

## Executable smoke example

```python
# probstats: execute
from probstats import Normal, mean, quantile

assert mean([1, 2, 3]) == 2.0
assert mean(Normal(0, 1)) == 0
assert quantile(Normal(0, 1), 0.5) == 0
```

## Stable facades and internal ownership

Large workflow namespaces remain stable public facades while implementation modules are split by responsibility. In particular, `probstats.inference`, `probstats.testing`, and `probstats.data` contain public re-exports rather than mixed implementation bodies. This preserves import paths while allowing internal modules to evolve independently without adding runtime dispatch layers.
