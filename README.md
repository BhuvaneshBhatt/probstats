# probstats

`probstats` is a probability and statistics package for Python. It combines a distribution algebra, structured event and parameter spaces, compositional probability models, numerical sampling, classical inference, Bayesian inference, algebraic statistics (essentially finite discrete algebraic statistical models plus general empirical algebraic moment/tensor statistics), survival analysis, random-matrix statistics, and data-analysis utilities behind an API built on SymPy and NumPy.

The package gives exact symbolic results when available. Numerical fallbacks are explicit, sampling uses a NumPy `Generator` or seed, and inferential routines return structured result objects with method and uncertainty information.

## Why use probstats?

Use `probstats` when a statistical workflow needs symbolic structure and numerical computation to agree instead of living in separate APIs. Distribution support and parameter constraints remain explicit, exact identities can flow into inference, and numerical methods expose their provenance. The package is especially useful for work that mixes probability algebra, classical or Bayesian inference, algebraic statistics, and reproducible simulation.

`probstats` favors mathematical semantics, inspectable result objects, and exact reasoning where those properties matter. For large-scale numerical estimation, simulation, or model fitting where symbolic structure is unnecessary, mature numerical systems such as SciPy, statsmodels, PyMC, Stan, or JAX-based libraries may provide substantially greater performance or more specialized algorithms.

If you're new to the package, start with the [workflow guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/workflows.md), which maps common statistical goals to relevant APIs and shows how calculations flow through the package. For polynomial-constraint testing, see the [worked semialgebraic example](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/algebraic-testing-worked-example.md).

## Installation

```bash
pip install probstats
```

Python 3.11 or newer is required. NumPy and SymPy are core dependencies; SciPy is optional. Optional dependency groups add capabilities:

| Installation | Capability |
| --- | --- |
| `pip install probstats` | Core symbolic probability, statistics, and modeling |
| `pip install "probstats[tabular]"` | Pandas-backed labeled inputs and result tables |
| `pip install "probstats[exact]"` | Structured exact integration and exact/certified probability workflows |
| `pip install "probstats[reasoning]"` | Semialgebraic and function-property reasoning |
| `pip install "probstats[algebraic-tensor]"` | Complete algebraic-statistics backend stack with semialg and TensorAtlas |

Other focused extras are available for certification, algebraic geometry, tensor methods, SciPy-backed Laplace routines, ArviZ, JAX, symbolic optimization, and asymptotics. See the [optional dependency guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/optional-dependencies.md) for the complete capability matrix.

## API organization, results, and labeled data

The package root is small: it contains core distribution constructors, the generic probability/evaluation functions, and major domain namespaces. Statistical procedures are owned by namespaces such as `probstats.stats`, `probstats.smoothing`, `probstats.survival`, `probstats.information`, `probstats.random_matrix`, and `probstats.bayes`.

A few mathematical summaries are root-level generics because the same operation applies naturally to either observed data or a distribution: `mean`, `median`, `quantile`, `variance`, and `standard_deviation`. The first argument selects the data or distribution name.

```python
from probstats import Normal, mean, quantile

mean([1, 2, 3, 4])
mean(Normal(0, 1))
quantile([1, 2, 3, 4], 0.5)
quantile(Normal(0, 1), 0.95)
```

Inferential result objects share `summary()`, `summary_data()`, `to_dict()`, and when Pandas is installed, `to_frame()`. Regression-style results provide coefficient tables with estimates and uncertainty information.

Pandas is an optional tabular layer. DataFrame and Series labels are captured at API boundaries, dense calculations stay in NumPy, and predictions/tables recover meaningful row and column labels. This avoids moving numerical inner loops into Pandas while retaining its indexing and presentation strengths.

```python
import pandas as pd
from probstats.stats import linear_regression

x = pd.DataFrame({"temperature": [18, 20, 23], "rain": [1, 0, 1]})
y = pd.Series([4.1, 5.2, 6.0], name="growth")
fit = linear_regression(x, y)
fit.coefficient_table()
fit.summary()
```

See the [API and tabular-data guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/results-and-tabular-data.md). See also the [public API guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/public-api.md) for namespace ownership and import conventions.

## Probability distributions

The distribution catalog includes common scalar, vector, simplex, and matrix-valued families. Scalar laws include Normal, Poisson, Gamma, Beta, Student-t, F and noncentral families, hypergeometric, Skellam, Zipf, inverse Gaussian, beta-prime, Rayleigh, Rice, Nakagami, Maxwell, Gumbel, Fréchet, Kumaraswamy, triangular, power, von Mises, and other standard families. Multivariate Normal and Student-t, Dirichlet, Wishart, inverse Wishart, LKJ, and compound count distributions are also included.

Every distribution exposes an event space and parameter space. Parameter spaces can represent scalar constraints, simplicies, ordered vectors, positive-definite matrices, correlation matrices, and cross-parameter relations.

```python
from probstats import (
    Gamma,
    Normal,
)

normal = Normal(0, 1)
gamma = Gamma(3, 2)

normal.event_space
normal.parameter_space
```

See the [distribution guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/distributions.md) for the complete catalog and parameter conventions.

## Distribution algebra and derived laws

Exact structural reasoning includes canonicalization, closed-family recognition, affine transformations, convolution, products, order statistics, transform inversion, and distribution-of-expression rules.

```python
from probstats import Poisson
from probstats.symbolic import convolution

convolution(Poisson(2), Poisson(3))
# Poisson(5)
```

`DistributionContext` and joint distributions retain dependence information. Linear images of multivariate Normal laws preserve covariance exactly; generic nonlinear maps can be retained as pushforward distributions. The algebra also includes truncation, censoring, finite mixtures, parameter mixtures, spliced laws, copulas, and user-defined symbolic distributions.

Characteristic functions, moment-generating functions, cumulant-generating functions, probability-generating functions, factorial-moment generating functions, multivariate transforms, and transform-based equivalence proofs are available for symbolic reasoning.

See the [derived-distribution guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/derived-distributions.md).

## Joint distributions and copulas

`JointDistribution` supports symbolic marginalization and conditioning for continuous and discrete joint laws. For structured continuous multiple integrals, the optional `probstats[exact]` extra uses `multiple-integrate` before falling back to SymPy. `CopulaDistribution` supports continuous, discrete, and mixed scalar marginals when the copula provides the required exact CDF operations.

Built-in copulas include independence, Gaussian, Clayton, Frank, and Gumbel families. Dependent convolutions and arbitrary scalar or vector pushforwards can be represented exactly when a closed family is unavailable.

## Probability and expectation

`probability()` and `expectation()` dispatch through distribution structure before attempting generic integration or summation. The functional API also includes survival and inverse-survival functions, continuous/discrete hazards, cumulative hazard, raw and central moments, cumulants, factorial moments, and exact IID likelihood/log-likelihood evaluation. Polynomial moments use known moment identities when possible. Monte Carlo fallback is an opt-in and can return provenance and standard-error information.

```python
import sympy as sp
from probstats import (
    Normal,
    expectation,
    probability,
)

x = sp.symbols("x", real=True)
d = Normal(0, 1)

probability(d, x > 0, variable=x)
expectation(d, x**4, variable=x)
```

See the [probability functional guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/probability.md).

## Random-variable algebra

`RandomVariable` objects participate directly in SymPy expressions. Statistical operators preserve exact symbolic structure, apply unconditional linear/bilinear rules, and use explicit `StatisticalAssumptions` only when a simplification requires a certified relationship.

```python
from probstats import (
    RandomVariable,
    StatisticalAssumptions,
    covariance,
    cumulant,
    expectation,
    independent,
    variance,
)

X = RandomVariable("X")
Y = RandomVariable("Y")
ctx = StatisticalAssumptions(independent(X, Y))

expectation(2 * X + 3)
variance(X + Y, assumptions=ctx)
covariance(X, Y, assumptions=ctx)
cumulant(X + Y, 3, assumptions=ctx)
```

Raw, central, mixed, conditional, and standardized statistics share the same symbolic semantics. Products factor only when the required `independent` relationship is known; pairwise independence is not promoted to independence from a joint conditioning set. Attached random-variable laws are authoritative for moments, cumulants, transforms, and registered distribution-closure rules. The subsystem also includes collection independence and identical-distribution assumptions, conditional covariance and total covariance, probability-event identities, entropy/mutual-information algebra, structured Taylor approximations and the delta method, and explicit Hermitian covariance/variance/pseudo-covariance for complex random variables. See the [random-variable algebra guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/random-variable-algebra.md).

## Sampling

Sampleable distributions use one NumPy-backed protocol:

```python
from probstats import Normal

Normal(0, 1).sample(size=1000, rng=123)
```

`size` describes sample dimensions only; vector and matrix event dimensions are appended automatically.

## Descriptive statistics, weighted data, and smoothing

The data-oriented API includes descriptive and robust summaries, covariance and correlation, empirical distributions, weighted observations, binning and histograms, rolling/running statistics, exponentially weighted statistics, running quantiles, kernel and local-polynomial smoothing, robust LOWESS, and sequence summaries.

```python
from probstats.smoothing import (
    WeightedData,
    local_polynomial_smooth,
    rolling_mean,
)

weighted = WeightedData([1, 2, 4], weights=[1, 2, 1])
rolling = rolling_mean([1, 2, 3, 4], window=2)
fit = local_polynomial_smooth([0, 1, 2], [1, 3, 5], bandwidth=1.0)
```

`WeightedData` integrates with descriptive functions, histograms, and KDE constructors. See the [data-analysis guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/data-analysis.md).

## Nonparametric distributions

`HistogramDistribution`, `KernelDensityDistribution`, and `MultivariateKernelDensityDistribution` participate in the same functional and sampling APIs as parametric laws. Univariate Gaussian KDE bandwidth selectors include Scott, Silverman, a Gaussian plug-in selector, and least-squares cross validation. Weighted observations are supported throughout.

```python
from probstats.distributions import (
    HistogramDistribution,
    KernelDensityDistribution,
)

kde = KernelDensityDistribution(data, bandwidth="plugin")
hist = HistogramDistribution(data, bins="fd")
```

## Estimation and statistical inference

The inference layer includes closed-form and numerical maximum likelihood, method of moments, confidence intervals, observed and expected Fisher information, profile likelihood, likelihood-ratio/Wald/score tests, bootstrap and permutation inference, exact small-sample rank tests where feasible, multiple-testing correction, goodness-of-fit testing, ANOVA, and regression inference.

Automatic numerical constraints can be derived from a distribution's `ParameterSpace` for common scalar parameter domains. See the [inference guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/inference.md) for assumptions and result-object conventions.

## Classical hypothesis tests

Classical procedures include t and z tests, exact binomial and chi-square tests, Pearson/Spearman/Kendall and distance-dependence procedures, variance-homogeneity tests, equivalence and noninferiority tests, Box-Pierce/Ljung-Box autocorrelation tests, Hotelling T-squared, Box's M, Mardia multivariate-normality diagnostics, Mann-Whitney U, Wilcoxon signed-rank, Kruskal-Wallis, Kolmogorov-Smirnov goodness-of-fit, and multiple-testing adjustments.

See the [statistics guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/statistics.md).

## Formula models, GLMs, and ANOVA

A common formula compiler powers weighted least squares, generalized linear models, ANCOVA, and multi-factor ANOVA. It retains model terms, categorical encodings, interactions, and transformed predictors so prediction and term-level inference use the same fitted design specification.

```python
from probstats.stats import (
    glm,
    linear_model,
)

ols = linear_model("y ~ log(x) + C(group)", data, weights="weight")
pois = glm(
    "count ~ treatment * C(site)",
    data,
    family="poisson",
    offset="log_exposure",
)
```

The formula language supports main effects, `:` interactions, hierarchical `*` expansion, intercept control, categorical factors, common transformations, integer powers, and raw polynomial bases. GLM families include Gaussian, binomial, Poisson, Gamma, Negative Binomial, quasi-Poisson, and quasi-binomial.

See the [modeling guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/modeling.md).

## Survival analysis and regression

Right-censored survival data, with optional delayed entry, is represented by `SurvivalData`. The survival layer includes Kaplan-Meier estimation with Greenwood uncertainty and log-minus-log confidence intervals, Nelson-Aalen cumulative hazards, risk tables, median survival, and weighted log-rank-family tests.

Regression includes Cox proportional hazards with Breslow/Efron ties, baseline hazard and survival, coefficient inference, Schoenfeld/martingale/deviance residuals, proportional-hazards diagnostics, and parametric AFT models for exponential, Weibull, log-normal, and log-logistic survival times.

See the [survival guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/survival.md).

## Information measures

`probstats` includes exact Kullback–Leibler (KL) and Renyi divergences, cross-entropy, Jensen-Shannon divergence, and Hellinger/Bhattacharyya measures with explicit support validation and optional numerical fallback.

See the [information-measures guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/information-measures.md).

## Algebraic statistics

`probstats.algebraic` provides polynomial and toric statistical models, exact contingency tables, sufficient-statistic fibers, exact conditional inference, probability tensors, complete-independence and conditional-independence models, and exact model invariants. Generic ideal geometry is delegated to the `semialg` package; generic tensor flattenings and Segre varieties are delegated to the `tensoratlas` package.

```python
from probstats.algebraic import ToricModel, conditional_test

A = ((1, 1, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0))
model = ToricModel(A)
conditional_test((1, 3, 3, 1), model).pvalue
```

Use the [algebraic-statistics reference](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/algebraic-statistics.md) for APIs, inputs, and complexity guards; the [cross-package tutorial](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/algebraic-statistics-guide.md) for worked workflows; and [certificate semantics](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/algebraic-statistics-certificates.md) for interpreting exact, complete, generic, certified, inconclusive, and numerical results.

## Bayesian inference

Bayesian inference lives under `probstats.bayes` and reuses the same probability distributions as the root package. It includes conjugate updates, exact symbolic inference, Laplace approximation, MCMC, nested sampling, Gaussian processes, Bayesian linear and multivariate-response regression, empirical-Bayes evidence optimization, diagnostics, certification/reasoning, neural regression utilities, and inference planning.

```python
from probstats import bayes

fit = bayes.BayesianLinearRegression().fit([[1], [2], [3]], [1, 2, 3])
fit.predict([[4], [5]])
fit.posterior_predictive([[4], [5]], size=1000, rng=123)
fit.predictive_interval([[4], [5]], level=0.95)
```

Exact predictive distributions returned when the fitted model has one; sampled inference uses the same predictive facade through a model-specific predicitive callback. `bayes.compare_models()` compares models with available marginal likelihoods and reports log evidence, Bayes factors, and posterior model probabilities.

See the [Bayesian subsystem guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/bayesian-inference.md), [tutorial](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/bayesian-tutorial.md), [method-selection guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/bayesian-methods.md), [conjugacy reference](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/conjugacy-reference.md), [MCMC guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/mcmc.md), [multivariate-regression guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/bayesian-multivariate-regression.md), and [empirical-Bayes guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/empirical-bayes.md).

## Random-matrix statistics

`probstats.random_matrix` separates finite matrix ensembles and spectral statistics from matrix-valued probability distributions. It includes GOE/GUE/GSE and Wishart ensembles, eigenvalue sampling, Wigner semicircle and Marchenko-Pastur laws, a pluggable Tracy-Widom interface, largest-eigenvalue scaling, and analytic or sampled trace/determinant/condition-number statistics where appropriate.

See the [random-matrix guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/random-matrix.md).

## Capability matrix

| Area | Symbolic / exact | Numerical | Main limitations |
| --- | --- | --- | --- |
| Probability distributions | Exact densities, supports, moments, transforms, structural algebra where implemented | Sampling and opt-in numerical functionals | Closed forms depend on the family and transformation; unsupported symbolic cases remain explicit |
| Classical statistics | Exact small-sample procedures where available | Regression, GLMs, resampling, nonparametric and multivariate tests | Procedures retain their usual statistical assumptions; optional SciPy paths require SciPy |
| Bayesian inference | Conjugacy and symbolic elimination for supported models | Laplace, MCMC, nested sampling, GP and neural utilities | No single inference method covers every model; diagnostics do not turn approximate inference into an exact certificate |
| Algebraic statistics | Polynomial models, fibers, invariants, certificates, semialgebraic hypotheses | Numerical latent fitting and SDL simulation/bootstrap | Some geometry and tensor operations require optional `semialg` or `tensoratlas`; combinatorial methods can grow rapidly |
| Data and smoothing | Exact descriptive operations when meaningful | NumPy-based rolling, weighted, KDE and smoothing routines | Pandas labels require the optional tabular extra; large numerical workloads are not the package's primary specialization |
| Random matrices | Exact finite-ensemble formulas where provided | Matrix/eigenvalue sampling and spectral summaries | Coverage is selective, not a general-purpose random-matrix solver |

The [distribution capability matrix](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/distribution-capabilities.md) gives method-level detail for probability families. Algebraic evidence levels and their limits are summarized in the [algebraic-statistics capability matrix](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/algebraic-statistics-certificates.md#capability-matrix).

## Limitations

`probstats` does not promise a closed form for every symbolic integral, transform, distributional expression, or posterior. An unresolved proof question is kept distinct from a false statement, and optional numerical fallback is used only where the API documents it. Exact conditional and algebraic-statistics procedures can be combinatorial, high-order kernel symmetrization can be expensive, and Monte Carlo results retain simulation error. Optional dependencies provide capabilities such as structured integration, semialgebraic reasoning, tensor geometry, SciPy-backed numerical methods, and Pandas-backed labels; their absence is reported separately from invalid mathematical input.

For operational details, see [failure semantics](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/failure-semantics.md), [optional dependencies](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/optional-dependencies.md), and [performance](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/performance.md).


## Design principles

- Exact symbolic computation when mathematically justified.
- Make numerical approximation explicit and report method/provenance where appropriate.
- Represent distribution support and parameter constraints structurally.
- Never infer independence for explicitly joint random variables.
- Raise clear exceptions for unsupported cases.
- Share one design-matrix and term representation across statistical models.
- Use explicit NumPy random-state semantics for sampling.

## Performance

Numerical hot paths use batched NumPy operations while exact polynomial-domain matrix work stays symbolic. See the [performance guide](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/performance.md) for batching, memory tradeoffs, and optional Flint-backed exact arithmetic.

## Development

For practical entry points, see the [task-oriented recipes](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/recipes.md). For exact-vs-numerical error behavior and backend distinctions, see [failure semantics](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/failure-semantics.md). The generated [distribution capability matrix](https://github.com/BhuvaneshBhatt/probstats/blob/main/docs/distribution-capabilities.md) distinguishes native symbolic methods from generic fallbacks.

```bash
python -m pytest
ruff check src tests
ruff format --check src tests
```

The test suite includes registry-driven distribution contracts, symbolic-to-numerical differential checks, randomized property tests, metamorphic identities, failure-semantics checks, model-design checks, and cross-feature regressions.

## License

`probstats` is licensed under the [GNU General Public License v3.0](https://github.com/BhuvaneshBhatt/probstats/blob/main/LICENSE).

