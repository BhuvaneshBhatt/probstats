# Bayesian inference method guide

`probstats.bayes` separates inference engines from the model representation. The automatic
planner is a policy layer: every engine can also be invoked explicitly.

| Method | Entry point | Result | Best suited to | Principal requirements |
| --- | --- | --- | --- | --- |
| Conjugate | `infer_conjugate` | exact | registered exponential-family pairs | likelihood, observations, compatible prior |
| Direct symbolic | `infer_exact` | exact | low-dimensional symbolic models | integrable/summable declared supports |
| Analytic linear regression | `BayesianLinearRegression.fit` | exact | univariate-response Gaussian linear regression | Normal-Inverse-Gamma model |
| Analytic multivariate linear regression | `BayesianMultivariateLinearRegression.fit` | exact | multivariate-response Gaussian linear regression | Matrix-Normal/Inverse-Wishart model |
| Analytic GP | `GaussianProcessRegressor.fit` | exact conditional Gaussian algebra | fixed GP hyperparameters | positive-definite covariance system |
| Laplace | `infer_laplace` | approximate, or exact for quadratic log densities | smooth continuous unimodal posteriors | mode + nonsingular negative Hessian; special singular support exists |
| Nested sampling | `infer_nested` / `infer_nested_model` | sampled | global evidence and nonconjugate posteriors | prior sampler/transform and numerical log likelihood |

## Planner policy

`InferencePlanner` estimates applicability and relative cost; the scores are policy values,
not runtime predictions. For a generic `Model`, the planner considers registered
conjugacy (when likelihood/data/prior context is supplied), direct exact normalization,
Laplace approximation, and nested sampling. Specialized Bayesian linear-regression and
Gaussian-process objects receive their analytic solvers directly.

The main exact-integration guards are `PlannerConfig.max_exact_latents` and
`PlannerConfig.max_exact_operations`. `allow_laplace` and `allow_nested` control numerical
fallbacks. Use `plan_inference(subject, **context).explain()` to inspect accepted and
rejected candidates before execution.

## Conjugate inference

Use conjugacy when the likelihood/prior family has a registered closed-form update. This
is normally the cheapest and strongest result: posterior hyperparameters, evidence, and
available predictive distributions are algebraic expressions rather than numerical fits.
See [Conjugacy reference](conjugacy-reference.md).

## Direct exact inference

`infer_exact` constructs the observed joint density, integrates or sums out latent
variables over declared supports, validates the normalizer, and returns a normalized
`SymbolicJointDistribution`. With `backend="auto"`, optional structured integration can be
tried before the SymPy fallback. A supplied conjugacy context may be used as a fallback if
direct integration fails.

Exactness here refers to the symbolic mathematical operation. If the backend cannot prove
a required integral or positivity condition, the engine fails rather than silently
turning an unresolved expression into a numerical answer.

## Laplace inference

Laplace inference finds a posterior mode and locally approximates

\[
\log p(\theta\mid y)
\]

by a quadratic form. Backends include symbolic optimization and optional SciPy/`symbopt`
integrations. The method is especially useful when exact integration is too costly but the
posterior is smooth and concentrated. Singular even-order local decay can use the
singular-Laplace path; genuinely coupled multivariate degenerate saddles remain more
restricted than ordinary nonsingular Laplace problems.

## Nested sampling

Nested sampling is a global numerical evidence method. The preferred interface is
`prior_transform(u)`, which maps the unit cube to the normalized prior and enables the
slice constrained sampler. A `prior_sampler` is also supported and uses the
correctness-oriented rejection constrained sampler by default. Results include a weighted
`EmpiricalPosterior`, log evidence, and evidence uncertainty diagnostics.

## Optional dependencies

The core model, distributions, conjugacy, and SymPy exact route use the base package.
Optional extras provide SciPy Laplace optimization (`laplace`), `symbopt`, higher-order
asymptotics (`asymptotic`), structured exact/certification backends (`exact`, `certify`,
`reasoning`), ArviZ conversion (`arviz`), and JAX neural regression (`jax`).

## Decision rule

Prefer, in order, a mathematically applicable analytic/conjugate method, then tractable
exact symbolic normalization, then a validated approximation appropriate to the posterior
geometry. Nested sampling is valuable when global evidence matters and local Gaussian
structure is insufficient. `method="auto"` embodies this policy but `plan_inference` should
be inspected for expensive or unusual models.


## MCMC

General MCMC is the posterior-sampling fallback for continuous models when exact/conjugate elimination is unavailable or when posterior draws are explicitly required. Supply `initial_positions` to automatic planning. The model adapter prefers NUTS when a symbolic gradient can be generated and otherwise uses Adaptive Metropolis. Warmup adaptation is frozen before retained samples. See [General MCMC framework](mcmc.md) for sampler and diagnostic details.

## Method-selection matrix

| Method | Exactness | Differentiability | Multimodality | Typical dimension | Evidence | Posterior draws | Key optional dependencies | Main diagnostics/failure modes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Conjugate | exact | not required | family-fixed | low to moderate | exact when family supplies it | usually direct sampling from posterior law | none | incompatible likelihood/prior family |
| Direct symbolic | exact if completed | not required | can represent multiple symbolic branches | usually low | exact normalizer | from resulting law when sampleable | structured exact extras optional | unresolved integral/sum, positivity/support proof |
| Analytic regression / GP | exact conditional algebra for the declared model | not required | model-specific unimodal Gaussian-family structure | moderate | model-dependent | yes | linear algebra stack only | singular/ill-conditioned covariance or invalid model assumptions |
| Laplace | local asymptotic approximation | smooth derivatives normally required | poor for separated modes | low to moderate | approximate | approximate Gaussian/local law | `laplace`, optional `symbopt`/`asymptotic` | missing/invalid mode, singular Hessian, boundary/nonregular geometry |
| MCMC (NUTS) | asymptotically exact sampling | gradient required | can struggle between separated modes | moderate to high | no direct evidence | yes | numerical stack | divergences, low ESS, poor R-hat/mixing |
| MCMC (Adaptive Metropolis) | asymptotically exact sampling | not required | can struggle between separated modes | low to moderate | no direct evidence | yes | numerical stack | slow mixing, adaptation/scale problems |
| Nested sampling | numerical global integration | not required | comparatively suitable | low to moderate | primary output | weighted posterior draws | none beyond numerical base for core path | constrained-sampling inefficiency, large evidence uncertainty |
| Neural/JAX utilities | approximate/model-specific | automatic differentiation | model-dependent | moderate to high | generally not primary | model-dependent | `jax` | optimizer/training instability, approximation error |

The matrix is a method-selection aid, not a ranking. The planner should reject inapplicable methods before comparing cost. When evidence is scientifically central, a method that directly estimates the normalizing constant can be preferable to a sampler optimized only for posterior expectations.

## Bayesian assumptions and failure modes

| Workflow | Assumptions to check | Returned evidence/diagnostics | Important failure modes |
| --- | --- | --- | --- |
| Conjugate update | registered likelihood/prior parameterization and support compatibility | posterior hyperparameters, often marginal likelihood/predictive law | incompatible family, invalid hyperparameters, inconsistent observations |
| Exact symbolic inference | declared supports and a normalizable joint density/mass | normalized symbolic posterior plus backend trace where available | unevaluated integral/sum, unproved finite/positive normalizer, unavailable exact backend |
| Laplace | sufficiently regular local posterior geometry near a relevant mode | mode, Hessian/local covariance, approximation metadata | nonstationary/boundary mode, indefinite or singular curvature, multimodality |
| MCMC | stationary target, appropriate transition kernel, adequate warmup/run length | chains plus convergence/effective-sample diagnostics | divergences, nonmixing, low ESS, unstable adaptation |
| Nested sampling | normalized prior transform/sampler and numerically evaluable likelihood | log evidence, uncertainty, weighted empirical posterior | poor constrained sampling, insufficient live points, large evidence uncertainty |
