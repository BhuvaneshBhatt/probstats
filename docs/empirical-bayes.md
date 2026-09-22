# Empirical Bayes and evidence optimization

`probstats.bayes` provides a general type-II maximum-likelihood / empirical-Bayes
interface together with MacKay fixed-point updates for Laplace models.

## General evidence optimization

`Hyperparameter` describes a natural-scale hyperparameter and its unconstrained
optimization coordinate.  The supported transforms are `identity`, `log` for
strictly positive quantities, and bounded `logit`.

```python
from probstats.bayes.empirical_bayes import Hyperparameter, optimize_evidence

result = optimize_evidence(
    objective,
    [Hyperparameter("alpha", 1.0, transform="log")],
)
```

The objective receives a mapping of natural-scale values.  It may return a
number, `EvidenceEvaluation`, `InferenceResult`, or any fit object exposing a
`log_evidence` attribute.  This allows the same optimizer to work
with exact conjugate evidence (including multivariate MNIW regression), Laplace
evidence, or user-defined evidence approximations.

Optimization occurs in unconstrained coordinates.  `method="auto"` uses SciPy
L-BFGS-B when SciPy is installed and otherwise falls back to a deterministic,
dependency-free coordinate search.  `EvidenceOptimizationResult` records the
final parameters, evidence, evaluation history, producing fit, and—when the
local evidence curvature is positive definite—an approximate hyperparameter
covariance obtained from the negative Hessian.

## Laplace evidence

`LaplaceEvidenceObjective` rebuilds a symbolic Bayesian `Model` at each
hyperparameter setting and runs `infer_laplace`.  `optimize_laplace_evidence`
is the corresponding convenience interface:

```python
result = optimize_laplace_evidence(
    model_factory,
    [Hyperparameter("alpha", 1.0, transform="log")],
    laplace_options={"backend": "sympy"},
)
```

An optional `hyperprior_logpdf` changes type-II maximum likelihood into a
hyperparameter MAP objective while retaining the conditional model evidence in
the evaluation metadata.

## MacKay fixed-point updates

`mackay_laplace` implements the evidence fixed-point scheme used for isotropic
Gaussian weight precisions.  For one precision `alpha`, with Laplace posterior
mean `m`, covariance `S`, and `k` weights, the no-hyperprior update is

\[
\alpha_{new} = \frac{k}{m^T m + \operatorname{tr}(S)}.
\]

`mackay_precision_update` also accepts the derivative of a log hyperprior with
respect to `log(alpha)`.

For Gaussian regression with both weight precision `alpha` and noise precision
`beta`, `mackay_precision_noise_update` implements the classical two-precision
update.  `mackay_fixed_point` is the lower-level reusable iterator; it accepts
any evaluator returning a Laplace-style Gaussian posterior plus evidence.

The result reports the actual final fixed point (or last iterate), not merely
the best intermediate evidence value, and retains the full path in `history`.

## Exact evidence as a reference

The general interface is not Laplace-specific.  Exact models such
as `BayesianMultivariateLinearRegression` can be passed directly to
`optimize_evidence`.  This provides a useful reference for testing approximate
empirical-Bayes strategies and avoids duplicating optimization infrastructure
inside each conjugate model family.
