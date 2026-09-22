# Laplace inference and optimization API

The Laplace layer provides local Gaussian posterior/evidence approximation while keeping MAP
optimization behind interchangeable backends. SymPy is always available; SciPy
and symbopt are optional.

## Public functions

### `symbolic_hessian`

API name: `symbolic_hessian`

Signature: `symbolic_hessian(expression, variables)`
Returns the exact SymPy Hessian of an expression with respect to the requested variables.

### `precision_at`

API name: `precision_at`

Signature: `precision_at(log_density, variables, point)`
Evaluates the negative Hessian at a candidate MAP and symmetrizes the numerical matrix.

### `laplace_log_evidence`

API name: `laplace_log_evidence`

Signature: `laplace_log_evidence(log_density_at_mode, precision)`
Computes the ordinary second-order Laplace log-integral from a mode value and
positive-definite precision matrix.

### `infer_laplace`

API name: `infer_laplace`

Signature: `infer_laplace(model, *, targets=None, backend="auto", initial_guess=None, assumptions=True, optimizer_options=None, geometry=None, corrector=None, order=2)`
Optimizes the observed joint log density, constructs the local Gaussian posterior,
and computes second- or higher-order Laplace evidence. With `backend="symbopt"`,
continuous supports and relational assumptions are translated to symbopt
constraints. Symbopt certification metadata is retained in the inference result.

### `fit_precision_at_max`

API name: `fit_precision_at_max`

Signature: `fit_precision_at_max(points, values)`
Fits the precision matrix of a local quadratic approximation from sampled points
and log-density/evidence values.

### `get_optimization_backend`

API name: `get_optimization_backend`

Signature: `get_optimization_backend(backend)`
Resolves `"sympy"`, `"scipy"`, `"symbopt"`, or `"auto"`. Auto prefers symbopt,
then SciPy, then SymPy.

## `SymboptOptimizationBackend`

The supported integration targets symbopt's public `maximize()` API. The backend:

1. converts real interval supports to exact inequalities;
2. appends relational `assumptions`;
3. forwards `optimizer_options` to `symbopt.maximize` (for example
   `working_precision` and `semialg_certification`);
4. reads the represented optimizer from `best_candidate.original_variable_values`;
5. preserves `certified`, `attained`, `status`, and `optimum_value` separately;
6. rejects certified-but-unattained suprema because Laplace requires a finite MAP.

Example:

```python
result = infer_laplace(
    model,
    backend="symbopt",
    optimizer_options={
        "working_precision": 50,
        "semialg_certification": "complete",
    },
)

assert result.metadata["optimization_backend"] == "symbopt"
print(result.metadata["optimization_certified"])
print(result.metadata["optimization_status"])
```

`symbopt` is optional and can be installed with `probstats[symbopt]`.

## Higher-order and singular Laplace with `asymptotic`

### `AsymptoticCorrector`

`AsymptoticCorrector` is the supported bridge to the optional `asymptotic`
package. It calls the public `laplace_asymptotic_integral()` API on an auxiliary
concentration family

```text
exp(t * log_density(theta)),  t -> +infinity
```

and evaluates the finite asymptotic evidence expansion at `t=1` by default.
This yields a formal higher-order refinement of ordinary second-order Laplace.
Set `terms=...` to control the number of terms requested from `asymptotic`, and
`certify=True` (the default) to request its theorem-oriented real-Laplace
certificate when the integrand lies in a supported class.

```python
corrector = AsymptoticCorrector(terms=4, certify=True)
result = infer_laplace(model, order=6, corrector=corrector)
```

The public method `AsymptoticCorrector.analyze(...)` returns an
`AsymptoticLaplaceAnalysis`; `AsymptoticCorrector.correction(...)` returns only
the additive log-evidence correction for the general
`HigherOrderCorrector` protocol.

### `AsymptoticLaplaceAnalysis`

Structured higher-order result containing the finite evidence expansion,
evaluated log evidence, additive second-order correction when defined,
`status`, `certified`, local derivative orders and coefficients, and the raw
remainder/certificate objects returned by `asymptotic`.

### `AsymptoticIntegrationError`

Raised when the optional dependency is unavailable or the model is outside the
supported asymptotic-Laplace geometry.

### `SingularLaplacePosterior`

When the MAP Hessian is positive semidefinite but singular, ordinary Gaussian
Laplace is invalid. With `AsymptoticCorrector`, the inference routine searches each separable
coordinate for the first nonzero even local decay term

```text
log p(theta) = log p(theta_hat) - a (theta-theta_hat)^m + ...,
```

and represents the leading posterior by the normalized exponential-power law
`exp(-a |delta|^m)`. Quartic, sextic, and other supported even-order saddles can
therefore be represented without inventing a covariance matrix.

`SingularLaplacePosterior.dimension`, `SingularLaplacePosterior.logpdf(...)`, and
`SingularLaplacePosterior.pdf(...)` are public.

### Current multivariate boundary

The current public `asymptotic.laplace_asymptotic_integral` API is
one-dimensional. `probstats.bayes` therefore supports multivariate higher-order
or singular Laplace through this backend when the log density is additively
coordinate-separable at the MAP, in which case the evidence integral factors
into one-dimensional components. A genuinely coupled degenerate multivariate
saddle is rejected explicitly; it will require a future multivariate Laplace
integral API in `asymptotic` rather than an unjustified product approximation.

Install the optional backend with:

```bash
pip install "probstats[asymptotic]"
```


## exprtest curvature certification

`infer_laplace` retains the exact symbolic precision determinant at the MAP
and applies `certify_zero` to it. A proved zero determinant supplies exact
degeneracy evidence for the singular-Laplace route; a proved nonzero
determinant distinguishes very small genuine curvature from an exactly flat
direction. Certification metadata is preserved in both `metadata` and
`diagnostics` for downstream quality assessment. See `docs/api/certification.md`.


## Robustness and constraint semantics

Laplace optimization uses each target's **model-effective support**, including
distribution-factor support restrictions. The SymPy and SciPy backends translate
simple one-variable relational assumptions into interval bounds. If an assumption
contains a coupled or otherwise unsupported target-variable constraint, those
backends fail explicitly and recommend `symbopt` rather than silently optimizing an
unconstrained problem.

Partial-target Laplace is rejected when another latent variable would remain
unresolved. Such a computation is neither a marginal posterior nor a conditional
posterior unless the omitted variables have first been eliminated or observed.

`GaussianLaplacePosterior` validates finite symmetric positive-definite covariance
and precision matrices, checks that they are numerical inverses, and uses
`numpy.linalg.slogdet` in `logpdf` to avoid determinant underflow. Hessian and
precision calculations also reject non-finite arrays.
