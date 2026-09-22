# Exact inference and structured integration API

Exact continuous integration is shared across the probability and Bayesian
layers. When the `exact` extra is installed, rectangular multivariate continuous
integrals first use `multiple-integrate` and fall back to native SymPy when the
structured backend is unavailable or does not evaluate the form. This shared
route is used by joint marginalization/conditioning and pushforwards,
multivariate characteristic and moment-generating transforms, symbolic
multivariate information measures, and Bayesian exact inference.

The Bayesian API additionally exposes backend selection and provenance for
posterior normalization, marginalization, and evidence calculations.

## `integrate_over`

```text
integrate_over(expr, variables, *, backend="auto", assumptions=None)
```

Exactly eliminates the requested `Variable` objects over their model-effective
supports. Distribution factors are authoritative: their supports are intersected
with any support declared on the variable, so a declared `Reals` variable with an
`Exponential` factor is integrated only over `[0, +infinity)`. `backend="auto"` uses `multiple-integrate` when all eliminated
variables have real/interval continuous support and the optional package is
installed. Discrete supports use SymPy summation.

`assumptions` is forwarded to `multiple_integrate`, which accepts ordinary
SymPy relations and `Q` predicates. This is useful for parameterized gamma,
beta, Gaussian, and other structured exact integrals.

## `evidence`

```text
evidence(model, *, backend="auto", assumptions=None)
```

Returns the exact marginal likelihood after substituting observations and
eliminating every latent variable.

## `normalize_posterior`

```text
normalize_posterior(
    model,
    *,
    targets=None,
    backend="auto",
    assumptions=None,
)
```

Computes the exact evidence and normalized `SymbolicJointDistribution`. When
`targets` is supplied, nuisance variables are marginalized exactly after
normalization.

## `marginalize`

```text
marginalize(model, variables, *, backend="auto", assumptions=None)
```

Eliminates selected latent variables from the observed joint density without
normalizing the remaining expression.

## `infer_exact`

```text
infer_exact(
    model=None,
    *,
    targets=None,
    likelihood=None,
    data=None,
    prior=None,
    registry=DEFAULT_REGISTRY,
    fallback_to_conjugacy=True,
    backend="auto",
    assumptions=None,
)
```

Runs direct exact normalization first. Its `InferenceResult.metadata["engine"]`
records the backend that actually succeeded (`"multiple-integrate"` or
`"sympy"`), while `integration_attempts` records the attempted route. If direct
integration fails and a conjugate likelihood/data/prior context was supplied,
registered conjugacy remains the exact fallback.

## Backend API

### `ExactIntegrationBackend`

Protocol for exact integration backends. Implementations expose a `name` and an
`integrate(expr, variables, *, assumptions=None)` method.

### `SymPyExactIntegrationBackend`

Always-available backend using `sympy.integrate` for continuous supports and
`sympy.summation` for discrete supports.

### `MultipleIntegrateBackend`

Optional structured backend using `multiple_integrate.multiple_integrate`.
Continuous variables are passed together in inner-first range order, matching
SymPy and MultipleIntegrate. This lets the external package recognize
multivariate Gaussian, beta/gamma, simplex/Dirichlet, polynomial-moment and
other structured families before resorting to generic antiderivatives.

```python
from probstats.bayes.exact import MultipleIntegrateBackend, integrate_over

value = integrate_over(
    expr,
    (x, y),
    backend=MultipleIntegrateBackend(),
    assumptions={a > 0},
)
```

Install it with:

```bash
pip install "probstats[exact]"
```

### `AutoExactIntegrationBackend`

Composite backend used by default. It tries `MultipleIntegrateBackend` for
purely continuous interval problems and then `SymPyExactIntegrationBackend`.

### `get_exact_integration_backend`

```python
get_exact_integration_backend(backend)
```

Resolves `"auto"`, `"sympy"`, or `"multiple-integrate"`. A custom object satisfying `ExactIntegrationBackend`
may be supplied directly.

### `ExactIntegrationTrace`

Immutable provenance record containing the backend that succeeded, variable
names eliminated, and ordered backend attempts.

### Errors

`ExactIntegrationError` is the user-facing failure when exact elimination
cannot be completed. `ExactBackendFailure` represents a backend-specific
unsupported/evaluation failure. `ExactBackendUnavailableError` is raised when
an explicitly requested optional backend is not installed.

## `SymbolicJointDistribution`

Represents a normalized symbolic joint density/mass and exposes `symbols`,
`logpdf`, `marginalize(...)`, and `marginal(...)`. The marginal methods accept
the same `backend=` and `assumptions=` options as `integrate_over`.


## exprtest normalizer certification

After exact integration, the inference layer calls `certify_zero` on the normalizing constant.
This uses certified-confidence
zero testing, so a proved-zero normalizer is rejected and a proved-nonzero
normalizer is recorded in `InferenceResult.metadata`. Unknown results remain
unknown; exact inference never converts heuristic evidence into a proof. See
`docs/api/certification.md`.


## Unified normalizer validation

`evidence`, `normalize_posterior`, and `infer_exact` share one normalizer
validity contract. The result must be finite and nonzero, and semialgebraic
reasoning is used when available to reject a normalizer that is certified
nonpositive under `assumptions`. The standalone exact APIs and `infer_exact` therefore apply the same validity rules.
