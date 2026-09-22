# Semialgebraic and function-property reasoning

The `probstats.bayes.reasoning` API is an optional proof and structural-analysis layer.
It uses `semialg` for exact real sign/inequality reasoning and `funcprops` for global
function properties when those packages are installed. Unsupported problems remain
`UNKNOWN`; no numerical or heuristic result is promoted to a proof.

## `ProofStatus`

Tri-state proof result: `TRUE`, `FALSE`, or `UNKNOWN`.

## `PropertyCertificate`

Structured exact property decision with backend, method, explanation, and optional
backend evidence. `value` is `True`/`False` only when the decision is certified.

## `certify_sign`

```text
certify_sign(expr, relation, *, variables=None, assumptions=True, use_semialg=True)
```

Certify one of `positive`, `nonnegative`, `negative`, `nonpositive`, `zero`, or
`nonzero`. The optional semialg backend uses its exact implication/CAD reasoning and
returns `FALSE` only when a counterexample certifies failure of the universal claim.

## `certify_positive`

Certify strict positivity, including parameter-conditioned semialgebraic expressions.

## `certify_nonnegative`

Certify nonnegativity.

## `certify_negative`

Certify strict negativity.

## `certify_nonpositive`

Certify nonpositivity.

## `MatrixDefinitenessCertificate`

Structured Sylvester-criterion result containing the leading principal minors and
their individual positivity certificates.

## `certify_positive_definite`

```text
certify_positive_definite(matrix, *, assumptions=True, use_semialg=True)
```

Certify positive definiteness of a real symmetric matrix by proving positivity of all
leading principal minors. This is used by Laplace inference to distinguish mathematically certified
MAP curvature from a merely numerically positive Hessian.

## `PosteriorGeometry`

Compact structural summary of a posterior log density: convexity/concavity,
continuity, singularities, backend source, and detailed provenance.

## `analyze_posterior_geometry`

```text
analyze_posterior_geometry(log_density, variables, *, domain=True, use_funcprops=True)
```

Use `funcprops` to classify global posterior geometry. Multivariate convexity uses the
public tuple-variable API. Continuity and singularity queries are currently applied to
one-dimensional posterior domains, where `funcprops` has its strongest global
real-analysis support. The inference planner uses favorable global concavity to lower Laplace's routing
cost and singular/discontinuous geometry to prefer global sampling when available.


## Domain-aware geometry and caching

One-dimensional singularity information is intersected with the requested
posterior domain/assumptions before it affects planner policy. A pole or branch
obstruction outside the effective support therefore does not penalize Laplace.
Posterior geometry queries are memoized by expression, variables, domain, and the
active funcprops API identity, allowing the planner and Laplace inference to reuse expensive structural
analysis without weakening correctness under backend replacement or tests.
