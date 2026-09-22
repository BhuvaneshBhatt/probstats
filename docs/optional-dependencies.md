# Optional dependencies

The base installation requires only NumPy and SymPy. Optional extras are separated so core probability and statistics remain usable without heavy numerical, tabular, symbolic-reasoning, or accelerator stacks.

| Extra | What it unlocks | Boundary expectation |
| --- | --- | --- |
| `tabular` | Pandas-aware inputs, labels, and tabular result presentation | Core array/list workflows must work with Pandas unavailable. |
| `laplace` | SciPy-backed numerical optimization and related numerical routines | Symbolic/core APIs must import without SciPy. |
| `exact` | Multiple integration, expression testing, and semialgebraic exact reasoning | Missing exact backends must produce explicit unavailable/unknown outcomes rather than silently changing semantics. |
| `certify` | Expression and semialgebraic certification | Certification APIs should clearly distinguish proved, disproved, and unresolved claims. |
| `reasoning` | Semialgebraic and function-property reasoning | Core distribution and statistics APIs must not import these packages eagerly. |
| `arviz` | Conversion of posterior samples to ArviZ structures | Diagnostics that do not require ArviZ remain available. |
| `jax` | JAX neural-network functionality | Importing `probstats.bayes` must not require JAX. |

CI runs a dependency matrix covering base, each major extra, and the combined environment. The no-Pandas regression test additionally installs an import guard around Pandas to enforce the tabular boundary in-process.

## Algebraic statistics and tensor geometry

- `probstats[algebraic]` installs `semialg` for exact dimensions, degrees, and singular loci of polynomial statistical models.
- `probstats[tensor]` installs `tensoratlas` for probability-tensor flattenings and generic tensor geometry.
- `probstats[algebraic-tensor]` installs both backends for independence models and exact algebraic tensor workflows.

The `probstats.algebraic` namespace itself imports without either optional backend; a backend is loaded only when an operation requires it.
