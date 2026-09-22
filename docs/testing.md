# Testing strategy

The test suite combines example-based regressions with registry-driven cross-cutting contracts. Public APIs are tested as exact discovery sets; fresh-import tests detect accidental root leakage; documentation imports and links are validated; and selected documentation examples execute in CI.

## Distribution contracts

`tests/distribution_contract_registry.py` is the catalogue contract. Every concrete scalar distribution exported from `probstats.distributions` must appear exactly once. Each entry supplies a safe numeric instance, probe points, probability levels, optional SciPy mapping, and explicit exceptions for protocols that are unavailable or prohibitively symbolic. A coverage test fails when a new scalar law is exported without corresponding contract metadata.

The registry drives support/density, sample-shape, CDF monotonicity, CDF-survival complement, generalized-inverse quantile, numerical normalization, and SciPy differential tests. Separate Hypothesis tests randomize valid parameters for representative native-quantile families, so testing is not limited to one hand-picked parameter tuple.

SciPy differential tests compare density/PMF, CDF, and quantiles where a compatible direct oracle exists. Exact symbolic formulas are also evaluated at high precision and compared with independent numerical references for rational-parameter cases. This specifically tests the transition from symbolic mathematics to floating-point evaluation.

## Metamorphic testing

Metamorphic tests cover identities that do not need an external oracle: affine location-scale transformation, Normal/Poisson/Gamma convolution closure, CDF-survival complement, KL identity/nonnegativity, and covariance symmetry/positive-semidefiniteness. New families should add algebraic identities when they have stable closure or invariance laws.

## Failure semantics

Failure categories are tested independently: invalid mathematical input, unsupported public operations, absent optional backends, available-backend evaluation failure, numerical nonconvergence metadata, and proof-oriented `UNKNOWN` states. See [Failure semantics](failure-semantics.md).

## Optional dependencies and CI

Tests for optional dependencies are split by environment rather than assuming the richest install. The CI matrix exercises base, tabular, Laplace/SciPy, exact, reasoning, JAX, and all-extras installations. Core regression is also tested with Pandas unavailable.

Coverage is used as a gap-finding tool rather than a release target. Priority branches are invalid-input handling, optional-backend fallbacks, symbolic condition handling, exact/certification outcomes, and numerical fallback transitions.

## Algebraic-statistics contracts

The algebraic-statistics tests are organized by mathematical contract rather than implementation chronology. Example-based theorem tests verify agreement across package boundaries; property tests exercise invariance under sample, variable, and tensor permutations; degeneracy tests cover singular and boundary cases; certificate tests ensure inconclusive evidence is not promoted to a theorem; and a registry-driven input-contract suite applies one exact-integer policy to every public algebraic parameter that represents a count, rank, order, cardinality, component count, iteration bound, or complexity guard.

Cross-representation properties verify that a contingency table and its canonical flat count vector produce the same toric statistic, fiber, likelihood equations, and exact conditional inference, and that equivalent probability-tensor representations preserve independence decisions. Symbolic adversarial tests exercise undecidable sign, normalization, integrality, and Segre residuals explicitly: statistical APIs that require a probability object demand proof of the required property, while three-valued query APIs retain `None` rather than promoting unresolved evidence to `True` or `False`.

New tests should prefer an independent mathematical invariant over reproducing the implementation. Numerical decomposition tests should assert stochastic constraints and certificate fields in addition to residual size, and exact tests should include a paired unsupported or inconclusive case whenever the public result type can represent one.

### Complexity guards and failure atomicity

Public complexity guards are tested at their operational boundary. For a problem with a known required budget, the exact budget must permit the operation and one less must fail before the protected enumeration, elimination, critical-system solve, or combinatorial expansion starts. These tests cover fiber candidate enumeration, latent-image elimination, likelihood face enumeration, and cumulant order. Cheap scalar/input validation is likewise required to occur before an optional backend is imported when the backend is not needed to determine that the request is invalid.

### Installed-wheel release contracts

`tests/test_installed_package_contracts.py` is the permanent installed-artifact contract suite. Ordinary source runs skip it. Release validation installs the built `probstats`, `semialg`, and TensorAtlas wheels into a clean target, copies the test outside the repository tree, and runs it with `PROBSTATS_INSTALLED_CONTRACTS=1`. This prevents repository-relative imports from satisfying the contract accidentally. The suite checks public imports, package ownership, the binary-independence theorem chain, contingency-table/vector coordinate equivalence, three-valued certificate semantics, and latent/moment smoke behavior.


## Internal facade coherence

The public `probstats.inference`, `probstats.testing`, and `probstats.data` modules are import-only facades. Their implementations are divided by stable responsibility—likelihood/resampling/regression/nonparametric/profile inference; testing primitives/parametric/dependence/variance/diagnostics; and weighted/rolling/exponentially-weighted/smoothing/sequence data operations. Architecture tests require these facade modules to remain definition-free so future growth does not recreate mixed-responsibility monoliths. Public names remain owned by the same facade modules.

## Numerical tolerance policy

Algebraic-statistics numerical defaults come from one internal tolerance policy. Exact symbolic decisions never consult numerical tolerances. Numerical decomposition uses the decomposition tolerance, stochastic normalization/recovery uses the stochastic tolerance, and iterative latent-model fitting uses the fitting tolerance. Public numerical entry points reject negative, non-finite, Boolean, or symbolic tolerance values before starting backend work.

## U-statistic and randomized-algorithm properties

Small U-statistics are checked against direct subset enumeration for randomized scalar and vector examples. Fixed-budget incomplete statistics must agree with complete enumeration when the budget covers every subset. Bernoulli selection is tested as a distribution: repeated seeded realizations check the Binomial count mean and variance rather than only a few selected seeds.

Tests for randomized APIs distinguish two contracts. Repeating the *same configuration* with the same seed must reproduce the same realization. Changing an execution parameter that changes random-number consumption, such as bootstrap batch size, need not preserve the exact stream unless the API explicitly promises that property.

Symbolic decision tests cover proof, disproof, and unresolved expressions. Tests of public consumers then verify that unresolved symbolic evidence is not silently converted to inequality or falsity.

## Performance contracts

Correctness tests enforce structural performance properties when they are deterministic. For example, exact polynomial-kernel construction at high order must not retain the factorial permutation group in memory. Wall-clock thresholds are not part of ordinary pytest because they are machine-dependent. Small manual benchmarks for complete and incomplete U-statistics live under `benchmarks/` and can be used when changing combinatorial inner loops.

## Release shards

Release validation should use deterministic file-level shards when one-process execution is affected by optional numerical backends or interpreter shutdown behavior. Every test file must belong to exactly one shard, and the aggregate counts—not an individual shard—form the release result. Sharding is an execution strategy only: tests must not rely on shard order or shared process state.


## Independent random streams

Tests of stochastic pipelines should verify both replayability and stream ownership. In SDL tests, augmentation, symmetrization, subset selection, projection shuffling, bootstrap multipliers, and calibration data generation use separate derived streams. A test that changes one option should assert that unrelated realized objects, such as selected subset indices or simulated calibration datasets, remain unchanged.

## Three-valued symbolic predicates

Certified symbolic predicates return `TruthValue.TRUE`, `TruthValue.FALSE`, or `TruthValue.UNKNOWN`. `TruthValue` rejects implicit Boolean conversion. Tests should exercise all three states and should verify the public operation consuming the result, not only the predicate helper.
