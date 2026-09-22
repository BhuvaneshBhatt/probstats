# Semialgebraic hypothesis representations

`probstats.algebraic.testing` represents null hypotheses whose parameter regions are
semialgebraic.  The representation is separate from the numerical test
algorithm: it records the mathematical null, ambient parameter space, estimator hooks,
and sample-space metadata without committing to a particular U-statistic or bootstrap.

A `BasicSemialgebraicNull` stores polynomial equalities and closed polynomial
inequalities. Inequalities are normalized internally to the convention `f <= 0`.
Equalities remain explicit so their provenance is not lost; `sdl_constraints` expands
each equality `g = 0` to `g <= 0` and `-g <= 0` when an all-inequality representation
is needed.

`SemialgebraicHypothesis` stores one or more basic components, so finite unions are
represented without flattening their component structure. `null_formula` intersects the
union with the ambient parameter region, and `alternative_formula` is the complement of
the null relative to that ambient region. The `basic_null` property rejects
multi-component nulls rather than silently applying a basic-set procedure to a union.

Existing `AlgebraicModel` instances can be adapted with
`SemialgebraicHypothesis.from_model()`. Probability normalization, nonnegativity, model
equalities, and model metadata are carried into the hypothesis representation rather than
reimplemented as a second statistical-model hierarchy.

Exact comparison of two represented null regions is available through
`semialgebraically_equivalent()`, which delegates the decision to `semialg`.

## Unbiased polynomial kernels

`probstats.algebraic.testing` can turn the canonical `f_j(theta) <= 0`
constraints of a basic null into the symmetric unbiased kernel required by the SDL
construction. `UnbiasedParameterEstimator(function, arity=eta)` records an unbiased
estimator of a parameter from `eta` independent observations. Plain callables are the
common `eta = 1` shorthand.

`polynomial_constraint_kernel(component, estimators)` expands each polynomial into
monomials, assigns independent observation blocks to the parameter estimators in each
monomial, and exactly symmetrizes the resulting kernel over all permutations. Its default
order is

```
eta * max(deg(f_j)).
```

All estimators used by a kernel must currently have the same arity, matching the SDL
construction. `hypothesis_kernel(hypothesis)` obtains the estimators directly from the
hypothesis object. It accepts only a basic null; use `sdl_union_test()` for finite-union nulls.

Exact full symmetrization is factorial in runtime, but permutations are generated lazily so
kernel construction does not also require factorial memory. Random partial symmetrization
is available explicitly when that computational approximation is appropriate.

## U-statistics

The general `probstats.u_statistics` module provides `u_statistic()` and
`incomplete_u_statistic()` for scalar or vector-valued symmetric kernels. Kernel order can
be supplied explicitly or read from a `Kernel`. Generic U-statistic functions assume the
caller supplies a mathematically symmetric kernel; checking symmetry from finitely many
evaluations would be unreliable and expensive. `polynomial_constraint_kernel(...,
symmetrization="exact")` constructs a symmetric kernel automatically.

`incomplete_u_statistic(..., selection="bernoulli")` implements the randomized SDL
selection law without enumerating every subset. If `M = C(n,m)`, it first draws
`Nhat ~ Binomial(M, N/M)` and then draws `Nhat` subset ranks uniformly without
replacement. This is distributionally identical to assigning independent
`Bernoulli(N/M)` selectors to all `m`-subsets. A zero selected count is reported explicitly
because the normalized incomplete U-statistic is then undefined.

For non-SDL uses, `selection="fixed"` evaluates exactly the requested number of uniformly
sampled subsets. `UStatisticResult` records the sample size, order, total subset count,
actual number of evaluations, requested budget, and—in the randomized case—the selected
subset indices. Callers can request retention of the concrete kernel values; SDL
studentization does so to avoid evaluating the selected subsets twice.


### Gaussian multiplier bootstrap

`sdl_multiplier_bootstrap()` uses an `SDLStudentizationResult` to generate the two conditional Gaussian multiplier processes for the incomplete-kernel and Hájek-projection terms, combines them as `m * U_g# + sqrt(alpha) * U_h#`, studentizes by the corresponding variance estimate, and returns the bootstrap maximum statistics and empirical p-value. The p-value follows the SDL definition `#{W >= T}/A`; it can therefore be zero for a finite bootstrap sample. Batching preserves the bootstrap law but can change the exact seeded realization because multiplier draws are interleaved by batch.


### End-to-end SDL test

`sdl_test(sample, hypothesis, budget=..., bootstrap_replicates=...)` composes the core SDL pipeline for a single basic semialgebraic null. It returns `SDLTestResult`, retaining the hypothesis, polynomial kernel, incomplete-U realization, Hájek/variance estimates, coordinate and maximum statistics, bootstrap distribution, p-value, and execution parameters. A finite-union null is rejected at this layer; componentwise intersection-union testing is reserved for the reducible-null extension.


### Constraint augmentation

`augment_constraints()` adds `r` redundant random convex combinations of the canonical SDL inequalities using Dirichlet weights. The original constraints are retained, so the represented null set is unchanged by construction: each added inequality is implied by the originals, while retaining the originals gives the reverse inclusion. `augment_hypothesis_constraints()` preserves the hypothesis context, and `sdl_test(..., augment_constraints_count=r)` uses a dedicated child random stream for augmentation. Exact duplicate generated coordinates are discarded, so requested augmentation count is an upper bound on added coordinates.


### Kernel symmetrization

Exact full permutation symmetrization remains the default kernel construction and iterates over the permutation group lazily rather than storing all `m!` permutations. For larger kernel orders, `symmetrization="random"` with `permutation_count=s` samples `s` independent uniform permutations once when the kernel is constructed and averages the corresponding unsymmetrized evaluations. The sampled approximate kernel is then fixed throughout the U-statistic, projection, variance, and bootstrap computation. This is an explicitly approximate computational option: the empirical partial-symmetrization strategy does not inherit the same theoretical status as exact full symmetrization. `sdl_test()` exposes it through `kernel_symmetrization` and `symmetrization_permutations` and draws the permutations from the procedure's common RNG stream.


### Reducible nulls and intersection-union testing

For a finite union null `H0 = H01 ∪ ... ∪ H0K`, `sdl_union_test()` runs the ordinary basic-component SDL test on every component. Rejecting the union requires rejecting every component, so the intersection-union combined p-value is `max(p_1, ..., p_K)`. The result retains all component `SDLTestResult` objects rather than collapsing their diagnostics. Both `seed=` and an explicit `Generator` derive independent child streams for components; the derived component seeds are recorded for replay. Constraint augmentation and symmetrization options are forwarded componentwise.

### Diagnostics and simulation calibration

`sdl_diagnostics()` summarizes a completed basic-null SDL run without introducing new randomness. It reports the realized-to-requested incomplete-U budget ratio, `alpha_n`, kernel order, projection size and block count, coordinatewise variance shares from the Hájek and incomplete-kernel terms, zero-standard-error coordinates, empirical bootstrap critical values, and the Monte Carlo standard error of the reported bootstrap p-value.

`sdl_calibrate()` estimates rejection frequency under an explicit caller-supplied null data-generating process. The sampler receives a simulation-only NumPy generator and must return one complete sample. Test randomness uses an independent child stream, so changing bootstrap or symmetrization settings does not change later simulated datasets. Calibration records the rejection rate at the requested significance level, its binomial Monte Carlo standard error, all p-values/statistics, realized incomplete-U budgets, kernel orders, and projection sizes. It does not infer a simulator from a symbolic semialgebraic hypothesis. This makes calibration assumptions explicit and permits controlled studies of the choices emphasized by the SDL methodology, including kernel order, `N`, `n1`, constraint augmentation, and partial symmetrization.

### Published trinomial reference models

`trinomial_reference_hypothesis(1..4)` reproduces the four trinomial semialgebraic descriptions used by Barnhill et al. (2025) to study SDL behavior. The probability simplex is kept as the ambient parameter space, while the published model equations and inequalities form the SDL null coordinates. `trinomial_sampler()` generates the standard-basis observations used in those experiments, and `TRINOMIAL_PAPER_CONFIGURATION` records the Section 3 settings `n=300`, `N=1000`, `A=1000`, and `n1=300`. `trinomial_model4_components()` supplies the three Model-2-like components used for the paper's intersection-union comparison. These helpers provide reproducible reference fixtures; they do not encode the paper's empirical conclusions as assertions, since those conclusions depend on Monte Carlo experiments and implementation choices.

See the [worked semialgebraic example](algebraic-testing-worked-example.md) for an end-to-end calculation and scaling table.


### Random-stream ownership and degenerate coordinates

A top-level SDL run derives separate random streams for constraint augmentation, random kernel symmetrization, incomplete-U subset selection, Hájek block shuffling, and the multiplier bootstrap. This makes methodological comparisons stable: changing one stochastic option does not silently change unrelated random choices.

Studentization treats zero estimated standard error according to the numerator. A `0/0` coordinate contributes zero. A nonzero numerator divided by zero remains signed infinity; diagnostics expose `zero_over_zero` and `nonzero_over_zero` masks so callers can distinguish these cases.
