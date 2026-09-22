# Algebraic statistics reference

This page is the **API and model reference** for `probstats.algebraic`: use it to look up public objects, accepted inputs, model semantics, complexity guards, and backend requirements. For a worked end-to-end tutorial, see [Algebraic Statistics with probstats + semialg + TensorAtlas](algebraic-statistics-guide.md). For the precise meaning of `exact`, `complete`, `certified`, genericity, inconclusive identifiability, and tensor-rank bounds, see [Certificate semantics](algebraic-statistics-certificates.md).

`probstats.algebraic` represents statistical models whose probability coordinates satisfy polynomial equations. Statistical semantics stay in `probstats`; generic polynomial geometry is delegated to `semialg`, and generic tensor geometry is delegated to `tensoratlas`.

## Algebraic models

```python
import sympy as sp
from probstats.algebraic import AlgebraicModel

p0, p1 = sp.symbols("p0 p1", real=True)
model = AlgebraicModel((p0, p1), equations=(p0 - p1,))

model.variety  # equalities, including normalization
model.region  # equalities plus probability nonnegativity
model.ideal()
```

`model.variety` is the affine equality model. `model.region` is the statistical model and also enforces `p_i >= 0`. Exact dimension, degree, and singular-locus queries require the `algebraic` extra, which installs `semialg`.

```python
model.dimension()
model.degree()
model.singular_locus()
```

## Contingency tables

```python
from probstats.algebraic import ContingencyTable

table = ContingencyTable([[12, 8, 4], [5, 9, 7]], ("X", "Y"))
table.total
table.marginal("X")
table.marginals()
```

Counts are exact nonnegative integers. Table shape and variable names are validated when the object is created.

## Input normalization and coordinate ordering

Discrete dimensions, latent-class counts, tensor orders, component counts, and count data are interpreted as **exact integers**. Numeric coercion is never used to turn a nonintegral value into an integer. For example, a cardinality of `2.5`, a moment order of `3/2`, or a latent-class count of `1.9` is invalid rather than being truncated.

`ContingencyTable.flat_counts` defines the canonical row-major vectorization used whenever a table is passed to a toric sufficient statistic, fiber constructor, likelihood calculation, or exact conditional test. This keeps table-shaped and vector-shaped inputs mathematically identical and prevents each consumer from inventing its own flattening convention.

## Complexity guards

Exact algebraic routines expose explicit budgets rather than starting potentially combinatorial work implicitly. The main guards are `Fiber.enumerate(max_candidates=...)`, `algebraic_mle(max_faces=...)`, `LatentClassModel.image_ideal(max_parameters=...)`, and `cumulant_tensor(max_order=...)`. A guard is checked before the expensive enumeration, elimination, critical-system solve, or set-partition expansion begins. Supplying the exact required budget permits the operation; supplying one less fails without starting that backend work.

Probability-mixture recovery has a stronger input contract than generic tensor decomposition. Exact symbolic cross-moments must have a provable total of one and provably nonnegative entries before their CP factors are interpreted as mixing weights and component probabilities. Floating-point cross-moments use the documented numerical tolerance. Generic symbolic tensors whose probability constraints are unresolved should be decomposed with TensorAtlas directly rather than interpreted as a statistical mixture.

## Complete independence

A joint categorical distribution is completely independent exactly when its probability tensor has rank one. `probstats` owns the probability-model interpretation; TensorAtlas supplies the Segre tensor variety and flattening equations.

```python
from probstats.algebraic import independent_model

model = independent_model((2, 3), variables=("X", "Y"))
model.equations
model.region
```

For a `2 x 3` table, the model equations are the three `2 x 2` minors of the probability matrix. Normalization and nonnegativity are added by the statistical model rather than by the generic Segre variety.

## Probability tensors

```python
from fractions import Fraction
from probstats.algebraic import ProbabilityTensor, tensor_independent

p = ProbabilityTensor(
    [[Fraction(1, 6), Fraction(1, 3)], [Fraction(1, 6), Fraction(1, 3)]],
    ("X", "Y"),
)

p.flatten(("X",), ("Y",))
tensor_independent(p)
```

Tensor operations require the `algebraic-tensor` extra. `tensor_independent` returns `True`, `False`, or `None`: `None` means the symbolic residuals were not sufficient to decide the rank-one equations exactly.

`ProbabilityTensor` keeps query semantics separate from constructive validation. `is_normalized()` and `is_nonnegative()` may return `None` when symbolic evidence is insufficient; `validate()` returns both states together, while `require_probability()` returns the same tensor only when normalization and nonnegativity are both proved. Constructive statistical routines should use the strict form rather than treating an unresolved symbolic property as true.

```python
validation = p.validate()
validation.valid
# True

p.require_probability() is p
# True
```

## Conditional independence

```python
from probstats.algebraic import conditionally_independent_model

model = conditionally_independent_model(
    {"X": 2, "Y": 2, "Z": 3},
    [("X", "Y", ("Z",))],
)
```

The conditional-independence constructor accepts statements `(left, right, conditioned)` whose groups partition the modeled variables. It constructs the determinantal equations of each conditioned probability-table slice. Requiring a partition avoids silently treating omitted variables as fixed when the correct statistical operation would be marginalization.

## Dependency boundaries

Install only the layer needed by the calculation:

```bash
pip install "probstats[algebraic]"        # semialg-backed exact model geometry
pip install "probstats[tensor]"           # TensorAtlas tensor operations
pip install "probstats[algebraic-tensor]" # both
```

Importing `probstats` or `probstats.algebraic` does not import either optional dependency. The dependency is resolved only when a method requires that backend.

## Toric models

`ToricModel` represents a normalized toric probability model from a homogeneous integer design matrix. If the columns of `A` are `a_i`, the probabilities have monomial form

```text
p_i(theta) = theta**a_i / sum_j theta**a_j.
```

Homogeneity is required because the statistical fibers use a fixed sample size. It also guarantees that every integer kernel move preserves total count.

```python
from probstats.algebraic import ToricModel

A = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)
model = ToricModel(A)

model.toric_ideal()
model.sufficient_statistic((1, 3, 3, 1))
model.markov_basis()
```

For this binary independence design, the toric ideal is generated by `p0*p3 - p1*p2` and the Markov basis contains the move `(1, -1, -1, 1)`.

Generic integer-kernel, lattice-ideal, toric-ideal, and Markov-basis computation belongs to `semialg.algebraic`. `probstats` adds the probability normalization, sufficient-statistic interpretation, and conditional count model.

## Fibers and exact conditional inference

A `Fiber` is the exact nonnegative integer set

```text
{x in N^n : A*x = b}.
```

```python
fiber = model.fiber((1, 3, 3, 1))
fiber.enumerate()
fiber.markov_basis()
```

Exact enumeration is available for ordinary nonnegative log-linear design matrices when each coordinate has a finite bound certified directly from `A*x=b`. The enumeration API has a candidate limit so an unexpectedly large exact test does not silently become an unbounded computation.

`conditional_test` conditions on the sufficient statistic. The resulting toric/log-linear count probability is proportional to `1 / prod(x_i!)` on the fiber.

```python
from probstats.algebraic import conditional_test

result = conditional_test((1, 3, 3, 1), model)
result.pvalue
result.fiber_size
result.exact
```

The default probability-ordering test sums the masses of all fiber points no more probable than the observed table. A callable statistic can instead request an exact upper or lower tail.

For larger fibers, `sample_fiber` uses Markov-basis moves. Its default `measure="conditional"` applies a Metropolis correction for the same `1 / prod(x_i!)` target; `measure="uniform"` is available when uniform fiber sampling is the intended distribution.

```python
from probstats.algebraic import sample_fiber

chain = sample_fiber(
    (1, 3, 3, 1),
    model=model,
    size=5_000,
    burnin=500,
    rng=123,
)
chain.acceptance_rate
```

## Likelihood geometry and algebraic MLE

`likelihood_equations(model, counts)` forms the logarithmic likelihood score equations, eliminates Lagrange multipliers with semialg, and saturates by the product of the probability coordinates. `critical_points` solves the resulting finite very-affine system exactly.

`algebraic_mle` also checks the coordinate faces permitted by zero observed counts, so boundary MLEs are not discarded. `maximum_likelihood_degree` reports the stable exact complex critical-point count at deterministic generic-data witnesses. The witness counts are exposed in the result; `generic_certified=False` distinguishes this reproducible generic evidence from a symbolic exceptional-locus certificate.

## Latent-class models and identifiability

`latent_class_model(cardinalities, latent_classes)` represents a finite mixture of product distributions. The statistical parameterization removes simplex redundancies before algebraic analysis: the final mixing weight and final category probability in every conditional vector are represented as one minus the remaining entries.

```python
from probstats.algebraic import latent_class_model, generic_identifiability

model = latent_class_model((2, 2, 2), 2, variables=("X", "Y", "Z"))
result = generic_identifiability(model)
```

Generic identifiability combines two distinct certificates. An exact Jacobian witness establishes the generic image/fiber dimension when it reaches the rigorous secant/ambient upper bound. TensorAtlas then applies the generic Kruskal criterion for essential CP uniqueness. A zero-dimensional generic fiber alone is reported only as finite-to-one; it is not silently promoted to global uniqueness.

For latent classes, essential uniqueness means uniqueness up to permutation of hidden labels. `IdentifiabilityResult.expected_label_orbit` records the corresponding factorial orbit size. Positive-dimensional fibers are certified non-identifiable. If available uniqueness criteria are insufficient, the result is explicitly inconclusive.

`identifiability(model, parameters)` performs pointwise Jacobian analysis at an exact parameter assignment. Full local rank certifies local finite-to-one behavior only; singular parameter points with a positive-dimensional local fiber are reported as non-identifiable.

`LatentClassModel.image_ideal()` delegates exact graph elimination to semialg and includes an explicit parameter-count guard because lexicographic elimination can grow rapidly. `fit_latent_class()` provides a numerical EM fit for contingency tables; statistical fitting remains in probstats while generic CP/secant geometry remains in TensorAtlas.

## Moment and tensor statistics

`moment_tensor`, `central_moment_tensor`, and `cumulant_tensor` construct exact finite-sample multivariate tensor statistics. Raw and central moments use the usual outer-power definitions, while cumulants use the exact set-partition formula with an explicit order guard because the number of partitions grows combinatorially.

```python
from probstats.algebraic import moment_tensor, cumulant_tensor

samples = ((1, 2), (3, 4))
M2 = moment_tensor(samples, 2)
K3 = cumulant_tensor(samples, 3)
```

The returned `MomentTensor` retains its tensor order, statistic kind, centering status, sample size, and exact SymPy entries. Its `.tensor` property converts lazily to TensorAtlas only when tensor geometry or decomposition is requested.

`multi_view_moment` computes an aligned cross moment `E[X1 ⊗ ... ⊗ Xk]`. For categorical observations, `categorical_multi_view_moment` performs one-hot encoding and therefore returns the empirical joint-probability tensor directly.

```python
from probstats.algebraic import categorical_multi_view_moment

M = categorical_multi_view_moment(
    ((0, 0, 1, 1), (0, 1, 0, 1)),
    cardinalities=(2, 2),
)
```

`decompose_moment_tensor` delegates generic CP or symmetric Waring decomposition to TensorAtlas and keeps the decomposition, rank bounds/certificate, and available Kruskal uniqueness information together in `MomentDecompositionResult`. Numerical decompositions do not become exact merely because they have a small residual.

For conditionally independent multi-view latent models, the cross-probability tensor has the form

```text
sum_h lambda_h * a_h^(1) tensor ... tensor a_h^(k).
```

`recover_multiview_mixture` converts TensorAtlas CP scaling into statistical parameters by normalizing each recovered mode vector to sum to one and absorbing the corresponding scale into the mixture weight. Because that normalization is only statistically meaningful for simplex-valued views, the input must be a nonnegative cross-probability tensor whose entries sum to one. Generic numeric feature moments should be decomposed with `decompose_moment_tensor` instead.

`fit_multiview_mixture` first constructs the empirical multi-view moment and then applies the same recovery path. `MultiViewMixtureResult` records convergence, exactness, residual, recovered weights/component means, and whether TensorAtlas supplies a certified generic CP-identifiability result. Label permutations remain an unavoidable symmetry; component ordering in the returned object is canonicalized only for deterministic presentation.
## Interpreting certificates

See [Certificate semantics for algebraic statistics](algebraic-statistics-certificates.md) for the exact meaning of `exact`, `complete`, `certified`, genericity, three-valued identifiability, tensor-rank bounds, and the capability matrix.

