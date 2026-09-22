# Certificate semantics for algebraic statistics

This page is the **evidence and certificate reference**. Use it when interpreting structured result fields and deciding how strong a conclusion is. It avoids duplicating the step-by-step tutorial in [Algebraic Statistics with probstats + semialg + TensorAtlas](algebraic-statistics-guide.md) or the API/input catalog in the [algebraic-statistics reference](algebraic-statistics.md).

Algebraic-statistics computations can return several different kinds of mathematical evidence: an exact symbolic answer, a complete enumeration, a rigorous lower or upper bound, a generic theorem, a local Jacobian certificate, or a numerical construction. These are represented by different result fields. The fields are not interchangeable.

This page is a reference for interpreting results from `probstats.algebraic`, `semialg`, and `tensoratlas.algebraic` without making a stronger claim than the computation supports.

## The central rule

A result should be read as the conjunction of its value and its evidence fields.

For example, a numerical CP decomposition with residual `1e-12` is strong evidence that a rank-two representation exists to numerical precision. It is not an exact proof that the tensor has CP rank two. Conversely, a flattening rank of two is an exact lower bound on CP rank, but it does not by itself construct a rank-two decomposition.

The library therefore keeps these statements separate instead of reducing them to one Boolean or scalar.

## Core evidence fields

| Field | Meaning | What it does **not** mean |
|---|---|---|
| `exact=True` | The reported algebraic/numeric value was obtained with exact arithmetic or exact symbolic reasoning appropriate to that result. | The result is necessarily exhaustive or generic. |
| `complete=True` | The relevant search/enumeration/solve was exhaustive under the stated model and domain. | Every intermediate or witness statement is generic. |
| `certified=True` | A theorem, exact rank/dimension argument, or other rigorous certificate supports the stated conclusion. | The conclusion necessarily holds globally when the certificate is local or generic. |
| `generic=True` | The statement concerns a generic/dense part of parameter space rather than every parameter point. | A symbolic exceptional-locus proof has necessarily been supplied. |
| `generic_certified=True` | Genericity itself has been certified, not merely observed at deterministic generic-looking witnesses. | The statement applies at singular or exceptional parameter values. |
| `identifiable=True` | The available analysis proves the relevant notion of uniqueness. | Parameters are uniquely labeled when label switching is an allowed symmetry. |
| `identifiable=False` | The available analysis proves non-identifiability. | A sufficient uniqueness criterion merely failed. |
| `identifiable=None` | Available criteria are inconclusive. | The model is non-identifiable. |
| `rank=None` | Exact rank is unresolved by current certificates. | The rank is zero or unavailable; lower/upper bounds can still be informative. |

## Exactness and completeness are independent

An exact calculation need not be complete. `maximum_likelihood_degree()` illustrates the distinction. The complex critical-point count at each deterministic witness can be exact, while proof that those witnesses avoid every exceptional locus may still be absent.

Accordingly, a result can legitimately have:

```python
result.exact is True
result.complete is True
result.generic is True
result.generic_certified is False
```

This means the witness calculations and their solution counts were exact and complete, and they agreed in the intended generic experiment, but no symbolic exceptional-locus certificate was produced.

## Local, generic, and global identifiability

These are separate statements.

### Pointwise/local

`identifiability(model, parameters)` examines a particular parameter point. A rank-deficient Jacobian can certify that the parameterization is locally non-finite-to-one there. This is especially important at:

- zero mixture weights;
- coincident latent components;
- rank-deficient factor matrices;
- boundary or singular parameter values.

A singular point does not contradict a theorem of generic identifiability.

### Generic

`generic_identifiability(model)` studies a dense generic part of parameter space. For latent-class models it combines generic fiber dimension with TensorAtlas CP-uniqueness information.

A positive-dimensional generic fiber is a genuine non-identifiability certificate. A zero-dimensional generic fiber only proves generic finite-to-one behavior; uniqueness still needs another argument.

### Label switching

For a model with `r` latent classes, `r!` parameter points can represent the same observed distribution by permuting hidden labels. When a result reports

```python
result.identifiable is True
result.up_to_label_swapping is True
```

it means uniqueness modulo this intrinsic symmetry, not literal equality of labeled parameter vectors.

## Sufficient criteria and inconclusive results

Kruskal's CP uniqueness inequality is sufficient, not necessary. TensorAtlas therefore uses three-valued semantics:

```python
result.identifiable is True  # criterion proves uniqueness
result.identifiable is None  # criterion does not decide
```

Failure of the inequality is not converted to `False`.

The same principle applies throughout the stack: unsupported or insufficient evidence should remain unresolved rather than being interpreted pessimistically or optimistically.

## Tensor-rank certificates

`tensoratlas.algebraic.tensor_rank()` reports lower and upper bounds separately:

```python
rank.lower_bound
rank.upper_bound
rank.rank
rank.exact
```

Typical sources are:

- matrix-flattening ranks: rigorous **lower bounds**;
- exact constructive decompositions: rigorous **upper bounds**;
- numerical CP decompositions: numerical constructive upper bounds;
- equality of certified lower and upper bounds: exact rank certificate.

If the bounds do not meet, `rank.rank` remains `None`.

### Rank equations

For rank one, all appropriate flattening minors describe the Segre variety and can be sufficient. For higher CP rank, flattening minors are generally only necessary. `TensorConstraintResult.sufficient` records that distinction.

## Numerical decomposition and recovery

`cp_decompose(..., method="numerical")` and nonnegative CP are constructive numerical procedures. Even a converged fit with a tiny residual retains:

```python
result.exact is False
```

`recover_multiview_mixture()` adds statistical normalization and nonnegativity semantics to such a decomposition. Its `identifiable` and `certified_identifiable` fields concern uniqueness evidence, while `converged` concerns the numerical fitting process. They answer different questions.

## Likelihood geometry

For algebraic MLE results:

- `exact` describes arithmetic/solution exactness;
- `complete` describes whether all relevant critical points/faces were treated;
- `mle` is the selected feasible maximizer when one is obtained;
- boundary solutions remain valid when observed counts contain zeros.

An interior critical-point calculation alone is not a global MLE certificate when probability-simplex faces can contain the maximum.

## Capability matrix

The table below summarizes the main public operations and the strength of evidence they provide.

| Task / API | Owner | Exact path | Numerical path | Typical certificate | Important limitation |
|---|---|---:|---:|---|---|
| Independence equations / `independent_model` | `probstats` + TensorAtlas | Yes | No | Segre/flattening equations plus probability normalization | Conditional-independence statements require supported variable partitions. |
| Toric ideal / Markov basis | `semialg` | Yes | No | Gröbner/lattice computation | Gröbner complexity can grow quickly. |
| Exact conditional test / `conditional_test` | `probstats` | Yes | Optional sampling separately | Exact rational fiber probability | Exact enumeration has an explicit candidate guard. |
| Likelihood critical points / `critical_points` | `probstats` + `semialg` | Yes | No | Complete zero-dimensional solve when supported | Positive-dimensional critical loci require different machinery. |
| Algebraic MLE / `algebraic_mle` | `probstats` + `semialg` | Yes | No | Exact feasible maximizer over treated faces | Face enumeration can become combinatorial with many zero counts. |
| ML degree / `maximum_likelihood_degree` | `probstats` + `semialg` | Witness counts exact | No | Exact complex counts at deterministic witnesses | Genericity remains uncertified without an exceptional-locus proof. |
| Tensor-rank bounds / `tensor_rank` | TensorAtlas | Yes for supported certificates | Yes | Flattening lower bounds + constructive upper bounds | Exact higher-order rank can remain unresolved. |
| CP decomposition / `cp_decompose` | TensorAtlas | Matrix/rank-one cases | Yes | Exact reconstruction or numerical residual | Numerical convergence is not an exact rank certificate. |
| Nonnegative CP | TensorAtlas | Limited | Yes | Nonnegative constructive fit | Non-convex; restart/conditioning sensitivity remains. |
| Generic CP uniqueness / `generic_cp_identifiability` | TensorAtlas | Yes | No | Kruskal sufficient criterion | Criterion failure is inconclusive, not negative. |
| Latent image dimension | `probstats` + TensorAtlas | Yes when witness reaches rigorous upper bound | No | Exact Jacobian minor/rank certificate | Potential defective cases can remain unsupported. |
| Generic latent identifiability | `probstats` + TensorAtlas | Yes when dimension + uniqueness certify | No | Fiber dimension + CP uniqueness | Reported up to hidden-label permutations. |
| Pointwise latent identifiability | `probstats` | Yes | No | Exact Jacobian rank at supplied point | Full local rank does not prove global uniqueness. |
| Exact latent image ideal / `image_ideal` | `probstats` + `semialg` | Yes | No | Elimination ideal | Explicit parameter-count guard limits expensive elimination. |
| Moment / cumulant tensors | `probstats` | Yes for exact samples/weights | Yes through numeric inputs | Exact empirical tensor entries | Exact cumulants have combinatorial set-partition growth. |
| Multi-view mixture recovery | `probstats` + TensorAtlas | Rank-one special cases | Yes | Reconstruction + optional Kruskal uniqueness | Numerical CP scaling/permutation must be normalized statistically. |

## Reading an unresolved result

An unresolved result is still useful. Prefer inspecting all evidence fields rather than converting it immediately to a Boolean.

For tensor rank:

```python
result = tensor_rank(tensor, method="exact")
if result.rank is None:
    print(result.lower_bound, result.upper_bound)
```

For CP uniqueness:

```python
result = generic_cp_identifiability(shape, rank)
if result.identifiable is None:
    # The current sufficient criterion does not settle the case.
    ...
```

For latent identifiability:

```python
result = generic_identifiability(model)
print(result.fiber_dimension)
print(result.identifiable)
print(result.certified)
```

The result object is designed to preserve the strongest statement justified by the computation without silently promoting bounds, numerical evidence, or failed sufficient criteria into theorems.

## Tests that protect these semantics

The algebraic-statistics test suite contains three complementary layers:

1. **Cross-package theorem tests** verify that independent representations agree across package boundaries.
2. **Certificate adversarial tests** verify that incomplete evidence stays incomplete.
3. **Metamorphic and degeneracy tests** verify invariance under harmless transformations and correct behavior at singular/boundary configurations.

Together these tests make the evidence fields part of the public mathematical contract rather than incidental metadata.
