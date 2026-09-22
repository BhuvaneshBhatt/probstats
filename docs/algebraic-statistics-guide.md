# Algebraic Statistics with probstats + semialg + TensorAtlas

This page is the **worked tutorial** for the three-package algebraic-statistics stack. It follows complete statistical workflows from data to model geometry, inference, identifiability, and moment recovery. For concise API/input details use the [algebraic-statistics reference](algebraic-statistics.md); for evidence-strength semantics use the [certificate reference](algebraic-statistics-certificates.md).

Algebraic statistics studies statistical models through polynomial equations, algebraic varieties, tensor geometry, and exact discrete structure. This stack separates those responsibilities across three packages:

- **`probstats`** owns statistical meaning: probability coordinates, contingency tables, sufficient statistics, likelihoods, latent classes, identifiability interpretation, empirical moments, and mixture recovery.
- **`semialg`** owns generic polynomial and semialgebraic computation: ideals, elimination, saturation, exact solving, dimension and degree, feasibility, and toric lattice calculations.
- **TensorAtlas (`tensoratlas`)** owns generic tensor geometry: flattenings, Segre and Veronese varieties, secants, CP/Waring decomposition, tensor-rank bounds, and Kruskal uniqueness certificates.

The separation matters because the same polynomial or tensor operation can have different statistical interpretations. TensorAtlas can prove that a tensor decomposition is essentially unique; `probstats` decides whether that uniqueness means a latent statistical model is identifiable up to hidden-label permutations. `semialg` can eliminate variables from a polynomial graph; `probstats` decides which eliminated coordinates are probabilities and which parameter values form a valid statistical model.

## Installing the layers

Use the smallest dependency set needed by the calculation:

```bash
pip install "probstats[algebraic]"        # semialg-backed exact algebra
pip install "probstats[tensor]"           # TensorAtlas tensor operations
pip install "probstats[algebraic-tensor]" # both
```

The optional packages are imported lazily. Importing `probstats.algebraic` alone does not eagerly load either backend.

## 1. Independence as a statistical model and a tensor variety

For two binary variables, complete independence says

```text
p00*p11 - p01*p10 = 0.
```

In `probstats`, this is a probability model:

```python
from probstats.algebraic import independent_model

model = independent_model((2, 2), variables=("X", "Y"))
model.probabilities
model.equations
model.region
```

TensorAtlas supplies the generic Segre geometry underlying those equations. `probstats` adds normalization, nonnegativity, variable names, and the statistical interpretation of rank one as independence.

The same binary-independence model also has a toric description. A homogeneous design matrix is

```python
A = (
    (1, 1, 1, 1),
    (1, 1, 0, 0),
    (1, 0, 1, 0),
)
```

and `semialg.algebraic.toric_ideal(A, probabilities)` produces the same determinant ideal as the Segre construction. This equivalence is tested directly by the cross-package theorem suite rather than assumed from the two implementations.

## 2. Toric fibers and exact conditional inference

A toric model turns a count table `x` into a sufficient statistic `A*x`:

```python
from probstats.algebraic import ToricModel

model = ToricModel(A)
observed = (1, 3, 3, 1)
model.sufficient_statistic(observed)
model.fiber(observed)
model.markov_basis()
```

The fiber is

```text
F_b = {x in N^n : A*x = b}.
```

The Markov basis comes from `semialg`; `probstats` uses it as a move set for the conditional statistical problem. A valid move must preserve `A*x`, and the integration tests verify that property with the actual basis returned by the algebra backend.

For exact conditional inference:

```python
from probstats.algebraic import conditional_test

result = conditional_test(observed, model)
result.pvalue
result.fiber_size
result.exact
```

Conditioning removes the toric parameters. The probability of a table on the fiber is proportional to `1 / prod(x_i!)`, so exact enumeration gives an exact rational p-value.

## 3. Likelihood geometry

For an algebraic probability model and count vector, `probstats` constructs logarithmic likelihood critical equations while `semialg` performs the generic algebraic operations:

```python
from probstats.algebraic import (
    algebraic_mle,
    critical_points,
    likelihood_equations,
    maximum_likelihood_degree,
)

system = likelihood_equations(model, (2, 3, 5, 7))
critical = critical_points(model, (2, 3, 5, 7))
mle = algebraic_mle(model, (2, 3, 5, 7))
ml_degree = maximum_likelihood_degree(model)
```

The score equations are polynomialized with Lagrange multipliers, the multipliers are eliminated, and the critical ideal is saturated by the active probability coordinates. Saturation removes coordinate-hyperplane artifacts introduced by polynomialization.

`algebraic_mle` is not merely "choose the largest interior critical point." Zero observed counts can place an MLE on a coordinate face, so relevant probability-simplex faces are solved exactly as well.

### ML-degree evidence versus proof

`MLDegreeResult` distinguishes two claims:

```python
ml_degree.exact
ml_degree.complete
ml_degree.generic
ml_degree.generic_certified
```

The current implementation counts complex critical points exactly at deterministic generic-data witnesses. Agreement across witnesses makes `generic=True`, but without a symbolic exceptional-locus argument it leaves `generic_certified=False`.

That distinction is a core library contract: **an exact computation at generic witnesses is not automatically a proof that the witnesses avoid every exceptional locus**.

## 4. Tensor rank: bounds, decompositions, and certificates

Tensor rank is not treated as a scalar when only bounds are known:

```python
from tensoratlas.algebraic import tensor_rank

rank = tensor_rank(tensor, method="exact")
rank.rank
rank.lower_bound
rank.upper_bound
rank.exact
rank.certificates
```

Matrix flattenings give rigorous lower bounds, while constructive decompositions give upper bounds. If the bounds do not meet, `rank.rank` remains `None`.

A numerical CP fit can tighten an upper bound, but it remains numerical:

```python
from tensoratlas.algebraic import cp_decompose

fit = cp_decompose(
    tensor,
    rank=2,
    method="numerical",
    nonnegative=True,
    rng=1,
)
fit.converged
fit.residual
fit.exact
```

Even when the residual is tiny, `fit.exact` is false. A numerical reconstruction and an exact rank certificate are different mathematical statements.

### Necessary versus sufficient rank equations

Flattening minors characterize rank one through the Segre variety, but higher-rank flattening minors are generally only necessary conditions. TensorAtlas makes this visible:

```python
from tensoratlas.algebraic import tensor_rank_constraints

constraints = tensor_rank_constraints((3, 3, 3), max_rank=2)
constraints.necessary
constraints.sufficient
```

The adversarial tests explicitly assert that higher-rank flattening constraints are not mislabeled sufficient.

## 5. Latent classes as secant models

A finite latent-class model is a mixture of product distributions:

```text
P(X1,...,Xd) = sum_h lambda_h prod_j P(Xj | H=h).
```

Construct one with:

```python
from probstats.algebraic import latent_class_model

latent = latent_class_model(
    (2, 2, 2),
    2,
    variables=("X", "Y", "Z"),
)
```

The statistical parameterization removes simplex redundancies before algebraic analysis. TensorAtlas supplies the corresponding secant geometry:

```python
from tensoratlas.algebraic import (
    generic_cp_identifiability,
    secant_expected_dimension,
)

secant_expected_dimension((2, 2, 2), 2)
generic_cp_identifiability((2, 2, 2), 2)
```

For the two-class three-binary model, both the parameter dimension and generic image dimension are seven. The generic fiber is therefore zero-dimensional.

Zero-dimensionality alone does **not** prove uniqueness. `probstats.generic_identifiability()` combines the generic-fiber calculation with TensorAtlas's Kruskal essential-uniqueness certificate:

```python
from probstats.algebraic import generic_identifiability

ident = generic_identifiability(latent)
ident.identifiable
ident.fiber_dimension
ident.up_to_label_swapping
ident.expected_label_orbit
ident.certified
```

For two latent classes, swapping the two hidden labels gives the same observed distribution. A certified two-element label orbit is therefore treated as statistical identifiability **up to label switching**, not as a failure of identifiability.

### Failure of a sufficient criterion is not a negative theorem

Kruskal's inequality is sufficient, not necessary. Consequently:

```python
from tensoratlas.algebraic import generic_cp_identifiability

result = generic_cp_identifiability((2, 2, 2), 3)
assert result.identifiable is None
```

`None` means "the available theorem does not decide this case." It does not mean the model is non-identifiable.

By contrast, a positive-dimensional generic parameter fiber is a genuine non-identifiability certificate:

```python
from probstats.algebraic import generic_identifiability, latent_class_model

result = generic_identifiability(latent_class_model((2, 2), 2))
assert result.identifiable is False
assert result.certified is True
```

## 6. Pointwise versus generic identifiability

Generic identifiability describes a dense generic part of parameter space. Particular parameter values can still be singular, for example when latent components coincide or a mixing weight degenerates.

```python
from probstats.algebraic import identifiability

pointwise = identifiability(latent, parameter_values)
pointwise.generic
pointwise.locally_finite_to_one
pointwise.fiber_dimension
```

A rank-deficient pointwise Jacobian certifies local non-identifiability. Conversely, a full-rank pointwise Jacobian certifies local finite-to-one behavior, not global uniqueness.

The test suite keeps these claims separate so a generic theorem cannot silently overwrite a singular pointwise result.

## 7. Moments and multi-view tensor statistics

For vector observations, `probstats` builds exact empirical tensor statistics:

```python
from probstats.algebraic import (
    central_moment_tensor,
    cumulant_tensor,
    moment_tensor,
)

samples = ((1, 2), (3, 4))
raw = moment_tensor(samples, 2)
central = central_moment_tensor(samples, 2)
cumulant = cumulant_tensor(samples, 3)
```

`MomentTensor` keeps the statistical metadata and exact SymPy entries. TensorAtlas is invoked lazily only when tensor geometry or decomposition is needed.

For aligned categorical views:

```python
from probstats.algebraic import categorical_multi_view_moment

cross = categorical_multi_view_moment(
    (
        (0, 0, 1, 1),
        (0, 1, 0, 1),
    ),
    cardinalities=(2, 2),
)
```

The one-hot cross moment is exactly the empirical joint-probability tensor.

## 8. Recovering a multi-view mixture

Under conditional independence given a latent class, a multi-view probability tensor has CP form

```text
M = sum_h lambda_h * a_h^(1) tensor ... tensor a_h^(d).
```

`probstats` delegates CP mechanics to TensorAtlas and then resolves CP scaling into statistical normalization:

```python
from probstats.algebraic import recover_multiview_mixture

recovered = recover_multiview_mixture(
    probability_tensor,
    components=2,
    method="numerical",
    rng=1,
)
recovered.weights
recovered.component_means
recovered.identifiable
recovered.certified_identifiable
```

For probability mixtures the numerical TensorAtlas path uses nonnegative CP. `probstats` then normalizes each recovered factor to a probability vector and transfers its scale into the mixture weight.

The input is required to be a nonnegative tensor summing to one. An arbitrary feature-moment tensor should instead be passed to `decompose_moment_tensor`; stochastic normalization would have no valid interpretation there.

## 9. Reading result objects: the certificate vocabulary

The packages use structured results because several notions that sound similar are not interchangeable.

| Field or state | Meaning |
| --- | --- |
| `exact=True` | The reported algebraic/numeric quantity was obtained exactly, not by floating approximation. |
| `complete=True` | The algorithm certifies that the requested finite solution/result set is complete for its stated problem. |
| `certified=True` | A mathematical certificate/theorem establishes the stated conclusion. |
| `generic=True` | The statement concerns generic parameter/data values, not every point. |
| `generic_certified=True` | Genericity itself has been certified, rather than inferred from stable witnesses. |
| `identifiable=True` | The implemented criteria establish identifiability under the result's stated equivalences. |
| `identifiable=False` | The implementation has a certificate of non-identifiability. |
| `identifiable=None` | Available criteria are inconclusive. |
| `rank=None` | Exact tensor rank has not been certified; inspect lower and upper bounds. |
| small numerical residual | A constructive approximation/upper-bound witness, not automatically an exact theorem. |

Two rules summarize the intended semantics:

1. **Do not promote an upper or lower bound to an equality without a certificate.**
2. **Do not turn failure of a sufficient theorem into proof of the opposite statement.**

The adversarial certificate-semantics tests exist specifically to prevent regressions against these rules.

## 10. Which package should receive a new feature?

Use the ownership test below before adding an API:

| Question | Owner |
| --- | --- |
| What probability model or statistical parameter is being represented? | `probstats` |
| What does conditioning, likelihood, MLE, or identifiability mean statistically? | `probstats` |
| How is a polynomial ideal eliminated, saturated, solved, or measured? | `semialg` |
| How are integer kernels, toric ideals, or generic semialgebraic constraints computed? | `semialg` |
| How is a tensor flattened, ranked, decomposed, or represented as a Segre/secant/Veronese object? | TensorAtlas |
| Is a CP decomposition essentially unique as a tensor decomposition? | TensorAtlas |
| Does that tensor uniqueness imply a latent statistical model is identifiable up to labels? | `probstats` |

This boundary keeps the libraries reusable: semialg does not need to know what a contingency table is, TensorAtlas does not need to know what an MLE is, and `probstats` does not duplicate Gröbner or CP algorithms.

## 11. Tested theorem chains

The cross-package theorem suite continuously checks several complete chains rather than isolated functions:

1. **Binary independence:** TensorAtlas Segre equations and semialg toric elimination describe the same ideal used by `probstats`.
2. **Toric inference and likelihood:** semialg Markov moves preserve the `probstats` sufficient statistic, while the same statistical model supports exact likelihood geometry and ML-degree computation.
3. **Latent identifiability:** TensorAtlas secant dimension and Kruskal uniqueness agree with `probstats` generic fiber and label-switching semantics.
4. **Moment recovery:** a `probstats` categorical cross moment passes through TensorAtlas rank certification and back into statistically normalized mixture recovery.

These integration tests are small and exact where possible. Their purpose is not exhaustive algorithm benchmarking; it is to ensure that the mathematical meanings agree at package boundaries.

## 12. When a result is inconclusive

An inconclusive result is a supported outcome, not an error condition. Typical examples include:

- flattening lower and constructive upper tensor-rank bounds do not meet;
- Kruskal's sufficient uniqueness inequality fails;
- an exact generic witness calculation lacks a symbolic genericity certificate;
- a requested elimination exceeds an explicit complexity guard;
- a numerical decomposition does not converge to the requested tolerance.

Inspect the structured result's bounds, method, certificate fields, and reason/notes before deciding whether to request a stronger exact algorithm, use a numerical method, add assumptions, or reformulate the model.
## Certificate reference

For a compact interpretation of every evidence field and a cross-package capability matrix, see [Certificate semantics for algebraic statistics](algebraic-statistics-certificates.md).

