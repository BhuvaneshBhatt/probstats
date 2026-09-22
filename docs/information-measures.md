# Information measures

`probstats` provides exact-first information-theoretic measures in the general
probability layer. They work with `Distribution` objects and use the same
symbolic-first philosophy as the rest of the package.

```python
from probstats import Normal
from probstats.information import (
    kl_divergence,
    renyi_divergence,
)

p = Normal(0, 1)
q = Normal(1, 2)

kl_divergence(p, q)
renyi_divergence(p, q, 2)
```

The public measures are:

- `information_entropy(p)`;
- `cross_entropy(p, q)`;
- `kl_divergence(p, q);
- `renyi_divergence(p, q, alpha)`;
- `jensen_shannon_divergence(p, q, weight=1/2)`;
- `bhattacharyya_coefficient(p, q)`;
- `bhattacharyya_distance(p, q)`;
- `hellinger_squared(p, q)` and `hellinger_distance(p, q)`.

The same operations are available as convenience methods on distributions,
for example `p.kl_divergence(q)` and `p.hellinger_distance(q)`.

## Support and absolute continuity

Relative information is meaningful only with the appropriate support and
measure relationship. `support_subset(p, q)` therefore returns a three-valued
`SupportRelation`:

- `PROVEN`: containment was established;
- `DISPROVEN`: containment was disproved;
- `UNKNOWN`: the symbolic set engine could establish neither result.

For KL divergence and cross-entropy, a disproven `support(p) ⊆ support(q)`
returns `oo`. Discrete and absolutely-continuous laws are also treated as
measure-incompatible rather than merely comparing their set-valued supports.
An `UNKNOWN` support relation is not silently promoted to `PROVEN`; symbolic
calculation may continue and the uncertainty is retained in `InformationResult`.

This keeps failed and unknown support proofs explicit rather than conflating
them.

## Exact and numerical evaluation

The default path constructs the defining sum or integral symbolically.
Multivariate continuous information integrals over product intervals use the
package's shared structured integration router: `multiple-integrate` is tried
when the `exact` extra is installed, with native SymPy integration as the
fallback. For example,

\[
D_{KL}(P\|Q)=E_P[\log p(X)-\log q(X)].
\]

If symbolic evaluation is unavailable, pass `numerical_fallback=True` for a
Monte Carlo estimate using samples from the appropriate source distribution.
Numerical log-density functions are compiled once with `sympy.lambdify`, so
Monte Carlo evaluation does not repeatedly invoke symbolic dispatch.

```python
from probstats.information import jensen_shannon_divergence

js = jensen_shannon_divergence(
    Normal(0, 1),
    Normal(1, 1),
    numerical_fallback=True,
    samples=50_000,
    rng=1234,
)
```

Use `return_result=True` to inspect the evaluation method, support status, and
whether the result is exact.

## Identities

The implementations are tested against the standard relationships

\[
H(P,Q)=H(P)+D_{KL}(P\|Q),
\]

\[
D_{1/2}(P\|Q)=-2\log BC(P,Q),
\]

and

\[
H^2(P,Q)=1-BC(P,Q).
\]

Jensen-Shannon divergence uses support-aware mixture densities and therefore
remains finite even when the two distributions do not share identical
supports.

## Closed-form registry

Common distribution pairs are dispatched through a closed-form registry before
the generic symbolic integration engine.  The built-in registry covers Normal,
MultivariateNormal, Gamma, Beta, and Dirichlet pairs for KL divergence and
Rényi divergence, and provides corresponding entropy formulas.  Cross-entropy
then uses `H(P, Q) = H(P) + KL(P || Q)`, while the Bhattacharyya and Hellinger
measures reuse the Rényi order-one-half identity.

Use `registered_information_formulas()` to inspect the active registry.  Third-
party distributions can extend it with `register_information_formula(measure,
p_type, q_type)`; registered formulas should return exact SymPy expressions and
must respect the distributions' parameter/support contracts.  Jensen-Shannon
divergence remains on the generic symbolic/numerical path because
most common continuous pairs, including unequal Normals, do not have a useful
elementary closed form.


## Built-in closed forms

The registry recognizes KL and Rényi formulas for Bernoulli, Categorical,
Poisson, Exponential, LogNormal, MatrixNormal, Wishart, and InverseWishart pairs
in addition to the core Normal, MultivariateNormal, Gamma, Beta, and Dirichlet
families. Entropy formulas are available for Bernoulli, Categorical,
Exponential, LogNormal, StudentT, MatrixNormal, Wishart, and InverseWishart.

LogNormal divergence is evaluated through invariance under the common logarithm
transformation.  MatrixNormal formulas use the covariance structure of the
vectorized matrix without replacing the public matrix-valued distribution.
Wishart and InverseWishart formulas use multivariate gamma and digamma terms and
check the mixed parameters required for finite Rényi integrals.

General Student-t pair divergences remain on the generic symbolic or
numerical path: unlike entropy, arbitrary Student-t KL and Rényi divergences do
not have a single simple closed form across unequal locations, scales, and
degrees of freedom.  Jensen-Shannon likewise continues to use the generic path.


Matching-trial-count `Binomial` and
`Multinomial` pairs. Their KL and Rényi divergences reduce exactly to `n` times
the corresponding Bernoulli or Categorical divergence. If the trial counts
differ, the registry declines the shortcut and the generic
support-aware evaluator is used instead.

`probstats.bayes.NormalInverseWishart` participates in the same registry even
though it is a conjugate joint law rather than a scalar/vector `Distribution`.
For NIW pairs, closed forms are available for KL divergence, Shannon entropy,
cross-entropy through `H(P,Q)=H(P)+KL(P||Q)`, Rényi divergence, and therefore
Bhattacharyya/Hellinger measures through the order-1/2 Rényi identity. The NIW
formula integrates the inverse-Wishart covariance marginal together with the
conditional multivariate Normal mean law. Non-`Distribution` objects are only
accepted by information functions when an explicit closed-form registry entry
exists.

## Certified transformation identities

When the optional `reasoning` dependencies are installed, information measures
reuse `funcprops` transformation certificates rather than trusting caller-declared
invertibility.

For a scalar common pushforward `g`, KL divergence applies

\[
D_{KL}(g_\#P\|g_\#Q)=D_{KL}(P\|Q)
\]

only after `funcprops.function_bijective` certifies one common map on the union
of the two base supports. If certification is false or unknown, ordinary KL
evaluation continues without the invariance shortcut.

For a continuous scalar `TransformedDistribution`, differential entropy applies

\[
h(g(X))=h(X)+E[\log |g'(X)|]
\]

only when `funcprops` certifies the map as globally bijective on the base support
and `probstats` verifies that the scalar derivative is globally nonzero. The exact
absolute derivative is then used as the change-of-variables factor. Discrete Shannon entropy uses the
corresponding bijection invariance without a Jacobian term. Unsupported,
non-bijective, or uncertified transformations fall back to the ordinary entropy
machinery rather than assuming the theorem hypotheses.
