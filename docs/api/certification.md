# Expression certification API

`probstats.bayes` uses the `exprtest` package as a proof-oriented
zero/nonzero oracle. Certification is tri-state: heuristic or
probable evidence is never promoted to a proof.

`exprtest` is a core dependency because symbolic equality and zero decisions across
the package use the same certification boundary. The `certify` extra adds the
semialgebraic reasoning backend used by higher-level certification features.

## `certify_zero`

```text
certify_zero(expr, *, use_exprtest=True)
```

Attempts to prove that `expr` is zero or nonzero. `exprtest` is a core dependency;
`probstats.bayes` calls:

```python
exprtest.zerotest(expr, confidence="certified")
```

so `False` means *proved nonzero*. The default exprtest `"probable"` mode is
not used for mathematical certification. If exprtest returns `None`, native
SymPy exact zero predicates are tried; unresolved cases remain `UNKNOWN`.

## `certify_equal`

```text
certify_equal(left, right, *, use_exprtest=True)
```

Certifies equality by applying `certify_zero(left - right)`.

## `ExpressionCertificate`

Immutable result containing:

- `expression`: simplified expression tested;
- `status`: `ZERO`, `NONZERO`, or `UNKNOWN`;
- `proven`: whether the conclusion is a proof;
- `backend`: `"exprtest"` or fallback `"sympy"`;
- `method` and `reason`: audit metadata;
- `raw`: optional backend result.

The `is_zero` property returns `True` or `False` only for proved conclusions and
`None` otherwise.

## `CertificationStatus`

Enumeration with values `ZERO`, `NONZERO`, and `UNKNOWN`.

## Use in exact inference

Exact inference certifies every computed posterior normalizer after checking for infinities
and NaNs. `infer_exact()` stores the result in:

```python
result.metadata["normalizer_certificate"]
result.metadata["normalizer_certified_nonzero"]
```

A proved-zero normalizer is rejected. An unresolved nonzeroness certificate is
retained as provenance rather than silently reported as proof.

## Use in Laplace inference

Laplace inference applies the same oracle to the exact symbolic determinant of the negative
Hessian evaluated at the MAP. The result is exposed as:

```python
result.diagnostics["precision_determinant_certificate"]
result.diagnostics["precision_determinant_certified_zero"]
result.diagnostics["precision_determinant_certified_nonzero"]
```

A proved zero determinant gives exact evidence of singular curvature and routes
to singular Laplace when an `AsymptoticCorrector` is supplied. A proved
nonzero determinant prevents a tiny-but-real curvature direction from being
classified as singular solely because of a floating threshold, provided no
negative curvature was detected.

## Use in quality assessment

`assess_quality()` consumes exact-inference and Laplace certificates. Exact results report whether the
normalizer has independently certified nonzeroness; Laplace quality reports
whether precision singularity/nonsingularity was certified. These certificates
are supporting evidence only: they do not turn a local Laplace approximation
into a globally exact posterior.
