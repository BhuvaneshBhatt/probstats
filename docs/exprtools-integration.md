# Exprtools integration (stage 6)

Shared symbolic predicates and Bayesian certification prefer exprtools. `exprtools_loader` reaches exact normalizers and Laplace Hessians. Legacy `exprtest_loader` and `use_exprtest` keywords remain supported; an explicit legacy loader retains its backend label. Supplying both loaders raises ValueError. An explicit `use_exprtools` value takes precedence over `use_exprtest`.

The new package is developed in the `exprtools/` subproject of
[BhuvaneshBhatt/exprtest](https://github.com/BhuvaneshBhatt/exprtest/pull/1).
The legacy exprtest distribution is a compatibility wrapper sharing the same
implementations and proof caches; new runtime imports use exprtools.

## Validation

Against baseline `998ac749b599d147bad5d88f24fa670d3fdbc511`, the selected downstream suite
passed 48 tests before import changes and 57
after changes and integration regressions. Tests were run on Python 3.12.14 /
SymPy 1.14.0 / python-flint 0.9.0. These are focused migration suites, not a claim
that this project's entire test suite or every optional integration was run.

Selected files:

- `bayes/test_certification.py`
- `bayes/test_exact.py`
- `test_algebraic_semialgebraic_testing.py`
- `test_failure_semantics.py`
- `test_exact_linear_algebra.py`
- `test_exprtools_integration.py`

A clean wheel environment containing exprtools, asymptotic, semialg, funcprops,
odeanalysis and probstats was tested outside all checkouts with exprtest absent
and its import actively blocked. It checks package provenance, dependency
metadata, certified nested radicals, independent refinement, domain holes,
parameter uncertainty, recurrence evidence, and exact posterior normalization.
A second smoke run installs the compatibility wrapper and checks shared classes,
functions and proof caches. Build artifacts and raw test results accompany the
migration report in the originating task.

## Release and development

No distributions were published. Before releasing this change, publish exprtools
first, then semialg and funcprops, then their consumers; choose new downstream
release versions at release time. The dependency name is `exprtools>=0.1.0`.
CI installs the immutable exprtools source commit
`b6c38bd2b5fc27d8f88e30026d52868050308e18` pending its first release.
The old exprtest wheel is optional for consumers that still use the legacy API.
