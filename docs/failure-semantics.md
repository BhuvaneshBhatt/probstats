# Failure semantics

`probstats` distinguishes mathematical invalidity, unsupported operations, optional-backend availability, backend evaluation failure, numerical nonconvergence, and unresolved proof/certification. Callers should not have to infer these states from error-message text.

| State | Representation | Meaning | Typical caller action |
| --- | --- | --- | --- |
| Invalid mathematical input | `ValueError` or a domain-specific `ValueError` subclass | Parameters/probabilities/data violate the mathematical domain | Fix the input; do not retry with another backend |
| Unsupported public operation | `NotImplementedError`, `ProbabilityFunctionalError`, or a documented domain-specific error | The requested operation is outside the implemented mathematical contract | Choose a supported formulation or explicit numerical method |
| Optional backend unavailable | `ImportError` subclass such as `ExactBackendUnavailableError` or `JAXUnavailableError` | The method is valid but an optional dependency/backend is absent | Install the matching extra or select another engine |
| Backend evaluation failure | `RuntimeError` subclass such as `ExactBackendFailure`, `LaplaceApproximationError`, or `EvidenceOptimizationError` | A selected backend was available but could not complete the requested computation | Inspect model geometry/numerics or try another valid engine |
| Numerical nonconvergence | result object with `converged=False` where the API supports recoverable optimization results, otherwise a numerical domain error | An iterative numerical method did not satisfy its convergence contract | Inspect diagnostics, starting values, scaling, or method choice |
| Unresolved symbolic condition/certification | explicit `UNKNOWN`/unproved result state | The system did not prove either direction; this is not evidence of falsehood | Add assumptions, use a stronger backend, or retain the unknown result |

## Design rules

1. Absence of an optional dependency must never masquerade as mathematical invalidity.
2. An available backend that fails must be distinguishable from a backend that is not installed.
3. Unresolved symbolic truth is a value-level three-state result where the API is proof-oriented, not an exception and not `False`.
4. Exact APIs do not silently downgrade to Monte Carlo or floating-point approximation unless that fallback is explicitly requested or documented by the entry point.
5. Public result objects should preserve convergence/certification metadata long enough for callers to audit how a value was obtained.

These distinctions are regression-tested in `tests/test_failure_semantics.py`.

## Parameter constraints

Distribution construction rejects a parameter set when its declared constraints are provably false. If symbolic parameters leave the constraints unresolved, the distribution remains symbolic and carries those constraints forward. This avoids treating “not proved valid” as “proved invalid.”

Internal certified equality/zero decisions use a three-valued `TruthValue`. It raises on implicit Boolean conversion, so `UNKNOWN` cannot accidentally enter ordinary `if not ...` control flow as though it meant false.
