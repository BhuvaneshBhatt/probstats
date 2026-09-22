# General MCMC framework

`probstats.bayes.mcmc` provides resumable log-density-first Markov-chain Monte Carlo for models that do not admit a cheaper exact or analytic route.

## Core objects

`MCMCState` records the current position, log density, transition/acceptance counters, sampler adaptation state, and RNG state. `MCMCChain` stores retained draws and sample statistics and can resume from its terminal state without restarting warmup.

```python
import numpy as np
from probstats.bayes.mcmc import NoUTurnSampler, sample_mcmc

logp = lambda x: -0.5 * np.dot(x, x)
grad = lambda x: -x

sampler = NoUTurnSampler(gradient=grad)
chain = sample_mcmc(logp, [2.0], sampler=sampler, warmup=500, draws=1000, rng=1)
continued = chain.resume(logp, sampler, 500)
```

Adaptation occurs during warmup and is frozen before retained draws. Adaptive Metropolis freezes its learned covariance; HMC and NUTS freeze their dual-averaged step size.

## Samplers

- `MetropolisHastings`: symmetric Gaussian random walk by default, with hooks for asymmetric proposals.
- `IndependentMetropolis`: independent proposal with the full Hastings correction.
- `TransformedMetropolis`: samples in unconstrained coordinates and includes the transformation Jacobian exactly.
- `AdaptiveMetropolis`: Haario-style online covariance learning with the conventional `2.38**2 / d` scale.
- `HamiltonianMonteCarlo`: leapfrog integration, Metropolis correction, mass matrix, warmup step-size adaptation, and divergence reporting.
- `NoUTurnSampler`: dynamic HMC trajectory construction with a no-U-turn stopping rule, warmup step-size adaptation, tree-depth statistics, and divergence reporting.
- `GibbsSampler`: user-supplied exact full-conditional component updates.
- `ComponentMetropolis`: component-wise random-walk fallback for mixed or difficult targets.

All samplers consume an unnormalized **log density**. HMC/NUTS use an analytic gradient when supplied. A finite-difference fallback exists for small numerical problems, but symbolic model inference uses a generated symbolic gradient whenever possible.

## Multiple chains and diagnostics

`run_chains` executes independent chains using child random streams. `diagnose_chains` reports classical split R-hat, rank-normalized/folded R-hat, autocorrelation ESS, bulk ESS, tail ESS, MCSE, per-chain acceptance, and HMC/NUTS divergence counts.

Additional diagnostics are available through `geweke`, `raftery_lewis`, and `heidelberger_welch`. They are primarily research and comparison diagnostics; rank-normalized R-hat, bulk/tail ESS, MCSE, and divergence checks should be preferred for routine decisions.

## Symbolic models and the inference planner

`infer_mcmc_model` compiles a `Model`'s observed joint log posterior to a numerical target. When SymPy can differentiate the joint density, the default sampler is NUTS; otherwise it falls back to Adaptive Metropolis.

The automatic planner considers MCMC when the model has continuous latent variables and the caller supplies explicit `initial_positions`. Exact/conjugate methods retain priority when they are applicable. MCMC can be forced indirectly by planner policy, or called explicitly when posterior sampling is required even though an analytic approximation exists.

## Nested sampling integration

`MCMCConstrainedSampler` is a likelihood-constrained MCMC replacement kernel for nested sampling. It is available **only** with `prior_transform`. It operates in unit-cube coordinates, where the target prior density is uniform, so a constrained MCMC transition preserves the correct prior measure. Using generic posterior-space random-walk MCMC inside nested sampling would not in general preserve the constrained prior and is therefore not supported.
