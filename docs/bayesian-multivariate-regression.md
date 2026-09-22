# Multivariate Bayesian linear regression

`BayesianMultivariateLinearRegression` fits a Gaussian multivariate-response
linear model with a Matrix-Normal/Inverse-Wishart (MNIW) conjugate prior.

For an `n x p` design matrix `X` and `n x m` response matrix `Y`, the model is

\[
Y\mid B,\Sigma \sim MN(XB, I_n, \Sigma),
\]

with prior

\[
B\mid\Sigma \sim MN(M_0,\Lambda_0^{-1},\Sigma),\qquad
\Sigma\sim IW(\Psi_0,\nu_0).
\]

The posterior remains MNIW:

\[
\Lambda_n=\Lambda_0+X^T X,
\qquad
M_n=\Lambda_n^{-1}(\Lambda_0M_0+X^TY),
\]

\[
\Psi_n=\Psi_0+Y^TY+M_0^T\Lambda_0M_0-M_n^T\Lambda_nM_n,
\qquad
\nu_n=\nu_0+n.
\]

## Basic use

```python
from probstats.bayes import BayesianMultivariateLinearRegression
from probstats.bayes.models import MatrixNormalInverseWishartPrior

prior = MatrixNormalInverseWishartPrior(
    coef_mean=[[0, 0], [0, 0]],
    precision=[[1, 0], [0, 1]],
    scale=[[1, 0], [0, 1]],
    df=3,
)
model = BayesianMultivariateLinearRegression(prior=prior)
fit = model.fit(
    [[1, 0], [1, 1], [1, 2]],
    [[1, 2], [2, 2], [4, 3]],
)
```

`fit.posterior` contains the exact updated MNIW parameters.  The covariance
marginal is an `InverseWishart`, while `fit.coefficient_distribution` is the
matrix-t marginal induced by integrating out `Sigma`.

For ordinary vector-valued marginals, use
`fit.coefficient_row_marginal(i)` for one coefficient row across responses or
`fit.response_coefficient_marginal(j)` for the coefficient vector associated
with one response column.  These return `MultivariateStudentT` distributions.

## Prediction

For a new design row `x`, integrating both `B` and `Sigma` gives a multivariate
Student-t predictive distribution.  If `m` is the response dimension, its
degrees of freedom are

\[
\nu_n-m+1,
\]

and its scale matrix is

\[
\frac{1+x^T\Lambda_n^{-1}x}{\nu_n-m+1}\Psi_n.
\]

```python
predictive = fit.predictive([1, 3])
latent_mean = fit.predictive([1, 3], observation=False)
joint = fit.joint_predictive([[1, 3], [1, 4]])
```

`joint_predictive` returns a matrix-t law and preserves posterior dependence
between multiple future response rows; `predict(...)` remains a convenient
rowwise tuple of multivariate Student-t marginals.

The `observation=False` form removes the observation-noise term and therefore
returns the posterior law of the latent regression mean at that design row.

## Exact evidence

The marginal likelihood is analytic:

\[
\log p(Y\mid X)=
-\frac{nm}{2}\log\pi
+\frac{m}{2}(\log|\Lambda_0|-\log|\Lambda_n|)
+\frac{\nu_0}{2}\log|\Psi_0|
-\frac{\nu_n}{2}\log|\Psi_n|
+\log\Gamma_m(\nu_n/2)-\log\Gamma_m(\nu_0/2).
\]

It is available as `fit.log_evidence` or `fit.evidence`.

## Sequential updates

`fit.update(X_new, Y_new)` performs another exact conjugate update using the
current posterior as the next prior.  Posterior parameters and cumulative log
evidence are algebraically identical to fitting the concatenated batch at once.
This makes the model suitable for online or chunked Bayesian updating without
retaining an approximation to the posterior.

## Planner integration

`plan_inference(model, x=X, y=Y)` recognizes the model as an analytic MNIW
regression problem and selects `analytic-multivariate-linear-regression` before
generic symbolic, Laplace, MCMC, or nested-sampling routes.
