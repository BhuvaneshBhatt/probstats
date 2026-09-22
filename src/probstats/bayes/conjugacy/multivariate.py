"""Normal-Inverse-Wishart conjugacy for multivariate normal observations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import sympy as sp

from ..._exact_linear_algebra import exact_solve, quadratic_form
from ..distributions.student_t import MultivariateStudentT


@dataclass(frozen=True, slots=True)
class NormalInverseWishart:
    mu: sp.ImmutableDenseMatrix
    lambda_: sp.Expr
    psi: sp.ImmutableDenseMatrix
    nu: sp.Expr

    def __init__(self, mu, lambda_, psi, nu):
        m = sp.ImmutableMatrix(mu)
        p = sp.ImmutableMatrix(psi)
        if m.cols != 1:
            m = sp.ImmutableMatrix(list(mu))
        if p.rows != p.cols or p.rows != m.rows:
            raise ValueError("NIW dimensions are incompatible")
        object.__setattr__(self, "mu", m)
        object.__setattr__(self, "lambda_", sp.sympify(lambda_))
        object.__setattr__(self, "psi", p)
        object.__setattr__(self, "nu", sp.sympify(nu))

    @property
    def dimension(self):
        return self.mu.rows

    @property
    def parameter_constraints(self):
        return sp.And(self.lambda_ > 0, self.nu > self.dimension - 1)

    @property
    def mean_marginal(self):
        df = sp.simplify(self.nu - self.dimension + 1)
        return MultivariateStudentT(self.mu, self.psi / (self.lambda_ * df), df)

    @property
    def predictive(self):
        df = sp.simplify(self.nu - self.dimension + 1)
        return MultivariateStudentT(
            self.mu, (self.lambda_ + 1) * self.psi / (self.lambda_ * df), df
        )


def default_multivariate_normal_prior(dimension: int):
    if dimension < 1:
        raise ValueError("dimension must be positive")
    return NormalInverseWishart(
        [0] * dimension,
        sp.Rational(1, 100),
        sp.eye(dimension) / 100,
        dimension - 1 + sp.Rational(1, 100),
    )


def update_multivariate_normal(
    data: Iterable[Iterable], prior: NormalInverseWishart | None = None
):
    rows = [sp.ImmutableMatrix(list(row)) for row in data]
    if not rows:
        if prior is None:
            raise ValueError(
                "dimension cannot be inferred from empty data without a prior"
            )
        return prior
    d = rows[0].rows
    if any(r.rows != d or r.cols != 1 for r in rows):
        raise ValueError("all observations must have equal dimension")
    prior = prior or default_multivariate_normal_prior(d)
    if prior.dimension != d:
        raise ValueError("prior dimension does not match observations")
    n = len(rows)
    mean = sum(rows, sp.zeros(d, 1)) / n
    scatter = sp.zeros(d, d)
    for r in rows:
        q = r - mean
        scatter += q * q.T
    lam = sp.simplify(prior.lambda_ + n)
    mu = (prior.lambda_ * prior.mu + n * mean) / lam
    diff = mean - prior.mu
    psi = prior.psi + scatter + (prior.lambda_ * n / lam) * (diff * diff.T)
    return NormalInverseWishart(mu, lam, psi, sp.simplify(prior.nu + n))


def _multigamma_log(a, dimension: int):
    return sp.Rational(dimension * (dimension - 1), 4) * sp.log(sp.pi) + sp.Add(
        *(sp.loggamma(a + sp.Rational(1 - j, 2)) for j in range(1, dimension + 1))
    )


def _multidigamma(a, dimension: int):
    return sp.Add(
        *(sp.digamma(a + sp.Rational(1 - j, 2)) for j in range(1, dimension + 1))
    )


def _register_information_formulas():
    # Import here to avoid a Bayesian/probability initialization cycle.
    from ...information_registry import register_information_formula

    @register_information_formula("kl", NormalInverseWishart, NormalInverseWishart)
    def _niw_kl(p: NormalInverseWishart, q: NormalInverseWishart):
        if p.dimension != q.dimension:
            return sp.oo
        d = p.dimension
        vp, vq = p.nu, q.nu
        # KL between the inverse-Wishart marginals.
        iw = sp.simplify(
            vq / 2 * sp.log(p.psi.det() / q.psi.det())
            + vp / 2 * (sp.trace(exact_solve(p.psi, q.psi)) - d)
            + _multigamma_log(vq / 2, d)
            - _multigamma_log(vp / 2, d)
            + (vp - vq) / 2 * _multidigamma(vp / 2, d)
        )
        delta = p.mu - q.mu
        conditional = sp.simplify(
            sp.Rational(1, 2)
            * (
                d * (q.lambda_ / p.lambda_ - 1 + sp.log(p.lambda_ / q.lambda_))
                + q.lambda_ * p.nu * quadratic_form(p.psi, delta)
            )
        )
        return sp.simplify(iw + conditional)

    @register_information_formula("entropy", NormalInverseWishart, NormalInverseWishart)
    def _niw_entropy(p: NormalInverseWishart, _q: NormalInverseWishart):
        d, v = p.dimension, p.nu
        elogdet = sp.log(p.psi.det()) - d * sp.log(2) - _multidigamma(v / 2, d)
        iw_entropy = sp.simplify(
            -v / 2 * sp.log(p.psi.det())
            + (v + d + 1) / 2 * elogdet
            + v * d / 2 * sp.log(2)
            + _multigamma_log(v / 2, d)
            + v * d / 2
        )
        normal_entropy = sp.simplify(
            sp.Rational(d, 2) * (1 + sp.log(2 * sp.pi) - sp.log(p.lambda_))
            + elogdet / 2
        )
        return sp.simplify(iw_entropy + normal_entropy)

    @register_information_formula("renyi", NormalInverseWishart, NormalInverseWishart)
    def _niw_renyi(p: NormalInverseWishart, q: NormalInverseWishart, alpha):
        a = sp.sympify(alpha)
        if a.is_positive is False or a == 0 or a == 1:
            raise ValueError(
                "closed-form Rényi formula requires alpha > 0 and alpha != 1"
            )
        if p.dimension != q.dimension:
            return sp.oo
        d = p.dimension
        kr = sp.simplify(a * p.lambda_ + (1 - a) * q.lambda_)
        vr = sp.simplify(a * p.nu + (1 - a) * q.nu)
        if kr.is_positive is False or (vr - (d - 1)).is_positive is False:
            return sp.oo
        delta = p.mu - q.mu
        correction = sp.simplify(
            a * (1 - a) * p.lambda_ * q.lambda_ / kr * (delta * delta.T)
        )
        psi_r = sp.simplify(a * p.psi + (1 - a) * q.psi + correction)
        if psi_r.is_positive_definite is False:
            return sp.oo
        log_integral = sp.simplify(
            a * (p.nu / 2 * sp.log(p.psi.det()) - _multigamma_log(p.nu / 2, d))
            + (1 - a) * (q.nu / 2 * sp.log(q.psi.det()) - _multigamma_log(q.nu / 2, d))
            - vr / 2 * sp.log(psi_r.det())
            + _multigamma_log(vr / 2, d)
            + sp.Rational(d, 2)
            * (a * sp.log(p.lambda_) + (1 - a) * sp.log(q.lambda_) - sp.log(kr))
        )
        return sp.simplify(log_integral / (a - 1))


_register_information_formulas()
