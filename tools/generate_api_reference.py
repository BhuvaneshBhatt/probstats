"""Generate deterministic public-API and distribution-capability references."""

from __future__ import annotations

import inspect
import re
from pathlib import Path
import enum
import probstats

ROOT = Path(__file__).resolve().parents[1]

SCIPY_ORACLE_DISTRIBUTIONS = {
    "Bernoulli",
    "Beta",
    "BetaBinomial",
    "BetaPrime",
    "Binomial",
    "Cauchy",
    "ChiSquared",
    "DiscreteUniform",
    "Exponential",
    "FDistribution",
    "Frechet",
    "Gamma",
    "Geometric",
    "Gumbel",
    "HalfCauchy",
    "HalfNormal",
    "Hypergeometric",
    "InverseGamma",
    "Laplace",
    "LogNormal",
    "Logistic",
    "Maxwell",
    "Nakagami",
    "NegativeBinomial",
    "NoncentralChiSquared",
    "NoncentralF",
    "NoncentralT",
    "Normal",
    "Pareto",
    "Poisson",
    "PowerDistribution",
    "Rayleigh",
    "Rice",
    "Skellam",
    "StudentT",
    "Triangular",
    "Uniform",
    "VonMises",
    "Weibull",
    "Zipf",
}
MULTIVARIATE_DISTRIBUTIONS = {
    "Categorical",  # scalar label but vector parameter; retain scalar event classification below
    "Dirichlet",
    "DirichletMultinomial",
    "InverseWishart",
    "LKJ",
    "LKJCholesky",
    "MatrixNormal",
    "Multinomial",
    "MultivariateNormal",
    "MultivariateStudentT",
    "Wishart",
    "MultivariateKDEDistribution",
    "MultivariateKernelDensityDistribution",
}
ABSTRACT_DISTRIBUTIONS = {"Distribution", "ExponentialFamily"}
WRAPPER_DISTRIBUTIONS = {
    "CensoredDistribution",
    "ProbabilityDistribution",
    "ParameterMixtureDistribution",
    "SplicedDistribution",
    "SymbolicDistribution",
    "HistogramDistribution",
    "KDEDistribution",
    "KernelDensityDistribution",
    "KernelMixtureDistribution",
}


def signature(obj):
    if inspect.isclass(obj) and issubclass(obj, enum.Enum):
        return ""
    try:
        text = str(inspect.signature(obj))
        return re.sub(r"0x[0-9a-fA-F]+", "0x…", text)
    except (TypeError, ValueError):
        return ""


def first_line(obj):
    doc = inspect.getdoc(obj) or ""
    return doc.splitlines()[0] if doc else ""


def api_reference():
    chunks = [
        "# Generated public API reference\n",
        "This file is generated from each namespace's `__all__`. Run `python tools/generate_api_reference.py` after public-API changes.\n",
    ]
    for label, module in [
        ("Package root", probstats),
        ("Algebraic statistics", probstats.algebraic),
        ("Statistics", probstats.stats),
        ("Bayesian inference", probstats.bayes),
        ("Distributions", probstats.distributions),
        ("Probability functionals", probstats.functionals),
        ("Information measures", probstats.information),
        ("Survival", probstats.survival),
    ]:
        chunks += [
            f"\n## {label}\n",
            "| Name | Kind | Signature | Summary |\n",
            "| --- | --- | --- | --- |\n",
        ]
        for name in module.__all__:
            obj = getattr(module, name)
            kind = (
                "module"
                if inspect.ismodule(obj)
                else "class"
                if inspect.isclass(obj)
                else "function"
                if inspect.isfunction(obj)
                else "object"
            )
            sig = signature(obj).replace("|", "\\|")
            summary = first_line(obj).replace("|", "\\|")
            chunks.append(f"| `{name}` | {kind} | `{sig}` | {summary} |\n")
    return "".join(chunks)


def _method_origin(cls, name):
    for base in cls.__mro__:
        if name in base.__dict__:
            return base.__name__
    return None


def _capability(cls, private_hook, public_method, *, scalar_only=False):
    if cls.__name__ in ABSTRACT_DISTRIBUTIONS:
        return "abstract"
    if (
        scalar_only
        and cls.__name__ in MULTIVARIATE_DISTRIBUTIONS
        and cls.__name__ != "Categorical"
    ):
        return "n/a (non-scalar)"
    if private_hook in cls.__dict__:
        return "native symbolic"
    origin = _method_origin(cls, public_method)
    if origin and origin != "Distribution":
        return f"native ({origin})"
    return "generic fallback"


def distribution_capabilities():
    rows = []
    for name in probstats.distributions.__all__:
        obj = getattr(probstats.distributions, name)
        if not inspect.isclass(obj):
            continue
        try:
            if not issubclass(obj, probstats.Distribution):
                continue
        except TypeError:
            continue
        if name in ABSTRACT_DISTRIBUTIONS:
            event = "abstract"
        elif name in MULTIVARIATE_DISTRIBUTIONS and name != "Categorical":
            event = "vector/matrix"
        else:
            event = "scalar"
        density_origin = _method_origin(obj, "pdf")
        density = (
            "abstract"
            if name in ABSTRACT_DISTRIBUTIONS
            else (
                "native symbolic"
                if density_origin and density_origin != "Distribution"
                else "generic from log-density"
            )
        )
        cdf = _capability(obj, "_cdf", "cdf", scalar_only=True)
        quant = _capability(obj, "_quantile", "quantile", scalar_only=True)
        sampling = (
            "native"
            if _method_origin(obj, "_sample") not in (None, "Distribution")
            else "generic/derived"
        )
        moments = (
            "native"
            if any(
                _method_origin(obj, hook) not in (None, "Distribution")
                for hook in ("_mean", "_variance", "_raw_moment")
            )
            else "generic exact"
        )
        entropy = (
            "native"
            if _method_origin(obj, "_entropy") not in (None, "Distribution")
            else "generic exact/numerical"
        )
        symbolic = (
            "yes"
            if name
            not in {
                "HistogramDistribution",
                "KDEDistribution",
                "KernelDensityDistribution",
                "KernelMixtureDistribution",
                "MultivariateKDEDistribution",
                "MultivariateKernelDensityDistribution",
            }
            else "data-driven"
        )
        scipy = (
            "yes" if name in SCIPY_ORACLE_DISTRIBUTIONS else "no/direct mapping absent"
        )
        category = (
            "wrapper/data-driven" if name in WRAPPER_DISTRIBUTIONS else "catalogue law"
        )
        rows.append(
            (
                name,
                category,
                event,
                density,
                cdf,
                quant,
                sampling,
                moments,
                entropy,
                symbolic,
                scipy,
            )
        )
    chunks = [
        "# Distribution capability matrix\n",
        "This generated matrix distinguishes native symbolic implementations from generic fallbacks. `generic fallback` means the protocol exists but may remain unevaluated or fail for parameterizations that SymPy cannot solve; it is not a promise of a closed form. SciPy-oracle availability records whether the test registry has a compatible direct differential reference.\n\n",
        "| Distribution | Category | Event | Density/PMF | CDF | Quantile | Sampling | Moments | Entropy | Symbolic parameters | SciPy oracle |\n",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    ]
    chunks += [
        "| "
        + " | ".join(
            f"`{value}`" if index == 0 else str(value)
            for index, value in enumerate(row)
        )
        + " |\n"
        for row in rows
    ]
    return "".join(chunks)


def main():
    (ROOT / "docs/api-reference.md").write_text(api_reference())
    (ROOT / "docs/distribution-capabilities.md").write_text(distribution_capabilities())


if __name__ == "__main__":
    main()
