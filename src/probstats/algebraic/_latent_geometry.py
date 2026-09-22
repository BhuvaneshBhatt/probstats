"""Latent-class models and algebraic identifiability diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import product
from math import factorial

import sympy as sp

from ._dependencies import require_latent_backends
from ._validation import cardinalities as _cardinalities
from ._validation import exact_integer
from .models import DiscreteAlgebraicModel, ModelIdeal


def _shape(cardinalities) -> tuple[int, ...]:
    dims = _cardinalities(cardinalities, minimum_variables=2)
    if any(dim < 2 for dim in dims):
        raise ValueError("latent-class variables must each have cardinality >= 2")
    return dims


def _probabilities(dims: tuple[int, ...]) -> tuple[sp.Symbol, ...]:
    return tuple(
        sp.Symbol("p_" + "_".join(map(str, index)))
        for index in product(*(range(d) for d in dims))
    )


def _reduced_parameterization(dims: tuple[int, ...], classes: int):
    weights_free = tuple(sp.Symbol(f"lambda_{h}") for h in range(classes - 1))
    weights = (*weights_free, sp.Integer(1) - sum(weights_free))
    conditionals: list[tuple[tuple[sp.Expr, ...], ...]] = []
    free: list[sp.Symbol] = list(weights_free)
    for h in range(classes):
        component = []
        for axis, dim in enumerate(dims):
            entries = tuple(sp.Symbol(f"theta_{h}_{axis}_{i}") for i in range(dim - 1))
            free.extend(entries)
            component.append((*entries, sp.Integer(1) - sum(entries)))
        conditionals.append(tuple(component))
    return tuple(free), tuple(weights), tuple(conditionals)


def _mapping(dims, probabilities, weights, conditionals):
    mapping = {}
    for p, index in zip(probabilities, product(*(range(d) for d in dims)), strict=True):
        mapping[p] = sp.expand(
            sum(
                weights[h]
                * sp.prod(
                    conditionals[h][axis][index[axis]] for axis in range(len(dims))
                )
                for h in range(len(weights))
            )
        )
    return mapping


@dataclass(frozen=True, slots=True)
class LatentClassModel(DiscreteAlgebraicModel):
    """Finite latent-class/naive-Bayes model with one hidden class variable.

    The public parameterization uses independent simplex coordinates: the last
    mixing weight and the last conditional probability in every observed mode
    are eliminated symbolically.  This removes scaling and normalization
    redundancies before Jacobian/fiber calculations.
    """

    latent_classes: int = 1
    mixing_weights: tuple[sp.Expr, ...] = field(default=(), compare=False)
    conditional_probabilities: tuple[tuple[tuple[sp.Expr, ...], ...], ...] = field(
        default=(), compare=False
    )

    @property
    def parameter_region(self):
        """Return simplex constraints on independent latent parameters."""
        constraints = [sp.Ge(weight, 0) for weight in self.mixing_weights]
        for component in self.conditional_probabilities:
            for vector in component:
                constraints.extend(sp.Ge(value, 0) for value in vector)
        return sp.And(*constraints)

    @property
    def parameter_dimension(self) -> int:
        return (self.latent_classes - 1) + self.latent_classes * sum(
            d - 1 for d in self.cardinalities
        )

    def image_dimension(self) -> int:
        """Return a certified generic image dimension from an exact Jacobian witness.

        A pointwise Jacobian rank is a rigorous lower bound on generic rank.  It
        becomes a certificate when it reaches the secant/ambient/parameter
        upper bound.  Unsupported defective or otherwise inconclusive cases
        raise rather than guessing a dimension.
        """
        backend = require_latent_backends()
        upper = min(
            self.parameter_dimension,
            len(self.probabilities) - 1,
            backend.secant_expected_dimension(self.cardinalities, self.latent_classes),
        )
        jac = sp.Matrix(
            tuple(self.parameterization[p] for p in self.probabilities)
        ).jacobian(self.parameters)
        # Deterministic interior rational witnesses; nonzero minors at any one
        # witness certify the corresponding generic-rank lower bound.
        for offset in (2, 3, 5):
            substitutions = {
                parameter: sp.Rational(offset + i, 10 * (len(self.parameters) + offset))
                for i, parameter in enumerate(self.parameters)
            }
            rank = int(jac.subs(substitutions).rank())
            if rank == upper:
                return rank
        raise NotImplementedError(
            "the exact Jacobian witnesses did not attain the rigorous image-dimension upper bound"
        )

    def generic_fiber_dimension(self) -> int:
        return self.parameter_dimension - self.image_dimension()

    def dimension(self) -> int:
        """Return the certified generic dimension of the latent-model image."""
        return self.image_dimension()

    def ideal(self) -> ModelIdeal:
        """Return the exact Zariski image ideal when bounded elimination is feasible."""
        return ModelIdeal(self.probabilities, self.image_ideal())

    @property
    def variety(self):
        """Return the exact Zariski closure when bounded elimination is feasible."""
        return sp.And(*(sp.Eq(eq, 0) for eq in self.image_ideal()))

    @property
    def region(self):
        raise NotImplementedError(
            "the exact latent statistical image is parametrically constrained; "
            "use parameter_region and parameterization rather than replacing it "
            "with the Zariski closure intersected with the probability simplex"
        )

    def image_ideal(self, *, max_parameters: int = 10) -> tuple[sp.Expr, ...]:
        """Eliminate latent parameters and return the exact Zariski image ideal.

        A guard prevents accidental lexicographic elimination on large models.
        """
        parameter_limit = exact_integer(
            max_parameters, name="max_parameters", minimum=0
        )
        if len(self.parameters) > parameter_limit:
            raise ValueError(
                f"exact latent image elimination has {len(self.parameters)} parameters; "
                f"increase max_parameters explicitly to proceed"
            )
        if self.image_dimension() == len(self.probabilities) - 1:
            return (sp.expand(sum(self.probabilities) - 1),)
        backend = require_latent_backends()
        graph = tuple(p - self.parameterization[p] for p in self.probabilities)
        variables = (*self.parameters, *self.probabilities)
        return backend.elimination_ideal_qq(graph, variables, self.parameters)


@dataclass(frozen=True, slots=True)
class IdentifiabilityResult:
    """Local/global information about a latent parameterization."""

    identifiable: bool | None
    up_to_label_swapping: bool | None
    generic: bool
    parameter_dimension: int
    image_dimension: int
    fiber_dimension: int
    expected_label_orbit: int
    locally_finite_to_one: bool
    certified: bool
    method: str
    reason: str


def latent_class_model(
    cardinalities, latent_classes: int, *, variables=()
) -> LatentClassModel:
    """Construct a finite latent-class model for observed categorical variables."""
    dims = _shape(cardinalities)
    classes = exact_integer(latent_classes, name="latent_classes", minimum=1)
    names = tuple(variables) or tuple(f"X{i}" for i in range(len(dims)))
    if len(names) != len(dims) or len(set(names)) != len(names):
        raise ValueError("variables must be distinct and match cardinalities")
    probabilities = _probabilities(dims)
    params, weights, conditionals = _reduced_parameterization(dims, classes)
    mapping = _mapping(dims, probabilities, weights, conditionals)
    return LatentClassModel(
        probabilities=probabilities,
        equations=(),
        inequalities=(),
        parameters=params,
        parameterization=mapping,
        metadata={"kind": "latent_class", "latent_classes": classes},
        cardinalities=dims,
        variable_names=names,
        latent_classes=classes,
        mixing_weights=weights,
        conditional_probabilities=conditionals,
    )


def generic_identifiability(model: LatentClassModel) -> IdentifiabilityResult:
    """Certify generic identifiability where dimension and Kruskal theory allow."""
    if not isinstance(model, LatentClassModel):
        raise TypeError("generic_identifiability requires LatentClassModel")
    backend = require_latent_backends()
    parameter_dim = model.parameter_dimension
    image_dim = model.image_dimension()
    fiber_dim = parameter_dim - image_dim
    label_orbit = factorial(model.latent_classes)
    if fiber_dim > 0:
        return IdentifiabilityResult(
            False,
            False,
            True,
            parameter_dim,
            image_dim,
            fiber_dim,
            label_orbit,
            False,
            True,
            "jacobian-dimension",
            "the generic parameter fiber has positive dimension",
        )
    cp = backend.generic_cp_identifiability(model.cardinalities, model.latent_classes)
    if cp.identifiable is True:
        return IdentifiabilityResult(
            True,
            True,
            True,
            parameter_dim,
            image_dim,
            0,
            label_orbit,
            True,
            True,
            "jacobian+generic-kruskal",
            "the generic fiber is finite and CP decomposition is essentially unique",
        )
    return IdentifiabilityResult(
        None,
        None,
        True,
        parameter_dim,
        image_dim,
        0,
        label_orbit,
        True,
        False,
        "jacobian-dimension",
        "the parameterization is generically finite-to-one, but available uniqueness criteria are inconclusive",
    )


def identifiability(
    model: LatentClassModel, parameters: Mapping[sp.Symbol, object] | None = None
) -> IdentifiabilityResult:
    """Return generic or pointwise local identifiability information.

    Pointwise analysis certifies the Jacobian's local fiber dimension.  It does
    not infer global uniqueness from a full-rank Jacobian.
    """
    if parameters is None:
        return generic_identifiability(model)
    supplied = {sp.sympify(k): sp.sympify(v) for k, v in parameters.items()}
    if set(supplied) != set(model.parameters):
        raise ValueError(
            "parameters must assign every independent model parameter exactly once"
        )
    jac = sp.Matrix(
        tuple(model.parameterization[p] for p in model.probabilities)
    ).jacobian(model.parameters)
    rank = int(jac.subs(supplied).rank())
    fiber_dim = model.parameter_dimension - rank
    if fiber_dim > 0:
        return IdentifiabilityResult(
            False,
            False,
            False,
            model.parameter_dimension,
            rank,
            fiber_dim,
            factorial(model.latent_classes),
            False,
            True,
            "pointwise-jacobian",
            "the parameterization loses rank at this parameter point",
        )
    return IdentifiabilityResult(
        None,
        None,
        False,
        model.parameter_dimension,
        rank,
        0,
        factorial(model.latent_classes),
        True,
        True,
        "pointwise-jacobian",
        "the map is locally finite-to-one here; global uniqueness requires a fiber/uniqueness certificate",
    )
