"""Polynomial models for independence and conditional independence."""

from __future__ import annotations

from itertools import product

import sympy as sp

from ._dependencies import require_tensoratlas
from ._validation import cardinalities as _cardinalities
from .models import DiscreteAlgebraicModel


def _probability_coords(cardinalities, prefix: str = "p"):
    dims = _cardinalities(cardinalities, minimum_variables=2)
    indices = tuple(product(*(range(dim) for dim in dims)))
    symbols = tuple(
        sp.Symbol(prefix + "_" + "_".join(map(str, index))) for index in indices
    )
    return dims, indices, symbols


def independent_model(cardinalities, *, variables=()) -> DiscreteAlgebraicModel:
    """Return the complete-independence model for a finite probability table."""
    dims, _, probabilities = _probability_coords(cardinalities)
    backend = require_tensoratlas()
    param = backend.segre_parameterization(
        dims, coordinate_prefix="p", parameter_prefix="a"
    )
    equations = backend.segre_equations(dims, coordinates=param.coordinates)
    coordinate_map = {str(symbol): symbol for symbol in probabilities}
    mapping = {
        coordinate_map[str(coord)]: expr for coord, expr in param.mapping.items()
    }
    params = tuple(symbol for factor in param.parameters for symbol in factor)
    return DiscreteAlgebraicModel(
        probabilities=probabilities,
        equations=equations,
        parameters=params,
        parameterization=mapping,
        metadata={"model": "complete_independent"},
        cardinalities=dims,
        variable_names=tuple(variables),
    )


def _name_group(value) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def conditionally_independent_model(variables, statements) -> DiscreteAlgebraicModel:
    """Return a discrete conditional-independence model.

    Each statement is ``(left, right, conditioned)``.  The three groups must partition the model variables, which gives the
    standard determinantal equations on every conditioned table slice.
    """
    if not isinstance(variables, dict) or not variables:
        raise TypeError(
            "variables must be a nonempty mapping of names to cardinalities"
        )
    names = tuple(variables)
    dims = _cardinalities((variables[name] for name in names), minimum_variables=2)
    dims, indices, probabilities = _probability_coords(dims)
    by_index = dict(zip(indices, probabilities, strict=True))
    equations = []
    normalized = []
    for statement in statements:
        if len(statement) != 3:
            raise ValueError("conditional-independence statements require three groups")
        left, right, given = (_name_group(group) for group in statement)
        groups = left + right + given
        if len(set(groups)) != len(groups) or set(groups) != set(names):
            raise ValueError("statement groups must partition all model variables")
        left_axes = tuple(names.index(name) for name in left)
        right_axes = tuple(names.index(name) for name in right)
        given_axes = tuple(names.index(name) for name in given)
        left_keys = tuple(product(*(range(dims[axis]) for axis in left_axes)))
        right_keys = tuple(product(*(range(dims[axis]) for axis in right_axes)))
        given_keys = tuple(product(*(range(dims[axis]) for axis in given_axes))) or (
            (),
        )

        def coordinate(
            left_key,
            right_key,
            given_key,
            left_axes=left_axes,
            right_axes=right_axes,
            given_axes=given_axes,
        ):
            index = [0] * len(dims)
            for axes, key in (
                (left_axes, left_key),
                (right_axes, right_key),
                (given_axes, given_key),
            ):
                for axis, value in zip(axes, key, strict=True):
                    index[axis] = value
            return by_index[tuple(index)]

        for given_key in given_keys:
            for i in range(len(left_keys)):
                for j in range(i + 1, len(left_keys)):
                    for k in range(len(right_keys)):
                        for m in range(k + 1, len(right_keys)):
                            p_ik = coordinate(left_keys[i], right_keys[k], given_key)
                            p_jm = coordinate(left_keys[j], right_keys[m], given_key)
                            p_im = coordinate(left_keys[i], right_keys[m], given_key)
                            p_jk = coordinate(left_keys[j], right_keys[k], given_key)
                            equations.append(sp.expand(p_ik * p_jm - p_im * p_jk))
        normalized.append((left, right, given))
    equations = tuple(dict.fromkeys(equations))
    return DiscreteAlgebraicModel(
        probabilities=probabilities,
        equations=equations,
        metadata={
            "model": "conditionally_independent",
            "statements": tuple(normalized),
        },
        cardinalities=dims,
        variable_names=names,
    )
