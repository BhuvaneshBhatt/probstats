"""Formula parsing and design-matrix construction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from ._tabular import column_mapping


def _as_columns(data: Any) -> dict[str, np.ndarray]:
    return column_mapping(data)[0]


@dataclass(frozen=True)
class TreatmentContrast:
    """Reference-cell treatment coding for a categorical factor."""

    reference: Any | None = None

    def matrix(
        self, values: np.ndarray, name: str
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[Any, ...]]:
        levels = tuple(dict.fromkeys(values.tolist()))
        if len(levels) < 2:
            return np.empty((len(values), 0)), (), levels
        reference = levels[0] if self.reference is None else self.reference
        if reference not in levels:
            raise ValueError(
                f"reference level {reference!r} is not present in factor {name!r}"
            )
        kept = tuple(level for level in levels if level != reference)
        matrix = np.column_stack([(values == level).astype(float) for level in kept])
        names = tuple(f"C({name})[T.{level}]" for level in kept)
        return matrix, names, levels


@dataclass(frozen=True)
class SumContrast:
    """Sum-to-zero coding with the final observed level as reference."""

    reference: Any | None = None

    def matrix(
        self, values: np.ndarray, name: str
    ) -> tuple[np.ndarray, tuple[str, ...], tuple[Any, ...]]:
        levels = tuple(dict.fromkeys(values.tolist()))
        if len(levels) < 2:
            return np.empty((len(values), 0)), (), levels
        reference = levels[-1] if self.reference is None else self.reference
        if reference not in levels:
            raise ValueError(
                f"reference level {reference!r} is not present in factor {name!r}"
            )
        kept = tuple(level for level in levels if level != reference)
        cols = []
        for level in kept:
            col = np.zeros(len(values), dtype=float)
            col[values == level] = 1.0
            col[values == reference] = -1.0
            cols.append(col)
        matrix = np.column_stack(cols)
        names = tuple(f"C({name})[S.{level}]" for level in kept)
        return matrix, names, levels


@dataclass(frozen=True)
class FormulaTerm:
    factors: tuple[str, ...]

    @property
    def name(self) -> str:
        return ":".join(self.factors)


def _split_top_level(text: str, separator: str) -> list[str]:
    """Split a formula fragment only at separators outside parentheses."""
    pieces: list[str] = []
    depth = 0
    start = 0
    for i, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses in formula")
        elif char == separator and depth == 0:
            pieces.append(text[start:i].strip())
            start = i + 1
    if depth != 0:
        raise ValueError("unbalanced parentheses in formula")
    pieces.append(text[start:].strip())
    return [piece for piece in pieces if piece]


def _split_plus(rhs: str) -> list[str]:
    return _split_top_level(rhs.replace("- 1", "+ 0").replace("-1", "+0"), "+")


def _expand_star(piece: str) -> list[FormulaTerm]:
    factors = tuple(
        part.strip() for part in _split_top_level(piece, "*") if part.strip()
    )
    if len(factors) == 1:
        return [FormulaTerm(factors)]
    terms: list[FormulaTerm] = []
    for mask in range(1, 1 << len(factors)):
        subset = tuple(factors[i] for i in range(len(factors)) if mask & (1 << i))
        terms.append(FormulaTerm(subset))
    return terms


def parse_formula(formula: str) -> tuple[str, tuple[FormulaTerm, ...], bool]:
    """Parse a compact R-style formula using +, :, *, 0/-1, and transformed factors."""
    if "~" not in formula:
        raise ValueError("formula must contain '~'")
    lhs, rhs = (part.strip() for part in formula.split("~", 1))
    if not re.fullmatch(r"[A-Za-z_]\w*", lhs):
        raise ValueError("response must be a simple column name")
    intercept = True
    ordered: dict[tuple[str, ...], FormulaTerm] = {}
    for piece in _split_plus(rhs):
        if piece in {"0", "-1"}:
            intercept = False
            continue
        if piece == "1":
            intercept = True
            continue
        for term in _expand_star(piece):
            flat = tuple(
                part.strip()
                for factor in term.factors
                for part in _split_top_level(factor, ":")
                if part.strip()
            )
            normalized = FormulaTerm(flat)
            ordered.setdefault(normalized.factors, normalized)
    return lhs, tuple(ordered.values()), intercept


def _factor_spec(token: str) -> tuple[str, bool, str, int | None]:
    """Return (source, categorical, transform kind, optional degree)."""
    match = re.fullmatch(r"C\(\s*([A-Za-z_]\w*)\s*\)", token)
    if match:
        return match.group(1), True, "identity", None
    match = re.fullmatch(r"(log|exp|sqrt)\(\s*([A-Za-z_]\w*)\s*\)", token)
    if match:
        return match.group(2), False, match.group(1), None
    match = re.fullmatch(r"I\(\s*([A-Za-z_]\w*)\s*\*\*\s*(\d+)\s*\)", token)
    if match:
        return match.group(1), False, "power", int(match.group(2))
    match = re.fullmatch(r"poly\(\s*([A-Za-z_]\w*)\s*,\s*(\d+)\s*\)", token)
    if match:
        degree = int(match.group(2))
        if degree < 1:
            raise ValueError("poly degree must be at least 1")
        return match.group(1), False, "poly", degree
    if not re.fullmatch(r"[A-Za-z_]\w*", token):
        raise ValueError(f"unsupported formula factor {token!r}")
    return token, False, "identity", None


@dataclass(frozen=True)
class FactorEncoding:
    token: str
    source_name: str
    categorical: bool
    levels: tuple[Any, ...] = ()
    contrast: Any = None
    column_names: tuple[str, ...] = ()
    transform_kind: str = "identity"
    degree: int | None = None

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply the fitted numeric transform or categorical contrast coding."""
        if not self.categorical:
            values = np.asarray(values, dtype=float)
            if self.transform_kind == "identity":
                out = values.reshape(-1, 1)
            elif self.transform_kind == "log":
                if np.any(values <= 0):
                    raise ValueError(
                        f"log-transformed predictor {self.source_name!r} must be positive"
                    )
                out = np.log(values).reshape(-1, 1)
            elif self.transform_kind == "exp":
                out = np.exp(values).reshape(-1, 1)
            elif self.transform_kind == "sqrt":
                if np.any(values < 0):
                    raise ValueError(
                        f"sqrt-transformed predictor {self.source_name!r} must be nonnegative"
                    )
                out = np.sqrt(values).reshape(-1, 1)
            elif self.transform_kind == "power":
                out = (values ** int(self.degree)).reshape(-1, 1)
            elif self.transform_kind == "poly":
                out = np.column_stack(
                    [values**power for power in range(1, int(self.degree) + 1)]
                )
            else:
                raise ValueError(f"unknown numeric transform {self.transform_kind!r}")
            if not np.all(np.isfinite(out)):
                raise ValueError(
                    f"transformed predictor {self.source_name!r} must be finite"
                )
            return out
        if isinstance(self.contrast, TreatmentContrast):
            levels = self.levels
            reference = (
                levels[0]
                if self.contrast.reference is None
                else self.contrast.reference
            )
            kept = tuple(level for level in levels if level != reference)
            unknown = set(values.tolist()) - set(levels)
            if unknown:
                raise ValueError(
                    f"unknown categorical levels for {self.source_name}: {sorted(map(str, unknown))}"
                )
            return np.column_stack([(values == level).astype(float) for level in kept])
        if isinstance(self.contrast, SumContrast):
            levels = self.levels
            reference = (
                levels[-1]
                if self.contrast.reference is None
                else self.contrast.reference
            )
            kept = tuple(level for level in levels if level != reference)
            unknown = set(values.tolist()) - set(levels)
            if unknown:
                raise ValueError(
                    f"unknown categorical levels for {self.source_name}: {sorted(map(str, unknown))}"
                )
            cols = []
            for level in kept:
                col = np.zeros(len(values), dtype=float)
                col[values == level] = 1.0
                col[values == reference] = -1.0
                cols.append(col)
            return np.column_stack(cols)
        raise TypeError("unknown categorical contrast")


@dataclass(frozen=True)
class DesignMatrix:
    formula: str
    response_name: str
    matrix: np.ndarray
    response: np.ndarray
    column_names: tuple[str, ...]
    terms: tuple[FormulaTerm, ...]
    term_slices: Mapping[str, tuple[int, ...]]
    factor_encodings: Mapping[str, FactorEncoding]
    intercept: bool
    training_data: Mapping[str, np.ndarray] | None = None

    def transform(self, data: Mapping[str, Any]) -> np.ndarray:
        columns = _as_columns(data)
        n = len(next(iter(columns.values())))
        pieces: list[np.ndarray] = []
        if self.intercept:
            pieces.append(np.ones((n, 1)))
        for term in self.terms:
            factor_matrices = []
            for token in term.factors:
                encoding = self.factor_encodings[token]
                if encoding.source_name not in columns:
                    raise ValueError(
                        f"missing predictor column {encoding.source_name!r}"
                    )
                factor_matrices.append(
                    encoding.transform(columns[encoding.source_name])
                )
            pieces.append(_interaction_matrix(factor_matrices)[0])
        return np.column_stack(pieces) if pieces else np.empty((n, 0))


def _interaction_matrix(
    matrices: list[np.ndarray], names: list[tuple[str, ...]] | None = None
):
    if not matrices:
        raise ValueError("interaction requires at least one factor")
    result = matrices[0]
    out_names = list(names[0]) if names else [""] * result.shape[1]
    for index, matrix in enumerate(matrices[1:], start=1):
        result = np.column_stack(
            [
                result[:, i] * matrix[:, j]
                for i, j in product(range(result.shape[1]), range(matrix.shape[1]))
            ]
        )
        if names:
            next_names = names[index]
            out_names = [
                f"{left}:{right}" for left, right in product(out_names, next_names)
            ]
    return result, tuple(out_names)


def _resolve_contrast(spec: Any) -> Any:
    if spec is None or spec == "treatment":
        return TreatmentContrast()
    if spec == "sum":
        return SumContrast()
    if isinstance(spec, (TreatmentContrast, SumContrast)):
        return spec
    raise ValueError(
        "contrast must be 'treatment', 'sum', TreatmentContrast, or SumContrast"
    )


def design_matrix(
    formula: str, data: Mapping[str, Any], *, contrasts: Mapping[str, Any] | None = None
) -> DesignMatrix:
    """Compile a formula and tabular mapping into a reusable design matrix."""
    columns = _as_columns(data)
    response_name, terms, intercept = parse_formula(formula)
    if response_name not in columns:
        raise ValueError(f"missing response column {response_name!r}")
    response = np.asarray(columns[response_name], dtype=float).reshape(-1)
    if not np.all(np.isfinite(response)):
        raise ValueError("response must be finite")
    contrast_map = contrasts or {}
    encodings: dict[str, FactorEncoding] = {}
    for term in terms:
        for token in term.factors:
            if token in encodings:
                continue
            source, explicit_cat, transform_kind, degree = _factor_spec(token)
            if source not in columns:
                raise ValueError(f"missing predictor column {source!r}")
            raw = columns[source]
            inferred_cat = explicit_cat or raw.dtype.kind in "OUSb"
            if inferred_cat:
                contrast = _resolve_contrast(contrast_map.get(source))
                matrix, names, levels = contrast.matrix(raw, source)
                encodings[token] = FactorEncoding(
                    token, source, True, levels, contrast, names
                )
            else:
                values = np.asarray(raw, dtype=float)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"predictor {source!r} must be finite")

                if transform_kind == "poly":
                    factor_names = tuple(
                        f"poly({source},{degree})[{power}]"
                        for power in range(1, int(degree) + 1)
                    )
                else:
                    factor_names = (token,)
                encodings[token] = FactorEncoding(
                    token, source, False, (), None, factor_names, transform_kind, degree
                )
    pieces: list[np.ndarray] = []
    names: list[str] = []
    slices: dict[str, tuple[int, ...]] = {}
    cursor = 0
    if intercept:
        pieces.append(np.ones((response.size, 1)))
        names.append("Intercept")
        slices["Intercept"] = (0,)
        cursor = 1
    for term in terms:
        mats = [
            encodings[token].transform(columns[encodings[token].source_name])
            for token in term.factors
        ]
        factor_names = [encodings[token].column_names for token in term.factors]
        matrix, term_names = _interaction_matrix(mats, factor_names)
        if matrix.shape[1] == 0:
            continue
        pieces.append(matrix)
        names.extend(term_names)
        indices = tuple(range(cursor, cursor + matrix.shape[1]))
        slices[term.name] = indices
        cursor += matrix.shape[1]
    matrix = np.column_stack(pieces) if pieces else np.empty((response.size, 0))
    return DesignMatrix(
        formula,
        response_name,
        matrix,
        response,
        tuple(names),
        terms,
        slices,
        encodings,
        intercept,
        columns,
    )


__all__ = [
    "DesignMatrix",
    "FactorEncoding",
    "FormulaTerm",
    "SumContrast",
    "TreatmentContrast",
    "design_matrix",
    "parse_formula",
]
