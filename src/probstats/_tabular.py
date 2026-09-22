"""Optional labeled-data interoperability without a hard pandas dependency."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _label_attr(data: Any, name: str):
    value = getattr(data, name, None)
    return None if callable(value) else value


def column_mapping(data: Any) -> tuple[dict[str, np.ndarray], Any | None]:
    if isinstance(data, Mapping):
        items = data.items()
    elif hasattr(data, "columns") and hasattr(data, "__getitem__"):
        items = ((name, data[name]) for name in data.columns)
    else:
        raise TypeError("data must be a mapping or a DataFrame-like table")
    columns = {
        str(name): (
            np.asarray(values.to_numpy(copy=False))
            if hasattr(values, "to_numpy")
            else np.asarray(values)
        )
        for name, values in items
    }
    if not columns:
        raise ValueError("data must contain at least one column")
    lengths = {len(values) for values in columns.values()}
    if len(lengths) != 1:
        raise ValueError("all data columns must have the same length")
    if next(iter(lengths)) == 0:
        raise ValueError("data must contain at least one row")
    return columns, _label_attr(data, "index")


def array_metadata(data: Any):
    """Return an ndarray plus optional row, column, and series labels."""
    if hasattr(data, "to_numpy"):
        try:
            array = data.to_numpy(copy=False)
        except TypeError:
            array = data.to_numpy()
    else:
        array = np.asarray(data)
    index = _label_attr(data, "index")
    columns = _label_attr(data, "columns")
    column_names = (
        tuple(str(value) for value in columns) if columns is not None else None
    )
    name = getattr(data, "name", None)
    return np.asarray(array), index, column_names, None if name is None else str(name)


def labeled_vector(values: Any, index: Any | None, *, name: str | None = None):
    array = np.asarray(values)
    if index is None:
        return array
    try:
        import pandas as pd
    except ImportError:
        return array
    return pd.Series(array, index=index, name=name)


def labeled_matrix(values: Any, index: Any | None, columns=None):
    array = np.asarray(values)
    if index is None:
        return array
    try:
        import pandas as pd
    except ImportError:
        return array
    return pd.DataFrame(array, index=index, columns=columns)


__all__ = ["array_metadata", "column_mapping", "labeled_matrix", "labeled_vector"]
