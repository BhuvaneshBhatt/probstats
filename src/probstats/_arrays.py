"""Internal array ownership helpers for immutable public results."""

from __future__ import annotations

import numpy as np


def readonly_array(value, *, dtype=None) -> np.ndarray:
    """Return an owned NumPy array whose contents cannot be mutated in place."""
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array
