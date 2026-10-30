#
# SpectRAL - Spectral reconstruction analysis toolkit
# Copyright (C) 2026 Julian Parrino (https://github.com/jparrino)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <https://www.gnu.org/licenses/>.
#

"""Shared helpers for the optional end-to-end arbitrary-precision backend."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any, Iterable, Sequence

import numpy as np


def require_mpmath():
    """Import and return mpmath's ``mp`` context with a useful error message."""
    try:
        from mpmath import mp
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise ImportError("the mpmath backend requires mpmath to be installed") from exc
    return mp


def to_mpf(value: Any, mp=None):
    """Convert a scalar without introducing an intermediate Python float.

    Decimal strings, :class:`~decimal.Decimal` values and fractions retain the
    information supplied by the caller.  A Python/NumPy float is converted
    exactly, but of course cannot contain more information than its original
    binary64 representation.
    """
    mp = require_mpmath() if mp is None else mp
    if hasattr(value, "_mpf_"):
        return value
    if isinstance(value, str):
        return mp.mpf(value)
    if isinstance(value, Decimal):
        return mp.mpf(str(value))
    if isinstance(value, Fraction):
        return mp.mpf(value.numerator) / value.denominator
    if isinstance(value, (np.integer, int)):
        return mp.mpf(int(value))
    if isinstance(value, np.floating):
        return mp.mpf(float(value))
    return mp.mpf(value)


def is_mp_scalar(value: Any) -> bool:
    return hasattr(value, "_mpf_") or hasattr(value, "_mpc_")


def is_mp_matrix(value: Any) -> bool:
    return (
        hasattr(value, "rows")
        and hasattr(value, "cols")
        and value.__class__.__module__.startswith("mpmath")
    )


def source_vector(values: Any) -> tuple[Any, ...]:
    """Copy a one-dimensional input while preserving its scalar objects."""
    if is_mp_matrix(values):
        if values.cols == 1:
            return tuple(values[i] for i in range(values.rows))
        if values.rows == 1:
            return tuple(values[0, i] for i in range(values.cols))
        raise ValueError("expected a vector")
    if isinstance(values, np.ndarray):
        if values.ndim != 1:
            raise ValueError("expected a one-dimensional array")
        return tuple(values.tolist())
    return tuple(values)


def source_matrix(values: Any) -> tuple[tuple[Any, ...], ...]:
    """Copy a two-dimensional input while preserving its scalar objects."""
    if is_mp_matrix(values):
        return tuple(
            tuple(values[i, j] for j in range(values.cols)) for i in range(values.rows)
        )
    if isinstance(values, np.ndarray):
        if values.ndim != 2:
            raise ValueError("expected a two-dimensional array")
        return tuple(tuple(row) for row in values.tolist())
    return tuple(tuple(row) for row in values)


def mp_vector(values: Iterable[Any], mp=None):
    mp = require_mpmath() if mp is None else mp
    return mp.matrix([to_mpf(value, mp) for value in values])


def mp_matrix(values: Any, mp=None):
    mp = require_mpmath() if mp is None else mp
    if is_mp_matrix(values):
        return mp.matrix(values)
    rows = source_matrix(values)
    return mp.matrix([[to_mpf(value, mp) for value in row] for row in rows])


def mp_values(values: Any, mp=None) -> list[Any]:
    mp = require_mpmath() if mp is None else mp
    return [to_mpf(value, mp) for value in source_vector(values)]


def mp_trapezoid_weights(times: Sequence[Any], mp=None) -> list[Any]:
    mp = require_mpmath() if mp is None else mp
    t = [to_mpf(value, mp) for value in times]
    if len(t) < 2:
        raise ValueError("at least two times are required to build trapezoid weights")
    if any(not mp.isfinite(value) for value in t):
        raise ValueError("times contain non-finite values")
    dt = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    if any(value <= 0 for value in dt):
        raise ValueError(
            "times must be strictly increasing for automatic trapezoid weights"
        )
    half = mp.mpf(1) / 2
    out = [half * dt[0]]
    out.extend(half * (dt[i - 1] + dt[i]) for i in range(1, len(t) - 1))
    out.append(half * dt[-1])
    return out


def mp_matrix_to_numpy(value: Any) -> np.ndarray:
    if not is_mp_matrix(value):
        return np.asarray(value, dtype=float)
    if value.cols == 1:
        return np.asarray([float(value[i]) for i in range(value.rows)], dtype=float)
    return np.asarray(
        [[float(value[i, j]) for j in range(value.cols)] for i in range(value.rows)],
        dtype=float,
    )


def mp_sequence_to_numpy(values: Any) -> np.ndarray:
    if is_mp_matrix(values):
        return mp_matrix_to_numpy(values)
    return np.asarray([float(value) for value in values], dtype=float)


def mp_inf_norm_vector(values: Any, mp=None):
    mp = require_mpmath() if mp is None else mp
    vals = (
        list(values)
        if not is_mp_matrix(values)
        else [values[i] for i in range(values.rows)]
    )
    return max((abs(value) for value in vals), default=mp.zero)


def mp_inf_norm_matrix(matrix: Any, mp=None):
    mp = require_mpmath() if mp is None else mp
    return max(
        (
            mp.fsum(abs(matrix[i, j]) for j in range(matrix.cols))
            for i in range(matrix.rows)
        ),
        default=mp.zero,
    )


def normalize_backend(backend: str) -> str:
    key = backend.lower()
    if key in {"numpy", "float", "float64", "double"}:
        return "numpy"
    if key in {"mpmath", "mp", "arbitrary"}:
        return "mpmath"
    raise ValueError("backend must be 'numpy' or 'mpmath'")
