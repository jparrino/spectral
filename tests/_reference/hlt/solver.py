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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .precision import (
    mp_matrix,
    mp_vector,
    normalize_backend,
    require_mpmath,
    source_vector,
    to_mpf,
)


@dataclass(frozen=True)
class SolveResult:
    q: Any
    eta: Any
    denominator: Any
    backend: str = "numpy"
    endpoint_eta: Any = 0
    omega_eta: tuple = ()


def constraint_columns(columns, values, *, mp=None):
    """Scale and select independent constraints, retaining multiplier indices.

    Identical scaled columns with identical values are redundant. Other
    dependent columns are rejected rather than silently relaxing constraints.
    """
    kept, scaled, rhs, scales = [], [], [], []
    for index, (col, value) in enumerate(zip(columns, values)):
        scale = max(abs(x) for x in col)
        if scale == 0:
            if value != 0:
                raise ValueError("nonzero constraint is impossible for a zero basis vector")
            continue
        normalized = [x / scale for x in col]
        desired = value / scale
        duplicate = next((j for j, old in enumerate(scaled) if normalized == old), None)
        if duplicate is not None:
            if desired != rhs[duplicate]:
                raise ValueError("conflicting duplicate constraints; constraints must be independent")
            continue
        kept.append(index)
        scaled.append(normalized)
        rhs.append(desired)
        scales.append(scale)
    if scaled:
        n, k = len(scaled[0]), len(scaled)
        if mp is None:
            independent = k <= n and np.linalg.matrix_rank(np.array(scaled).T) == k
        else:
            # Reorthogonalized Gram-Schmidt, entirely at working precision.
            orthogonal = []
            independent = k <= n
            for col in scaled:
                v = mp.matrix(col)
                for _ in range(2):
                    for u in orthogonal:
                        v -= mp.fdot(u, v) * u
                norm = mp.norm(v)
                if norm <= 100 * max(n, k) * mp.eps * mp.norm(mp.matrix(col)):
                    independent = False
                    break
                orthogonal.append(v / norm)
        if not independent:
            raise ValueError("constraints must be independent; reduce constraints or increase precision")
    return kept, scaled, rhs, scales


def solve_hlt_endpoint_constrained(
    W, f, v, value, *, R=None, c=None, backend="numpy", mp_dps=80,
    omega_vectors=(), omega_values=(),
) -> SolveResult:
    """Solve the HLT quadratic with area, endpoint and arbitrary point constraints.

    Constraint columns and objective are scaled before the KKT solve. Returned
    multipliers use the original scaling, including zeros for redundant points.
    """
    backend = normalize_backend(backend)
    if backend == "mpmath" and mp_dps < 30:
        raise ValueError("mp_dps should be at least 30 for the mpmath backend")
    from contextlib import nullcontext
    mp = require_mpmath() if backend == "mpmath" else None
    with mp.workdps(mp_dps) if mp else nullcontext():
        vector = (lambda x: mp_vector(source_vector(x), mp)) if mp else (lambda x: np.asarray(x, dtype=float))
        scalar = (lambda x: to_mpf(x, mp)) if mp else float
        W = mp_matrix(W, mp) if mp else np.asarray(W, dtype=float)
        f = vector(f)
        columns, values = [], []
        area_index = endpoint_index = None
        if R is not None:
            area_index = len(columns)
            columns.append(vector(R)); values.append(scalar(c))
        if v is not None:
            endpoint_index = len(columns)
            columns.append(vector(v)); values.append(scalar(value))
        omega_start = len(columns)
        if len(omega_vectors) != len(omega_values):
            raise ValueError("constraint vectors and values must have matching lengths")
        columns.extend(vector(col) for col in omega_vectors)
        values.extend(scalar(d) for d in omega_values)
        for col, d in zip(columns, values):
            if mp:
                _validate_mp_inputs(W, f, col, d, mp)
            else:
                _validate_numpy_inputs(W, f, col, d)
        kept, scaled, rhs_values, scales = constraint_columns(columns, values, mp=mp)
        # Preserve the established solve when only the area constraint remains.
        if not kept or (area_index is not None and kept == [area_index]):
            old = (solve_hlt_unconstrained(W, f, backend=backend, mp_dps=mp_dps)
                   if not kept else solve_hlt_constrained(W, f, R, c, backend=backend, mp_dps=mp_dps))
            from dataclasses import replace
            return replace(old, omega_eta=tuple(scalar(0) for _ in omega_vectors))
        n, k = len(f), len(kept)
        objective_scale = max(abs(x) for x in W) if mp else np.max(np.abs(W))
        if objective_scale == 0:
            raise ValueError("HLT objective matrix must have a nonzero finite scale")
        if mp:
            kkt, rhs = mp.matrix(n+k), mp.matrix(n+k, 1)
            for i in range(n):
                rhs[i] = f[i] / objective_scale
                for j in range(n):
                    kkt[i,j] = W[i,j] / objective_scale
                for j in range(k):
                    kkt[i,n+j] = kkt[n+j,i] = scaled[j][i]
            for j in range(k):
                rhs[n+j] = rhs_values[j]
            try:
                answer = mp.lu_solve(kkt, rhs)
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError("constrained system is singular; check constraints or increase mp_dps") from exc
            q = mp.matrix([answer[i] for i in range(n)])
        else:
            C = np.asarray(scaled).T
            kkt = np.block([[W / objective_scale, C], [C.T, np.zeros((k,k))]])
            answer = np.linalg.solve(kkt, np.concatenate([f / objective_scale, rhs_values]))
            q = answer[:n]
        finite = mp.isfinite if mp else np.isfinite
        if not all(finite(x) for x in answer):
            raise FloatingPointError("constrained solve produced non-finite output")
        multipliers = [scalar(0) for _ in columns]
        for j, index in enumerate(kept):
            multipliers[index] = answer[n+j] * objective_scale / scales[j]
        return SolveResult(
            q=q, eta=scalar("nan") if area_index is None else multipliers[area_index],
            denominator=scalar("nan"), backend=backend,
            endpoint_eta=scalar(0) if endpoint_index is None else multipliers[endpoint_index],
            omega_eta=tuple(multipliers[omega_start:]),
        )


def solve_hlt_constrained(
    W: Any,
    f: Any,
    R: Any,
    c: Any,
    *,
    backend: str = "numpy",
    mp_dps: int = 80,
) -> SolveResult:
    """Solve ``W q + eta R = f`` with ``R^T q = c``.

    With ``backend='mpmath'`` MP-native inputs and outputs are retained; no
    conversion through NumPy occurs.
    """
    backend = normalize_backend(backend)
    if backend == "numpy":
        W_np = np.asarray(W, dtype=float)
        f_np = np.asarray(f, dtype=float)
        R_np = np.asarray(R, dtype=float)
        _validate_numpy_inputs(W_np, f_np, R_np, c)
        return _solve_numpy(W_np, f_np, R_np, c)
    return _solve_mpmath(W, f, R, c, mp_dps=mp_dps)


def solve_hlt_unconstrained(
    W: Any,
    f: Any,
    *,
    backend: str = "numpy",
    mp_dps: int = 80,
) -> SolveResult:
    """Solve the unconstrained HLT normal equations ``W q = f``."""
    backend = normalize_backend(backend)
    if backend == "numpy":
        W_np = np.asarray(W, dtype=float)
        f_np = np.asarray(f, dtype=float)
        if W_np.ndim != 2 or W_np.shape[0] != W_np.shape[1]:
            raise ValueError("W must be a square matrix")
        if f_np.shape != (W_np.shape[0],):
            raise ValueError("f has incompatible shape")
        if not np.all(np.isfinite(W_np)) or not np.all(np.isfinite(f_np)):
            raise ValueError("W and f must be finite")
        q = np.linalg.solve(W_np, f_np)
        if not np.all(np.isfinite(q)):
            raise FloatingPointError("unconstrained solve produced non-finite output")
        return SolveResult(
            q=q, eta=float("nan"), denominator=float("nan"), backend="numpy"
        )

    if mp_dps < 30:
        raise ValueError("mp_dps should be at least 30 for the mpmath backend")
    mp = require_mpmath()
    with mp.workdps(int(mp_dps)):
        W_mp = mp_matrix(W, mp)
        f_mp = mp_vector(source_vector(f), mp)
        _validate_mp_inputs(W_mp, f_mp, None, None, mp)
        q = mp.lu_solve(W_mp, f_mp)
        if any(not mp.isfinite(q[i]) for i in range(q.rows)):
            raise FloatingPointError("unconstrained solve produced non-finite output")
        return SolveResult(q=q, eta=mp.nan, denominator=mp.nan, backend="mpmath")


def _validate_numpy_inputs(W: np.ndarray, f: np.ndarray, R: np.ndarray, c: Any) -> None:
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError("W must be a square matrix")
    if f.shape != (W.shape[0],):
        raise ValueError("f has incompatible shape")
    if R.shape != (W.shape[0],):
        raise ValueError("R has incompatible shape")
    if (
        not np.all(np.isfinite(W))
        or not np.all(np.isfinite(f))
        or not np.all(np.isfinite(R))
        or not np.isfinite(c)
    ):
        raise ValueError("W, f, R, and c must be finite")


def _validate_mp_inputs(W, f, R, c, mp) -> None:
    if W.rows != W.cols:
        raise ValueError("W must be a square matrix")
    if f.rows != W.rows or f.cols != 1:
        raise ValueError("f has incompatible shape")
    if R is not None and (R.rows != W.rows or R.cols != 1):
        raise ValueError("R has incompatible shape")
    values = [W[i, j] for i in range(W.rows) for j in range(W.cols)]
    values.extend(f[i] for i in range(f.rows))
    if R is not None:
        values.extend(R[i] for i in range(R.rows))
    if c is not None:
        values.append(c)
    if any(not mp.isfinite(value) for value in values):
        raise ValueError("W, f, R, and c must be finite")


def _solve_numpy(W: np.ndarray, f: np.ndarray, R: np.ndarray, c: Any) -> SolveResult:
    x = np.linalg.solve(W, f)
    y = np.linalg.solve(W, R)
    denom = float(np.dot(R, y))
    if not np.isfinite(denom) or abs(denom) < 1e-300:
        raise np.linalg.LinAlgError(
            "constrained solve denominator is zero or non-finite"
        )
    alpha = float((c - np.dot(R, x)) / denom)
    q = x + alpha * y
    return SolveResult(q=q, eta=-alpha, denominator=denom, backend="numpy")


def _solve_mpmath(W: Any, f: Any, R: Any, c: Any, *, mp_dps: int) -> SolveResult:
    if mp_dps < 30:
        raise ValueError("mp_dps should be at least 30 for the mpmath backend")
    mp = require_mpmath()
    with mp.workdps(int(mp_dps)):
        W_mp = mp_matrix(W, mp)
        f_mp = mp_vector(source_vector(f), mp)
        R_mp = mp_vector(source_vector(R), mp)
        c_mp = to_mpf(c, mp)
        _validate_mp_inputs(W_mp, f_mp, R_mp, c_mp, mp)

        # The constrained solve needs two right-hand sides.  Calling
        # ``mp.lu_solve`` twice repeats the O(n^3) factorization.  HLT matrices
        # are symmetric positive definite after regularization, so use one
        # Cholesky factorization when possible and fall back to one shared LU
        # factorization for unusual custom inputs.
        with mp.extraprec(10):
            try:
                lower = mp.cholesky(W_mp)

                def solve_factored(rhs):
                    forward = mp.matrix(rhs)
                    for i in range(lower.rows):
                        forward[i] -= mp.fsum(
                            lower[i, j] * forward[j] for j in range(i)
                        )
                        forward[i] /= lower[i, i]
                    return mp.U_solve(lower.H, forward)

                x = solve_factored(f_mp)
                y = solve_factored(R_mp)
            except ValueError:
                lu, permutation = mp.LU_decomp(mp.matrix(W_mp))

                def solve_factored(rhs):
                    forward = mp.L_solve(lu, mp.matrix(rhs), permutation)
                    return mp.U_solve(lu, forward)

                x = solve_factored(f_mp)
                y = solve_factored(R_mp)
        x = +x
        y = +y
        denom = mp.fdot(R_mp, y)
        if not mp.isfinite(denom) or denom == 0:
            raise np.linalg.LinAlgError(
                "constrained solve denominator is zero or non-finite"
            )

        alpha = (c_mp - mp.fdot(R_mp, x)) / denom
        q = x + alpha * y
        eta = -alpha
        if any(not mp.isfinite(q[i]) for i in range(q.rows)) or not mp.isfinite(eta):
            raise FloatingPointError("mpmath solve produced non-finite output")
        return SolveResult(q=q, eta=eta, denominator=denom, backend="mpmath")
