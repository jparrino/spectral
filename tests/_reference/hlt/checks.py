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

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .precision import mp_inf_norm_vector


@dataclass
class SanityReport:
    area_abs: Any
    area_rel: Any
    stationarity_rel: Any
    variance: Any
    condition_number: Any
    max_abs_q: Any
    mismatch_abs: Any | None = None
    mismatch_rel: Any | None = None
    warnings: list[str] = field(default_factory=list)
    backend: str = "numpy"
    endpoint_abs: Any | None = None
    endpoint_rel: Any | None = None
    omega_abs: tuple = ()
    omega_rel: tuple = ()

    @property
    def ok(self) -> bool:
        return len(self.warnings) == 0

    def to_float(self) -> SanityReport:
        def convert(value):
            return None if value is None else float(value)

        return SanityReport(
            area_abs=convert(self.area_abs),
            area_rel=convert(self.area_rel),
            stationarity_rel=convert(self.stationarity_rel),
            variance=convert(self.variance),
            condition_number=convert(self.condition_number),
            max_abs_q=convert(self.max_abs_q),
            mismatch_abs=convert(self.mismatch_abs),
            mismatch_rel=convert(self.mismatch_rel),
            warnings=list(self.warnings),
            backend="numpy",
            endpoint_abs=convert(self.endpoint_abs),
            endpoint_rel=convert(self.endpoint_rel),
            omega_abs=tuple(float(x) for x in self.omega_abs),
            omega_rel=tuple(float(x) for x in self.omega_rel),
        )


def make_report(
    *,
    q: np.ndarray,
    W: np.ndarray,
    f: np.ndarray,
    R: np.ndarray,
    c: float,
    eta: float,
    covariance: np.ndarray,
    area_constraint: bool = True,
    mismatch_abs: float | None = None,
    mismatch_rel: float | None = None,
    area_rel_tol: float = 1e-8,
    stationarity_tol: float = 1e-8,
    condition_warn: float = 1e20,
    variance_tol: float = 1e-14,
    endpoint_vector=None,
    endpoint_value=None,
    endpoint_eta=0,
    omega_vectors=(),
    omega_values=(),
    omega_eta=(),
) -> SanityReport:
    eps = np.finfo(float).eps
    if area_constraint:
        area_abs = float(abs(np.dot(R, q) - c))
        area_rel = float(area_abs / max(abs(c), eps))
        stationarity = W @ q + eta * R - f
    else:
        area_abs = float("nan")
        area_rel = float("nan")
        stationarity = W @ q - f
    endpoint_abs = endpoint_rel = None
    if endpoint_vector is not None:
        v = np.asarray(endpoint_vector, dtype=float)
        stationarity += endpoint_eta * v
        endpoint_abs = float(abs(v @ q - endpoint_value))
        endpoint_rel = endpoint_abs / max(abs(endpoint_value), np.sum(np.abs(v * q)), eps)
    omega_abs, omega_rel = [], []
    for v, value, multiplier in zip(omega_vectors, omega_values, omega_eta):
        stationarity += multiplier * v
        residual = float(abs(v @ q - value))
        omega_abs.append(residual)
        omega_rel.append(residual / max(abs(value), np.sum(np.abs(v * q)), eps))
    stationarity_rel = float(np.linalg.norm(stationarity) / max(np.linalg.norm(f), eps))
    variance = float(q.T @ covariance @ q)
    if omega_vectors:
        from .solver import constraint_columns
        columns = ([R] if area_constraint else []) + ([endpoint_vector] if endpoint_vector is not None else []) + list(omega_vectors)
        _, scaled, _, _ = constraint_columns(columns, [0.] * len(columns))
        if scaled:
            C = np.asarray(scaled).T
            matrix = np.block([[W / np.max(np.abs(W)), C], [C.T, np.zeros((C.shape[1], C.shape[1]))]])
        else:
            matrix = W
        condition_number = float(np.linalg.cond(matrix))
    else:
        condition_number = float(np.linalg.cond(W))
    max_abs_q = float(np.max(np.abs(q)))

    warnings: list[str] = []
    for i, residual in enumerate(omega_rel):
        if residual > area_rel_tol:
            warnings.append(f"omega constraint {i} scaled residual {residual:.3e} exceeds {area_rel_tol:.3e}")
    if endpoint_rel is not None and endpoint_rel > area_rel_tol:
        warnings.append(f"endpoint constraint scaled residual {endpoint_rel:.3e} exceeds {area_rel_tol:.3e}")
    if area_constraint and area_rel > area_rel_tol:
        warnings.append(
            "area constraint relative residual "
            f"{area_rel:.3e} exceeds {area_rel_tol:.3e}"
        )
    if stationarity_rel > stationarity_tol:
        warnings.append(
            "stationarity relative residual "
            f"{stationarity_rel:.3e} exceeds {stationarity_tol:.3e}"
        )
    if variance < -variance_tol:
        warnings.append(f"variance is negative: {variance:.3e}")
    if condition_number > condition_warn:
        warnings.append(f"condition number is large: {condition_number:.3e}")

    return SanityReport(
        area_abs=area_abs,
        area_rel=area_rel,
        stationarity_rel=stationarity_rel,
        variance=variance,
        condition_number=condition_number,
        max_abs_q=max_abs_q,
        mismatch_abs=mismatch_abs,
        mismatch_rel=mismatch_rel,
        warnings=warnings,
        backend="numpy",
        endpoint_abs=endpoint_abs,
        endpoint_rel=endpoint_rel,
        omega_abs=tuple(omega_abs),
        omega_rel=tuple(omega_rel),
    )


def make_report_mpmath(
    *,
    q,
    W,
    f,
    R,
    c,
    eta,
    covariance,
    mp,
    area_constraint: bool = True,
    mismatch_abs=None,
    mismatch_rel=None,
    condition_number=None,
    endpoint_vector=None,
    endpoint_value=None,
    endpoint_eta=0,
    omega_vectors=(),
    omega_values=(),
    omega_eta=(),
) -> SanityReport:
    """Build sanity diagnostics without converting the MP system to NumPy."""
    tiny = mp.power(10, -(mp.dps - 5))
    if area_constraint:
        area_abs = abs(mp.fdot(R, q) - c)
        area_rel = area_abs / max(abs(c), tiny)
        stationarity = W * q + eta * R - f
    else:
        area_abs = mp.nan
        area_rel = mp.nan
        stationarity = W * q - f
    endpoint_abs = endpoint_rel = None
    if endpoint_vector is not None:
        stationarity += endpoint_eta * endpoint_vector
        endpoint_abs = abs(mp.fdot(endpoint_vector, q) - endpoint_value)
        endpoint_rel = endpoint_abs / max(
            abs(endpoint_value), mp.fsum(abs(endpoint_vector[i] * q[i]) for i in range(q.rows)), tiny
        )
    omega_abs, omega_rel = [], []
    for v, value, multiplier in zip(omega_vectors, omega_values, omega_eta):
        stationarity += multiplier * v
        residual = abs(mp.fdot(v, q) - value)
        omega_abs.append(residual)
        omega_rel.append(residual / max(abs(value), mp.fsum(abs(v[i] * q[i]) for i in range(q.rows)), tiny))
    stationarity_rel = mp_inf_norm_vector(stationarity, mp) / max(
        mp_inf_norm_vector(f, mp), tiny
    )
    variance = (q.T * covariance * q)[0]

    if condition_number is None:
        columns = [R] if area_constraint else []
        if endpoint_vector is not None and any(endpoint_vector):
            columns.append(endpoint_vector)
        columns.extend(omega_vectors)
        from .solver import constraint_columns
        kept, _, _, _ = constraint_columns(columns, [mp.zero] * len(columns), mp=mp)
        columns = [columns[i] for i in kept]
        if columns:
            size = W.rows
            kkt = mp.matrix(size + len(columns), size + len(columns))
            for i in range(size):
                for j in range(size):
                    kkt[i, j] = W[i, j]
                for j, col in enumerate(columns):
                    kkt[i, size + j] = col[i]
                    kkt[size + j, i] = col[i]
            condition_number = mp.cond(kkt)
        else:
            condition_number = mp.cond(W)
    max_abs_q = max(abs(q[i]) for i in range(q.rows))

    residual_tol = mp.power(10, -max(10, min(30, mp.dps // 2)))
    warnings: list[str] = []
    for i, residual in enumerate(omega_rel):
        if residual > residual_tol:
            warnings.append(f"omega constraint {i} scaled residual {mp.nstr(residual, 6)} exceeds {mp.nstr(residual_tol, 3)}")
    if endpoint_rel is not None and endpoint_rel > residual_tol:
        warnings.append(
            f"endpoint constraint scaled residual {mp.nstr(endpoint_rel, 6)} exceeds {mp.nstr(residual_tol, 3)}"
        )
    if area_constraint and area_rel > residual_tol:
        warnings.append(
            "area constraint relative residual "
            f"{mp.nstr(area_rel, 6)} exceeds {mp.nstr(residual_tol, 3)}"
        )
    if stationarity_rel > residual_tol:
        warnings.append(
            "stationarity relative residual "
            f"{mp.nstr(stationarity_rel, 6)} exceeds {mp.nstr(residual_tol, 3)}"
        )
    if variance < -residual_tol:
        warnings.append(f"variance is negative: {mp.nstr(variance, 6)}")
    if condition_number > mp.power(10, mp.dps - 10):
        warnings.append(
            f"condition number {mp.nstr(condition_number, 6)} leaves fewer than "
            "ten nominal decimal digits"
        )

    return SanityReport(
        area_abs=area_abs,
        area_rel=area_rel,
        stationarity_rel=stationarity_rel,
        variance=variance,
        condition_number=condition_number,
        max_abs_q=max_abs_q,
        mismatch_abs=mismatch_abs,
        mismatch_rel=mismatch_rel,
        warnings=warnings,
        backend="mpmath",
        endpoint_abs=endpoint_abs,
        endpoint_rel=endpoint_rel,
        omega_abs=tuple(omega_abs),
        omega_rel=tuple(omega_rel),
    )
