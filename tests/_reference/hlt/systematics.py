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

from .basis import HLTBasis
from .precision import (
    mp_matrix_to_numpy,
    mp_trapezoid_weights,
    require_mpmath,
    source_vector,
    to_mpf,
)
from .targets import TargetKernel, trapezoid_weights


@dataclass(frozen=True)
class KernelMismatchGrid:
    """Grid-level kernel reconstruction diagnostics.

    The arrays have the same shape as ``omega``.  ``weighted_mismatch`` is
    populated only when a model spectral density ``rho_p`` is provided.
    """

    omega: Any
    target: Any
    reconstructed: Any
    mismatch: Any
    relative_mismatch: Any
    weights: Any | None = None
    rho_p: Any | None = None
    weighted_target: Any | None = None
    weighted_reconstructed: Any | None = None
    weighted_mismatch: Any | None = None
    backend: str = "numpy"
    mp_dps: int | None = None

    def to_numpy(self) -> KernelMismatchGrid:
        if self.backend == "numpy":
            return self

        def convert(value):
            return None if value is None else mp_matrix_to_numpy(value)

        return KernelMismatchGrid(
            omega=convert(self.omega),
            target=convert(self.target),
            reconstructed=convert(self.reconstructed),
            mismatch=convert(self.mismatch),
            relative_mismatch=convert(self.relative_mismatch),
            weights=convert(self.weights),
            rho_p=convert(self.rho_p),
            weighted_target=convert(self.weighted_target),
            weighted_reconstructed=convert(self.weighted_reconstructed),
            weighted_mismatch=convert(self.weighted_mismatch),
            backend="numpy",
        )


@dataclass(frozen=True)
class KernelSystematic:
    """Observable-level estimate of kernel-reconstruction bias.

    The estimator assumes that ``rho_p`` is the spectral density in the same
    representation as the HLT target, i.e. the observable is

        O = int d omega T(omega) rho_p(omega).

    For the p=2 convention used in the two-pion/timelike tests,

        rho_p(omega) = rho^{I=1}(omega^2) = R(omega^2)/(12*pi^2).

    The signed bias allows cancellations in omega.  The absolute bias is the
    conservative bound-like quantity

        int d omega |T_hat(omega)-T(omega)| |rho_p(omega)|.
    """

    signed_bias: Any
    absolute_bias: Any
    relative_signed_bias: Any
    relative_absolute_bias: Any
    target_observable: Any
    reconstructed_observable: Any
    stat_error: Any | None = None
    total_error_with_absolute_bias: Any | None = None
    backend: str = "numpy"
    mp_dps: int | None = None

    def to_float(self) -> KernelSystematic:
        def convert(value):
            return None if value is None else float(value)

        return KernelSystematic(
            signed_bias=convert(self.signed_bias),
            absolute_bias=convert(self.absolute_bias),
            relative_signed_bias=convert(self.relative_signed_bias),
            relative_absolute_bias=convert(self.relative_absolute_bias),
            target_observable=convert(self.target_observable),
            reconstructed_observable=convert(self.reconstructed_observable),
            stat_error=convert(self.stat_error),
            total_error_with_absolute_bias=convert(self.total_error_with_absolute_bias),
            backend="numpy",
        )


def kernel_mismatch_on_grid(
    *,
    basis: HLTBasis,
    target: TargetKernel,
    q: np.ndarray,
    omega: np.ndarray,
    rho_p: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    relative_floor: float = 1e-14,
    strict_omega0: bool = True,
    backend: str = "numpy",
    mp_dps: int = 80,
    target_backend: str | None = None,
) -> KernelMismatchGrid:
    """Evaluate target, reconstructed kernel, and mismatch on an omega grid.

    Parameters
    ----------
    basis, target, q:
        The HLT basis, target kernel, and reconstructed coefficient vector.
    omega:
        One-dimensional energy grid.  By default all points must satisfy
        ``omega >= basis.omega0`` because the HLT problem is defined on
        ``[omega0, infinity)``.
    rho_p:
        Optional spectral density values in the same p-representation as the
        target.  If supplied, the returned object also contains the pointwise
        weighted mismatch ``(T_hat - T) * rho_p``.
    weights:
        Optional quadrature weights for ``omega``.  These are not needed for
        plotting, but are stored when supplied so the same grid object can also
        be used for integrated diagnostics.
    relative_floor:
        Values with ``abs(target) <= relative_floor`` get ``nan`` for the
        relative mismatch to avoid meaningless infinities.
    strict_omega0:
        If True, raise an error when the grid contains points below
        ``basis.omega0``.  If False, values are still returned, but such points
        are outside the minimization domain and should be interpreted with care.
    """
    if backend == "mpmath":
        return _kernel_mismatch_on_grid_mpmath(
            basis=basis,
            target=target,
            q=q,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            relative_floor=relative_floor,
            strict_omega0=strict_omega0,
            mp_dps=mp_dps,
            target_backend=target_backend or "mpmath",
        )
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'mpmath'")
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(q, dtype=float)

    if omega.ndim != 1:
        raise ValueError("omega must be one-dimensional")
    if len(omega) == 0:
        raise ValueError("omega grid must be non-empty")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega contains non-finite values")
    if np.any(omega < 0):
        raise ValueError("omega must be non-negative")
    if strict_omega0 and np.any(omega < basis.omega0):
        raise ValueError("omega grid contains points below basis.omega0")
    if relative_floor <= 0 or not np.isfinite(relative_floor):
        raise ValueError("relative_floor must be positive and finite")

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != omega.shape:
            raise ValueError("weights must have the same shape as omega")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contains non-finite values")

    reconstructed = basis.reconstructed_kernel(q, omega)
    target_values = np.asarray(target.eval(omega), dtype=float)
    mismatch = reconstructed - target_values

    relative_mismatch = np.full_like(mismatch, np.nan, dtype=float)
    mask = np.abs(target_values) > relative_floor
    relative_mismatch[mask] = mismatch[mask] / target_values[mask]

    rho_arr = None
    weighted_target = None
    weighted_reconstructed = None
    weighted_mismatch = None
    if rho_p is not None:
        rho_arr = np.asarray(rho_p, dtype=float)
        if rho_arr.shape != omega.shape:
            raise ValueError("rho_p must have the same shape as omega")
        if not np.all(np.isfinite(rho_arr)):
            raise ValueError("rho_p contains non-finite values")
        weighted_target = target_values * rho_arr
        weighted_reconstructed = reconstructed * rho_arr
        weighted_mismatch = mismatch * rho_arr

    return KernelMismatchGrid(
        omega=omega,
        target=target_values,
        reconstructed=reconstructed,
        mismatch=mismatch,
        relative_mismatch=relative_mismatch,
        weights=weights,
        rho_p=rho_arr,
        weighted_target=weighted_target,
        weighted_reconstructed=weighted_reconstructed,
        weighted_mismatch=weighted_mismatch,
        backend="numpy",
    )


def estimate_kernel_systematic_on_grid(
    *,
    basis: HLTBasis,
    target: TargetKernel,
    q: np.ndarray,
    omega: np.ndarray,
    rho_p: np.ndarray,
    weights: np.ndarray | None = None,
    stat_error: float | None = None,
    backend: str = "numpy",
    mp_dps: int = 80,
    target_backend: str | None = None,
) -> KernelSystematic:
    """Estimate kernel-reconstruction systematic on a quadrature grid.

    Parameters
    ----------
    basis, target, q:
        The HLT basis, target kernel, and reconstructed coefficient vector.
    omega:
        One-dimensional omega grid.
    rho_p:
        Spectral density values in the same representation as the HLT target.
        In other words, the target observable is approximated as
        ``sum weights * target.eval(omega) * rho_p``.
    weights:
        Optional quadrature weights for the omega grid.  These must include the
        integration measure d omega.  If omitted, trapezoid-rule weights are
        constructed automatically from ``omega``.  The omega grid must then be
        strictly increasing.
    stat_error:
        Optional statistical error of the HLT estimator.  If supplied, the
        returned object also contains ``sqrt(stat_error**2 + absolute_bias**2)``.

    Returns
    -------
    KernelSystematic
        Signed and absolute bias estimates, plus relative versions normalized
        by the target observable estimated with ``rho_p``.
    """
    if backend == "mpmath":
        mp = require_mpmath()
        with mp.workdps(mp_dps):
            effective_weights = weights
            if effective_weights is None:
                omega_values = _mp_list(omega, mp)
                effective_weights = mp.matrix(mp_trapezoid_weights(omega_values, mp))
            grid = _kernel_mismatch_on_grid_mpmath(
                basis=basis,
                target=target,
                q=q,
                omega=omega,
                rho_p=rho_p,
                weights=effective_weights,
                relative_floor="1e-14",
                strict_omega0=True,
                mp_dps=mp_dps,
                target_backend=target_backend or "mpmath",
            )
            return _systematic_from_mp_grid(grid, stat_error, mp, mp_dps)
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'mpmath'")
    omega = np.asarray(omega, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    if weights is None:
        weights = trapezoid_weights(omega)
    else:
        weights = np.asarray(weights, dtype=float)

    if weights.shape != omega.shape:
        raise ValueError("weights must have the same shape as omega")
    if rho_p.shape != omega.shape:
        raise ValueError("rho_p must have the same shape as omega")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contains non-finite values")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative for the absolute-bias estimate")
    if stat_error is not None and (not np.isfinite(stat_error) or stat_error < 0):
        raise ValueError("stat_error must be non-negative and finite")

    grid = kernel_mismatch_on_grid(
        basis=basis,
        target=target,
        q=q,
        omega=omega,
        rho_p=rho_p,
        weights=weights,
        strict_omega0=True,
    )

    assert grid.weighted_target is not None
    assert grid.weighted_reconstructed is not None
    assert grid.weighted_mismatch is not None

    target_observable = float(np.sum(weights * grid.weighted_target))
    reconstructed_observable = float(np.sum(weights * grid.weighted_reconstructed))
    signed_bias = float(np.sum(weights * grid.weighted_mismatch))
    absolute_bias = float(np.sum(weights * np.abs(grid.mismatch) * np.abs(rho_p)))

    denom = max(abs(target_observable), np.finfo(float).eps)
    relative_signed_bias = float(signed_bias / denom)
    relative_absolute_bias = float(absolute_bias / denom)

    total = None
    if stat_error is not None:
        total = float(np.sqrt(stat_error**2 + absolute_bias**2))

    return KernelSystematic(
        signed_bias=signed_bias,
        absolute_bias=absolute_bias,
        relative_signed_bias=relative_signed_bias,
        relative_absolute_bias=relative_absolute_bias,
        target_observable=target_observable,
        reconstructed_observable=reconstructed_observable,
        stat_error=None if stat_error is None else float(stat_error),
        total_error_with_absolute_bias=total,
        backend="numpy",
    )


def estimate_target_difference_on_grid(
    *,
    target_a: TargetKernel,
    target_b: TargetKernel,
    omega: np.ndarray,
    rho_p: np.ndarray,
    weights: np.ndarray | None = None,
    stat_error: float | None = None,
    backend: str = "numpy",
    mp_dps: int = 80,
    target_backend: str | None = None,
) -> KernelSystematic:
    """Estimate an observable-level weighted difference between two target kernels.

    The signed difference is

        int d omega [target_a(omega) - target_b(omega)] rho_p(omega).

    The absolute difference is the conservative bound-like quantity

        int d omega |target_a(omega) - target_b(omega)| |rho_p(omega)|.

    ``target_b`` is treated as the reference target for the relative quantities
    and is stored as ``target_observable`` in the returned ``KernelSystematic``.
    ``target_a`` is stored as ``reconstructed_observable`` for compatibility
    with the existing ``KernelSystematic`` container.

    If ``weights`` is omitted, trapezoid-rule weights are built from ``omega``.
    """
    if backend == "mpmath":
        return _estimate_target_difference_mpmath(
            target_a=target_a,
            target_b=target_b,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            stat_error=stat_error,
            mp_dps=mp_dps,
            target_backend=target_backend or "mpmath",
        )
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'mpmath'")
    omega = np.asarray(omega, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    if weights is None:
        weights = trapezoid_weights(omega)
    else:
        weights = np.asarray(weights, dtype=float)

    if omega.ndim != 1:
        raise ValueError("omega must be one-dimensional")
    if len(omega) == 0:
        raise ValueError("omega grid must be non-empty")
    if weights.shape != omega.shape:
        raise ValueError("weights must have the same shape as omega")
    if rho_p.shape != omega.shape:
        raise ValueError("rho_p must have the same shape as omega")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega contains non-finite values")
    if np.any(omega < 0):
        raise ValueError("omega must be non-negative")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contains non-finite values")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative for the absolute-bias estimate")
    if not np.all(np.isfinite(rho_p)):
        raise ValueError("rho_p contains non-finite values")
    if stat_error is not None and (not np.isfinite(stat_error) or stat_error < 0):
        raise ValueError("stat_error must be non-negative and finite")

    values_a = np.asarray(target_a.eval(omega), dtype=float)
    values_b = np.asarray(target_b.eval(omega), dtype=float)
    if values_a.shape != omega.shape:
        values_a = np.broadcast_to(values_a, omega.shape).astype(float)
    if values_b.shape != omega.shape:
        values_b = np.broadcast_to(values_b, omega.shape).astype(float)
    if not np.all(np.isfinite(values_a)):
        raise ValueError("target_a evaluates to non-finite values")
    if not np.all(np.isfinite(values_b)):
        raise ValueError("target_b evaluates to non-finite values")

    diff = values_a - values_b
    target_observable = float(np.sum(weights * values_b * rho_p))
    reconstructed_observable = float(np.sum(weights * values_a * rho_p))
    signed_bias = float(np.sum(weights * diff * rho_p))
    absolute_bias = float(np.sum(weights * np.abs(diff) * np.abs(rho_p)))

    denom = max(abs(target_observable), np.finfo(float).eps)
    relative_signed_bias = float(signed_bias / denom)
    relative_absolute_bias = float(absolute_bias / denom)

    total = None
    if stat_error is not None:
        total = float(np.sqrt(stat_error**2 + absolute_bias**2))

    return KernelSystematic(
        signed_bias=signed_bias,
        absolute_bias=absolute_bias,
        relative_signed_bias=relative_signed_bias,
        relative_absolute_bias=relative_absolute_bias,
        target_observable=target_observable,
        reconstructed_observable=reconstructed_observable,
        stat_error=None if stat_error is None else float(stat_error),
        total_error_with_absolute_bias=total,
        backend="numpy",
    )


def estimate_reconstructed_target_difference_on_grid(
    *,
    basis: HLTBasis,
    q: np.ndarray,
    reference_target: TargetKernel,
    omega: np.ndarray,
    rho_p: np.ndarray,
    weights: np.ndarray | None = None,
    stat_error: float | None = None,
    backend: str = "numpy",
    mp_dps: int = 80,
    target_backend: str | None = None,
) -> KernelSystematic:
    """Estimate weighted difference between a reconstructed kernel and a target.

    The signed difference is

        int d omega [T_hat(omega) - reference_target(omega)] rho_p(omega).

    The absolute difference is the conservative bound-like quantity

        int d omega |T_hat(omega) - reference_target(omega)| |rho_p(omega)|.

    ``reference_target`` is treated as the reference target for the relative
    quantities and is stored as ``target_observable`` in the returned
    ``KernelSystematic``.  The reconstructed-kernel observable is stored as
    ``reconstructed_observable``.

    If ``weights`` is omitted, trapezoid-rule weights are built from ``omega``.
    """
    if backend == "mpmath":
        return _estimate_reconstructed_difference_mpmath(
            basis=basis,
            q=q,
            reference_target=reference_target,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            stat_error=stat_error,
            mp_dps=mp_dps,
            target_backend=target_backend or "mpmath",
        )
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'mpmath'")
    omega = np.asarray(omega, dtype=float)
    rho_p = np.asarray(rho_p, dtype=float)
    q = np.asarray(q, dtype=float)
    if weights is None:
        weights = trapezoid_weights(omega)
    else:
        weights = np.asarray(weights, dtype=float)

    if omega.ndim != 1:
        raise ValueError("omega must be one-dimensional")
    if len(omega) == 0:
        raise ValueError("omega grid must be non-empty")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega contains non-finite values")
    if np.any(omega < 0):
        raise ValueError("omega must be non-negative")
    if np.any(omega < basis.omega0):
        raise ValueError("omega grid contains points below basis.omega0")
    if weights.shape != omega.shape:
        raise ValueError("weights must have the same shape as omega")
    if rho_p.shape != omega.shape:
        raise ValueError("rho_p must have the same shape as omega")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weights contains non-finite values")
    if np.any(weights < 0):
        raise ValueError("weights must be non-negative for the absolute-bias estimate")
    if not np.all(np.isfinite(rho_p)):
        raise ValueError("rho_p contains non-finite values")
    if stat_error is not None and (not np.isfinite(stat_error) or stat_error < 0):
        raise ValueError("stat_error must be non-negative and finite")

    reconstructed = np.asarray(basis.reconstructed_kernel(q, omega), dtype=float)
    reference_values = np.asarray(reference_target.eval(omega), dtype=float)
    if reconstructed.shape != omega.shape:
        reconstructed = np.broadcast_to(reconstructed, omega.shape).astype(float)
    if reference_values.shape != omega.shape:
        reference_values = np.broadcast_to(reference_values, omega.shape).astype(float)
    if not np.all(np.isfinite(reconstructed)):
        raise ValueError("reconstructed kernel contains non-finite values")
    if not np.all(np.isfinite(reference_values)):
        raise ValueError("reference_target evaluates to non-finite values")

    diff = reconstructed - reference_values
    target_observable = float(np.sum(weights * reference_values * rho_p))
    reconstructed_observable = float(np.sum(weights * reconstructed * rho_p))
    signed_bias = float(np.sum(weights * diff * rho_p))
    absolute_bias = float(np.sum(weights * np.abs(diff) * np.abs(rho_p)))

    denom = max(abs(target_observable), np.finfo(float).eps)
    relative_signed_bias = float(signed_bias / denom)
    relative_absolute_bias = float(absolute_bias / denom)

    total = None
    if stat_error is not None:
        total = float(np.sqrt(stat_error**2 + absolute_bias**2))

    return KernelSystematic(
        signed_bias=signed_bias,
        absolute_bias=absolute_bias,
        relative_signed_bias=relative_signed_bias,
        relative_absolute_bias=relative_absolute_bias,
        target_observable=target_observable,
        reconstructed_observable=reconstructed_observable,
        stat_error=None if stat_error is None else float(stat_error),
        total_error_with_absolute_bias=total,
        backend="numpy",
    )


def _mp_list(values, mp):
    return [to_mpf(value, mp) for value in source_vector(values)]


def _mp_grid_inputs(omega, rho_p, weights, mp):
    omega_values = _mp_list(omega, mp)
    if len(omega_values) == 0:
        raise ValueError("omega grid must be non-empty")
    if any(not mp.isfinite(value) for value in omega_values):
        raise ValueError("omega contains non-finite values")
    if any(value < 0 for value in omega_values):
        raise ValueError("omega must be non-negative")

    if weights is None:
        weight_values = mp_trapezoid_weights(omega_values, mp)
    else:
        weight_values = _mp_list(weights, mp)
    if len(weight_values) != len(omega_values):
        raise ValueError("weights must have the same shape as omega")
    if any(not mp.isfinite(value) for value in weight_values):
        raise ValueError("weights contains non-finite values")

    rho_values = None if rho_p is None else _mp_list(rho_p, mp)
    if rho_values is not None:
        if len(rho_values) != len(omega_values):
            raise ValueError("rho_p must have the same shape as omega")
        if any(not mp.isfinite(value) for value in rho_values):
            raise ValueError("rho_p contains non-finite values")
    return omega_values, rho_values, weight_values


def _kernel_mismatch_on_grid_mpmath(
    *,
    basis,
    target,
    q,
    omega,
    rho_p,
    weights,
    relative_floor,
    strict_omega0,
    mp_dps,
    target_backend,
):
    from .mpmath_backend import MPWorkspace, reconstructed_kernel, target_eval

    mp = require_mpmath()
    with mp.workdps(mp_dps):
        omega_values, rho_values, weight_values = _mp_grid_inputs(
            omega, rho_p, weights, mp
        )
        omega0 = to_mpf(basis._omega0_source, mp)
        if strict_omega0 and any(value < omega0 for value in omega_values):
            raise ValueError("omega grid contains points below basis.omega0")
        floor = to_mpf(relative_floor, mp)
        if floor <= 0 or not mp.isfinite(floor):
            raise ValueError("relative_floor must be positive and finite")

        workspace = MPWorkspace(mp_dps)
        reconstructed_values = [
            reconstructed_kernel(basis, q, value, mp) for value in omega_values
        ]
        if target_backend == "numpy":
            target_values = [
                to_mpf(float(target.eval(float(value))), mp)
                for value in omega_values
            ]
        elif target_backend == "mpmath":
            target_values = [
                target_eval(target, value, mp, workspace) for value in omega_values
            ]
        else:
            raise ValueError("target_backend must be 'numpy' or 'mpmath'")
        mismatch_values = [
            rec - tar for rec, tar in zip(reconstructed_values, target_values)
        ]
        relative_values = [
            diff / tar if abs(tar) > floor else mp.nan
            for diff, tar in zip(mismatch_values, target_values)
        ]

        weighted_target = weighted_reconstructed = weighted_mismatch = None
        if rho_values is not None:
            weighted_target = mp.matrix(
                [tar * rho for tar, rho in zip(target_values, rho_values)]
            )
            weighted_reconstructed = mp.matrix(
                [rec * rho for rec, rho in zip(reconstructed_values, rho_values)]
            )
            weighted_mismatch = mp.matrix(
                [diff * rho for diff, rho in zip(mismatch_values, rho_values)]
            )

        return KernelMismatchGrid(
            omega=mp.matrix(omega_values),
            target=mp.matrix(target_values),
            reconstructed=mp.matrix(reconstructed_values),
            mismatch=mp.matrix(mismatch_values),
            relative_mismatch=mp.matrix(relative_values),
            weights=None if weights is None else mp.matrix(weight_values),
            rho_p=None if rho_values is None else mp.matrix(rho_values),
            weighted_target=weighted_target,
            weighted_reconstructed=weighted_reconstructed,
            weighted_mismatch=weighted_mismatch,
            backend="mpmath",
            mp_dps=mp_dps,
        )


def _systematic_from_mp_grid(grid, stat_error, mp, mp_dps):
    if grid.rho_p is None:
        raise ValueError("rho_p is required for an integrated systematic")
    weights = [grid.weights[i] for i in range(grid.weights.rows)]
    target = [grid.target[i] for i in range(grid.target.rows)]
    reconstructed = [grid.reconstructed[i] for i in range(grid.reconstructed.rows)]
    mismatch = [grid.mismatch[i] for i in range(grid.mismatch.rows)]
    rho = [grid.rho_p[i] for i in range(grid.rho_p.rows)]
    if any(value < 0 for value in weights):
        raise ValueError("weights must be non-negative for the absolute-bias estimate")
    target_observable = mp.fsum(
        w * value * density for w, value, density in zip(weights, target, rho)
    )
    reconstructed_observable = mp.fsum(
        w * value * density for w, value, density in zip(weights, reconstructed, rho)
    )
    signed_bias = mp.fsum(
        w * value * density for w, value, density in zip(weights, mismatch, rho)
    )
    absolute_bias = mp.fsum(
        w * abs(value) * abs(density)
        for w, value, density in zip(weights, mismatch, rho)
    )
    floor = mp.power(10, -(mp.dps - 5))
    denom = max(abs(target_observable), floor)
    stat = None if stat_error is None else to_mpf(stat_error, mp)
    if stat is not None and (not mp.isfinite(stat) or stat < 0):
        raise ValueError("stat_error must be non-negative and finite")
    total = (
        None if stat is None else mp.sqrt(stat * stat + absolute_bias * absolute_bias)
    )
    return KernelSystematic(
        signed_bias=signed_bias,
        absolute_bias=absolute_bias,
        relative_signed_bias=signed_bias / denom,
        relative_absolute_bias=absolute_bias / denom,
        target_observable=target_observable,
        reconstructed_observable=reconstructed_observable,
        stat_error=stat,
        total_error_with_absolute_bias=total,
        backend="mpmath",
        mp_dps=mp_dps,
    )


def _systematic_from_mp_values(
    values_a, values_b, omega, rho_p, weights, stat_error, mp, mp_dps
):
    omega_values, rho_values, weight_values = _mp_grid_inputs(omega, rho_p, weights, mp)
    assert rho_values is not None
    if len(values_a) != len(omega_values) or len(values_b) != len(omega_values):
        raise ValueError("target values must have the same shape as omega")
    if any(value < 0 for value in weight_values):
        raise ValueError("weights must be non-negative for the absolute-bias estimate")
    if any(not mp.isfinite(value) for value in values_a + values_b):
        raise ValueError("target evaluates to non-finite values")
    diff = [a - b for a, b in zip(values_a, values_b)]
    target_observable = mp.fsum(
        w * b * rho for w, b, rho in zip(weight_values, values_b, rho_values)
    )
    reconstructed_observable = mp.fsum(
        w * a * rho for w, a, rho in zip(weight_values, values_a, rho_values)
    )
    signed_bias = mp.fsum(
        w * d * rho for w, d, rho in zip(weight_values, diff, rho_values)
    )
    absolute_bias = mp.fsum(
        w * abs(d) * abs(rho) for w, d, rho in zip(weight_values, diff, rho_values)
    )
    floor = mp.power(10, -(mp.dps - 5))
    denom = max(abs(target_observable), floor)
    stat = None if stat_error is None else to_mpf(stat_error, mp)
    if stat is not None and (not mp.isfinite(stat) or stat < 0):
        raise ValueError("stat_error must be non-negative and finite")
    return KernelSystematic(
        signed_bias=signed_bias,
        absolute_bias=absolute_bias,
        relative_signed_bias=signed_bias / denom,
        relative_absolute_bias=absolute_bias / denom,
        target_observable=target_observable,
        reconstructed_observable=reconstructed_observable,
        stat_error=stat,
        total_error_with_absolute_bias=None
        if stat is None
        else mp.sqrt(stat * stat + absolute_bias * absolute_bias),
        backend="mpmath",
        mp_dps=mp_dps,
    )


def _estimate_target_difference_mpmath(
    *,
    target_a,
    target_b,
    omega,
    rho_p,
    weights,
    stat_error,
    mp_dps,
    target_backend,
):
    from .mpmath_backend import MPWorkspace, target_eval

    mp = require_mpmath()
    with mp.workdps(mp_dps):
        omega_values = _mp_list(omega, mp)
        workspace = MPWorkspace(mp_dps)
        if target_backend == "numpy":
            values_a = [
                to_mpf(float(target_a.eval(float(value))), mp)
                for value in omega_values
            ]
            values_b = [
                to_mpf(float(target_b.eval(float(value))), mp)
                for value in omega_values
            ]
        elif target_backend == "mpmath":
            values_a = [
                target_eval(target_a, value, mp, workspace)
                for value in omega_values
            ]
            values_b = [
                target_eval(target_b, value, mp, workspace)
                for value in omega_values
            ]
        else:
            raise ValueError("target_backend must be 'numpy' or 'mpmath'")
        return _systematic_from_mp_values(
            values_a, values_b, omega_values, rho_p, weights, stat_error, mp, mp_dps
        )


def _estimate_reconstructed_difference_mpmath(
    *,
    basis,
    q,
    reference_target,
    omega,
    rho_p,
    weights,
    stat_error,
    mp_dps,
    target_backend,
):
    from .mpmath_backend import MPWorkspace, reconstructed_kernel, target_eval

    mp = require_mpmath()
    with mp.workdps(mp_dps):
        omega_values = _mp_list(omega, mp)
        omega0 = to_mpf(basis._omega0_source, mp)
        if any(value < omega0 for value in omega_values):
            raise ValueError("omega grid contains points below basis.omega0")
        workspace = MPWorkspace(mp_dps)
        reconstructed = [
            reconstructed_kernel(basis, q, value, mp) for value in omega_values
        ]
        if target_backend == "numpy":
            reference = [
                to_mpf(float(reference_target.eval(float(value))), mp)
                for value in omega_values
            ]
        elif target_backend == "mpmath":
            reference = [
                target_eval(reference_target, value, mp, workspace)
                for value in omega_values
            ]
        else:
            raise ValueError("target_backend must be 'numpy' or 'mpmath'")
        return _systematic_from_mp_values(
            reconstructed,
            reference,
            omega_values,
            rho_p,
            weights,
            stat_error,
            mp,
            mp_dps,
        )
