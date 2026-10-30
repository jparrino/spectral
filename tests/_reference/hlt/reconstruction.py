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
from scipy.integrate import quad

from .basis import HLTBasis
from .checks import SanityReport, make_report, make_report_mpmath
from .data import CorrelatorData
from .precision import mp_matrix_to_numpy, normalize_backend, require_mpmath, to_mpf
from .solver import solve_hlt_constrained, solve_hlt_unconstrained, solve_hlt_endpoint_constrained, constraint_columns
from .targets import NonIntegrableTargetAreaError, QuadOptions, TargetKernel


@dataclass(frozen=True)
class ReconstructionResult:
    q: Any
    estimate: Any
    variance: Any
    error: Any
    checks: SanityReport
    W: Any
    b: Any
    c: Any
    R: Any
    regulator: str
    area_constraint: bool
    basis: HLTBasis
    target: TargetKernel
    backend: str = "numpy"
    mp_dps: int | None = None
    target_integration_backend: str = "numpy"
    quadrature_errors: dict[str, Any] = field(default_factory=dict)
    endpoint_value: Any | None = None
    omega_constraints: tuple = ()

    def kernel_values(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the reconstructed HLT kernel on an omega grid."""
        if self.backend == "mpmath":
            from .mpmath_backend import reconstructed_kernel

            mp = require_mpmath()
            with mp.workdps(self.mp_dps or 80):
                return reconstructed_kernel(self.basis, self.q, omega, mp)
        return self.basis.reconstructed_kernel(self.q, np.asarray(omega, dtype=float))

    def target_values(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate the target kernel on an omega grid."""
        if self.backend == "mpmath":
            if self.target_integration_backend == "numpy":
                mp = require_mpmath()
                with mp.workdps(self.mp_dps or 80):
                    values = np.asarray(
                        self.target.eval(np.asarray(omega, dtype=float)), dtype=float
                    )
                    if values.ndim == 0:
                        return to_mpf(values.item(), mp)
                    return mp.matrix([to_mpf(value, mp) for value in values])
            from .mpmath_backend import MPWorkspace, target_values

            mp = require_mpmath()
            with mp.workdps(self.mp_dps or 80):
                return target_values(
                    self.target, omega, mp, MPWorkspace(self.mp_dps or 80)
                )
        return np.asarray(self.target.eval(np.asarray(omega, dtype=float)), dtype=float)

    def kernel_difference(self, omega: np.ndarray) -> np.ndarray:
        """Evaluate reconstructed kernel minus target kernel on an omega grid."""
        if self.backend == "mpmath":
            return self.kernel_values(omega) - self.target_values(omega)
        omega = np.asarray(omega, dtype=float)
        return self.kernel_values(omega) - self.target_values(omega)

    def to_numpy(self) -> ReconstructionResult:
        """Explicitly down-convert an MP result for plotting/interoperability."""
        if self.backend == "numpy":
            return self
        return ReconstructionResult(
            q=mp_matrix_to_numpy(self.q),
            estimate=float(self.estimate),
            variance=float(self.variance),
            error=float(self.error),
            checks=self.checks.to_float(),
            W=mp_matrix_to_numpy(self.W),
            b=mp_matrix_to_numpy(self.b),
            c=float(self.c),
            R=mp_matrix_to_numpy(self.R),
            regulator=self.regulator,
            area_constraint=self.area_constraint,
            basis=self.basis,
            target=self.target,
            backend="numpy",
            mp_dps=None,
            target_integration_backend=self.target_integration_backend,
            quadrature_errors={
                key: float(value) for key, value in self.quadrature_errors.items()
            },
            endpoint_value=None if self.endpoint_value is None else float(self.endpoint_value),
            omega_constraints=tuple((float(w), float(d)) for w, d in self.omega_constraints),
        )

    def kernel_mismatch_on_grid(
        self,
        omega: np.ndarray,
        rho_p: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        *,
        relative_floor: float = 1e-14,
        strict_omega0: bool = True,
    ):
        """Return grid-level target, reconstructed kernel, and mismatch arrays.

        The returned object contains ``omega``, ``target``, ``reconstructed``,
        ``mismatch = reconstructed - target``, and ``relative_mismatch``.  If
        ``rho_p`` is supplied, it also contains ``weighted_mismatch`` equal to
        ``mismatch * rho_p``.

        By default, the omega grid must lie inside the HLT minimization domain,
        i.e. ``omega >= self.basis.omega0``.
        """
        from .systematics import kernel_mismatch_on_grid

        return kernel_mismatch_on_grid(
            basis=self.basis,
            target=self.target,
            q=self.q,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            relative_floor=relative_floor,
            strict_omega0=strict_omega0,
            backend=self.backend,
            mp_dps=self.mp_dps or 80,
            target_backend=self.target_integration_backend,
        )

    def estimate_kernel_systematic_on_grid(
        self,
        omega: np.ndarray,
        rho_p: np.ndarray,
        weights: np.ndarray | None = None,
        *,
        include_stat_error: bool = True,
    ):
        """Estimate observable-level kernel systematic on a quadrature grid.

        ``rho_p`` must be the spectral density in the same p-representation as
        the HLT target.  For p=2 in the convention used here, this is typically
        ``rho_p(omega) = R(omega**2)/(12*pi**2)``.  If ``weights`` is omitted,
        trapezoid-rule weights are constructed from ``omega`` automatically.

        This method uses the same pointwise mismatch returned by
        ``kernel_mismatch_on_grid``.
        """
        from .systematics import estimate_kernel_systematic_on_grid

        stat_error = self.error if include_stat_error else None
        return estimate_kernel_systematic_on_grid(
            basis=self.basis,
            target=self.target,
            q=self.q,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            stat_error=stat_error,
            backend=self.backend,
            mp_dps=self.mp_dps or 80,
            target_backend=self.target_integration_backend,
        )

    def estimate_target_smearing_systematic_on_grid(
        self,
        omega: np.ndarray,
        rho_p: np.ndarray,
        sharp_target: TargetKernel,
        weights: np.ndarray | None = None,
    ):
        """Estimate the Delta-smearing systematic against a sharp target.

        This compares the smeared target kernel used in this reconstruction,
        ``self.target``, with an externally supplied sharp target kernel,
        ``sharp_target``.  The signed bias is

            int d omega [T_delta(omega) - T_sharp(omega)] rho_p(omega),

        and the absolute bias is

            int d omega |T_delta(omega) - T_sharp(omega)| |rho_p(omega)|.

        ``rho_p`` must be normalized in the same p-representation as the HLT
        target.  If ``weights`` is omitted, trapezoid-rule weights are built
        automatically from ``omega``.
        """
        from .systematics import estimate_target_difference_on_grid

        return estimate_target_difference_on_grid(
            target_a=self.target,
            target_b=sharp_target,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            stat_error=None,
            backend=self.backend,
            mp_dps=self.mp_dps or 80,
            target_backend=self.target_integration_backend,
        )

    def estimate_total_sharp_systematic_on_grid(
        self,
        omega: np.ndarray,
        rho_p: np.ndarray,
        sharp_target: TargetKernel,
        weights: np.ndarray | None = None,
        *,
        include_stat_error: bool = True,
    ):
        """Estimate total sharp-target systematic on a quadrature grid.

        This compares the reconstructed HLT kernel, ``T_hat``, with an
        externally supplied sharp target kernel, ``sharp_target``.  The signed
        bias is

            int d omega [T_hat(omega) - T_sharp(omega)] rho_p(omega),

        and the absolute bias is

            int d omega |T_hat(omega) - T_sharp(omega)| |rho_p(omega)|.

        This quantity includes both the HLT reconstruction mismatch and the
        Delta-smearing shift relative to the sharp target.  ``rho_p`` must be
        normalized in the same p-representation as the HLT target.  If
        ``weights`` is omitted, trapezoid-rule weights are built automatically
        from ``omega``.
        """
        from .systematics import estimate_reconstructed_target_difference_on_grid

        stat_error = self.error if include_stat_error else None
        return estimate_reconstructed_target_difference_on_grid(
            basis=self.basis,
            q=self.q,
            reference_target=sharp_target,
            omega=omega,
            rho_p=rho_p,
            weights=weights,
            stat_error=stat_error,
            backend=self.backend,
            mp_dps=self.mp_dps or 80,
            target_backend=self.target_integration_backend,
        )


def _checked_selected_data_mpmath(data, basis, mp, mp_dps):
    """Convert selected correlator data once and validate covariance symmetry."""
    from .mpmath_backend import selected_data_matrices

    if (
        basis.periodic
        and data.T is not None
        and abs(basis.T - data.T) > 1e-12
    ):
        raise ValueError("basis.T and data.T disagree")

    mean_basis, cov_basis = selected_data_matrices(data, basis.times, mp)
    symmetry_scale = max(
        [
            abs(cov_basis[i, j])
            for i in range(cov_basis.rows)
            for j in range(cov_basis.cols)
        ]
        + [mp.one]
    )
    symmetry_tol = mp.power(10, -(mp_dps - 10)) * symmetry_scale
    if any(
        abs(cov_basis[i, j] - cov_basis[j, i]) > symmetry_tol
        for i in range(cov_basis.rows)
        for j in range(i)
    ):
        raise ValueError("selected covariance matrix is not symmetric")
    return mean_basis, cov_basis


def _condition_number_mpmath(W, R, area_constraint, mp, endpoint_vector=None, omega_vectors=()):
    columns = [R] if area_constraint else []
    if endpoint_vector is not None and any(endpoint_vector):
        columns.append(endpoint_vector)
    columns.extend(omega_vectors)
    kept, _, _, _ = constraint_columns(columns, [mp.zero] * len(columns), mp=mp)
    columns = [columns[i] for i in kept]
    if not columns:
        return mp.cond(W)
    size = W.rows
    kkt = mp.matrix(size + len(columns), size + len(columns))
    for i in range(size):
        for j in range(size):
            kkt[i, j] = W[i, j]
        for j, col in enumerate(columns):
            kkt[i, size + j] = col[i]
            kkt[size + j, i] = col[i]
    return mp.cond(kkt)


@dataclass(frozen=True)
class PreparedHLTSolution:
    """One lambda solution of a reusable arbitrary-precision HLT problem."""

    problem: Any
    lambda_hlt: Any
    q: Any
    W: Any
    f: Any
    eta: Any
    denominator: Any
    condition_number: Any
    mismatch_abs: Any | None
    mismatch_rel: Any | None
    regulator_data: CorrelatorData | None = None
    endpoint_eta: Any = 0
    omega_eta: tuple = ()

    def evaluate(
        self,
        data: CorrelatorData | None = None,
        *,
        strict: bool | None = None,
    ) -> ReconstructionResult:
        """Contract this shared solution with one correlator data set."""
        problem = self.problem
        selected_data = problem.default_data if data is None else data
        if selected_data is None:
            raise ValueError("data is required to evaluate a prepared HLT solution")
        if (
            problem.regulator == "covariance"
            and self.regulator_data is not None
            and selected_data is not self.regulator_data
        ):
            raise ValueError(
                "a covariance-regulated prepared solution may only be evaluated "
                "with the data used to construct its regulator"
            )
        effective_strict = problem.default_strict if strict is None else bool(strict)

        mp = require_mpmath()
        with mp.workdps(problem.mp_dps):
            mean_basis, cov_basis = _checked_selected_data_mpmath(
                selected_data, problem.basis, mp, problem.mp_dps
            )
            estimate = mp.fdot(self.q, mean_basis)
            from .mpmath_backend import matrix_quadratic

            variance = matrix_quadratic(self.q, cov_basis, mp)
            roundoff_floor = mp.power(10, -(problem.mp_dps - 10)) * max(
                abs(estimate) ** 2, mp.one
            )
            if variance < 0 and abs(variance) <= roundoff_floor:
                variance = mp.zero
            error = mp.sqrt(variance) if variance >= 0 else mp.nan

            checks = make_report_mpmath(
                q=self.q,
                W=self.W,
                f=self.f,
                R=problem.R,
                c=problem.c,
                eta=self.eta,
                covariance=cov_basis,
                mp=mp,
                area_constraint=problem.area_constraint,
                mismatch_abs=self.mismatch_abs,
                mismatch_rel=self.mismatch_rel,
                condition_number=self.condition_number,
                endpoint_vector=problem.endpoint_vector,
                endpoint_value=problem.endpoint_value,
                endpoint_eta=self.endpoint_eta,
                omega_vectors=problem.omega_vectors,
                omega_values=tuple(d for _, d in problem.omega_constraints),
                omega_eta=self.omega_eta,
            )
            if effective_strict and checks.warnings:
                raise RuntimeError(
                    "sanity checks failed: " + "; ".join(checks.warnings)
                )

            return ReconstructionResult(
                q=self.q,
                estimate=estimate,
                variance=variance,
                error=error,
                checks=checks,
                W=self.W,
                b=problem.b,
                c=problem.c,
                R=problem.R,
                regulator=problem.regulator,
                area_constraint=problem.area_constraint,
                basis=problem.basis,
                target=problem.target,
                backend="mpmath",
                mp_dps=problem.mp_dps,
                target_integration_backend=problem.target_integration_backend,
                quadrature_errors=dict(problem.quadrature_errors),
                endpoint_value=problem.endpoint_value,
                omega_constraints=problem.omega_constraints,
            )


@dataclass(frozen=True)
class PreparedHLTProblem:
    """Lambda-independent arbitrary-precision HLT construction.

    ``A``, ``R``, ``b``, the target area and (optionally) target norm are
    evaluated once.  The object can then solve many regulator values and each
    solution can be contracted with multiple correlator discretizations.
    """

    basis: HLTBasis
    target: TargetKernel
    A: Any
    R: Any
    b: Any
    c: Any
    area_constraint: bool
    target_norm_squared: Any | None
    regulator: str
    mp_dps: int
    target_integration_backend: str
    quad_options: QuadOptions
    quadrature_errors: dict[str, Any]
    shared_basis_cache: dict[Any, Any] = field(
        default_factory=dict, repr=False, compare=False
    )
    default_data: CorrelatorData | None = None
    default_covariance_norm_source: Any = None
    default_strict: bool = True
    endpoint_vector: Any | None = None
    endpoint_value: Any | None = None
    omega_constraints: tuple = ()
    omega_vectors: tuple = ()

    def _regulator_matrix(self, data, covariance_norm, mp):
        from .mpmath_backend import (
            frobenius_norm,
            identity,
            matrix_trace,
        )

        if self.regulator == "identity":
            return identity(self.A.rows, mp)
        if self.regulator == "identity_scaled":
            scale = matrix_trace(self.A, mp) / self.A.rows
            if not mp.isfinite(scale) or scale <= 0:
                scale = frobenius_norm(self.A, mp) / self.A.rows
            if not mp.isfinite(scale) or scale <= 0:
                raise ValueError(
                    "could not build a positive scaled-identity regulator from A"
                )
            return scale * identity(self.A.rows, mp)

        selected_data = self.default_data if data is None else data
        if selected_data is None:
            raise ValueError("data is required for the covariance regulator")
        _, cov_basis = _checked_selected_data_mpmath(
            selected_data, self.basis, mp, self.mp_dps
        )
        norm_source = (
            self.default_covariance_norm_source
            if covariance_norm is None
            else covariance_norm
        )
        if norm_source is not None:
            norm = to_mpf(norm_source, mp)
        else:
            matches = np.where(
                np.isclose(selected_data.times, 0.0, rtol=0.0, atol=1e-12)
            )[0]
            if len(matches) != 1:
                raise ValueError(
                    "could not identify C(0); provide covariance_norm explicitly"
                )
            c0 = to_mpf(selected_data._mean_source[int(matches[0])], mp)
            if c0 == 0:
                raise ValueError("C(0) is zero; provide covariance_norm explicitly")
            norm = c0 * c0
        return cov_basis / norm

    def solve(
        self,
        lambda_hlt: Any,
        *,
        data: CorrelatorData | None = None,
        covariance_norm: Any = None,
        compute_condition_number: bool = True,
        _identity_eigenvalue_bounds: tuple[Any, Any] | None = None,
    ) -> PreparedHLTSolution:
        """Solve one lambda without repeating target or basis integration."""
        mp = require_mpmath()
        with mp.workdps(self.mp_dps):
            lam = to_mpf(lambda_hlt, mp)
            if not (0 <= lam < 1):
                raise ValueError("lambda_hlt must satisfy 0 <= lambda_hlt < 1")
            regulator_matrix = self._regulator_matrix(data, covariance_norm, mp)
            W = (1 - lam) * self.A + lam * regulator_matrix
            f = (1 - lam) * self.b
            if self.endpoint_vector is not None or self.omega_vectors:
                solved = solve_hlt_endpoint_constrained(
                    W, f, self.endpoint_vector, self.endpoint_value,
                    omega_vectors=self.omega_vectors,
                    omega_values=tuple(d for _, d in self.omega_constraints),
                    R=self.R if self.area_constraint else None,
                    c=self.c if self.area_constraint else None,
                    backend="mpmath", mp_dps=self.mp_dps,
                )
            elif self.area_constraint:
                solved = solve_hlt_constrained(
                    W, f, self.R, self.c, backend="mpmath", mp_dps=self.mp_dps
                )
            else:
                solved = solve_hlt_unconstrained(
                    W, f, backend="mpmath", mp_dps=self.mp_dps
                )

            mismatch_abs = mismatch_rel = None
            if self.target_norm_squared is not None:
                qAq = mp.fdot(solved.q, self.A * solved.q)
                qb = mp.fdot(solved.q, self.b)
                mismatch_squared = qAq - 2 * qb + self.target_norm_squared
                cancellation_scale = max(
                    abs(qAq), 2 * abs(qb), abs(self.target_norm_squared), mp.one
                )
                negative_floor = (
                    mp.power(10, -(self.mp_dps - 8)) * cancellation_scale
                )
                if mismatch_squared < 0:
                    if abs(mismatch_squared) <= negative_floor:
                        mismatch_squared = mp.zero
                    else:
                        raise FloatingPointError(
                            "quadratic-form kernel mismatch is negative; increase mp_dps"
                        )
                mismatch_abs = mp.sqrt(mismatch_squared)
                norm_floor = mp.power(10, -(self.mp_dps - 5))
                mismatch_rel = mismatch_abs / max(
                    mp.sqrt(max(self.target_norm_squared, mp.zero)), norm_floor
                )

            if not compute_condition_number:
                condition_number = mp.nan
            elif (
                _identity_eigenvalue_bounds is not None
                and not self.area_constraint
                and self.endpoint_vector is None
                and not self.omega_vectors
                and self.regulator in {"identity", "identity_scaled"}
            ):
                eig_min, eig_max = _identity_eigenvalue_bounds
                regulator_scale = regulator_matrix[0, 0]
                shifted_min = (1 - lam) * eig_min + lam * regulator_scale
                shifted_max = (1 - lam) * eig_max + lam * regulator_scale
                condition_number = abs(shifted_max / shifted_min)
            else:
                condition_number = _condition_number_mpmath(
                    W, self.R, self.area_constraint, mp, self.endpoint_vector, self.omega_vectors
                )
            return PreparedHLTSolution(
                problem=self,
                lambda_hlt=lam,
                q=solved.q,
                W=W,
                f=f,
                eta=solved.eta,
                denominator=solved.denominator,
                condition_number=condition_number,
                mismatch_abs=mismatch_abs,
                mismatch_rel=mismatch_rel,
                endpoint_eta=solved.endpoint_eta,
                omega_eta=solved.omega_eta,
                regulator_data=(
                    self.default_data if data is None else data
                )
                if self.regulator == "covariance"
                else None,
            )

    def solve_many(
        self,
        lambda_values,
        *,
        data: CorrelatorData | None = None,
        covariance_norm: Any = None,
        compute_condition_number: bool = True,
    ) -> list[PreparedHLTSolution]:
        eigenvalue_bounds = None
        if (
            compute_condition_number
            and not self.area_constraint
            and self.endpoint_vector is None
            and not self.omega_vectors
            and self.regulator in {"identity", "identity_scaled"}
        ):
            mp = require_mpmath()
            with mp.workdps(self.mp_dps):
                cache_key = ("identity_eigenvalue_bounds", self.mp_dps)
                eigenvalue_bounds = self.shared_basis_cache.get(cache_key)
                if eigenvalue_bounds is None:
                    eigenvalues = mp.eigsy(self.A, eigvals_only=True)
                    eig_min = eigenvalues[0]
                    eig_max = eigenvalues[eigenvalues.rows - 1]
                    negative_tolerance = (
                        mp.power(10, -(self.mp_dps - 5))
                        * max(abs(eig_max), mp.one)
                    )
                    if eig_min < 0 and abs(eig_min) <= negative_tolerance:
                        eig_min = mp.zero
                    if eig_min >= 0:
                        eigenvalue_bounds = (eig_min, eig_max)
                        self.shared_basis_cache[cache_key] = eigenvalue_bounds
        return [
            self.solve(
                value,
                data=data,
                covariance_norm=covariance_norm,
                compute_condition_number=compute_condition_number,
                _identity_eigenvalue_bounds=eigenvalue_bounds,
            )
            for value in lambda_values
        ]

    def reconstruct_many(
        self,
        lambda_values,
        *,
        data: CorrelatorData | None = None,
        strict: bool | None = None,
        covariance_norm: Any = None,
        compute_condition_number: bool = True,
    ) -> list[ReconstructionResult]:
        return [
            solution.evaluate(data=data, strict=strict)
            for solution in self.solve_many(
                lambda_values,
                data=data,
                covariance_norm=covariance_norm,
                compute_condition_number=compute_condition_number,
            )
        ]


@dataclass(frozen=True)
class HLTReconstructor:
    """HLT fit with optional area, endpoint and point constraints.

    ``omega_constraints=[(omega, value), ...]`` fixes kernel values at energies
    in lattice units. None or an empty sequence adds no point constraints.

    ``endpoint_value=None`` preserves the ordinary HLT solve. A finite scalar
    imposes T_hat(basis.omega0)=endpoint_value; ``"target"`` instead matches
    the target at that point. Both constraints are enforced in the solve,
    up to working precision, without changing the target or correlator data.
    """
    data: CorrelatorData
    basis: HLTBasis
    lambda_hlt: float
    covariance_norm: float | None = None
    quad_options: QuadOptions = QuadOptions()
    strict: bool = True
    solver_backend: str = "numpy"
    mp_dps: int = 80
    regulator: str = "covariance"
    area_constraint: bool | str = "auto"
    numeric_backend: str | None = None
    target_integration_backend: str = "mpmath"
    endpoint_value: Any | None = None
    omega_constraints: tuple = ()
    _lambda_source: Any = field(init=False, repr=False, compare=False)
    _covariance_norm_source: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        lambda_source = self.lambda_hlt
        covariance_norm_source = self.covariance_norm
        lambda_hlt = float(self.lambda_hlt)
        covariance_norm = (
            None if self.covariance_norm is None else float(self.covariance_norm)
        )
        backend = normalize_backend(self.numeric_backend or self.solver_backend)
        target_integration_backend = normalize_backend(self.target_integration_backend)
        if not (0.0 <= lambda_hlt < 1.0):
            raise ValueError(
                "lambda_hlt must satisfy 0 <= lambda_hlt < 1 for the standard HLT form"
            )
        if covariance_norm is not None and covariance_norm <= 0:
            raise ValueError("covariance_norm must be positive")
        if (
            self.basis.periodic
            and self.data.T is not None
            and abs(self.basis.T - self.data.T) > 1e-12
        ):
            raise ValueError("basis.T and data.T disagree")
        if self.mp_dps < 30:
            raise ValueError("mp_dps should be at least 30")
        if self.regulator not in {"covariance", "identity", "identity_scaled"}:
            raise ValueError(
                "regulator must be 'covariance', 'identity', or 'identity_scaled'"
            )
        if self.area_constraint not in {True, False, "auto"}:
            raise ValueError("area_constraint must be True, False, or 'auto'")
        if self.endpoint_value is not None and self.endpoint_value != "target":
            try:
                if backend == "mpmath":
                    mp = require_mpmath()
                    with mp.workdps(self.mp_dps):
                        finite = mp.isfinite(to_mpf(self.endpoint_value, mp))
                else:
                    finite = np.isfinite(float(self.endpoint_value))
            except (TypeError, ValueError):
                finite = False
            if not finite:
                raise ValueError("endpoint_value must be None, 'target', or a finite scalar")
        from contextlib import nullcontext
        mp = require_mpmath() if backend == "mpmath" else None
        with mp.workdps(self.mp_dps) if mp else nullcontext():
            scalar = (lambda x: to_mpf(x, mp)) if mp else float
            finite = mp.isfinite if mp else np.isfinite
            constraints = []
            try:
                for pair in (() if self.omega_constraints is None else self.omega_constraints):
                    if isinstance(pair, (str, bytes)) or len(pair) != 2:
                        raise ValueError
                    w, d = (scalar(x) for x in pair)
                    if not finite(w) or not finite(d) or w < 0:
                        raise ValueError
                    if w == 0 and self.basis.p < 0:
                        raise ValueError
                    old = next((value for energy, value in constraints if energy == w), None)
                    if old is not None:
                        if old != d:
                            raise ValueError("conflicting duplicate omega_constraints")
                        continue
                    constraints.append((w, d))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("omega_constraints must contain finite [omega, value] pairs with omega >= 0 and a finite basis; conflicting duplicates are invalid") from exc
            object.__setattr__(self, "omega_constraints", tuple(constraints))
        object.__setattr__(self, "lambda_hlt", lambda_hlt)
        object.__setattr__(self, "covariance_norm", covariance_norm)
        object.__setattr__(self, "solver_backend", backend)
        object.__setattr__(self, "numeric_backend", backend)
        object.__setattr__(
            self, "target_integration_backend", target_integration_backend
        )
        object.__setattr__(self, "_lambda_source", lambda_source)
        object.__setattr__(self, "_covariance_norm_source", covariance_norm_source)

    def selected_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.data.sliced(self.basis.times)

    def _endpoint_spec(self, target, mp=None, workspace=None):
        if self.endpoint_value is None:
            return None, None
        if self.basis.omega0 == 0 and self.basis.p < 0:
            raise ValueError("endpoint constraint requires a finite basis at omega0")
        if mp is None:
            v = np.asarray([self.basis.phi(t, self.basis.omega0) for t in self.basis.times])
            value = float(target.eval(self.basis.omega0) if self.endpoint_value == "target"
                          else self.endpoint_value)
            if not np.all(np.isfinite(v)) or not np.isfinite(value):
                raise ValueError("endpoint basis and target value must be finite")
            if self.basis.omega0 > 0 and not np.any(v):
                raise FloatingPointError("endpoint basis underflowed; use the mpmath backend")
        else:
            from .mpmath_backend import basis_phi, target_eval

            omega0 = to_mpf(self.basis._omega0_source, mp)
            v = mp.matrix([basis_phi(self.basis, t, omega0, mp) for t in self.basis._times_source])
            if self.endpoint_value == "target":
                value = (to_mpf(target.eval(self.basis.omega0), mp)
                         if self.target_integration_backend == "numpy"
                         else target_eval(target, omega0, mp, workspace))
            else:
                value = to_mpf(self.endpoint_value, mp)
            if not all(mp.isfinite(x) for x in v) or not mp.isfinite(value):
                raise ValueError("endpoint basis and target value must be finite")
        return v, value

    def _omega_vectors(self, endpoint_value=None, mp=None):
        columns = []
        for omega, value in self.omega_constraints:
            if endpoint_value is not None:
                omega0 = to_mpf(self.basis._omega0_source, mp) if mp else self.basis.omega0
                if omega == omega0 and value != endpoint_value:
                    raise ValueError("omega_constraints conflicts with endpoint_value at omega0")
            if mp is None:
                col = np.asarray([self.basis.phi(t, omega) for t in self.basis.times])
                if not np.all(np.isfinite(col)):
                    raise ValueError("constraint basis must be finite")
                if omega > 0 and not np.any(col):
                    raise FloatingPointError("constraint basis underflowed; use the mpmath backend")
            else:
                from .mpmath_backend import basis_phi
                col = mp.matrix([basis_phi(self.basis, t, omega, mp) for t in self.basis._times_source])
                if not all(mp.isfinite(x) for x in col):
                    raise ValueError("constraint basis must be finite")
            columns.append(col)
        return tuple(columns)

    def covariance_normalization(self) -> float:
        if self.covariance_norm is not None:
            return float(self.covariance_norm)
        matches = np.where(np.isclose(self.data.times, 0.0, rtol=0.0, atol=1e-12))[0]
        if len(matches) == 1:
            c0 = float(self.data.mean[matches[0]])
            if c0 == 0.0:
                raise ValueError("C(0) is zero; provide covariance_norm explicitly")
            return c0 * c0
        raise ValueError("could not identify C(0); provide covariance_norm explicitly")

    def prepare(
        self,
        target: TargetKernel,
        *,
        compute_mismatch: bool = True,
        reuse_basis_from: PreparedHLTProblem | None = None,
    ) -> PreparedHLTProblem:
        """Prepare all lambda-independent quantities for an MP scan.

        The returned problem is intended for regulator scans.  It evaluates
        the expensive target vector and target norm only once, then exposes
        ``solve_many``/``reconstruct_many``.  An identity-regulated solution
        may also be evaluated against multiple correlator data sets sharing
        the same basis.
        """
        if self.numeric_backend != "mpmath":
            raise ValueError("prepared scan problems currently require mpmath")
        from .mpmath_backend import (
            MPWorkspace,
            basis_A,
            basis_R,
            target_area,
            target_b_vector,
            target_norm_squared,
        )

        mp = require_mpmath()
        with mp.workdps(self.mp_dps):
            workspace = MPWorkspace(self.mp_dps)
            if reuse_basis_from is not None:
                if reuse_basis_from.basis is not self.basis:
                    raise ValueError(
                        "reuse_basis_from must use the identical HLTBasis object"
                    )
                if reuse_basis_from.mp_dps != self.mp_dps:
                    raise ValueError("reuse_basis_from has a different mp_dps")
                A = reuse_basis_from.A
                R = reuse_basis_from.R
            else:
                A = basis_A(self.basis, mp)
                R = basis_R(self.basis, mp)
            if self.target_integration_backend == "numpy":
                b_float = target.b_vector(self.basis, self.quad_options)
                if not np.all(np.isfinite(b_float)):
                    raise ValueError("target b-vector contains non-finite values")
                b = mp.matrix([to_mpf(value, mp) for value in b_float])
            else:
                b = target_b_vector(
                    target, self.basis, self.quad_options, mp, workspace
                )

            use_area_constraint = self.area_constraint is not False
            if use_area_constraint:
                try:
                    if self.target_integration_backend == "numpy":
                        c_float = target.area(
                            self.quad_options, omega0=self.basis.omega0
                        )
                        if not np.isfinite(c_float):
                            raise ValueError("target area is non-finite")
                        c = to_mpf(c_float, mp)
                    else:
                        c = target_area(
                            target,
                            to_mpf(self.basis._omega0_source, mp),
                            self.quad_options,
                            mp,
                            workspace,
                        )
                except NonIntegrableTargetAreaError:
                    if self.area_constraint == "auto":
                        use_area_constraint = False
                        c = mp.nan
                    else:
                        raise
            else:
                c = mp.nan

            norm_squared = None
            if compute_mismatch:
                if self.target_integration_backend == "numpy":
                    opts = self.quad_options
                    value, _ = quad(
                        lambda omega: float(target.eval(omega)) ** 2,
                        self.basis.omega0,
                        np.inf,
                        epsabs=opts.epsabs,
                        epsrel=opts.epsrel,
                        limit=opts.limit,
                    )
                    norm_squared = to_mpf(value, mp)
                else:
                    norm_squared = target_norm_squared(
                        target,
                        to_mpf(self.basis._omega0_source, mp),
                        self.quad_options,
                        mp,
                        workspace,
                    )

            endpoint_vector, endpoint_value = self._endpoint_spec(target, mp, workspace)
            omega_vectors = self._omega_vectors(endpoint_value, mp)
            return PreparedHLTProblem(
                basis=self.basis,
                target=target,
                A=A,
                R=R,
                b=b,
                c=c,
                area_constraint=use_area_constraint,
                target_norm_squared=norm_squared,
                regulator=self.regulator,
                mp_dps=self.mp_dps,
                target_integration_backend=self.target_integration_backend,
                quad_options=self.quad_options,
                quadrature_errors=dict(workspace.quadrature_errors),
                shared_basis_cache=(
                    reuse_basis_from.shared_basis_cache
                    if reuse_basis_from is not None
                    else {}
                ),
                default_data=self.data,
                default_covariance_norm_source=self._covariance_norm_source,
                default_strict=self.strict,
                endpoint_vector=endpoint_vector,
                endpoint_value=endpoint_value,
                omega_vectors=omega_vectors,
                omega_constraints=self.omega_constraints,
            )

    def W_and_f(
        self, A: np.ndarray, b: np.ndarray, covariance_basis: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Construct the regularized HLT matrix and right-hand side.

        The default is the standard covariance-regulated HLT form

            W = (1-lambda) A + lambda Cov / C(0)^2,
            f = (1-lambda) b.

        With ``regulator='identity'`` one instead uses

            W = (1-lambda) A + lambda I,
            f = (1-lambda) b.

        The identity option is a stabilizing regulator rather than the usual
        statistical HLT error functional; therefore lambda values are not
        directly comparable between the two regulator choices.
        """
        lam = self.lambda_hlt
        if self.regulator == "covariance":
            norm = self.covariance_normalization()
            regulator_matrix = covariance_basis / norm
        elif self.regulator == "identity":
            regulator_matrix = np.eye(A.shape[0], dtype=float)
        elif self.regulator == "identity_scaled":
            scale = float(np.trace(A) / A.shape[0])
            if not np.isfinite(scale) or scale <= 0.0:
                scale = float(np.linalg.norm(A, ord="fro") / A.shape[0])
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(
                    "could not build a positive scaled-identity regulator from A"
                )
            regulator_matrix = scale * np.eye(A.shape[0], dtype=float)
        else:  # guarded in __post_init__, kept for defensive programming
            raise ValueError(
                "regulator must be 'covariance', 'identity', or 'identity_scaled'"
            )

        W = (1.0 - lam) * A + lam * regulator_matrix
        f = (1.0 - lam) * b
        return W, f

    def _target_area_for_solve(self, target: TargetKernel) -> tuple[bool, float]:
        """Return (use_area_constraint, c).

        ``area_constraint='auto'`` preserves the usual constrained HLT solve
        whenever the target has a finite area.  If a target explicitly reports
        that its unweighted area is non-integrable, the solve is switched to the
        unconstrained normal equation, because the constraint R^T q = c is then
        mathematically undefined.
        """
        if self.area_constraint is False:
            return False, float("nan")

        try:
            c = target.area(self.quad_options, omega0=self.basis.omega0)
        except NonIntegrableTargetAreaError:
            if self.area_constraint == "auto":
                return False, float("nan")
            raise

        if not np.isfinite(c):
            if self.area_constraint == "auto":
                return False, float("nan")
            raise ValueError("target area is non-finite")

        return True, float(c)

    def kernel_mismatch(
        self, target: TargetKernel, q: np.ndarray
    ) -> tuple[float, float]:
        def diff_sq(x: float) -> float:
            omega = np.asarray([x], dtype=float)
            rec = self.basis.reconstructed_kernel(q, omega)[0]
            tar = float(target.eval(x))
            return float((rec - tar) ** 2)

        def target_sq(x: float) -> float:
            tar = float(target.eval(x))
            return float(tar * tar)

        opts = self.quad_options
        mismatch, _ = quad(
            diff_sq,
            self.basis.omega0,
            np.inf,
            epsabs=opts.epsabs,
            epsrel=opts.epsrel,
            limit=opts.limit,
        )
        norm, _ = quad(
            target_sq,
            self.basis.omega0,
            np.inf,
            epsabs=opts.epsabs,
            epsrel=opts.epsrel,
            limit=opts.limit,
        )
        mismatch_abs = float(np.sqrt(max(mismatch, 0.0)))
        mismatch_rel = float(
            mismatch_abs / max(np.sqrt(max(norm, 0.0)), np.finfo(float).eps)
        )
        return mismatch_abs, mismatch_rel

    def _kernel_mismatch_mixed_mpmath(self, target: TargetKernel, q, mp):
        """Use a float target quadrature without rounding MP coefficients.

        SciPy controls the outer diagnostic integral and the target/window is
        evaluated in binary64, while cancellation in the reconstructed kernel
        is still handled at the reconstruction's mpmath precision.
        """
        from .mpmath_backend import reconstructed_kernel

        def diff_sq(x: float) -> float:
            reconstructed = reconstructed_kernel(self.basis, q, x, mp)
            target_value = to_mpf(float(target.eval(x)), mp)
            return float((reconstructed - target_value) ** 2)

        def target_sq(x: float) -> float:
            target_value = float(target.eval(x))
            return target_value * target_value

        opts = self.quad_options
        mismatch, _ = quad(
            diff_sq,
            self.basis.omega0,
            np.inf,
            epsabs=opts.epsabs,
            epsrel=opts.epsrel,
            limit=opts.limit,
        )
        norm, _ = quad(
            target_sq,
            self.basis.omega0,
            np.inf,
            epsabs=opts.epsabs,
            epsrel=opts.epsrel,
            limit=opts.limit,
        )
        mismatch_abs = mp.sqrt(max(to_mpf(mismatch, mp), mp.zero))
        norm_sqrt = mp.sqrt(max(to_mpf(norm, mp), mp.zero))
        mismatch_rel = mismatch_abs / max(norm_sqrt, mp.eps)
        return mismatch_abs, mismatch_rel

    def reconstruct(
        self, target: TargetKernel, compute_mismatch: bool = True
    ) -> ReconstructionResult:
        if self.numeric_backend == "mpmath":
            return self._reconstruct_mpmath(target, compute_mismatch=compute_mismatch)

        _, mean_basis, cov_basis = self.selected_data()

        if not np.allclose(cov_basis, cov_basis.T, rtol=1e-10, atol=1e-12):
            raise ValueError("selected covariance matrix is not symmetric")

        A = self.basis.A()
        R = self.basis.R()
        b = target.b_vector(self.basis, self.quad_options)
        use_area_constraint, c = self._target_area_for_solve(target)
        W, f = self.W_and_f(A, b, cov_basis)

        endpoint_vector, endpoint_value = self._endpoint_spec(target)
        omega_vectors = self._omega_vectors(endpoint_value)
        if endpoint_vector is not None or omega_vectors:
            solution = solve_hlt_endpoint_constrained(
                W, f, endpoint_vector, endpoint_value,
                omega_vectors=omega_vectors,
                omega_values=tuple(d for _, d in self.omega_constraints),
                R=R if use_area_constraint else None,
                c=c if use_area_constraint else None,
                backend=self.solver_backend, mp_dps=self.mp_dps,
            )
        elif use_area_constraint:
            solution = solve_hlt_constrained(
                W,
                f,
                R,
                c,
                backend=self.solver_backend,
                mp_dps=self.mp_dps,
            )
        else:
            solution = solve_hlt_unconstrained(
                W,
                f,
                backend=self.solver_backend,
                mp_dps=self.mp_dps,
            )
        q = solution.q

        estimate = float(np.dot(q, mean_basis))
        variance = float(q.T @ cov_basis @ q)
        if variance < 0 and variance > -1e-14:
            variance = 0.0
        error = float(np.sqrt(variance)) if variance >= 0 else float("nan")

        mismatch_abs = None
        mismatch_rel = None
        if compute_mismatch:
            mismatch_abs, mismatch_rel = self.kernel_mismatch(target, q)

        checks = make_report(
            q=q,
            W=W,
            f=f,
            R=R,
            c=c,
            eta=solution.eta,
            covariance=cov_basis,
            area_constraint=use_area_constraint,
            mismatch_abs=mismatch_abs,
            mismatch_rel=mismatch_rel,
            endpoint_vector=endpoint_vector,
            endpoint_value=endpoint_value,
            endpoint_eta=solution.endpoint_eta,
            omega_vectors=omega_vectors,
            omega_values=tuple(d for _, d in self.omega_constraints),
            omega_eta=solution.omega_eta,
        )

        if self.strict and checks.warnings:
            raise RuntimeError("sanity checks failed: " + "; ".join(checks.warnings))

        return ReconstructionResult(
            q=q,
            estimate=estimate,
            variance=variance,
            error=error,
            checks=checks,
            W=W,
            b=b,
            c=c,
            R=R,
            regulator=self.regulator,
            area_constraint=use_area_constraint,
            basis=self.basis,
            target=target,
            backend="numpy",
            mp_dps=None,
            target_integration_backend="numpy",
            endpoint_value=endpoint_value,
            omega_constraints=self.omega_constraints,
        )

    def _reconstruct_mpmath(
        self, target: TargetKernel, *, compute_mismatch: bool
    ) -> ReconstructionResult:
        prepared = self.prepare(target, compute_mismatch=compute_mismatch)
        solution = prepared.solve(
            self._lambda_source,
            data=self.data,
            covariance_norm=self._covariance_norm_source,
        )
        return solution.evaluate(self.data, strict=self.strict)

        # Kept below as unreachable source compatibility context for the
        # original 0.5.1 implementation.  The prepared path above owns all MP
        # reconstruction work and avoids rebuilding lambda-independent data.
        from .mpmath_backend import (
            MPWorkspace,
            basis_A,
            basis_R,
            frobenius_norm,
            identity,
            kernel_mismatch_integrals,
            matrix_quadratic,
            matrix_trace,
            selected_data_matrices,
            target_area,
            target_b_vector,
        )

        mp = require_mpmath()
        with mp.workdps(self.mp_dps):
            workspace = MPWorkspace(self.mp_dps)
            mean_basis, cov_basis = selected_data_matrices(
                self.data, self.basis.times, mp
            )

            symmetry_scale = max(
                [
                    abs(cov_basis[i, j])
                    for i in range(cov_basis.rows)
                    for j in range(cov_basis.cols)
                ]
                + [mp.one]
            )
            symmetry_tol = mp.power(10, -(self.mp_dps - 10)) * symmetry_scale
            if any(
                abs(cov_basis[i, j] - cov_basis[j, i]) > symmetry_tol
                for i in range(cov_basis.rows)
                for j in range(i)
            ):
                raise ValueError("selected covariance matrix is not symmetric")

            A = basis_A(self.basis, mp)
            R = basis_R(self.basis, mp)
            if self.target_integration_backend == "numpy":
                # The target/window quadratures are smooth and inexpensive in
                # binary64 compared with their deeply nested arbitrary-
                # precision counterparts.  Promote their computed values only
                # after integration; A, R, W, the solve, and the observable
                # contractions remain native mpmath calculations.
                b_float = target.b_vector(self.basis, self.quad_options)
                if not np.all(np.isfinite(b_float)):
                    raise ValueError("target b-vector contains non-finite values")
                b = mp.matrix([to_mpf(value, mp) for value in b_float])
            else:
                b = target_b_vector(
                    target, self.basis, self.quad_options, mp, workspace
                )

            use_area_constraint = self.area_constraint is not False
            if use_area_constraint:
                try:
                    if self.target_integration_backend == "numpy":
                        c_float = target.area(
                            self.quad_options, omega0=self.basis.omega0
                        )
                        if not np.isfinite(c_float):
                            raise ValueError("target area is non-finite")
                        c = to_mpf(c_float, mp)
                    else:
                        c = target_area(
                            target,
                            to_mpf(self.basis._omega0_source, mp),
                            self.quad_options,
                            mp,
                            workspace,
                        )
                except NonIntegrableTargetAreaError:
                    if self.area_constraint == "auto":
                        use_area_constraint = False
                        c = mp.nan
                    else:
                        raise
            else:
                c = mp.nan

            lam = to_mpf(self._lambda_source, mp)
            if self.regulator == "covariance":
                if self._covariance_norm_source is not None:
                    norm = to_mpf(self._covariance_norm_source, mp)
                else:
                    matches = np.where(
                        np.isclose(self.data.times, 0.0, rtol=0.0, atol=1e-12)
                    )[0]
                    if len(matches) != 1:
                        raise ValueError(
                            "could not identify C(0); provide covariance_norm "
                            "explicitly"
                        )
                    c0 = to_mpf(self.data._mean_source[int(matches[0])], mp)
                    if c0 == 0:
                        raise ValueError(
                            "C(0) is zero; provide covariance_norm explicitly"
                        )
                    norm = c0 * c0
                regulator_matrix = cov_basis / norm
            elif self.regulator == "identity":
                regulator_matrix = identity(A.rows, mp)
            else:
                scale = matrix_trace(A, mp) / A.rows
                if not mp.isfinite(scale) or scale <= 0:
                    scale = frobenius_norm(A, mp) / A.rows
                if not mp.isfinite(scale) or scale <= 0:
                    raise ValueError(
                        "could not build a positive scaled-identity regulator from A"
                    )
                regulator_matrix = scale * identity(A.rows, mp)

            W = (1 - lam) * A + lam * regulator_matrix
            f = (1 - lam) * b
            if use_area_constraint:
                solution = solve_hlt_constrained(
                    W, f, R, c, backend="mpmath", mp_dps=self.mp_dps
                )
            else:
                solution = solve_hlt_unconstrained(
                    W, f, backend="mpmath", mp_dps=self.mp_dps
                )
            q = solution.q

            estimate = mp.fdot(q, mean_basis)
            variance = matrix_quadratic(q, cov_basis, mp)
            roundoff_floor = mp.power(10, -(self.mp_dps - 10)) * max(
                abs(estimate) ** 2, mp.one
            )
            if variance < 0 and abs(variance) <= roundoff_floor:
                variance = mp.zero
            error = mp.sqrt(variance) if variance >= 0 else mp.nan

            mismatch_abs = mismatch_rel = None
            if compute_mismatch:
                if self.target_integration_backend == "numpy":
                    mismatch_abs, mismatch_rel = (
                        self._kernel_mismatch_mixed_mpmath(target, q, mp)
                    )
                else:
                    mismatch_abs, mismatch_rel = kernel_mismatch_integrals(
                        target, self.basis, q, self.quad_options, mp, workspace
                    )

            checks = make_report_mpmath(
                q=q,
                W=W,
                f=f,
                R=R,
                c=c,
                eta=solution.eta,
                covariance=cov_basis,
                mp=mp,
                area_constraint=use_area_constraint,
                mismatch_abs=mismatch_abs,
                mismatch_rel=mismatch_rel,
            )
            if self.strict and checks.warnings:
                raise RuntimeError(
                    "sanity checks failed: " + "; ".join(checks.warnings)
                )

            return ReconstructionResult(
                q=q,
                estimate=estimate,
                variance=variance,
                error=error,
                checks=checks,
                W=W,
                b=b,
                c=c,
                R=R,
                regulator=self.regulator,
                area_constraint=use_area_constraint,
                basis=self.basis,
                target=target,
                backend="mpmath",
                mp_dps=self.mp_dps,
                target_integration_backend=self.target_integration_backend,
                quadrature_errors=dict(workspace.quadrature_errors),
            )
