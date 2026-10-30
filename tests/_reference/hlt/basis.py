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
from scipy.special import gamma, gammaincc

from .precision import require_mpmath, source_vector


@dataclass(frozen=True)
class HLTBasis:
    """HLT basis functions phi_t(omega) = omega^p B_T(t, omega)."""

    times: np.ndarray
    p: float = 0.0
    T: float | None = None
    periodic: bool = False
    omega0: float = 0.0
    _times_source: tuple[Any, ...] = field(init=False, repr=False, compare=False)
    _p_source: Any = field(init=False, repr=False, compare=False)
    _T_source: Any = field(init=False, repr=False, compare=False)
    _omega0_source: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        times_source = source_vector(self.times)
        p_source, T_source, omega0_source = self.p, self.T, self.omega0
        times = np.asarray(times_source, dtype=float)
        p = float(self.p)
        T = None if self.T is None else float(self.T)
        omega0 = float(self.omega0)
        if times.ndim != 1 or len(times) == 0:
            raise ValueError("basis times must be a non-empty one-dimensional array")
        if not np.all(np.isfinite(times)):
            raise ValueError("basis times contain non-finite values")
        if len(np.unique(times)) != len(times):
            raise ValueError("basis times must be unique")
        if p <= -1:
            raise ValueError(
                "p must be > -1 for the normalization integral to converge"
            )
        if omega0 < 0 or not np.isfinite(omega0):
            raise ValueError("omega0 must be finite and >= 0")
        if np.any(times <= 0):
            raise ValueError("basis times must be > 0")
        if self.periodic:
            if T is None:
                raise ValueError("T is required for periodic=True")
            if T <= 0:
                raise ValueError("T must be positive")
            if np.any(times >= T):
                raise ValueError("periodic basis times must satisfy 0 < t < T")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "omega0", omega0)
        object.__setattr__(self, "_times_source", times_source)
        object.__setattr__(self, "_p_source", p_source)
        object.__setattr__(self, "_T_source", T_source)
        object.__setattr__(self, "_omega0_source", omega0_source)

    def B(self, t: float, omega: np.ndarray | float) -> np.ndarray | float:
        if self.periodic:
            return np.exp(-omega * t) + np.exp(-omega * (self.T - t))
        return np.exp(-omega * t)

    def phi(self, t: float, omega: np.ndarray | float) -> np.ndarray | float:
        return np.power(omega, self.p) * self.B(t, omega)

    def _exp_integral(self, power: float, alpha: float) -> float:
        """Return int_omega0^inf omega^power exp(-alpha omega) d omega.

        This is Gamma(power+1, alpha*omega0) / alpha**(power+1),
        evaluated through scipy.special.gammaincc, the regularized upper
        incomplete gamma function.
        """
        if alpha <= 0:
            raise ValueError("exponential decay parameter alpha must be positive")
        a = power + 1.0
        z = alpha * self.omega0
        upper_gamma = gamma(a) * gammaincc(a, z)
        return float(upper_gamma / alpha**a)

    def A(self) -> np.ndarray:
        """Analytic Gram matrix int_omega0^inf phi_n(omega) phi_m(omega) d omega."""
        n = len(self.times)
        A = np.empty((n, n), dtype=float)
        power = 2.0 * self.p
        for i, ti in enumerate(self.times):
            for j, tj in enumerate(self.times):
                if self.periodic:
                    alphas = (
                        ti + tj,
                        ti + self.T - tj,
                        self.T - ti + tj,
                        2.0 * self.T - ti - tj,
                    )
                else:
                    alphas = (ti + tj,)
                A[i, j] = sum(self._exp_integral(power, a) for a in alphas)
        return A

    def R(self) -> np.ndarray:
        """Analytic area vector int_omega0^inf phi_n(omega) d omega."""
        out = np.empty(len(self.times), dtype=float)
        for i, ti in enumerate(self.times):
            if self.periodic:
                out[i] = self._exp_integral(self.p, ti) + self._exp_integral(
                    self.p, self.T - ti
                )
            else:
                out[i] = self._exp_integral(self.p, ti)
        return out

    def reconstructed_kernel(self, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        omega = np.asarray(omega, dtype=float)
        if q.shape != (len(self.times),):
            raise ValueError("q has incompatible shape")
        total = np.zeros_like(omega, dtype=float)
        for coeff, t in zip(q, self.times):
            total += coeff * self.phi(t, omega)
        return total

    def A_mpmath(self, mp_dps: int = 80):
        """Return the analytic Gram matrix without a binary64 intermediate."""
        from .mpmath_backend import basis_A

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return basis_A(self, mp)

    def R_mpmath(self, mp_dps: int = 80):
        """Return the analytic area vector without a binary64 intermediate."""
        from .mpmath_backend import basis_R

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return basis_R(self, mp)

    def reconstructed_kernel_mpmath(self, q: Any, omega: Any, mp_dps: int = 80):
        """Evaluate the reconstructed kernel entirely with mpmath arithmetic."""
        from .mpmath_backend import reconstructed_kernel

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return reconstructed_kernel(self, q, omega, mp)
