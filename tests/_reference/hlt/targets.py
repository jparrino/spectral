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
from functools import lru_cache
from typing import Any, Iterable, Sequence

import numpy as np
from scipy.integrate import quad
from scipy.special import expit, kv, modstruve

from .basis import HLTBasis
from .precision import (
    mp_trapezoid_weights,
    require_mpmath,
    source_vector,
)

HBARC_MEV_FM = 197.3269804
MUON_MASS_MEV = 105.6583745
ALPHA_EM = 1.0 / 137.035999084


def fm_to_lattice_units(
    length_fm: float, a_lat_inv_MeV: float, *, backend: str = "numpy", mp_dps: int = 80
):
    """Convert a physical length in fm to lattice units using a^{-1} in MeV."""
    if backend == "mpmath":
        from .precision import to_mpf

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            a_inv = to_mpf(a_lat_inv_MeV, mp)
            if a_inv <= 0 or not mp.isfinite(a_inv):
                raise ValueError("a_lat_inv_MeV must be positive and finite")
            return to_mpf(length_fm, mp) * a_inv / mp.mpf("197.3269804")
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'mpmath'")
    if a_lat_inv_MeV <= 0 or not np.isfinite(a_lat_inv_MeV):
        raise ValueError("a_lat_inv_MeV must be positive and finite")
    return float(length_fm * a_lat_inv_MeV / HBARC_MEV_FM)


def _standard_window_lattice(name: str, a_lat_inv_MeV: float) -> tuple[float, float]:
    key = name.upper()
    t0 = fm_to_lattice_units(0.4, a_lat_inv_MeV)
    t1 = fm_to_lattice_units(1.0, a_lat_inv_MeV)
    if key in {"ID", "INT", "INTERMEDIATE", "WIN", "WINDOW"}:
        return (t0, t1)
    if key in {"SD", "SHORT", "SHORT_DISTANCE"}:
        return (-np.inf, t0)
    if key in {"LD", "LONG", "LONG_DISTANCE"}:
        return (t1, np.inf)
    raise ValueError(
        "unknown euclidean_window string; use 'FULL', 'ID', 'SD', 'LD', or a "
        "two-element interval"
    )


def _default_window_delta_lattice(a_lat_inv_MeV: float) -> float:
    return fm_to_lattice_units(0.15, a_lat_inv_MeV)


@lru_cache(maxsize=16)
def _legendre_nodes_weights_0_1(order: int) -> tuple[np.ndarray, np.ndarray]:
    if order < 8:
        raise ValueError("quadrature order must be at least 8")
    x, w = np.polynomial.legendre.leggauss(order)
    return 0.5 * (x + 1.0), 0.5 * w


def _smooth_step(t: np.ndarray | float, t0: float, delta: float) -> np.ndarray | float:
    # expit(2*x) is algebraically (1+tanh(x))/2 and is stable in both tails.
    return expit(2.0 * (t - t0) / delta)


def _bernecker_meyer_inner_analytic_float(t: float, m_mu: float) -> float:
    """Binary64 evaluation of the exact analytic TMR inner kernel.

    This is ``m_mu**2*f_tilde(t)/pi**2`` from Eqs. (55) and (57) of
    arXiv:1705.01775, with the Meijer-G rewritten using Bessel K and modified
    Struve L functions.  The short-distance series from Eq. (58) avoids loss
    of significance in the closed form near the origin.
    """
    u = float(t * m_mu)
    if u <= 0.0 or not np.isfinite(u):
        return 0.0
    if u < 0.2:
        u2 = u * u
        log_u = np.log(u)
        return float(
            u2**2 / 9.0
            + u2**3 * (120.0 * log_u + 120.0 * np.euler_gamma - 169.0) / 5400.0
            + u2**4 * (210.0 * log_u + 210.0 * np.euler_gamma - 401.0) / 88200.0
            + u2**5 * (360.0 * log_u + 360.0 * np.euler_gamma - 787.0) / 2916000.0
            + u2**6 * (3080.0 * log_u + 3080.0 * np.euler_gamma - 7353.0)
            / 768398400.0
        )

    z = 2.0 * u
    k0 = kv(0, z)
    k1 = kv(1, z)
    closed_form_g = -4.0 * u * (
        k1
        + np.pi
        * u
        * (k0 * modstruve(-1, z) + k1 * modstruve(0, z))
    )
    return float(
        2.0
        * (
            -2.0
            + 8.0 * np.euler_gamma
            + 4.0 / u**2
            + u**2
            - 8.0 * k1 / u
            + 8.0 * np.log(u)
            + closed_form_g
        )
    )


def _hvp_kernel_K_analytic_float(omega: float, m_mu: float) -> float:
    """Return the standard dispersive HVP kernel without quadrature.

    This evaluates

        K(r) = int_0^1 dx x^2 (1-x) / (x^2 + r (1-x)),
        r = (omega/m_mu)^2,

    using closed forms on the two sides of the threshold ``r=4``.  The
    large-r expression uses the usual conformal variable and a short series
    for ``log1p(x)-x+x^2/2`` so that the cancellation at ``x -> 0`` does not
    discard binary64 accuracy.
    """
    omega = float(omega)
    m_mu = float(m_mu)
    if omega <= 0.0 or m_mu <= 0.0 or not np.isfinite(omega):
        return 0.0
    r = (omega / m_mu) ** 2
    if not np.isfinite(r):
        return 0.0

    if r > 4.0:
        beta = np.sqrt(1.0 - 4.0 / r)
        # This form avoids cancellation in 1-beta at large r.
        x = 4.0 / (r * (1.0 + beta) ** 2)
        if x < 0.075:
            remainder = 0.0
            power = x**3
            sign = 1.0
            for n in range(3, 24):
                remainder += sign * power / n
                power *= x
                sign = -sign
        else:
            remainder = np.log1p(x) - x + 0.5 * x * x
        return float(
            0.5 * x * x * (2.0 - x * x)
            + (1.0 + x) ** 2 * (1.0 + x**-2) * remainder
            + (1.0 + x) / (1.0 - x) * x * x * np.log(x)
        )

    if r == 4.0:
        return float(8.0 * np.log(2.0) - 5.5)

    h = np.sqrt(r * (4.0 - r))
    reciprocal_integral = (2.0 / h) * (
        np.arctan((2.0 - r) / h) + np.arctan(r / h)
    )
    log_coefficient = 0.5 * r * (2.0 - r)
    reciprocal_coefficient = -0.5 * r * (r * r - 4.0 * r + 2.0)
    return float(
        0.5
        - r
        - log_coefficient * np.log(r)
        + reciprocal_coefficient * reciprocal_integral
    )


def trapezoid_weights(
    times: Iterable[float], *, backend: str = "numpy", mp_dps: int = 80
):
    """Return trapezoid-rule weights for a one-dimensional time grid.

    These weights convert kernel values K(t_i) into quadrature weights
    w_i K(t_i) for approximating int dt K(t) G(t).  The grid may be
    non-uniform, but it must be strictly increasing.
    """
    values = source_vector(times)
    if backend == "mpmath":
        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return mp.matrix(mp_trapezoid_weights(values, mp))
    if backend != "numpy":
        raise ValueError("backend must be 'numpy' or 'mpmath'")
    t = np.asarray(values, dtype=float)
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional")
    if len(t) < 2:
        raise ValueError("at least two times are required to build trapezoid weights")
    if not np.all(np.isfinite(t)):
        raise ValueError("times contain non-finite values")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError(
            "times must be strictly increasing for automatic trapezoid weights"
        )

    w = np.empty_like(t, dtype=float)
    w[0] = 0.5 * dt[0]
    w[-1] = 0.5 * dt[-1]
    if len(t) > 2:
        w[1:-1] = 0.5 * (dt[:-1] + dt[1:])
    return w


@dataclass(frozen=True)
class QuadOptions:
    epsabs: float = 1e-11
    epsrel: float = 1e-9
    limit: int = 200
    mp_maxdegree: int | None = 8
    mp_method: str = "gauss-legendre"

    def __post_init__(self) -> None:
        if self.mp_maxdegree is not None and self.mp_maxdegree < 3:
            raise ValueError("mp_maxdegree must be at least 3 or None")
        if self.mp_method not in {"gauss-legendre", "tanh-sinh"}:
            raise ValueError("mp_method must be 'gauss-legendre' or 'tanh-sinh'")


class NonIntegrableTargetAreaError(ValueError):
    """Raised when the target has no finite unweighted area integral."""


class TargetKernel:
    """Abstract target kernel interface."""

    def eval(self, omega: np.ndarray | float) -> np.ndarray | float:
        raise NotImplementedError

    def area(
        self, quad_options: QuadOptions | None = None, omega0: float = 0.0
    ) -> float:
        opts = quad_options or QuadOptions()
        if omega0 < 0 or not np.isfinite(omega0):
            raise ValueError("omega0 must be finite and >= 0")
        val, _ = quad(
            lambda x: float(self.eval(x)),
            omega0,
            np.inf,
            epsabs=opts.epsabs,
            epsrel=opts.epsrel,
            limit=opts.limit,
        )
        return float(val)

    def b_vector(
        self, basis: HLTBasis, quad_options: QuadOptions | None = None
    ) -> np.ndarray:
        opts = quad_options or QuadOptions()
        b = np.empty(len(basis.times), dtype=float)
        for i, t in enumerate(basis.times):

            def integrand(x, tt=t):
                return float(basis.phi(tt, x) * self.eval(x))

            val, _ = quad(
                integrand,
                basis.omega0,
                np.inf,
                epsabs=opts.epsabs,
                epsrel=opts.epsrel,
                limit=opts.limit,
            )
            b[i] = val
        return b

    def eval_mpmath(self, omega: Any, mp_dps: int = 80):
        """Evaluate this built-in target without a binary64 intermediate."""
        from .mpmath_backend import MPWorkspace, target_values

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return target_values(self, omega, mp, MPWorkspace(mp_dps))

    def area_mpmath(
        self, quad_options: QuadOptions | None = None, omega0: Any = 0, mp_dps: int = 80
    ):
        """Evaluate the target area using arbitrary-precision quadrature."""
        from .mpmath_backend import MPWorkspace, target_area

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return target_area(
                self, omega0, quad_options or QuadOptions(), mp, MPWorkspace(mp_dps)
            )

    def b_vector_mpmath(
        self, basis: HLTBasis, quad_options: QuadOptions | None = None, mp_dps: int = 80
    ):
        """Evaluate the HLT target vector using arbitrary-precision quadrature."""
        from .mpmath_backend import MPWorkspace, target_b_vector

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            return target_b_vector(
                self, basis, quad_options or QuadOptions(), mp, MPWorkspace(mp_dps)
            )


@dataclass(frozen=True)
class HighEnergyTarget(TargetKernel):
    """Represented high-energy target theta_delta(omega-am_tau) omega^p B_T(t0, omega).

    The parameter name ``am_tau`` is kept for compatibility with the original
    tau-threshold use case, but it can represent any energy cut in the same
    units as omega.
    """

    t0: float
    am_tau: float
    delta: float
    p: float = 0.0
    T: float | None = None
    periodic: bool = False
    _mp_sources: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sources = {
            "t0": self.t0,
            "am_tau": self.am_tau,
            "delta": self.delta,
            "p": self.p,
            "T": self.T,
        }
        t0, am_tau, delta, p = (
            float(self.t0),
            float(self.am_tau),
            float(self.delta),
            float(self.p),
        )
        T = None if self.T is None else float(self.T)
        if delta <= 0:
            raise ValueError("delta must be positive")
        if t0 <= 0:
            raise ValueError("target time t0 must be > 0")
        if p <= -1:
            raise ValueError("p must be > -1")
        if self.periodic:
            if T is None:
                raise ValueError("T is required for periodic=True")
            if not (0 < t0 < T):
                raise ValueError("periodic target time must satisfy 0 < t0 < T")
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "am_tau", am_tau)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "_mp_sources", sources)

    def theta(self, omega: np.ndarray | float) -> np.ndarray | float:
        return expit(2.0 * (omega - self.am_tau) / self.delta)

    def B(self, omega: np.ndarray | float) -> np.ndarray | float:
        if self.periodic:
            return np.exp(-omega * self.t0) + np.exp(-omega * (self.T - self.t0))
        return np.exp(-omega * self.t0)

    def eval(self, omega: np.ndarray | float) -> np.ndarray | float:
        return self.theta(omega) * np.power(omega, self.p) * self.B(omega)


@dataclass(frozen=True)
class SharpHighEnergyTarget(TargetKernel):
    """Sharp high-energy target theta(omega-am_tau) omega^p B_T(t0, omega).

    This is intended as the unsmeared counterpart of ``HighEnergyTarget`` for
    post-processing diagnostics such as the Delta-smearing systematic.  The
    parameter name ``am_tau`` is kept for compatibility, but it can represent
    any energy cut in the same units as omega.
    """

    t0: float
    am_tau: float
    p: float = 0.0
    T: float | None = None
    periodic: bool = False
    _mp_sources: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sources = {"t0": self.t0, "am_tau": self.am_tau, "p": self.p, "T": self.T}
        t0, am_tau, p = float(self.t0), float(self.am_tau), float(self.p)
        T = None if self.T is None else float(self.T)
        if am_tau < 0 or not np.isfinite(am_tau):
            raise ValueError("am_tau must be finite and >= 0")
        if t0 <= 0:
            raise ValueError("target time t0 must be > 0")
        if p <= -1:
            raise ValueError("p must be > -1")
        if self.periodic:
            if T is None:
                raise ValueError("T is required for periodic=True")
            if not (0 < t0 < T):
                raise ValueError("periodic target time must satisfy 0 < t0 < T")
        object.__setattr__(self, "t0", t0)
        object.__setattr__(self, "am_tau", am_tau)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "_mp_sources", sources)

    def theta(self, omega: np.ndarray | float) -> np.ndarray | float:
        omega_arr = np.asarray(omega, dtype=float)
        out = np.where(omega_arr >= self.am_tau, 1.0, 0.0)
        if np.ndim(omega) == 0:
            return float(out)
        return out

    def B(self, omega: np.ndarray | float) -> np.ndarray | float:
        if self.periodic:
            return np.exp(-omega * self.t0) + np.exp(-omega * (self.T - self.t0))
        return np.exp(-omega * self.t0)

    def eval(self, omega: np.ndarray | float) -> np.ndarray | float:
        return self.theta(omega) * np.power(omega, self.p) * self.B(omega)


@dataclass(frozen=True)
class IntegratedHVPTarget(TargetKernel):
    """Direct omega-space HVP target for high-energy/window observables.

    The target represents

        T_p(omega) = omega**(p-2) T_2^HVP(omega) Theta_win_tilde(omega**2)
                     theta_delta(omega - am_tau),

    where ``omega``, ``am_tau``, ``delta``, ``am_mu``, and all Euclidean window
    parameters are in lattice units.  The p-dependence matches the convention
    ``rho_p(omega) = omega**(2-p) rho(omega**2)`` used by the HLT basis
    ``phi_t^(p)(omega) = omega**p B_T(t, omega)``.
    """

    am_tau: float
    delta: float | None
    p: float = 2.0
    euclidean_window: str | Sequence[float] | None = None
    window_delta: float | None = None
    a_lat_inv_MeV: float | None = None
    am_mu: float | None = None
    m_mu_MeV: float = MUON_MASS_MEV
    alpha: float = ALPHA_EM
    sharp: bool = False
    kernel_quadrature_order: int = 96
    window_quad_options: QuadOptions = QuadOptions(epsabs=1e-9, epsrel=1e-7, limit=100)
    _mp_sources: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        sources = {
            "am_tau": self.am_tau,
            "delta": self.delta,
            "p": self.p,
            "euclidean_window": self.euclidean_window,
            "window_delta": self.window_delta,
            "a_lat_inv_MeV": self.a_lat_inv_MeV,
            "am_mu": self.am_mu,
            "m_mu_MeV": self.m_mu_MeV,
            "alpha": self.alpha,
        }
        am_tau = float(self.am_tau)
        delta = None if self.delta is None else float(self.delta)
        p = float(self.p)
        alpha = float(self.alpha)
        m_mu_MeV = float(self.m_mu_MeV)
        if am_tau < 0 or not np.isfinite(am_tau):
            raise ValueError("am_tau must be finite and >= 0")
        if self.sharp:
            if delta is not None and (delta <= 0 or not np.isfinite(delta)):
                raise ValueError("delta must be positive when supplied")
        elif delta is None or delta <= 0 or not np.isfinite(delta):
            raise ValueError("delta must be positive for a smooth high-energy target")
        if p <= -1:
            raise ValueError("p must be > -1")
        if alpha <= 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be positive and finite")
        if m_mu_MeV <= 0 or not np.isfinite(m_mu_MeV):
            raise ValueError("m_mu_MeV must be positive and finite")
        if self.kernel_quadrature_order < 8:
            raise ValueError("kernel_quadrature_order must be at least 8")

        a_inv = None if self.a_lat_inv_MeV is None else float(self.a_lat_inv_MeV)
        if a_inv is not None and (a_inv <= 0 or not np.isfinite(a_inv)):
            raise ValueError("a_lat_inv_MeV must be positive and finite")

        am_mu = None if self.am_mu is None else float(self.am_mu)
        if am_mu is None:
            if a_inv is None:
                raise ValueError(
                    "provide either am_mu or a_lat_inv_MeV so the muon mass can "
                    "be expressed in lattice units"
                )
            am_mu = m_mu_MeV / a_inv
        if am_mu <= 0 or not np.isfinite(am_mu):
            raise ValueError("am_mu must be positive and finite")
        object.__setattr__(self, "am_tau", am_tau)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "p", p)
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "m_mu_MeV", m_mu_MeV)
        object.__setattr__(self, "a_lat_inv_MeV", a_inv)
        object.__setattr__(self, "am_mu", float(am_mu))

        window = self._normalize_window(self.euclidean_window, a_inv)
        object.__setattr__(self, "euclidean_window", window)

        if window is None:
            if self.window_delta is not None and (
                self.window_delta <= 0 or not np.isfinite(self.window_delta)
            ):
                raise ValueError(
                    "window_delta must be positive and finite when supplied"
                )
            object.__setattr__(
                self,
                "window_delta",
                None if self.window_delta is None else float(self.window_delta),
            )
        else:
            if self.window_delta is None:
                if a_inv is None:
                    raise ValueError(
                        "window_delta is required for numeric windows unless "
                        "a_lat_inv_MeV is supplied for the 0.15 fm default"
                    )
                win_delta = _default_window_delta_lattice(a_inv)
            else:
                win_delta = float(self.window_delta)
            if win_delta <= 0 or not np.isfinite(win_delta):
                raise ValueError("window_delta must be positive and finite")
            object.__setattr__(self, "window_delta", win_delta)
        object.__setattr__(self, "_mp_sources", sources)

    @staticmethod
    def _normalize_window(
        euclidean_window: str | Sequence[float] | None,
        a_lat_inv_MeV: float | None,
    ) -> tuple[float, float] | None:
        if euclidean_window is None:
            return None
        if isinstance(euclidean_window, str):
            key = euclidean_window.upper()
            if key in {"FULL", "TOTAL", "NONE"}:
                return None
            if a_lat_inv_MeV is None:
                raise ValueError(
                    "a_lat_inv_MeV is required for standard string windows "
                    "'ID', 'SD', and 'LD'"
                )
            return _standard_window_lattice(key, a_lat_inv_MeV)
        vals = tuple(float(x) for x in euclidean_window)
        if len(vals) != 2:
            raise ValueError("euclidean_window intervals must have exactly two entries")
        left, right = vals
        if np.isnan(left) or np.isnan(right):
            raise ValueError("euclidean_window bounds cannot be NaN")
        if left >= right:
            raise ValueError("euclidean_window must satisfy left < right")
        return (left, right)

    @classmethod
    def from_high_energy(
        cls,
        am_tau: float,
        delta: float,
        p: float = 2.0,
        euclidean_window: str | Sequence[float] | None = None,
        window_delta: float | None = None,
        a_lat_inv_MeV: float | None = None,
        am_mu: float | None = None,
        m_mu_MeV: float = MUON_MASS_MEV,
        alpha: float = ALPHA_EM,
        kernel_quadrature_order: int = 96,
        window_quad_options: QuadOptions = QuadOptions(
            epsabs=1e-9, epsrel=1e-7, limit=100
        ),
    ) -> IntegratedHVPTarget:
        """Build the smooth high-energy direct HVP target.

        All dimensionful inputs are in lattice units, except the optional
        physical conversion parameters ``a_lat_inv_MeV`` and ``m_mu_MeV``.
        String windows use the standard physical definitions converted with
        ``a_lat_inv_MeV``: ID=[0.4,1.0] fm, SD=(-inf,0.4] fm, LD=[1.0,inf) fm.
        Numeric two-entry windows and numeric ``window_delta`` are interpreted
        directly in lattice units.
        """
        return cls(
            am_tau=am_tau,
            delta=delta,
            p=p,
            euclidean_window=euclidean_window,
            window_delta=window_delta,
            a_lat_inv_MeV=a_lat_inv_MeV,
            am_mu=am_mu,
            m_mu_MeV=m_mu_MeV,
            alpha=alpha,
            sharp=False,
            kernel_quadrature_order=kernel_quadrature_order,
            window_quad_options=window_quad_options,
        )

    @classmethod
    def from_sharp_high_energy(
        cls,
        am_tau: float,
        p: float = 2.0,
        euclidean_window: str | Sequence[float] | None = None,
        window_delta: float | None = None,
        a_lat_inv_MeV: float | None = None,
        am_mu: float | None = None,
        m_mu_MeV: float = MUON_MASS_MEV,
        alpha: float = ALPHA_EM,
        kernel_quadrature_order: int = 96,
        window_quad_options: QuadOptions = QuadOptions(
            epsabs=1e-9, epsrel=1e-7, limit=100
        ),
    ) -> IntegratedHVPTarget:
        """Build the sharp-theta counterpart of ``from_high_energy``."""
        return cls(
            am_tau=am_tau,
            delta=None,
            p=p,
            euclidean_window=euclidean_window,
            window_delta=window_delta,
            a_lat_inv_MeV=a_lat_inv_MeV,
            am_mu=am_mu,
            m_mu_MeV=m_mu_MeV,
            alpha=alpha,
            sharp=True,
            kernel_quadrature_order=kernel_quadrature_order,
            window_quad_options=window_quad_options,
        )

    def theta_cut(self, omega: np.ndarray | float) -> np.ndarray | float:
        omega_arr = np.asarray(omega, dtype=float)
        if self.sharp:
            out = np.where(omega_arr >= self.am_tau, 1.0, 0.0)
        else:
            assert self.delta is not None
            out = 0.5 * (1.0 + np.tanh((omega_arr - self.am_tau) / self.delta))
        if np.ndim(omega) == 0:
            return float(out)
        return out

    def window_theta_t(self, t: np.ndarray | float) -> np.ndarray | float:
        """Euclidean-time window Theta(t) corresponding to ``euclidean_window``."""
        if self.euclidean_window is None:
            if np.ndim(t) == 0:
                return 1.0
            return np.ones_like(np.asarray(t, dtype=float), dtype=float)
        assert self.window_delta is not None
        left, right = self.euclidean_window
        t_arr = np.asarray(t, dtype=float)
        if np.isneginf(left) and np.isposinf(right):
            out = np.ones_like(t_arr, dtype=float)
        elif np.isneginf(left):
            out = 1.0 - _smooth_step(t_arr, right, self.window_delta)
        elif np.isposinf(right):
            out = _smooth_step(t_arr, left, self.window_delta)
        else:
            out = _smooth_step(t_arr, left, self.window_delta) - _smooth_step(
                t_arr, right, self.window_delta
            )
        if np.ndim(t) == 0:
            return float(out)
        return out

    @lru_cache(maxsize=8192)
    def _hvp_kernel_K_scalar(self, omega: float) -> float:
        """Exact standard dispersive HVP kernel K(s), with s=omega**2."""
        omega = float(omega)
        if omega <= 0 or not np.isfinite(omega):
            return 0.0
        return _hvp_kernel_K_analytic_float(omega, self.am_mu)

    def _hvp_kernel_K(self, omega: np.ndarray | float) -> np.ndarray | float:
        omega_arr = np.asarray(omega, dtype=float)
        if omega_arr.ndim == 0:
            return self._hvp_kernel_K_scalar(float(omega_arr))
        return np.asarray(
            [self._hvp_kernel_K_scalar(float(x)) for x in omega_arr], dtype=float
        )

    def _hvp_kernel_Khat_scalar(self, omega: float) -> float:
        if omega <= 0:
            return 0.0
        return float(
            3.0 * (omega / self.am_mu) ** 2 * self._hvp_kernel_K_scalar(float(omega))
        )

    def _full_T2_scalar(self, omega: float) -> float:
        if omega <= 0 or not np.isfinite(omega):
            return 0.0
        return float(8.0 * self.alpha**2 * self._hvp_kernel_K_scalar(omega) / omega)

    @lru_cache(maxsize=20000)
    def _bernecker_meyer_inner_scalar(self, t: float) -> float:
        """Analytic inner kernel appearing in Eq. (12) of arXiv:2205.12963."""
        t = float(t)
        if t <= 0 or not np.isfinite(t):
            return 0.0
        return _bernecker_meyer_inner_analytic_float(t, self.am_mu)

    @lru_cache(maxsize=8192)
    def _window_laplace_scalar(self, omega: float) -> float:
        omega = float(omega)
        if omega <= 0 or not np.isfinite(omega):
            return 0.0
        opts = self.window_quad_options
        
        def integrand(t: float) -> float:
            return float(
                self.window_theta_t(t)
                * np.exp(-omega * t)
                * self._bernecker_meyer_inner_scalar(t)
            )

        val, _ = quad(
            integrand,
            0.0,
            np.inf,
            epsabs=opts.epsabs,
            epsrel=opts.epsrel,
            limit=opts.limit,
        )
        return float(val)

    @lru_cache(maxsize=8192)
    def _tilde_theta_scalar(self, omega: float) -> float:
        if self.euclidean_window is None:
            return 1.0
        omega = float(omega)
        if omega <= 0 or not np.isfinite(omega):
            return 0.0
        khat = self._hvp_kernel_Khat_scalar(omega)
        if khat <= 0 or not np.isfinite(khat):
            return 0.0
        val = self._window_laplace_scalar(omega)
        prefactor = 3.0 * omega**5 / (8.0 * self.am_mu**4 * khat)
        return float(prefactor * val)

    def tilde_theta(self, omega: np.ndarray | float) -> np.ndarray | float:
        """Evaluate the energy-space window weight \u03d1Theta(omega**2)."""
        omega_arr = np.asarray(omega, dtype=float)
        if omega_arr.ndim == 0:
            return self._tilde_theta_scalar(float(omega_arr))
        return np.asarray(
            [self._tilde_theta_scalar(float(x)) for x in omega_arr], dtype=float
        )

    def tilde_theta_mpmath(self, omega: Any, mp_dps: int = 80):
        """Evaluate the energy-space window without binary64 intermediates."""
        from .mpmath_backend import MPWorkspace, tilde_theta

        mp = require_mpmath()
        with mp.workdps(mp_dps):
            workspace = MPWorkspace(mp_dps)
            if isinstance(omega, (list, tuple)) or getattr(omega, "ndim", 0) == 1:
                return mp.matrix(
                    [
                        tilde_theta(self, value, mp, workspace)
                        for value in source_vector(omega)
                    ]
                )
            return tilde_theta(self, omega, mp, workspace)

    def _area_diverges_asymptotically(self) -> bool:
        """Return True if int^inf T_p(omega) d omega is UV divergent.

        For the full HVP kernel K(s) ~ am_mu**2/(3 omega**2), hence
        T_2(omega) ~ const/omega**3 and T_p(omega) ~ const*omega**(p-5).
        The unweighted area integral therefore diverges for p >= 4 whenever
        the Euclidean window has nonzero support at t=0.
        """
        if self.p < 4.0:
            return False
        if self.euclidean_window is None:
            return True
        # Smooth tanh windows are not compactly supported.  Their UV tail is
        # controlled by Theta(0).  Treat tiny leakage as zero for the purpose of
        # the area constraint; this preserves constrained behavior for standard
        # ID/LD windows where the t=0 leakage is exponentially small.
        theta0 = abs(float(self.window_theta_t(0.0)))
        return theta0 > 1e-4

    def area(
        self, quad_options: QuadOptions | None = None, omega0: float = 0.0
    ) -> float:
        if self._area_diverges_asymptotically():
            raise NonIntegrableTargetAreaError(
                "IntegratedHVPTarget has a divergent unweighted area for this "
                "choice of p/window.  Use an unconstrained HLT solve, e.g. "
                "HLTReconstructor(..., area_constraint=False), or keep "
                "area_constraint='auto'."
            )
        return super().area(quad_options=quad_options, omega0=omega0)

    def eval(self, omega: np.ndarray | float) -> np.ndarray | float:
        omega_arr = np.asarray(omega, dtype=float)

        def scalar(x: float) -> float:
            if x <= 0 or not np.isfinite(x):
                return 0.0
            theta = self.theta_cut(x)
            if self.euclidean_window is not None:
                # The dispersive K(s) cancels exactly between T_2^HVP and
                # tilde-Theta.  This reduced form is both faster and more stable.
                return float(
                    self.alpha**2
                    * np.power(x, self.p)
                    * self._window_laplace_scalar(x)
                    * theta
                    / self.am_mu**2
                )
            return float(np.power(x, self.p - 2.0) * self._full_T2_scalar(x) * theta)

        if omega_arr.ndim == 0:
            return scalar(float(omega_arr))
        return np.asarray([scalar(float(x)) for x in omega_arr], dtype=float)


@dataclass(frozen=True)
class IntegratedTarget(TargetKernel):
    """Linear combination of target kernels, e.g. sum_i K_i T_i(omega)."""

    weights: np.ndarray
    targets: tuple[TargetKernel, ...]
    _weight_sources: tuple[Any, ...] = field(init=False, repr=False, compare=False)
    _mp_weight_spec: tuple[tuple[Any, ...], tuple[Any, ...], bool] | None = field(
        init=False, default=None, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        weight_sources = source_vector(self.weights)
        weights = np.asarray(weight_sources, dtype=float)
        if weights.ndim != 1:
            raise ValueError("weights must be one-dimensional")
        if len(weights) != len(self.targets):
            raise ValueError("weights and targets must have the same length")
        if len(weights) == 0:
            raise ValueError("IntegratedTarget requires at least one component")
        if not np.all(np.isfinite(weights)):
            raise ValueError("weights contain non-finite values")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "_weight_sources", weight_sources)

    @classmethod
    def from_high_energy_times(
        cls,
        times: Iterable[float],
        weights: Iterable[float],
        am_tau: float,
        delta: float,
        p: float = 0.0,
        T: float | None = None,
        periodic: bool = False,
        apply_quadrature_weights: bool = True,
    ) -> IntegratedTarget:
        """Build an integrated high-energy target from kernel values on a time grid.

        Parameters
        ----------
        times:
            Target times t_i at which the high-energy correlator appears in the
            numerical time integral.  These times do not have to coincide with
            the HLT basis times.
        weights:
            By default, these are interpreted as kernel values K(t_i), e.g. TMR
            or window-kernel values.  The method automatically multiplies them
            by trapezoid-rule integration weights derived from ``times`` so that
            the integrated target represents sum_i w_i K(t_i) G(t_i).
        apply_quadrature_weights:
            Keep the default ``True`` when ``weights`` are kernel values.  Set to
            ``False`` only if ``weights`` already include the numerical
            integration measure.
        """
        times_source = source_vector(times)
        weights_source = source_vector(weights)
        times_arr = np.asarray(times_source, dtype=float)
        weights_arr = np.asarray(weights_source, dtype=float)
        if times_arr.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if weights_arr.shape != times_arr.shape:
            raise ValueError("weights must have the same shape as times")
        if apply_quadrature_weights:
            weights_arr = trapezoid_weights(times_arr) * weights_arr
        targets = tuple(
            HighEnergyTarget(
                t0=t, am_tau=am_tau, delta=delta, p=p, T=T, periodic=periodic
            )
            for t in times_source
        )
        result = cls(weights=weights_arr, targets=targets)
        object.__setattr__(
            result,
            "_mp_weight_spec",
            (times_source, weights_source, apply_quadrature_weights),
        )
        return result

    @classmethod
    def from_sharp_high_energy_times(
        cls,
        times: Iterable[float],
        weights: Iterable[float],
        am_tau: float,
        p: float = 0.0,
        T: float | None = None,
        periodic: bool = False,
        apply_quadrature_weights: bool = True,
    ) -> IntegratedTarget:
        """Build the sharp-theta counterpart of ``from_high_energy_times``.

        This represents

            sum_i w_i K_i theta(omega-am_tau) omega^p B_T(t_i, omega),

        with the same quadrature-weight convention as
        ``from_high_energy_times``.  Use this as the ``sharp_target`` in
        smearing-systematic diagnostics, while reconstructing with the smooth
        ``IntegratedTarget.from_high_energy_times`` target.
        """
        times_source = source_vector(times)
        weights_source = source_vector(weights)
        times_arr = np.asarray(times_source, dtype=float)
        weights_arr = np.asarray(weights_source, dtype=float)
        if times_arr.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if weights_arr.shape != times_arr.shape:
            raise ValueError("weights must have the same shape as times")
        if apply_quadrature_weights:
            weights_arr = trapezoid_weights(times_arr) * weights_arr
        targets = tuple(
            SharpHighEnergyTarget(t0=t, am_tau=am_tau, p=p, T=T, periodic=periodic)
            for t in times_source
        )
        result = cls(weights=weights_arr, targets=targets)
        object.__setattr__(
            result,
            "_mp_weight_spec",
            (times_source, weights_source, apply_quadrature_weights),
        )
        return result

    def eval(self, omega: np.ndarray | float) -> np.ndarray | float:
        total = 0.0
        for w, target in zip(self.weights, self.targets):
            total = total + w * target.eval(omega)
        return total

    def area(
        self, quad_options: QuadOptions | None = None, omega0: float = 0.0
    ) -> float:
        return float(
            sum(
                w * target.area(quad_options, omega0=omega0)
                for w, target in zip(self.weights, self.targets)
            )
        )

    def b_vector(
        self, basis: HLTBasis, quad_options: QuadOptions | None = None
    ) -> np.ndarray:
        out = np.zeros(len(basis.times), dtype=float)
        for w, target in zip(self.weights, self.targets):
            out += w * target.b_vector(basis, quad_options)
        return out
