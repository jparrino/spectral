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

"""Analytic HVP/TMR formulas retained from the supplied implementation."""

import numpy as np
from mpmath import mp
from scipy.special import kv, modstruve
from functools import lru_cache


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
    if u > 50:
        # The omitted exponential term is below binary64 precision here.
        return float(
            2 * u * u
            - 4 * np.pi * u
            + 4 * (-1 + 4 * np.euler_gamma + 4 * np.log(u))
            + 8 / (u * u)
        )
    if u < 0.2:
        u2 = u * u
        log_u = np.log(u)
        return float(
            u2**2 / 9.0
            + u2**3 * (120.0 * log_u + 120.0 * np.euler_gamma - 169.0) / 5400.0
            + u2**4 * (210.0 * log_u + 210.0 * np.euler_gamma - 401.0) / 88200.0
            + u2**5 * (360.0 * log_u + 360.0 * np.euler_gamma - 787.0) / 2916000.0
            + u2**6 * (3080.0 * log_u + 3080.0 * np.euler_gamma - 7353.0) / 768398400.0
        )

    z = 2.0 * u
    k0 = kv(0, z)
    k1 = kv(1, z)
    closed_form_g = (
        -4.0 * u * (k1 + np.pi * u * (k0 * modstruve(-1, z) + k1 * modstruve(0, z)))
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
    reciprocal_integral = (2.0 / h) * (np.arctan((2.0 - r) / h) + np.arctan(r / h))
    log_coefficient = 0.5 * r * (2.0 - r)
    reciprocal_coefficient = -0.5 * r * (r * r - 4.0 * r + 2.0)
    return float(
        0.5
        - r
        - log_coefficient * np.log(r)
        + reciprocal_coefficient * reciprocal_integral
    )


def _hvp_mp(omega, mass):
    r = mp.power(omega / mass, 2)

    # Closed form of int_0^1 dx x^2(1-x)/(x^2+r(1-x)).  The r>4 branch
    # uses a cancellation-free conformal variable.  For small x, evaluate the
    # log1p remainder directly as its alternating series; evaluating three
    # nearly equal terms separately becomes inaccurate at the extreme nodes
    # of the outer omega quadrature.
    with mp.extradps(15):
        if r > 4:
            beta = mp.sqrt(1 - 4 / r)
            x = 4 / (r * (1 + beta) ** 2)
            if x < mp.mpf("0.075"):
                remainder = mp.zero
                power = x**3
                sign = 1
                n = 3
                while True:
                    term = sign * power / n
                    updated = remainder + term
                    if updated == remainder:
                        break
                    remainder = updated
                    power *= x
                    sign = -sign
                    n += 1
            else:
                remainder = mp.log1p(x) - x + x * x / 2
            value = (
                x * x * (2 - x * x) / 2
                + (1 + x) ** 2 * (1 + x**-2) * remainder
                + (1 + x) * x * x * mp.log(x) / (1 - x)
            )
        elif r == 4:
            value = 8 * mp.log(2) - mp.mpf("5.5")
        else:
            h = mp.sqrt(r * (4 - r))
            reciprocal_integral = 2 * (mp.atan((2 - r) / h) + mp.atan(r / h)) / h
            log_coefficient = r * (2 - r) / 2
            reciprocal_coefficient = -r * (r * r - 4 * r + 2) / 2
            value = (
                mp.mpf("0.5")
                - r
                - log_coefficient * mp.log(r)
                + reciprocal_coefficient * reciprocal_integral
            )
    return +value


def _tmr_mp(time, mass):
    u = time * mass

    # Use analytic limiting representations only once their conservative
    # omitted-term estimates are below the current working precision.  This
    # prevents special-function failures at the extreme nodes generated by an
    # infinite-interval quadrature without imposing a fixed numerical cutoff.
    log_u = mp.log(u)
    u2 = u * u
    short_value = (
        u2**2 / 9
        + u2**3 * (120 * log_u + 120 * mp.euler - 169) / 5400
        + u2**4 * (210 * log_u + 210 * mp.euler - 401) / 88200
        + u2**5 * (360 * log_u + 360 * mp.euler - 787) / 2916000
        + u2**6 * (3080 * log_u + 3080 * mp.euler - 7353) / 768398400
    )
    short_remainder_bound = 10 * u**14 * (abs(log_u) + 1)

    large_value = 2 * u2 - 4 * mp.pi * u + 4 * (-1 + 4 * mp.euler + 4 * log_u) + 8 / u2
    large_remainder_bound = 4 * mp.sqrt(mp.pi / u) * mp.exp(-2 * u) * (1 + 1 / u)

    relative_goal = mp.eps / 10
    if u < 1 and short_remainder_bound <= relative_goal * abs(short_value):
        value = short_value
    elif u > 1 and large_remainder_bound <= relative_goal * abs(large_value):
        value = large_value
    else:
        # Individual terms scale as u**-2 while the result scales as u**4
        # near the origin.  Guard digits protect those six powers of
        # cancellation in the exact central-region representation.
        cancellation_digits = max(0, int(mp.ceil(-6 * mp.log10(u))))
        with mp.extradps(max(20, cancellation_digits + 10)):
            z = 2 * u
            k0 = mp.besselk(0, z)
            k1 = mp.besselk(1, z)
            closed_form_g = (
                -4
                * u
                * (k1 + mp.pi * u * (k0 * mp.struvel(-1, z) + k1 * mp.struvel(0, z)))
            )
            value = 2 * (
                -2
                + 8 * mp.euler
                + 4 / (u * u)
                + u * u
                - 8 * k1 / u
                + 8 * mp.log(u)
                + closed_form_g
            )
        value = +value
    return value


def hvp_kernel(omega, mass, ctx):
    if omega <= 0:
        return ctx.number(0)
    return (
        _hvp_mp(omega, mass) if ctx.is_mp else _hvp_kernel_K_analytic_float(omega, mass)
    )


@lru_cache(maxsize=32768)
def _cached_tmr(time, mass, policy):
    with policy.context() as ctx:
        time, mass = ctx.number(time), ctx.number(mass)
        if time <= 0:
            return ctx.number(0)
        return (
            _tmr_mp(time, mass)
            if ctx.is_mp
            else _bernecker_meyer_inner_analytic_float(time, mass)
        )


def tmr_inner(time, mass, ctx):
    return _cached_tmr(time, mass, ctx.policy)
