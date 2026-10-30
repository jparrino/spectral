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

"""All time-window-specific mathematics and caches live with the target."""

from dataclasses import dataclass
from functools import lru_cache
import math
from decimal import Decimal, localcontext

from .._registry import serializable
from .hvp import HVP
from ._hvp_math import tmr_inner


def _sigmoid(x, ctx):
    return 1 / (1 + ctx.exp(-x)) if x >= 0 else ctx.exp(x) / (1 + ctx.exp(x))


@serializable
@dataclass(frozen=True)
class TimeWindow:
    start: object = None
    stop: object = None
    width: object = "0.15"

    def __post_init__(self):
        if not math.isfinite(float(self.width)) or float(self.width) <= 0:
            raise ValueError("time-window width must be positive and finite")
        for value in (self.start, self.stop):
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0
            ):
                raise ValueError("finite window endpoints must be nonnegative")
        if (
            self.start is not None
            and self.stop is not None
            and float(self.start) >= float(self.stop)
        ):
            raise ValueError("time-window endpoints must be increasing")

    def scalar(self, time, ctx):
        width = ctx.number(self.width)
        if self.start is None and self.stop is None:
            return ctx.number(1)
        if self.start is None:
            return _sigmoid(2 * (ctx.number(self.stop) - time) / width, ctx)
        if self.stop is None:
            return _sigmoid(2 * (time - ctx.number(self.start)) / width, ctx)
        a, b = (
            2 * (time - ctx.number(self.start)) / width,
            2 * (time - ctx.number(self.stop)) / width,
        )
        return (
            _sigmoid(a, ctx) - _sigmoid(b, ctx)
            if a + b <= 0
            else _sigmoid(-b, ctx) - _sigmoid(-a, ctx)
        )


@lru_cache(maxsize=8192)
def _window_laplace(window, mass, omega, policy):
    with policy.context() as ctx:
        omega, mass = ctx.number(omega), ctx.number(mass)
        points = [x for x in (window.start, window.stop) if x is not None]
        return ctx.integrate(
            lambda t: window.scalar(t, ctx)
            * ctx.exp(-omega * t)
            * tmr_inner(t, mass, ctx),
            points=points,
        )


@serializable
@dataclass(frozen=True)
class WindowHVP(HVP):
    window: TimeWindow = TimeWindow()

    def scalar(self, omega, ctx):
        if omega < 0:
            raise ValueError("energy must be nonnegative")
        if omega == 0 and self.window.stop is None:
            raise ValueError("long-distance HVP target is singular at zero energy")
        mass = ctx.number(self.mass)
        # The dispersive kernel cancels analytically; cache the Laplace transform
        # independently of the smearing cutoff so scan targets share it.
        return (
            self.alpha_value(ctx) ** 2
            * omega ** ctx.number(self.p)
            / mass**2
            * _window_laplace(self.window, mass, omega, ctx.policy)
        )

    def validate_integral(self, ctx, lower, upper, power=1):
        if self.window.stop is None:
            return super().validate_integral(ctx, lower, upper, power)
        p = ctx.number(self.p)
        if upper is None and power * (p - 5) >= -1:
            raise ValueError(
                "smooth-window HVP integral diverges at infinity (the t=0 tail is nonzero)"
            )
        if ctx.number(lower) == 0 and power * p <= -1:
            raise ValueError(
                "window HVP integral diverges at zero for this spectral power"
            )


def intermediate_window(*, mass, start, stop, width, p=2, alpha=None, energy_unit=None):
    return WindowHVP(mass, p, alpha, energy_unit, TimeWindow(start, stop, width))


def hvp_window(*, name, a_lat_inv_MeV, p=2, m_mu_MeV="105.6583745", alpha=None):
    """Standard FULL/SD/ID/LD windows, converting physical fm to lattice units."""
    if float(a_lat_inv_MeV) <= 0:
        raise ValueError("inverse lattice spacing must be positive")
    with localcontext() as ctx:
        ctx.prec = 100
        a = Decimal(str(a_lat_inv_MeV))
        mass = str(Decimal(str(m_mu_MeV)) / a)
        t0 = str(Decimal("0.4") * a / Decimal("197.3269804"))
        t1 = str(Decimal("1.0") * a / Decimal("197.3269804"))
        width = str(Decimal("0.15") * a / Decimal("197.3269804"))
    key = name.upper()
    if key == "FULL":
        return HVP(mass, p, alpha, "lattice")
    endpoints = {"SD": (None, t0), "ID": (t0, t1), "LD": (t1, None)}
    if key not in endpoints:
        raise ValueError("window name must be FULL, SD, ID or LD")
    return WindowHVP(mass, p, alpha, "lattice", TimeWindow(*endpoints[key], width))
