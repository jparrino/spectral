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

"""Forward correlator kernels, independent of reconstruction algorithms."""

from dataclasses import dataclass
import math

from .core.kernel import Kernel
from ._registry import serializable


@serializable
@dataclass(frozen=True)
class CorrelatorKernel:
    extent: object = None
    energy_unit: str | None = None

    def __post_init__(self):
        if self.extent is not None and (
            not math.isfinite(float(self.extent)) or float(self.extent) <= 0
        ):
            raise ValueError("periodic extent must be positive and finite")

    def at_time(self, time):
        if not math.isfinite(float(time)) or float(time) <= 0:
            raise ValueError("correlator target times must be positive and finite")
        if self.extent is not None and float(time) >= float(self.extent):
            raise ValueError("periodic times must be smaller than the temporal extent")
        return ExponentialKernel(time, self.extent, energy_unit=self.energy_unit)


@serializable
@dataclass(frozen=True)
class ExponentialKernel(Kernel):
    time: object
    extent: object = None
    p: object = 0
    energy_unit: str | None = None

    def __post_init__(self):
        CorrelatorKernel(self.extent, self.energy_unit).__post_init__()
        if not math.isfinite(float(self.time)) or float(self.time) <= 0:
            raise ValueError("time must be positive and finite")
        if self.extent is not None and float(self.time) >= float(self.extent):
            raise ValueError("time must be smaller than extent")

    @property
    def spectral_power(self):
        return self.p

    def rates(self, ctx):
        t = ctx.number(self.time)
        return (t,) if self.extent is None else (t, ctx.number(self.extent) - t)

    def scalar(self, omega, ctx):
        return omega ** ctx.number(self.p) * ctx.sum(
            ctx.exp(-omega * t) for t in self.rates(ctx)
        )

    def integral(self, ctx, lower=0, upper=None):
        if upper is not None:
            return super().integral(ctx, lower, upper)
        return ctx.sum(ctx.exp_integral(self.p, t, lower) for t in self.rates(ctx))

    def basis_integrals(self, basis, ctx):
        """Analytic overlaps for exponential targets, owned by the kernel."""
        p = ctx.number(basis.p) + ctx.number(self.p)
        b = [
            ctx.sum(
                ctx.exp_integral(p, a + b, basis.omega_0)
                for a in basis.at_time(t).rates(ctx)
                for b in self.rates(ctx)
            )
            for t in basis.times
        ]
        norm = ctx.sum(
            ctx.exp_integral(2 * ctx.number(self.p), a + b, basis.omega_0)
            for a in self.rates(ctx)
            for b in self.rates(ctx)
        )
        return b, norm


def exponential(*, energy_unit=None):
    return CorrelatorKernel(energy_unit=energy_unit)


def periodic_correlator(*, extent, energy_unit=None):
    return CorrelatorKernel(extent=extent, energy_unit=energy_unit)
