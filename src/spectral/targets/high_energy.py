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

from dataclasses import dataclass, replace
import math
from .._registry import serializable
from ..core.kernel import Kernel, as_kernel


@serializable
@dataclass(frozen=True)
class HighEnergy(Kernel):
    kernel: object
    cutoff: object
    width: object = None

    def __post_init__(self):
        object.__setattr__(self, "kernel", as_kernel(self.kernel))
        if not math.isfinite(float(self.cutoff)) or float(self.cutoff) < 0:
            raise ValueError("cutoff must be finite and nonnegative")
        if self.width is not None and (
            not math.isfinite(float(self.width)) or float(self.width) <= 0
        ):
            raise ValueError(
                "smearing width must be positive and finite; use sharp_high_energy for a sharp cut"
            )

    @property
    def is_time_dependent(self):
        return getattr(
            self.kernel, "is_time_dependent", hasattr(self.kernel, "at_time")
        )

    def at_time(self, time):
        if not self.is_time_dependent:
            raise ValueError("this target already has a fixed time")
        return replace(self, kernel=self.kernel.at_time(time))

    @property
    def spectral_power(self):
        return getattr(self.kernel, "spectral_power", None)

    @property
    def energy_unit(self):
        return getattr(self.kernel, "energy_unit", None)

    @property
    def breakpoints(self):
        cut = float(self.cutoff)
        if self.width is None:
            return (self.cutoff,)
        width = float(self.width)
        return tuple(
            x
            for x in (cut - 8 * width, cut - width, cut, cut + width, cut + 8 * width)
            if x > 0
        )

    def integration_points(self, ctx):
        cut = ctx.number(self.cutoff)
        points = (
            (cut,)
            if self.width is None
            else tuple(cut + i * ctx.number(self.width) for i in (-8, -1, 0, 1, 8))
        )
        return tuple(self.kernel.integration_points(ctx)) + points

    def validate_integral(self, ctx, lower, upper, power=1):
        if self.width is None:
            lower = max(ctx.number(lower), ctx.number(self.cutoff))
        self.kernel.validate_integral(ctx, lower, upper, power)

    def scalar(self, omega, ctx):
        if self.is_time_dependent:
            raise ValueError("target_times are required for this kernel")
        cut = ctx.number(self.cutoff)
        if self.width is None:
            if omega < cut:
                return ctx.number(0)
            step = ctx.number(1)
        else:
            x = 2 * (omega - cut) / ctx.number(self.width)
            step = 1 / (1 + ctx.exp(-x)) if x >= 0 else ctx.exp(x) / (1 + ctx.exp(x))
        return step * self.kernel.scalar(omega, ctx)

    def integral(self, ctx, lower=0, upper=None):
        if self.width is None:
            lower = max(ctx.number(lower), ctx.number(self.cutoff))
            if upper is not None and lower >= ctx.number(upper):
                return ctx.number(0)
            return self.kernel.integral(ctx, lower, upper)
        return super().integral(ctx, lower, upper)


def smooth_high_energy(*, kernel, cutoff, width):
    return HighEnergy(kernel, cutoff, width)


def sharp_high_energy(*, kernel, cutoff):
    return HighEnergy(kernel, cutoff)
