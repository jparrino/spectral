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
from ..kernels import CorrelatorKernel


@serializable
@dataclass(frozen=True)
class Basis:
    kernel: CorrelatorKernel
    times: tuple
    p: object = 0
    omega_0: object = 0

    def __post_init__(self):
        from .grid import validate_axis

        times = validate_axis("basis times", self.times)
        if not math.isfinite(float(self.p)):
            raise ValueError("p must be finite")
        if not math.isfinite(float(self.omega_0)) or float(self.omega_0) < 0:
            raise ValueError("omega_0 must be finite and nonnegative")
        # The squared basis must be integrable, not just its area.
        if float(self.omega_0) == 0 and float(self.p) <= -0.5:
            raise ValueError(
                "p must exceed -1/2 when omega_0=0 for an integrable Gram matrix"
            )
        for t in times:
            self.kernel.at_time(t)
        object.__setattr__(self, "times", times)

    @property
    def energy_unit(self):
        return self.kernel.energy_unit

    def at_time(self, time):
        return replace(self.kernel.at_time(time), p=self.p)

    def values(self, omega, ctx):
        return ctx.array([self.at_time(t).scalar(omega, ctx) for t in self.times])

    def gram(self, ctx):
        kernels = [self.at_time(t) for t in self.times]
        return ctx.array(
            [
                [
                    ctx.sum(
                        ctx.exp_integral(2 * ctx.number(self.p), a + b, self.omega_0)
                        for a in left.rates(ctx)
                        for b in right.rates(ctx)
                    )
                    for right in kernels
                ]
                for left in kernels
            ]
        )

    def area(self, ctx):
        return ctx.array(
            [self.at_time(t).integral(ctx, self.omega_0) for t in self.times]
        )
