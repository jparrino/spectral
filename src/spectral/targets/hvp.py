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

"""Full HVP energy kernel in the supplied rho_p convention."""

from dataclasses import dataclass
import math
from .._registry import serializable
from ..core.kernel import Kernel
from ._hvp_math import hvp_kernel


@serializable
@dataclass(frozen=True)
class HVP(Kernel):
    mass: object
    p: object = 2
    alpha: object = None
    energy_unit: str | None = None

    def __post_init__(self):
        if not math.isfinite(float(self.mass)) or float(self.mass) <= 0:
            raise ValueError("muon mass must be positive and finite")
        if not math.isfinite(float(self.p)):
            raise ValueError("p must be finite")
        if self.alpha is not None and (
            not math.isfinite(float(self.alpha)) or float(self.alpha) <= 0
        ):
            raise ValueError("alpha must be positive and finite")

    @property
    def spectral_power(self):
        return self.p

    @property
    def breakpoints(self):
        return (float(self.mass), 2 * float(self.mass))

    def integration_points(self, ctx):
        mass = ctx.number(self.mass)
        return (mass, 2 * mass)

    def alpha_value(self, ctx):
        return (
            1 / ctx.number("137.035999084")
            if self.alpha is None
            else ctx.number(self.alpha)
        )

    def scalar(self, omega, ctx):
        if omega <= 0:
            raise ValueError(
                "full HVP energy kernel must be evaluated at positive energies"
            )
        return (
            8
            * self.alpha_value(ctx) ** 2
            * omega ** (ctx.number(self.p) - 3)
            * hvp_kernel(omega, ctx.number(self.mass), ctx)
        )

    def validate_integral(self, ctx, lower, upper, power=1):
        p = ctx.number(self.p)
        if upper is None and power * (p - 5) >= -1:
            raise ValueError(
                "HVP integral diverges at infinity for this spectral power"
            )
        if ctx.number(lower) == 0 and power * (p - 3) <= -1:
            raise ValueError(
                "HVP integral diverges at zero; provide a physical positive omega_0"
            )


def hvp(*, mass, p=2, alpha=None, energy_unit=None):
    return HVP(mass, p, alpha, energy_unit)
