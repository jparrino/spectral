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

from dataclasses import dataclass
import math
from .._registry import serializable
from ..core.kernel import Kernel


@serializable
@dataclass(frozen=True)
class Gaussian(Kernel):
    center: object
    width: object
    energy_unit: str | None = None

    def __post_init__(self):
        if (
            not math.isfinite(float(self.center))
            or not math.isfinite(float(self.width))
            or float(self.width) <= 0
        ):
            raise ValueError(
                "Gaussian center must be finite and width positive and finite"
            )

    @property
    def breakpoints(self):
        return tuple(
            float(self.center) + k * float(self.width) for k in (-8, -1, 0, 1, 8)
        )

    def integration_points(self, ctx):
        center, width = ctx.number(self.center), ctx.number(self.width)
        return tuple(center + k * width for k in (-8, -1, 0, 1, 8))

    def scalar(self, omega, ctx):
        width = ctx.number(self.width)
        x = (omega - ctx.number(self.center)) / width
        return ctx.exp(-x * x / 2) / (ctx.sqrt(2 * ctx.pi) * width)


def gaussian(*, center, width, energy_unit=None):
    return Gaussian(center, width, energy_unit)
