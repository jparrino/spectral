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

"""Linear interpolation of physical spectral-density samples."""

from dataclasses import dataclass
import bisect
import math

from .._registry import serializable
from ..core.spectral_density import SpectralDensity


@serializable
@dataclass(frozen=True)
class TabulatedSpectralDensity(SpectralDensity):
    energies: tuple
    values: tuple
    energy_unit: str | None = None
    outside: str = "raise"

    def __post_init__(self):
        energies, values = tuple(self.energies), tuple(self.values)
        if len(energies) < 2 or len(values) != len(energies):
            raise ValueError(
                "tabulated model needs matching arrays of at least two points"
            )
        if any(not math.isfinite(float(x)) for x in (*energies, *values)):
            raise ValueError("tabulated model contains non-finite values")
        if any(float(b) <= float(a) for a, b in zip(energies[:-1], energies[1:])):
            raise ValueError("model energies must be strictly increasing")
        if self.outside not in {"raise", "zero"}:
            raise ValueError("outside must be raise or zero")
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "values", values)

    @property
    def domain(self):
        return (self.energies[0], self.energies[-1])

    @property
    def breakpoints(self):
        return self.energies

    def scalar(self, omega, ctx):
        energies, values = ctx.cached(
            self, lambda: (ctx.array(self.energies), ctx.array(self.values))
        )
        if omega < energies[0] or omega > energies[-1]:
            if self.outside == "zero":
                return ctx.number(0)
            raise ValueError("energy outside tabulated model domain")
        i = min(len(energies) - 2, max(0, bisect.bisect_right(energies, omega) - 1))
        f = (omega - energies[i]) / (energies[i + 1] - energies[i])
        return (1 - f) * values[i] + f * values[i + 1]
