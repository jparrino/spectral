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

"""The supplied rho-dominance two-pion toy model, in GeV units."""

from dataclasses import dataclass
import math
from .._registry import serializable
from ..core.spectral_density import SpectralDensity
from ..numerics import Numerics


@serializable
@dataclass(frozen=True)
class TwoPionModel(SpectralDensity):
    """Physical density omega**2 * R(omega**2) / (12*pi**2), in GeV units."""

    m_pi: object = "0.13957039"
    m_rho: object = "0.77526"
    gamma_rho: object = "0.1491"
    energy_unit: str = "GeV"

    def __post_init__(self):
        if any(
            not math.isfinite(float(x)) for x in (self.m_pi, self.m_rho, self.gamma_rho)
        ):
            raise ValueError("model parameters must be finite")
        if (
            float(self.m_pi) <= 0
            or float(self.m_rho) <= 2 * float(self.m_pi)
            or float(self.gamma_rho) <= 0
        ):
            raise ValueError("model requires m_pi>0, m_rho>2*m_pi, gamma_rho>0")

    @property
    def domain(self):
        from decimal import Decimal, localcontext

        mass = Decimal(str(self.m_pi))
        with localcontext() as ctx:
            ctx.prec = max(50, len(mass.as_tuple().digits) + 10)
            return (str(2 * mass), None)

    @property
    def breakpoints(self):
        return (self.m_rho, "0.6", "1.1", "2.5")

    def r_ratio(self, s, ctx):
        mpi, mrho, width = map(ctx.number, (self.m_pi, self.m_rho, self.gamma_rho))
        if s <= 4 * mpi**2:
            return ctx.number(0)
        beta = ctx.sqrt(1 - 4 * mpi**2 / s)
        beta0 = ctx.sqrt(1 - 4 * mpi**2 / mrho**2)
        running_width = (
            width * (mrho**2 / s) * (beta / beta0) ** 3 * (ctx.sqrt(s) / mrho)
        )
        form_squared = mrho**4 / ((mrho**2 - s) ** 2 + s * running_width**2)
        return beta**3 * form_squared / 4

    def R(self, s, *, numerics=None):
        with (numerics or Numerics()).context() as ctx:
            return self.r_ratio(ctx.number(s), ctx)

    def scalar(self, omega, ctx):
        if omega <= ctx.number(self.domain[0]):
            return ctx.number(0)
        return omega**2 * self.r_ratio(omega**2, ctx) / (12 * ctx.pi**2)
