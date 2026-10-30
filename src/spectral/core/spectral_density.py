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

"""Physical spectral densities and integration against physical kernels."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

from .._registry import serializable
from ..numerics import Numerics
from .kernel import Kernel, as_kernel


class SpectralDensity(ABC):
    """A physical density per unit energy, independent of reconstruction bases.

    Implement ``scalar(omega, ctx)`` using the active numerical context. Declare
    the support in ``domain``, the energy unit, and any quadrature breakpoints.
    Distribution-valued models may override ``integrate`` to pair the complete
    spectrum with a smooth kernel, and reject pointwise evaluation in ``scalar``.
    Basis conversion and systematic estimation preserve this integration hook.
    """

    domain = (0, None)
    energy_unit = None
    breakpoints = ()

    @abstractmethod
    def scalar(self, omega, ctx):
        """Evaluate the physical density at one energy."""

    def __call__(self, energies, *, numerics=None):
        with (numerics or Numerics()).context() as ctx:
            values = np.asarray(energies, dtype=object)
            out = [self.scalar(ctx.number(w), ctx) for w in values.flat]
            ctx.require_finite(out, "spectral density")
            if values.ndim == 0:
                return out[0]
            return np.array(out, dtype=object if ctx.is_mp else float).reshape(
                values.shape
            )

    @staticmethod
    def from_function(function, *, domain=(0, None), breakpoints=(), energy_unit=None):
        """Adapt a physical density f(omega, ctx); callable code is not archived."""
        return CallableSpectralDensity(
            function, tuple(domain), tuple(breakpoints), energy_unit
        )

    def integrand(self, kernel):
        """Multiply by a physical kernel, with no reconstruction conversion."""
        return SpectralIntegrand(self, as_kernel(kernel))

    def integrate(self, kernel, *, numerics=None, lower=None, upper=None):
        """Integrate the physical density times the supplied physical kernel."""
        return self.integrand(kernel)._quadrature(
            numerics=numerics, lower=lower, upper=upper
        )


@dataclass(frozen=True)
class CallableSpectralDensity(SpectralDensity):
    function: object
    domain: tuple = (0, None)
    breakpoints: tuple = ()
    energy_unit: str | None = None

    def scalar(self, omega, ctx):
        return ctx.number(self.function(omega, ctx))


@serializable
@dataclass(frozen=True)
class SpectralIntegrand(Kernel):
    density: object
    kernel: object

    def __post_init__(self):
        object.__setattr__(self, "kernel", as_kernel(self.kernel))
        left, right = self.density.energy_unit, self.kernel.energy_unit
        if left and right and left != right:
            raise ValueError("density and kernel energy units disagree")

    @property
    def energy_unit(self):
        return self.density.energy_unit or self.kernel.energy_unit

    def integration_points(self, ctx):
        return tuple(self.density.breakpoints) + tuple(
            self.kernel.integration_points(ctx)
        )

    def scalar(self, omega, ctx):
        lo, hi = self.density.domain
        if omega < ctx.number(lo) or (hi is not None and omega > ctx.number(hi)):
            return ctx.number(0)
        return self.density.scalar(omega, ctx) * self.kernel.scalar(omega, ctx)

    def integrate(self, *, numerics=None, lower=None, upper=None):
        if isinstance(self.density, SpectralDensity):
            return self.density.integrate(
                self.kernel, numerics=numerics, lower=lower, upper=upper
            )
        # Historical archive densities predate SpectralDensity.
        return self._quadrature(numerics=numerics, lower=lower, upper=upper)

    def _quadrature(self, *, numerics=None, lower=None, upper=None):
        with (numerics or Numerics()).context() as ctx:
            lo, hi = self.density.domain
            lo = (
                ctx.number(lo)
                if lower is None
                else max(ctx.number(lo), ctx.number(lower))
            )
            if upper is not None:
                hi = (
                    ctx.number(upper)
                    if hi is None
                    else min(ctx.number(hi), ctx.number(upper))
                )
            if hi is not None and ctx.number(hi) <= lo:
                return ctx.number(0)
            return ctx.integrate(
                lambda w: self.scalar(w, ctx), lo, hi, self.integration_points(ctx)
            )


def integrate(*, density, kernel, numerics=None, lower=None, upper=None):
    """Integrate a physical density against a physical kernel."""
    return density.integrate(kernel, numerics=numerics, lower=lower, upper=upper)
