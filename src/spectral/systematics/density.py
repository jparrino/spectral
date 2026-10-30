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

"""Convert physical spectral densities to the convention of a basis."""

from dataclasses import dataclass, replace

from .._registry import serializable
from ..core.grid import coordinate_equal
from ..core.kernel import Kernel, LinearCombination, PrecisionKernel, as_kernel
from ..core.result import ApproximatedKernel
from ..core.spectral_density import SpectralDensity, SpectralIntegrand
from ..kernels import ExponentialKernel
from ..targets.high_energy import HighEnergy


@serializable
@dataclass(frozen=True)
class BasisDensity(Kernel):
    """rho_phys / omega**p; construct with for_basis(density, basis)."""

    density: SpectralDensity
    spectral_power: object

    @property
    def domain(self):
        return self.density.domain

    @property
    def energy_unit(self):
        return self.density.energy_unit

    @property
    def breakpoints(self):
        return self.density.breakpoints

    def scalar(self, omega, ctx):
        lo, hi = self.domain
        if omega < ctx.number(lo) or (hi is not None and omega > ctx.number(hi)):
            return ctx.number(0)
        p = ctx.number(self.spectral_power)
        if omega == 0 and p > 0:
            raise ValueError(
                "converted density is undefined at zero; evaluate the combined integrand instead"
            )
        return self.density.scalar(omega, ctx) * omega ** (-p)

    def integrand(self, kernel):
        return SpectralIntegrand(
            self.density, _physical_kernel(as_kernel(kernel), self.spectral_power)
        )


def for_basis(density, basis):
    """Resolve a physical density using the power and energy units of a basis."""
    return _convert(density, basis.p, basis.energy_unit)


def _convert(density, power, energy_unit):
    unit = getattr(density, "energy_unit", None)
    if unit and energy_unit and unit != energy_unit:
        raise ValueError("density energy unit disagrees with basis/kernel")
    if isinstance(density, SpectralDensity):
        return BasisDensity(density, power)
    # Old archives can contain densities whose power convention was already set.
    if not isinstance(density, BasisDensity) and not getattr(
        density, "_legacy_density", False
    ):
        raise TypeError("density must implement SpectralDensity")
    old_power = getattr(density, "spectral_power", None)
    if old_power is not None and not coordinate_equal(old_power, power):
        raise ValueError("spectral powers of basis, density and reference must agree")
    return density


def integrand(*, density, kernel, basis=None):
    """Return rho_phys * K_p / omega**p, taking p from the kernel or basis.

    For custom kernels with no spectral-power metadata, supply their basis.
    Known kernel factors are cancelled before evaluation, including at zero.
    """
    kernel = as_kernel(kernel)
    power = getattr(kernel, "spectral_power", None)
    if basis is not None:
        if power is not None and not coordinate_equal(power, basis.p):
            raise ValueError("spectral powers of basis and kernel must agree")
        converted = for_basis(density, basis)
    else:
        if power is None:
            raise ValueError(
                "provide basis for a kernel without spectral_power metadata"
            )
        converted = _convert(density, power, kernel.energy_unit)
    if isinstance(converted, BasisDensity):
        return converted.integrand(kernel)
    return SpectralIntegrand(converted, kernel)


@serializable
@dataclass(frozen=True)
class _DividedKernel(Kernel):
    kernel: object
    power: object

    @property
    def energy_unit(self):
        return self.kernel.energy_unit

    def integration_points(self, ctx):
        return self.kernel.integration_points(ctx)

    def scalar(self, omega, ctx):
        p = ctx.number(self.power)
        if omega == 0 and p > 0:
            raise ValueError(
                "custom kernel / omega**p is undefined at zero; provide a positive integration lower bound"
            )
        return self.kernel.scalar(omega, ctx) * omega ** (-p)


def _physical_kernel(kernel, power):
    """Cancel known power factors exactly; preserve full custom kernel weights."""
    declared = getattr(kernel, "spectral_power", None)
    if declared is not None and not coordinate_equal(declared, power):
        raise ValueError("spectral powers of basis, density and reference must agree")
    if coordinate_equal(power, 0):
        return kernel
    if isinstance(kernel, ExponentialKernel):
        return replace(kernel, p=0)
    if isinstance(kernel, (PrecisionKernel, HighEnergy)):
        return replace(kernel, kernel=_physical_kernel(kernel.kernel, power))
    if isinstance(kernel, LinearCombination):
        return replace(
            kernel, kernels=tuple(_physical_kernel(k, power) for k in kernel.kernels)
        )
    if isinstance(kernel, ApproximatedKernel):
        return LinearCombination(
            tuple(
                _physical_kernel(kernel.basis.at_time(t), power)
                for t in kernel.basis.times
            ),
            tuple(kernel.coefficients),
        )
    return _DividedKernel(kernel, power)
