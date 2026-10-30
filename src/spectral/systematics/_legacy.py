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

"""Readers for historical archives whose model values already included p."""

from dataclasses import dataclass

from .._registry import serializable, TYPES
from ..core.kernel import Kernel
from ..models import TwoPionModel, TabulatedSpectralDensity
from .density import BasisDensity
from .deformations import DeformedDensity


@serializable
@dataclass(frozen=True)
class _LegacyDensity(Kernel):
    # Historical tables stored rho_p, sometimes with no declared convention.
    # Preserve their values without interpreting them as a new physical density.
    values: object
    spectral_power: object = None
    _legacy_density = True

    @property
    def domain(self):
        return self.values.domain

    @property
    def energy_unit(self):
        return self.values.energy_unit

    @property
    def breakpoints(self):
        return self.values.breakpoints

    def scalar(self, omega, ctx):
        return self.values.scalar(omega, ctx)


def _tabulated(
    energies, values, spectral_power=None, energy_unit=None, outside="raise"
):
    table = TabulatedSpectralDensity(
        energies, values, energy_unit=energy_unit, outside=outside
    )
    return _LegacyDensity(table, spectral_power)


def _two_pion(**fields):
    power = fields.pop("p", None)
    physical = TwoPionModel(**fields)
    return physical if power is None else BasisDensity(physical, power)


def _two_pion_density(model, spectral_power):
    return BasisDensity(model, spectral_power)


def _deformed(model, energies, factors):
    if isinstance(model, BasisDensity):
        return BasisDensity(
            DeformedDensity(model.density, energies, factors), model.spectral_power
        )
    if isinstance(model, _LegacyDensity):
        return _LegacyDensity(
            DeformedDensity(model.values, energies, factors), model.spectral_power
        )
    return DeformedDensity(model, energies, factors)


TYPES.update(
    {
        "spectral.systematics.model.TabulatedModel": _tabulated,
        "spectral.systematics.two_pion.TwoPionModel": _two_pion,
        "spectral.systematics.two_pion._TwoPionDensity": _two_pion_density,
        "spectral.systematics.deformations.DeformedModel": _deformed,
    }
)
