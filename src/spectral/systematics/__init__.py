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

from .density import for_basis, integrand
from .deformations import DeformationSet, DeformedDensity, gaussian_deformations
from .estimate import SystematicReport, estimate
from . import _legacy  # Register explicit migrations for historical archives.

__all__ = [
    "for_basis",
    "integrand",
    "DeformationSet",
    "DeformedDensity",
    "gaussian_deformations",
    "SystematicReport",
    "estimate",
]
