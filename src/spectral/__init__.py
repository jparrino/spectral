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

"""Reconstruct spectral integrals, then assess and export their uncertainties."""

from .core import (
    Basis,
    CorrelatorData,
    Reconstructor,
    Result,
    ResultCollection,
    Kernel,
    CallableKernel,
    LinearCombination,
    TargetGrid,
    ReconstructionError,
    SpectralDensity,
    integrate,
)
from . import (
    kernels,
    targets,
    numerics,
    execution,
    algorithms,
    models,
    systematics,
    io,
    scan,
)

__version__ = "0.1.0"

__all__ = [
    "SpectralDensity",
    "integrate",
    "models",
    "Basis",
    "CorrelatorData",
    "Reconstructor",
    "Result",
    "ResultCollection",
    "Kernel",
    "CallableKernel",
    "LinearCombination",
    "TargetGrid",
    "ReconstructionError",
    "kernels",
    "targets",
    "numerics",
    "execution",
    "algorithms",
    "systematics",
    "io",
    "scan",
]
