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

"""Minimal HLT reconstruction toolkit."""

from .basis import HLTBasis
from .data import CorrelatorData
from .reconstruction import (
    HLTReconstructor,
    PreparedHLTProblem,
    PreparedHLTSolution,
    ReconstructionResult,
)
from .solver import solve_hlt_constrained, solve_hlt_unconstrained, solve_hlt_endpoint_constrained
from .systematics import (
    KernelMismatchGrid,
    KernelSystematic,
    estimate_kernel_systematic_on_grid,
    estimate_reconstructed_target_difference_on_grid,
    estimate_target_difference_on_grid,
    kernel_mismatch_on_grid,
)
from .targets import (
    HighEnergyTarget,
    IntegratedHVPTarget,
    IntegratedTarget,
    NonIntegrableTargetAreaError,
    QuadOptions,
    SharpHighEnergyTarget,
    TargetKernel,
    fm_to_lattice_units,
    trapezoid_weights,
)

__all__ = [
    "CorrelatorData",
    "HLTBasis",
    "HighEnergyTarget",
    "SharpHighEnergyTarget",
    "IntegratedTarget",
    "IntegratedHVPTarget",
    "NonIntegrableTargetAreaError",
    "QuadOptions",
    "TargetKernel",
    "trapezoid_weights",
    "fm_to_lattice_units",
    "HLTReconstructor",
    "PreparedHLTProblem",
    "PreparedHLTSolution",
    "ReconstructionResult",
    "solve_hlt_constrained",
    "solve_hlt_unconstrained",
    "solve_hlt_endpoint_constrained",
    "KernelMismatchGrid",
    "KernelSystematic",
    "estimate_kernel_systematic_on_grid",
    "estimate_target_difference_on_grid",
    "estimate_reconstructed_target_difference_on_grid",
    "kernel_mismatch_on_grid",
]
