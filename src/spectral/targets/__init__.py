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

"""Built-in targets and explicit Cartesian grids of target parameters."""

from itertools import product
from .correlator import correlator
from .high_energy import smooth_high_energy, sharp_high_energy
from .gaussian import Gaussian, gaussian
from .hvp import HVP, hvp
from .intermediate_window import TimeWindow, WindowHVP, intermediate_window, hvp_window
from ..core.kernel import Kernel, CallableKernel, LinearCombination, from_function
from ..core.grid import TargetGrid, validate_axis

__all__ = [
    "correlator",
    "smooth_high_energy",
    "sharp_high_energy",
    "Gaussian",
    "gaussian",
    "HVP",
    "hvp",
    "TimeWindow",
    "WindowHVP",
    "intermediate_window",
    "hvp_window",
    "Kernel",
    "CallableKernel",
    "LinearCombination",
    "from_function",
    "grid",
    "named",
]


def grid(factory, *, vary, **fixed):
    if not vary:
        raise ValueError("vary must contain at least one target parameter")
    if set(vary) & set(fixed):
        raise ValueError("a parameter cannot be both fixed and scanned")
    axes = {key: validate_axis(key, values) for key, values in vary.items()}
    targets = tuple(
        factory(**fixed, **dict(zip(axes, point))) for point in product(*axes.values())
    )
    return TargetGrid(axes, targets)


def named(targets):
    return TargetGrid({"kernel": tuple(targets)}, tuple(targets.values()))
