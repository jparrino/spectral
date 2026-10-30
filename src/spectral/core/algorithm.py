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

"""Structural interface for additional linear reconstruction algorithms."""

from typing import Protocol


class LinearAlgorithm(Protocol):
    @property
    def parameter_axes(self) -> dict:
        """Only scanned parameters; scalar settings do not create axes."""
        ...

    @property
    def parameter_points(self) -> tuple[dict, ...]:
        """Cover the Cartesian product of parameter_axes exactly once.

        Additional scalar parameters must be constant across points. Parameter
        names must not overlap target-grid axes or the reserved target_time,
        ensemble, and lambda_ names. Parameters passed to solve_batch are read-only.
        """
        ...

    def prepare(self, basis, data, numerics):
        """Return reusable, process-serializable numerical preparation."""
        ...

    def prepare_target(self, problem, target, numerics): ...

    def solve_batch(self, problem, targets, parameters, numerics):
        """Return one pair or execution.Outcome per target.

        Shared preparation/factorization may raise for the whole batch. Target
        errors should be Outcome(error=...) so neighboring targets survive.
        """
        ...
