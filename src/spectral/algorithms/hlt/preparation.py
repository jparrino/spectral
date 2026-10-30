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

from dataclasses import dataclass
import numpy as np

from . import constraints as constraint_ops


@dataclass(frozen=True)
class Problem:
    basis: object
    gram: np.ndarray
    regulator: np.ndarray
    columns: tuple
    constraints: tuple


@dataclass(frozen=True)
class TargetProblem:
    target: object
    overlaps: np.ndarray
    norm_squared: object
    constraint_values: np.ndarray


def prepare(basis, data, solver, numerics):
    with numerics.context() as ctx:
        gram = basis.gram(ctx)
        n = len(basis.times)
        if solver.regulator == "identity":
            regulator = ctx.array(np.eye(n))
        else:
            if solver.covariance_norm is None:
                raise ValueError(
                    "covariance regulator requires an explicit positive covariance_norm"
                )
            norm = ctx.number(solver.covariance_norm)
            if not ctx.math.isfinite(norm) or norm <= 0:
                raise ValueError("covariance_norm must be positive and finite")
            regulator = ctx.array(data.covariance) / norm
        return Problem(
            basis,
            gram,
            regulator,
            tuple(constraint_ops.columns(basis, solver.constraints, ctx)),
            solver.constraints,
        )


def prepare_target(problem, target, numerics):
    basis = problem.basis
    if (
        target.energy_unit
        and basis.energy_unit
        and target.energy_unit != basis.energy_unit
    ):
        raise ValueError("target and basis energy units disagree")
    if target.spectral_power is not None and str(target.spectral_power) != str(basis.p):
        from ...core.grid import coordinate_equal

        if not coordinate_equal(target.spectral_power, basis.p):
            raise ValueError(
                "target and basis spectral powers disagree; construct targets.correlator(basis=basis)"
            )
    with numerics.context() as ctx:
        # Optimized target integration is dispatched by the target itself.
        b, norm = target.basis_integrals(basis, ctx)
        ctx.require_finite(ctx.number(norm), "target norm")
        return TargetProblem(
            target,
            ctx.array(b),
            norm,
            constraint_ops.right_hand_side(target, basis, problem.constraints, ctx),
        )
