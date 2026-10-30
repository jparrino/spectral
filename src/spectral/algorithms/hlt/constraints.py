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

"""Explicit linear constraints on the approximated kernel."""

from dataclasses import dataclass
import math
from ..._registry import serializable


@serializable
@dataclass(frozen=True)
class Area:
    value: object = "target"


@serializable
@dataclass(frozen=True)
class Point:
    energy: object
    value: object = "target"

    def __post_init__(self):
        if not math.isfinite(float(self.energy)) or float(self.energy) < 0:
            raise ValueError("constraint energy must be finite and nonnegative")


def area(value="target"):
    return Area(value)


def point(*, energy, value="target"):
    return Point(energy, value)


def columns(basis, constraints, ctx):
    vectors = []
    for constraint in constraints:
        if isinstance(constraint, Area):
            vectors.append(basis.area(ctx))
        elif isinstance(constraint, Point):
            vectors.append(basis.values(ctx.number(constraint.energy), ctx))
        else:
            raise TypeError(f"unsupported HLT constraint {type(constraint).__name__}")
    return vectors


def right_hand_side(target, basis, constraints, ctx):
    values = []
    for constraint in constraints:
        if constraint.value != "target":
            value = ctx.number(constraint.value)
        elif isinstance(constraint, Area):
            value = target.integral(ctx, basis.omega_0)
        else:
            value = target.scalar(ctx.number(constraint.energy), ctx)
        if not ctx.math.isfinite(value):
            raise ValueError("constraint value is non-finite")
        values.append(value)
    return ctx.array(values)
