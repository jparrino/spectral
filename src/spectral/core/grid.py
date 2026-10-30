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
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from itertools import product
from fractions import Fraction
import numpy as np

from .._registry import serializable
from ._state import FrozenMapping


def coordinate_key(value):
    """Context-independent coordinate identity, preserving stored mpf digits.

    Ordinary float/string coordinates retain decimal-label matching. Mpmath
    coordinates use their exact stored binary value; reuse the original mpf
    label when selecting a non-binary-exact decimal coordinate.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if hasattr(value, "_mpf_"):
        sign, mantissa, exponent, bits = map(int, value._mpf_)
        if bits < 0:
            raise ValueError("coordinates must be finite")
        numerator = -mantissa if sign else mantissa
        number = (
            Fraction(numerator << exponent)
            if exponent >= 0
            else Fraction(numerator, 1 << -exponent)
        )
        return ("number", number)
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("coordinates must be hashable scalar labels") from exc
        return ("label", value)
    if not number.is_finite():
        raise ValueError("coordinates must be finite")
    return ("number", Fraction(number))


def coordinate_equal(a, b):
    return coordinate_key(a) == coordinate_key(b)


def validate_axis(name, values):
    values = tuple(values)
    if not values:
        raise ValueError(f"axis {name!r} is empty")
    if len(set(coordinate_key(value) for value in values)) != len(values):
        raise ValueError(f"axis {name!r} has duplicate coordinates")
    return values


@serializable
@dataclass(frozen=True)
class TargetGrid:
    axes: Mapping
    targets: tuple

    def __post_init__(self):
        axes = {k: validate_axis(k, v) for k, v in self.axes.items()}
        if any(k in {"lambda", "lambda_", "target_time", "ensemble"} for k in axes):
            raise ValueError("target-grid axes use a reserved coordinate name")
        if int(np.prod([len(v) for v in axes.values()])) != len(self.targets):
            raise ValueError("target count does not match target-grid dimensions")
        object.__setattr__(self, "axes", FrozenMapping(axes))
        object.__setattr__(self, "targets", tuple(self.targets))

    @property
    def coordinates(self):
        return tuple(
            dict(zip(self.axes, values)) for values in product(*self.axes.values())
        )
