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

"""Snapshots for scientific mappings and exact, context-independent identity."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
from types import MappingProxyType

import numpy as np


class FrozenMapping(Mapping):
    """A detached, recursively frozen mapping supporting process transport."""

    __slots__ = ("_data",)

    def __init__(self, values=()):
        object.__setattr__(
            self,
            "_data",
            MappingProxyType(
                {key: freeze(value) for key, value in dict(values).items()}
            ),
        )

    def __setattr__(self, name, value):
        raise TypeError("scientific mappings are immutable")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    def __repr__(self):
        return repr(dict(self))

    def __reduce__(self):
        return FrozenMapping, (dict(self),)


def freeze(value):
    """Snapshot standard containers; opaque application objects stay opaque."""
    if isinstance(value, FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return FrozenMapping(value)
    if isinstance(value, (tuple, list)):
        return tuple(freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        result = value.copy()
        if result.dtype == object:
            for index in np.ndindex(result.shape):
                result[index] = freeze(result[index])
        result.setflags(write=False)
        return result
    return value


def state_digest(value):
    """Hash stored values without ambient-precision rounding or display repr.

    Custom non-dataclass objects have instance identity. Such objects are not
    supported by native archives; registered dataclasses have structural identity.
    """

    def state(item):
        if isinstance(item, np.generic):
            return state(item.item())
        if hasattr(item, "_mpf_"):
            return ("mpf", tuple(int(x) for x in item._mpf_))
        if isinstance(item, float):
            return ("float", item.hex())
        if item is None or isinstance(item, (str, bool, int)):
            return (type(item).__name__, item)
        if isinstance(item, np.ndarray):
            return ("array", item.shape, tuple(state(x) for x in item.flat))
        if isinstance(item, Mapping):
            return (
                "mapping",
                tuple(
                    sorted(((state(k), state(v)) for k, v in item.items()), key=repr)
                ),
            )
        if isinstance(item, (tuple, list)):
            return ("sequence", tuple(state(x) for x in item))
        if is_dataclass(item):
            return (
                type(item).__module__,
                type(item).__qualname__,
                tuple(
                    (f.name, state(getattr(item, f.name)))
                    for f in fields(item)
                    if f.init
                ),
            )
        return ("instance", id(item))

    return hashlib.sha256(repr(state(value)).encode()).hexdigest()
