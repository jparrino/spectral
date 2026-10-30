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

"""Budgets applied before parsing native manifests or allocating their arrays."""

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class LoadLimits:
    """Configurable native archive input limits; all values are positive integers.

    Array bytes are cumulative NumPy storage, including object-array pointers;
    Python scalar/container overhead and constructor copies are not included.
    Manifest bytes and JSON node counts additionally bound serialized objects.
    max_integer_bits also bounds the absolute binary exponent of encoded mpf
    values, whose use as coordinates can otherwise allocate enormous integers.
    """

    max_manifest_bytes: int = 64 * 1024**2
    max_depth: int = 64
    max_nodes: int = 1_000_000
    max_array_ndim: int = 32
    max_array_elements: int = 10_000_000
    max_array_bytes: int = 256 * 1024**2
    max_integer_bits: int = 16_384

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a positive integer")


def check_json_limits(document, limits):
    """Count nesting and tokens before json.loads can allocate their containers.

    Strings (including object keys), scalar literals, arrays and objects each
    count as a node. Brackets and escaped quotes inside strings are ignored.
    JSON syntax validation remains the standard parser's responsibility.
    """
    depth = nodes = 0
    in_string = escaped = in_atom = False
    for char in document:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            in_atom = False
            nodes += 1
        elif char in "[{":
            depth += 1
            nodes += 1
            in_atom = False
            if depth > limits.max_depth:
                raise ValueError("archive JSON exceeds max_depth")
        elif char in "]}":
            depth -= 1
            in_atom = False
        elif char in ",: \t\r\n":
            in_atom = False
        elif not in_atom:
            in_atom = True
            nodes += 1
        if nodes > limits.max_nodes:
            raise ValueError("archive JSON exceeds max_nodes")
