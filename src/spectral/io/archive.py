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

"""Versioned, atomic JSON archives with exact mpmath values and shared objects."""

from dataclasses import dataclass, fields, is_dataclass
from collections.abc import Mapping
from pathlib import Path
import json
import math
import os
import tempfile
import zipfile
import mpmath as mp
import numpy as np

from .._registry import TYPES, serializable
from ._limits import LoadLimits, check_json_limits


@serializable
@dataclass(frozen=True)
class Archive:
    results: object
    systematics: object = None
    metadata: object = None


class Encoder:
    def __init__(self):
        self.seen = {}

    def encode(self, value):
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if hasattr(value, "_mpf_"):
            return {"mpf": [int(x) for x in value._mpf_]}
        if isinstance(value, np.generic):
            return self.encode(value.item())
        if isinstance(value, float):
            return {"float": value.hex()}
        if isinstance(value, tuple):
            return {"tuple": [self.encode(v) for v in value]}
        if isinstance(value, list):
            return {"list": [self.encode(v) for v in value]}
        if isinstance(value, Mapping):
            return {
                "dict": [[self.encode(k), self.encode(v)] for k, v in value.items()]
            }
        previous = self.seen.get(id(value))
        if previous is not None:
            return {"ref": previous}
        index = len(self.seen)
        self.seen[id(value)] = index
        if isinstance(value, np.ndarray):
            return {
                "id": index,
                "array": self.encode(value.tolist()),
                "shape": value.shape,
                "dtype": "object" if value.dtype == object else str(value.dtype),
            }
        name = type(value).__module__ + "." + type(value).__qualname__
        if is_dataclass(value) and name in TYPES:
            return {
                "id": index,
                "type": name,
                "fields": {
                    f.name: self.encode(getattr(value, f.name))
                    for f in fields(value)
                    if f.init
                },
            }
        raise TypeError(
            f"{name} is not a registered archive type; use a registered dataclass instead of an arbitrary callable"
        )


class Decoder:
    def __init__(self, limits=None):
        self.limits = limits if limits is not None else LoadLimits()
        if not isinstance(self.limits, LoadLimits):
            raise TypeError("limits must be a LoadLimits instance")
        self.objects = {}
        self.array_bytes = 0

    def decode(self, value):
        if not isinstance(value, dict):
            return value
        if "ref" in value:
            return self.objects[value["ref"]]
        if "mpf" in value:
            parts = value["mpf"]
            if (
                not isinstance(parts, (tuple, list))
                or len(parts) != 4
                or any(type(x) is not int for x in parts)
            ):
                raise ValueError("invalid archive mpf")
            sign, mantissa, exponent, bits = parts
            special = (sign, mantissa, exponent, bits) in {
                (0, 0, 0, 0),
                (0, 0, -456, -2),
                (1, 0, -789, -3),
                (0, 0, -123, -1),
            }
            if not special and (
                sign not in (0, 1)
                or mantissa <= 0
                or mantissa % 2 == 0
                or bits != mantissa.bit_length()
            ):
                raise ValueError("invalid archive mpf")
            if (
                bits > self.limits.max_integer_bits
                or abs(exponent) > self.limits.max_integer_bits
            ):
                raise ValueError("archive mpf exceeds max_integer_bits")
            return mp.make_mpf(tuple(parts))
        if "float" in value:
            return float.fromhex(value["float"])
        if "tuple" in value:
            return tuple(self.decode(x) for x in value["tuple"])
        if "list" in value:
            return [self.decode(x) for x in value["list"]]
        if "dict" in value:
            return {self.decode(k): self.decode(v) for k, v in value["dict"]}
        if "array" in value:
            result = self._array(value)
        elif "type" in value:
            if value["type"] not in TYPES:
                raise ValueError(f"unregistered archive type: {value['type']}")
            fields = {k: self.decode(v) for k, v in value["fields"].items()}
            result = TYPES[value["type"]](**fields)
        else:
            raise ValueError("invalid archive object")
        self.objects[value["id"]] = result
        return result

    def _array(self, value):
        shape = value["shape"]
        if not isinstance(shape, (list, tuple)) or any(
            type(n) is not int or n < 0 for n in shape
        ):
            raise ValueError("invalid archive array shape")
        if len(shape) > self.limits.max_array_ndim:
            raise ValueError("archive array exceeds max_array_ndim")
        size = math.prod(shape)
        if size > self.limits.max_array_elements or any(
            n > self.limits.max_array_elements for n in shape
        ):
            raise ValueError("archive array exceeds max_array_elements")
        name = value["dtype"]
        if not isinstance(name, str) or len(name) > 64:
            raise ValueError("unsupported archive array dtype")
        try:
            dtype = np.dtype(name)
        except (TypeError, ValueError) as exc:
            raise ValueError("unsupported archive array dtype") from exc
        if (
            dtype.fields is not None
            or dtype.subdtype is not None
            or dtype.kind not in "biufUO"
            or dtype.itemsize == 0
        ):
            raise ValueError("unsupported archive array dtype")
        # Charge allocations cumulatively, counting a shared array only once.
        # The itemsize check also rejects oversized dtypes on empty arrays.
        nbytes = size * dtype.itemsize
        if (
            dtype.itemsize > self.limits.max_array_bytes
            or nbytes > self.limits.max_array_bytes - self.array_bytes
        ):
            raise ValueError("archive arrays exceed max_array_bytes")
        self.array_bytes += nbytes
        scalars = []

        def flatten(item, axis):
            if axis < len(shape):
                if (
                    not isinstance(item, dict)
                    or set(item) != {"list"}
                    or not isinstance(item["list"], list)
                    or len(item["list"]) != shape[axis]
                ):
                    raise ValueError("archive array data do not match shape")
                for child in item["list"]:
                    flatten(child, axis + 1)
            else:
                # Object arrays hold scalar data, never container/array refs or
                # constructors that could expand beyond the declared shape.
                if isinstance(item, dict):
                    if set(item) not in ({"mpf"}, {"float"}):
                        raise ValueError("archive arrays require scalar elements")
                    item = self.decode(item)
                elif item is not None and type(item) not in (str, bool, int, float):
                    raise ValueError("archive arrays require scalar elements")
                scalars.append(item)

        flatten(value["array"], 0)
        return np.array(scalars, dtype=dtype).reshape(shape)


def save(path, *, results, systematics=None, metadata=None, overwrite=False):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    payload = {
        "format": "spectral",
        "version": 1,
        "archive": Encoder().encode(Archive(results, systematics, metadata or {})),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".spectral-", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("manifest.json", json.dumps(payload, allow_nan=False))
        if overwrite:
            os.replace(temporary, path)
        else:
            # Both names are on the same filesystem. Linking publishes the
            # completed file atomically and refuses an existing destination,
            # including dangling symlinks and files created after our precheck.
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load(path, *, limits=None):
    """Load a native archive using configurable manifest and array budgets.

    See LoadLimits for defaults. Limits bound input decoding, not memory or CPU
    used by application-registered dataclass constructors.
    """
    decoder = Decoder(limits)
    with zipfile.ZipFile(path) as archive:
        info = archive.getinfo("manifest.json")
        maximum = decoder.limits.max_manifest_bytes
        if info.file_size > maximum:
            raise ValueError("archive manifest exceeds max_manifest_bytes")
        with archive.open(info) as stream:
            document = stream.read(maximum + 1)
        if len(document) > maximum:
            raise ValueError("archive manifest exceeds max_manifest_bytes")
    document = document.decode("utf-8-sig")
    check_json_limits(document, decoder.limits)

    def parse_integer(text):
        if len(text.lstrip("-")) > decoder.limits.max_integer_bits:
            raise ValueError("archive integer exceeds max_integer_bits")
        number = int(text)
        if number.bit_length() > decoder.limits.max_integer_bits:
            raise ValueError("archive integer exceeds max_integer_bits")
        return number

    try:
        payload = json.loads(document, parse_int=parse_integer)
    except RecursionError as exc:
        raise ValueError("archive JSON nesting exceeds parser capacity") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("format") != "spectral"
        or payload.get("version") != 1
    ):
        raise ValueError("unsupported spectral archive format/version")
    return decoder.decode(payload["archive"])
