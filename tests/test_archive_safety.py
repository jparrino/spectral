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

"""Bound native decoding and preserve atomic no-overwrite behavior."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
from threading import Barrier
import zipfile

import mpmath as mp
import numpy as np
import pytest

import spectral as sp
from spectral.io import archive


def write_payload(path, result):
    payload = {
        "format": "spectral",
        "version": 1,
        "archive": {
            "id": 0,
            "type": "spectral.io.archive.Archive",
            "fields": {"results": result},
        },
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        handle.writestr("manifest.json", json.dumps(payload))
    return path


def array_payload(values=None, dtype="float64", shape=None, id=1):
    return {
        "id": id,
        "dtype": dtype,
        "shape": [1] if shape is None else shape,
        "array": {"list": [1] if values is None else values},
    }


def test_manifest_limit_is_checked_before_decompression(tmp_path, monkeypatch):
    path = write_payload(tmp_path / "large.spectral", "x" * 10000)

    def forbidden(*args, **kwargs):
        pytest.fail("oversized manifest must not be decompressed")

    monkeypatch.setattr(zipfile.ZipFile, "open", forbidden)
    with pytest.raises(ValueError, match="max_manifest_bytes"):
        sp.io.load(path, limits=sp.io.LoadLimits(max_manifest_bytes=1024))


@pytest.mark.parametrize(
    "result,limits,message",
    [
        ({"list": [{"list": [1]}]}, sp.io.LoadLimits(max_depth=4), "max_depth"),
        ({"list": list(range(50))}, sp.io.LoadLimits(max_nodes=30), "max_nodes"),
    ],
)
def test_json_limits_apply_before_parsing(
    tmp_path, monkeypatch, result, limits, message
):
    path = write_payload(tmp_path / "nested.spectral", result)
    monkeypatch.setattr(archive.json, "loads", lambda *a, **kw: pytest.fail("parsed"))
    with pytest.raises(ValueError, match=message):
        sp.io.load(path, limits=limits)


def test_json_scanner_ignores_brackets_and_escaped_quotes_in_strings(tmp_path):
    text = '[{\\"' * 100 + '"\\}]'
    path = write_payload(tmp_path / "text.spectral", text)
    assert sp.io.load(path, limits=sp.io.LoadLimits(max_depth=4)).results == text


@pytest.mark.parametrize(
    "payload,limits,message",
    [
        (
            array_payload(dtype="<U8388608"),
            sp.io.LoadLimits(max_array_bytes=1024),
            "max_array_bytes",
        ),
        (array_payload(dtype="(1000,)f8"), sp.io.LoadLimits(), "dtype"),
        (array_payload(dtype="U0"), sp.io.LoadLimits(), "dtype"),
        (array_payload(shape=[-1]), sp.io.LoadLimits(), "shape"),
        (array_payload(shape=[True]), sp.io.LoadLimits(), "shape"),
        (array_payload(shape=[2**62, 2**62]), sp.io.LoadLimits(), "max_array_elements"),
        (
            array_payload(shape=[1, 1, 1]),
            sp.io.LoadLimits(max_array_ndim=2),
            "max_array_ndim",
        ),
        (array_payload(shape=[2]), sp.io.LoadLimits(), "match shape"),
        (
            array_payload([{"list": [1]}], dtype="object"),
            sp.io.LoadLimits(),
            "scalar elements",
        ),
    ],
)
def test_arrays_are_validated_before_numpy_allocation(
    tmp_path, monkeypatch, payload, limits, message
):
    path = write_payload(tmp_path / "array.spectral", payload)
    monkeypatch.setattr(archive.np, "array", lambda *a, **kw: pytest.fail("allocated"))
    with pytest.raises(ValueError, match=message):
        sp.io.load(path, limits=limits)


def test_array_budget_is_cumulative_and_can_be_increased(tmp_path):
    path = write_payload(
        tmp_path / "many.spectral",
        {
            "tuple": [array_payload(id=1), array_payload(id=2)],
        },
    )
    limits = sp.io.LoadLimits(max_array_bytes=8)
    with pytest.raises(ValueError, match="max_array_bytes"):
        sp.io.load(path, limits=limits)
    result = sp.io.load(path, limits=replace(limits, max_array_bytes=16)).results
    assert len(result) == 2
    assert all(a.tolist() == [1] for a in result)


def test_shared_arrays_are_charged_once(tmp_path):
    path = write_payload(
        tmp_path / "shared.spectral",
        {
            "tuple": [array_payload(), {"ref": 1}],
        },
    )
    first, second = sp.io.load(path, limits=sp.io.LoadLimits(max_array_bytes=8)).results
    assert first is second


def test_empty_arrays_do_not_consume_remaining_storage_budget(tmp_path):
    path = write_payload(
        tmp_path / "empty.spectral",
        {
            "tuple": [array_payload(), array_payload([], shape=[0], id=2)],
        },
    )
    first, empty = sp.io.load(path, limits=sp.io.LoadLimits(max_array_bytes=8)).results
    assert first.tolist() == [1]
    assert empty.shape == (0,)


def test_exact_object_and_string_arrays_still_round_trip(tmp_path):
    with mp.workdps(120):
        number = mp.sqrt(2)
    arrays = (
        np.array([number, "1e-400", None, mp.inf], dtype=object),
        np.array([["alpha", "β"]]),
        np.array(number, dtype=object),
        np.empty((2, 0, 3)),
    )
    path = tmp_path / "normal.spectral"
    sp.io.save(path, results=arrays)
    restored = sp.io.load(path).results
    for before, after in zip(arrays, restored):
        assert before.shape == after.shape
        assert before.dtype == after.dtype
        np.testing.assert_array_equal(before, after)
    assert restored[0][0]._mpf_ == number._mpf_
    assert restored[2].item()._mpf_ == number._mpf_


@pytest.mark.parametrize(
    "value", [[0, 1, 10**9, 1], [0, 3, 0, 1], [0, 2, 0, 2], [2, 1, 0, 1]]
)
def test_malformed_or_excessive_mpf_is_rejected(tmp_path, value):
    path = write_payload(tmp_path / "mpf.spectral", {"mpf": value})
    with pytest.raises(ValueError, match="mpf"):
        sp.io.load(path)


def test_large_integer_budget(tmp_path):
    path = write_payload(tmp_path / "integer.spectral", 2**100)
    with pytest.raises(ValueError, match="max_integer_bits"):
        sp.io.load(path, limits=sp.io.LoadLimits(max_integer_bits=64))
    assert sp.io.load(path).results == 2**100


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_load_limits_require_positive_integers(value):
    with pytest.raises(ValueError, match="positive integer"):
        sp.io.LoadLimits(max_array_bytes=value)


def test_concurrent_writers_publish_only_one_complete_archive(tmp_path, monkeypatch):
    path = tmp_path / "race.spectral"
    barrier = Barrier(2)
    original = archive.tempfile.mkstemp

    def synchronize(*args, **kwargs):
        # Both writers have passed the initial existence check before either
        # can publish its archive. Completion order remains nondeterministic.
        result = original(*args, **kwargs)
        barrier.wait(timeout=10)
        return result

    monkeypatch.setattr(archive.tempfile, "mkstemp", synchronize)

    def write(value):
        try:
            sp.io.save(path, results=value)
            return value
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, ["first", "second"]))
    winners = [value for value in outcomes if value is not None]
    assert len(winners) == 1
    assert sp.io.load(path).results == winners[0]
    assert not list(tmp_path.glob(".spectral-*.tmp"))


def test_no_overwrite_preserves_dangling_symlink(tmp_path):
    path, target = tmp_path / "link.spectral", tmp_path / "missing.spectral"
    path.symlink_to(target)
    with pytest.raises(FileExistsError):
        sp.io.save(path, results="replacement")
    assert path.is_symlink()
    assert not target.exists()
    assert not list(tmp_path.glob(".spectral-*.tmp"))


def test_explicit_overwrite_replaces_archive(tmp_path):
    path = tmp_path / "overwrite.spectral"
    sp.io.save(path, results="first")
    sp.io.save(path, results="second", overwrite=True)
    assert sp.io.load(path).results == "second"
    assert not list(tmp_path.glob(".spectral-*.tmp"))


def test_publish_failure_preserves_destination_and_cleans_temporary(
    tmp_path, monkeypatch
):
    path = tmp_path / "failure.spectral"
    sp.io.save(path, results="first")

    def fail(*args, **kwargs):
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="simulated"):
        sp.io.save(path, results="second", overwrite=True)
    assert sp.io.load(path).results == "first"
    assert not list(tmp_path.glob(".spectral-*.tmp"))
