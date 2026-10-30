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

"""Optional JKS adapter. Correlations come from retained source resamples."""

import copy
import hashlib
import json
import numpy as np

from ..core.data import CorrelatorData, SourceData
from ..core.result import Result
from .csv import result_items


def _jks():
    try:
        import jks
    except ImportError as exc:
        raise ImportError(
            "install spectral-reconstruction[jks] to use the JKS adapter"
        ) from exc
    return jks


def _fingerprint(name, mean, blocks, tags):
    h = hashlib.sha256(name.encode())
    h.update(np.asarray(mean, dtype=float).tobytes())
    for tag, row in sorted(zip(tags, blocks)):
        h.update(tag.encode())
        h.update(np.asarray(row, dtype=float).tobytes())
    return h.hexdigest()


def read_correlator(database, observable, *, times=None, energy_unit=None):
    """Read every timeslice of a JKS correlator.

    By default, stored components are labelled ``0, 1, ..., N-1``. Optional
    ``times`` supplies a different full time axis in storage order. To keep a
    subset, call ``read_correlator(...).select_times(selected_times)``.
    The original resamples are retained for correlation-preserving export.
    """
    r = database.get(observable)
    if not getattr(r, "_clone_type_str", "").endswith("_jk_resamples"):
        raise ValueError(
            "this adapter currently requires prepared JKS jackknife resamples"
        )
    mean, covariance = np.asarray(r.mean()), np.asarray(r.cov())
    times = tuple(range(mean.size)) if times is None else tuple(times)
    n = len(times)
    if mean.shape != (n,) or covariance.shape != (n, n):
        raise ValueError(
            f"JKS observable {observable!r} has mean shape {mean.shape} and "
            f"covariance shape {covariance.shape}, but {n} time labels require "
            f"shapes {(n,)} and {(n, n)}. "
            "Omit 'times' to use all timeslices numbered from zero, or supply "
            "one time label per stored component in order. "
            "Use .select_times(...) afterwards to select a subset."
        )
    source = SourceData(
        times,
        mean,
        covariance,
        _fingerprint(observable, mean, r.blocks, r.tags),
        observable,
        tuple(r.tags),
        np.asarray(r.blocks),
        copy.deepcopy(r.info),
    )
    return CorrelatorData(
        times, mean, covariance, energy_unit, source, np.eye(len(mean))
    )


def _validate_source(original, current):
    if not np.array_equal(
        np.asarray(original.mean, float), np.asarray(current.mean(), float)
    ):
        raise ValueError(f"source mean changed for {original.observable}")
    old = {
        t: row
        for t, row in zip(original.tags, original.resamples)
        if not t.startswith("!")
    }
    new = {
        t: row for t, row in zip(current.tags, current.blocks) if not t.startswith("!")
    }
    if old.keys() != new.keys() or any(
        not np.array_equal(np.asarray(old[t], float), np.asarray(new[t], float))
        for t in old
    ):
        raise ValueError(f"source resamples changed for {original.observable}")


def to_jks(results, *, source=None):
    """Return a JKS vector in flattened result order; do not write or modify a database.

    Kernel-weighted sums are evaluated at each result's working precision,
    then converted to the binary64 values used by JKS.
    """
    items = result_items(results)
    if not items or any(not isinstance(r, Result) for r in items):
        raise ValueError("JKS export requires successful reconstruction results")
    sources = {}
    for result in items:
        record = result.data.source
        if record.resamples is None or record.observable is None:
            raise ValueError(
                "correlation-preserving export requires original JKS resamples"
            )
        if record.observable in sources and record.id != sources[record.observable].id:
            raise ValueError(
                "different sources share an observable name; use a common, consistently named JKS database"
            )
        sources[record.observable] = record
    jks = _jks()
    if source is None:
        source = jks.resamples()
        for name, record in sources.items():
            source.add(
                name,
                jks.jackknife(
                    np.asarray(record.mean, float).tolist(),
                    np.asarray(record.resamples, float).tolist(),
                    tags=list(record.tags),
                    info=copy.deepcopy(record.info),
                ),
            )
    else:
        for name, record in sources.items():
            _validate_source(record, source.get(name))
    weights = [r.source_weights for r in items]

    def evaluate(values):
        output = []
        for r, w in zip(items, weights):
            with r.numerics.context() as ctx:
                value = ctx.dot(
                    ctx.array(w), ctx.array(values[r.data.source.observable])
                )
            output.append(float(value))
        return output

    observable = source.apply(evaluate)
    if not np.all(np.isfinite(observable.mean())) or not np.all(
        np.isfinite(observable.blocks)
    ):
        raise ArithmeticError("JKS binary64 export overflowed")
    expected = np.asarray([float(r.variance) for r in items])
    actual = np.diag(np.atleast_2d(observable.cov()))
    if not np.allclose(actual, expected, rtol=1e-7, atol=1e-300):
        raise ValueError(
            "JKS statistical variances changed during export; check shared resampling identities and covariance normalization"
        )
    observable.info = {
        **observable.info,
        "spectral/result_ids": json.dumps([r.id for r in items]),
        "spectral/coordinates": json.dumps(
            [dict(r.coordinates) for r in items], default=str
        ),
    }
    return observable


def with_systematic(observable, *, tag, shift, description="", replace=False):
    """Return a new JKS observable carrying one shared signed systematic source."""
    if not isinstance(tag, str) or not tag or tag.startswith("!"):
        raise ValueError("systematic tag must be nonempty and must not start with !")
    if tag in observable.vars() and not replace:
        raise ValueError(f"systematic {tag!r} already exists")
    shift = np.asarray(shift, dtype=float).reshape(-1)
    mean = np.asarray(observable.mean(), dtype=float)
    if shift.shape != mean.shape or not np.all(np.isfinite(shift)):
        raise ValueError("provide one finite signed shift per observable component")
    return observable.addvar(
        tag, (mean + shift).tolist(), description or f"Signed systematic source: {tag}"
    )


def add_named(database, results, *, systematics=None, systematic_tag=None):
    """Add a mapping of scalar results together, preflighting names and provenance."""
    if set(results) & set(database.keys()):
        raise ValueError("destination already contains one of the requested names")
    names, items = tuple(results), tuple(results.values())
    vector = to_jks(items, source=database)
    if systematics is not None:
        if tuple(r.id for r in items) != systematics.result_ids or not systematic_tag:
            raise ValueError(
                "systematic report order must match results and a shared tag is required"
            )
        vector = with_systematic(
            vector,
            tag=systematic_tag,
            shift=-np.asarray(systematics.bias_at_maximum, dtype=float).reshape(-1),
        )
    temporary = _jks().resamples()
    temporary.add("vector", vector)
    components = [
        temporary.apply(lambda r, i=i: [r["vector"][i]]) for i in range(len(items))
    ]
    for name, component in zip(names, components):
        database.add(name, component)
