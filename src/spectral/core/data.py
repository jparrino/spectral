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

from dataclasses import dataclass, field
from collections.abc import Mapping
from uuid import uuid4
import numpy as np
import mpmath as mp

from .._registry import serializable
from ._state import FrozenMapping


def readonly(values):
    a = np.array(values, dtype=object, copy=True)
    a.setflags(write=False)
    return a


def same_source_values(a, b):
    """Compare fields that determine a source's numerical/correlation identity."""
    return a is b or all(
        np.array_equal(getattr(a, field), getattr(b, field))
        for field in ("times", "mean", "covariance", "resamples", "tags", "observable")
    )


def _validate_projection(source, projection, mean, covariance):
    """Require the stated moments to come from the retained source projection."""
    from ..numerics import mpmath

    with mpmath(80).context() as ctx:
        p, original_mean, original_covariance = map(
            ctx.array, (projection, source.mean, source.covariance)
        )
        size = len(original_mean)
        if (
            size == 0
            or len(source.times) != size
            or original_mean.shape != (size,)
            or original_covariance.shape != (size, size)
        ):
            raise ValueError("source moments have incompatible shapes")
        # Selection and scaling have at most one nonzero per row. Avoid dense
        # matrix products in the common case, including repeated scan planning.
        nonzero = [np.flatnonzero(row) for row in p]
        if all(len(indices) <= 1 for indices in nonzero):
            indices = [int(row[0]) if len(row) else 0 for row in nonzero]
            weights = ctx.array([row[index] for row, index in zip(p, indices)])
            expected_mean = weights * original_mean[indices]
            expected_covariance = (
                np.outer(weights, weights)
                * original_covariance[np.ix_(indices, indices)]
            )
            mean_scale, covariance_scale = abs(expected_mean), abs(expected_covariance)
        else:
            expected_mean = p @ original_mean
            expected_covariance = p @ original_covariance @ p.T
            mean_scale = abs(p) @ abs(original_mean)
            covariance_scale = abs(p) @ abs(original_covariance) @ abs(p.T)
        # Match the input covariance's binary64 validation tolerance, with a
        # cancellation scale and no absolute floor that would hide tiny errors.
        for label, actual, expected, scale in (
            ("mean", mean, expected_mean, mean_scale),
            ("covariance", covariance, expected_covariance, covariance_scale),
        ):
            actual = ctx.array(actual)
            if any(
                abs(a - e) > ctx.number("1e-12") * max(abs(a), bound)
                for a, e, bound in zip(actual.flat, expected.flat, scale.flat)
            ):
                raise ValueError(
                    f"data {label} disagrees with its source projection; create a new source for independent data"
                )


@serializable
@dataclass(frozen=True)
class SourceData:
    times: tuple
    mean: np.ndarray
    covariance: np.ndarray
    id: str = field(default_factory=lambda: uuid4().hex)
    observable: str | None = None
    tags: tuple = ()
    resamples: object = None
    info: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "times", tuple(self.times))
        object.__setattr__(self, "tags", tuple(self.tags))
        object.__setattr__(self, "info", FrozenMapping(self.info))
        object.__setattr__(self, "mean", readonly(self.mean))
        object.__setattr__(self, "covariance", readonly(self.covariance))
        if self.resamples is not None:
            object.__setattr__(self, "resamples", readonly(self.resamples))


@serializable
@dataclass(frozen=True)
class CorrelatorData:
    times: tuple
    mean: np.ndarray
    covariance: np.ndarray
    energy_unit: str | None = None
    source: SourceData | None = None
    projection: object = None

    def __post_init__(self):
        from .grid import validate_axis

        times = validate_axis("data times", self.times)
        mean, covariance = readonly(self.mean), readonly(self.covariance)
        n = len(times)
        if mean.shape != (n,) or covariance.shape != (n, n):
            raise ValueError(
                "mean and covariance must have shapes (n_times,) and (n_times,n_times)"
            )
        with mp.workdps(50):
            for x in (*times, *mean.flat, *covariance.flat):
                if not mp.isfinite(
                    mp.mpf(x.item() if isinstance(x, np.generic) else x)
                ):
                    raise ValueError("data contain non-finite values")
            scale = max(map(lambda x: abs(mp.mpf(x)), covariance.flat))
            if scale:
                normalized = np.array(
                    [[float(mp.mpf(x) / scale) for x in row] for row in covariance]
                )
                if not np.allclose(normalized, normalized.T, rtol=1e-12, atol=1e-14):
                    raise ValueError("covariance must be symmetric")
                if np.linalg.eigvalsh(normalized).min() < -1e-12:
                    raise ValueError("covariance must be positive semidefinite")
        source = self.source or SourceData(times, mean, covariance)
        projection = readonly(np.eye(n) if self.projection is None else self.projection)
        if projection.shape != (n, len(source.mean)):
            raise ValueError("source projection has incompatible shape")
        if self.source is not None or self.projection is not None:
            _validate_projection(source, projection, mean, covariance)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "projection", projection)

    @classmethod
    def from_arrays(cls, *, times, mean, covariance=None, energy_unit=None):
        if covariance is None:
            covariance = np.zeros((len(mean), len(mean)))
        return cls(times, mean, covariance, energy_unit)

    def select_times(self, times):
        from .grid import coordinate_equal

        requested = tuple(times)
        indices = []
        for t in requested:
            found = [i for i, old in enumerate(self.times) if coordinate_equal(t, old)]
            if len(found) != 1:
                raise ValueError(
                    f"time {t!r} is not present exactly once in the source data"
                )
            indices.append(found[0])
        return CorrelatorData(
            requested,
            self.mean[indices],
            self.covariance[np.ix_(indices, indices)],
            self.energy_unit,
            self.source,
            self.projection[indices],
        )

    def scaled(self, factor, *, dps=None):
        """Scale at inferred operand precision, or at an explicit decimal precision."""

        def digits(value):
            if hasattr(value, "_mpf_"):
                return int(value._mpf_[3] * 0.30103) + 2
            return len(str(value))

        # The factor enters covariance quadratically. Include every operand;
        # decimal strings and mpf mantissas may exceed the ambient precision.
        precision = (
            max(
                80,
                mp.mp.dps,
                *(
                    digits(x)
                    for x in (
                        *self.mean.flat,
                        *self.covariance.flat,
                        *self.projection.flat,
                    )
                ),
            )
            + 2 * digits(factor)
            + 20
        )
        if dps is not None:
            if not isinstance(dps, int) or dps < 25:
                raise ValueError("scaling dps must be an integer >= 25")
            precision = dps
        with mp.workdps(precision):
            f = mp.mpf(factor)
            return CorrelatorData(
                self.times,
                np.array([mp.mpf(x) * f for x in self.mean], dtype=object),
                np.array(
                    [[mp.mpf(x) * f * f for x in row] for row in self.covariance],
                    dtype=object,
                ),
                self.energy_unit,
                self.source,
                np.array(
                    [[mp.mpf(x) * f for x in row] for row in self.projection],
                    dtype=object,
                ),
            )
