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

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from uuid import uuid4
from itertools import product
import numpy as np

from .._registry import serializable
from .data import readonly, same_source_values
from .grid import coordinate_equal, validate_axis
from .kernel import Kernel, PrecisionKernel
from ._state import FrozenMapping, state_digest


@serializable
@dataclass(frozen=True)
class ApproximatedKernel(Kernel):
    basis: object
    coefficients: object
    numerics: object = None

    @property
    def spectral_power(self):
        return self.basis.p

    @property
    def energy_unit(self):
        return self.basis.energy_unit

    def scalar(self, omega, ctx):
        return ctx.dot(ctx.array(self.coefficients), self.basis.values(omega, ctx))


@serializable
@dataclass(frozen=True)
class Result:
    value: object
    variance: object
    coefficients: np.ndarray
    basis: object
    target_kernel: object
    data: object
    numerics: object
    diagnostics: Mapping = field(default_factory=dict)
    coordinates: Mapping = field(default_factory=dict)
    metadata: Mapping = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid4().hex)
    algorithm: object = None
    _identity_state: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if len(self.basis.times) != len(self.data.times) or any(
            not coordinate_equal(a, b)
            for a, b in zip(self.basis.times, self.data.times)
        ):
            raise ValueError("result basis and data times must agree in order")
        with self.numerics.context() as ctx:
            actual = ctx.array((self.value, self.variance))
            if actual.shape != (2,):
                raise ValueError("result value and variance must be scalars")
            q = ctx.array(self.coefficients)
            if q.shape != (len(self.basis.times),):
                raise ValueError("result coefficients must match the basis size")
            if actual[1] < 0:
                raise ValueError("negative result variance")
            mean, covariance = (
                ctx.array(self.data.mean),
                ctx.array(self.data.covariance),
            )
            expected = (ctx.dot(q, mean), ctx.dot(q, covariance @ q))
            scales = (
                ctx.sum(abs(a * b) for a, b in zip(q, mean)),
                ctx.sum(
                    abs(q[i] * covariance[i, j] * q[j])
                    for i in range(len(q))
                    for j in range(len(q))
                ),
            )
            ctx.require_finite((*expected, *scales), "result contraction")
            tolerance = 1000 * len(q) * ctx.eps
            for label, a, e, scale in zip(
                ("value", "variance"), actual, expected, scales
            ):
                if abs(a - e) > tolerance * max(abs(a), abs(e), scale):
                    raise ValueError(
                        f"result {label} disagrees with coefficients and data"
                    )
            object.__setattr__(self, "value", actual[0])
            object.__setattr__(self, "variance", actual[1])
        object.__setattr__(self, "coefficients", readonly(self.coefficients))
        target = self.target_kernel
        if isinstance(target, PrecisionKernel):
            target = target.kernel
        object.__setattr__(
            self, "target_kernel", PrecisionKernel(target, self.numerics)
        )
        for name in ("coordinates", "metadata", "diagnostics"):
            object.__setattr__(self, name, FrozenMapping(getattr(self, name)))
        for value in self.coordinates.values():
            validate_axis("result coordinate", (value,))
        source = self.data.source
        identity = state_digest(
            (
                self.value,
                self.variance,
                self.coefficients,
                self.basis,
                self.target_kernel,
                self.numerics,
                self.algorithm,
                self.data.times,
                self.data.mean,
                self.data.covariance,
                self.data.projection,
                self.data.energy_unit,
                source.id,
                source.times,
                source.mean,
                source.covariance,
                source.resamples,
                source.tags,
                source.observable,
            )
        )
        if self._identity_state is not None and self._identity_state != identity:
            object.__setattr__(self, "id", uuid4().hex)
        object.__setattr__(self, "_identity_state", identity)

    def rebind_data(self, data):
        """Retain identity when resuming against an equivalent current source."""
        if not same_source_values(self.data.source, data.source) or any(
            not np.array_equal(getattr(self.data, name), getattr(data, name))
            for name in ("times", "mean", "covariance", "projection", "energy_unit")
        ):
            raise ValueError("rebinding requires equivalent data and source contents")
        return replace(self, data=data, _identity_state=None)

    @property
    def statistical_error(self):
        with self.numerics.context() as ctx:
            return ctx.sqrt(ctx.number(self.variance))

    @property
    def input_times(self):
        return self.basis.times

    @property
    def approximated_kernel(self):
        return ApproximatedKernel(self.basis, self.coefficients, self.numerics)

    def _source_weights(self, ctx):
        coefficients, projection = (
            ctx.array(self.coefficients),
            ctx.array(self.data.projection),
        )
        return ctx.array([ctx.dot(coefficients, column) for column in projection.T])

    @property
    def source_weights(self):
        with self.numerics.context() as ctx:
            return self._source_weights(ctx)


@serializable
@dataclass(frozen=True)
class Failure:
    coordinates: Mapping
    error: str

    def __post_init__(self):
        object.__setattr__(self, "coordinates", FrozenMapping(self.coordinates))
        for value in self.coordinates.values():
            validate_axis("failure coordinate", (value,))


@serializable
@dataclass(frozen=True)
class ResultCollection:
    axes: Mapping
    results: tuple

    def __post_init__(self):
        axes = {k: validate_axis(k, v) for k, v in self.axes.items()}
        if int(np.prod([len(v) for v in axes.values()])) != len(self.results):
            raise ValueError("result count does not match collection axes")
        for result, values in zip(self.results, product(*axes.values())):
            if not isinstance(result, (Result, Failure)):
                raise TypeError("collections require Result or Failure objects")
            if any(
                key not in result.coordinates
                or not coordinate_equal(result.coordinates[key], value)
                for key, value in zip(axes, values)
            ):
                raise ValueError(
                    "result coordinates do not match collection axis order"
                )
        # Extra coordinates are retained by selected subsets, but must be fixed.
        extras = [set(r.coordinates) - set(axes) for r in self.results]
        if any(extra != extras[0] for extra in extras) or any(
            not coordinate_equal(r.coordinates[key], self.results[0].coordinates[key])
            for r in self.results
            for key in extras[0]
        ):
            raise ValueError("coordinates outside collection axes must be constant")
        object.__setattr__(self, "axes", FrozenMapping(axes))
        object.__setattr__(self, "results", tuple(self.results))

    @property
    def shape(self):
        return tuple(len(v) for v in self.axes.values())

    @property
    def coordinates(self):
        return tuple(dict(zip(self.axes, v)) for v in product(*self.axes.values()))

    @property
    def failures(self):
        return tuple(r for r in self.results if isinstance(r, Failure))

    def _values(self, attribute):
        return np.array(
            [
                getattr(r, attribute) if isinstance(r, Result) else np.nan
                for r in self.results
            ],
            dtype=object,
        ).reshape(self.shape)

    @property
    def values(self):
        return self._values("value")

    @property
    def statistical_errors(self):
        return self._values("statistical_error")

    @property
    def coefficients(self):
        if self.failures:
            raise ValueError(
                "coefficient array is unavailable for a collection with failed points"
            )
        lengths = {len(r.coefficients) for r in self.results}
        if len(lengths) != 1:
            raise ValueError(
                "input bases have different sizes; select an ensemble first"
            )
        return np.array([r.coefficients for r in self.results], dtype=object).reshape(
            self.shape + (lengths.pop(),)
        )

    @property
    def lambda_values(self):
        return self.axes["lambda"]

    @property
    def target_times(self):
        return self.axes["target_time"]

    def select(self, **coordinates):
        if "lambda_" in coordinates:
            coordinates["lambda"] = coordinates.pop("lambda_")
        if set(coordinates) - set(self.axes):
            raise KeyError(f"unknown axes: {set(coordinates) - set(self.axes)}")
        selected = tuple(
            r
            for r, c in zip(self.results, self.coordinates)
            if all(coordinate_equal(c[k], v) for k, v in coordinates.items())
        )
        if not selected:
            raise KeyError(f"no result at {coordinates}")
        axes = {k: v for k, v in self.axes.items() if k not in coordinates}
        return ResultCollection(axes, selected) if axes else selected[0]

    def at_lambda(self, value):
        return self.select(lambda_=value)

    def at_time(self, value):
        return self.select(target_time=value)

    def approximated_kernel(self, energies):
        if self.failures:
            raise ValueError("kernel evaluation requires successful results")
        values = [
            r.approximated_kernel(energies, numerics=r.numerics) for r in self.results
        ]
        return np.array(values, dtype=object).reshape(self.shape + np.shape(energies))

    def target_kernel(self, energies):
        values = [r.target_kernel(energies, numerics=r.numerics) for r in self.results]
        return np.array(values, dtype=object).reshape(self.shape + np.shape(energies))

    @property
    def statistical_covariance(self):
        """Full covariance in flattened, row-major coordinate order."""
        if self.failures:
            raise ValueError("covariance requires successful results")
        source = self.results[0].data.source
        for result in self.results:
            if result.data.source.id != source.id:
                raise ValueError(
                    "cross-source covariance requires aligned resamples; use joint JKS export"
                )
            if not same_source_values(source, result.data.source):
                raise ValueError("inconsistent data for a shared source identity")
        # One context for the entire matrix keeps diagonal and off-diagonal
        # entries coherent. Do not pre-round projections at each result's dps.
        policy = max(
            (result.numerics for result in self.results),
            key=lambda p: (p.backend == "mpmath", p.dps),
        )
        with policy.context() as ctx:
            weights = ctx.array(
                [result._source_weights(ctx) for result in self.results]
            )
            covariance_weights = ctx.array(source.covariance) @ weights.T
            matrix = ctx.array(
                [
                    [ctx.dot(left, right) for right in covariance_weights.T]
                    for left in weights
                ]
            )
        return matrix
