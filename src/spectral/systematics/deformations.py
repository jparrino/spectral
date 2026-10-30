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
import bisect
import math
from numbers import Integral
import numpy as np

from .._registry import serializable
from ..numerics import Numerics
from ..core._state import FrozenMapping
from ..core.spectral_density import SpectralDensity
from .density import for_basis


@serializable
@dataclass(frozen=True)
class DeformedDensity(SpectralDensity):
    density: SpectralDensity
    energies: tuple
    factors: tuple

    @property
    def domain(self):
        return self.density.domain

    @property
    def energy_unit(self):
        return self.density.energy_unit

    @property
    def breakpoints(self):
        return tuple(self.density.breakpoints) + tuple(self.energies)

    def scalar(self, omega, ctx):
        energies, factors = ctx.cached(
            self, lambda: (ctx.array(self.energies), ctx.array(self.factors))
        )
        # The deformation is explicitly confined to this energy interval.
        if omega < energies[0] or omega > energies[-1]:
            return self.density.scalar(omega, ctx)
        i = min(len(energies) - 2, max(0, bisect.bisect_right(energies, omega) - 1))
        a = (omega - energies[i]) / (energies[i + 1] - energies[i])
        factor = (1 - a) * factors[i] + a * factors[i + 1]
        return self.density.scalar(omega, ctx) * factor


@serializable
@dataclass(frozen=True)
class DeformationSet:
    models: tuple
    ids: tuple
    metadata: Mapping

    def __post_init__(self):
        object.__setattr__(self, "models", tuple(self.models))
        object.__setattr__(self, "ids", tuple(self.ids))
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata))
        if (
            len(self.models) != len(self.ids)
            or len(set(self.ids)) != len(self.ids)
            or "nominal" in self.ids
        ):
            raise ValueError(
                "deformation IDs must be unique, aligned, and different from nominal"
            )


def trapezoid_weights(energies, ctx):
    x = ctx.array(energies)
    if (
        x.ndim != 1
        or len(x) < 2
        or any(not ctx.math.isfinite(v) for v in x)
        or any(b <= a for a, b in zip(x[:-1], x[1:]))
    ):
        raise ValueError(
            "integration grid must be finite, one-dimensional and strictly increasing"
        )
    w = ctx.array(np.zeros(len(x)))
    w[0], w[-1] = (x[1] - x[0]) / 2, (x[-1] - x[-2]) / 2
    for i in range(1, len(x) - 1):
        w[i] = (x[i + 1] - x[i - 1]) / 2
    return w


def gaussian_deformations(
    *,
    density,
    energies,
    count,
    relative_sigma,
    correlation_length,
    seed,
    preserve_integral=False,
    normalization_weight=None,
    basis=None,
    require_positive=True,
    max_attempts=10000,
    numerics=None,
):
    """Generate reusable multiplicative Gaussian fields on an energy grid.

    Integral preservation uses the supplied normalization weight (or model
    density) with trapezoid quadrature on this grid. Outside the grid the
    original model is unchanged. RNG draws are binary64; model arithmetic
    and normalization use the selected numerical backend.
    By default preservation refers to the physical density. Pass ``basis`` to
    preserve the converted density integral, or supply normalization weights.
    """
    policy = numerics or Numerics()
    energies = tuple(energies)
    x = np.asarray(energies, float)
    if (
        not isinstance(count, Integral)
        or count < 1
        or not isinstance(max_attempts, Integral)
        or max_attempts < count
    ):
        raise ValueError(
            "count must be a positive integer and max_attempts must be an integer >= count"
        )
    if any(
        value is None or not math.isfinite(float(value)) or float(value) < 0
        for value in (relative_sigma, correlation_length)
    ):
        raise ValueError("sigma and correlation length must be finite and nonnegative")
    if (
        x.ndim != 1
        or len(x) < 2
        or not np.all(np.isfinite(x))
        or np.any(np.diff(x) <= 0)
    ):
        raise ValueError("deformation grid must be finite and strictly increasing")
    rng = np.random.default_rng(seed)
    length = float(correlation_length)
    smooth = (
        np.eye(len(x))
        if length == 0
        else np.exp(-(((x[:, None] - x[None, :]) / length) ** 2))
    )
    smooth /= np.sqrt(np.sum(smooth * smooth, axis=1))[:, None]
    models, attempts = [], 0
    with policy.context() as ctx:
        sigma = ctx.number(relative_sigma)
        if preserve_integral or normalization_weight is not None:
            normalization = for_basis(density, basis) if basis is not None else density
            weight = (
                ctx.array(normalization_weight)
                if normalization_weight is not None
                else ctx.array(
                    [normalization.scalar(ctx.number(e), ctx) for e in energies]
                )
            )
            if weight.shape != x.shape:
                raise ValueError("normalization_weight must match energies")
            if any(not ctx.math.isfinite(w) for w in weight):
                raise ValueError("normalization_weight must be finite")
            measure = trapezoid_weights(energies, ctx) * weight
            integral = ctx.sum(measure)
            if preserve_integral and (integral <= 0 or any(w < 0 for w in measure)):
                raise ValueError(
                    "integral preservation requires a nonnegative, nonzero normalization measure"
                )
        while len(models) < count and attempts < max_attempts:
            attempts += 1
            field = ctx.array(smooth @ rng.standard_normal(len(x)))
            if preserve_integral:
                field -= ctx.dot(measure, field) / integral
            factors = 1 + sigma * field
            if require_positive and any(f <= 0 for f in factors):
                continue
            models.append(DeformedDensity(density, energies, tuple(factors)))
    if len(models) != count:
        raise ValueError(
            f"only {len(models)} positive deformations accepted in {attempts} attempts"
        )
    return DeformationSet(
        tuple(models),
        tuple(f"deformation/{i}" for i in range(count)),
        dict(
            seed=seed,
            attempts=attempts,
            rejected=attempts - count,
            relative_sigma=relative_sigma,
            correlation_length=correlation_length,
            preserve_integral=preserve_integral,
            normalization_measure=(
                "explicit" if normalization_weight is not None else "model"
            ),
            normalization_spectral_power=(
                getattr(normalization, "spectral_power", 0)
                if preserve_integral and normalization_weight is None
                else None
            ),
            deformation_interval=(energies[0], energies[-1]),
        ),
    )
