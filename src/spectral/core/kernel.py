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

"""Energy kernels implement scalar(omega, context); target families bind a time."""

from dataclasses import dataclass
import inspect
from typing import Callable
import numpy as np

from .._registry import serializable
from ..numerics import Numerics


class Kernel:
    energy_unit = None
    spectral_power = None
    breakpoints = ()

    def scalar(self, omega, ctx):
        raise NotImplementedError

    def integration_points(self, ctx):
        """Quadrature breakpoints, calculated at the active precision if needed."""
        return self.breakpoints

    def validate_integral(self, ctx, lower, upper, power=1):
        """Reject known divergent integrals of this kernel raised to power."""

    def __call__(self, energies, *, numerics=None):
        with (
            numerics or getattr(self, "numerics", None) or Numerics()
        ).context() as ctx:
            values = np.asarray(energies, dtype=object)
            out = [self.scalar(ctx.number(x), ctx) for x in values.flat]
            if values.ndim == 0:
                return out[0]
            return np.array(out, dtype=object if ctx.is_mp else float).reshape(
                values.shape
            )

    def integral(self, ctx, lower=0, upper=None):
        self.validate_integral(ctx, lower, upper)
        return ctx.integrate(
            lambda w: self.scalar(w, ctx), lower, upper, self.integration_points(ctx)
        )

    def basis_integrals(self, basis, ctx):
        """Overlaps with the basis and squared norm; targets can optimize these."""
        self.validate_integral(ctx, basis.omega_0, None, power=2)
        b = [
            ctx.integrate(
                lambda w, t=t: basis.at_time(t).scalar(w, ctx) * self.scalar(w, ctx),
                basis.omega_0,
                points=self.integration_points(ctx),
            )
            for t in basis.times
        ]
        norm = ctx.integrate(
            lambda w: self.scalar(w, ctx) ** 2,
            basis.omega_0,
            points=self.integration_points(ctx),
        )
        return b, norm


@dataclass(frozen=True)
class CallableKernel(Kernel):
    """Custom function f(omega, ctx); use ctx math to support arbitrary precision.

    Top-level functions support process execution. Native archives deliberately
    require a registered Kernel class instead of serializing executable functions.
    """

    function: Callable
    breakpoints: tuple = ()
    energy_unit: str | None = None
    spectral_power: object = None

    def scalar(self, omega, ctx):
        return ctx.number(self.function(omega, ctx))


@dataclass(frozen=True)
class _FunctionKernel(Kernel):
    function: Callable
    takes_time: bool
    takes_context: bool
    time: object = None
    breakpoints: tuple = ()
    energy_unit: str | None = None
    spectral_power: object = None

    @property
    def is_time_dependent(self):
        return self.takes_time and self.time is None

    def at_time(self, time):
        from dataclasses import replace
        from .grid import validate_axis

        if not self.is_time_dependent:
            raise ValueError("this function target already has a fixed time")
        validate_axis("target_time", (time,))
        return replace(self, time=time)

    def scalar(self, omega, ctx):
        if self.is_time_dependent:
            raise ValueError("target_times are required for this function kernel")
        args = (omega, ctx.number(self.time)) if self.takes_time else (omega,)
        kwargs = {"ctx": ctx} if self.takes_context else {}
        value = ctx.number(self.function(*args, **kwargs))
        ctx.require_finite(value, "function kernel value")
        return value


def from_function(function, *, breakpoints=(), energy_unit=None, spectral_power=None):
    """Adapt f(omega) or f(omega, t), optionally accepting a keyword ctx.

    The function returns the complete target weight, including any omega**p.
    Use ctx math to preserve the selected numerical backend's precision. Ordinary
    math/NumPy functions are suitable for binary64 calculations. Importable
    functions support process execution; executable functions are not archived.
    """
    message = (
        "function targets require f(omega) or f(omega, t), optionally with "
        "a keyword ctx; bind other required parameters with functools.partial"
    )
    if not callable(function):
        raise TypeError(message)
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError) as exc:
        raise TypeError(message) from exc
    positional = [
        p
        for p in signature.parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.name != "ctx"
    ]
    if not 1 <= len(positional) <= 2 or any(
        p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        for p in signature.parameters.values()
    ):
        raise TypeError(message)
    takes_time = len(positional) == 2
    takes_context = "ctx" in signature.parameters
    args = (object(), object()) if takes_time else (object(),)
    kwargs = {"ctx": object()} if takes_context else {}
    try:
        signature.bind(*args, **kwargs)
    except TypeError as exc:
        raise TypeError(message) from exc
    from .grid import validate_axis

    points = tuple(breakpoints)
    if points:
        validate_axis("breakpoints", points)
    return _FunctionKernel(
        function,
        takes_time,
        takes_context,
        breakpoints=points,
        energy_unit=energy_unit,
        spectral_power=spectral_power,
    )


def as_kernel(target):
    """Accept plain function targets without changing existing kernel protocols."""
    if (
        callable(target)
        and not hasattr(target, "scalar")
        and not hasattr(target, "at_time")
    ):
        return from_function(target)
    return target


@serializable
@dataclass(frozen=True)
class PrecisionKernel(Kernel):
    kernel: object
    numerics: object

    @property
    def spectral_power(self):
        return self.kernel.spectral_power

    @property
    def energy_unit(self):
        return self.kernel.energy_unit

    @property
    def breakpoints(self):
        return self.kernel.breakpoints

    def scalar(self, omega, ctx):
        return self.kernel.scalar(omega, ctx)

    def integration_points(self, ctx):
        return self.kernel.integration_points(ctx)

    def validate_integral(self, ctx, lower, upper, power=1):
        return self.kernel.validate_integral(ctx, lower, upper, power)

    def integral(self, ctx, lower=0, upper=None):
        return self.kernel.integral(ctx, lower, upper)

    def basis_integrals(self, basis, ctx):
        return self.kernel.basis_integrals(basis, ctx)


@serializable
@dataclass(frozen=True)
class LinearCombination(Kernel):
    kernels: tuple
    weights: tuple

    def __post_init__(self):
        if not self.kernels or len(self.kernels) != len(self.weights):
            raise ValueError("kernels and weights must have the same nonzero length")
        # Access both properties here to reject incompatible constituents eagerly.
        self.energy_unit
        self.spectral_power

    def _convention(self, name):
        from .grid import coordinate_equal

        values = [getattr(k, name, None) for k in self.kernels]
        known = [v for v in values if v is not None]
        equal = coordinate_equal if name == "spectral_power" else lambda a, b: a == b
        if any(not equal(known[0], v) for v in known[1:]):
            raise ValueError(f"constituent kernel {name} conventions disagree")
        return known[0] if known else None

    @property
    def energy_unit(self):
        return self._convention("energy_unit")

    @property
    def spectral_power(self):
        return self._convention("spectral_power")

    def validate_integral(self, ctx, lower, upper, power=1):
        # Conservative: divergent constituent cancellations require an explicit
        # kernel with its own integrability proof, not implicit cancellation.
        for kernel, weight in zip(self.kernels, self.weights):
            value = ctx.number(weight)
            ctx.require_finite(value, "kernel weight")
            if value != 0:
                kernel.validate_integral(ctx, lower, upper, power)

    @property
    def breakpoints(self):
        return tuple(x for k in self.kernels for x in k.breakpoints)

    def integration_points(self, ctx):
        return tuple(x for k in self.kernels for x in k.integration_points(ctx))

    def scalar(self, omega, ctx):
        return ctx.sum(
            ctx.number(w) * k.scalar(omega, ctx)
            for k, w in zip(self.kernels, self.weights)
        )


def kernel_cache_key(kernel):
    """Exact archive state for supported kernels, identity for custom kernels.

    Callers must keep kernels alive for the lifetime of their local cache.
    """
    import json
    from ..io.archive import Encoder

    try:
        return ("state", json.dumps(Encoder().encode(kernel), sort_keys=True))
    except TypeError:
        return ("identity", id(kernel))
