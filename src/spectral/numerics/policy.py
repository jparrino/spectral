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

from contextlib import contextmanager
from dataclasses import dataclass
import math
import warnings
from numbers import Integral

import mpmath as mp
import numpy as np
from scipy.integrate import quad, IntegrationWarning
from scipy.special import gammaincc, gamma

from .._registry import serializable


@serializable
@dataclass(frozen=True)
class Numerics:
    backend: str = "numpy"
    dps: int = 80
    epsabs: float = 1e-12
    epsrel: float = 1e-11
    maxdegree: int = 9

    def __post_init__(self):
        if self.backend not in {"numpy", "mpmath"}:
            raise ValueError("backend must be numpy or mpmath")
        if (
            not isinstance(self.dps, Integral)
            or not isinstance(self.maxdegree, Integral)
            or self.dps < 25
            or self.maxdegree < 3
        ):
            raise ValueError(
                "dps and maxdegree must be integers with dps >= 25 and maxdegree >= 3"
            )
        if any(not math.isfinite(x) or x <= 0 for x in (self.epsabs, self.epsrel)):
            raise ValueError("quadrature tolerances must be positive and finite")

    @contextmanager
    def context(self):
        # The supported executors are serial and processes; no shared mp threads.
        with mp.workdps(self.dps):
            yield Backend(self)


def numpy(**kwargs):
    return Numerics(backend="numpy", **kwargs)


def mpmath(dps=80, **kwargs):
    return Numerics(backend="mpmath", dps=dps, **kwargs)


class Backend:
    def __init__(self, policy):
        self._prepared = {}
        self.policy = policy
        self.is_mp = policy.backend == "mpmath"
        self.math = mp if self.is_mp else math
        self.pi = self.math.pi
        self.eps = mp.eps if self.is_mp else np.finfo(float).eps
        self.exp = self.math.exp
        self.sqrt = self.math.sqrt
        self.tanh = self.math.tanh
        self.log = self.math.log
        self.erf = self.math.erf
        self.inf = mp.inf if self.is_mp else math.inf

    def cached(self, owner, prepare):
        """Prepare immutable object data once within this precision context."""
        key = id(owner)
        if key not in self._prepared:
            self._prepared[key] = (owner, prepare())
        return self._prepared[key][1]

    def number(self, x):
        if isinstance(x, np.generic):
            x = x.item()
        return mp.mpf(x) if self.is_mp else float(x)

    def array(self, values):
        a = np.asarray(values, dtype=object)
        result = np.array(
            [self.number(x) for x in a.flat], dtype=object if self.is_mp else float
        ).reshape(a.shape)
        self.require_finite(result, "converted array")
        return result

    def require_finite(self, values, label="numerical output"):
        if any(
            not self.math.isfinite(x) for x in np.asarray(values, dtype=object).flat
        ):
            raise ArithmeticError(f"non-finite {label}")

    def sum(self, values):
        return mp.fsum(values) if self.is_mp else math.fsum(values)

    def dot(self, a, b):
        return self.sum(x * y for x, y in zip(a, b, strict=True))

    def exp_integral(self, power, rate, lower):
        power, rate, lower = map(self.number, (power, rate, lower))
        if rate <= 0 or lower < 0:
            raise ValueError(
                "exponential integral requires positive rate and nonnegative lower bound"
            )
        if lower == 0 and power <= -1:
            raise ValueError("integral diverges at omega_0=0")
        a = power + 1
        if self.is_mp:
            return mp.gammainc(a, rate * lower, mp.inf) / rate**a
        if a > 0:
            try:
                g, tail, denominator = (
                    float(gamma(a)),
                    float(gammaincc(a, rate * lower)),
                    rate**a,
                )
                if all(math.isfinite(v) and v > 0 for v in (g, tail, denominator)):
                    value = g * tail / denominator
                    if math.isfinite(value) and value > 0:
                        return value
            except OverflowError:
                pass
        # Evaluate the combination before converting to binary64 to avoid
        # intermediate under/overflow. This also handles nonpositive gamma
        # orders, for which unscaled quadrature can miss a narrow endpoint peak.
        with mp.workdps(max(50, self.policy.dps)):
            r, lo, exponent = mp.mpf(rate), mp.mpf(lower), mp.mpf(power) + 1
            value = float(mp.gammainc(exponent, r * lo, mp.inf) / r**exponent)
        self.require_finite(value, "exponential integral")
        return value

    def integrate(self, function, lower=0, upper=None, points=()):
        lo = self.number(lower)
        hi = self.inf if upper is None else self.number(upper)
        if not lo < hi:
            raise ValueError("integration bounds must be increasing")
        knots = (
            [lo]
            + sorted(set(self.number(x) for x in points if lo < self.number(x) < hi))
            + [hi]
        )
        if self.is_mp:
            value, error = mp.quad(
                function,
                knots,
                method="tanh-sinh",
                maxdegree=self.policy.maxdegree,
                error=True,
            )
            if not mp.isfinite(value):
                raise ArithmeticError("non-finite quadrature result")
            # mp.quad error estimates are heuristic; convergence is also tested independently.
            tolerance = mp.power(10, -(self.policy.dps - 10)) * max(1, abs(value))
            if error > tolerance:
                raise ArithmeticError(
                    f"quadrature did not converge: estimated error {error}; split the domain or increase maxdegree"
                )
            return value
        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            value = self.sum(
                quad(
                    function,
                    float(a),
                    float(b),
                    epsabs=self.policy.epsabs,
                    epsrel=self.policy.epsrel,
                    limit=300,
                )[0]
                for a, b in zip(knots[:-1], knots[1:])
            )
        if not math.isfinite(value):
            raise ArithmeticError("non-finite quadrature result")
        return value
