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
import numpy as np
import mpmath as mp

from ..._registry import serializable
from ...core.grid import validate_axis
from . import preparation, linear


@serializable
@dataclass(frozen=True)
class Solver:
    lambda_: object = "1e-6"
    regulator: str = "identity"
    constraints: tuple = ()
    covariance_norm: object = None

    def __post_init__(self):
        if np.ndim(self.lambda_) not in (0, 1):
            raise ValueError("lambda_ must be a scalar or a one-dimensional array")
        values = (self.lambda_,) if np.ndim(self.lambda_) == 0 else tuple(self.lambda_)
        values = tuple(
            x.item() if isinstance(x, (np.generic, np.ndarray)) else x for x in values
        )
        validate_axis("lambda", values)
        with mp.workdps(80):
            if any(
                not mp.isfinite(mp.mpf(x)) or not 0 <= mp.mpf(x) < 1 for x in values
            ):
                raise ValueError("HLT lambda values must satisfy 0 <= lambda < 1")
        if self.regulator not in {"identity", "covariance"}:
            raise ValueError("regulator must be identity or covariance")
        if np.ndim(self.lambda_) != 0:
            object.__setattr__(self, "lambda_", values)
        else:
            object.__setattr__(self, "lambda_", values[0])
        object.__setattr__(self, "constraints", tuple(self.constraints))

    @property
    def parameter_axes(self):
        return {} if np.ndim(self.lambda_) == 0 else {"lambda": tuple(self.lambda_)}

    @property
    def parameter_points(self):
        values = (self.lambda_,) if np.ndim(self.lambda_) == 0 else self.lambda_
        return tuple({"lambda": value} for value in values)

    def prepare(self, basis, data, numerics):
        return preparation.prepare(basis, data, self, numerics)

    def preparation_key(self, basis, data, numerics):
        # Local task identity only; native archives never use pickle.
        import pickle

        covariance = data.covariance if self.regulator == "covariance" else None
        return pickle.dumps(
            (
                basis,
                self.regulator,
                self.covariance_norm,
                self.constraints,
                numerics,
                covariance,
            ),
            protocol=5,
        )

    def prepare_target(self, problem, target, numerics):
        return preparation.prepare_target(problem, target, numerics)

    def solve_batch(self, problem, targets, parameters, numerics):
        return linear.solve_batch(problem, targets, parameters["lambda"], numerics)
