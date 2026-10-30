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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from .precision import source_matrix, source_vector


@dataclass(frozen=True)
class CorrelatorData:
    """Euclidean correlator central values and covariance matrix.

    Parameters
    ----------
    times:
        One-dimensional array of available Euclidean times.
    mean:
        Central values C(t) at the same times.
    covariance:
        Covariance matrix corresponding to ``mean``.
    T:
        Optional temporal lattice extent. Required for periodic basis checks.
    """

    times: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    T: float | None = None
    _mean_source: tuple[Any, ...] = field(init=False, repr=False, compare=False)
    _covariance_source: tuple[tuple[Any, ...], ...] = field(
        init=False, repr=False, compare=False
    )
    _T_source: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mean_source = source_vector(self.mean)
        covariance_source = source_matrix(self.covariance)
        T_source = self.T
        T = None if self.T is None else float(self.T)
        times = np.asarray(self.times, dtype=float)
        mean = np.asarray(mean_source, dtype=float)
        cov = np.asarray(covariance_source, dtype=float)

        if times.ndim != 1:
            raise ValueError("times must be one-dimensional")
        if mean.ndim != 1:
            raise ValueError("mean must be one-dimensional")
        if len(times) != len(mean):
            raise ValueError("times and mean must have the same length")
        if cov.shape != (len(mean), len(mean)):
            raise ValueError("covariance must have shape (N, N)")
        if not np.all(np.isfinite(times)):
            raise ValueError("times contain non-finite values")
        if not np.all(np.isfinite(mean)):
            raise ValueError("mean contains non-finite values")
        if not np.all(np.isfinite(cov)):
            raise ValueError("covariance contains non-finite values")
        if len(np.unique(times)) != len(times):
            raise ValueError("times must be unique")

        object.__setattr__(self, "times", times)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", cov)
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "_mean_source", mean_source)
        object.__setattr__(self, "_covariance_source", covariance_source)
        object.__setattr__(self, "_T_source", T_source)

    @classmethod
    def from_arrays(
        cls,
        mean: Iterable[float],
        covariance: Iterable[Iterable[float]],
        times: Iterable[float] | None = None,
        T: float | None = None,
    ) -> CorrelatorData:
        mean_values = source_vector(mean)
        if times is None:
            times_arr = np.arange(len(mean_values), dtype=float)
        else:
            times_arr = np.asarray(times, dtype=float)
        return cls(times=times_arr, mean=mean_values, covariance=covariance, T=T)

    def select_times(
        self,
        t_min: float | None = None,
        t_max: float | None = None,
        stride: int = 1,
        times: Iterable[float] | None = None,
    ) -> np.ndarray:
        """Return basis/target times either explicitly or from a cut/stride selection.

        If ``times`` is supplied, it is validated against available correlator times.
        Otherwise the available times are filtered by ``t_min <= t <= t_max`` and
        then thinned by ``stride``.
        """
        if stride < 1:
            raise ValueError("stride must be >= 1")

        if times is not None:
            selected = np.asarray(list(times), dtype=float)
            self.indices_for_times(selected)
            return selected

        mask = np.ones(len(self.times), dtype=bool)
        if t_min is not None:
            mask &= self.times >= t_min
        if t_max is not None:
            mask &= self.times <= t_max
        selected = self.times[mask][::stride]
        if len(selected) == 0:
            raise ValueError("time selection is empty")
        return selected

    def indices_for_times(self, selected_times: Iterable[float]) -> np.ndarray:
        selected = np.asarray(list(selected_times), dtype=float)
        indices = []
        for t in selected:
            matches = np.where(np.isclose(self.times, t, rtol=0.0, atol=1e-12))[0]
            if len(matches) != 1:
                raise ValueError(
                    f"selected time {t} is not present exactly once in correlator data"
                )
            indices.append(int(matches[0]))
        return np.asarray(indices, dtype=int)

    def sliced(
        self, selected_times: Iterable[float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return selected times, mean vector, and covariance submatrix."""
        selected = np.asarray(list(selected_times), dtype=float)
        idx = self.indices_for_times(selected)
        return selected, self.mean[idx], self.covariance[np.ix_(idx, idx)]

    def sliced_sources(
        self, selected_times: Iterable[float]
    ) -> tuple[tuple[Any, ...], tuple[tuple[Any, ...], ...]]:
        """Return original-precision mean and covariance values for selected times."""
        idx = self.indices_for_times(selected_times)
        mean = tuple(self._mean_source[int(i)] for i in idx)
        covariance = tuple(
            tuple(self._covariance_source[int(i)][int(j)] for j in idx) for i in idx
        )
        return mean, covariance
