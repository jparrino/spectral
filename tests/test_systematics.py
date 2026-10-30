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

import numpy as np
import pytest
import spectral as sp
from scipy.integrate import trapezoid
from spectral.algorithms import hlt


def test_nominal_is_included_and_report_is_separate():
    basis = sp.Basis(sp.kernels.exponential(), (1, 2))
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1, 0.5])
    result = sp.Reconstructor(basis, hlt.Solver(".1")).reconstruct(
        data, target=basis.at_time(3)
    )
    nominal = sp.models.TabulatedSpectralDensity((0, 1, 2), (1, 1, 1), outside="zero")
    zero = sp.models.TabulatedSpectralDensity((0, 1, 2), (0, 0, 0), outside="zero")
    report = sp.systematics.estimate(
        result,
        reference_kernel=basis.at_time(3),
        density=nominal,
        deformations=sp.systematics.DeformationSet((zero,), ("zero",), {}),
    )
    assert report.max_abs_bias == abs(report.nominal_bias)
    assert report.model_at_maximum == "nominal"
    assert not hasattr(result, "systematics")
    assert report.result_ids == (result.id,)


def test_seeded_deformations_preserve_specified_integral():
    model = sp.models.TabulatedSpectralDensity((0, 1, 2), (1, 2, 1))
    energies = np.linspace(0, 2, 51)
    kwargs = dict(
        density=model,
        energies=energies,
        count=5,
        relative_sigma=0.1,
        correlation_length=0.2,
        seed=42,
        preserve_integral=True,
    )
    a, b = sp.systematics.gaussian_deformations(
        **kwargs
    ), sp.systematics.gaussian_deformations(**kwargs)
    for left, right in zip(a.models, b.models):
        np.testing.assert_array_equal(left.factors, right.factors)
        rho = model(energies)
        assert trapezoid(
            rho * np.asarray(left.factors, float), energies
        ) == pytest.approx(trapezoid(rho, energies), rel=1e-13)
        assert min(left.factors) > 0


def test_reference_convention_must_match():
    basis = sp.Basis(sp.kernels.exponential(), (1, 2), p=2)
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1, 0.5])
    result = sp.Reconstructor(basis, hlt.Solver(".1")).reconstruct(
        data, target=basis.at_time(3)
    )
    with pytest.raises(ValueError, match="spectral powers"):
        sp.systematics.estimate(
            result,
            reference_kernel=sp.kernels.exponential().at_time(3),
            density=sp.models.TabulatedSpectralDensity((0, 1), (1, 1)),
        )
