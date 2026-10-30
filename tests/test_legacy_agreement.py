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

"""Compare the refactored numerics with the user-supplied implementation."""

from pathlib import Path
import sys
import numpy as np
import pytest
import spectral as sp
from spectral.algorithms import hlt as new_hlt

sys.path.insert(0, str(Path(__file__).parent / "_reference"))
import hlt as legacy


@pytest.mark.parametrize("regulator", ["identity", "covariance"])
@pytest.mark.parametrize("p", [0, 2])
def test_smeared_hlt_coefficients_agree(regulator, p):
    times = (1, 2, 3, 4)
    mean = np.exp(-0.5 * np.array(times))
    covariance = np.diag((0.01 * mean) ** 2)
    old_basis = legacy.HLTBasis(times, p=p, omega0=0.3)
    old_data = legacy.CorrelatorData(times, mean, covariance)
    old_target = legacy.HighEnergyTarget(t0=2, am_tau=0.7, delta=0.15, p=p)
    old = legacy.HLTReconstructor(
        old_data,
        old_basis,
        lambda_hlt=1e-4,
        regulator=regulator,
        covariance_norm=1,
        area_constraint=True,
        strict=False,
    ).reconstruct(old_target)
    basis = sp.Basis(sp.kernels.exponential(), times, p=p, omega_0=0.3)
    data = sp.CorrelatorData.from_arrays(times=times, mean=mean, covariance=covariance)
    target = sp.targets.smooth_high_energy(
        kernel=basis.at_time(2), cutoff=0.7, width=0.15
    )
    result = sp.Reconstructor(
        basis,
        new_hlt.Solver(
            1e-4,
            regulator=regulator,
            covariance_norm=1,
            constraints=[new_hlt.constraints.area()],
        ),
    ).reconstruct(data, target=target)
    np.testing.assert_allclose(
        np.asarray(result.coefficients, float),
        np.asarray(old.q, float),
        rtol=1e-7,
        atol=1e-9,
    )
    assert float(result.value) == pytest.approx(float(old.estimate), rel=1e-8)


def test_hvp_full_formula_matches_reference():
    old = legacy.IntegratedHVPTarget.from_sharp_high_energy(
        am_tau=0, p=2, am_mu=0.105, alpha=0.007
    )
    new = sp.targets.hvp(mass=0.105, p=2, alpha=0.007)
    energies = np.array([0.01, 0.15, 0.21, 0.4, 0.7, 2.0, 20.0])
    np.testing.assert_allclose(new(energies), old.eval(energies), rtol=1e-13)
    policy = sp.numerics.mpmath(35)
    for energy in (".15", ".7", "20"):
        a = new(energy, numerics=policy)
        b = old.eval_mpmath(energy, mp_dps=35)
        assert abs(float((a - b) / b)) < 1e-28


def test_intermediate_window_matches_reference():
    old = legacy.IntegratedHVPTarget.from_sharp_high_energy(
        am_tau=0,
        p=2,
        am_mu=0.105,
        alpha=0.007,
        euclidean_window=(2.0, 5.0),
        window_delta=0.5,
    )
    new = sp.targets.intermediate_window(
        mass=0.105, start=2.0, stop=5.0, width=0.5, p=2, alpha=0.007
    )
    # The old binary64 infinite quadrature can hit 0*inf in its special
    # functions. Use its stable mpmath implementation as the reference.
    energy = ".7"
    reference = old.eval_mpmath(energy, mp_dps=30)
    assert new(float(energy)) == pytest.approx(float(reference), rel=1e-9)
    high_precision = new(energy, numerics=sp.numerics.mpmath(30))
    assert abs(float((high_precision - reference) / reference)) < 1e-20
