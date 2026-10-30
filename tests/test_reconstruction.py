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
import mpmath as mp
import pytest
import spectral as sp
from spectral.algorithms import hlt


def problem(policy=None, lambdas=("0", "1e-6"), regulator="identity", constraints=()):
    times = (1, 2, 3)
    basis = sp.Basis(sp.kernels.exponential(), times, p=2, omega_0="0.3")
    data = sp.CorrelatorData.from_arrays(
        times=times, mean=[1, 0.5, 0.2], covariance=np.eye(3) * 0.001
    )
    solver = hlt.Solver(
        lambdas, regulator=regulator, covariance_norm=1, constraints=constraints
    )
    return sp.Reconstructor(basis, solver, policy or sp.numerics.mpmath(60)), data


@pytest.mark.parametrize("regulator", ["identity", "covariance"])
@pytest.mark.parametrize(
    "constraints",
    [(), (hlt.constraints.area(),), (hlt.constraints.point(energy="0.4"),)],
)
def test_exact_input_time(regulator, constraints):
    recon, data = problem(regulator=regulator, constraints=constraints)
    result = recon.reconstruct(
        data, target=sp.targets.correlator(basis=recon.basis), target_times=[1, 2, 3]
    )
    exact = result.at_lambda("0")
    for i, item in enumerate(exact.results):
        assert abs(item.value - mp.mpf(data.mean[i])) < mp.mpf("1e-50")
        assert max(
            abs(x - (1 if j == i else 0)) for j, x in enumerate(item.coefficients)
        ) < mp.mpf("1e-50")
    assert result.values.shape == (2, 3)
    assert result.coefficients.shape == (2, 3, 3)


def test_numpy_matches_mpmath():
    a, data = problem(policy=sp.numerics.numpy())
    b, _ = problem()
    target = sp.targets.correlator(basis=a.basis)
    left = a.reconstruct(data, target=target, target_times=[4])
    right = b.reconstruct(data, target=target, target_times=[4])
    np.testing.assert_allclose(
        np.asarray(left.values, float), np.asarray(right.values, float), rtol=1e-10
    )


def test_covariance_requires_normalization():
    recon, data = problem()
    recon = sp.Reconstructor(recon.basis, hlt.Solver(".1", regulator="covariance"))
    with pytest.raises(sp.ReconstructionError, match="covariance_norm"):
        recon.reconstruct(data, target=recon.basis.at_time(2))


def test_constraints_reject_conflicts():
    recon, data = problem(
        constraints=(
            hlt.constraints.point(energy=1, value=0),
            hlt.constraints.point(energy=1, value=2),
        )
    )
    with pytest.raises(sp.ReconstructionError, match="conflicting"):
        recon.reconstruct(data, target=recon.basis.at_time(2))


def test_time_validation_and_singleton_axes():
    recon, data = problem(lambdas=["0"])
    with pytest.raises(ValueError, match="target_times"):
        recon.reconstruct(data, target=sp.targets.correlator(basis=recon.basis))
    with pytest.raises(ValueError, match="target_times"):
        recon.reconstruct(data, target=recon.basis.at_time(1), target_times=[1])
    result = recon.reconstruct(data, target=recon.basis.at_time(1))
    assert result.values.shape == (1,)
    with pytest.raises(ValueError, match="duplicate"):
        hlt.Solver(["1e-3", ".001"])


def test_correlations_are_preserved():
    recon, data = problem(lambdas=[0])
    result = recon.reconstruct(
        data, target=sp.targets.correlator(basis=recon.basis), target_times=[1, 2, 3]
    )
    np.testing.assert_allclose(
        np.asarray(result.statistical_covariance, float),
        np.asarray(data.covariance, float),
        atol=1e-50,
    )


def test_record_failure_keeps_coordinates():
    recon, data = problem(constraints=(hlt.constraints.point(energy=0, value=1),))
    result = recon.reconstruct(data, target=recon.basis.at_time(2), on_error="record")
    assert len(result.failures) == 2
    assert result.failures[0].coordinates == {"lambda": "0"}
