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
from hlt import (
    CorrelatorData,
    HighEnergyTarget,
    HLTBasis,
    HLTReconstructor,
    IntegratedHVPTarget,
    NonIntegrableTargetAreaError,
    QuadOptions,
    SharpHighEnergyTarget,
    solve_hlt_unconstrained,
)
from mpmath import mp


def _toy_problem(backend="mpmath", dps=40, target_backend="numpy"):
    mean = ["1.0", "0.7", "0.5", "0.36"]
    covariance = [["0.01" if i == j else "0.0001" for j in range(4)] for i in range(4)]
    data = CorrelatorData.from_arrays(mean, covariance, times=[0, 1, 2, 3])
    basis = HLTBasis(times=[1, 2, 3], p="2", omega0="0.1")
    target = HighEnergyTarget(t0="1.5", am_tau="0.4", delta="0.1", p="2")
    reconstructor = HLTReconstructor(
        data=data,
        basis=basis,
        lambda_hlt="1e-4",
        covariance_norm="1.0",
        numeric_backend=backend,
        target_integration_backend=target_backend,
        mp_dps=dps,
        strict=False,
    )
    return reconstructor, target


def test_mp_basis_uses_original_decimal_sources():
    basis = HLTBasis(times=["0.1"], p="0", omega0="0")
    with mp.workdps(60):
        assert abs(basis.A_mpmath(60)[0, 0] - 5) < mp.mpf("1e-55")
        assert abs(basis.R_mpmath(60)[0] - 10) < mp.mpf("1e-55")


def test_mp_solver_retains_mp_native_output():
    result = solve_hlt_unconstrained(
        [["1", "0"], ["0", "1e-40"]],
        ["1", "1e-80"],
        backend="mpmath",
        mp_dps=70,
    )
    assert result.backend == "mpmath"
    assert hasattr(result.q[0], "_mpf_")
    with mp.workdps(70):
        assert abs(result.q[1] - mp.mpf("1e-40")) < mp.mpf("1e-100")


def test_end_to_end_mp_reconstruction_and_explicit_downconversion():
    reconstructor, target = _toy_problem()
    result = reconstructor.reconstruct(target, compute_mismatch=True)
    assert result.backend == "mpmath"
    assert hasattr(result.estimate, "_mpf_")
    assert hasattr(result.error, "_mpf_")
    assert hasattr(result.q[0], "_mpf_")
    assert result.checks.backend == "mpmath"
    assert result.target_integration_backend == "numpy"
    assert result.checks.stationarity_rel < mp.mpf("1e-30")
    assert not result.quadrature_errors

    converted = result.to_numpy()
    assert converted.backend == "numpy"
    assert isinstance(converted.q, np.ndarray)
    np.testing.assert_allclose(
        converted.estimate, float(result.estimate), rtol=0, atol=0
    )

    numpy_reconstructor, numpy_target = _toy_problem(backend="numpy")
    numpy_result = numpy_reconstructor.reconstruct(numpy_target, compute_mismatch=False)
    np.testing.assert_allclose(
        [float(result.estimate), float(result.error)],
        [numpy_result.estimate, numpy_result.error],
        rtol=2e-14,
        atol=0,
    )

    higher_reconstructor, higher_target = _toy_problem(dps=60)
    higher = higher_reconstructor.reconstruct(higher_target, compute_mismatch=False)
    with mp.workdps(60):
        assert abs(result.estimate - higher.estimate) < mp.mpf("1e-38")


def test_mixed_backend_promotes_float_target_but_builds_A_in_mpmath():
    reconstructor, target = _toy_problem(dps=55)
    result = reconstructor.reconstruct(target, compute_mismatch=False)
    expected_b = target.b_vector(reconstructor.basis, reconstructor.quad_options)

    with mp.workdps(55):
        for index, value in enumerate(expected_b):
            assert result.b[index] == mp.mpf(float(value))
        assert result.W[0, 0] != mp.mpf(float(result.W[0, 0]))


def test_full_mp_target_quadrature_remains_available_as_validation_mode():
    reconstructor, target = _toy_problem(dps=40, target_backend="mpmath")
    result = reconstructor.reconstruct(target, compute_mismatch=False)
    assert result.target_integration_backend == "mpmath"
    assert result.quadrature_errors

    mixed_reconstructor, mixed_target = _toy_problem(dps=40)
    mixed = mixed_reconstructor.reconstruct(mixed_target, compute_mismatch=False)
    with mp.workdps(40):
        assert abs(result.estimate - mixed.estimate) < mp.mpf("1e-9")


def test_mp_grid_systematics_remain_mp_native():
    reconstructor, target = _toy_problem(dps=35)
    result = reconstructor.reconstruct(target, compute_mismatch=False)
    omega = ["0.1", "0.2", "0.4", "0.8"]
    rho = ["1", "0.9", "0.7", "0.4"]
    grid = result.kernel_mismatch_on_grid(omega, rho_p=rho)
    systematic = result.estimate_kernel_systematic_on_grid(omega, rho)
    assert grid.backend == "mpmath"
    assert hasattr(grid.mismatch[0], "_mpf_")
    assert systematic.backend == "mpmath"
    assert hasattr(systematic.signed_bias, "_mpf_")
    assert isinstance(grid.to_numpy().mismatch, np.ndarray)


def test_direct_window_mp_evaluation_and_exact_divergence_rule():
    options = QuadOptions(mp_maxdegree=4, mp_method="gauss-legendre")
    target = IntegratedHVPTarget.from_high_energy(
        am_tau="0.2",
        delta="0.05",
        p="2",
        euclidean_window="ID",
        a_lat_inv_MeV="2500",
        window_quad_options=options,
    )
    value = target.tilde_theta_mpmath("0.2", mp_dps=30)
    assert hasattr(value, "_mpf_")
    assert abs(value - mp.mpf("0.1307693")) < mp.mpf("5e-6")

    divergent = IntegratedHVPTarget.from_sharp_high_energy(
        am_tau="0.2",
        p="4",
        euclidean_window="ID",
        a_lat_inv_MeV="2500",
        window_quad_options=options,
    )
    with pytest.raises(NonIntegrableTargetAreaError):
        divergent.area_mpmath(mp_dps=30)


def test_mp_target_difference_systematic():
    reconstructor, smooth = _toy_problem(dps=35)
    result = reconstructor.reconstruct(smooth, compute_mismatch=False)
    sharp = SharpHighEnergyTarget(t0="1.5", am_tau="0.4", p="2")
    systematic = result.estimate_target_smearing_systematic_on_grid(
        omega=["0.1", "0.2", "0.4", "0.8"],
        rho_p=["1", "0.9", "0.7", "0.4"],
        sharp_target=sharp,
    )
    assert systematic.backend == "mpmath"
    assert hasattr(systematic.absolute_bias, "_mpf_")
