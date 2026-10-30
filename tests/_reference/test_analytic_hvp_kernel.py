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
from hlt import CorrelatorData, HLTBasis, HLTReconstructor, IntegratedHVPTarget
from hlt.mpmath_backend import MPWorkspace, _bernecker_meyer_inner
from hlt.targets import _bernecker_meyer_inner_analytic_float
from mpmath import mp


def _target_with_unit_muon_mass():
    return IntegratedHVPTarget(
        am_tau="0.2",
        delta="0.05",
        p="2",
        euclidean_window=("0.4", "1.0"),
        window_delta="0.15",
        am_mu="1",
    )


def _numeric_s_prime_reference(u):
    """Original production integral, retained only as an independent test."""

    def weight(r):
        root = mp.sqrt(r * (r + 4))
        return 16 / (root * (r + 2 + root) ** 2)

    def integrand(y):
        if y == 0:
            return mp.zero
        z = u * y / 2
        one_minus_sinc_squared = 1 - (mp.sin(z) / z) ** 2
        return 2 * y * weight(y * y) * u * u * one_minus_sinc_squared

    return mp.quad(integrand, [0, 1, mp.inf], method="tanh-sinh")


@pytest.mark.parametrize("u_source", ["0.001", "0.01", "0.1", "0.5", "1", "2", "5"])
def test_analytic_inner_kernel_reproduces_original_s_prime_integral(u_source):
    target = _target_with_unit_muon_mass()
    with mp.workdps(60):
        u = mp.mpf(u_source)
        analytic = _bernecker_meyer_inner(target, u, mp, MPWorkspace(60))
        numeric = _numeric_s_prime_reference(u)
        # The tolerance is set by convergence of the independent oscillatory
        # improper integral, not by the analytic expression.
        assert abs(analytic - numeric) <= mp.mpf("5e-11") * abs(analytic)


@pytest.mark.parametrize("u", [1e-6, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0])
def test_float_analytic_kernel_agrees_with_mp_closed_form(u):
    target = _target_with_unit_muon_mass()
    with mp.workdps(60):
        exact = _bernecker_meyer_inner(target, mp.mpf(str(u)), mp, MPWorkspace(60))
        approximate = mp.mpf(_bernecker_meyer_inner_analytic_float(u, 1.0))
        assert abs(approximate - exact) <= mp.mpf("5e-11") * abs(exact)


def test_mpmath_reconstruction_defaults_to_full_mp_target_integration():
    data = CorrelatorData.from_arrays(
        mean=["1", "0.7"],
        covariance=[["0.01", "0"], ["0", "0.01"]],
        times=[0, 1],
    )
    reconstructor = HLTReconstructor(
        data=data,
        basis=HLTBasis(times=[1], p="2"),
        lambda_hlt="1e-8",
        covariance_norm="1",
        solver_backend="mpmath",
        strict=False,
    )
    assert reconstructor.target_integration_backend == "mpmath"


def test_analytic_kernel_records_no_s_prime_quadrature_error():
    target = _target_with_unit_muon_mass()
    workspace = MPWorkspace(50)
    with mp.workdps(50):
        value = _bernecker_meyer_inner(target, mp.mpf("0.5"), mp, workspace)
    assert mp.isfinite(value)
    assert "bernecker_meyer_inner" not in workspace.quadrature_errors
    assert np.isfinite(float(value))


@pytest.mark.parametrize("u_source", ["1e-40", "1e-20", "1e-10", "100", "1000"])
def test_analytic_kernel_is_stable_at_extreme_quadrature_nodes(u_source):
    target = _target_with_unit_muon_mass()
    with mp.workdps(50):
        value = _bernecker_meyer_inner(
            target, mp.mpf(u_source), mp, MPWorkspace(50)
        )
        assert mp.isfinite(value)
        assert value > 0
