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

"""Smoke tests for HLTBasis omega0 support."""

import numpy as np
from scipy.integrate import quad

from hlt import CorrelatorData, HLTBasis, HLTReconstructor, HighEnergyTarget


def assert_close(a, b, rtol=1e-9, atol=1e-11):
    if not np.allclose(a, b, rtol=rtol, atol=atol):
        raise AssertionError(f"not close:\n{a}\n{b}")


def test_analytic_A_R_against_quad():
    times = np.array([1.0, 3.0, 5.0])
    basis = HLTBasis(times=times, p=1.3, T=16.0, periodic=True, omega0=0.4)

    A = basis.A()
    R = basis.R()

    A_quad = np.empty_like(A)
    for i, ti in enumerate(times):
        for j, tj in enumerate(times):
            val, _ = quad(lambda w: basis.phi(ti, w) * basis.phi(tj, w), basis.omega0, np.inf)
            A_quad[i, j] = val

    R_quad = np.empty_like(R)
    for i, ti in enumerate(times):
        val, _ = quad(lambda w: basis.phi(ti, w), basis.omega0, np.inf)
        R_quad[i] = val

    assert_close(A, A_quad)
    assert_close(R, R_quad)


def test_reconstruction_with_omega0_and_identity_regulator():
    Nt = 32
    T = 64
    full_times = np.arange(Nt, dtype=float)
    mean = np.exp(-0.25 * full_times) + 0.1 * np.exp(-0.6 * full_times)
    cov = np.diag((0.02 * np.maximum(mean, 1e-8)) ** 2)

    data = CorrelatorData.from_arrays(mean=mean, covariance=cov, times=full_times, T=T)
    basis_times = data.select_times(t_min=1, t_max=15, stride=2)
    basis = HLTBasis(times=basis_times, p=2.0, T=T, periodic=True, omega0=0.3)
    target = HighEnergyTarget(t0=8.0, am_tau=0.8, delta=0.12, p=2.0, T=T, periodic=True)

    recon = HLTReconstructor(
        data=data,
        basis=basis,
        lambda_hlt=1e-4,
        regulator="identity",
        strict=False,
    )
    result = recon.reconstruct(target, compute_mismatch=False)
    if result.checks.area_rel > 1e-8:
        raise AssertionError(f"area constraint failed: {result.checks.area_rel}")
    if result.checks.stationarity_rel > 1e-8:
        raise AssertionError(f"stationarity failed: {result.checks.stationarity_rel}")


if __name__ == "__main__":
    test_analytic_A_R_against_quad()
    test_reconstruction_with_omega0_and_identity_regulator()
    print("omega0 tests passed")
