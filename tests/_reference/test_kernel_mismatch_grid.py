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

from hlt import CorrelatorData, HLTBasis, HLTReconstructor, HighEnergyTarget


def main():
    Nt = 40
    T = 80
    times = np.arange(Nt, dtype=float)
    mean = np.exp(-0.25 * times) + 0.05 * np.exp(-0.7 * times)
    cov = np.diag((0.02 * np.maximum(mean, 1e-8)) ** 2)

    data = CorrelatorData.from_arrays(mean=mean, covariance=cov, times=times, T=T)
    basis_times = data.select_times(t_min=1, t_max=18, stride=2)
    basis = HLTBasis(times=basis_times, p=2, T=T, periodic=True, omega0=0.2)
    recon = HLTReconstructor(
        data=data,
        basis=basis,
        lambda_hlt=1e-6,
        covariance_norm=mean[0] ** 2,
        regulator="identity",
        strict=False,
    )
    target = HighEnergyTarget(t0=6, am_tau=0.7, delta=0.05, p=2, T=T, periodic=True)
    result = recon.reconstruct(target, compute_mismatch=False)

    omega = np.linspace(basis.omega0, 4.0, 300)
    rho_p = np.exp(-omega)
    grid = result.kernel_mismatch_on_grid(omega=omega, rho_p=rho_p)

    assert grid.omega.shape == omega.shape
    assert grid.target.shape == omega.shape
    assert grid.reconstructed.shape == omega.shape
    assert grid.mismatch.shape == omega.shape
    assert grid.weighted_mismatch.shape == omega.shape
    np.testing.assert_allclose(grid.mismatch, grid.reconstructed - grid.target)
    np.testing.assert_allclose(grid.weighted_mismatch, grid.mismatch * rho_p)

    syst = result.estimate_kernel_systematic_on_grid(omega=omega, rho_p=rho_p, include_stat_error=False)
    weights = np.empty_like(omega)
    d = np.diff(omega)
    weights[0] = 0.5 * d[0]
    weights[-1] = 0.5 * d[-1]
    weights[1:-1] = 0.5 * (d[:-1] + d[1:])
    np.testing.assert_allclose(syst.signed_bias, np.sum(weights * grid.weighted_mismatch))
    print("kernel mismatch grid tests passed")


if __name__ == "__main__":
    main()
