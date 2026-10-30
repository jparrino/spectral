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

"""Minimal example for the hlt package.

Replace the synthetic data below with your jks-system correlator:

    import jks
    db = jks.resamples("data/48I.jks")
    cor = db.get("C(t).r")
    mean = cor.mean()
    cov = cor.cov()
"""

import numpy as np

from hlt import CorrelatorData, HLTBasis, HLTReconstructor, HighEnergyTarget, IntegratedTarget


Nt = 48
T = 96
times = np.arange(Nt, dtype=float)
mean = np.exp(-0.25 * times) + 0.1 * np.exp(-0.6 * times)
errors = 0.02 * np.maximum(mean, 1e-8)
cov = np.diag(errors**2)

p = 2
omega_cut = 0.82
delta = 0.14
lambda_hlt = 1e-4

data = CorrelatorData.from_arrays(mean=mean, covariance=cov, times=times, T=T)

basis_times = data.select_times(t_min=1, t_max=24, stride=2)
basis = HLTBasis(times=basis_times, p=p, T=T, periodic=True, omega0=0.0)

recon = HLTReconstructor(
    data=data,
    basis=basis,
    lambda_hlt=lambda_hlt,
    covariance_norm=mean[0] ** 2,
    regulator="covariance",  # or "identity"
    strict=False,
)

target = HighEnergyTarget(t0=8, am_tau=omega_cut, delta=delta, p=p, T=T, periodic=True)
result = recon.reconstruct(target)

print("single-time estimate:", result.estimate)
print("single-time error:   ", result.error)
print("regulator:           ", result.regulator)
print("area rel residual:  ", result.checks.area_rel)
print("stationarity:       ", result.checks.stationarity_rel)
print("warnings:           ", result.checks.warnings)

# Direct integrated reconstruction with arbitrary toy weights.
target_times = np.arange(2, 16, dtype=float)
K = np.exp(-0.1 * target_times)
int_target = IntegratedTarget.from_high_energy_times(
    times=target_times,
    weights=K,
    am_tau=omega_cut,
    delta=delta,
    p=p,
    T=T,
    periodic=True,
)
int_result = recon.reconstruct(int_target)

print("integrated estimate:", int_result.estimate)
print("integrated error:   ", int_result.error)
print("integrated warnings:", int_result.checks.warnings)
