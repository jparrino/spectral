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

"""Numerical regressions from the pre-publication review."""

import mpmath as mp
import numpy as np
import pytest

import spectral as sp
from spectral.algorithms import hlt
from spectral.core.result import Failure


@pytest.mark.parametrize("power", [-1, -2, -2.5])
@pytest.mark.parametrize("rate", [1e-5, 1e4, 1e5])
def test_negative_power_integrals_across_energy_scales(power, rate):
    lower = 1 / rate
    # Independent quadrature in the dimensionless coordinate x = rate * omega
    # resolves the peak regardless of the physical energy scale.
    with mp.workdps(60):
        r, lo, p = map(mp.mpf, (rate, lower, power))
        expected = float(
            mp.quad(lambda x: x**p * mp.exp(-x), [r * lo, mp.inf]) / r ** (p + 1)
        )
    with sp.numerics.numpy().context() as ctx:
        actual = ctx.exp_integral(power, rate, lower)
    assert actual == pytest.approx(expected, rel=1e-13, abs=0)


def test_concentrated_negative_power_reconstruction():
    basis = sp.Basis(sp.kernels.exponential(), (10000,), p=-1, omega_0="0.0001")
    data = sp.CorrelatorData.from_arrays(
        times=basis.times, mean=[1], covariance=[[0.01]]
    )
    result = sp.Reconstructor(basis, hlt.Solver(".001")).reconstruct(
        data, target=basis.at_time(10000)
    )
    assert result.value == pytest.approx(0.99999733310771841875075, rel=1e-14)
    assert result.variance == pytest.approx(0.01 * result.value**2, rel=1e-14)


@pytest.mark.parametrize("policy", [sp.numerics.numpy(), sp.numerics.mpmath(25)])
def test_lambda_rounding_to_one_is_rejected_at_working_precision(policy):
    basis = sp.Basis(sp.kernels.exponential(), (1,))
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1])
    recon = sp.Reconstructor(basis, hlt.Solver("0." + "9" * 40), policy)
    with pytest.raises(sp.ReconstructionError, match="after backend conversion"):
        recon.reconstruct(data, target=basis.at_time(1))
    result = recon.reconstruct(data, target=basis.at_time(1), on_error="record")
    assert isinstance(result, Failure)
    assert "higher precision" in result.error


def test_lambda_near_one_works_at_sufficient_precision():
    basis = sp.Basis(sp.kernels.exponential(), (1,))
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1])
    lam = "0." + "9" * 40
    result = sp.Reconstructor(
        basis, hlt.Solver(lam), sp.numerics.mpmath(80)
    ).reconstruct(data, target=basis.at_time(1))
    with mp.workdps(80):
        expected = (1 - mp.mpf(lam)) / (1 + mp.mpf(lam))
        assert result.value > 0
        assert abs(result.value / expected - 1) < mp.mpf("1e-75")


def test_unrepresentable_lambda_does_not_fail_neighboring_scan_points():
    basis = sp.Basis(sp.kernels.exponential(), (1,))
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1])
    last_float = np.nextafter(1.0, 0.0)
    result = sp.Reconstructor(
        basis, hlt.Solver([0, last_float, "0." + "9" * 40])
    ).reconstruct(data, target=basis.at_time(1), on_error="record")
    assert result.results[0].value == 1
    assert result.results[1].value > 0
    assert isinstance(result.results[2], Failure)
