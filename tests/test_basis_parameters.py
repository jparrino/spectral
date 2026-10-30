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

import mpmath as mp
import pytest
import spectral as sp
from spectral.algorithms import hlt


@pytest.mark.parametrize("p,lower", [(0, 0), (2, ".3"), (-1, ".3")])
def test_basis_gram_and_area(p, lower):
    basis = sp.Basis(sp.kernels.exponential(), (1, 2), p=p, omega_0=lower)
    with sp.numerics.mpmath(45).context() as ctx:
        gram = basis.gram(ctx)
        expected = mp.quad(
            lambda w: w ** (2 * p) * mp.exp(-3 * w), [mp.mpf(lower), mp.inf]
        )
        assert abs(gram[0, 1] - expected) < mp.mpf("1e-40")
        area = basis.area(ctx)
        assert abs(
            area[0] - mp.quad(lambda w: w**p * mp.exp(-w), [mp.mpf(lower), mp.inf])
        ) < mp.mpf("1e-40")


def test_periodic_exact_recovery():
    basis = sp.Basis(
        sp.kernels.periodic_correlator(extent=16), (1, 2, 3), p=2, omega_0=".2"
    )
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1, 0.5, 0.2])
    result = sp.Reconstructor(basis, hlt.Solver(0), sp.numerics.mpmath(60)).reconstruct(
        data, target=basis.at_time(2)
    )
    assert abs(result.value - mp.mpf(".5")) < mp.mpf("1e-50")


def test_invalid_domains_and_covariance():
    with pytest.raises(ValueError, match="Gram matrix"):
        sp.Basis(sp.kernels.exponential(), (1, 2), p=-0.75)
    with pytest.raises(ValueError, match="positive semidefinite"):
        sp.CorrelatorData.from_arrays(
            times=[1, 2], mean=[1, 1], covariance=[[1, 2], [2, 1]]
        )
    with pytest.raises(ValueError, match="extent"):
        sp.Basis(sp.kernels.periodic_correlator(extent=3), (1, 3))
