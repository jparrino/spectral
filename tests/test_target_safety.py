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

import pytest
import spectral as sp


@pytest.mark.parametrize(
    "power,lower,squared",
    [(2, 0, False), (4, ".1", False), (2, 0, True), ("4.5", ".1", True)],
)
def test_divergent_hvp_targets_fail_before_quadrature(power, lower, squared):
    target = sp.targets.smooth_high_energy(
        kernel=sp.targets.hvp(mass=".1", p=power), cutoff=".4", width=".1"
    )
    with sp.numerics.mpmath(50).context() as ctx:
        with pytest.raises(ValueError, match="diverges"):
            if squared:
                target.basis_integrals(
                    sp.Basis(sp.kernels.exponential(), (1, 2), p=power, omega_0=lower),
                    ctx,
                )
            else:
                target.integral(ctx, lower)


def test_sharp_cut_avoids_evaluating_singular_excluded_energy():
    target = sp.targets.sharp_high_energy(kernel=sp.targets.hvp(mass=".1"), cutoff=".4")
    assert target(0) == 0
    with sp.numerics.mpmath(50).context() as ctx:
        target.validate_integral(ctx, 0, None, power=2)


def test_breakpoints_retain_arbitrary_precision_smearing_width():
    target = sp.targets.smooth_high_energy(
        kernel=sp.kernels.exponential().at_time(1), cutoff="1", width="1e-25"
    )
    with sp.numerics.mpmath(60).context() as ctx:
        points = target.integration_points(ctx)
        assert len(set(points)) == 5
        assert abs(points[3] - points[2] - ctx.number("1e-25")) < ctx.number("1e-59")
