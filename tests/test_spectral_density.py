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

"""Physical-density contracts independent of any concrete model's implementation."""

from dataclasses import dataclass
import numpy as np
import pytest
from scipy.integrate import trapezoid

import spectral as sp
from spectral.algorithms import hlt


@dataclass(frozen=True)
class ExponentialDensity(sp.SpectralDensity):
    energy_unit: str = "GeV"

    def scalar(self, omega, ctx):
        return ctx.exp(-omega)


def test_distribution_integration_override_survives_basis_conversion():
    class DeltaDensity(sp.SpectralDensity):
        energy_unit = "GeV"

        def scalar(self, omega, ctx):
            raise TypeError("a delta distribution cannot be sampled")

        def integrate(self, kernel, *, numerics=None, lower=None, upper=None):
            with (numerics or sp.numerics.numpy()).context() as ctx:
                return kernel.scalar(ctx.number(1), ctx)

    density = DeltaDensity()
    basis = sp.Basis(sp.kernels.exponential(energy_unit="GeV"), (1, 2), p=4)
    integrand = sp.systematics.integrand(density=density, kernel=basis.at_time(3))
    assert integrand.integrate() == pytest.approx(np.exp(-3))
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=np.exp(-np.array([1, 2])))
    result = sp.Reconstructor(basis, hlt.Solver(".01")).reconstruct(data, target=basis.at_time(3))
    report = sp.systematics.estimate(result, reference_kernel=basis.at_time(3), density=density)
    assert report.nominal_bias == pytest.approx(float(result.value)-np.exp(-3))


@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
def test_custom_density_integrates_physical_kernels_without_a_basis(backend):
    density = ExponentialDensity()
    policy = sp.numerics.Numerics(backend, dps=45)
    kernel = sp.kernels.exponential(energy_unit="GeV").at_time(2)
    with policy.context() as ctx:
        expected = ctx.number(1) / 3
        actual = sp.integrate(density=density, kernel=kernel, numerics=policy)
        tolerance = ctx.number("1e-40" if backend == "mpmath" else "1e-11")
        assert abs(actual - expected) < tolerance
        finite = density.integrate(kernel, lower=".2", upper="1.7", numerics=policy)
        assert (
            abs(finite - (ctx.exp(-ctx.number(".6")) - ctx.exp(-ctx.number("5.1"))) / 3)
            < tolerance
        )
    np.testing.assert_allclose(
        np.asarray(density([0, 1, 2]), float), np.exp(-np.arange(3))
    )
    assert not hasattr(density, "p")


def test_tabulated_density_integration_clips_to_support():
    density = sp.models.TabulatedSpectralDensity((1, 2, 3), (1, 1, 1))
    kernel = sp.kernels.exponential().at_time(1)
    assert density.integrate(kernel, lower=0, upper=2) == pytest.approx(
        np.exp(-1) - np.exp(-2)
    )
    assert density.integrate(kernel, lower=4) == 0
    assert density.integrand(kernel)(0) == 0
    assert density.integrand(kernel)(4) == 0


@pytest.mark.parametrize("p", [0, 2, 4, ".5"])
def test_basis_integrand_cancels_power_at_origin_and_for_periodic_kernels(p):
    density = ExponentialDensity()
    basis = sp.Basis(
        sp.kernels.periodic_correlator(extent=10, energy_unit="GeV"), (1, 2), p=p
    )
    integrand = sp.systematics.integrand(density=density, kernel=basis.at_time(1))
    assert integrand(0) == 2
    assert integrand.integrate() == pytest.approx(1 / 2 + 1 / 10, rel=1e-11)
    # Cut and linear-combination wrappers must preserve the cancellation.
    cut = sp.targets.smooth_high_energy(
        kernel=basis.at_time(1), cutoff=".7", width=".1"
    )
    combination = sp.LinearCombination((cut, basis.at_time(2)), (1, -1))
    actual = sp.systematics.integrand(density=density, kernel=combination)(0)
    assert actual == pytest.approx(2 / (1 + np.exp(14)) - 2)


@pytest.mark.parametrize("p", [0, 2, 4])
@pytest.mark.parametrize("grid", [False, True])
def test_arbitrary_physical_density_uses_basis_power_in_systematics(p, grid):
    density = ExponentialDensity()
    basis = sp.Basis(sp.kernels.exponential(energy_unit="GeV"), (1, 2), p=p)
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1 / 2, 1 / 3])
    result = sp.Reconstructor(basis, hlt.Solver(".01")).reconstruct(
        data, target=basis.at_time(3)
    )
    energies = np.linspace(0, 12, 1201)
    options = {"energies": energies} if grid else {}
    report = sp.systematics.estimate(
        result, reference_kernel=basis.at_time(3), density=density, **options
    )
    q = np.asarray(result.coefficients, float)
    expected = (
        trapezoid(
            np.exp(-energies)
            * (
                q[0] * np.exp(-energies)
                + q[1] * np.exp(-2 * energies)
                - np.exp(-3 * energies)
            ),
            energies,
        )
        if grid
        else q[0] / 2 + q[1] / 3 - 1 / 4
    )
    assert report.nominal_bias == pytest.approx(expected, rel=1e-10, abs=1e-14)


def test_custom_kernel_uses_explicit_basis_and_validates_conventions():
    density = ExponentialDensity()
    basis = sp.Basis(sp.kernels.exponential(energy_unit="GeV"), (1,), p=4)
    custom = sp.targets.from_function(lambda w, ctx: w**4 * ctx.exp(-w))
    with pytest.raises(ValueError, match="provide basis"):
        sp.systematics.integrand(density=density, kernel=custom)
    integrand = sp.systematics.integrand(density=density, kernel=custom, basis=basis)
    assert integrand.integrate(lower=".1") == pytest.approx(np.exp(-0.2) / 2, rel=1e-11)
    with pytest.raises(ValueError, match="spectral powers"):
        sp.systematics.integrand(
            density=density, kernel=basis.kernel.at_time(1), basis=basis
        )
    with pytest.raises(ValueError, match="energy unit"):
        density.integrate(sp.kernels.exponential(energy_unit="MeV").at_time(1))
    with pytest.raises(ValueError, match="energy unit"):
        sp.systematics.for_basis(
            density, sp.Basis(sp.kernels.exponential(energy_unit="MeV"), (1,), p=4)
        )


def test_deformations_preserve_physical_integral_by_default():
    density = sp.models.TwoPionModel()
    energies = np.linspace(float(density.domain[0]), 3, 101)
    variations = sp.systematics.gaussian_deformations(
        density=density,
        energies=energies,
        count=2,
        relative_sigma=".1",
        correlation_length=".2",
        seed=42,
        preserve_integral=True,
    )
    expected = trapezoid(density(energies), energies)
    assert variations.metadata["normalization_spectral_power"] == 0
    for variant in variations.models:
        assert isinstance(variant, sp.SpectralDensity)
        assert trapezoid(variant(energies), energies) == pytest.approx(
            expected, rel=1e-13
        )
        assert variant(4) == density(4)


@pytest.mark.parametrize("power", [None, 4])
def test_legacy_tabulated_and_deformed_archives_keep_original_values(power):
    from spectral.io.archive import Decoder, Encoder

    table = sp.models.TabulatedSpectralDensity((".1", 1, 2), (1, 1, 1))
    variation = sp.systematics.DeformedDensity(table, (".1", 2), (2, 2))
    payload = Encoder().encode(variation)
    payload["type"] = "spectral.systematics.deformations.DeformedModel"
    old_table = payload["fields"].pop("density")
    payload["fields"]["model"] = old_table
    old_table["type"] = "spectral.systematics.model.TabulatedModel"
    old_table["fields"]["spectral_power"] = power
    restored = Decoder().decode(payload)
    basis = sp.Basis(sp.kernels.exponential(), (1,), p=4)
    converted = sp.systematics.for_basis(restored, basis)
    assert converted(0.5) == 2
    assert Decoder().decode(Encoder().encode(restored))(0.5) == 2


def test_physical_density_adaptive_estimation_in_processes():
    density = sp.models.TabulatedSpectralDensity((0, 1, 2), (1, 2, 1))
    basis = sp.Basis(sp.kernels.exponential(), (1, 2), p=4)
    data = sp.CorrelatorData.from_arrays(times=basis.times, mean=[1, 0.5])
    result = sp.Reconstructor(basis, hlt.Solver(".01")).reconstruct(
        data, target=basis.at_time(3)
    )
    expected = sp.systematics.estimate(
        result, reference_kernel=basis.at_time(3), density=density
    )
    with sp.execution.processes(2) as execution:
        actual = sp.systematics.estimate(
            result,
            reference_kernel=basis.at_time(3),
            density=density,
            execution=execution,
        )
    np.testing.assert_array_equal(actual.biases, expected.biases)
