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

"""Independent forward Laplace reference for the supplied two-pion toy model."""

import mpmath as mp
import numpy as np
import pytest
from dataclasses import replace
import spectral as sp
from spectral.algorithms import hlt


@pytest.fixture(scope="module")
def exact_correlator():
    with mp.workdps(65):
        mpi, mrho, gamma = map(mp.mpf, ("0.13957039", "0.77526", "0.1491"))
        threshold = 2 * mpi

        def rho_weight(w):
            beta = mp.sqrt(1 - (threshold / w) ** 2)
            beta_rho = mp.sqrt(1 - (threshold / mrho) ** 2)
            width = gamma * (mrho / w) * (beta / beta_rho) ** 3
            modulus = abs(mrho**2 / mp.mpc(mrho**2 - w**2, -w * width)) ** 2
            return w**2 * beta**3 * modulus / (48 * mp.pi**2)

        return {
            t: mp.quad(
                lambda w: rho_weight(w) * mp.exp(-w * t),
                [threshold, mp.mpf("0.6"), mrho, mp.mpf("1.1"), mp.mpf("2.5"), mp.inf],
            )
            for t in (1, 2, 3, 4, 5)
        }


@pytest.mark.parametrize("p", [0, 2])
@pytest.mark.parametrize("regulator", ["identity", "covariance"])
def test_two_pion_lambda_scan_recovers_input_correlator(exact_correlator, p, regulator):
    policy = sp.numerics.mpmath(55)
    model = sp.models.TwoPionModel()
    basis = sp.Basis(
        sp.kernels.exponential(energy_unit="GeV"),
        (1, 2, 3, 4),
        p=p,
        omega_0=model.domain[0],
    )
    with policy.context() as ctx:
        mean = ctx.array([exact_correlator[t] for t in basis.times])
        errors = mean * ctx.number(".005")
        covariance = ctx.array(
            [
                [
                    errors[i] * errors[j] * (1 if i == j else ctx.number(".95"))
                    for j in range(4)
                ]
                for i in range(4)
            ]
        )
    data = sp.CorrelatorData(basis.times, mean, covariance, energy_unit="GeV")
    solver = hlt.Solver(
        ["0", "1e-40", "1e-12", "1e-3"], regulator=regulator, covariance_norm="1"
    )
    recon = sp.Reconstructor(basis, solver, policy)
    scan = recon.reconstruct(
        data, target=sp.targets.correlator(basis=basis), target_times=[2, 5]
    )
    with policy.context() as ctx:
        exact = scan.at_lambda("0").at_time(2)
        assert abs(exact.value - exact_correlator[2]) / abs(
            exact_correlator[2]
        ) < ctx.number("1e-45")
        small = scan.at_lambda("1e-40").at_time(2)
        large = scan.at_lambda("1e-3").at_time(2)
        assert abs(small.value - exact_correlator[2]) < abs(
            large.value - exact_correlator[2]
        )
        # An omitted time is a prediction, not an exact basis-recovery identity.
        extrapolated = scan.at_lambda("0").at_time(5)
        assert abs(extrapolated.value - exact_correlator[5]) > ctx.number("1e-15")


@pytest.mark.parametrize("p", [0, 2, 4])
def test_two_pion_signed_bias_matches_independent_reference(exact_correlator, p):
    policy = sp.numerics.mpmath(50)
    model = sp.models.TwoPionModel()
    basis = sp.Basis(
        sp.kernels.exponential(energy_unit="GeV"),
        (1, 2, 3),
        p=p,
        omega_0=model.domain[0],
    )
    data = sp.CorrelatorData.from_arrays(
        times=basis.times,
        mean=[exact_correlator[t] for t in basis.times],
        energy_unit="GeV",
    )
    recon = sp.Reconstructor(basis, hlt.Solver(["0", "1e-20", "1e-6"]), policy)
    scan = recon.reconstruct(
        data, target=sp.targets.correlator(basis=basis), target_times=[2, 5]
    )
    report = sp.systematics.estimate(
        scan, reference_kernel=sp.targets.correlator(basis=basis), density=model
    )
    with policy.context() as ctx:
        for r, bias in zip(scan.results, report.biases[0]):
            difference = r.value - exact_correlator[r.coordinates["target_time"]]
            assert abs(difference - bias) < ctx.number("1e-40")


@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
@pytest.mark.parametrize("p", [0, 2, 4, "1.5"])
def test_two_pion_forward_correlator_is_independent_of_basis_power(
    exact_correlator, p, backend
):
    model = sp.models.TwoPionModel()
    basis = sp.Basis(
        sp.kernels.exponential(energy_unit="GeV"),
        (2,),
        p=p,
        omega_0=model.domain[0],
    )
    policy = sp.numerics.Numerics(backend, dps=50)
    with policy.context() as ctx:
        kernel = basis.at_time(2)
        actual = sp.systematics.integrand(density=model, kernel=kernel).integrate(
            numerics=policy
        )
        tolerance = ctx.number("1e-40" if backend == "mpmath" else "1e-9")
        expected = ctx.number(exact_correlator[2])
        assert abs(actual - expected) < tolerance * abs(expected)
        energies = ctx.array([0, model.domain[0], ".5", "1.2"])
        values = model(energies, numerics=policy)
        converted = sp.systematics.for_basis(model, basis)(energies, numerics=policy)
        assert values[0] == values[1] == 0
        for w, value, rho_p in zip(energies[2:], values[2:], converted[2:]):
            physical = w**2 * model.r_ratio(w**2, ctx) / (12 * ctx.pi**2)
            assert abs(value - physical) < tolerance * physical
            assert abs(rho_p * w ** ctx.number(p) - physical) < tolerance * physical
        direct = sp.integrate(
            density=model, kernel=basis.kernel.at_time(2), numerics=policy
        )
        assert abs(direct - expected) < tolerance * abs(expected)
    assert not hasattr(model, "p")
    assert not hasattr(model, "for_basis")


@pytest.mark.parametrize("grid", [False, True])
def test_shared_two_pion_models_resolve_each_results_basis(
    exact_correlator, grid, tmp_path
):
    model = sp.models.TwoPionModel()
    powers = (0, 2, 4)
    results = []
    for p in powers:
        basis = sp.Basis(
            sp.kernels.exponential(energy_unit="GeV"),
            (1, 2, 3),
            p=p,
            omega_0=model.domain[0],
        )
        data = sp.CorrelatorData.from_arrays(
            times=basis.times, mean=[float(exact_correlator[t]) for t in basis.times]
        )
        result = sp.Reconstructor(basis, hlt.Solver(".01")).reconstruct(
            data, target=basis.at_time(5)
        )
        results.append(replace(result, coordinates={"power": p}))
    results = sp.ResultCollection({"power": powers}, tuple(results))
    references = sp.TargetGrid(
        {"power": powers}, tuple(r.basis.at_time(5) for r in results.results)
    )
    # Shared multiplicative variations must also follow each basis's convention.
    noise = sp.systematics.gaussian_deformations(
        density=model,
        energies=(model.domain[0], ".8", "2"),
        count=1,
        relative_sigma=".1",
        correlation_length=".2",
        seed=42,
    )
    variations = sp.systematics.DeformationSet(
        (replace(model, m_rho=".78526"), *noise.models),
        ("mass_shift", *noise.ids),
        {},
    )
    options = {"energies": np.linspace(0, 3, 201)} if grid else {}
    report = sp.systematics.estimate(
        results,
        reference_kernel=references,
        density=model,
        deformations=variations,
        **options,
    )
    for j, r in enumerate(results.results):
        separate = sp.systematics.estimate(
            r,
            reference_kernel=r.basis.at_time(5),
            density=model,
            deformations=variations,
            **options,
        )
        np.testing.assert_allclose(
            np.asarray(report.biases[:, j], float),
            np.asarray(separate.biases[:, 0], float),
            rtol=1e-11,
        )
    path = tmp_path / "two_pion.spectral"
    sp.io.save(path, results=results, systematics=report)
    loaded = sp.io.load(path)
    assert loaded.systematics.models == report.models
    repeated = sp.systematics.estimate(
        loaded.results,
        reference_kernel=references,
        density=loaded.systematics.models[0],
        deformations=sp.systematics.DeformationSet(
            loaded.systematics.models[1:], loaded.systematics.model_ids[1:], {}
        ),
        **options,
    )
    np.testing.assert_array_equal(repeated.biases, report.biases)


def test_two_pion_deformations_preserve_density_integral_for_basis():
    model = sp.models.TwoPionModel()
    basis = sp.Basis(sp.kernels.exponential(energy_unit="GeV"), (1,), p=4)
    energies = np.linspace(float(model.domain[0]), 3, 101)
    variations = sp.systematics.gaussian_deformations(
        density=model,
        basis=basis,
        energies=energies,
        count=2,
        relative_sigma=".1",
        correlation_length=".2",
        seed=42,
        preserve_integral=True,
    )
    assert variations.metadata["normalization_spectral_power"] == basis.p
    with sp.numerics.numpy().context() as ctx:
        from spectral.systematics.deformations import trapezoid_weights

        weights = trapezoid_weights(energies, ctx)
        expected = ctx.dot(weights, sp.systematics.for_basis(model, basis)(energies))
        for variant in variations.models:
            actual = ctx.dot(
                weights,
                sp.systematics.for_basis(variant, basis)(energies),
            )
            assert actual == pytest.approx(expected, rel=1e-13)


def test_legacy_two_pion_archive_retains_its_density_convention():
    from spectral.io.archive import Decoder, Encoder

    model = sp.models.TwoPionModel()
    payload = Encoder().encode(model)
    payload["type"] = "spectral.systematics.two_pion.TwoPionModel"
    payload["fields"]["p"] = 4
    restored = Decoder().decode(payload)
    basis = sp.Basis(sp.kernels.exponential(energy_unit="GeV"), (1,), p=4)
    assert restored.spectral_power == 4
    assert restored.density == model
    assert restored(0.5) == sp.systematics.for_basis(model, basis)(0.5)
    assert Decoder().decode(Encoder().encode(restored))(0.5) == restored(0.5)


def test_supplied_numpy_model_matches_new_model():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).parents[1] / "two_pion_HLTtest/two_pion_model.py"
    if not path.is_file():
        pytest.skip("Optional supplied two-pion reference model is not bundled.")
    spec = importlib.util.spec_from_file_location("two_pion_supplied", path)
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    old, new = module.TwoPionModel(), sp.models.TwoPionModel()
    for omega in (0.28, 0.5, 0.77526, 1, 2):
        assert new.R(omega**2) == pytest.approx(float(old.R(omega**2)), rel=5e-14)
