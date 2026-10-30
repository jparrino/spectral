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

"""User functions as targets, including time binding and process transport."""

from functools import partial
from dataclasses import replace
import math

import mpmath as mp
import numpy as np
import pytest

import spectral as sp
from spectral.algorithms import hlt


def mykernel(omega, t):
    return omega**2 * np.exp(-t * omega)


def precise_kernel(omega, t, *, ctx):
    return omega**2 * ctx.exp(-t * omega)


def fixed_kernel(omega):
    return omega**2 * math.exp(-2 * omega)


def precise_fixed_kernel(omega, *, ctx):
    return omega**2 * ctx.exp(-2 * omega)


def periodic_kernel(omega, t, *, extent, ctx):
    return omega**2 * (ctx.exp(-t * omega) + ctx.exp(-(extent - t) * omega))


def bad_kernel(omega, t):
    raise TypeError("failure inside user function")


def nonfinite_kernel(omega):
    return np.nan


def setup(policy=None, periodic=False):
    basis = sp.Basis(
        (
            sp.kernels.periodic_correlator(extent=16)
            if periodic
            else sp.kernels.exponential()
        ),
        (1, 2),
        p=2,
        omega_0=".2",
    )
    data = sp.CorrelatorData.from_arrays(
        times=basis.times,
        mean=(1, 0.5),
        covariance=np.eye(2) * 0.01,
    )
    return (
        sp.Reconstructor(basis, hlt.Solver([0, ".001"]), policy or sp.numerics.numpy()),
        data,
    )


@pytest.mark.parametrize("function", [mykernel, precise_kernel])
def test_direct_function_matches_builtin_for_all_times_and_lambdas(function):
    recon, data = setup()
    actual = recon.reconstruct(data, target=function, target_times=[1, 2, 3])
    expected = recon.reconstruct(
        data,
        target=sp.targets.correlator(basis=recon.basis),
        target_times=[1, 2, 3],
    )
    assert actual.axes == expected.axes
    np.testing.assert_allclose(
        actual.coefficients.astype(float),
        expected.coefficients.astype(float),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual.statistical_covariance.astype(float),
        expected.statistical_covariance.astype(float),
        rtol=1e-10,
        atol=1e-12,
    )


@pytest.mark.parametrize("function", [fixed_kernel, precise_fixed_kernel])
def test_fixed_functions_and_legacy_context_wrapper(function):
    recon, data = setup()
    actual = recon.reconstruct(data, target=function)
    expected = recon.reconstruct(data, target=recon.basis.at_time(2))
    np.testing.assert_allclose(
        actual.values.astype(float),
        expected.values.astype(float),
        rtol=1e-12,
    )
    legacy = sp.CallableKernel(lambda omega, ctx: ctx.exp(-omega))
    assert legacy(1) == pytest.approx(math.exp(-1))


def test_partial_can_fix_time_and_local_functions_work_serially():
    recon, data = setup()
    fixed = recon.reconstruct(data, target=partial(mykernel, t=2))
    local = recon.reconstruct(data, target=lambda omega: omega**2 * np.exp(-2 * omega))
    np.testing.assert_array_equal(fixed.values, local.values)


@pytest.mark.parametrize("policy", [sp.numerics.numpy(), sp.numerics.mpmath(45)])
def test_explicit_adapter_and_partial_periodic_function(policy):
    recon, data = setup(policy, periodic=True)
    target = sp.targets.from_function(
        partial(periodic_kernel, extent=16),
        spectral_power=2,
        breakpoints=(".5",),
    )
    actual = recon.reconstruct(data, target=target, target_times=[2])
    tolerance = 1e-12 if policy.backend == "numpy" else mp.mpf("1e-33")
    assert abs(actual.at_lambda(0).at_time(2).value - 0.5) < tolerance
    stored = actual.at_lambda(0).at_time(2).target_kernel
    with policy.context() as ctx:
        omega = ctx.number(".7")
        expected = omega**2 * (ctx.exp(-2 * omega) + ctx.exp(-14 * omega))
        assert abs(stored(omega) - expected) < tolerance


def test_binding_converts_time_to_backend_precision():
    target = sp.targets.from_function(precise_kernel).at_time(
        "2.00000000000000000000001"
    )
    with sp.numerics.mpmath(60).context() as ctx:
        expected = ctx.exp(-ctx.number("2.00000000000000000000001"))
        assert target.scalar(ctx.number(1), ctx) == expected
    assert not target.is_time_dependent
    with pytest.raises(ValueError, match="fixed time"):
        target.at_time(3)


def test_functions_work_inside_cut_grids_and_as_systematic_references():
    recon, data = setup()
    targets = sp.targets.grid(
        sp.targets.smooth_high_energy,
        vary={"width": [".05", ".1"]},
        kernel=mykernel,
        cutoff=".7",
    )
    results = recon.reconstruct(data, target=targets, target_times=[1, 2])
    assert results.shape == (2, 2, 2)
    model = sp.models.TabulatedSpectralDensity((".2", ".5", 1), (1, 1, 1))
    report = sp.systematics.estimate(
        results,
        reference_kernel=mykernel,
        density=model,
        energies=[".2", ".5", 1],
    )
    expected = sp.systematics.estimate(
        results,
        reference_kernel=sp.targets.correlator(basis=recon.basis),
        density=model,
        energies=[".2", ".5", 1],
    )
    assert report.biases.shape == (1, 8)
    np.testing.assert_allclose(
        report.biases.astype(float),
        expected.biases.astype(float),
        rtol=1e-12,
        atol=1e-14,
    )


def test_named_plain_functions_are_adapted():
    recon, data = setup()
    result = recon.reconstruct(
        data,
        target=sp.targets.named({"numpy": mykernel, "context": precise_kernel}),
        target_times=[2],
    )
    np.testing.assert_allclose(
        result.select(kernel="numpy").values.astype(float),
        result.select(kernel="context").values.astype(float),
    )


def test_time_requirements_and_unit_checks_remain_explicit():
    recon, data = setup()
    with pytest.raises(ValueError, match="target_times"):
        recon.reconstruct(data, target=mykernel)
    with pytest.raises(ValueError, match="target_times"):
        recon.reconstruct(data, target=fixed_kernel, target_times=[1])
    target = sp.targets.from_function(fixed_kernel, spectral_power=0)
    with pytest.raises(sp.ReconstructionError, match="spectral powers"):
        recon.reconstruct(data, target=target)
    with pytest.raises(ValueError, match="target_times"):
        sp.targets.from_function(mykernel)(1)
    dimensional = replace(
        recon,
        basis=replace(recon.basis, kernel=sp.kernels.exponential(energy_unit="GeV")),
    )
    with pytest.raises(sp.ReconstructionError, match="energy units"):
        dimensional.reconstruct(
            data,
            target=sp.targets.from_function(fixed_kernel, energy_unit="lattice"),
        )


def test_multi_ensemble_scanner_accepts_plain_functions():
    recon, data = setup()
    results = sp.scan.reconstruct(
        ensembles={
            "A": sp.scan.Ensemble(data, recon.basis),
            "B": sp.scan.Ensemble(data.scaled(2), recon.basis),
        },
        algorithm=recon.algorithm,
        target=mykernel,
        target_times=[1, 2],
    )
    assert results.shape == (2, 2, 2)
    np.testing.assert_allclose(
        results.select(ensemble="B").values.astype(float),
        2 * results.select(ensemble="A").values.astype(float),
    )


@pytest.mark.parametrize(
    "function",
    [
        lambda: 1,
        lambda omega, t, extra: 1,
        lambda *args: 1,
        lambda omega, **kwargs: 1,
        lambda omega, *, missing: 1,
    ],
)
def test_unsupported_signatures_fail_before_computation(function):
    with pytest.raises(TypeError, match="function targets require"):
        sp.targets.from_function(function)


def test_user_errors_are_not_retried_as_other_signatures():
    recon, data = setup()
    result = recon.reconstruct(
        data, target=bad_kernel, target_times=[1], on_error="record"
    )
    assert len(result.failures) == 2
    assert all("failure inside user function" in f.error for f in result.failures)
    with pytest.raises(sp.ReconstructionError, match="non-finite function kernel"):
        recon.reconstruct(data, target=nonfinite_kernel)


@pytest.mark.parametrize("policy", [sp.numerics.numpy(), sp.numerics.mpmath(40)])
def test_importable_functions_support_process_execution(policy):
    recon, data = setup(policy)
    serial = recon.reconstruct(data, target=precise_kernel, target_times=[1, 2])
    with sp.execution.processes(2) as execution:
        parallel = recon.reconstruct(
            data,
            target=precise_kernel,
            target_times=[1, 2],
            execution=execution,
        )
    np.testing.assert_array_equal(parallel.coefficients, serial.coefficients)


def test_plain_functions_remain_explicitly_unsupported_by_native_archives(tmp_path):
    recon, data = setup()
    result = recon.reconstruct(data, target=fixed_kernel)
    with pytest.raises(TypeError, match="registered archive type"):
        sp.io.save(tmp_path / "custom.spectral", results=result)
