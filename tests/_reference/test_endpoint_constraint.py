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

"""Endpoint constraints: independent minimizer, estimator and scan checks."""
from dataclasses import replace

import numpy as np
import pytest
from mpmath import mp
from scipy.integrate import trapezoid
from scipy.linalg import null_space

from hlt import CorrelatorData, HLTBasis, HLTReconstructor, HighEnergyTarget, IntegratedHVPTarget
from hlt.mpmath_backend import MPWorkspace, target_eval
from hlt.solver import solve_hlt_endpoint_constrained


def problem(backend="numpy", area=True, periodic=True, regulator="identity", endpoint=0.0):
    times = np.arange(7)
    data = CorrelatorData.from_arrays(
        mean=np.exp(-0.4 * times), covariance=np.eye(7) * 1e-4,
        times=times, T=16 if periodic else None,
    )
    basis = HLTBasis(times=[1, 2, 3, 4, 5], p=2, omega0="0.2",
                     periodic=periodic, T=16 if periodic else None)
    target = HighEnergyTarget(t0=2.5, am_tau=0.7, delta=0.1, p=2,
                              periodic=periodic, T=16 if periodic else None)
    recon = HLTReconstructor(
        data=data, basis=basis, lambda_hlt=1e-4, regulator=regulator,
        area_constraint=area, endpoint_value=endpoint,
        solver_backend=backend, target_integration_backend="numpy",
        mp_dps=50, strict=True,
    )
    return recon, target


@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
@pytest.mark.parametrize("area", [True, False])
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("regulator", ["identity", "identity_scaled", "covariance"])
def test_endpoint_and_independent_nullspace_minimizer(backend, area, periodic, regulator):
    recon, target = problem(backend, area, periodic, regulator)
    raw = recon.reconstruct(target, compute_mismatch=False)
    result = raw.to_numpy()
    v = np.array([recon.basis.phi(t, recon.basis.omega0) for t in recon.basis.times])
    C = np.column_stack([result.R, v]) if area else v[:, None]
    d = np.array([result.c, 0.]) if area else np.array([0.])
    # Independent reference: eliminate the constraints, then minimize in the
    # null space. This does not use the production KKT implementation.
    q0 = np.linalg.lstsq(C.T, d, rcond=None)[0]
    Z = null_space(C.T)
    f = (1 - recon.lambda_hlt) * result.b
    expected = q0 + Z @ np.linalg.solve(Z.T @ result.W @ Z, Z.T @ (f - result.W @ q0))
    np.testing.assert_allclose(result.q, expected, rtol=1e-8, atol=2e-10)
    assert abs(v @ result.q) < 1e-13
    if area:
        assert abs(result.R @ result.q - result.c) < 1e-12
    _, mean, covariance = recon.selected_data()
    # Contraction after downconversion can lose digits through cancellation.
    np.testing.assert_allclose(result.estimate, result.q @ mean, rtol=1e-12, atol=5e-15)
    np.testing.assert_allclose(result.variance, result.q @ covariance @ result.q, rtol=1e-12)
    assert raw.checks.ok
    assert result.checks.endpoint_abs < 1e-13
    assert result.checks.stationarity_rel < 1e-8
    if backend == "mpmath":
        assert hasattr(raw.endpoint_value, "_mpf_")
        with mp.workdps(50):
            assert abs(raw.kernel_values(["0.2"])[0]) < mp.mpf("1e-45")


@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
@pytest.mark.parametrize("value", ["target", "0.012345678901234567890123456789"])
def test_nonzero_endpoint_and_systematics(backend, value):
    recon, target = problem(backend, endpoint=value)
    result = recon.reconstruct(target, compute_mismatch=False)
    if backend == "mpmath":
        with mp.workdps(50):
            expected = mp.mpf(float(target.eval(0.2))) if value == "target" else mp.mpf(value)
            assert abs(result.kernel_values(["0.2"])[0] - expected) < mp.mpf("1e-45")
    else:
        expected = float(target.eval(0.2)) if value == "target" else float(value)
        assert abs(result.kernel_values(np.array([0.2]))[0] - expected) < 1e-13
    w = np.linspace(.2, 5, 81)
    rho = np.exp(-w)
    grid = result.kernel_mismatch_on_grid(w, rho_p=rho).to_numpy()
    syst = result.estimate_kernel_systematic_on_grid(w, rho).to_float()
    np.testing.assert_allclose(syst.signed_bias, trapezoid(grid.weighted_mismatch, x=w),
                               rtol=1e-11, atol=1e-15)


@pytest.mark.parametrize("area", [True, False])
def test_prepared_scan_keeps_endpoint_constraint(area):
    recon, target = problem("mpmath", area=area)
    prepared = recon.prepare(target, compute_mismatch=True)
    results = prepared.reconstruct_many(["1e-3", "1e-5", "1e-8"])
    for lam, result in zip(["1e-3", "1e-5", "1e-8"], results):
        independent = replace(recon, lambda_hlt=lam).reconstruct(target)
        with mp.workdps(50):
            assert abs(result.kernel_values(["0.2"])[0]) < mp.mpf("1e-42")
            assert abs(result.estimate - independent.estimate) < mp.mpf("1e-42")
            assert result.checks.mismatch_abs == independent.checks.mismatch_abs
        assert result.checks.ok


def test_mp_target_matching_uses_mp_target_evaluation():
    recon, target = problem("mpmath", endpoint="target")
    recon = replace(recon, target_integration_backend="mpmath")
    result = recon.reconstruct(target, compute_mismatch=False)
    with mp.workdps(50):
        exact = target_eval(target, mp.mpf("0.2"), mp, MPWorkspace(50))
        assert abs(result.endpoint_value - exact) < mp.mpf("1e-45")
        assert abs(result.kernel_values(["0.2"])[0] - exact) < mp.mpf("1e-45")


@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
def test_redundant_zero_endpoint_and_impossible_nonzero(backend):
    recon, target = problem(backend)
    recon = replace(recon, basis=replace(recon.basis, omega0=0.0))
    result = recon.reconstruct(target, compute_mismatch=False)
    baseline = replace(recon, endpoint_value=None).reconstruct(target, compute_mismatch=False)
    np.testing.assert_array_equal(result.to_numpy().q, baseline.to_numpy().q)
    assert result.checks.endpoint_abs == 0
    with pytest.raises(ValueError, match="impossible"):
        replace(recon, endpoint_value=1.).reconstruct(target, compute_mismatch=False)


@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
def test_incompatible_constraints_rejected(backend):
    with pytest.raises(ValueError, match="independent"):
        solve_hlt_endpoint_constrained([[1]], [1], [1], 0, R=[1], c=1, backend=backend)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "zero", "nonsense"])
def test_invalid_endpoint_value(bad):
    with pytest.raises(ValueError, match="endpoint_value"):
        problem(endpoint=bad)


def test_hvp_target_endpoint_constraint():
    recon, _ = problem(endpoint=0.)
    target = IntegratedHVPTarget.from_high_energy(
        am_tau=0.7, delta=0.08, p=2, euclidean_window="FULL", am_mu=0.06)
    result = recon.reconstruct(target, compute_mismatch=False)
    assert abs(result.kernel_values(np.array([.2]))[0]) < 1e-18
    assert result.checks.ok


def test_default_is_unchanged():
    from hlt.solver import solve_hlt_constrained
    recon, target = problem(endpoint=None)
    result = recon.reconstruct(target, compute_mismatch=False)
    old = solve_hlt_constrained(result.W, (1-recon.lambda_hlt)*result.b, result.R, result.c)
    np.testing.assert_array_equal(result.q, old.q)
    assert result.endpoint_value is None
    assert result.checks.endpoint_abs is None
