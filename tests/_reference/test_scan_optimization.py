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
import pytest
from mpmath import mp

from hlt import CorrelatorData, HLTBasis, HLTReconstructor, IntegratedHVPTarget
from hlt.mpmath_backend import (
    MPWorkspace,
    _hvp_kernel_K,
    reconstructed_kernel,
    target_eval,
)
from hlt.targets import _hvp_kernel_K_analytic_float


def _full_target():
    return IntegratedHVPTarget.from_high_energy(
        am_tau="0.7",
        delta="0.08",
        p="4",
        euclidean_window="FULL",
        am_mu="0.06",
    )


@pytest.mark.parametrize("r_source", ["1e-8", "0.1", "1", "3.9", "4", "4.1", "10", "1e4"])
def test_analytic_dispersive_kernel_reproduces_numeric_x_integral(r_source):
    target = _full_target()
    with mp.workdps(70):
        r = mp.mpf(r_source)
        omega = mp.sqrt(r) * mp.mpf("0.06")
        analytic = _hvp_kernel_K(target, omega, mp, MPWorkspace(70))
        numeric = mp.quad(
            lambda x: x * x * (1 - x) / (x * x + (1 - x) * r),
            [0, 1],
        )
        assert abs(analytic - numeric) <= mp.mpf("1e-58") * abs(numeric)


@pytest.mark.parametrize("r", [1e-5, 0.1, 1.0, 3.9, 4.0, 4.1, 10.0, 1e4])
def test_float_analytic_dispersive_kernel_matches_mp(r):
    with mp.workdps(60):
        omega = np.sqrt(r) * 0.06
        approximate = mp.mpf(_hvp_kernel_K_analytic_float(omega, 0.06))
        exact = mp.quad(
            lambda x: x * x * (1 - x) / (x * x + (1 - x) * mp.mpf(str(r))),
            [0, 1],
        )
        assert abs(approximate - exact) <= mp.mpf("2e-12") * abs(exact)


def _prepared_problem():
    times = np.arange(6, dtype=float)
    data = CorrelatorData.from_arrays(
        mean=np.exp(-0.2 * times),
        covariance=np.eye(6) * 1e-4,
        times=times,
    )
    basis = HLTBasis(times=[1, 2, 3, 4, 5], p="2", omega0="0.1")
    target = IntegratedHVPTarget.from_high_energy(
        am_tau="0.7",
        delta="0.08",
        p="2",
        euclidean_window="FULL",
        am_mu="0.06",
    )
    reconstructor = HLTReconstructor(
        data=data,
        basis=basis,
        lambda_hlt="1e-8",
        regulator="identity",
        solver_backend="mpmath",
        mp_dps=40,
        strict=False,
    )
    return reconstructor, target


def test_prepared_scan_matches_independent_reconstructions():
    reconstructor, target = _prepared_problem()
    lambdas = ["1e-6", "1e-8", "1e-10"]
    prepared = reconstructor.prepare(target, compute_mismatch=True)
    scan_results = prepared.reconstruct_many(lambdas, strict=False)

    for lambda_source, scan_result in zip(lambdas, scan_results):
        independent = HLTReconstructor(
            data=reconstructor.data,
            basis=reconstructor.basis,
            lambda_hlt=lambda_source,
            regulator="identity",
            solver_backend="mpmath",
            mp_dps=40,
            strict=False,
        ).reconstruct(target, compute_mismatch=True)
        with mp.workdps(40):
            assert abs(scan_result.estimate - independent.estimate) < mp.mpf("1e-36")
            assert abs(scan_result.error - independent.error) < mp.mpf("1e-36")
            assert abs(
                scan_result.checks.mismatch_abs - independent.checks.mismatch_abs
            ) < mp.mpf("1e-34")


def test_identity_solution_can_be_shared_across_correlator_discretizations():
    reconstructor, target = _prepared_problem()
    alternate = CorrelatorData.from_arrays(
        mean=np.linspace(1.0, 0.4, 6),
        covariance=np.eye(6) * 2e-4,
        times=np.arange(6, dtype=float),
    )
    solution = reconstructor.prepare(target, compute_mismatch=False).solve("1e-9")
    first = solution.evaluate(reconstructor.data, strict=False)
    second = solution.evaluate(alternate, strict=False)
    assert first.q is second.q
    assert first.estimate != second.estimate
    assert first.error != second.error


def test_multiple_targets_can_share_prepared_basis_matrices():
    reconstructor, target = _prepared_problem()
    first = reconstructor.prepare(target, compute_mismatch=False)
    second_target = IntegratedHVPTarget.from_high_energy(
        am_tau="0.8",
        delta="0.08",
        p="2",
        euclidean_window="FULL",
        am_mu="0.06",
    )
    second = reconstructor.prepare(
        second_target, compute_mismatch=False, reuse_basis_from=first
    )
    assert second.A is first.A
    assert second.R is first.R
    assert second.b is not first.b
    assert second.shared_basis_cache is first.shared_basis_cache


def test_quadratic_mismatch_matches_direct_mp_integration():
    reconstructor, target = _prepared_problem()
    prepared = reconstructor.prepare(target, compute_mismatch=True)
    solution = prepared.solve("1e-8", compute_condition_number=False)
    with mp.workdps(40):
        workspace = MPWorkspace(40)
        direct = mp.quad(
            lambda omega: (
                reconstructed_kernel(reconstructor.basis, solution.q, omega, mp)
                - target_eval(target, omega, mp, workspace)
            )
            ** 2,
            [mp.mpf("0.1"), mp.mpf("0.5"), mp.mpf("0.7"), mp.mpf("0.9"), mp.inf],
        )
        assert abs(solution.mismatch_abs**2 - direct) < mp.mpf("1e-28")


def test_full_target_records_no_inner_kernel_quadrature():
    target = _full_target()
    workspace = MPWorkspace(40)
    with mp.workdps(40):
        value = _hvp_kernel_K(target, mp.mpf("0.8"), mp, workspace)
    assert value > 0
    assert "hvp_kernel" not in workspace.quadrature_errors
