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

"""Multiple point constraints, checked against an independent minimization."""
from dataclasses import replace
import csv
import io
import json

import numpy as np
import pytest
from mpmath import mp
from scipy.linalg import null_space
from scipy.integrate import trapezoid

from test_endpoint_constraint import problem
from hlt.solver import solve_hlt_endpoint_constrained


@pytest.mark.parametrize('backend', ['numpy', 'mpmath'])
@pytest.mark.parametrize('area', [True, False])
@pytest.mark.parametrize('periodic', [True, False])
@pytest.mark.parametrize('endpoint', [None, 0., 'target'])
def test_multiple_points_independent_minimum(backend, area, periodic, endpoint):
    recon, target = problem(backend, area=area, periodic=periodic, endpoint=endpoint)
    recon = replace(recon, omega_constraints=[[.6, 0.], [.7, .001]])
    raw = recon.reconstruct(target, compute_mismatch=False)
    result = raw.to_numpy()
    basis = recon.basis
    columns = [result.R] if area else []
    values = [result.c] if area else []
    if endpoint is not None:
        columns.append(np.array([basis.phi(t, basis.omega0) for t in basis.times]))
        values.append(result.endpoint_value)
    for w, value in result.omega_constraints:
        columns.append(np.array([basis.phi(t, w) for t in basis.times]))
        values.append(value)
    C = np.column_stack(columns)
    q0 = np.linalg.lstsq(C.T, values, rcond=None)[0]
    Z = null_space(C.T)
    f = (1-recon.lambda_hlt) * result.b
    expected = q0 + Z @ np.linalg.solve(Z.T @ result.W @ Z, Z.T @ (f-result.W @ q0))
    np.testing.assert_allclose(result.q, expected, rtol=2e-8, atol=3e-9)
    np.testing.assert_allclose(result.kernel_values([.6, .7]), [0., .001], atol=2e-12)
    assert raw.checks.ok
    assert len(raw.checks.omega_abs) == 2
    assert max(result.checks.omega_abs) < 2e-12
    _, mean, cov = recon.selected_data()
    np.testing.assert_allclose(result.estimate, result.q @ mean, atol=1e-12)
    np.testing.assert_allclose(result.variance, result.q @ cov @ result.q, rtol=1e-12)
    if backend == 'mpmath':
        with mp.workdps(50):
            assert max(raw.checks.omega_abs) < mp.mpf('1e-43')
            assert all(hasattr(w, '_mpf_') for w, _ in raw.omega_constraints)


@pytest.mark.parametrize('endpoint', [None, 0.])
@pytest.mark.parametrize('area', [True, False])
def test_prepared_scan_and_basis_reuse(endpoint, area):
    recon, target = problem('mpmath', area=area, endpoint=endpoint)
    old = recon.prepare(target, compute_mismatch=False)
    recon = replace(recon, omega_constraints=[['0.6', '0'], ['0.7', '0']])
    prepared = recon.prepare(target, reuse_basis_from=old)
    results = prepared.reconstruct_many(['1e-3', '1e-6'])
    for lam, raw in zip(['1e-3', '1e-6'], results):
        direct = replace(recon, lambda_hlt=lam).reconstruct(target)
        with mp.workdps(50):
            assert max(abs(x) for x in raw.kernel_values(['0.6', '0.7'])) < mp.mpf('1e-42')
            assert mp.norm(raw.q - direct.q) < mp.mpf('1e-42')
        assert raw.checks.ok


@pytest.mark.parametrize('backend', ['numpy', 'mpmath'])
def test_duplicates_and_endpoint_overlap(backend):
    recon, target = problem(backend)
    recon = replace(recon, omega_constraints=[['0.2', 0], [.6, 0], [.6, 0]])
    result = recon.reconstruct(target, compute_mismatch=False)
    assert len(result.omega_constraints) == 2
    assert result.checks.ok
    with pytest.raises(ValueError, match='conflict'):
        replace(recon, omega_constraints=[['0.2', 1]]).reconstruct(target, compute_mismatch=False)
    with pytest.raises(ValueError, match='conflict'):
        replace(recon, omega_constraints=[[.6, 0], [.6, 1]])


@pytest.mark.parametrize('backend', ['numpy', 'mpmath'])
def test_zero_basis_and_systematics(backend):
    recon, target = problem(backend, endpoint=None)
    recon = replace(recon, omega_constraints=[[0, 0], [.6, 0], [.7, 0]])
    raw = recon.reconstruct(target, compute_mismatch=False)
    assert raw.checks.omega_abs[0] == 0
    w = np.linspace(.2, 5, 60)
    rho = np.exp(-w)
    grid = raw.kernel_mismatch_on_grid(w, rho).to_numpy()
    syst = raw.estimate_kernel_systematic_on_grid(w, rho).to_float()
    np.testing.assert_allclose(syst.signed_bias, trapezoid(grid.weighted_mismatch, x=w), rtol=1e-10)
    with pytest.raises(ValueError, match='impossible'):
        replace(recon, omega_constraints=[[0, 1]]).reconstruct(target, compute_mismatch=False)


@pytest.mark.parametrize('bad', [[[.6]], [[.6, 0, 1]], [0.6, 0], [[-.1, 0]], [[float('nan'), 0]], [[.6, float('inf')]], [['target', 0]]])
def test_invalid_pairs(bad):
    recon, _ = problem()
    with pytest.raises(ValueError, match='omega_constraints'):
        replace(recon, omega_constraints=bad)


@pytest.mark.parametrize('backend', ['numpy', 'mpmath'])
def test_too_many_independent_conditions(backend):
    recon, target = problem(backend, endpoint=None)
    with pytest.raises(ValueError, match='independent'):
        replace(recon, omega_constraints=[[w, 0] for w in [.3, .4, .5, .6, .7]]).reconstruct(target, compute_mismatch=False)
    # Three linearly dependent but distinct columns: not just an identical pair.
    with pytest.raises(ValueError, match='independent'):
        solve_hlt_endpoint_constrained(np.eye(3), [1, 1, 1], None, None,
            omega_vectors=([1,0,0], [0,1,0], [1,1,0]), omega_values=(0,0,0), backend=backend)


@pytest.mark.parametrize('backend', ['numpy', 'mpmath'])
def test_empty_constraints_preserve_solution(backend):
    recon, target = problem(backend)
    expected = recon.reconstruct(target, compute_mismatch=False)
    for empty in [None, [], ()]:
        result = replace(recon, omega_constraints=empty).reconstruct(target, compute_mismatch=False)
        np.testing.assert_array_equal(result.to_numpy().q, expected.to_numpy().q)
        assert result.checks.omega_abs == ()


def test_mp_constraint_precision():
    recon, target = problem('mpmath', endpoint=None)
    w, d = '0.612345678901234567890123456789', '0.0012345678901234567890123456789'
    raw = replace(recon, omega_constraints=[[w,d]]).reconstruct(target, compute_mismatch=False)
    with mp.workdps(50):
        assert raw.omega_constraints[0] == (mp.mpf(w), mp.mpf(d))
        assert abs(raw.kernel_values([w])[0]-mp.mpf(d)) < mp.mpf('1e-45')
