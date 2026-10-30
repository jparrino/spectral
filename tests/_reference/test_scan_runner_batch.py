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

import importlib
import sys
import types
from dataclasses import replace

import numpy as np

sys.modules.setdefault("jks", types.ModuleType("jks"))
scan_runner = importlib.import_module("run_multi_ensemble_hlt_scan")

from hlt import CorrelatorData, HLTBasis, HLTReconstructor, IntegratedHVPTarget
from scan_inputs import (
    ENSEMBLES,
    GLOBAL_HLT,
    SCAN,
    EnsembleConfig,
    GlobalHLTConfig,
    ScanConfig,
    SystematicConfig,
)


def test_attached_scan_collapses_64_rows_to_four_target_groups():
    points = scan_runner.make_scan_points(ENSEMBLES, SCAN)
    groups = scan_runner.make_scan_groups(points, ENSEMBLES, GLOBAL_HLT)
    assert len(points) == 64
    assert len(groups) == 4
    assert all(len(group) == 16 for group in groups)

    expanded = replace(
        SCAN,
        time_like_cut_MeV_array=(1200.0, 1776.86),
        kernel_names=("FULL", "ID"),
    )
    expanded_points = scan_runner.make_scan_points(ENSEMBLES, expanded)
    expanded_groups = scan_runner.make_scan_groups(
        expanded_points, ENSEMBLES, GLOBAL_HLT
    )
    assert len(expanded_points) == 256
    assert len(expanded_groups) == 4


def test_batched_systematic_contractions_match_scalar_formulas():
    omega = np.asarray([0.1, 0.2, 0.5, 0.9])
    weights = scan_runner._trapezoidal_point_weights(omega)
    reference = np.asarray([1.0, 0.8, 0.4, 0.2])
    reconstructed = np.asarray([0.9, 0.75, 0.45, 0.22])
    rho = np.asarray(
        [[1.0, 0.9, 0.7, 0.4], [1.1, 0.8, 0.75, 0.35]], dtype=float
    )
    batched = scan_runner._systematic_metrics_batch(
        reference=reference,
        reconstructed=reconstructed,
        rho=rho,
        weights=weights,
    )
    for index, spectrum in enumerate(rho):
        difference = reconstructed - reference
        target = np.sum(weights * reference * spectrum)
        rebuilt = np.sum(weights * reconstructed * spectrum)
        signed = np.sum(weights * difference * spectrum)
        absolute = np.sum(weights * np.abs(difference) * np.abs(spectrum))
        np.testing.assert_allclose(batched["target_observable"][index], target)
        np.testing.assert_allclose(batched["reconstructed_observable"][index], rebuilt)
        np.testing.assert_allclose(batched["signed"][index], signed)
        np.testing.assert_allclose(batched["absolute"][index], absolute)


def test_shared_mp_grid_matches_individual_result_evaluations():
    data = CorrelatorData.from_arrays(
        mean=[1.0, 0.8, 0.6, 0.4],
        covariance=np.eye(4) * 1e-4,
        times=[0, 1, 2, 3],
    )
    basis = HLTBasis(times=[1, 2, 3], p="2", omega0="0.1")
    target = IntegratedHVPTarget.from_high_energy(
        am_tau="0.5", delta="0.1", p="2", euclidean_window="FULL", am_mu="0.06"
    )
    sharp = IntegratedHVPTarget.from_sharp_high_energy(
        am_tau="0.5", p="2", euclidean_window="FULL", am_mu="0.06"
    )
    reconstructor = HLTReconstructor(
        data=data,
        basis=basis,
        lambda_hlt="1e-6",
        regulator="identity",
        solver_backend="mpmath",
        mp_dps=35,
        strict=False,
    )
    prepared = reconstructor.prepare(target, compute_mismatch=False)
    solutions = prepared.solve_many(["1e-6", "1e-8"])
    omega = np.asarray([0.1, 0.2, 0.4, 0.8])
    shared = scan_runner.evaluate_scan_kernels_mpmath(
        prepared=prepared,
        solutions=solutions,
        sharp_target=sharp,
        omega=omega,
    )
    for column, solution in enumerate(solutions):
        result = solution.evaluate(data, strict=False)
        np.testing.assert_allclose(
            shared["reconstructed"][:, column],
            np.asarray([float(result.kernel_values(value)) for value in omega]),
            rtol=2e-15,
        )
        np.testing.assert_allclose(
            shared["smooth"],
            np.asarray([float(result.target_values(value)) for value in omega]),
            rtol=2e-15,
        )


def test_group_runner_matches_independent_lambda_reconstructions(monkeypatch):
    common = dict(
        name="toy",
        a_lat_inv_MeV=1700.0,
        m_pi_MeV=140.0,
        ZV=0.72,
        T=16,
        L_lattice=8,
        basis_t_min=1,
        basis_t_max=3,
        basis_stride=1,
        nrec=0,
        t_rec=4,
        jks_filename="toy.jks",
    )
    ensembles = (
        EnsembleConfig(discretization="ll", **common),
        EnsembleConfig(discretization="lc", **common),
    )
    hlt = GlobalHLTConfig(
        p=2,
        regulator="identity",
        solver_backend="mpmath",
        mp_dps=30,
        strict=False,
        compute_mismatch=True,
    )
    scan = ScanConfig(
        lambda_array=("1e-6", "1e-8"),
        delta_array=(1.0,),
        time_like_cut_MeV_array=(800.0,),
        kernel_names=("FULL",),
    )
    systematic = SystematicConfig(
        r_ratio_noise_n_samples=3,
        r_ratio_max_points=20,
    )
    times = np.arange(4, dtype=float)
    data_by_disc = {
        "ll": CorrelatorData.from_arrays(
            [1.0, 0.8, 0.6, 0.45], np.eye(4) * 1e-4, times=times, T=16
        ),
        "lc": CorrelatorData.from_arrays(
            [1.0, 0.75, 0.55, 0.4], np.eye(4) * 2e-4, times=times, T=16
        ),
    }
    basis = HLTBasis(times=[1, 2, 3], p="2", T="16", periodic=True, omega0="0.1")
    omega = np.linspace(0.1, 1.5, 20)
    rho = np.exp(-omega)

    def fake_load_ensemble_objects(*, ens, **_kwargs):
        return {
            "data": data_by_disc[ens.discretization],
            "basis": basis,
            "omega0": 0.1,
            "omega_lat_cut": omega,
            "rho_p_cut": rho,
        }

    monkeypatch.setattr(
        scan_runner, "load_ensemble_objects", fake_load_ensemble_objects
    )
    points = scan_runner.make_scan_points(ensembles, scan)
    groups = scan_runner.make_scan_groups(points, ensembles, hlt)
    assert len(groups) == 1
    rows = scan_runner.run_scan_group(
        groups[0],
        ensembles=ensembles,
        hlt=hlt,
        scan=scan,
        syst_cfg=systematic,
        data_dir_str="unused",
    )
    assert all(row["status"] == "success" for row in rows)
    for row in rows:
        ens = ensembles[int(row["ensemble_index"])]
        derived = scan_runner.derived_parameters(ens)
        target = IntegratedHVPTarget.from_high_energy(
            am_tau=800.0 / ens.a_lat_inv_MeV,
            delta=derived["m_pi"],
            p=2,
            euclidean_window="FULL",
            window_delta=derived["window_delta"],
            a_lat_inv_MeV=ens.a_lat_inv_MeV,
        )
        independent = HLTReconstructor(
            data=data_by_disc[ens.discretization],
            basis=basis,
            lambda_hlt=str(row["lambda_hlt"]),
            regulator="identity",
            solver_backend="mpmath",
            mp_dps=30,
            strict=False,
        ).reconstruct(target, compute_mismatch=True)
        np.testing.assert_allclose(
            row["estimate_raw"], float(independent.estimate), rtol=2e-14
        )
        np.testing.assert_allclose(
            row["error_raw"], float(independent.error), rtol=2e-14
        )
        if int(row["scan_index"]) == 1:
            sharp = IntegratedHVPTarget.from_sharp_high_energy(
                am_tau=800.0 / ens.a_lat_inv_MeV,
                p=2,
                euclidean_window="FULL",
                window_delta=derived["window_delta"],
                a_lat_inv_MeV=ens.a_lat_inv_MeV,
            )
            expected_kernel = independent.estimate_kernel_systematic_on_grid(
                omega=omega, rho_p=rho
            )
            expected_total = independent.estimate_total_sharp_systematic_on_grid(
                omega=omega, rho_p=rho, sharp_target=sharp
            )
            np.testing.assert_allclose(
                row["syst_kernel_signed_raw"],
                float(expected_kernel.signed_bias),
                rtol=3e-14,
                atol=1e-18,
            )
            np.testing.assert_allclose(
                row["syst_total_signed_raw"],
                float(expected_total.signed_bias),
                rtol=3e-14,
                atol=1e-18,
            )
            expected_noise = (
                scan_runner.estimate_r_ratio_gaussian_noise_envelope(
                    int_result=independent,
                    sharp_target=sharp,
                    omega_lat_cut=omega,
                    rho_p_cut=rho,
                    n_samples=3,
                    relative_sigma=systematic.r_ratio_noise_relative_sigma,
                    correlation_length_MeV=(
                        systematic.r_ratio_noise_correlation_length_MeV
                    ),
                    random_seed=systematic.r_ratio_noise_random_seed,
                    preserve_integral=systematic.r_ratio_noise_preserve_integral,
                    require_positive=systematic.r_ratio_noise_require_positive,
                    a_lat_inv_MeV=ens.a_lat_inv_MeV,
                )
            )
            np.testing.assert_allclose(
                row["syst_kernel_noise_signed_min_raw"],
                expected_noise["syst_kernel_noise_signed_min"],
                rtol=3e-14,
                atol=1e-18,
            )
            np.testing.assert_allclose(
                row["syst_total_noise_signed_max_raw"],
                expected_noise["syst_total_noise_signed_max"],
                rtol=3e-14,
                atol=1e-18,
            )
