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

import csv
import io
import json
import pytest
import numpy as np
from test_scan_runner_batch import scan_runner
from hlt import CorrelatorData, HLTBasis, HLTReconstructor, IntegratedHVPTarget
from scan_inputs import EnsembleConfig, GlobalHLTConfig, ScanConfig, SystematicConfig

@pytest.mark.parametrize("endpoint", [None, 0.0])
@pytest.mark.parametrize("backend", ["numpy", "mpmath"])
@pytest.mark.parametrize("points_config", [(), ((0.6, 0.0), (0.7, 0.0))])
def test_scanner_point_constraints(monkeypatch, backend, points_config, endpoint):
    common = dict(
        name="toy",
        a_lat_inv_MeV=1700.0,
        m_pi_MeV=140.0,
        ZV=0.72,
        T=16,
        L_lattice=8,
        basis_t_min=1,
        basis_t_max=4,
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
        solver_backend=backend,
        omega_constraints=points_config,
        endpoint_value=endpoint,
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
    times = np.arange(5, dtype=float)
    data_by_disc = {
        "ll": CorrelatorData.from_arrays(
            [1.0, 0.8, 0.6, 0.45, 0.3], np.eye(5) * 1e-4, times=times, T=16
        ),
        "lc": CorrelatorData.from_arrays(
            [1.0, 0.75, 0.55, 0.4, 0.25], np.eye(5) * 2e-4, times=times, T=16
        ),
    }
    basis = HLTBasis(times=[1, 2, 3, 4], p="2", T="16", periodic=True, omega0="0.1")
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
    rows = (scan_runner.run_scan_group(
        groups[0],
        ensembles=ensembles,
        hlt=hlt,
        scan=scan,
        syst_cfg=systematic,
        data_dir_str="unused",
    ) if backend == "mpmath" else [scan_runner.run_one_scan_point(
        point, ensembles=ensembles, hlt=hlt, scan=scan, syst_cfg=systematic,
        data_dir_str="unused") for point in points])
    assert all(row["status"] == "success" for row in rows), rows
    fields = scan_runner.all_fieldnames(points[0], ensembles, hlt, scan, systematic)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    for row in rows:
        residuals = json.loads(row["omega_constraint_abs"])
        assert len(residuals) == len(points_config)
        assert all(abs(float(x)) < 1e-12 for x in residuals)
        assert len(json.loads(row["omega_constraints_config"])) == len(points_config)
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
            solver_backend=backend,
            omega_constraints=points_config,
            endpoint_value=endpoint,
            mp_dps=30,
            strict=False,
        ).reconstruct(target, compute_mismatch=True)
        np.testing.assert_allclose(
            row["estimate_raw"], float(independent.estimate), rtol=1e-9, atol=1e-15
        )
        np.testing.assert_allclose(
            row["error_raw"], float(independent.error), rtol=1e-9, atol=1e-15
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
                rtol=1e-9,
                atol=1e-18,
            )
            np.testing.assert_allclose(
                row["syst_total_signed_raw"],
                float(expected_total.signed_bias),
                rtol=1e-9,
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
                rtol=1e-9,
                atol=1e-18,
            )
            np.testing.assert_allclose(
                row["syst_total_noise_signed_max_raw"],
                expected_noise["syst_total_noise_signed_max"],
                rtol=1e-9,
                atol=1e-18,
            )
