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

"""Input parameters for the multi-ensemble HLT scan.

Edit this file to add ensembles or change scan ranges.  The runner imports
only these configuration objects; all scan logic lives in
`run_multi_ensemble_hlt_scan.py`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnsembleConfig:
    name: str
    a_lat_inv_MeV: float
    m_pi_MeV: float
    ZV: float
    T: int
    L_lattice: int
    basis_t_min: int
    basis_t_max: int
    basis_stride: int
    nrec: int
    t_rec: int
    discretization: str
    jks_filename: str | None = None

    def resolved_jks_filename(self) -> str:
        if self.jks_filename is not None:
            return self.jks_filename
        return f"{self.name}.jks"


@dataclass(frozen=True)
class GlobalHLTConfig:
    # Exponent in the HLT basis phi_t^(p)(omega).
    # The scanner converts rho_2 -> rho_p consistently using rho_p = rho_2/omega^(p-2).
    p: float = 4.0

    # Use "identity_scaled" for lambda scans that are more comparable across p.
    regulator: str = "identity"
    solver_backend: str = "mpmath"
    mp_dps: int = 30
    # Kernel constraints in lattice energy units; each pair is (omega, value).
    # Example: omega_constraints=((0.6, 0.0), (0.7, 0.0))
    omega_constraints: tuple[tuple[float, float], ...] = ()
    endpoint_value: float | str | None = None
    strict: bool = True
    compute_mismatch: bool = True

    # MP quadrature controls.  Keep these at the converged values established
    # for the smallest lambda in the scan.
    mp_quadrature_method: str = "gauss-legendre"
    mp_quadrature_maxdegree: int = 8
    mp_window_quadrature_method: str = "gauss-legendre"
    mp_window_quadrature_maxdegree: int = 8

    # Direct IntegratedHVPTarget settings.
    # window_delta is computed ensemble-by-ensemble in lattice units from the
    # standard 0.15 fm window width and passed explicitly to the target.
    # Retained for API compatibility; K(s) is analytic in genericHLT >= 0.6.
    kernel_quadrature_order: int = 96


@dataclass(frozen=True)
class ScanConfig:
    # Decimal strings reach mpmath without a binary64 rounding step.
    lambda_array: tuple[str, ...] = (
        "1e-15",
        "1e-14",
        "1e-13",
        "1e-12",
        "1e-11",
        "1e-10",
        "1e-9",
        "1e-8",


    )
    delta_array: tuple[float, ...] = (1,)
    time_like_cut_MeV_array: tuple[float, ...] = (
        1776.86,
    )
    kernel_names: tuple[str, ...] = ("FULL",)#("FULL", "ID", "LD", "SD")


@dataclass(frozen=True)
class SystematicConfig:
    r_ratio_filename: str = "r_ratio_pdg.csv"
    r_ratio_max_points: int = 1000
    # Uses rho_2 <- R/(12*pi^2) * 2/ZV**2 * systblind.
    systblind: float = 2.0

    # Gaussian-random R-ratio distortions.
    #
    # Each realization uses
    #     R_noise(E) = R(E) * [1 + sigma_R * eta(E)],
    # where eta has unit R-weighted RMS after optional energy smoothing.
    r_ratio_noise_n_samples: int = 100

    # RMS fractional size of the random distortion. 0.05 means a 5% RMS
    # deformation of R(E).
    r_ratio_noise_relative_sigma: float = 0.05

    # Gaussian energy-correlation length. Set to 0.0 for independent noise at
    # each R-ratio grid point. Positive values produce smooth random shapes.
    r_ratio_noise_correlation_length_MeV: float = 100.0

    # Fixed seed makes all scan points use the same reproducible realizations.
    r_ratio_noise_random_seed: int = 12345

    # Remove the R-weighted mean of each random field so that
    # integral R_noise(E) dE = integral R(E) dE numerically.
    r_ratio_noise_preserve_integral: bool = True

    # Reject random realizations that make R_noise(E) negative.
    r_ratio_noise_require_positive: bool = True


ENSEMBLES: tuple[EnsembleConfig, ...] = (
    EnsembleConfig(
        name="96I",
        a_lat_inv_MeV=2630,
        m_pi_MeV=131.29,
        ZV= 0.760,
        T=192,
        L_lattice=96,
        basis_t_min=1,
        basis_t_max=48,
        basis_stride=1,
        nrec=10,
        discretization="ll",
        t_rec=26,
    ),
    EnsembleConfig(
        name="96I",
        a_lat_inv_MeV=2630,
        m_pi_MeV=131.29,
        ZV= 0.760,
        T=192,
        L_lattice=96,
        basis_t_min=1,
        basis_t_max=48,
        basis_stride=1,
        nrec=10,
        discretization="lc",
        t_rec=26,
    ),
    EnsembleConfig(
        name="Ca",
        a_lat_inv_MeV=1731.2,
        m_pi_MeV=139.32,
        ZV=0.711,
        T=128,
        L_lattice=64,
        basis_t_min=1,
        basis_t_max=32,
        basis_stride=1,
        nrec=10,
        discretization="ll",
        t_rec=18,
    ),
    EnsembleConfig(
        name="Ca",
        a_lat_inv_MeV=1731.2,
        m_pi_MeV=139.32,
        ZV=0.711,
        T=128,
        L_lattice=64,
        basis_t_min=1,
        basis_t_max=32,
        basis_stride=1,
        nrec=10,
        discretization="lc",
        t_rec=18,
    ),
    EnsembleConfig(
        name="48I",
        a_lat_inv_MeV=1731.2,
        m_pi_MeV=139.32,
        ZV=0.711,
        T=96,
        L_lattice=48,
        basis_t_min=1,
        basis_t_max=48,
        basis_stride=1,
        nrec=5,
        discretization="ll",
        t_rec=16,
    ),
    EnsembleConfig(
        name="48I",
        a_lat_inv_MeV=1731.2,
        m_pi_MeV=139.32,
        ZV=0.711,
        T=96,
        L_lattice=48,
        basis_t_min=1,
        basis_t_max=48,
        basis_stride=1,
        nrec=5,
        discretization="lc",
        t_rec=16,
    ),
    EnsembleConfig(
        name="64I",
        a_lat_inv_MeV=2354,
        m_pi_MeV=138.98,
        ZV=0.743,
        T=128,
        L_lattice=64,
        basis_t_min=1,
        basis_t_max=48,
        basis_stride=1,
        nrec=5,
        discretization="lc",
        t_rec=20,
    ),
    EnsembleConfig(
        name="64I",
        a_lat_inv_MeV=2354,
        m_pi_MeV=138.98,
        ZV=0.743,
        T=128,
        L_lattice=64,
        basis_t_min=1,
        basis_t_max=48,
        basis_stride=1,
        nrec=5,
        discretization="ll",
        t_rec=20,
    ),

)

GLOBAL_HLT = GlobalHLTConfig()
SCAN = ScanConfig()
SYSTEMATIC = SystematicConfig()
