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

"""Edit the ensembles and scan parameters here; endpoint matching defaults to zero."""

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
    discretization: str
    jks_filename: str | None = None

    def resolved_jks_filename(self):
        return self.jks_filename or f"{self.name}.jks"


@dataclass(frozen=True)
class GlobalHLTConfig:
    p: float = 4.0
    regulator: str = "identity"  # "identity" or "covariance"
    solver_backend: str = "numpy"
    mp_dps: int = 40
    mp_quadrature_maxdegree: int = 8
    # Constraint at each ensemble's omega0: None disables it, "target" matches
    # the smooth HVP target, or use a number/decimal string for a fixed value.
    endpoint_value: str | float | None = 0


@dataclass(frozen=True)
class ScanConfig:
    lambda_array: tuple[str, ...] = (
        "1e-14",
        "1e-13",
        "1e-12",
        "1e-11",
        "2e-11",
        "1e-10",
        "2e-10",
        "1e-9",
        "2e-9",
        "1e-8",
        "1e-7",
        "1e-6",
        "1e-5",
        "1e-4",
        "1e-3",
    )
    delta_array: tuple[float, ...] = (0.2, 0.5, 1, 2)
    time_like_cut_MeV_array: tuple[float, ...] = (1200, 1400, 1600, 1776.86, 2000)
    kernel_names: tuple[str, ...] = ("ID",)  # Also supports FULL, SD, LD.


@dataclass(frozen=True)
class SystematicConfig:
    r_ratio_filename: str = "r_ratio_pdg.csv"
    r_ratio_max_points: int = 1000
    systfac: float = 1.0
    r_ratio_noise_n_samples: int = 100  # Per correlation length.
    r_ratio_noise_relative_sigma: float = 0.05  # R-weighted fractional RMS.
    r_ratio_noise_correlation_lengths_MeV: tuple[float, ...] = (
        20,
        40,
        60,
        80,
        100,
        120,
        140,
        160,
        180,
        200,
    )
    r_ratio_noise_random_seed: int = 12345
    r_ratio_noise_preserve_integral: bool = False
    r_ratio_noise_require_positive: bool = True


# Positional fields follow EnsembleConfig above; ll and lc share one JKS file.
ENSEMBLES = (
    EnsembleConfig("48I", 1731.2, 139.32, 0.711, 96, 48, 1, 48, 1, 5, "ll"),
    EnsembleConfig("48I", 1731.2, 139.32, 0.711, 96, 48, 1, 48, 1, 5, "lc"),
)

GLOBAL_HLT = GlobalHLTConfig()
SCAN = ScanConfig()
SYSTEMATIC = SystematicConfig()
