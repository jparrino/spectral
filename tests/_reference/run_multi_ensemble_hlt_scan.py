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

"""Run a grouped parallel MP HLT scan with batched R-ratio systematics.

Settings live in `scan_inputs.py`.

For the mpmath backend, work is scheduled by basis/target group rather than by
individual lambda.  Lambda-independent MP integrals and LL/LC identity-
regulated solutions are shared automatically.

Short test
----------
python run_multi_ensemble_hlt_scan.py \
    --data-dir data \
    --output-dir scan_output_test \
    --max-scan-points 20 \
    --n-workers 2 \
    --progress-every 1 \
    --verbose

Full scan
---------
python run_multi_ensemble_hlt_scan.py \
    --data-dir data \
    --output-dir scan_output \
    --n-workers 4 \
    --progress-every 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jks
import numpy as np
from hlt import (
    CorrelatorData,
    HLTBasis,
    HLTReconstructor,
    IntegratedHVPTarget,
    QuadOptions,
)
from scan_inputs import (
    ENSEMBLES,
    GLOBAL_HLT,
    SCAN,
    SYSTEMATIC,
    EnsembleConfig,
    GlobalHLTConfig,
    ScanConfig,
    SystematicConfig,
)


_WORKER_CACHE: dict[tuple[Any, ...], Any] = {}


class ScanPoint:
    """Small pickle-friendly scan-point container."""

    def __init__(
        self,
        *,
        index: int,
        n_total: int,
        ensemble_index: int,
        kernel_name: str,
        time_like_cut_MeV: float,
        delta_factor: float,
        lambda_hlt: Any,
    ) -> None:
        self.index = index
        self.n_total = n_total
        self.ensemble_index = ensemble_index
        self.kernel_name = kernel_name
        self.time_like_cut_MeV = time_like_cut_MeV
        self.delta_factor = delta_factor
        self.lambda_hlt = lambda_hlt


def format_seconds(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "unknown"

    seconds = int(round(seconds))
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    if days > 0:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours > 0:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    if minutes > 0:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"



def correlator_root(ens: EnsembleConfig) -> str:
    """Return the discretization-specific correlator root."""
    discretization = ens.discretization.lower()
    if discretization not in {"ll", "lc"}:
        raise ValueError(
            f"ensemble {ens.name!r} has unsupported discretization "
            f"{ens.discretization!r}; expected 'll' or 'lc'"
        )
    return f"C_{discretization}_sub{ens.nrec}"


def omega0_root(ens: EnsembleConfig) -> str:
    """Human-readable source for omega0 used in the output CSV.

    For nrec=0 there is no lower reconstructed level star/En-{nrec-1}.
    In that case use the two-pion threshold Epipi.
    """
    if ens.nrec == 0:
        return "Epipi"
    if ens.nrec < 0:
        raise ValueError(
            f"ensemble {ens.name!r} has nrec={ens.nrec}; "
            "this scan code expects nrec >= 0"
        )
    return f"star/En-{ens.nrec - 1}"


def resolve_omega0(
    *,
    db: Any,
    ens: EnsembleConfig,
    derived: dict[str, float],
) -> float:
    """Return omega0 for the ensemble/nrec convention.

    nrec=0: use Epipi = 2*sqrt(m_pi^2 + (2*pi/L)^2).
    nrec>=1: use the level star/En-{nrec-1} from the JKS database.
    """
    if ens.nrec == 0:
        return float(derived["Epipi"])
    if ens.nrec < 0:
        raise ValueError(
            f"ensemble {ens.name!r} has nrec={ens.nrec}; "
            "this scan code expects nrec >= 0"
        )
    return float(np.asarray(db.get(f"star/En-{ens.nrec - 1}").mean(), dtype=float)[0])


def derived_parameters(ens: EnsembleConfig) -> dict[str, float]:
    alpha_em = 1.0 / 137.035999084
    m_mu = 105.6583755 / ens.a_lat_inv_MeV
    a_lat_fm = 0.197327 / (ens.a_lat_inv_MeV * 0.001)
    m_pi = ens.m_pi_MeV / ens.a_lat_inv_MeV

    t0 = 0.4 / a_lat_fm
    t1 = 1.0 / a_lat_fm
    window_delta = 0.15 / a_lat_fm
    epipi = 2.0 * np.sqrt(m_pi**2 + (2.0 * np.pi / ens.L_lattice) ** 2)

    return {
        "alpha_em": alpha_em,
        "m_mu": m_mu,
        "a_lat_fm": a_lat_fm,
        "m_pi": m_pi,
        "t0": t0,
        "t1": t1,
        "window_delta": window_delta,
        "Epipi": epipi,
    }


def hlt_quad_options(hlt: GlobalHLTConfig) -> QuadOptions:
    return QuadOptions(
        mp_method=getattr(hlt, "mp_quadrature_method", "gauss-legendre"),
        mp_maxdegree=getattr(hlt, "mp_quadrature_maxdegree", 8),
    )


def window_quad_options(hlt: GlobalHLTConfig) -> QuadOptions:
    return QuadOptions(
        mp_method=getattr(
            hlt, "mp_window_quadrature_method", "gauss-legendre"
        ),
        mp_maxdegree=getattr(hlt, "mp_window_quadrature_maxdegree", 8),
        epsabs=1e-9,
        epsrel=1e-7,
        limit=100,
    )



def load_systematic_spectrum(
    r_ratio_path: Path,
    *,
    ens: EnsembleConfig,
    hlt: GlobalHLTConfig,
    syst_cfg: SystematicConfig,
) -> dict[str, np.ndarray]:
    """Load the nominal R-ratio spectrum and its rho_p representation.

    The undistorted R-ratio is retained because Gaussian-noise realizations
    are generated and normalized in physical energy before conversion to rho_p.

    rho_2 = R(s)/(12*pi^2) * (2/ZV^2) * systblind
    rho_p = rho_2 / omega_lat**(p-2)
    """

    cache_key = (
        "systematic_spectrum",
        str(r_ratio_path.resolve()),
        ens.name,
        ens.a_lat_inv_MeV,
        ens.ZV,
        hlt.p,
        syst_cfg.systblind,
        syst_cfg.r_ratio_max_points,
    )
    if cache_key in _WORKER_CACHE:
        return _WORKER_CACHE[cache_key]

    omega_GeV, r_ratio = np.loadtxt(
        r_ratio_path,
        delimiter=",",
        skiprows=1,
        usecols=(0, 3),
        unpack=True,
    )

    sort_idx = np.argsort(omega_GeV, kind="stable")
    omega_GeV = omega_GeV[sort_idx]
    r_ratio = r_ratio[sort_idx]

    keep = np.concatenate(([True], np.diff(omega_GeV) > 0))
    omega_GeV = omega_GeV[keep][: syst_cfg.r_ratio_max_points]
    r_ratio = r_ratio[keep][: syst_cfg.r_ratio_max_points]

    valid = (
        np.isfinite(omega_GeV)
        & np.isfinite(r_ratio)
        & (omega_GeV >= 0.0)
        & (r_ratio >= 0.0)
    )
    omega_GeV = omega_GeV[valid]
    r_ratio = r_ratio[valid]

    if len(omega_GeV) < 2:
        raise ValueError("R-ratio spectrum has fewer than two valid points")

    omega_MeV = 1000.0 * omega_GeV
    omega_lat = omega_MeV / ens.a_lat_inv_MeV

    rho_2 = r_ratio / (12.0 * np.pi**2)
    rho_2 *= 2.0 / ens.ZV**2 * syst_cfg.systblind
    rho_p = rho_2 / omega_lat ** (hlt.p - 2.0)

    out = {
        "omega_GeV": omega_GeV,
        "omega_MeV": omega_MeV,
        "omega_lat": omega_lat,
        "r_ratio": r_ratio,
        "rho_p": rho_p,
    }
    _WORKER_CACHE[cache_key] = out
    return out


def load_ensemble_objects(
    *,
    data_dir: Path,
    ens: EnsembleConfig,
    hlt: GlobalHLTConfig,
    syst_cfg: SystematicConfig,
) -> dict[str, Any]:
    """Load and cache ensemble data, basis, omega0, and systematic grid."""

    cache_key = (
        "ensemble_objects",
        str(data_dir.resolve()),
        ens.name,
        ens.discretization,
        ens.resolved_jks_filename(),
        ens.nrec,
        ens.m_pi_MeV,
        ens.L_lattice,
        ens.basis_t_min,
        ens.basis_t_max,
        ens.basis_stride,
        ens.T,
        hlt.p,
        syst_cfg.r_ratio_filename,
        syst_cfg.r_ratio_max_points,
        syst_cfg.systblind,
    )
    if cache_key in _WORKER_CACHE:
        return _WORKER_CACHE[cache_key]

    jks_path = data_dir / ens.resolved_jks_filename()
    r_ratio_path = data_dir / syst_cfg.r_ratio_filename

    db_key = ("jks_database", str(jks_path.resolve()))
    db = _WORKER_CACHE.get(db_key)
    if db is None:
        db = jks.resamples(str(jks_path))
        _WORKER_CACHE[db_key] = db

    approx = db.get(correlator_root(ens))
    mean = np.asarray(approx.mean(), dtype=float)
    cov = np.asarray(approx.cov(), dtype=float)
    times = np.arange(len(mean), dtype=float)

    derived = derived_parameters(ens)
    omega0 = resolve_omega0(db=db, ens=ens, derived=derived)

    data = CorrelatorData.from_arrays(
        mean=mean,
        covariance=cov,
        times=times,
        T=ens.T,
    )

    basis_times = data.select_times(
        t_min=ens.basis_t_min,
        t_max=ens.basis_t_max,
        stride=ens.basis_stride,
    )

    basis = HLTBasis(
        times=basis_times,
        p=hlt.p,
        T=ens.T,
        periodic=True,
        omega0=omega0,
    )

    spectrum = load_systematic_spectrum(
        r_ratio_path,
        ens=ens,
        hlt=hlt,
        syst_cfg=syst_cfg,
    )

    mask = spectrum["omega_lat"] >= omega0
    omega_lat_cut = spectrum["omega_lat"][mask]
    rho_p_cut = spectrum["rho_p"][mask]

    if len(omega_lat_cut) < 2:
        raise ValueError(
            f"systematic omega grid for ensemble {ens.name} has fewer than two "
            f"points above omega0={omega0}"
        )

    out = {
        "data": data,
        "basis": basis,
        "omega0": omega0,
        "omega_lat_cut": omega_lat_cut,
        "rho_p_cut": rho_p_cut,
        "systematic_omega_MeV_all": spectrum["omega_MeV"],
        "systematic_r_ratio_all": spectrum["r_ratio"],
        "systematic_cut_mask": mask,
        "basis_times": basis_times,
    }
    _WORKER_CACHE[cache_key] = out
    return out


def get_check(checks: Any, name: str) -> float:
    value = getattr(checks, name, None)
    return math.nan if value is None else float(value)


def get_attr_float(obj: Any, name: str) -> float:
    return float(getattr(obj, name, math.nan))


def get_systematic_float(cfg: SystematicConfig, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def get_systematic_bool(cfg: SystematicConfig, name: str, default: bool) -> bool:
    return bool(getattr(cfg, name, default))


def systematic_result_to_dict(prefix: str, result: Any) -> dict[str, float]:
    return {
        f"{prefix}_signed": get_attr_float(result, "signed_bias"),
        f"{prefix}_absolute": get_attr_float(result, "absolute_bias"),
        f"{prefix}_relative_signed": get_attr_float(result, "relative_signed_bias"),
        f"{prefix}_relative_absolute": get_attr_float(result, "relative_absolute_bias"),
        f"{prefix}_target_observable": get_attr_float(result, "target_observable"),
        f"{prefix}_reconstructed_observable": get_attr_float(result, "reconstructed_observable"),
    }


def get_systematic_int(cfg: SystematicConfig, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def smooth_gaussian_noise_on_energy_grid(
    *,
    energy_MeV: np.ndarray,
    noise: np.ndarray,
    correlation_length_MeV: float,
) -> np.ndarray:
    """Smooth standard-normal noise with an approximately Gaussian kernel.

    The R-ratio energy grid is not assumed to be perfectly uniform.  The
    smoothing width in grid points is determined from the median positive
    energy spacing.  A zero correlation length returns the original
    pointwise-independent noise.
    """
    energy_MeV = np.asarray(energy_MeV, dtype=float)
    noise = np.asarray(noise, dtype=float)

    if correlation_length_MeV <= 0.0:
        return noise.copy()

    spacings = np.diff(energy_MeV)
    spacings = spacings[np.isfinite(spacings) & (spacings > 0.0)]
    if len(spacings) == 0:
        raise ValueError("cannot determine a positive R-ratio energy spacing")

    median_spacing = float(np.median(spacings))
    sigma_points = correlation_length_MeV / median_spacing
    if sigma_points <= 0.25:
        return noise.copy()

    half_width = max(1, int(math.ceil(4.0 * sigma_points)))
    offsets = np.arange(-half_width, half_width + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma_points) ** 2)
    kernel /= np.sum(kernel)

    # Reflect padding avoids artificially suppressing the noise near the ends.
    padded = np.pad(noise, half_width, mode="reflect")
    return np.convolve(padded, kernel, mode="valid")


def _trapezoidal_point_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if len(x) < 2 or np.any(np.diff(x) <= 0.0):
        raise ValueError("energy grid must contain at least two increasing points")
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    if len(x) > 2:
        w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def build_gaussian_noise_factor(
    *,
    energy_MeV: np.ndarray,
    spectral_weight: np.ndarray,
    rng: np.random.Generator,
    relative_sigma: float,
    correlation_length_MeV: float,
    preserve_integral: bool,
    require_positive: bool,
    max_resampling_attempts: int = 100,
) -> tuple[np.ndarray, dict[str, float]]:
    energy_MeV = np.asarray(energy_MeV, dtype=float)
    spectral_weight = np.asarray(spectral_weight, dtype=float)
    if energy_MeV.shape != spectral_weight.shape:
        raise ValueError("energy_MeV and spectral_weight must have the same shape")
    if relative_sigma < 0.0:
        raise ValueError("r_ratio_noise_relative_sigma must be non-negative")
    weights = _trapezoidal_point_weights(energy_MeV)
    measure = weights * np.maximum(spectral_weight, 0.0)
    measure_sum = float(np.sum(measure))
    if not np.isfinite(measure_sum) or measure_sum <= 0.0:
        raise ValueError("systematic spectral weight has zero or non-finite integral")
    for attempt in range(max_resampling_attempts):
        eta = rng.normal(0.0, 1.0, size=len(energy_MeV))
        eta = smooth_gaussian_noise_on_energy_grid(
            energy_MeV=energy_MeV,
            noise=eta,
            correlation_length_MeV=correlation_length_MeV,
        )
        weighted_mean_before = float(np.sum(measure * eta) / measure_sum)
        if preserve_integral:
            eta = eta - weighted_mean_before
        weighted_rms_sq = float(np.sum(measure * eta**2) / measure_sum)
        if not np.isfinite(weighted_rms_sq) or weighted_rms_sq <= 1e-20:
            continue
        eta /= math.sqrt(weighted_rms_sq)
        factor = 1.0 + relative_sigma * eta
        if require_positive and np.any(factor <= 0.0):
            continue
        diagnostics = {
            "integral_ratio": float(np.sum(measure * factor) / measure_sum),
            "minimum_factor": float(np.min(factor)),
            "maximum_factor": float(np.max(factor)),
            "noise_weighted_mean_before_centering": weighted_mean_before,
            "noise_weighted_rms": float(math.sqrt(np.sum(measure * eta**2) / measure_sum)),
            "resampling_attempts": float(attempt + 1),
        }
        return factor, diagnostics
    raise ValueError(
        "could not generate a positive Gaussian-noise realization after "
        f"{max_resampling_attempts} attempts; reduce relative_sigma or set "
        "r_ratio_noise_require_positive=False"
    )


def estimate_r_ratio_gaussian_noise_envelope(
    *,
    int_result: Any,
    sharp_target: Any,
    omega_lat_cut: np.ndarray,
    rho_p_cut: np.ndarray,
    n_samples: int,
    relative_sigma: float,
    correlation_length_MeV: float,
    random_seed: int,
    preserve_integral: bool,
    require_positive: bool,
    a_lat_inv_MeV: float,
) -> dict[str, float]:
    if n_samples < 1:
        raise ValueError("r_ratio_noise_n_samples must be >= 1")
    omega_lat = np.asarray(omega_lat_cut, dtype=float)
    rho_nominal = np.asarray(rho_p_cut, dtype=float)
    valid = np.isfinite(omega_lat) & np.isfinite(rho_nominal) & (omega_lat > 0.0)
    omega_lat = omega_lat[valid]
    rho_nominal = rho_nominal[valid]
    if len(omega_lat) < 2:
        raise ValueError("fewer than two valid systematic-grid points")
    energy_MeV = omega_lat * float(a_lat_inv_MeV)
    values = {
        "syst_kernel_noise_signed": [],
        "syst_kernel_noise_absolute": [],
        "syst_kernel_noise_relative_signed": [],
        "syst_kernel_noise_relative_absolute": [],
        "syst_kernel_noise_target_observable": [],
        "syst_kernel_noise_reconstructed_observable": [],
        "syst_total_noise_signed": [],
        "syst_total_noise_absolute": [],
        "syst_total_noise_relative_signed": [],
        "syst_total_noise_relative_absolute": [],
        "syst_total_noise_target_observable": [],
        "syst_total_noise_reconstructed_observable": [],
    }
    diagnostics_all = []
    n_valid = 0
    n_rejected = 0
    rng = np.random.default_rng(random_seed)
    for sample_index in range(n_samples):
        try:
            factor, diagnostics = build_gaussian_noise_factor(
                energy_MeV=energy_MeV,
                spectral_weight=rho_nominal,
                rng=rng,
                relative_sigma=relative_sigma,
                correlation_length_MeV=correlation_length_MeV,
                preserve_integral=preserve_integral,
                require_positive=require_positive,
            )
        except ValueError:
            n_rejected += 1
            continue
        rho_var = rho_nominal * factor
        syst_kernel_var = int_result.estimate_kernel_systematic_on_grid(
            omega=omega_lat, rho_p=rho_var
        )
        syst_total_var = int_result.estimate_total_sharp_systematic_on_grid(
            omega=omega_lat, rho_p=rho_var, sharp_target=sharp_target
        )
        for key, value in systematic_result_to_dict("syst_kernel_noise", syst_kernel_var).items():
            values[key].append((float(value), int(sample_index)))
        for key, value in systematic_result_to_dict("syst_total_noise", syst_total_var).items():
            values[key].append((float(value), int(sample_index)))
        diagnostics_all.append(diagnostics)
        n_valid += 1
    if n_valid == 0:
        raise ValueError(
            "all Gaussian-noise samples were rejected; reduce "
            "r_ratio_noise_relative_sigma or set "
            "r_ratio_noise_require_positive=False"
        )
    out = {
        "syst_noise_n_requested": float(n_samples),
        "syst_noise_n_valid": float(n_valid),
        "syst_noise_n_rejected": float(n_rejected),
        "syst_noise_relative_sigma": float(relative_sigma),
        "syst_noise_correlation_length_MeV": float(correlation_length_MeV),
        "syst_noise_random_seed": float(random_seed),
        "syst_noise_integral_ratio_min": float(min(d["integral_ratio"] for d in diagnostics_all)),
        "syst_noise_integral_ratio_max": float(max(d["integral_ratio"] for d in diagnostics_all)),
        "syst_noise_factor_min": float(min(d["minimum_factor"] for d in diagnostics_all)),
        "syst_noise_factor_max": float(max(d["maximum_factor"] for d in diagnostics_all)),
    }
    for key, samples in values.items():
        finite = [(v, i) for v, i in samples if np.isfinite(v)]
        if not finite:
            out[f"{key}_min"] = math.nan
            out[f"{key}_max"] = math.nan
            out[f"{key}_sample_index_at_min"] = math.nan
            out[f"{key}_sample_index_at_max"] = math.nan
            continue
        min_value, min_index = min(finite, key=lambda item: item[0])
        max_value, max_index = max(finite, key=lambda item: item[0])
        out[f"{key}_min"] = float(min_value)
        out[f"{key}_max"] = float(max_value)
        out[f"{key}_sample_index_at_min"] = float(min_index)
        out[f"{key}_sample_index_at_max"] = float(max_index)
    return out


def build_r_ratio_noise_batch(
    *,
    omega_lat: np.ndarray,
    rho_nominal: np.ndarray,
    n_samples: int,
    relative_sigma: float,
    correlation_length_MeV: float,
    random_seed: int,
    preserve_integral: bool,
    require_positive: bool,
    a_lat_inv_MeV: float,
) -> dict[str, Any]:
    """Generate the reproducible noise ensemble once for an entire scan group."""
    omega_lat = np.asarray(omega_lat, dtype=float)
    rho_nominal = np.asarray(rho_nominal, dtype=float)
    energy_MeV = omega_lat * float(a_lat_inv_MeV)
    rng = np.random.default_rng(random_seed)
    rho_samples: list[np.ndarray] = []
    sample_indices: list[int] = []
    diagnostics_all: list[dict[str, float]] = []
    n_rejected = 0
    for sample_index in range(n_samples):
        try:
            factor, diagnostics = build_gaussian_noise_factor(
                energy_MeV=energy_MeV,
                spectral_weight=rho_nominal,
                rng=rng,
                relative_sigma=relative_sigma,
                correlation_length_MeV=correlation_length_MeV,
                preserve_integral=preserve_integral,
                require_positive=require_positive,
            )
        except ValueError:
            n_rejected += 1
            continue
        rho_samples.append(rho_nominal * factor)
        sample_indices.append(sample_index)
        diagnostics_all.append(diagnostics)
    if not rho_samples:
        raise ValueError(
            "all Gaussian-noise samples were rejected; reduce "
            "r_ratio_noise_relative_sigma or disable positivity"
        )
    return {
        "rho": np.asarray(rho_samples, dtype=float),
        "sample_indices": np.asarray(sample_indices, dtype=int),
        "diagnostics": diagnostics_all,
        "n_requested": n_samples,
        "n_rejected": n_rejected,
        "relative_sigma": relative_sigma,
        "correlation_length_MeV": correlation_length_MeV,
        "random_seed": random_seed,
        "preserve_integral": preserve_integral,
        "require_positive": require_positive,
    }


def _systematic_metrics_batch(
    *,
    reference: np.ndarray,
    reconstructed: np.ndarray,
    rho: np.ndarray,
    weights: np.ndarray,
    difference: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate one kernel difference against every spectrum in one BLAS batch."""
    reference = np.asarray(reference, dtype=float)
    reconstructed = np.asarray(reconstructed, dtype=float)
    rho = np.atleast_2d(np.asarray(rho, dtype=float))
    weights = np.asarray(weights, dtype=float)
    difference = (
        reconstructed - reference
        if difference is None
        else np.asarray(difference, dtype=float)
    )
    target_observable = rho @ (weights * reference)
    reconstructed_observable = rho @ (weights * reconstructed)
    signed_bias = rho @ (weights * difference)
    absolute_bias = np.abs(rho) @ (weights * np.abs(difference))
    denominator = np.maximum(np.abs(target_observable), np.finfo(float).eps)
    return {
        "signed": signed_bias,
        "absolute": absolute_bias,
        "relative_signed": signed_bias / denominator,
        "relative_absolute": absolute_bias / denominator,
        "target_observable": target_observable,
        "reconstructed_observable": reconstructed_observable,
    }


def _nominal_systematic(metrics: dict[str, np.ndarray], stat_error: Any):
    absolute = float(metrics["absolute"][0])
    return SimpleNamespace(
        signed_bias=float(metrics["signed"][0]),
        absolute_bias=absolute,
        relative_signed_bias=float(metrics["relative_signed"][0]),
        relative_absolute_bias=float(metrics["relative_absolute"][0]),
        target_observable=float(metrics["target_observable"][0]),
        reconstructed_observable=float(metrics["reconstructed_observable"][0]),
        stat_error=float(stat_error),
        total_error_with_absolute_bias=float(np.hypot(float(stat_error), absolute)),
    )


def _noise_envelope_from_batches(
    *,
    kernel_metrics: dict[str, np.ndarray],
    total_metrics: dict[str, np.ndarray],
    noise_batch: dict[str, Any],
) -> dict[str, float]:
    diagnostics = noise_batch["diagnostics"]
    indices = noise_batch["sample_indices"]
    out: dict[str, float] = {
        "syst_noise_n_requested": float(noise_batch["n_requested"]),
        "syst_noise_n_valid": float(len(indices)),
        "syst_noise_n_rejected": float(noise_batch["n_rejected"]),
        "syst_noise_relative_sigma": float(noise_batch["relative_sigma"]),
        "syst_noise_correlation_length_MeV": float(
            noise_batch["correlation_length_MeV"]
        ),
        "syst_noise_random_seed": float(noise_batch["random_seed"]),
        "syst_noise_preserve_integral": float(noise_batch["preserve_integral"]),
        "syst_noise_require_positive": float(noise_batch["require_positive"]),
        "syst_noise_integral_ratio_min": float(
            min(item["integral_ratio"] for item in diagnostics)
        ),
        "syst_noise_integral_ratio_max": float(
            max(item["integral_ratio"] for item in diagnostics)
        ),
        "syst_noise_factor_min": float(
            min(item["minimum_factor"] for item in diagnostics)
        ),
        "syst_noise_factor_max": float(
            max(item["maximum_factor"] for item in diagnostics)
        ),
    }
    for prefix, metrics in (
        ("syst_kernel_noise", kernel_metrics),
        ("syst_total_noise", total_metrics),
    ):
        for metric_name, values in metrics.items():
            values = np.asarray(values, dtype=float)
            finite = np.flatnonzero(np.isfinite(values))
            key = f"{prefix}_{metric_name}"
            if len(finite) == 0:
                out[f"{key}_min"] = math.nan
                out[f"{key}_max"] = math.nan
                out[f"{key}_sample_index_at_min"] = math.nan
                out[f"{key}_sample_index_at_max"] = math.nan
                continue
            local_min = finite[int(np.argmin(values[finite]))]
            local_max = finite[int(np.argmax(values[finite]))]
            out[f"{key}_min"] = float(values[local_min])
            out[f"{key}_max"] = float(values[local_max])
            out[f"{key}_sample_index_at_min"] = float(indices[local_min])
            out[f"{key}_sample_index_at_max"] = float(indices[local_max])
    return out


def evaluate_scan_kernels_mpmath(
    *,
    prepared: Any,
    solutions: list[Any],
    sharp_target: Any,
    omega: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate one MP basis matrix and all lambda kernels simultaneously."""
    from hlt.mpmath_backend import MPWorkspace, target_values
    from hlt.precision import require_mpmath, to_mpf

    mp = require_mpmath()
    with mp.workdps(prepared.mp_dps):
        workspace = MPWorkspace(prepared.mp_dps)
        omega_mp = [to_mpf(value, mp) for value in np.asarray(omega, dtype=float)]
        times_mp = [to_mpf(value, mp) for value in prepared.basis._times_source]
        basis_power = to_mpf(prepared.basis._p_source, mp)
        temporal_extent = (
            None
            if prepared.basis._T_source is None
            else to_mpf(prepared.basis._T_source, mp)
        )
        phi = mp.matrix(len(omega_mp), len(times_mp))
        for row, omega_value in enumerate(omega_mp):
            omega_power = mp.power(omega_value, basis_power)
            for column, time_value in enumerate(times_mp):
                basis_value = mp.exp(-omega_value * time_value)
                if prepared.basis.periodic:
                    basis_value += mp.exp(
                        -omega_value * (temporal_extent - time_value)
                    )
                phi[row, column] = omega_power * basis_value
        coefficients = mp.matrix(len(times_mp), len(solutions))
        for column, solution in enumerate(solutions):
            for row in range(len(times_mp)):
                coefficients[row, column] = solution.q[row]
        reconstructed = phi * coefficients
        smooth = target_values(prepared.target, omega_mp, mp, workspace)
        sharp = target_values(sharp_target, omega_mp, mp, workspace)
        kernel_difference = mp.matrix(reconstructed.rows, reconstructed.cols)
        total_difference = mp.matrix(reconstructed.rows, reconstructed.cols)
        for i in range(reconstructed.rows):
            for j in range(reconstructed.cols):
                kernel_difference[i, j] = reconstructed[i, j] - smooth[i]
                total_difference[i, j] = reconstructed[i, j] - sharp[i]

        def matrix_to_numpy(matrix):
            return np.asarray(
                [
                    [float(matrix[i, j]) for j in range(matrix.cols)]
                    for i in range(matrix.rows)
                ],
                dtype=float,
            )

        return {
            "reconstructed": matrix_to_numpy(reconstructed),
            "kernel_difference": matrix_to_numpy(kernel_difference),
            "total_difference": matrix_to_numpy(total_difference),
            "smooth": np.asarray([float(smooth[i]) for i in range(smooth.rows)]),
            "sharp": np.asarray([float(sharp[i]) for i in range(sharp.rows)]),
        }


def result_field_names() -> list[str]:
    nominal_fields = [
        "endpoint_value_resolved", "endpoint_abs", "endpoint_rel",
        "omega_constraints_resolved", "omega_constraint_abs", "omega_constraint_rel",
        "estimate_raw",
        "error_raw",
        "variance_raw",
        "estimate_scaled",
        "error_scaled",
        "variance_scaled",
        "syst_kernel_signed_raw",
        "syst_kernel_absolute_raw",
        "syst_kernel_relative_signed",
        "syst_kernel_relative_absolute",
        "syst_kernel_target_observable",
        "syst_kernel_reconstructed_observable",
        "syst_kernel_signed_scaled",
        "syst_kernel_absolute_scaled",
        "syst_total_signed_raw",
        "syst_total_absolute_raw",
        "syst_total_relative_signed",
        "syst_total_relative_absolute",
        "syst_total_target_observable",
        "syst_total_reconstructed_observable",
        "syst_total_signed_scaled",
        "syst_total_absolute_scaled",
    ]

    noise_fields = [
        # Final quoted envelope from the signed Gaussian-noise variations:
        # max(abs(signed_min), abs(signed_max)).
        "syst_kernel_noise_final_raw",
        "syst_kernel_noise_final_scaled",
        "syst_total_noise_final_raw",
        "syst_total_noise_final_scaled",
        # Statistical and final systematic uncertainty added in quadrature.
        "total_error_noise_raw",
        "total_error_noise_scaled",
        "syst_noise_n_requested",
        "syst_noise_n_valid",
        "syst_noise_n_rejected",
        "syst_noise_relative_sigma",
        "syst_noise_correlation_length_MeV",
        "syst_noise_random_seed",
        "syst_noise_preserve_integral",
        "syst_noise_require_positive",
        "syst_noise_integral_ratio_min",
        "syst_noise_integral_ratio_max",
        "syst_noise_factor_min",
        "syst_noise_factor_max",
    ]

    for base in (
        "syst_kernel_noise_signed",
        "syst_kernel_noise_absolute",
        "syst_kernel_noise_relative_signed",
        "syst_kernel_noise_relative_absolute",
        "syst_kernel_noise_target_observable",
        "syst_kernel_noise_reconstructed_observable",
        "syst_total_noise_signed",
        "syst_total_noise_absolute",
        "syst_total_noise_relative_signed",
        "syst_total_noise_relative_absolute",
        "syst_total_noise_target_observable",
        "syst_total_noise_reconstructed_observable",
    ):
        noise_fields.extend(
            [
                f"{base}_min_raw",
                f"{base}_max_raw",
                f"{base}_sample_index_at_min",
                f"{base}_sample_index_at_max",
            ]
        )
        if (
            "relative" not in base
            and "target_observable" not in base
            and "reconstructed_observable" not in base
        ):
            noise_fields.extend(
                [f"{base}_min_scaled", f"{base}_max_scaled"]
            )

    check_fields = [
        "mismatch_abs",
        "mismatch_rel",
        "area_abs",
        "area_rel",
        "stationarity_rel",
        "condition_number",
        "max_abs_q",
    ]

    return nominal_fields + noise_fields + check_fields

def base_row(
    *,
    point: ScanPoint,
    ens: EnsembleConfig,
    hlt: GlobalHLTConfig,
    scan: ScanConfig,
    syst_cfg: SystematicConfig,
    derived: dict[str, float],
    omega0: float | None,
) -> dict[str, Any]:
    am_cut = point.time_like_cut_MeV / ens.a_lat_inv_MeV
    delta_lattice = point.delta_factor * derived["m_pi"]
    scale_factor = 0.5 * ens.ZV**2

    row: dict[str, Any] = {
        "scan_index": point.index,
        "scan_total": point.n_total,
        "ensemble_index": point.ensemble_index,
        "ensemble": ens.name,
        "discretization": ens.discretization,
        "jks_filename": ens.resolved_jks_filename(),
        "nrec": ens.nrec,
        "correlator_root": correlator_root(ens),
        "omega0_root": omega0_root(ens),
        "omega0": math.nan if omega0 is None else float(omega0),
        "a_lat_inv_MeV": ens.a_lat_inv_MeV,
        "m_pi_MeV": ens.m_pi_MeV,
        "ZV": ens.ZV,
        "T": ens.T,
        "L_lattice": ens.L_lattice,
        "a_lat_fm": derived["a_lat_fm"],
        "m_pi_lat": derived["m_pi"],
        "m_mu_lat": derived["m_mu"],
        "kernel": point.kernel_name,
        "time_like_cut_MeV": float(point.time_like_cut_MeV),
        "am_cut": float(am_cut),
        "delta_factor": float(point.delta_factor),
        "delta_lattice": float(delta_lattice),
        "lambda_hlt": float(point.lambda_hlt),
        "basis_t_min": ens.basis_t_min,
        "basis_t_max": ens.basis_t_max,
        "basis_stride": ens.basis_stride,
        "target_type": "IntegratedHVPTarget",
        "euclidean_window": point.kernel_name,
        "window_t0_lattice": derived["t0"],
        "window_t1_lattice": derived["t1"],
        "window_delta_lattice": derived["window_delta"],
        "p": hlt.p,
        "regulator": hlt.regulator,
        "endpoint_value_config": getattr(hlt, "endpoint_value", None),
        "omega_constraints_config": json.dumps(getattr(hlt, "omega_constraints", ()) or (), default=str),
        "solver_backend": hlt.solver_backend,
        "mp_dps": hlt.mp_dps,
        "strict": hlt.strict,
        "compute_mismatch": hlt.compute_mismatch,
        "mp_quadrature_method": getattr(
            hlt, "mp_quadrature_method", "gauss-legendre"
        ),
        "mp_quadrature_maxdegree": getattr(hlt, "mp_quadrature_maxdegree", 8),
        "mp_window_quadrature_method": getattr(
            hlt, "mp_window_quadrature_method", "gauss-legendre"
        ),
        "mp_window_quadrature_maxdegree": getattr(
            hlt, "mp_window_quadrature_maxdegree", 8
        ),
        "kernel_quadrature_order": hlt.kernel_quadrature_order,
        "scale_factor": float(scale_factor),
        "systblind": syst_cfg.systblind,
        "r_ratio_filename": syst_cfg.r_ratio_filename,
        "r_ratio_max_points": syst_cfg.r_ratio_max_points,
        "r_ratio_noise_n_samples_config": get_systematic_int(
            syst_cfg, "r_ratio_noise_n_samples", 100
        ),
        "r_ratio_noise_relative_sigma_config": get_systematic_float(
            syst_cfg, "r_ratio_noise_relative_sigma", 0.05
        ),
        "r_ratio_noise_correlation_length_MeV_config": get_systematic_float(
            syst_cfg, "r_ratio_noise_correlation_length_MeV", 100.0
        ),
        "r_ratio_noise_random_seed_config": get_systematic_int(
            syst_cfg, "r_ratio_noise_random_seed", 12345
        ),
        "r_ratio_noise_preserve_integral_config": get_systematic_bool(
            syst_cfg, "r_ratio_noise_preserve_integral", True
        ),
        "r_ratio_noise_require_positive_config": get_systematic_bool(
            syst_cfg, "r_ratio_noise_require_positive", True
        ),
        "worker_pid": os.getpid(),
    }

    for key in result_field_names():
        row[key] = math.nan

    row["status"] = "not_run"
    row["error_message"] = ""
    row["traceback"] = ""
    row["warnings"] = ""
    return row


def fill_success_row(
    *,
    row: dict[str, Any],
    int_result: Any,
    syst_kernel: Any,
    syst_total: Any,
    ens: EnsembleConfig,
    noise_envelope: dict[str, float] | None = None,
) -> dict[str, Any]:
    scale = 0.5 * ens.ZV**2

    row["estimate_raw"] = float(int_result.estimate)
    row["error_raw"] = float(int_result.error)
    row["variance_raw"] = float(int_result.variance)
    row["estimate_scaled"] = scale * row["estimate_raw"]
    row["error_scaled"] = scale * row["error_raw"]
    row["variance_scaled"] = scale**2 * row["variance_raw"]

    row["syst_kernel_signed_raw"] = get_attr_float(syst_kernel, "signed_bias")
    row["syst_kernel_absolute_raw"] = get_attr_float(syst_kernel, "absolute_bias")
    row["syst_kernel_relative_signed"] = get_attr_float(syst_kernel, "relative_signed_bias")
    row["syst_kernel_relative_absolute"] = get_attr_float(syst_kernel, "relative_absolute_bias")
    row["syst_kernel_target_observable"] = get_attr_float(syst_kernel, "target_observable")
    row["syst_kernel_reconstructed_observable"] = get_attr_float(syst_kernel, "reconstructed_observable")
    row["syst_kernel_signed_scaled"] = scale * row["syst_kernel_signed_raw"]
    row["syst_kernel_absolute_scaled"] = scale * row["syst_kernel_absolute_raw"]

    row["syst_total_signed_raw"] = get_attr_float(syst_total, "signed_bias")
    row["syst_total_absolute_raw"] = get_attr_float(syst_total, "absolute_bias")
    row["syst_total_relative_signed"] = get_attr_float(syst_total, "relative_signed_bias")
    row["syst_total_relative_absolute"] = get_attr_float(syst_total, "relative_absolute_bias")
    row["syst_total_target_observable"] = get_attr_float(syst_total, "target_observable")
    row["syst_total_reconstructed_observable"] = get_attr_float(syst_total, "reconstructed_observable")
    row["syst_total_signed_scaled"] = scale * row["syst_total_signed_raw"]
    row["syst_total_absolute_scaled"] = scale * row["syst_total_absolute_raw"]

    if noise_envelope is not None:
        for key, value in noise_envelope.items():
            if key.startswith("syst_kernel_noise") or key.startswith("syst_total_noise"):
                if key.endswith("_sample_index_at_min") or key.endswith("_sample_index_at_max"):
                    row[key] = float(value)
                elif key.endswith("_min") or key.endswith("_max"):
                    row[f"{key}_raw"] = float(value)
                else:
                    row[key] = float(value)
            else:
                row[key] = float(value)

        for prefix in (
            "syst_kernel_noise_signed",
            "syst_kernel_noise_absolute",
            "syst_total_noise_signed",
            "syst_total_noise_absolute",
        ):
            row[f"{prefix}_min_scaled"] = scale * row.get(
                f"{prefix}_min_raw", math.nan
            )
            row[f"{prefix}_max_scaled"] = scale * row.get(
                f"{prefix}_max_raw", math.nan
            )

        def max_abs_pair(a: float, b: float) -> float:
            values = np.asarray([a, b], dtype=float)
            values = values[np.isfinite(values)]
            if len(values) == 0:
                return math.nan
            return float(np.max(np.abs(values)))

        # Final quoted systematic uncertainty from the signed envelope.
        row["syst_kernel_noise_final_raw"] = max_abs_pair(
            row.get("syst_kernel_noise_signed_min_raw", math.nan),
            row.get("syst_kernel_noise_signed_max_raw", math.nan),
        )
        row["syst_kernel_noise_final_scaled"] = max_abs_pair(
            row.get("syst_kernel_noise_signed_min_scaled", math.nan),
            row.get("syst_kernel_noise_signed_max_scaled", math.nan),
        )
        row["syst_total_noise_final_raw"] = max_abs_pair(
            row.get("syst_total_noise_signed_min_raw", math.nan),
            row.get("syst_total_noise_signed_max_raw", math.nan),
        )
        row["syst_total_noise_final_scaled"] = max_abs_pair(
            row.get("syst_total_noise_signed_min_scaled", math.nan),
            row.get("syst_total_noise_signed_max_scaled", math.nan),
        )

        # Final total uncertainty: statistical and quoted systematic errors
        # are treated as independent and added in quadrature.
        row["total_error_noise_raw"] = float(np.hypot(
            row["error_raw"],
            row["syst_total_noise_final_raw"],
        ))
        row["total_error_noise_scaled"] = float(np.hypot(
            row["error_scaled"],
            row["syst_total_noise_final_scaled"],
        ))

    checks = int_result.checks
    endpoint = getattr(int_result, "endpoint_value", None)
    row["endpoint_value_resolved"] = None if endpoint is None else str(endpoint)
    row["endpoint_abs"] = get_check(checks, "endpoint_abs")
    row["endpoint_rel"] = get_check(checks, "endpoint_rel")
    row["omega_constraints_resolved"] = json.dumps(
        [[str(w), str(d)] for w, d in getattr(int_result, "omega_constraints", ())])
    row["omega_constraint_abs"] = json.dumps([str(x) for x in getattr(checks, "omega_abs", ())])
    row["omega_constraint_rel"] = json.dumps([str(x) for x in getattr(checks, "omega_rel", ())])
    row["mismatch_abs"] = get_check(checks, "mismatch_abs")
    row["mismatch_rel"] = get_check(checks, "mismatch_rel")
    row["area_abs"] = get_check(checks, "area_abs")
    row["area_rel"] = get_check(checks, "area_rel")
    row["stationarity_rel"] = get_check(checks, "stationarity_rel")
    row["condition_number"] = get_check(checks, "condition_number")
    row["max_abs_q"] = get_check(checks, "max_abs_q")
    row["warnings"] = " | ".join(getattr(checks, "warnings", []))

    row["status"] = "success"
    return row


def reconstruct_with_optional_mismatch_arg(
    reconstructor: HLTReconstructor,
    target: Any,
    compute_mismatch: bool,
) -> Any:
    try:
        return reconstructor.reconstruct(target, compute_mismatch=compute_mismatch)
    except TypeError:
        return reconstructor.reconstruct(target)


def run_one_scan_point(
    point: ScanPoint,
    *,
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
    scan: ScanConfig,
    syst_cfg: SystematicConfig,
    data_dir_str: str,
) -> dict[str, Any]:
    """Worker function. Keep this top-level for multiprocessing."""

    ens = ensembles[point.ensemble_index]
    derived = derived_parameters(ens)

    row = base_row(
        point=point,
        ens=ens,
        hlt=hlt,
        scan=scan,
        syst_cfg=syst_cfg,
        derived=derived,
        omega0=None,
    )

    try:
        data_dir = Path(data_dir_str)
        objects = load_ensemble_objects(
            data_dir=data_dir,
            ens=ens,
            hlt=hlt,
            syst_cfg=syst_cfg,
        )
        row["omega0"] = float(objects["omega0"])

        am_cut = point.time_like_cut_MeV / ens.a_lat_inv_MeV
        delta_lattice = point.delta_factor * derived["m_pi"]

        reconstructor = HLTReconstructor(
            data=objects["data"],
            basis=objects["basis"],
            lambda_hlt=point.lambda_hlt,
            strict=hlt.strict,
            regulator=hlt.regulator,
            solver_backend=hlt.solver_backend,
            mp_dps=hlt.mp_dps,
            endpoint_value=getattr(hlt, "endpoint_value", None),
            omega_constraints=getattr(hlt, "omega_constraints", ()),
            quad_options=hlt_quad_options(hlt),
        )

        int_target = IntegratedHVPTarget.from_high_energy(
            am_tau=am_cut,
            delta=delta_lattice,
            p=hlt.p,
            euclidean_window=point.kernel_name,
            window_delta=derived["window_delta"],
            a_lat_inv_MeV=ens.a_lat_inv_MeV,
            kernel_quadrature_order=hlt.kernel_quadrature_order,
            window_quad_options=window_quad_options(hlt),
        )

        sharp_int_target = IntegratedHVPTarget.from_sharp_high_energy(
            am_tau=am_cut,
            p=hlt.p,
            euclidean_window=point.kernel_name,
            window_delta=derived["window_delta"],
            a_lat_inv_MeV=ens.a_lat_inv_MeV,
            kernel_quadrature_order=hlt.kernel_quadrature_order,
            window_quad_options=window_quad_options(hlt),
        )

        int_result = reconstruct_with_optional_mismatch_arg(
            reconstructor,
            int_target,
            compute_mismatch=hlt.compute_mismatch,
        )

        syst_kernel = int_result.estimate_kernel_systematic_on_grid(
            omega=objects["omega_lat_cut"],
            rho_p=objects["rho_p_cut"],
        )

        syst_total = int_result.estimate_total_sharp_systematic_on_grid(
            omega=objects["omega_lat_cut"],
            rho_p=objects["rho_p_cut"],
            sharp_target=sharp_int_target,
        )

        noise_envelope = estimate_r_ratio_gaussian_noise_envelope(
            int_result=int_result,
            sharp_target=sharp_int_target,
            omega_lat_cut=objects["omega_lat_cut"],
            rho_p_cut=objects["rho_p_cut"],
            n_samples=get_systematic_int(
                syst_cfg, "r_ratio_noise_n_samples", 100
            ),
            relative_sigma=get_systematic_float(
                syst_cfg, "r_ratio_noise_relative_sigma", 0.05
            ),
            correlation_length_MeV=get_systematic_float(
                syst_cfg, "r_ratio_noise_correlation_length_MeV", 100.0
            ),
            random_seed=get_systematic_int(
                syst_cfg, "r_ratio_noise_random_seed", 12345
            ),
            preserve_integral=get_systematic_bool(
                syst_cfg, "r_ratio_noise_preserve_integral", True
            ),
            require_positive=get_systematic_bool(
                syst_cfg, "r_ratio_noise_require_positive", True
            ),
            a_lat_inv_MeV=ens.a_lat_inv_MeV,
        )

        row = fill_success_row(
            row=row,
            int_result=int_result,
            syst_kernel=syst_kernel,
            syst_total=syst_total,
            ens=ens,
            noise_envelope=noise_envelope,
        )

    except Exception as exc:
        row["status"] = "failed"
        row["error_message"] = repr(exc)
        row["traceback"] = traceback.format_exc()

    return row


def run_scan_group(
    points: list[ScanPoint],
    *,
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
    scan: ScanConfig,
    syst_cfg: SystematicConfig,
    data_dir_str: str,
) -> list[dict[str, Any]]:
    """Run all lambdas/discretizations sharing one MP target preparation."""
    if not points:
        return []
    rows_by_index: dict[int, dict[str, Any]] = {}
    objects_by_ensemble: dict[int, dict[str, Any]] = {}
    try:
        data_dir = Path(data_dir_str)
        for ensemble_index in dict.fromkeys(point.ensemble_index for point in points):
            ens = ensembles[ensemble_index]
            objects_by_ensemble[ensemble_index] = load_ensemble_objects(
                data_dir=data_dir,
                ens=ens,
                hlt=hlt,
                syst_cfg=syst_cfg,
            )

        first_point = points[0]
        first_ens = ensembles[first_point.ensemble_index]
        first_objects = objects_by_ensemble[first_point.ensemble_index]
        first_basis = first_objects["basis"]
        for point in points[1:]:
            other = objects_by_ensemble[point.ensemble_index]
            other_basis = other["basis"]
            if not (
                np.array_equal(first_basis.times, other_basis.times)
                and first_basis.p == other_basis.p
                and first_basis.T == other_basis.T
                and first_basis.periodic == other_basis.periodic
                and first_basis.omega0 == other_basis.omega0
            ):
                raise ValueError(
                    "scan grouping found incompatible basis objects; split this group"
                )
            if not (
                np.array_equal(
                    first_objects["omega_lat_cut"], other["omega_lat_cut"]
                )
                and np.array_equal(first_objects["rho_p_cut"], other["rho_p_cut"])
            ):
                raise ValueError(
                    "scan grouping found incompatible systematic grids; split this group"
                )

        derived = derived_parameters(first_ens)
        am_cut = first_point.time_like_cut_MeV / first_ens.a_lat_inv_MeV
        delta_lattice = first_point.delta_factor * derived["m_pi"]
        int_target = IntegratedHVPTarget.from_high_energy(
            am_tau=am_cut,
            delta=delta_lattice,
            p=hlt.p,
            euclidean_window=first_point.kernel_name,
            window_delta=derived["window_delta"],
            a_lat_inv_MeV=first_ens.a_lat_inv_MeV,
            kernel_quadrature_order=hlt.kernel_quadrature_order,
            window_quad_options=window_quad_options(hlt),
        )
        sharp_int_target = IntegratedHVPTarget.from_sharp_high_energy(
            am_tau=am_cut,
            p=hlt.p,
            euclidean_window=first_point.kernel_name,
            window_delta=derived["window_delta"],
            a_lat_inv_MeV=first_ens.a_lat_inv_MeV,
            kernel_quadrature_order=hlt.kernel_quadrature_order,
            window_quad_options=window_quad_options(hlt),
        )

        lambda_sources = list(dict.fromkeys(str(point.lambda_hlt) for point in points))
        template = HLTReconstructor(
            data=first_objects["data"],
            basis=first_basis,
            lambda_hlt=lambda_sources[0],
            strict=hlt.strict,
            regulator=hlt.regulator,
            solver_backend=hlt.solver_backend,
            mp_dps=hlt.mp_dps,
            endpoint_value=getattr(hlt, "endpoint_value", None),
            omega_constraints=getattr(hlt, "omega_constraints", ()),
            quad_options=hlt_quad_options(hlt),
        )
        if template.numeric_backend != "mpmath":
            raise ValueError(
                "grouped scan execution requires solver_backend='mpmath'; "
                "use the legacy point runner for NumPy scans"
            )
        prepared_basis_key = (
            "prepared_mp_basis",
            id(first_basis),
            hlt.mp_dps,
        )
        reuse_basis_from = _WORKER_CACHE.get(prepared_basis_key)
        prepared = template.prepare(
            int_target,
            compute_mismatch=hlt.compute_mismatch,
            reuse_basis_from=reuse_basis_from,
        )
        if reuse_basis_from is None:
            _WORKER_CACHE[prepared_basis_key] = prepared
        solve_data = first_objects["data"] if hlt.regulator == "covariance" else None
        solutions = prepared.solve_many(lambda_sources, data=solve_data)
        solution_index = {source: index for index, source in enumerate(lambda_sources)}

        omega = np.asarray(first_objects["omega_lat_cut"], dtype=float)
        rho_nominal = np.asarray(first_objects["rho_p_cut"], dtype=float)
        weights = _trapezoidal_point_weights(omega)
        kernel_values = evaluate_scan_kernels_mpmath(
            prepared=prepared,
            solutions=solutions,
            sharp_target=sharp_int_target,
            omega=omega,
        )

        noise_key = (
            "r_ratio_noise_batch",
            id(first_objects["omega_lat_cut"]),
            id(first_objects["rho_p_cut"]),
            get_systematic_int(syst_cfg, "r_ratio_noise_n_samples", 100),
            get_systematic_float(
                syst_cfg, "r_ratio_noise_relative_sigma", 0.05
            ),
            get_systematic_float(
                syst_cfg, "r_ratio_noise_correlation_length_MeV", 100.0
            ),
            get_systematic_int(syst_cfg, "r_ratio_noise_random_seed", 12345),
            get_systematic_bool(
                syst_cfg, "r_ratio_noise_preserve_integral", True
            ),
            get_systematic_bool(
                syst_cfg, "r_ratio_noise_require_positive", True
            ),
        )
        noise_batch = _WORKER_CACHE.get(noise_key)
        if noise_batch is None:
            noise_batch = build_r_ratio_noise_batch(
                omega_lat=omega,
                rho_nominal=rho_nominal,
                n_samples=get_systematic_int(
                    syst_cfg, "r_ratio_noise_n_samples", 100
                ),
                relative_sigma=get_systematic_float(
                    syst_cfg, "r_ratio_noise_relative_sigma", 0.05
                ),
                correlation_length_MeV=get_systematic_float(
                    syst_cfg, "r_ratio_noise_correlation_length_MeV", 100.0
                ),
                random_seed=get_systematic_int(
                    syst_cfg, "r_ratio_noise_random_seed", 12345
                ),
                preserve_integral=get_systematic_bool(
                    syst_cfg, "r_ratio_noise_preserve_integral", True
                ),
                require_positive=get_systematic_bool(
                    syst_cfg, "r_ratio_noise_require_positive", True
                ),
                a_lat_inv_MeV=first_ens.a_lat_inv_MeV,
            )
            _WORKER_CACHE[noise_key] = noise_batch

        systematics_by_lambda: dict[str, tuple[Any, Any, dict[str, float]]] = {}
        for source, index in solution_index.items():
            reconstructed = kernel_values["reconstructed"][:, index]
            nominal_kernel_metrics = _systematic_metrics_batch(
                reference=kernel_values["smooth"],
                reconstructed=reconstructed,
                rho=rho_nominal,
                weights=weights,
                difference=kernel_values["kernel_difference"][:, index],
            )
            nominal_total_metrics = _systematic_metrics_batch(
                reference=kernel_values["sharp"],
                reconstructed=reconstructed,
                rho=rho_nominal,
                weights=weights,
                difference=kernel_values["total_difference"][:, index],
            )
            noise_kernel_metrics = _systematic_metrics_batch(
                reference=kernel_values["smooth"],
                reconstructed=reconstructed,
                rho=noise_batch["rho"],
                weights=weights,
                difference=kernel_values["kernel_difference"][:, index],
            )
            noise_total_metrics = _systematic_metrics_batch(
                reference=kernel_values["sharp"],
                reconstructed=reconstructed,
                rho=noise_batch["rho"],
                weights=weights,
                difference=kernel_values["total_difference"][:, index],
            )
            systematics_by_lambda[source] = (
                nominal_kernel_metrics,
                nominal_total_metrics,
                _noise_envelope_from_batches(
                    kernel_metrics=noise_kernel_metrics,
                    total_metrics=noise_total_metrics,
                    noise_batch=noise_batch,
                ),
            )

        for point in points:
            ens = ensembles[point.ensemble_index]
            objects = objects_by_ensemble[point.ensemble_index]
            row = base_row(
                point=point,
                ens=ens,
                hlt=hlt,
                scan=scan,
                syst_cfg=syst_cfg,
                derived=derived_parameters(ens),
                omega0=objects["omega0"],
            )
            try:
                source = str(point.lambda_hlt)
                solution = solutions[solution_index[source]]
                int_result = solution.evaluate(objects["data"], strict=hlt.strict)
                kernel_metrics, total_metrics, noise_envelope = (
                    systematics_by_lambda[source]
                )
                row = fill_success_row(
                    row=row,
                    int_result=int_result,
                    syst_kernel=_nominal_systematic(
                        kernel_metrics, int_result.error
                    ),
                    syst_total=_nominal_systematic(total_metrics, int_result.error),
                    ens=ens,
                    noise_envelope=noise_envelope,
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error_message"] = repr(exc)
                row["traceback"] = traceback.format_exc()
            rows_by_index[point.index] = row
    except Exception as exc:
        failure_traceback = traceback.format_exc()
        for point in points:
            ens = ensembles[point.ensemble_index]
            objects = objects_by_ensemble.get(point.ensemble_index)
            row = base_row(
                point=point,
                ens=ens,
                hlt=hlt,
                scan=scan,
                syst_cfg=syst_cfg,
                derived=derived_parameters(ens),
                omega0=None if objects is None else objects["omega0"],
            )
            row["status"] = "failed"
            row["error_message"] = repr(exc)
            row["traceback"] = failure_traceback
            rows_by_index[point.index] = row
    return [rows_by_index[index] for index in sorted(rows_by_index)]


def make_scan_points(ensembles: tuple[EnsembleConfig, ...], scan: ScanConfig) -> list[ScanPoint]:
    raw: list[tuple[int, str, float, float, Any]] = []
    for ensemble_index, _ens in enumerate(ensembles):
        for kernel_name in scan.kernel_names:
            for time_like_cut_MeV in scan.time_like_cut_MeV_array:
                for delta_factor in scan.delta_array:
                    for lambda_hlt in scan.lambda_array:
                        raw.append(
                            (
                                ensemble_index,
                                kernel_name,
                                float(time_like_cut_MeV),
                                float(delta_factor),
                                lambda_hlt,
                            )
                        )

    n_total = len(raw)
    return [
        ScanPoint(
            index=i + 1,
            n_total=n_total,
            ensemble_index=ensemble_index,
            kernel_name=kernel_name,
            time_like_cut_MeV=time_like_cut_MeV,
            delta_factor=delta_factor,
            lambda_hlt=lambda_hlt,
        )
        for i, (ensemble_index, kernel_name, time_like_cut_MeV, delta_factor, lambda_hlt) in enumerate(raw)
    ]


def _scan_group_key(
    point: ScanPoint,
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
) -> tuple[Any, ...]:
    """Key all scan rows that share one physical MP basis geometry.

    Identity-regulated LL/LC rows share the same basis, target and coefficients.
    A covariance regulator remains discretization-specific because its matrix
    enters the solve.
    """
    ens = ensembles[point.ensemble_index]
    discretization_key = (
        ens.discretization.lower() if hlt.regulator == "covariance" else None
    )
    return (
        ens.name,
        ens.resolved_jks_filename(),
        ens.a_lat_inv_MeV,
        ens.m_pi_MeV,
        ens.ZV,
        ens.T,
        ens.L_lattice,
        ens.basis_t_min,
        ens.basis_t_max,
        ens.basis_stride,
        ens.nrec,
        hlt.p,
        hlt.regulator,
        discretization_key,
    )


def _target_group_key(point: ScanPoint) -> tuple[Any, ...]:
    return (
        point.kernel_name,
        point.time_like_cut_MeV,
        point.delta_factor,
    )


def make_scan_groups(
    points: list[ScanPoint],
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
) -> list[list[ScanPoint]]:
    groups: dict[tuple[Any, ...], list[ScanPoint]] = {}
    for point in points:
        groups.setdefault(_scan_group_key(point, ensembles, hlt), []).append(point)
    return list(groups.values())


def run_scan_task(
    points: list[ScanPoint],
    *,
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
    scan: ScanConfig,
    syst_cfg: SystematicConfig,
    data_dir_str: str,
) -> list[dict[str, Any]]:
    if hlt.solver_backend.lower() in {"mpmath", "mp", "arbitrary"}:
        target_groups: dict[tuple[Any, ...], list[ScanPoint]] = {}
        for point in points:
            target_groups.setdefault(_target_group_key(point), []).append(point)
        rows: list[dict[str, Any]] = []
        for target_points in target_groups.values():
            rows.extend(
                run_scan_group(
                    target_points,
                    ensembles=ensembles,
                    hlt=hlt,
                    scan=scan,
                    syst_cfg=syst_cfg,
                    data_dir_str=data_dir_str,
                )
            )
        return sorted(rows, key=lambda row: int(row["scan_index"]))
    return [
        run_one_scan_point(
            point,
            ensembles=ensembles,
            hlt=hlt,
            scan=scan,
            syst_cfg=syst_cfg,
            data_dir_str=data_dir_str,
        )
        for point in points
    ]


def all_fieldnames(
    first_point: ScanPoint,
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
    scan: ScanConfig,
    syst_cfg: SystematicConfig,
) -> list[str]:
    ens = ensembles[first_point.ensemble_index]
    row = base_row(
        point=first_point,
        ens=ens,
        hlt=hlt,
        scan=scan,
        syst_cfg=syst_cfg,
        derived=derived_parameters(ens),
        omega0=None,
    )
    return list(row.keys())


def write_metadata(
    path: Path,
    *,
    ensembles: tuple[EnsembleConfig, ...],
    hlt: GlobalHLTConfig,
    scan: ScanConfig,
    syst_cfg: SystematicConfig,
    n_rows: int,
    n_workers: int,
) -> None:
    metadata = {
        "ensembles": [asdict(e) for e in ensembles],
        "global_hlt": asdict(hlt),
        "scan": asdict(scan),
        "systematic": asdict(syst_cfg),
        "n_rows": n_rows,
        "n_workers": n_workers,
        "normalization": {
            "target_convention": "IntegratedHVPTarget direct omega-space HVP kernel; no target-time quadrature grid is used",
            "window_convention": "kernel_names FULL/ID/LD/SD are passed as euclidean_window; window_delta is passed in lattice units",
            "integrated_result_scale_factor": "0.5 * ZV**2, ensemble by ensemble",
            "rho_p_conversion": "rho_2 = R/(12*pi^2) * 2/ZV**2 * systblind, ensemble by ensemble",
            "final_noise_systematic": "syst_total_noise_final_scaled = max(abs(syst_total_noise_signed_min_scaled), abs(syst_total_noise_signed_max_scaled))",
            "final_total_error": "total_error_noise_scaled = sqrt(error_scaled**2 + syst_total_noise_final_scaled**2)",
            "r_ratio_gaussian_noise_envelope": "multiplicative Gaussian-random field R_noise=R*(1+sigma_R*eta); configurable sample count, RMS amplitude, energy correlation length and seed; optional exact preservation of integral R(E)dE; min/max saved over valid realizations",
            "omega0_convention": "nrec=0 uses Epipi=2*sqrt(m_pi^2+(2*pi/L)^2); nrec>=1 uses db.get(f'star/En-{nrec-1}').mean()[0]",
            "jks_filename_convention": "{ensemble_name}.jks; ll and lc entries for the same ensemble share the same file",
            "correlator_root_convention": "C_{discretization}_sub{nrec}, with discretization in {'ll','lc'}",
            "mp_scan_convention": "A, R, b and target norm are prepared once per basis/target group; identity-regulated q is shared by compatible ll/lc rows",
            "systematic_batch_convention": "target and reconstructed grid kernels are evaluated in mpmath once; external binary64 R-ratio samples are then contracted in batched NumPy operations",
            "mismatch_convention": "sqrt(q^T A q - 2 q^T b + ||target||^2), with the target norm integrated once per group",
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def print_progress(
    completed: int,
    total: int,
    n_success: int,
    n_failed: int,
    row: dict[str, Any],
    start_time: float,
) -> None:
    elapsed = time.time() - start_time

    if completed > 0:
        sec_per_point = elapsed / completed
        remaining = sec_per_point * (total - completed)
        estimated_total = sec_per_point * total
        eta = datetime.now() + timedelta(seconds=remaining)
        eta_str = eta.strftime("%Y-%m-%d %H:%M:%S")
    else:
        sec_per_point = math.nan
        remaining = math.nan
        estimated_total = math.nan
        eta_str = "unknown"

    print(
        f"[{completed:5d}/{total:5d}] "
        f"success={n_success:5d} failed={n_failed:5d} "
        f"elapsed={format_seconds(elapsed)} "
        f"remaining≈{format_seconds(remaining)} "
        f"total≈{format_seconds(estimated_total)} "
        f"avg={sec_per_point:.2f}s/pt "
        f"ETA={eta_str} "
        f"last: ens={row['ensemble']} disc={row['discretization']} nrec={row['nrec']} "
        f"kernel={row['kernel']} cut={row['time_like_cut_MeV']:g}MeV "
        f"delta={row['delta_factor']:g} lambda={row['lambda_hlt']:.0e} "
        f"status={row['status']} pid={row['worker_pid']}",
        flush=True,
    )


def run_scan(args: argparse.Namespace) -> None:
    ensembles = ENSEMBLES
    hlt = GLOBAL_HLT
    scan = SCAN
    syst_cfg = SYSTEMATIC

    if args.mp_dps is not None:
        hlt = replace(hlt, mp_dps=args.mp_dps)

    if args.n_workers < 1:
        raise ValueError("--n-workers must be >= 1")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points = make_scan_points(ensembles, scan)
    if args.max_scan_points is not None:
        points = points[: args.max_scan_points]

    if not points:
        raise ValueError("no scan points selected")

    if hlt.solver_backend.lower() in {"mpmath", "mp", "arbitrary"}:
        task_groups = make_scan_groups(points, ensembles, hlt)
    else:
        task_groups = [[point] for point in points]

    output_csv = output_dir / args.output_csv
    output_json = output_dir / args.output_json
    fieldnames = all_fieldnames(points[0], ensembles, hlt, scan, syst_cfg)

    print(
        f"Starting multi-ensemble HLT scan: {len(points)} point(s), "
        f"{len(task_groups)} reusable target group(s), "
        f"{len(ensembles)} ensemble row(s), {args.n_workers} worker(s).",
        flush=True,
    )
    print(f"Writing rows incrementally to {output_csv}", flush=True)

    start_time = time.time()
    completed = 0
    n_success = 0
    n_failed = 0
    rows: list[dict[str, Any]] = []

    def record_row(writer: csv.DictWriter, row: dict[str, Any]) -> None:
        nonlocal completed, n_success, n_failed
        writer.writerow(row)
        rows.append(row)
        completed += 1
        n_success += row["status"] == "success"
        n_failed += row["status"] == "failed"
        if args.verbose:
            print(
                f"Finished [{completed}/{len(points)}] "
                f"scan_index={row['scan_index']} ens={row['ensemble']} "
                f"disc={row['discretization']} nrec={row['nrec']} "
                f"kernel={row['kernel']} cut={row['time_like_cut_MeV']:g}MeV "
                f"delta={row['delta_factor']:g} "
                f"lambda={float(row['lambda_hlt']):.0e} "
                f"status={row['status']} pid={row['worker_pid']}",
                flush=True,
            )
        if args.progress_every > 0 and (
            completed % args.progress_every == 0 or completed == len(points)
        ):
            print_progress(
                completed, len(points), n_success, n_failed, row, start_time
            )

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        if args.n_workers == 1:
            for group_index, group in enumerate(task_groups, start=1):
                if args.verbose:
                    point = group[0]
                    ens = ensembles[point.ensemble_index]
                    print(
                        f"Running target group [{group_index}/{len(task_groups)}] "
                        f"with {len(group)} output row(s): ens={ens.name} "
                        f"nrec={ens.nrec} kernel={point.kernel_name} "
                        f"cut={point.time_like_cut_MeV:g}MeV "
                        f"delta={point.delta_factor:g}",
                        flush=True,
                    )
                group_rows = run_scan_task(
                    group,
                    ensembles=ensembles,
                    hlt=hlt,
                    scan=scan,
                    syst_cfg=syst_cfg,
                    data_dir_str=str(data_dir),
                )
                for row in group_rows:
                    record_row(writer, row)
                f.flush()

        else:
            with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
                future_to_group = {
                    executor.submit(
                        run_scan_task,
                        group,
                        ensembles=ensembles,
                        hlt=hlt,
                        scan=scan,
                        syst_cfg=syst_cfg,
                        data_dir_str=str(data_dir),
                    ): group
                    for group in task_groups
                }

                for future in as_completed(future_to_group):
                    group = future_to_group[future]
                    try:
                        group_rows = future.result()
                    except Exception as exc:
                        group_rows = []
                        failure_traceback = traceback.format_exc()
                        for point in group:
                            ens = ensembles[point.ensemble_index]
                            row = base_row(
                                point=point,
                                ens=ens,
                                hlt=hlt,
                                scan=scan,
                                syst_cfg=syst_cfg,
                                derived=derived_parameters(ens),
                                omega0=None,
                            )
                            row["status"] = "failed"
                            row["error_message"] = repr(exc)
                            row["traceback"] = failure_traceback
                            group_rows.append(row)
                    for row in group_rows:
                        record_row(writer, row)
                    f.flush()

    # Rewrite once in scan-index order so the final CSV is deterministic.
    rows.sort(key=lambda r: int(r["scan_index"]))
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_metadata(
        output_json,
        ensembles=ensembles,
        hlt=hlt,
        scan=scan,
        syst_cfg=syst_cfg,
        n_rows=len(rows),
        n_workers=args.n_workers,
    )

    elapsed = time.time() - start_time
    print(
        f"Finished scan in {format_seconds(elapsed)}. "
        f"Wrote {len(rows)} rows to {output_csv}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing ensemble JKS files and r_ratio_pdg.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="scan_output",
        help="Directory where scan results are written",
    )
    parser.add_argument(
        "--output-csv",
        default="multi_ensemble_hlt_scan_results.csv",
    )
    parser.add_argument(
        "--output-json",
        default="multi_ensemble_hlt_scan_metadata.json",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N completed scan points. Use 1 for every point.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every completed scan point.",
    )
    parser.add_argument(
        "--max-scan-points",
        type=int,
        default=None,
        help="Optional early stop for testing.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=1,
        help="Number of parallel worker processes. Use 1 for serial execution.",
    )
    parser.add_argument(
        "--mp-dps",
        type=int,
        default=None,
        help="Override GLOBAL_HLT.mp_dps from scan_inputs.py.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    run_scan(parse_args())
