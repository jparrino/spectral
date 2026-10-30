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

"""Demo-compatible R-ratio sampling and batched, finite-grid bias estimates.

This adapter deliberately retains the demo's Gaussian precursor covariance,
R-weighted RMS normalization, and noise-only envelope. These differ from the
library's general-purpose gaussian_deformations prescription.
"""

from dataclasses import replace
import math

import numpy as np


def weights(x):
    x = np.asarray(x, float)
    if len(x) < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("The integration grid needs at least two increasing energies.")
    return np.r_[np.diff(x)[0], x[2:] - x[:-2], np.diff(x)[-1]] / 2


def load_r_ratio(path, max_points):
    energy, ratio = np.loadtxt(
        path, delimiter=",", skiprows=1, usecols=(0, 3), unpack=True
    )
    order = np.argsort(energy, kind="stable")
    energy, ratio = energy[order], ratio[order]
    unique = np.r_[True, np.diff(energy) > 0]
    energy, ratio = energy[unique][:max_points], ratio[unique][:max_points]
    valid = np.isfinite(energy) & np.isfinite(ratio) & (energy > 0) & (ratio >= 0)
    energy, ratio = energy[valid], ratio[valid]
    weights(energy)
    return 1000 * energy, ratio  # CSV energies are GeV; noise lengths are MeV.


def noise_spectra(omega, rho, a_inv, p, cfg):
    """Return distorted spectra and diagnostics, reusing the demo's random stream."""
    energy = omega * a_inv
    measure = weights(energy) * np.maximum(rho * omega ** (p - 2), 0)
    measure_sum = measure.sum()
    lengths = cfg.r_ratio_noise_correlation_lengths_MeV
    sigma = cfg.r_ratio_noise_relative_sigma
    count = cfg.r_ratio_noise_n_samples
    if (
        not lengths
        or len(set(lengths)) != len(lengths)
        or any(not np.isfinite(x) or x < 0 for x in lengths)
        or count < 1
        or not np.isfinite(sigma)
        or sigma < 0
        or not np.isfinite(measure_sum)
        or measure_sum <= 0
    ):
        raise ValueError("Invalid R-ratio noise settings or zero spectral weight.")
    samples, indices, sample_lengths, diagnostics = [], [], [], []
    seeds = np.random.SeedSequence(cfg.r_ratio_noise_random_seed).spawn(len(lengths))
    for length_index, (length, seed) in enumerate(zip(lengths, seeds)):
        covariance = (
            np.eye(len(energy))
            if length == 0
            else np.exp(-0.5 * ((energy[:, None] - energy[None, :]) / length) ** 2)
        )
        if length:
            covariance.flat[:: len(energy) + 1] += 1e-12
        factor = np.linalg.cholesky(covariance)
        rng = np.random.default_rng(seed)
        n_before = len(samples)
        for sample in range(count):
            for _ in range(100):
                eta = factor @ rng.normal(0, 1, len(energy))
                if cfg.r_ratio_noise_preserve_integral:
                    eta -= np.sum(measure * eta) / measure_sum
                rms2 = np.sum(measure * eta**2) / measure_sum
                if not np.isfinite(rms2) or rms2 <= 1e-20:
                    continue
                deformation = 1 + sigma * eta / math.sqrt(rms2)
                if cfg.r_ratio_noise_require_positive and np.any(deformation <= 0):
                    continue
                samples.append(rho * deformation)
                indices.append(length_index * count + sample)
                sample_lengths.append(length)
                diagnostics.append(
                    (
                        np.sum(measure * deformation) / measure_sum,
                        deformation.min(),
                        deformation.max(),
                    )
                )
                break
        if len(samples) == n_before:
            raise ValueError(f"All R-ratio samples rejected for length {length} MeV.")
    diagnostics = np.asarray(diagnostics)
    info = {
        "syst_noise_n_requested": count * len(lengths),
        "syst_noise_n_valid": len(samples),
        "syst_noise_n_rejected": count * len(lengths) - len(samples),
        "syst_noise_n_correlation_lengths": len(lengths),
        "syst_noise_n_samples_per_length": count,
        "syst_noise_relative_sigma": sigma,
        "syst_noise_random_seed": cfg.r_ratio_noise_random_seed,
        "syst_noise_preserve_integral": cfg.r_ratio_noise_preserve_integral,
        "syst_noise_require_positive": cfg.r_ratio_noise_require_positive,
        "syst_noise_correlation_length_MeV": lengths[0]
        if len(lengths) == 1
        else math.nan,
        "syst_noise_correlation_length_min_MeV": min(lengths),
        "syst_noise_correlation_length_max_MeV": max(lengths),
        "syst_noise_integral_ratio_min": diagnostics[:, 0].min(),
        "syst_noise_integral_ratio_max": diagnostics[:, 0].max(),
        "syst_noise_factor_min": diagnostics[:, 1].min(),
        "syst_noise_factor_max": diagnostics[:, 2].max(),
    }
    return np.asarray(samples), np.asarray(indices), np.asarray(sample_lengths), info


def metrics(reference, reconstructed, difference, spectra, measure):
    target = spectra @ (measure * reference).T
    signed = spectra @ (measure * difference).T
    absolute = np.abs(spectra) @ (measure * np.abs(difference)).T
    denominator = np.maximum(np.abs(target), np.finfo(float).eps)
    return {
        "signed": signed,
        "absolute": absolute,
        "relative_signed": signed / denominator,
        "relative_absolute": absolute / denominator,
        "target_observable": target,
        "reconstructed_observable": spectra @ (measure * reconstructed).T,
    }


def evaluate(results, ens, energy_MeV, ratio, cfg):
    """Compute smooth-kernel and sharp-target biases for a whole ensemble scan."""
    basis, policy = results[0].basis, results[0].numerics
    omega = energy_MeV / ens.a_lat_inv_MeV
    rho = ratio / (12 * np.pi**2) * (2 / ens.ZV**2) * cfg.systfac
    rho = rho / omega ** (float(basis.p) - 2)
    keep = omega >= float(basis.omega_0)
    omega, rho = omega[keep], rho[keep]
    measure = weights(omega)
    samples, indices, lengths, info = noise_spectra(
        omega,
        rho,
        ens.a_lat_inv_MeV,
        float(basis.p),
        cfg,
    )
    spectra = np.vstack([rho, samples])
    # Form kernels and subtract at reconstruction precision before casting.
    with policy.context() as ctx:
        grid = ctx.array(omega)
        phi = ctx.array([basis.values(w, ctx) for w in grid]).T
        reconstructed_mp = ctx.array([r.coefficients for r in results]) @ phi
        cache = {}

        def values(kernel):
            key = repr(kernel)
            if key not in cache:
                cache[key] = ctx.array([kernel.scalar(w, ctx) for w in grid])
            return cache[key]

        smooth = ctx.array([values(r.target_kernel.kernel) for r in results])
        sharp = ctx.array(
            [values(replace(r.target_kernel.kernel, width=None)) for r in results]
        )
        differences = {
            "kernel": np.asarray(reconstructed_mp - smooth, float),
            "total": np.asarray(reconstructed_mp - sharp, float),
        }
        reconstructed = np.asarray(reconstructed_mp, float)
    rows = [dict(info) for _ in results]
    scale = 0.5 * ens.ZV**2
    for kind, reference in (("kernel", smooth), ("total", sharp)):
        batch = metrics(
            np.asarray(reference, float),
            reconstructed,
            differences[kind],
            spectra,
            measure,
        )
        if any(not np.isfinite(v).all() for v in batch.values()):
            raise ArithmeticError("Non-finite R-ratio systematic estimate.")
        for j, row in enumerate(rows):
            for metric, values_array in batch.items():
                prefix = f"syst_{kind}_{metric}"
                dimensional = metric in {"signed", "absolute"}
                row[prefix + ("_raw" if dimensional else "")] = values_array[0, j]
                if dimensional:
                    row[prefix + "_scaled"] = scale * values_array[0, j]
                noise_values = values_array[1:, j]  # Match demo: nominal is separate.
                for edge, index in (
                    ("min", np.argmin(noise_values)),
                    ("max", np.argmax(noise_values)),
                ):
                    prefix = f"syst_{kind}_noise_{metric}"
                    row[f"{prefix}_{edge}_raw"] = noise_values[index]
                    row[f"{prefix}_sample_index_at_{edge}"] = indices[index]
                    if dimensional:
                        row[f"{prefix}_{edge}_scaled"] = scale * noise_values[index]
            signed = batch["signed"][1:, j]
            winner = np.argmax(np.abs(signed))
            row[f"syst_noise_{kind}_winning_sample_index"] = indices[winner]
            row[f"syst_noise_{kind}_winning_correlation_length_MeV"] = lengths[winner]
            row[f"syst_{kind}_noise_final_raw"] = abs(signed[winner])
            row[f"syst_{kind}_noise_final_scaled"] = scale * abs(signed[winner])
    return rows
