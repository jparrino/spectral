#!/usr/bin/env python3
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

"""Scan every configured ensemble with spectral.scan; paths default to this folder."""

import argparse
import csv
from dataclasses import asdict, replace
from importlib import import_module
import json
from pathlib import Path
import sys
import time

import numpy as np

APP = Path(__file__).resolve().parent
ROOT = APP.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import spectral as sp  # noqa: E402
from spectral.algorithms import hlt  # noqa: E402
from spectral.core.grid import TargetGrid  # noqa: E402
from spectral.core.result import Failure  # noqa: E402

# Keep copied applications independent, including in spawned worker processes.
config = import_module(f"applications.{APP.name}.scan_inputs")
systematics = import_module(f"applications.{APP.name}.systematics")


def target_grid(ens, settings, scan):
    axes = {
        "kernel": scan.kernel_names,
        "time_like_cut_MeV": scan.time_like_cut_MeV_array,
        "delta_factor": scan.delta_array,
    }
    targets = []
    for name in scan.kernel_names:
        kernel = sp.targets.hvp_window(
            name=name,
            a_lat_inv_MeV=ens.a_lat_inv_MeV,
            p=settings.p,
        )
        if name.upper() != "FULL":
            # The demo passes this width explicitly, using hbar*c = 197.327.
            kernel = replace(
                kernel,
                window=replace(
                    kernel.window,
                    width=0.15 / (197.327 / ens.a_lat_inv_MeV),
                ),
            )
        for cut in scan.time_like_cut_MeV_array:
            for delta in scan.delta_array:
                targets.append(
                    sp.targets.smooth_high_energy(
                        kernel=kernel,
                        cutoff=cut / ens.a_lat_inv_MeV,
                        width=delta * (ens.m_pi_MeV / ens.a_lat_inv_MeV),
                    )
                )
    return TargetGrid(axes, tuple(targets))


def prepare_inputs(data_dir, ensembles, settings, scan, numerics):
    import jks

    databases, prepared, algorithms, targets, definitions = {}, {}, {}, {}, {}
    for ens in ensembles:
        disc = ens.discretization.lower()
        key = f"{ens.name}/{disc}/sub{ens.nrec}"
        if key in prepared:
            raise ValueError(f"Duplicate ensemble/discretization/nrec entry: {key}")
        if disc not in {"ll", "lc"} or ens.nrec < 0:
            raise ValueError(f"Invalid discretization or nrec for {key}")
        if not 0 < ens.basis_t_min <= ens.basis_t_max < ens.T or ens.basis_stride < 1:
            raise ValueError(f"Invalid periodic basis times for {key}")
        path = data_dir / ens.resolved_jks_filename()
        if path not in databases:
            databases[path] = jks.resamples(str(path))
        db = databases[path]
        observable = f"C_{disc}_sub{ens.nrec}"
        correlator = db.get(observable)
        mean = np.asarray(correlator.mean(), float)
        covariance = np.asarray(correlator.cov(), float)
        if ens.nrec:
            source = f"star/En-{ens.nrec - 1}"
            omega0 = float(np.asarray(db.get(source).mean()).reshape(-1)[0])
        else:
            source = "Epipi"
            omega0 = 2 * np.sqrt(
                (ens.m_pi_MeV / ens.a_lat_inv_MeV) ** 2
                + (2 * np.pi / ens.L_lattice) ** 2
            )
        data = sp.CorrelatorData.from_arrays(
            times=tuple(range(len(mean))),
            mean=mean,
            covariance=covariance,
            energy_unit="lattice",
        ).select_times(range(ens.basis_t_min, ens.basis_t_max + 1, ens.basis_stride))
        basis = sp.Basis(
            kernel=sp.kernels.periodic_correlator(extent=ens.T, energy_unit="lattice"),
            times=data.times,
            p=settings.p,
            omega_0=omega0,
        )
        prepared[key] = sp.scan.Ensemble(
            data,
            basis,
            {
                **asdict(ens),
                "correlator_root": observable,
                "omega0_root": source,
                "omega0": omega0,
            },
        )
        # Match legacy area_constraint="auto": HVP area diverges at p >= 4.
        constraints = [hlt.constraints.area()] if settings.p < 4 else []
        if settings.endpoint_value is not None:
            constraints.append(
                hlt.constraints.point(
                    energy=basis.omega_0,
                    value=settings.endpoint_value,
                )
            )
        with numerics.context() as ctx:
            covariance_norm = ctx.number(mean[0]) ** 2
        algorithms[key] = hlt.Solver(
            lambda_=scan.lambda_array,
            regulator=settings.regulator,
            covariance_norm=covariance_norm,
            constraints=constraints,
        )
        targets[key] = target_grid(ens, settings, scan)
        definitions[key] = ens
    return prepared, algorithms, targets, definitions


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run_scan(
    data_dir,
    output_dir,
    *,
    ensembles=config.ENSEMBLES,
    settings=config.GLOBAL_HLT,
    scan=config.SCAN,
    systematic=config.SYSTEMATIC,
    workers=1,
    resume=False,
    check_inputs=False,
):
    start = time.perf_counter()
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    expected = list(
        dict.fromkeys(
            [e.resolved_jks_filename() for e in ensembles]
            + [systematic.r_ratio_filename]
        )
    )
    missing = [
        str(data_dir / name) for name in expected if not (data_dir / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Copy the required data files into the data directory:\n"
            + "\n".join(missing)
        )
    if workers < 1 or not ensembles:
        raise ValueError("Provide at least one worker and one ensemble.")
    numerics = sp.numerics.Numerics(
        backend=settings.solver_backend,
        dps=settings.mp_dps,
        maxdegree=settings.mp_quadrature_maxdegree,
    )
    prepared, algorithms, targets, definitions = prepare_inputs(
        data_dir,
        ensembles,
        settings,
        scan,
        numerics,
    )
    energy, ratio = systematics.load_r_ratio(
        data_dir / systematic.r_ratio_filename,
        systematic.r_ratio_max_points,
    )
    for key, item in prepared.items():
        grid = energy / definitions[key].a_lat_inv_MeV
        systematics.weights(grid[grid >= float(item.basis.omega_0)])
    npoints = (
        len(ensembles) * len(scan.kernel_names) * len(scan.time_like_cut_MeV_array)
    )
    npoints *= len(scan.delta_array) * len(scan.lambda_array)
    print(
        f"Validated {len(prepared)} ensemble entries; {npoints} HLT scan points.",
        flush=True,
    )
    if check_inputs:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "ensembles": [asdict(e) for e in ensembles],
        "global_hlt": asdict(settings),
        "scan": asdict(scan),
        "systematic": asdict(systematic),
        "data_dir": str(data_dir.resolve()),
        "n_workers": workers,
        "numerics": asdict(numerics),
        "conventions": {
            "scale": "0.5 * ZV**2 (both ll and lc, matching demo)",
            "rho_p": "R/(12*pi**2) * 2/ZV**2 * systfac / omega_lat**(p-2)",
            "systematic": "max absolute signed sharp-target bias over noise samples",
            "noise_normalization": "R-weighted RMS on the grid above omega0",
            "integrals": "HLT: omega0 to infinity; systematics: finite R-ratio grid",
            "area_constraint": "enabled for p<4; disabled for divergent HVP area at p>=4",
        },
    }
    execution = (
        sp.execution.serial() if workers == 1 else sp.execution.processes(workers)
    )
    with execution:
        print("Preparing HVP targets and solving the lambda scan...", flush=True)
        results = sp.scan.reconstruct(
            ensembles=prepared,
            algorithm=algorithms,
            target=targets,
            numerics=numerics,
            execution=execution,
            on_error="record",
            checkpoint_dir=output_dir / "checkpoints",
            resume=resume,
        )
        sp.io.save(
            output_dir / "scan.spectral",
            results=results,
            metadata=metadata,
            overwrite=True,
        )
        print("Evaluating nominal and distorted R-ratio systematics...", flush=True)
        tasks, successful = [], {}
        for key, ens in definitions.items():
            items = tuple(
                r
                for r in results.select(ensemble=key).results
                if not isinstance(r, Failure)
            )
            if items:
                successful[key] = items
                tasks.append(
                    (systematics.evaluate, (items, ens, energy, ratio, systematic))
                )
        biases, systematic_errors = {}, {}
        for (key, items), outcome in zip(successful.items(), execution.map(tasks)):
            if outcome.error:
                systematic_errors[key] = outcome.error
            else:
                biases.update((r.id, row) for r, row in zip(items, outcome.value))
    rows = []
    for index, result in enumerate(results.results, 1):
        key = result.coordinates["ensemble"]
        ens = definitions[key]
        coordinates = result.coordinates
        scale = 0.5 * ens.ZV**2
        row = {
            "scan_index": index,
            "scan_total": len(results.results),
            **asdict(ens),
            "ensemble": ens.name,
            "discretization": ens.discretization.lower(),
            "kernel": coordinates["kernel"],
            "time_like_cut_MeV": coordinates["time_like_cut_MeV"],
            "delta_factor": coordinates["delta_factor"],
            "lambda_hlt": coordinates["lambda"],
            "omega0": prepared[key].basis.omega_0,
            "correlator_root": prepared[key].metadata["correlator_root"],
            "omega0_root": prepared[key].metadata["omega0_root"],
            "scale_factor": scale,
            "endpoint_value": settings.endpoint_value,
            "p": settings.p,
            "regulator": settings.regulator,
            "status": "success",
            "error_message": "",
        }
        row["jks_filename"] = ens.resolved_jks_filename()
        row.pop("name")
        if isinstance(result, Failure) or key in systematic_errors:
            row.update(
                status="failed",
                error_message=(
                    result.error
                    if isinstance(result, Failure)
                    else systematic_errors[key]
                ),
            )
        else:
            value, error = float(result.value), float(result.statistical_error)
            row.update(biases[result.id])
            row.update(
                estimate_raw=value,
                error_raw=error,
                variance_raw=float(result.variance),
                estimate_scaled=scale * value,
                error_scaled=scale * error,
                variance_scaled=scale**2 * float(result.variance),
                total_error_noise_raw=float(
                    np.hypot(error, row["syst_total_noise_final_raw"])
                ),
                total_error_noise_scaled=float(
                    np.hypot(
                        scale * error,
                        row["syst_total_noise_final_scaled"],
                    )
                ),
                mismatch_abs=float(
                    np.sqrt(float(result.diagnostics["kernel_mismatch_squared"]))
                ),
                stationarity_rel=float(result.diagnostics["stationarity_residual"]),
                max_abs_q=max(abs(float(q)) for q in result.coefficients),
            )
        rows.append(row)
    write_csv(output_dir / "multi_ensemble_hlt_scan_results.csv", rows)
    metadata.update(
        n_results=len(rows),
        n_success=sum(r["status"] == "success" for r in rows),
        elapsed_seconds=time.perf_counter() - start,
    )
    (output_dir / "multi_ensemble_hlt_scan_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
    )
    print(
        f"Saved {metadata['n_success']}/{len(rows)} successful points to {output_dir}."
    )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=APP / "data")
    parser.add_argument("--output-dir", type=Path, default=APP / "scan_output")
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--mp-dps", type=int)
    parser.add_argument(
        "--resume", action="store_true", help="Reuse matching scanner checkpoints."
    )
    parser.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate files, observables, basis times and spectral coverage; do not scan.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run the first ensemble, window, cut, width and two lambdas.",
    )
    args = parser.parse_args()
    settings = config.GLOBAL_HLT
    if args.mp_dps is not None:
        settings = replace(settings, mp_dps=args.mp_dps)
    ensembles, scan = config.ENSEMBLES, config.SCAN
    if args.smoke_test:
        ensembles = ensembles[:1]
        scan = replace(
            scan,
            kernel_names=scan.kernel_names[:1],
            time_like_cut_MeV_array=scan.time_like_cut_MeV_array[:1],
            delta_array=scan.delta_array[:1],
            lambda_array=scan.lambda_array[:2],
        )
    try:
        rows = run_scan(
            args.data_dir,
            args.output_dir,
            ensembles=ensembles,
            settings=settings,
            scan=scan,
            workers=args.n_workers,
            resume=args.resume,
            check_inputs=args.check_inputs,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    if any(row["status"] != "success" for row in rows):
        parser.exit(1, "Some scan points failed; see error_message in the CSV.\n")


if __name__ == "__main__":
    main()
