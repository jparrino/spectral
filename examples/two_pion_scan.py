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

"""Run from the project root after installation: python examples/two_pion_scan.py.

Use --workers 2 for process parallelism, or --backend mpmath for a complete
arbitrary-precision calculation. Outputs are written only to --output.
"""

from argparse import ArgumentParser
from pathlib import Path
import spectral as sp
from spectral.algorithms import hlt


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--backend", choices=["numpy", "mpmath"], default="numpy")
    parser.add_argument("--dps", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("example_output/two_pion"))
    args = parser.parse_args()
    numerics = sp.numerics.Numerics(args.backend, dps=args.dps)
    model = sp.models.TwoPionModel()
    times = (1, 2, 3, 4)
    basis = sp.Basis(
        sp.kernels.exponential(energy_unit="GeV"), times, p=2, omega_0=model.domain[0]
    )
    with numerics.context() as ctx:
        mean = ctx.array(
            [model.integrate(basis.kernel.at_time(t), numerics=numerics) for t in times]
        )
        sigma = mean * ctx.number(".005")
        covariance = ctx.array(
            [
                [
                    sigma[i] * sigma[j] * (1 if i == j else ctx.number(".95"))
                    for j in range(len(times))
                ]
                for i in range(len(times))
            ]
        )
        lower = ctx.number(model.domain[0])
        deformation_grid = [
            lower + (ctx.number("2.5") - lower) * i / 30 for i in range(31)
        ]
        integration_grid = [
            lower + (ctx.number(5) - lower) * i / 1000 for i in range(1001)
        ]
    data = sp.CorrelatorData(times, mean, covariance, energy_unit="GeV")
    solver = hlt.Solver(["0", "1e-20", "1e-8", "1e-4"], regulator="identity")
    reconstructor = sp.Reconstructor(basis, solver, numerics)
    targets = sp.targets.grid(
        sp.targets.smooth_high_energy,
        vary={"width": [".05", ".15"]},
        kernel=basis.at_time(2),
        cutoff=".7",
    )
    sharp = sp.targets.sharp_high_energy(kernel=basis.at_time(2), cutoff=".7")
    # The small reproducible set is shared across every scan point.
    deformations = sp.systematics.gaussian_deformations(
        density=model,
        energies=deformation_grid,
        count=4,
        relative_sigma=".1",
        correlation_length=".15",
        seed=2026,
        numerics=numerics,
    )
    execution = (
        sp.execution.serial()
        if args.workers == 1
        else sp.execution.processes(args.workers)
    )
    with execution:
        prediction = reconstructor.reconstruct(
            data,
            target=sp.targets.correlator(basis=basis),
            target_times=[2, 5],
            execution=execution,
        )
        scan = reconstructor.reconstruct(data, target=targets, execution=execution)
        # Explicit grid integration makes the demonstration quick. Adaptive
        # integration is the default when energies is omitted.
        report = sp.systematics.estimate(
            scan,
            reference_kernel=sharp,
            density=model,
            deformations=deformations,
            energies=integration_grid,
            execution=execution,
        )
    args.output.mkdir(parents=True, exist_ok=True)
    sp.io.save(
        args.output / "scan.spectral", results=scan, systematics=report, overwrite=True
    )
    sp.io.write_csv(
        args.output / "scan.csv", results=scan, systematics=report, overwrite=True
    )
    print(
        "Exact input-time recovery at lambda=0:",
        prediction.at_lambda("0").at_time(2).value,
    )
    print("Forward-transform reference:", mean[1])
    print("Scan dimensions:", scan.axes)
    print("Saved:", args.output / "scan.spectral")


if __name__ == "__main__":
    main()
