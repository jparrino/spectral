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

"""Small multi-ensemble scan with parallel execution and native checkpoints."""

import numpy as np
import spectral as sp
from spectral.algorithms import hlt


def main():
    times = (1, 2, 3, 4)
    basis = sp.Basis(sp.kernels.exponential(), times, omega_0=".2")
    ensembles = {}
    for name, amplitude in (("A", 1), ("B", 1.2)):
        mean = amplitude * np.exp(-0.4 * np.array(times))
        data = sp.CorrelatorData.from_arrays(
            times=times, mean=mean, covariance=np.diag((0.01 * mean) ** 2)
        )
        ensembles[name] = sp.scan.Ensemble(data, basis, {"discretization": "ll"})
    targets = sp.targets.grid(
        sp.targets.smooth_high_energy,
        vary={"width": [".05", ".1"]},
        kernel=basis.at_time(2),
        cutoff=".4",
    )
    with sp.execution.processes(workers=2) as execution:
        result = sp.scan.reconstruct(
            ensembles=ensembles,
            algorithm=hlt.Solver(["1e-6", "1e-3"]),
            target=targets,
            execution=execution,
            checkpoint_dir="example_output/checkpoints",
        )
    sp.io.save("example_output/ensembles.spectral", results=result, overwrite=True)
    print(result.axes)
    print(result.values)


if __name__ == "__main__":
    main()
