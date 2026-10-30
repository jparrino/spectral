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

"""Self-contained correlated JKS reconstruction and systematic export example."""

from pathlib import Path
import getpass
import os
import numpy as np
import jks
import spectral as sp
from spectral.algorithms import hlt


def main():
    times = np.arange(1, 9)
    rng = np.random.default_rng(2026)
    truth = np.exp(-0.4 * times) + 0.2 * np.exp(-1.1 * times)
    rows = {
        f"toy:{i:04d}": (
            truth * (1 + rng.normal(0, 0.02)) + rng.normal(0, 0.002, len(times))
        ).tolist()
        for i in range(40)
    }
    database = jks.resamples()
    database.add("C", jks.measurements(rows).jackknife().prepare(lambda m: m.mean()))
    data = sp.io.jks.read_correlator(
        database, observable="C", times=times
    ).select_times([1, 2, 3, 4])
    basis = sp.Basis(sp.kernels.exponential(), data.times)
    reconstructor = sp.Reconstructor(basis, hlt.Solver("1e-8"), sp.numerics.mpmath(50))
    prediction = reconstructor.reconstruct(
        data, target=sp.targets.correlator(basis=basis), target_times=[5, 6, 7]
    )
    observable = sp.io.jks.to_jks(prediction, source=database)
    observable = sp.io.jks.with_systematic(
        observable,
        tag="finite_volume/toy",
        shift=0.01 * np.asarray(prediction.values, float),
    )
    database.add("C_predicted", observable)
    print("Predictions:", observable.mean())
    print("Statistical covariance:\n", np.asarray(observable.cov()))
    print("Total covariance:\n", np.asarray(observable.tcov()))
    output = Path("example_output/jks")
    output.mkdir(parents=True, exist_ok=True)
    sp.io.save(output / "prediction.spectral", results=prediction, overwrite=True)
    # JKS 1.1.10 uses os.getlogin() for provenance; a headless terminal may
    # not have a controlling login session. Restore it after the save.
    login = os.getlogin
    try:
        try:
            login()
        except OSError:
            os.getlogin = getpass.getuser
        database.save(str(output / "analysis.jks"))
    finally:
        os.getlogin = login


if __name__ == "__main__":
    main()
