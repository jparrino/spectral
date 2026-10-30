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

import numpy as np
import pytest
import spectral as sp
from spectral.algorithms import hlt


def setup():
    basis = sp.Basis(sp.kernels.exponential(), (1, 2, 3))
    data = sp.CorrelatorData.from_arrays(
        times=basis.times, mean=[1, 0.5, 0.2], covariance=np.eye(3) * 0.001
    )
    targets = sp.targets.grid(
        sp.targets.smooth_high_energy,
        vary={"width": [".05", ".1"]},
        kernel=basis.at_time(2),
        cutoff=".4",
    )
    recon = sp.Reconstructor(basis, hlt.Solver(["1e-6", "1e-3"]), sp.numerics.numpy())
    return recon, data, targets


def test_parallel_width_lambda_scan_matches_serial():
    recon, data, targets = setup()
    serial = recon.reconstruct(data, target=targets)
    with sp.execution.processes(2) as execution:
        parallel = recon.reconstruct(data, target=targets, execution=execution)
    assert parallel.axes == {"width": (".05", ".1"), "lambda": ("1e-6", "1e-3")}
    np.testing.assert_allclose(
        np.asarray(parallel.coefficients, float),
        np.asarray(serial.coefficients, float),
        rtol=1e-13,
    )
    assert parallel.select(width="0.05", lambda_=".001").coordinates == {
        "width": ".05",
        "lambda": "1e-3",
    }


def test_mpmath_parallel_lambda_scan():
    recon, data, _ = setup()
    recon = sp.Reconstructor(recon.basis, recon.algorithm, sp.numerics.mpmath(40))
    serial = recon.reconstruct(data, target=recon.basis.at_time(4))
    with sp.execution.processes(2) as execution:
        parallel = recon.reconstruct(
            data, target=recon.basis.at_time(4), execution=execution
        )
    assert [r.value for r in serial.results] == [r.value for r in parallel.results]


def test_ensemble_scan_and_checkpoint(tmp_path):
    recon, data, targets = setup()
    ensembles = {
        "A": sp.scan.Ensemble(data, recon.basis, {"discretization": "ll"}),
        "B": sp.scan.Ensemble(data.scaled(2), recon.basis, {"discretization": "lc"}),
    }
    first = sp.scan.reconstruct(
        ensembles=ensembles,
        algorithm=recon.algorithm,
        target=targets,
        checkpoint_dir=tmp_path,
    )
    second = sp.scan.reconstruct(
        ensembles=ensembles,
        algorithm=recon.algorithm,
        target=targets,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert first.axes == second.axes
    assert [r.id for r in first.results] == [r.id for r in second.results]
    np.testing.assert_allclose(
        np.asarray(first.select(ensemble="B").values, float),
        2 * np.asarray(first.select(ensemble="A").values, float),
    )
    # Simulate a new process rebuilding the same input arrays with fresh source IDs.
    fresh = sp.CorrelatorData.from_arrays(
        times=data.times, mean=data.mean, covariance=data.covariance
    )
    restarted = {
        "A": sp.scan.Ensemble(fresh, recon.basis, {"discretization": "ll"}),
        "B": sp.scan.Ensemble(fresh.scaled(2), recon.basis, {"discretization": "lc"}),
    }
    resumed = sp.scan.reconstruct(
        ensembles=restarted,
        algorithm=recon.algorithm,
        target=targets,
        checkpoint_dir=tmp_path,
        resume=True,
    )
    assert [r.id for r in resumed.results] == [r.id for r in first.results]
    restarted["A"] = sp.scan.Ensemble(
        fresh.scaled(3), recon.basis, {"discretization": "ll"}
    )
    with pytest.raises(ValueError, match="checkpoint settings differ"):
        sp.scan.reconstruct(
            ensembles=restarted,
            algorithm=recon.algorithm,
            target=targets,
            checkpoint_dir=tmp_path,
            resume=True,
        )


def test_width_scan_reuses_factorizations(monkeypatch):
    recon, data, targets = setup()
    from spectral.algorithms.hlt import linear

    original = linear.factor
    calls = []

    def counted(matrix, ctx):
        calls.append(matrix.shape)
        return original(matrix, ctx)

    monkeypatch.setattr(linear, "factor", counted)
    result = recon.reconstruct(data, target=targets)
    assert len(result.results) == 4
    assert len(calls) == 2  # One matrix factorization per lambda, shared across widths.


def test_checkpoint_rejects_previous_numerical_revision(tmp_path, monkeypatch):
    from spectral.numerics.policy import Backend
    from spectral.scan import runner

    basis = sp.Basis(sp.kernels.exponential(), (10000,), p=-1, omega_0="0.0001")
    data = sp.CorrelatorData.from_arrays(
        times=basis.times, mean=[1], covariance=[[0.01]]
    )
    options = dict(
        ensembles={"A": sp.scan.Ensemble(data, basis)},
        algorithm=hlt.Solver(".001"),
        target=basis.at_time(10000),
        checkpoint_dir=tmp_path,
    )
    current_integral = Backend.exp_integral

    def previous_integral(self, power, rate, lower):
        power, rate, lower = map(self.number, (power, rate, lower))
        if not self.is_mp and power <= -1:
            return self.integrate(lambda w: w**power * self.exp(-rate * w), lower)
        return current_integral(self, power, rate, lower)

    # Reproduce a v3 checkpoint containing the old, silently incorrect result.
    with monkeypatch.context() as old:
        old.setattr(
            runner, "_CHECKPOINT_REVISION", "numerical-validation-v3", raising=False
        )
        old.setattr(Backend, "exp_integral", previous_integral)
        stale = sp.scan.reconstruct(**options)
    assert abs(float(stale.results[0].value)) < 1e-10
    path = next(tmp_path.glob("*.spectral"))
    original_bytes = path.read_bytes()

    with pytest.raises(ValueError, match="checkpoint settings differ"):
        sp.scan.reconstruct(**options, resume=True)
    assert path.read_bytes() == original_bytes

    # Explicit regeneration replaces stale data; current checkpoints still resume.
    fresh = sp.scan.reconstruct(**options, resume=False)
    assert fresh.results[0].value == pytest.approx(0.9999973331077184, rel=1e-14)
    resumed = sp.scan.reconstruct(**options, resume=True)
    assert resumed.results[0].id == fresh.results[0].id
    assert resumed.results[0].value == fresh.results[0].value
