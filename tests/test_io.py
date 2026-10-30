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


def test_exact_archive_and_shared_source(tmp_path):
    basis = sp.Basis(sp.kernels.exponential(), (1, 2, 3))
    data = sp.CorrelatorData.from_arrays(
        times=basis.times, mean=["1.000000000000000000000001", ".5", ".2"]
    )
    result = sp.Reconstructor(
        basis, hlt.Solver([0, "1e-12"]), sp.numerics.mpmath(60)
    ).reconstruct(data, target=basis.at_time(2))
    path = tmp_path / "result.spectral"
    sp.io.save(path, results=result)
    loaded = sp.io.load(path).results
    assert loaded.results[0].value._mpf_ == result.results[0].value._mpf_
    assert loaded.results[0].data.source is loaded.results[1].data.source
    for a, b in zip(result.results, loaded.results):
        assert all(x == y for x, y in zip(a.coefficients, b.coefficients))
        assert b.algorithm == a.algorithm
        assert b.target_kernel(".7") == a.target_kernel(".7")
    sp.io.write_csv(tmp_path / "summary.csv", results=loaded)
    assert (
        "1.000" not in (tmp_path / "summary.csv").read_text()
    )  # the selected target is time 2
    with pytest.raises(FileExistsError):
        sp.io.save(path, results=result)


@pytest.mark.parametrize("selected_times", [[1, 2], [2, 1], [1]])
def test_jks_reads_all_timeslices_then_selects(selected_times):
    jks = pytest.importorskip("jks")
    rows = {
        f"A:{i:04d}": [1 + i * 0.01, 0.5 + i * 0.02, 0.2 - i * 0.001]
        for i in range(20)
    }
    correlator = jks.measurements(rows).jackknife().prepare(lambda m: m.mean())
    database = jks.resamples()
    database.add("C(t).r", correlator)

    data = sp.io.jks.read_correlator(database, "C(t).r")
    assert data.times == (0, 1, 2)
    np.testing.assert_array_equal(data.mean, correlator.mean())
    np.testing.assert_array_equal(data.covariance, correlator.cov())

    selected = data.select_times(np.array(selected_times))
    assert selected.times == tuple(selected_times)
    np.testing.assert_array_equal(
        selected.mean, np.asarray(correlator.mean())[selected_times]
    )
    np.testing.assert_array_equal(
        selected.covariance,
        np.asarray(correlator.cov())[np.ix_(selected_times, selected_times)],
    )
    assert selected.source is data.source
    np.testing.assert_array_equal(selected.source.resamples, correlator.blocks)
    np.testing.assert_array_equal(selected.projection, np.eye(3)[selected_times])


def test_jks_preserves_full_covariance_and_named_shifts():
    jks = pytest.importorskip("jks")
    rows = {
        f"A:{i:04d}": [1 + i * 0.01, 0.5 + i * 0.02, 0.2 - i * 0.001] for i in range(20)
    }
    correlator = jks.measurements(rows).jackknife().prepare(lambda m: m.mean())
    database = jks.resamples()
    database.add("C", correlator)
    data = sp.io.jks.read_correlator(database, observable="C", times=[1, 2, 3])
    basis = sp.Basis(sp.kernels.exponential(), (1, 2, 3))
    results = sp.Reconstructor(
        basis, hlt.Solver([0, ".001"]), sp.numerics.mpmath(50)
    ).reconstruct(data, target=sp.targets.correlator(basis=basis), target_times=[1, 2])
    exported = sp.io.jks.to_jks(results, source=database)
    np.testing.assert_allclose(
        exported.cov(),
        np.asarray(results.statistical_covariance, float),
        rtol=1e-10,
        atol=1e-20,
    )
    database.add("predictions", exported)
    cross = database.apply(lambda r: [r["C"][0], r["predictions"][0]])
    cov = np.asarray(cross.cov())
    assert cov[0, 1] == pytest.approx(cov[0, 0], rel=1e-13)
    shift = np.array([0.1, -0.2, 0.3, -0.4])
    enriched = sp.io.jks.with_systematic(exported, tag="finite_volume", shift=shift)
    np.testing.assert_allclose(enriched.cov("finite_volume"), np.outer(shift, shift))
    np.testing.assert_array_equal(enriched.mean(), exported.mean())
    np.testing.assert_array_equal(enriched.cov(), exported.cov())
    with pytest.raises(ValueError, match="already exists"):
        sp.io.jks.with_systematic(enriched, tag="finite_volume", shift=shift)


def test_jks_refuses_missing_resamples():
    data = sp.CorrelatorData.from_arrays(times=[1], mean=[1])
    basis = sp.Basis(sp.kernels.exponential(), (1,))
    result = sp.Reconstructor(basis, hlt.Solver(0)).reconstruct(
        data, target=basis.at_time(1)
    )
    with pytest.raises(ValueError, match="original JKS resamples"):
        sp.io.jks.to_jks(result)


def test_joint_systematic_export_survives_native_reload(tmp_path):
    jks = pytest.importorskip("jks")
    rows = {
        f"ensemble:{i}": [1 + i * 0.01, 0.5 - i * 0.002, 0.2 + i * 0.003]
        for i in range(20)
    }
    source = jks.resamples()
    source.add("C", jks.measurements(rows).jackknife().prepare(lambda m: m.mean()))
    data = sp.io.jks.read_correlator(source, observable="C").select_times([1, 2])
    basis = sp.Basis(sp.kernels.exponential(), data.times)
    results = sp.Reconstructor(
        basis, hlt.Solver(".001"), sp.numerics.mpmath(45)
    ).reconstruct(data, target=sp.targets.correlator(basis=basis), target_times=[2, 3])
    model = sp.models.TabulatedSpectralDensity((0, 1, 2), (1, 2, 1))
    report = sp.systematics.estimate(
        results, reference_kernel=sp.targets.correlator(basis=basis), density=model
    )
    path = tmp_path / "joint.spectral"
    sp.io.save(path, results=results, systematics=report)
    loaded = sp.io.load(path)
    assert loaded.systematics.models[0] == model
    np.testing.assert_array_equal(loaded.systematics.biases, report.biases)
    exported = sp.io.jks.to_jks(loaded.results)
    np.testing.assert_allclose(
        exported.cov(), np.asarray(results.statistical_covariance, float), rtol=1e-10
    )
    sp.io.jks.add_named(
        source,
        dict(zip(("prediction_2", "prediction_3"), loaded.results.results)),
        systematics=loaded.systematics,
        systematic_tag="reconstruction",
    )
    joint = source.apply(lambda r: [r["prediction_2"][0], r["prediction_3"][0]])
    shifts = -np.asarray(report.bias_at_maximum, float)
    np.testing.assert_allclose(
        joint.cov("reconstruction"), np.outer(shifts, shifts), rtol=1e-12
    )
    np.testing.assert_allclose(joint.cov(), exported.cov(), rtol=1e-12)


def test_jks_export_rejects_changed_source():
    jks = pytest.importorskip("jks")
    rows = {f"A:{i}": [1 + i * 0.01, 0.5 + i * 0.02] for i in range(10)}
    observable = jks.measurements(rows).jackknife().prepare(lambda m: m.mean())
    source = jks.resamples()
    source.add("C", observable)
    data = sp.io.jks.read_correlator(source, observable="C", times=[1, 2])
    basis = sp.Basis(sp.kernels.exponential(), data.times)
    result = sp.Reconstructor(basis, hlt.Solver(0)).reconstruct(
        data, target=basis.at_time(1)
    )
    changed = jks.resamples()
    changed.add("C", source.apply(lambda r: [2 * x for x in r["C"]]))
    with pytest.raises(ValueError, match="source mean changed"):
        sp.io.jks.to_jks(result, source=changed)
