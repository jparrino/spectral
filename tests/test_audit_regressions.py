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

"""Acceptance reproductions for the strict 0.1.0 findings (F01–F14)."""

from dataclasses import replace
import hashlib
import pickle

import mpmath as mp
import numpy as np
import pytest
import spectral as sp
from spectral.algorithms import hlt
from spectral.core.kernel import LinearCombination
from spectral.core.reconstruction import ReconstructionError
from spectral.core.result import Failure


def inputs(*, p=0, policy=None, mean=(1, 0.5), covariance=None):
    basis = sp.Basis(sp.kernels.exponential(), (1, 2), p=p)
    data = sp.CorrelatorData.from_arrays(
        times=basis.times,
        mean=mean,
        covariance=np.eye(2) * 0.01 if covariance is None else covariance,
    )
    return (
        basis,
        data,
        sp.Reconstructor(basis, hlt.Solver(0), policy or sp.numerics.numpy()),
    )


def echo(value):
    return value


def test_f01_exact_transport_both_directions():
    with mp.workdps(160):
        value = mp.sqrt(2)
        complex_value = mp.mpc(value, mp.pi)
    payload = {"nested": np.array([value, complex_value], dtype=object)}
    with mp.workdps(15), sp.execution.processes(2) as execution:
        actual = execution.map([(echo, (payload,))])[0]
        assert actual.error is None
        assert actual.value["nested"][0]._mpf_ == value._mpf_
        assert actual.value["nested"][1]._mpc_ == complex_value._mpc_
    # Transport must not globally patch mpmath's reducer.
    assert pickle.dumps(value)


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("partial", [False, True])
def test_f02_resume_source_graph(tmp_path, shared, partial):
    basis, data, _ = inputs()

    def ensembles(data):
        second = data.scaled(2) if shared else inputs()[1]
        return {
            "A": sp.scan.Ensemble(data, basis),
            "B": sp.scan.Ensemble(second, basis),
        }

    arguments = dict(
        algorithm=hlt.Solver(0), target=basis.at_time(1), checkpoint_dir=tmp_path
    )
    original = sp.scan.reconstruct(ensembles=ensembles(data), **arguments)
    if partial:
        (tmp_path / (hashlib.sha256(b"B").hexdigest()[:16] + ".spectral")).unlink()
    current = ensembles(inputs()[1])
    resumed = sp.scan.reconstruct(ensembles=current, resume=True, **arguments)
    fresh = sp.scan.reconstruct(
        ensembles=current, algorithm=hlt.Solver(0), target=basis.at_time(1)
    )
    for result in resumed.results:
        assert result.data.source is current[result.coordinates["ensemble"]].data.source
    if shared:
        np.testing.assert_allclose(
            np.asarray(resumed.statistical_covariance, float),
            np.asarray(fresh.statistical_covariance, float),
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            np.asarray(resumed.statistical_covariance, float),
            np.asarray(original.statistical_covariance, float),
            rtol=1e-14,
        )
    else:
        assert resumed.results[0].data.source.id != resumed.results[1].data.source.id
        with pytest.raises(ValueError, match="cross-source"):
            _ = resumed.statistical_covariance


@pytest.mark.parametrize("was_shared", [True, False])
def test_f02_incompatible_source_graph_rejected(tmp_path, was_shared):
    basis, data, _ = inputs()

    def run(shared, resume=False):
        other = data if shared else inputs()[1]
        return sp.scan.reconstruct(
            ensembles={
                "A": sp.scan.Ensemble(data, basis),
                "B": sp.scan.Ensemble(other, basis),
            },
            algorithm=hlt.Solver(0),
            target=basis.at_time(1),
            checkpoint_dir=tmp_path,
            resume=resume,
        )

    run(was_shared)
    with pytest.raises(ValueError, match="checkpoint settings differ"):
        run(not was_shared, True)


@pytest.mark.parametrize("workers", [0, 1, 2, 3])
def test_f04_neighboring_target_failure(workers):
    basis, data, _ = inputs(p=1)
    targets = sp.TargetGrid(
        {"target": ("valid", "invalid")},
        (basis.at_time(1), replace(basis.at_time(1), p=0)),
    )
    # Same convention, but a nonzero value at zero conflicts with the zero basis.
    from spectral.core.kernel import CallableKernel

    targets = replace(
        targets,
        targets=(basis.at_time(1), CallableKernel(nonzero_at_zero, spectral_power=1)),
    )
    recon = sp.Reconstructor(
        basis, hlt.Solver(0, constraints=(hlt.constraints.point(energy=0),))
    )
    execution = (
        sp.execution.serial() if not workers else sp.execution.processes(workers)
    )
    with execution:
        results = recon.reconstruct(
            data, target=targets, execution=execution, on_error="record"
        )
    assert not isinstance(results.results[0], Failure)
    assert isinstance(results.results[1], Failure)
    assert "zero basis vector" in results.results[1].error


def nonzero_at_zero(w, ctx):
    return ctx.exp(-w)


@pytest.mark.parametrize("attribute,value", [("energy_unit", "MeV"), ("p", 1)])
def test_f05_combination_preserves_conventions(attribute, value):
    basis, data, recon = inputs()
    if attribute == "energy_unit":
        basis = replace(basis, kernel=sp.kernels.exponential(energy_unit="GeV"))
        recon = replace(recon, basis=basis)
    target = replace(basis.at_time(1), **{attribute: value})
    wrapped = LinearCombination((target,), (1,))
    with pytest.raises(ReconstructionError, match="disagree"):
        recon.reconstruct(data, target=wrapped)


def test_f05_combination_integrability_and_incompatible_constituents():
    target = sp.targets.hvp(mass=".1", p=2)
    with sp.numerics.mpmath(40).context() as ctx:
        with pytest.raises(ValueError, match="diverges"):
            LinearCombination((target,), (1,)).integral(ctx)
    basis, _, _ = inputs()
    with pytest.raises(ValueError, match="conventions disagree"):
        LinearCombination((basis.at_time(1), replace(basis.at_time(1), p=1)), (1, 1))


def constant_model(w, ctx):
    if w < ctx.number(".23") or w > ctx.number("1.17"):
        raise AssertionError("evaluated outside model support")
    return ctx.number(1)


def test_f06_grid_domain_boundary_weights():
    basis, data, recon = inputs()
    result = recon.reconstruct(data, target=basis.at_time(1))
    model = sp.SpectralDensity.from_function(constant_model, domain=(".23", "1.17"))
    options = dict(reference_kernel=basis.at_time(2), density=model)
    adaptive = sp.systematics.estimate(result, **options)
    coarse = sp.systematics.estimate(result, energies=np.linspace(0, 2, 101), **options)
    fine = sp.systematics.estimate(result, energies=np.linspace(0, 2, 2001), **options)
    exact = float(adaptive.nominal_bias)
    assert abs(float(fine.nominal_bias) - exact) < 1e-7
    assert (
        abs(float(fine.nominal_bias) - exact)
        < abs(float(coarse.nominal_bias) - exact) / 100
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_f07_nonfinite_grid_and_report(value):
    basis, data, recon = inputs()
    result = recon.reconstruct(data, target=basis.at_time(1))
    with pytest.raises(ArithmeticError, match="non-finite"):
        sp.systematics.estimate(
            result,
            reference_kernel=basis.at_time(2),
            density=sp.SpectralDensity.from_function(lambda w, ctx: value),
            energies=[0.1, 0.2, 1],
        )
    with pytest.raises(ArithmeticError, match="non-finite"):
        report_for((result,), [[value]], {})


def test_f07_overflow_recorded():
    basis, data, recon = inputs(mean=("1e400", "1e400"))
    result = recon.reconstruct(data, target=basis.at_time(1), on_error="record")
    assert isinstance(result, Failure)
    assert "non-finite" in result.error


def report_for(results, biases, axes, policy=None):
    return sp.systematics.SystematicReport(
        tuple(r.id for r in results),
        axes,
        ("nominal",) + tuple(f"model/{i}" for i in range(1, len(biases))),
        np.asarray(biases, dtype=object),
        tuple(r.target_kernel for r in results),
        policy or sp.numerics.numpy(),
        {},
    )


def test_f08_envelope_ambient_precision_and_identity_scaling():
    basis, data, recon = inputs(policy=sp.numerics.mpmath(120))
    result = recon.reconstruct(data, target=basis.at_time(1))
    with mp.workdps(120):
        larger = mp.mpf(1) + mp.mpf("1e-90")
        precise = mp.sqrt(2)
        biases = [[mp.mpf(1)], [-larger]]
    report = report_for((result,), biases, {}, sp.numerics.mpmath(120))
    with mp.workdps(15):
        assert report.maximum_indices == (1,)
        assert report.max_abs_bias == larger
        data = sp.CorrelatorData.from_arrays(
            times=(1,), mean=[precise], covariance=[[precise]]
        )
        scaled = data.scaled(1)
        assert scaled.mean[0]._mpf_ == precise._mpf_
        assert scaled.covariance[0, 0]._mpf_ == precise._mpf_


@pytest.mark.parametrize(
    "outer,inner",
    [
        ("sharp_high_energy", "smooth_high_energy"),
        ("smooth_high_energy", "sharp_high_energy"),
    ],
)
def test_f10_nested_fixed_time_wrappers(outer, inner):
    basis, data, recon = inputs()

    def wrap(name, kernel):
        return getattr(sp.targets, name)(
            kernel=kernel,
            cutoff=".3",
            **({"width": ".1"} if name.startswith("smooth") else {}),
        )

    target = wrap(outer, wrap(inner, basis.at_time(1)))
    assert not target.is_time_dependent
    assert np.isfinite(float(recon.reconstruct(data, target=target).value))


def test_f11_metadata_does_not_fragment_groups():
    basis, data, recon = inputs()
    result = recon.reconstruct(
        data, target=basis.at_time(1), metadata={"ensemble": "A"}
    )
    results = tuple(
        replace(
            result,
            id=str(i),
            coordinates={"lambda": i},
            metadata={"ensemble": "A", "provenance": {"nested": [i]}, "incidental": i},
        )
        for i in range(2)
    )
    selected = sp.scan.select(
        results, systematics=report_for(results, [[0, 0]], {"lambda": (0, 1)})
    )
    assert selected.groups == ({"ensemble": "A"},)
    assert len(selected.results) == 1


def test_f12_cached_failure_honors_current_policy(tmp_path):
    basis, data, _ = inputs()
    options = dict(
        ensembles={"A": sp.scan.Ensemble(data, basis)},
        algorithm=hlt.Solver(0),
        target=replace(basis.at_time(1), p=1),
        checkpoint_dir=tmp_path,
    )
    first = sp.scan.reconstruct(**options, on_error="record")
    assert first.failures
    with pytest.raises(ReconstructionError):
        sp.scan.reconstruct(**options, resume=True, on_error="raise")


def test_f13_missing_resamples_before_dependency_import(monkeypatch):
    basis, data, recon = inputs()
    result = recon.reconstruct(data, target=basis.at_time(1))
    monkeypatch.setattr(
        sp.io.jks, "_jks", lambda: pytest.fail("dependency must not be imported")
    )
    with pytest.raises(ValueError, match="original JKS resamples"):
        sp.io.jks.to_jks(result)


@pytest.mark.parametrize("operation", ["selection", "csv"])
def test_f14_one_envelope_scan_per_operation(monkeypatch, tmp_path, operation):
    basis, data, recon = inputs()
    result = recon.reconstruct(data, target=basis.at_time(1))
    results = tuple(
        replace(result, id=str(i), coordinates={"lambda": i}) for i in range(25)
    )
    report = report_for(results, np.zeros((5, 25)), {"lambda": tuple(range(25))})
    descriptor = type(report).maximum_indices
    calls = []

    def counted(self):
        calls.append(self.biases.size)
        return descriptor.fget(self)

    monkeypatch.setattr(type(report), "maximum_indices", property(counted))
    if operation == "selection":
        sp.scan.select(results, systematics=report)
    else:
        sp.io.write_csv(tmp_path / "results.csv", results=results, systematics=report)
    assert calls == [125]


def test_f02_shared_identity_requires_consistent_source_contents():
    basis, data, _ = inputs()
    inconsistent = replace(
        data, covariance=np.eye(2), source=replace(data.source, covariance=np.eye(2))
    )
    with pytest.raises(ValueError, match="inconsistent data"):
        sp.scan.reconstruct(
            ensembles={
                "A": sp.scan.Ensemble(data, basis),
                "B": sp.scan.Ensemble(inconsistent, basis),
            },
            algorithm=hlt.Solver(0),
            target=basis.at_time(1),
        )


def test_f02_fresh_scan_does_not_require_archivable_source_metadata():
    basis, data, _ = inputs()
    data = replace(
        data, source=replace(data.source, info={"application_object": object()})
    )
    result = sp.scan.reconstruct(
        ensembles={"A": sp.scan.Ensemble(data, basis)},
        algorithm=hlt.Solver(0),
        target=basis.at_time(1),
    )
    assert not result.failures
