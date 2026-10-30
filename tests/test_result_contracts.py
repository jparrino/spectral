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

"""Public scientific-state contracts, including non-HLT algorithm extensions."""

from dataclasses import replace
import pickle

import numpy as np
import pytest

import spectral as sp
from spectral.algorithms import hlt
from spectral.core.result import Failure
from spectral.execution import _pack


def sample(policy=None):
    basis = sp.Basis(sp.kernels.exponential(), (1, 2))
    data = sp.CorrelatorData.from_arrays(
        times=(1, 2), mean=(1, 0.5), covariance=np.eye(2)
    )
    return sp.Reconstructor(
        basis, hlt.Solver(0), policy or sp.numerics.numpy()
    ).reconstruct(data, target=sp.targets.correlator(basis=basis), target_times=(1, 2))


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", 100),
        ("variance", 100),
        ("coefficients", [1]),
        ("coefficients", [[1, 0]]),
    ],
)
def test_inconsistent_result_changes_rejected(field, value):
    result = sample().at_time(1)
    with pytest.raises(ValueError, match="result"):
        replace(result, **{field: value})


def test_basis_data_order_must_match():
    result = sample().at_time(1)
    with pytest.raises(ValueError, match="times"):
        replace(result, data=result.data.select_times((2, 1)))


def test_scalar_inputs_cannot_mutate_a_validated_result():
    original = sample().at_time(1)
    value, variance = np.array(original.value), np.array(original.variance)
    result = replace(original, value=value, variance=variance)
    value[...] = 100
    variance[...] = 100
    assert result.value == original.value
    assert result.variance == original.variance
    with pytest.raises(ValueError, match="scalars"):
        replace(original, value=[original.value], variance=[original.variance])


@pytest.mark.parametrize(
    "original_policy,new_policy",
    [
        (sp.numerics.numpy(), sp.numerics.mpmath(70)),
        (sp.numerics.mpmath(40), sp.numerics.mpmath(70)),
        (sp.numerics.mpmath(70), sp.numerics.numpy()),
    ],
)
def test_changed_precision_applies_to_both_stored_kernels(
    tmp_path, original_policy, new_policy
):
    basis = sp.Basis(sp.kernels.exponential(), (1,))
    data = sp.CorrelatorData.from_arrays(times=(1,), mean=(1,), covariance=[[1]])
    original = sp.Reconstructor(basis, hlt.Solver(0), original_policy).reconstruct(
        data, target=basis.at_time(1)
    )
    changed = replace(original, numerics=new_policy)
    assert changed.id != original.id
    path = tmp_path / "changed-precision.spectral"
    sp.io.save(path, results=changed)
    restored = sp.io.load(path).results
    assert restored.id == changed.id
    for result in (changed, restored):
        assert result.target_kernel.numerics == new_policy
        with new_policy.context() as ctx:
            expected = basis.at_time(1).scalar(ctx.number(".4"), ctx)
        assert result.target_kernel(".4") == expected
        assert result.approximated_kernel(".4") == expected


@pytest.mark.parametrize("policy", [sp.numerics.numpy(), sp.numerics.mpmath(70)])
def test_tiny_values_have_no_unit_scale_validation_floor(policy):
    original = sample(policy).at_time(1)
    data = original.data.scaled("1e-30")
    result = sp.Reconstructor(original.basis, hlt.Solver(0), policy).reconstruct(
        data, target=original.basis.at_time(1)
    )
    with pytest.raises(ValueError, match="value"):
        replace(result, value=result.value * 2)
    with pytest.raises(ValueError, match="variance"):
        replace(result, variance=result.variance * 2)


def test_collection_rejects_permuted_missing_and_failure_coordinates():
    results = sample()
    with pytest.raises(ValueError, match="axis order"):
        sp.ResultCollection(results.axes, tuple(reversed(results.results)))
    with pytest.raises(ValueError, match="axis order"):
        sp.ResultCollection({"other": (1, 2)}, results.results)
    with pytest.raises(ValueError, match="axis order"):
        sp.ResultCollection(
            results.axes,
            (
                results.results[0],
                Failure({"target_time": 1}, "failed"),
            ),
        )
    with pytest.raises(ValueError, match="constant"):
        sp.ResultCollection(
            {"case": ("a", "b")},
            tuple(
                replace(r, coordinates={**r.coordinates, "case": c})
                for r, c in zip(results.results, ("a", "b"))
            ),
        )


def test_subsets_keep_results_and_fixed_coordinates():
    results = sample()
    expanded = sp.ResultCollection(
        {"case": ("a", "b"), **results.axes},
        tuple(
            replace(r, coordinates={**r.coordinates, "case": case})
            for case in ("a", "b")
            for r in results.results
        ),
    )
    subset = expanded.select(case="a")
    assert subset.at_time(2) is expanded.results[1]
    assert subset.at_time(2).coordinates == {"case": "a", "target_time": 2}


def test_scientific_change_gets_new_identity_and_rejects_old_report():
    result = sample().at_time(1)
    report = sp.systematics.SystematicReport(
        (result.id,),
        {},
        ("nominal",),
        [[0]],
        (result.target_kernel,),
        result.numerics,
        {},
    )
    changed = replace(
        result,
        coefficients=result.coefficients * 2,
        value=result.value * 2,
        variance=result.variance * 4,
    )
    assert changed.id != result.id
    assert replace(result, target_kernel=result.basis.at_time(2)).id != result.id
    assert replace(result, metadata={"note": "annotated"}).id == result.id
    assert replace(result, coordinates={"case": "copy"}).id == result.id
    with pytest.raises(ValueError, match="do not belong"):
        sp.scan.select(changed, systematics=report)


def test_rebinding_only_preserves_identity_for_equivalent_sources():
    result = sample().at_time(1)
    fresh = sp.CorrelatorData.from_arrays(
        times=result.data.times,
        mean=result.data.mean,
        covariance=result.data.covariance,
    )
    assert replace(result, data=fresh).id != result.id
    rebound = result.rebind_data(fresh)
    assert rebound.id == result.id
    assert rebound.data.source is fresh.source
    with pytest.raises(ValueError, match="equivalent"):
        result.rebind_data(fresh.scaled(2))


def test_mappings_are_detached_and_recursively_immutable():
    original = {"provenance": {"stages": ["input", "scan"]}}
    result = replace(sample().at_time(1), metadata=original)
    original["provenance"]["stages"].append("later")
    assert result.metadata["provenance"]["stages"] == ("input", "scan")
    with pytest.raises(TypeError):
        result.metadata["provenance"]["stages"] = ()
    with pytest.raises(TypeError):
        result.coordinates["target_time"] = 10
    with pytest.raises(TypeError):
        result.diagnostics["new"] = 1
    results = sample()
    with pytest.raises(TypeError):
        results.axes["target_time"] = (10, 20)
    with pytest.raises(TypeError):
        Failure({"target_time": 1}, "error").coordinates["target_time"] = 2


def test_grid_report_and_policy_mappings_are_immutable():
    result = sample().at_time(1)
    grid = sp.TargetGrid({"width": [1]}, (result.target_kernel,))
    report = sp.systematics.SystematicReport(
        (result.id,),
        {"width": [1]},
        ("nominal",),
        [[0]],
        (result.target_kernel,),
        result.numerics,
        {"integration": {"bounds": [0, 1]}},
    )
    ranges = {"lambda": [0, 1]}
    policy = sp.scan.StatDominantMinimum(
        ranges=ranges, window_ranges={"ID": {"width": [0, 1]}}
    )
    ranges["lambda"][0] = 10
    assert policy.ranges["lambda"] == (0, 1)
    for mapping, key in (
        (grid.axes, "width"),
        (report.axes, "width"),
        (report.metadata["integration"], "bounds"),
        (policy.ranges, "lambda"),
        (policy.window_ranges["ID"], "width"),
    ):
        with pytest.raises(TypeError):
            mapping[key] = ()


def test_source_ensemble_and_deformation_metadata_are_frozen():
    result = sample().at_time(1)
    source = replace(result.data.source, info={"nested": {"stages": [1]}})
    ensemble = sp.scan.Ensemble(
        result.data, result.basis, metadata={"nested": {"stages": [1]}}
    )
    deformations = sp.systematics.DeformationSet((), (), {"nested": {"stages": [1]}})
    for metadata in (source.info, ensemble.metadata, deformations.metadata):
        with pytest.raises(TypeError):
            metadata["nested"]["stages"] = ()


def test_report_rejects_ambiguous_ids_and_mismatched_axes():
    result = sample().at_time(1)
    args = dict(
        result_ids=(result.id, result.id),
        axes={"case": (1, 2)},
        model_ids=("nominal",),
        biases=[[0, 0]],
        reference_kernels=(result.target_kernel,) * 2,
        numerics=result.numerics,
        metadata={},
    )
    with pytest.raises(ValueError, match="unique"):
        sp.systematics.SystematicReport(**args)
    with pytest.raises(ValueError, match="axes"):
        sp.systematics.SystematicReport(**{**args, "axes": {}})


def test_older_archive_without_identity_snapshot_retains_id():
    from spectral.io.archive import Encoder, Decoder

    result = sample().at_time(1)
    encoded = Encoder().encode(result)
    del encoded["fields"]["_identity_state"]
    restored = Decoder().decode(encoded)
    assert restored.id == result.id
    assert replace(restored, metadata={"restored": True}).id == result.id
    with pytest.raises(ValueError, match="variance"):
        replace(restored, variance=100)


@pytest.mark.parametrize("policy", [sp.numerics.numpy(), sp.numerics.mpmath(70)])
def test_identity_and_immutability_survive_archive_and_process_transport(
    tmp_path, policy
):
    results = sample(policy)
    path = tmp_path / "results.spectral"
    sp.io.save(path, results=results)
    for restored in (sp.io.load(path).results, pickle.loads(_pack(results))):
        assert [r.id for r in restored.results] == [r.id for r in results.results]
        assert (
            replace(restored.at_time(1), metadata={"a": 1}).id == results.at_time(1).id
        )
        np.testing.assert_allclose(
            np.asarray(restored.statistical_covariance, float),
            np.diag([float(r.variance) for r in restored.results]),
            atol=1e-12,
        )
        with pytest.raises(TypeError):
            restored.axes["target_time"] = (10, 20)
        with pytest.raises(TypeError):
            restored.at_time(1).coordinates["target_time"] = 10


class AlternativeAlgorithm:
    """An extension whose parameter name is deliberately not lambda."""

    def __init__(self, axis="alpha", points=None):
        self.parameter_axes = {axis: (0, 0.001)}
        self.parameter_points = (
            points
            if points is not None
            else (
                {axis: 0},
                {axis: 0.001},
            )
        )
        self.axis = axis
        self.preparations = 0

    def prepare(self, basis, data, numerics):
        self.preparations += 1
        return hlt.Solver(0).prepare(basis, data, numerics)

    def prepare_target(self, problem, target, numerics):
        return hlt.Solver(0).prepare_target(problem, target, numerics)

    def solve_batch(self, problem, targets, parameters, numerics):
        return hlt.Solver(0).solve_batch(
            problem,
            targets,
            {"lambda": parameters[self.axis]},
            numerics,
        )


def extension_request(algorithm):
    result = sample().at_time(1)
    targets = sp.targets.grid(
        sp.targets.smooth_high_energy,
        vary={"width": (".05", ".1")},
        kernel=result.basis.at_time(1),
        cutoff=".4",
    )
    return sp.Reconstructor(result.basis, algorithm).reconstruct(
        result.data, target=targets
    )


@pytest.mark.parametrize("axis", ["width", "ensemble", "target_time", "lambda_"])
def test_axis_collisions_fail_before_algorithm_preparation(axis):
    algorithm = AlternativeAlgorithm(axis)
    with pytest.raises(ValueError, match="overlap|reserved"):
        extension_request(algorithm)
    assert algorithm.preparations == 0


@pytest.mark.parametrize(
    "points",
    [
        ({"alpha": 0},),
        ({"alpha": 0}, {"alpha": 0}),
        ({"beta": 0}, {"beta": 0.001}),
        ({"alpha": 0, "width": 1}, {"alpha": 0.001, "width": 1}),
        ({"alpha": 0, "extra": 1}, {"alpha": 0.001, "extra": 2}),
    ],
)
def test_malformed_parameter_grids_fail_before_preparation(points):
    algorithm = AlternativeAlgorithm(points=points)
    with pytest.raises(ValueError, match="parameter"):
        extension_request(algorithm)
    assert algorithm.preparations == 0


def test_non_hlt_parameter_grid_retains_every_combination():
    results = extension_request(AlternativeAlgorithm())
    assert results.shape == (2, 2)
    for width in (".05", ".1"):
        for alpha in (0, 0.001):
            assert results.select(width=width, alpha=alpha).coordinates == {
                "width": width,
                "alpha": alpha,
            }
