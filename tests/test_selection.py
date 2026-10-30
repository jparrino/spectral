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

from dataclasses import replace
import numpy as np
import pytest
import spectral as sp
from spectral.algorithms import hlt


def selection_inputs():
    basis = sp.Basis(sp.kernels.exponential(), (1, 2))
    data = sp.CorrelatorData.from_arrays(
        times=basis.times, mean=[1, 0.5], covariance=np.eye(2) * 0.01
    )
    results = sp.Reconstructor(basis, hlt.Solver(["1e-6", "1e-3", ".1"])).reconstruct(
        data, target=basis.at_time(3), metadata={"ensemble": "A"}
    )
    stats = np.asarray(results.statistical_errors, float)
    report = sp.systematics.SystematicReport(
        tuple(r.id for r in results.results),
        results.axes,
        ("nominal",),
        np.array([stats * [3, 0.1, 0.2]]),
        tuple(r.target_kernel for r in results.results),
        sp.numerics.numpy(),
        {},
    )
    return results, report


def test_selection_applies_dominance_and_preserves_result_objects():
    results, report = selection_inputs()
    selected = sp.scan.select(results, systematics=report)
    expected = min(
        results.results[1:],
        key=lambda r: (
            float(r.variance) + float(report.biases[0, results.results.index(r)]) ** 2
        ),
    )
    assert selected.results[0] is expected
    assert selected.groups == ({"ensemble": "A"},)
    restricted = sp.scan.select(
        results,
        systematics=report,
        policy=sp.scan.StatDominantMinimum(ranges={"lambda_": ("1e-3", "1e-3")}),
    )
    assert restricted.results[0] is results.results[1]


def test_selection_empty_groups_and_wrong_report_are_explicit():
    results, report = selection_inputs()
    with pytest.raises(ValueError, match="no statistically dominated candidate"):
        sp.scan.select(
            results,
            systematics=report,
            policy=sp.scan.StatDominantMinimum(ranges={"lambda": (".5", None)}),
        )
    selected = sp.scan.select(
        results,
        systematics=report,
        policy=sp.scan.StatDominantMinimum(
            ranges={"lambda": (".5", None)}, missing="skip"
        ),
    )
    assert selected.results == ()
    assert selected.skipped_groups == ({"ensemble": "A"},)
    wrong = replace(report, result_ids=("a", "b", "c"))
    with pytest.raises(ValueError, match="do not belong"):
        sp.scan.select(results, systematics=wrong)


def test_dataframe_and_library_share_signed_bias_tie_policy():
    pd = pytest.importorskip("pandas")
    from spectral.scan.selection import select_dataframe_group

    results, _ = selection_inputs()
    data = sp.CorrelatorData.from_arrays(
        times=(1, 2), mean=(1, 0), covariance=np.eye(2)
    )
    items = tuple(
        replace(r, value=1, variance=1, coefficients=[1, 0], data=data)
        for r in results.results[:2]
    )
    report = sp.systematics.SystematicReport(
        tuple(r.id for r in items),
        {"lambda": (0, 1)},
        ("nominal",),
        np.array([[-0.2, 0.1]]),
        tuple(r.target_kernel for r in items),
        sp.numerics.numpy(),
        {},
    )
    selected = sp.scan.select(items, systematics=report)
    frame = pd.DataFrame(
        [
            dict(
                estimate_scaled=1,
                error_scaled=1,
                syst_total_noise_final_scaled=bias,
                total_error_noise_scaled=np.hypot(1, bias),
                lambda_hlt=float(r.coordinates["lambda"]),
                delta_factor=1,
                id=r.id,
            )
            for r, bias in zip(items, [-0.2, 0.1])
        ]
    )
    row = select_dataframe_group(
        frame,
        factor=0.5,
        limits={
            k: None for k in ("lambda_min", "lambda_max", "delta_min", "delta_max")
        },
    )
    assert row.id == selected.results[0].id
