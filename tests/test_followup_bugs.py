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

"""Regressions discovered in the follow-up review after F01–F14."""

from dataclasses import dataclass, replace

import mpmath as mp
import numpy as np
import pytest
import spectral as sp
from spectral.algorithms import hlt
from spectral.core.grid import coordinate_equal


def inputs():
    basis = sp.Basis(sp.kernels.exponential(), (1, 2))
    data = sp.CorrelatorData.from_arrays(
        times=basis.times, mean=[1, 0.5], covariance=np.eye(2)
    )
    return basis, data


def test_distinct_mpf_scan_coordinates_survive_ambient_precision():
    with mp.workdps(100):
        first = mp.mpf(".1")
        second = first + mp.mpf("1e-70")
    basis, data = inputs()
    with mp.workdps(15):
        assert not coordinate_equal(first, second)
        recon = sp.Reconstructor(
            basis, hlt.Solver((first, second)), sp.numerics.mpmath(95)
        )
        results = recon.reconstruct(data, target=basis.at_time(3))
        assert results.at_lambda(first) is results.results[0]
        assert results.at_lambda(second) is results.results[1]
        assert results.results[0].value != results.results[1].value
        for coordinate, actual in zip((first, second), results.results):
            expected = replace(recon, algorithm=hlt.Solver(coordinate)).reconstruct(
                data, target=basis.at_time(3)
            )
            assert actual.value == expected.value
    assert coordinate_equal(".1", 0.1)
    assert coordinate_equal("1", mp.mpf(1))


@pytest.mark.parametrize("coordinate", [np.nan, np.inf, "-Infinity", mp.nan, mp.inf])
def test_nonfinite_coordinates_rejected(coordinate):
    with pytest.raises(ValueError, match="finite"):
        sp.TargetGrid(
            {"cutoff": (coordinate,)}, (sp.targets.gaussian(center=1, width=1),)
        )


def test_mpf_input_time_selection_retains_distinct_labels():
    with mp.workdps(100):
        times = (mp.mpf(1), mp.mpf(1) + mp.mpf("1e-70"))
    with mp.workdps(15):
        basis = sp.Basis(sp.kernels.exponential(), times)
        data = sp.CorrelatorData.from_arrays(times=times, mean=[1, 2])
        assert data.select_times((basis.times[1],)).mean[0] == 2


@dataclass(frozen=True)
class SameReprKernel(sp.Kernel):
    rate: int

    def __repr__(self):
        return "SameReprKernel()"

    def scalar(self, w, ctx):
        return ctx.exp(-ctx.number(self.rate) * w)


def test_systematic_cache_does_not_use_custom_kernel_repr():
    basis, data = inputs()
    result = sp.Reconstructor(basis, hlt.Solver(0)).reconstruct(
        data, target=basis.at_time(1)
    )
    items = sp.ResultCollection(
        {"kernel": ("one", "two")},
        tuple(
            replace(result, id=str(i), coordinates={"kernel": name})
            for i, name in enumerate(("one", "two"))
        ),
    )
    references = sp.targets.named({"one": SameReprKernel(1), "two": SameReprKernel(2)})
    report = sp.systematics.estimate(
        items,
        reference_kernel=references,
        density=sp.SpectralDensity.from_function(lambda w, ctx: 1, domain=(0, 1)),
    )
    expected = (1 - np.exp(-1)) - (1 - np.exp(-2)) / 2
    np.testing.assert_allclose(
        report.biases.astype(float), [[0, expected]], rtol=1e-13, atol=1e-15
    )


@pytest.mark.parametrize("grid", [False, True])
def test_systematics_preserves_high_precision_kernel_parameters(grid):
    with mp.workdps(100):
        first = mp.mpf(1)
        second = first + mp.mpf("1e-60")
    basis, data = inputs()
    result = sp.Reconstructor(basis, hlt.Solver(0), sp.numerics.mpmath(85)).reconstruct(
        data, target=basis.at_time(1)
    )
    references = sp.targets.named(
        {"one": basis.at_time(first), "two": basis.at_time(second)}
    )
    results = sp.ResultCollection(
        {"kernel": ("one", "two")},
        tuple(
            replace(result, id=str(i), coordinates={"kernel": name})
            for i, name in enumerate(("one", "two"))
        ),
    )
    options = {"energies": [0, ".5", 1]} if grid else {}
    with mp.workdps(15):
        report = sp.systematics.estimate(
            results,
            reference_kernel=references,
            density=sp.SpectralDensity.from_function(lambda w, ctx: 1, domain=(0, 1)),
            **options,
        )
    with mp.workdps(85):
        if grid:
            expected = mp.fsum(
                weight * (mp.exp(-first * w) - mp.exp(-second * w))
                for w, weight in [
                    (0, mp.mpf(".25")),
                    (mp.mpf(".5"), mp.mpf(".5")),
                    (1, mp.mpf(".25")),
                ]
            )
        else:
            expected = -mp.expm1(-first) / first + mp.expm1(-second) / second
        assert abs(report.biases[0, 1] - expected) < mp.mpf("1e-82")
        assert report.biases[0, 1] > 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("mean", [2, 0.5]),
        ("covariance", np.eye(2) * 100),
        ("projection", np.eye(2) * 2),
    ],
)
def test_inconsistent_retained_source_projection_is_rejected(field, value):
    _, data = inputs()
    with pytest.raises(ValueError, match="source projection"):
        replace(data, **{field: value})


def test_general_projection_and_scaling_preserve_full_covariance():
    basis, data = inputs()
    projection = np.array([[2.0, -1.0], [1.0, 3.0]])
    projected = replace(
        data,
        mean=projection @ np.asarray(data.mean, float),
        covariance=projection @ np.asarray(data.covariance, float) @ projection.T,
        projection=projection,
    ).scaled("1e-30")
    results = sp.Reconstructor(basis, hlt.Solver(0)).reconstruct(
        projected, target=sp.targets.correlator(basis=basis), target_times=(1, 2)
    )
    np.testing.assert_allclose(
        np.asarray(results.statistical_covariance, float),
        np.asarray(projected.covariance, float),
        rtol=1e-14,
        atol=0,
    )
    np.testing.assert_allclose(
        np.asarray([r.variance for r in results.results], float),
        np.diag(np.asarray(projected.covariance, float)),
        rtol=1e-14,
        atol=0,
    )


def test_manually_combined_results_reject_conflicting_source_identity():
    basis, data = inputs()
    other = replace(
        data,
        covariance=np.eye(2) * 100,
        source=replace(data.source, covariance=np.eye(2) * 100),
    )
    recon = sp.Reconstructor(basis, hlt.Solver(0))
    results = sp.ResultCollection(
        {"case": ("A", "B")},
        tuple(
            replace(
                recon.reconstruct(value, target=basis.at_time(1)),
                coordinates={"case": label},
            )
            for label, value in zip(("A", "B"), (data, other))
        ),
    )
    with pytest.raises(ValueError, match="inconsistent data"):
        _ = results.statistical_covariance


@pytest.mark.parametrize(
    "power,rate,lower", [(0, 1e-200, 1e203), (200, 100.0, 0), (2, 1e200, 1e-200)]
)
def test_exponential_integral_avoids_intermediate_underflow_and_overflow(
    power, rate, lower
):
    with mp.workdps(90):
        expected = float(
            mp.gammainc(mp.mpf(power) + 1, mp.mpf(rate) * mp.mpf(lower), mp.inf)
            / mp.mpf(rate) ** (power + 1)
        )
    with sp.numerics.numpy().context() as ctx:
        actual = ctx.exp_integral(power, rate, lower)
    assert actual == pytest.approx(expected, rel=1e-13, abs=0)
    if expected != 0:
        assert actual > 0


def test_exponential_integral_reports_unrepresentable_output():
    with sp.numerics.numpy().context() as ctx:
        with pytest.raises(ArithmeticError, match="non-finite"):
            ctx.exp_integral(200, 0.001, 0)


def test_implicit_source_rejects_inconsistent_explicit_projection():
    with pytest.raises(ValueError, match="source projection"):
        sp.CorrelatorData((1,), [1], [[1]], projection=[[2]])


def test_cross_covariance_uses_joint_precision_for_both_projections():
    from spectral.core.data import SourceData

    source = SourceData((1,), [1], [[1]])
    projection = np.array([[10**16], [1], [-(10**16)]], dtype=object)
    data = sp.CorrelatorData(
        (1, 2, 3),
        projection[:, 0],
        projection @ projection.T,
        source=source,
        projection=projection,
    )
    basis = sp.Basis(sp.kernels.exponential(), data.times)
    low = sp.Result(
        1,
        1,
        np.ones(3),
        basis,
        basis.at_time(1),
        data,
        sp.numerics.numpy(),
        coordinates={"precision": "low"},
    )
    high = replace(
        low,
        numerics=sp.numerics.mpmath(80),
        target_kernel=basis.at_time(1),
        coordinates={"precision": "high"},
    )
    covariance = sp.ResultCollection(
        {"precision": ("low", "high")}, (low, high)
    ).statistical_covariance
    # Reusing low.source_weights would cancel 1e16 + 1 - 1e16 to zero in binary64.
    np.testing.assert_array_equal(covariance, np.ones((2, 2)))
