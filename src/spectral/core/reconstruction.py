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

"""Algorithm-independent orchestration of linear reconstruction tasks."""

from dataclasses import dataclass
from itertools import product
import math
import numpy as np

from ..execution import Serial, Outcome
from ..numerics import Numerics
from .grid import TargetGrid, validate_axis, coordinate_key
from .result import Result, ResultCollection, Failure
from ._state import FrozenMapping
from .kernel import as_kernel


class ReconstructionError(RuntimeError):
    pass


def is_time_dependent(target):
    return getattr(target, "is_time_dependent", hasattr(target, "at_time"))


@dataclass(frozen=True)
class Reconstructor:
    basis: object
    algorithm: object
    numerics: Numerics = Numerics()

    def reconstruct(
        self,
        data,
        *,
        target,
        target_times=None,
        execution=None,
        on_error="raise",
        metadata=None,
    ):
        request = dict(
            reconstructor=self,
            data=data,
            target=target,
            target_times=target_times,
            metadata=metadata or {},
        )
        return reconstruct_requests([request], execution=execution, on_error=on_error)[
            0
        ]


def _make_plan(request):
    recon = request["reconstructor"]
    data = request["data"].select_times(recon.basis.times)
    if (
        data.energy_unit
        and recon.basis.energy_unit
        and data.energy_unit != recon.basis.energy_unit
    ):
        raise ValueError("data and basis energy units disagree")
    target = request["target"]
    grid = target if isinstance(target, TargetGrid) else TargetGrid({}, (target,))
    times = request.get("target_times")
    if times is not None:
        times = validate_axis("target_time", times)
    targets, coordinates = [], []
    for item, coordinate in zip(grid.targets, grid.coordinates):
        item = as_kernel(item)
        if is_time_dependent(item) != (times is not None):
            raise ValueError(
                "time-dependent targets require target_times; fixed targets reject target_times"
            )
        for t in times if times is not None else (None,):
            targets.append(item.at_time(t) if times is not None else item)
            coordinates.append(
                {**coordinate, **({"target_time": t} if times is not None else {})}
            )
    algorithm_axes = {
        key: validate_axis(key, values)
        for key, values in recon.algorithm.parameter_axes.items()
    }
    overlap = set(grid.axes) & set(algorithm_axes)
    if overlap:
        raise ValueError(f"target and algorithm axes overlap: {sorted(overlap)}")
    if set(algorithm_axes) & {"target_time", "ensemble", "lambda_"}:
        raise ValueError("algorithm axes use a reserved coordinate name")
    parameters = tuple(FrozenMapping(p) for p in recon.algorithm.parameter_points)
    expected = {
        tuple(coordinate_key(v) for v in point)
        for point in product(*algorithm_axes.values())
    }
    actual = []
    for point in parameters:
        if set(point) & (set(grid.axes) | {"target_time", "ensemble", "lambda_"}):
            raise ValueError(
                "algorithm parameters overlap target or reserved coordinates"
            )
        if not set(algorithm_axes) <= set(point):
            raise ValueError("algorithm parameter point is missing a declared axis")
        actual.append(tuple(coordinate_key(point[k]) for k in algorithm_axes))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(
            "algorithm parameter points must cover their axes exactly once"
        )
    scalar_parameters = set(parameters[0]) - set(algorithm_axes)
    if any(
        set(point) != set(parameters[0])
        or any(
            coordinate_key(point[k]) != coordinate_key(parameters[0][k])
            for k in scalar_parameters
        )
        for point in parameters
    ):
        raise ValueError("varying algorithm parameters must declare an axis")
    axes = {**grid.axes, **algorithm_axes}
    if times is not None:
        axes["target_time"] = times
    return dict(
        request=request,
        recon=recon,
        data=data,
        targets=targets,
        coordinates=coordinates,
        axes=axes,
        parameters=parameters,
        outputs={},
    )


def _coordinate_key(coordinates):
    return tuple((key, coordinate_key(value)) for key, value in coordinates.items())


def reconstruct_requests(requests, *, execution=None, on_error="raise"):
    """Schedule preparation and solve phases across all requested ensembles."""
    if on_error not in {"raise", "record"}:
        raise ValueError("on_error must be raise or record")
    executor = execution or Serial()
    plans = [_make_plan(request) for request in requests]
    from .data import CorrelatorData

    tasks, prepare_indices = [], {}
    for index, p in enumerate(plans):
        recon = p["recon"]
        key = (
            recon.algorithm.preparation_key(recon.basis, p["data"], recon.numerics)
            if hasattr(recon.algorithm, "preparation_key")
            else index
        )
        if key not in prepare_indices:
            prepare_indices[key] = len(tasks)
            # Workers receive numerical inputs, not entire JKS source databases.
            numerical_data = CorrelatorData.from_arrays(
                times=p["data"].times,
                mean=p["data"].mean,
                covariance=p["data"].covariance,
                energy_unit=p["data"].energy_unit,
            )
            tasks.append(
                (recon.algorithm.prepare, (recon.basis, numerical_data, recon.numerics))
            )
        p["prepare_index"] = prepare_indices[key]
    prepared = executor.map(tasks)
    target_jobs, target_locations = [], []
    for plan in plans:
        outcome = prepared[plan["prepare_index"]]
        plan["problem"] = outcome
        plan["prepared_targets"] = {}
        for i, target in enumerate(plan["targets"]):
            if outcome.error:
                _failure(plan, i, outcome.error, on_error)
            else:
                target_jobs.append(
                    (
                        plan["recon"].algorithm.prepare_target,
                        (outcome.value, target, plan["recon"].numerics),
                    )
                )
                target_locations.append((plan, i))
    for (plan, i), outcome in zip(target_locations, executor.map(target_jobs)):
        if outcome.error:
            _failure(plan, i, outcome.error, on_error)
        else:
            plan["prepared_targets"][i] = outcome.value
    solve_jobs, solve_locations = [], []
    total_parameters = sum(len(p["parameters"]) for p in plans)
    for plan in plans:
        indices = tuple(plan["prepared_targets"])
        if not indices:
            continue
        # Enough chunks to use spare workers, with target RHS batching otherwise.
        chunks = min(
            len(indices), max(1, math.ceil(executor.workers / max(1, total_parameters)))
        )
        batches = [
            tuple(int(i) for i in chunk) for chunk in np.array_split(indices, chunks)
        ]
        for parameters in plan["parameters"]:
            for batch in batches:
                solve_jobs.append(
                    (
                        plan["recon"].algorithm.solve_batch,
                        (
                            plan["problem"].value,
                            tuple(plan["prepared_targets"][i] for i in batch),
                            parameters,
                            plan["recon"].numerics,
                        ),
                    )
                )
                solve_locations.append((plan, batch, parameters))
    for (plan, indices, parameters), outcome in zip(
        solve_locations, executor.map(solve_jobs)
    ):
        for j, index in enumerate(indices):
            coordinates = _coordinates(plan, index, parameters)
            if outcome.error:
                result = _error_result(coordinates, outcome.error, on_error)
            else:
                try:
                    point = outcome.value[j]
                    if isinstance(point, Outcome):
                        if point.error:
                            raise ArithmeticError(point.error)
                        point = point.value
                    q, diagnostics = point
                    result = _build_result(plan, index, q, diagnostics, coordinates)
                except Exception as exc:
                    result = _error_result(
                        coordinates, f"{type(exc).__name__}: {exc}", on_error
                    )
            plan["outputs"][_coordinate_key(coordinates)] = result
    collections = []
    for plan in plans:
        coordinates = [
            dict(zip(plan["axes"], point)) for point in product(*plan["axes"].values())
        ]
        results = tuple(plan["outputs"][_coordinate_key(c)] for c in coordinates)
        collections.append(
            ResultCollection(plan["axes"], results) if plan["axes"] else results[0]
        )
    return collections


def _coordinates(plan, index, parameters):
    all_coords = {**plan["coordinates"][index], **parameters}
    return {key: all_coords[key] for key in plan["axes"]}


def _error_result(coordinates, error, on_error):
    if on_error == "raise":
        raise ReconstructionError(f"reconstruction failed at {coordinates}: {error}")
    return Failure(coordinates, error)


def _failure(plan, index, error, on_error):
    for parameters in plan["parameters"]:
        coordinates = _coordinates(plan, index, parameters)
        plan["outputs"][_coordinate_key(coordinates)] = _error_result(
            coordinates, error, on_error
        )


def _build_result(plan, index, coefficients, diagnostics, coordinates):
    recon, data = plan["recon"], plan["data"]
    with recon.numerics.context() as ctx:
        q = ctx.array(coefficients)
        mean, covariance = ctx.array(data.mean), ctx.array(data.covariance)
        value = ctx.dot(q, mean)
        variance = ctx.dot(q, covariance @ q)
        ctx.require_finite((value, variance), "reconstructed value/variance")
        scale = ctx.sum(
            abs(q[i] * covariance[i, j] * q[j])
            for i in range(len(q))
            for j in range(len(q))
        )
        ctx.require_finite(scale, "variance contraction scale")
        if variance < -1000 * len(q) * ctx.eps * max(scale, ctx.number("1e-300")):
            raise ArithmeticError("negative reconstructed variance")
        variance = max(ctx.number(0), variance)
        return Result(
            value,
            variance,
            q,
            recon.basis,
            plan["targets"][index],
            data,
            recon.numerics,
            {
                **diagnostics,
                "algorithm": type(recon.algorithm).__name__,
                "regulator": getattr(recon.algorithm, "regulator", None),
            },
            coordinates,
            plan["request"].get("metadata", {}),
            algorithm=recon.algorithm,
        )
