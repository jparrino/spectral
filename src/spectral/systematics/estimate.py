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

from dataclasses import dataclass
from collections.abc import Mapping
import numpy as np

from .._registry import serializable
from ..core.result import Result, ResultCollection
from ..core.grid import coordinate_equal, coordinate_key, TargetGrid
from ..core.kernel import kernel_cache_key, as_kernel
from ..core.spectral_density import SpectralIntegrand
from ..core.reconstruction import is_time_dependent
from ..core._state import FrozenMapping
from ..core.grid import validate_axis
from ..execution import Serial
from .deformations import DeformationSet, trapezoid_weights
from .density import for_basis, BasisDensity, _physical_kernel


@serializable
@dataclass(frozen=True)
class SystematicReport:
    result_ids: tuple
    axes: Mapping
    model_ids: tuple
    biases: np.ndarray  # (n_models, n_results), always includes nominal first
    reference_kernels: tuple
    numerics: object
    metadata: Mapping
    models: tuple = ()

    def __post_init__(self):
        from ..core.data import readonly

        axes = {k: validate_axis(k, v) for k, v in self.axes.items()}
        if int(np.prod([len(v) for v in axes.values()])) != len(self.result_ids):
            raise ValueError("systematic result count does not match report axes")
        if len(set(self.result_ids)) != len(self.result_ids):
            raise ValueError("systematic result IDs must be unique")
        if len(set(self.model_ids)) != len(self.model_ids):
            raise ValueError("systematic model IDs must be unique")
        biases = readonly(self.biases)
        if (
            biases.shape != (len(self.model_ids), len(self.result_ids))
            or not self.model_ids
            or self.model_ids[0] != "nominal"
        ):
            raise ValueError(
                "systematic report must contain aligned biases with nominal first"
            )
        with self.numerics.context() as ctx:
            ctx.array(biases)  # Reject non-finite biases after backend conversion.
        object.__setattr__(self, "biases", biases)
        object.__setattr__(self, "axes", FrozenMapping(axes))
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata))
        for name in ("result_ids", "model_ids", "reference_kernels", "models"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def shape(self):
        return tuple(len(v) for v in self.axes.values())

    def _shaped(self, values):
        a = np.array(values, dtype=object).reshape(self.shape)
        return a.item() if not self.axes else a

    @property
    def nominal_bias(self):
        return self._shaped(self.biases[0])

    @property
    def maximum_indices(self):
        with self.numerics.context() as ctx:
            return tuple(
                max(
                    range(len(self.model_ids)),
                    key=lambda i: abs(ctx.number(self.biases[i, j])),
                )
                for j in range(len(self.result_ids))
            )

    @property
    def bias_at_maximum(self):
        return self._shaped(
            [self.biases[i, j] for j, i in enumerate(self.maximum_indices)]
        )

    @property
    def max_abs_bias(self):
        with self.numerics.context() as ctx:
            return self._shaped(
                [
                    abs(ctx.number(self.biases[i, j]))
                    for j, i in enumerate(self.maximum_indices)
                ]
            )

    @property
    def model_at_maximum(self):
        return self._shaped([self.model_ids[i] for i in self.maximum_indices])

    @property
    def biases_by_model(self):
        return {
            name: self._shaped(row) for name, row in zip(self.model_ids, self.biases)
        }

    @property
    def reference_kernel(self):
        if len(self.reference_kernels) != 1:
            raise ValueError("collection report has multiple reference_kernels")
        return self.reference_kernels[0]

    def combined_biases(self, weights):
        """Signed bias of a linear combination, before taking its envelope."""
        with self.numerics.context() as ctx:
            weights = ctx.array(weights).reshape(-1)
            if len(weights) != len(self.result_ids):
                raise ValueError("weights must match flattened result order")
            return ctx.array([ctx.dot(weights, ctx.array(row)) for row in self.biases])


def _integral(model, kernel, lower, upper, policy):
    integrand = (
        model.integrand(kernel)
        if isinstance(model, BasisDensity)
        else SpectralIntegrand(model, kernel)
    )
    return integrand.integrate(numerics=policy, lower=lower, upper=upper)


def _reference(reference, result):
    if isinstance(reference, TargetGrid):
        matches = [
            k
            for k, c in zip(reference.targets, reference.coordinates)
            if all(
                key in result.coordinates
                and coordinate_equal(result.coordinates[key], value)
                for key, value in c.items()
            )
        ]
        if len(matches) != 1:
            raise ValueError(
                "reference grid must align uniquely with result coordinates"
            )
        reference = matches[0]
    reference = as_kernel(reference)
    if is_time_dependent(reference):
        if "target_time" not in result.coordinates:
            raise ValueError(
                "a time-dependent reference requires recorded target times"
            )
        reference = reference.at_time(result.coordinates["target_time"])
    return reference


def estimate(
    results,
    *,
    reference_kernel,
    density,
    deformations=None,
    execution=None,
    upper=None,
    energies=None,
):
    """Evaluate signed reconstruction-minus-reference biases; never change results.

    By default use adaptive quadrature. Supplying energies explicitly requests
    batched trapezoid integration over that finite grid; its bounds are recorded.
    Physical densities are converted using each result's basis power.
    """
    items = results.results if isinstance(results, ResultCollection) else (results,)
    if any(not isinstance(r, Result) for r in items):
        raise ValueError(
            "select successful reconstructions before estimating systematics"
        )
    axes = results.axes if isinstance(results, ResultCollection) else {}
    if deformations is None:
        deformations = DeformationSet((), (), {})
    if not isinstance(deformations, DeformationSet):
        raise TypeError(
            "deformations must be a DeformationSet with stable model identities"
        )
    models = (density,) + deformations.models
    model_ids = ("nominal",) + deformations.ids
    references = tuple(_reference(reference_kernel, r) for r in items)
    policy = max(
        (r.numerics for r in items), key=lambda p: (p.backend == "mpmath", p.dps)
    )
    densities = {}
    models_by_result = []
    for m in models:
        row = []
        for r in items:
            key = (id(m), id(r.basis))
            if key not in densities:
                densities[key] = for_basis(m, r.basis)
            row.append(densities[key])
        models_by_result.append(tuple(row))
    for j, (r, ref) in enumerate(zip(items, references)):
        for obj in (*(row[j] for row in models_by_result), ref):
            p = getattr(obj, "spectral_power", None)
            if p is not None and not coordinate_equal(p, r.basis.p):
                raise ValueError(
                    "spectral powers of basis, model and reference must agree"
                )
            unit = getattr(obj, "energy_unit", None)
            if unit and r.basis.energy_unit and unit != r.basis.energy_unit:
                raise ValueError("model/reference energy unit disagrees with basis")
    if energies is not None:
        if upper is not None:
            raise ValueError(
                "upper and energies cannot both specify integration bounds"
            )
        biases = _on_grid(items, references, models_by_result, energies, policy)
        integration = {
            "method": "trapezoid",
            "bounds": (energies[0], energies[-1]),
            "model_domains": tuple(m.domain for m in models),
            "support": "intersection of grid, model domain and basis domain; exact boundary nodes",
        }
    else:
        executor = execution or Serial()
        tasks, task_indices = [], {}

        def request(m, k, lo):
            key = (id(m), kernel_cache_key(k), coordinate_key(lo))
            if key not in task_indices:
                task_indices[key] = len(tasks)
                tasks.append((_integral, (m, k, lo, upper, policy)))
            return task_indices[key]

        indices = []
        for resolved in models_by_result:
            row = []
            for m, r, ref in zip(resolved, items, references):
                moments = tuple(
                    request(m, r.basis.at_time(t), r.basis.omega_0)
                    for t in r.basis.times
                )
                row.append((moments, request(m, ref, r.basis.omega_0)))
            indices.append(row)
        outcomes = executor.map(tasks)
        if any(o.error for o in outcomes):
            i = next(i for i, o in enumerate(outcomes) if o.error)
            raise ArithmeticError(
                f"systematic integral {i} failed: {outcomes[i].error}"
            )
        with policy.context() as ctx:
            biases = ctx.array(
                [
                    [
                        ctx.dot(
                            ctx.array(r.coefficients),
                            ctx.array([outcomes[i].value for i in moments]),
                        )
                        - outcomes[reference].value
                        for r, (moments, reference) in zip(items, row)
                    ]
                    for row in indices
                ]
            )
        integration = {
            "method": "adaptive",
            "upper": upper,
            "model_domains": tuple(m.domain for m in models),
        }
    return SystematicReport(
        tuple(r.id for r in items),
        axes,
        model_ids,
        biases,
        references,
        policy,
        {
            "deformations": deformations.metadata,
            "integration": integration,
            "sign": "approximated minus reference",
            "nominal_included": True,
        },
        models,
    )


def _on_grid(items, references, models_by_result, energies, policy):
    with policy.context() as ctx:
        grid = ctx.array(energies)
        trapezoid_weights(grid, ctx)  # Validate the requested grid once.
        biases = ctx.array(np.zeros((len(models_by_result), len(items))))
        # Reuse kernel differences for models sharing a support interval and
        # basis evaluations for results sharing a basis on that interval.
        differences, basis_values = {}, {}
        for i, resolved in enumerate(models_by_result):
            spectra = {}
            for j, (model, result, reference) in enumerate(
                zip(resolved, items, references)
            ):
                lo_model = ctx.number(model.domain[0])
                hi_model = (
                    grid[-1] if model.domain[1] is None else ctx.number(model.domain[1])
                )
                lo = max(grid[0], lo_model, ctx.number(result.basis.omega_0))
                hi = min(grid[-1], hi_model)
                if hi <= lo:
                    continue
                support = (lo, hi)
                physical = model.density if isinstance(model, BasisDensity) else model
                power = model.spectral_power if isinstance(model, BasisDensity) else 0
                spectrum_key = (kernel_cache_key(physical), support)
                if spectrum_key not in spectra:
                    # Evaluate the actual integrand at inserted boundaries;
                    # masking nodes would give incorrect endpoint weights.
                    nodes = ctx.array([lo, *(w for w in grid if lo < w < hi), hi])
                    measure = trapezoid_weights(nodes, ctx) * ctx.array(
                        [physical.scalar(w, ctx) for w in nodes]
                    )
                    ctx.require_finite(measure, "grid integration measure")
                    spectra[spectrum_key] = nodes, measure
                nodes, measure = spectra[spectrum_key]
                key = (j, support, coordinate_key(power))
                if key not in differences:
                    basis_key = (
                        kernel_cache_key(result.basis),
                        support,
                        coordinate_key(power),
                    )
                    if basis_key not in basis_values:
                        kernels = (
                            tuple(
                                _physical_kernel(result.basis.at_time(t), power)
                                for t in result.basis.times
                            )
                            if isinstance(model, BasisDensity)
                            else tuple(
                                result.basis.at_time(t) for t in result.basis.times
                            )
                        )
                        basis_values[basis_key] = ctx.array(
                            [[k.scalar(w, ctx) for k in kernels] for w in nodes]
                        )
                    approximation = basis_values[basis_key] @ ctx.array(
                        result.coefficients
                    )
                    ref = (
                        _physical_kernel(reference, power)
                        if isinstance(model, BasisDensity)
                        else reference
                    )
                    differences[key] = approximation - ctx.array(
                        [ref.scalar(w, ctx) for w in nodes]
                    )
                    ctx.require_finite(differences[key], "grid kernel difference")
                biases[i, j] = ctx.dot(differences[key], measure)
        ctx.require_finite(biases, "grid systematic biases")
        return biases
