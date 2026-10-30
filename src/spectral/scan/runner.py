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

from dataclasses import dataclass, field, replace
from collections.abc import Mapping
from pathlib import Path
import hashlib
import json

from .._registry import serializable
from ..core.reconstruction import Reconstructor, reconstruct_requests, _error_result
from ..core.result import ResultCollection, Failure
from ..core.data import same_source_values
from ..numerics import Numerics
from ..core._state import FrozenMapping

# Bump when numerical behavior changes, independently of the archive format.
# v4 includes the negative-power integral fix and backend lambda validation.
_CHECKPOINT_REVISION = "numerical-validation-v4"


@serializable
@dataclass(frozen=True)
class Ensemble:
    data: object
    basis: object
    metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", FrozenMapping(self.metadata))


def reconstruct(
    *,
    ensembles,
    algorithm,
    target,
    target_times=None,
    numerics=None,
    execution=None,
    on_error="raise",
    checkpoint_dir=None,
    resume=False,
):
    """Reconstruct ensemble/target/algorithm grids using one execution context.

    Checkpoints are versioned archives per completed ensemble. Resume validates
    the numerical inputs, settings and source-sharing graph before reusing a result.
    Cached results are rebound to current sources. Cached failures are reused,
    with the current on_error policy (raise or record); they are not retried.
    Checkpoints from earlier numerical-validation revisions must be regenerated
    by running with resume=False.
    """
    from ..io.archive import Encoder, save, load

    if not ensembles:
        raise ValueError("ensembles cannot be empty")
    if on_error not in {"raise", "record"}:
        raise ValueError("on_error must be raise or record")
    # Identity is a relation, not a function of numerical contents. Validate each
    # identity, then hash its ensemble membership independently of transient UUIDs.
    source_groups, sources = {}, {}
    for name, ensemble in ensembles.items():
        source = ensemble.data.source
        previous = sources.get(source.id)
        if previous is not None and not same_source_values(source, previous):
            raise ValueError("inconsistent data for a shared source identity")
        sources[source.id] = source
        source_groups.setdefault(source.id, []).append(name)
    source_graph = tuple(
        sorted(
            (tuple(sorted(names, key=str)) for names in source_groups.values()), key=str
        )
    )
    policy = numerics or Numerics()
    requests, names, completed, paths, keys = [], [], {}, {}, {}
    for name, ensemble in ensembles.items():
        selected_target = target[name] if isinstance(target, Mapping) else target
        selected_times = (
            target_times[name] if isinstance(target_times, Mapping) else target_times
        )
        selected_algorithm = (
            algorithm[name] if isinstance(algorithm, Mapping) else algorithm
        )
        request = dict(
            reconstructor=Reconstructor(ensemble.basis, selected_algorithm, policy),
            data=ensemble.data,
            target=selected_target,
            target_times=selected_times,
            metadata={**ensemble.metadata, "ensemble": name},
        )
        if checkpoint_dir is not None:
            # Array-only inputs acquire fresh UUIDs in a new interpreter. Their
            # numerical content and source mapping, not that transient identity,
            # determine whether a checkpoint can be reused.
            stable_data = replace(
                ensemble.data, source=replace(ensemble.data.source, id="")
            )
            settings = (
                replace(ensemble, data=stable_data),
                selected_algorithm,
                selected_target,
                selected_times,
                policy,
                source_graph,
                _CHECKPOINT_REVISION,
            )
            key = hashlib.sha256(
                json.dumps(Encoder().encode(settings), sort_keys=True).encode()
            ).hexdigest()
            path = Path(checkpoint_dir) / (
                hashlib.sha256(str(name).encode()).hexdigest()[:16] + ".spectral"
            )
            paths[name], keys[name] = path, key
            if resume and path.exists():
                previous = load(path)
                if previous.metadata.get("request_hash") != key:
                    raise ValueError(
                        f"checkpoint settings differ for ensemble {name}; "
                        "rerun with resume=False to regenerate checkpoints"
                    )

                def rebind(result):
                    if isinstance(result, Failure):
                        return _error_result(result.coordinates, result.error, on_error)
                    return result.rebind_data(
                        ensemble.data.select_times(result.basis.times)
                    )

                cached = previous.results
                completed[name] = (
                    replace(cached, results=tuple(rebind(r) for r in cached.results))
                    if isinstance(cached, ResultCollection)
                    else rebind(cached)
                )
                continue
        requests.append(request)
        names.append(name)
    for name, result in zip(
        names, reconstruct_requests(requests, execution=execution, on_error=on_error)
    ):
        completed[name] = result
        if checkpoint_dir is not None:
            save(
                paths[name],
                results=result,
                metadata={"request_hash": keys[name]},
                overwrite=True,
            )
    collections = [completed[name] for name in ensembles]
    first_axes = (
        collections[0].axes if isinstance(collections[0], ResultCollection) else {}
    )
    if any(
        (c.axes if isinstance(c, ResultCollection) else {}) != first_axes
        for c in collections
    ):
        raise ValueError(
            "ensemble result axes differ; use common coordinate labels for ensemble-dependent targets"
        )
    results = []
    for name, collection in zip(ensembles, collections):
        items = (
            collection.results
            if isinstance(collection, ResultCollection)
            else (collection,)
        )
        for r in items:
            results.append(replace(r, coordinates={"ensemble": name, **r.coordinates}))
    return ResultCollection(
        {"ensemble": tuple(ensembles), **first_axes}, tuple(results)
    )
