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

from dataclasses import dataclass, field
from collections.abc import Mapping
from collections import defaultdict
import math

from .._registry import serializable
from ..core.result import Result
from ..io.csv import result_items
from ..core._state import FrozenMapping, freeze


@serializable
@dataclass(frozen=True)
class StatDominantMinimum:
    factor: object = 0.5
    ranges: Mapping = field(default_factory=dict)
    window_ranges: Mapping = field(default_factory=dict)
    missing: str = "raise"

    def __post_init__(self):
        object.__setattr__(self, "ranges", FrozenMapping(self.ranges))
        object.__setattr__(self, "window_ranges", FrozenMapping(self.window_ranges))
        if (
            not math.isfinite(float(self.factor))
            or float(self.factor) < 0
            or self.missing not in {"raise", "skip"}
        ):
            raise ValueError(
                "factor must be finite and nonnegative; missing must be raise or skip"
            )


@serializable
@dataclass(frozen=True)
class Selection:
    results: tuple
    policy: StatDominantMinimum
    groups: tuple
    skipped_groups: tuple

    def __post_init__(self):
        object.__setattr__(self, "results", tuple(self.results))
        object.__setattr__(self, "groups", freeze(self.groups))
        object.__setattr__(self, "skipped_groups", freeze(self.skipped_groups))


def within_range(value, lo, hi):
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("selection range is reversed")
    return (lo is None or value >= lo) and (hi is None or value <= hi)


def candidate_order(
    *,
    value,
    stat,
    syst,
    total,
    lambda_value,
    width,
    index,
    factor,
    isfinite=math.isfinite,
):
    """Shared finite/dominance policy and stable tie order for all scan frontends."""
    if any(not isfinite(x) for x in (value, stat, syst, total, lambda_value, width)):
        return None
    if stat < 0 or total < 0 or stat < factor * abs(syst):
        return None
    return (total, stat, abs(syst), lambda_value, width, index)


def select_dataframe_group(group, *, factor, limits):
    """Optional pandas frontend for legacy scan columns; returns row + counts."""
    import numpy as np

    if not math.isfinite(factor) or factor < 0:
        raise ValueError("The dominance factor must be finite and nonnegative.")
    candidates = group
    for column, prefix in (("lambda_hlt", "lambda"), ("delta_factor", "delta")):
        lo, hi = limits[f"{prefix}_min"], limits[f"{prefix}_max"]
        within_range(0, lo, hi)  # Validate ranges even for an empty frame.
        if lo is not None:
            candidates = candidates.loc[candidates[column] >= lo]
        if hi is not None:
            candidates = candidates.loc[candidates[column] <= hi]
    n_window = len(candidates)
    columns = [
        "estimate_scaled",
        "error_scaled",
        "syst_total_noise_final_scaled",
        "total_error_noise_scaled",
        "lambda_hlt",
        "delta_factor",
    ]
    finite = np.isfinite(candidates[columns]).all(axis=1)
    finite &= (candidates.error_scaled >= 0) & (
        candidates.total_error_noise_scaled >= 0
    )
    candidates = candidates.loc[finite]
    n_finite = len(candidates)
    eligible = []
    for position, (_, row) in enumerate(candidates.iterrows()):
        index = row.get("scan_index", position)
        index = index if np.isfinite(index) else np.inf
        order = candidate_order(
            value=row.estimate_scaled,
            stat=row.error_scaled,
            syst=row.syst_total_noise_final_scaled,
            total=row.total_error_noise_scaled,
            lambda_value=row.lambda_hlt,
            width=row.delta_factor,
            index=index,
            factor=factor,
        )
        if order is not None:
            eligible.append((order, row))
    if not eligible:
        raise ValueError(
            "No finite, statistically dominated result inside the selection ranges."
        )
    selected = min(eligible, key=lambda entry: entry[0])[1].copy()
    systematic = abs(selected.syst_total_noise_final_scaled)
    selected["selection_statistical_dominance_factor"] = factor
    selected["selection_stat_over_syst"] = (
        selected.error_scaled / systematic if systematic else np.inf
    )
    selected["selection_n_window_candidates"] = n_window
    selected["selection_n_finite_candidates"] = n_finite
    selected["selection_n_dominant_candidates"] = len(eligible)
    selected["selection_metric"] = "total_error_noise_scaled"
    selected["selection_rank"] = 1
    for key, value in limits.items():
        selected[f"selection_{key}"] = np.nan if value is None else value
    return selected


def select(results, *, systematics, policy=None, group_by=None):
    policy = policy or StatDominantMinimum()
    items = tuple(r for r in result_items(results) if isinstance(r, Result))
    report_indices = {key: i for i, key in enumerate(systematics.result_ids)}
    if any(r.id not in report_indices for r in items):
        raise ValueError("systematics do not belong to the supplied results")
    maximum_indices = systematics.maximum_indices
    if group_by is None:
        group_by = tuple(
            dict.fromkeys(
                k
                for r in items
                for k in {
                    **{
                        name: r.metadata[name]
                        for name in ("ensemble", "discretization", "kernel")
                        if name in r.metadata
                    },
                    **r.coordinates,
                }
                if k not in {"lambda", "width", "delta_factor"}
            )
        )
    groups = defaultdict(list)
    for r in items:
        labels = {**r.metadata, **r.coordinates}
        groups[tuple(labels.get(k) for k in group_by)].append(r)
    chosen, labels, skipped = [], [], []
    for key, group in groups.items():
        candidates = []
        for r in group:
            coords = {**r.metadata, **r.coordinates}
            ranges = {
                **policy.ranges,
                **policy.window_ranges.get(str(coords.get("kernel", "")).upper(), {}),
            }
            with r.numerics.context() as ctx:
                allowed = True
                for name, (lo, hi) in ranges.items():
                    name = "lambda" if name == "lambda_" else name
                    if name not in coords:
                        raise ValueError(
                            f"selection range references missing coordinate {name}"
                        )
                    value = ctx.number(coords[name])
                    allowed &= within_range(
                        value,
                        None if lo is None else ctx.number(lo),
                        None if hi is None else ctx.number(hi),
                    )
                j = report_indices[r.id]
                bias = systematics.biases[maximum_indices[j], j]
                syst = abs(ctx.number(bias))
                stat = ctx.number(r.statistical_error)
                ctx.require_finite(
                    (ctx.number(r.value), stat, syst), "selection candidate"
                )
                if not allowed:
                    continue
                total = ctx.sqrt(stat**2 + syst**2)
                order = candidate_order(
                    value=ctx.number(r.value),
                    stat=stat,
                    syst=syst,
                    total=total,
                    lambda_value=ctx.number(coords.get("lambda", 0)),
                    width=ctx.number(
                        coords.get("width", coords.get("delta_factor", 0))
                    ),
                    index=len(candidates),
                    factor=ctx.number(policy.factor),
                    isfinite=ctx.math.isfinite,
                )
                if order is not None:
                    candidates.append((order, r))
        if not candidates:
            if policy.missing == "raise":
                raise ValueError(
                    f"no statistically dominated candidate in group {dict(zip(group_by,key))}"
                )
            skipped.append(dict(zip(group_by, key)))
            continue
        chosen.append(min(candidates, key=lambda c: c[0])[1])
        labels.append(dict(zip(group_by, key)))
    return Selection(tuple(chosen), policy, tuple(labels), tuple(skipped))
