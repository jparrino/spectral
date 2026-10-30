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

import csv
from pathlib import Path
import mpmath as mp
from ..core.result import Result, Failure


def result_items(results):
    if isinstance(results, (Result, Failure)):
        return (results,)
    return tuple(results.results) if hasattr(results, "results") else tuple(results)


def number_text(value):
    if hasattr(value, "_mpf_"):
        digits = max(17, int(value._mpf_[3] * 0.30103) + 3)
        return mp.nstr(value, digits)
    return str(value)


def write_csv(
    path, *, results, systematics=None, include_coefficients=False, overwrite=False
):
    items = result_items(results)
    reports = (
        {}
        if systematics is None
        else {key: j for j, key in enumerate(systematics.result_ids)}
    )
    if systematics is not None and any(
        r.id not in reports for r in items if isinstance(r, Result)
    ):
        raise ValueError("systematic report does not belong to the exported results")
    maximum_indices = systematics.maximum_indices if systematics is not None else ()
    rows = []
    for r in items:
        row = {**getattr(r, "metadata", {}), **r.coordinates}
        if isinstance(r, Failure):
            row.update(status="failed", error=r.error)
        else:
            for key in ("lambda", "regulator", "algorithm"):
                if key in r.diagnostics:
                    row.setdefault(key, number_text(r.diagnostics[key]))
            row.update(
                status="ok",
                value=number_text(r.value),
                statistical_error=number_text(r.statistical_error),
            )
            if include_coefficients:
                row.update(
                    {
                        f"coefficient_{i}": number_text(q)
                        for i, q in enumerate(r.coefficients)
                    }
                )
            if systematics is not None:
                j = reports[r.id]
                i = maximum_indices[j]
                bias = systematics.biases[i, j]
                with r.numerics.context() as ctx:
                    total = ctx.sqrt(ctx.number(r.variance) + ctx.number(bias) ** 2)
                    magnitude = abs(ctx.number(bias))
                    ctx.require_finite(total, "CSV total error")
                row.update(
                    systematic_error=number_text(magnitude),
                    signed_bias_at_maximum=number_text(bias),
                    total_error=number_text(total),
                )
        rows.append(row)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w" if overwrite else "x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
