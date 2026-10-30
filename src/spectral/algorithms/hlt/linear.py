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

"""Factor a constrained system once and apply it to several target RHS vectors."""

import numpy as np
import mpmath as mp
from scipy.linalg import lu_factor, lu_solve
import warnings

from ...execution import Outcome


def independent_constraints(columns, ctx):
    """Normalize columns and reject inconsistent or dependent constraints."""
    kept, scaled, scales, checks = [], [], [], []
    for i, column in enumerate(columns):
        scale = max(abs(x) for x in column)
        if scale == 0:
            checks.append((i, None, None, None))
            continue
        vector = column / scale
        duplicate = next(
            (
                j
                for j, old in enumerate(scaled)
                if max(abs(x - y) for x, y in zip(old, vector)) < 100 * ctx.eps
            ),
            None,
        )
        if duplicate is not None:
            previous = kept[duplicate]
            checks.append((i, previous, scale, scales[duplicate]))
            continue
        kept.append(i)
        scales.append(scale)
        scaled.append(vector)
    # Reorthogonalized Gram-Schmidt avoids accepting numerically dependent columns.
    orthogonal = []
    for vector in scaled:
        v = vector.copy()
        for _ in range(2):
            for u in orthogonal:
                v -= ctx.dot(u, v) * u
        norm = ctx.sqrt(ctx.dot(v, v))
        if norm <= 100 * len(vector) * ctx.eps:
            raise ValueError(
                "constraints are linearly dependent at the working precision"
            )
        orthogonal.append(v / norm)
    return kept, scaled, scales, checks


def factor(matrix, ctx):
    if not ctx.is_mp:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lu, piv = lu_factor(np.asarray(matrix, dtype=float))
        return lambda rhs: lu_solve((lu, piv), np.asarray(rhs, dtype=float))
    P, L, U = mp.lu(mp.matrix(matrix.tolist()))
    n = len(matrix)

    def solve(rhs):
        b = P * mp.matrix(list(rhs))
        y = [ctx.number(0)] * n
        x = [ctx.number(0)] * n
        for i in range(n):
            y[i] = b[i] - ctx.sum(L[i, j] * y[j] for j in range(i))
        for i in reversed(range(n)):
            x[i] = (y[i] - ctx.sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
        return ctx.array(x)

    return solve


def solve_batch(problem, targets, lambda_value, numerics):
    with numerics.context() as ctx:
        lam = ctx.number(lambda_value)
        if not ctx.math.isfinite(lam) or not 0 <= lam < 1:
            raise ValueError(
                "HLT lambda must satisfy 0 <= lambda < 1 after backend conversion; "
                "use the mpmath backend with higher precision if lambda rounds to 1"
            )
        A, regulator = ctx.array(problem.gram), ctx.array(problem.regulator)
        W = (1 - lam) * A + lam * regulator
        ctx.require_finite(W, "HLT matrix")
        n = len(W)
        kept, columns, scales, checks = independent_constraints(
            [ctx.array(c) for c in problem.columns], ctx
        )
        objective_scale = max(abs(x) for x in W.flat)
        if objective_scale == 0:
            raise ValueError("HLT matrix is zero")
        k = len(kept)
        matrix = ctx.array(np.zeros((n + k, n + k)))
        matrix[:n, :n] = W / objective_scale
        for i, column in enumerate(columns):
            matrix[:n, n + i] = column
            matrix[n + i, :n] = column
        solve = factor(matrix, ctx)
        solutions = []

        def solve_target(target):
            values = ctx.array(target.constraint_values)
            for i, previous, scale, previous_scale in checks:
                if previous is None:
                    if values[i] != 0:
                        raise ValueError("nonzero constraint on a zero basis vector")
                elif abs(values[i] / scale - values[previous] / previous_scale) > (
                    1000 * ctx.eps * max(1, abs(values[i] / scale))
                ):
                    raise ValueError("conflicting duplicate constraints")
            b = ctx.array(target.overlaps)
            rhs = ctx.array(
                list((1 - lam) * b / objective_scale)
                + [values[j] / scales[i] for i, j in enumerate(kept)]
            )
            answer = solve(rhs)
            if not all(ctx.math.isfinite(x) for x in answer):
                raise ArithmeticError("non-finite HLT solution")
            residual = matrix @ answer - rhs
            scale = max(
                ctx.number(1),
                max(abs(x) for x in rhs),
                max(abs(x) for x in matrix.flat) * max(abs(x) for x in answer),
            )
            ctx.require_finite(residual, "HLT residual")
            ctx.require_finite(scale, "HLT residual scale")
            residual_norm = max(abs(x) for x in residual) / scale
            if residual_norm > 10000 * len(matrix) * ctx.eps:
                raise ArithmeticError(
                    "HLT solve residual exceeds the working-precision tolerance"
                )
            q = answer[:n]
            mismatch = (
                ctx.dot(q, A @ q) - 2 * ctx.dot(q, b) + ctx.number(target.norm_squared)
            )
            ctx.require_finite(mismatch, "kernel mismatch")
            cancellation_scale = max(
                1,
                abs(ctx.dot(q, A @ q)),
                abs(2 * ctx.dot(q, b)),
                abs(ctx.number(target.norm_squared)),
            )
            if mismatch < -1000 * n * ctx.eps * cancellation_scale:
                raise ArithmeticError(
                    "negative kernel mismatch; increase numerical precision"
                )
            return (
                q,
                {
                    "lambda": lam,
                    "stationarity_residual": residual_norm,
                    "kernel_mismatch_squared": max(ctx.number(0), mismatch),
                    "constraint_residual": max(
                        [
                            abs(ctx.dot(ctx.array(c), q) - v)
                            for c, v in zip(problem.columns, values)
                        ]
                        or [ctx.number(0)]
                    ),
                },
            )

        for target in targets:
            try:
                solutions.append(Outcome(solve_target(target)))
            except Exception as exc:
                solutions.append(Outcome(error=f"{type(exc).__name__}: {exc}"))
        return solutions
