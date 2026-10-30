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

"""End-to-end mpmath implementation for the built-in HLT objects.

This module deliberately avoids NumPy and SciPy in every numerically
significant operation.  Callers are expected to enter an ``mp.workdps``
context before invoking these functions.  The Bernecker--Meyer inner kernel
uses the exact analytic representation of Appendix B.1 of
arXiv:1705.01775 instead of a nested numerical s-prime quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .precision import (
    is_mp_matrix,
    mp_matrix,
    mp_trapezoid_weights,
    mp_values,
    mp_vector,
    source_vector,
    to_mpf,
)

HBARC_MEV_FM = "197.3269804"
MUON_MASS_MEV = "105.6583745"
ALPHA_EM_DENOMINATOR = "137.035999084"


@dataclass
class MPWorkspace:
    """Precision-scoped caches and quadrature diagnostics."""

    dps: int
    quadrature_errors: dict[str, Any] = field(default_factory=dict)
    hvp_parameters: dict[int, dict[str, Any]] = field(default_factory=dict)
    hvp_kernel: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    bm_inner: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    window_laplace: dict[tuple[Any, ...], Any] = field(default_factory=dict)
    tilde_theta: dict[tuple[Any, ...], Any] = field(default_factory=dict)

    def key(self, owner: Any, value: Any) -> tuple[Any, ...]:
        value_key = getattr(value, "_mpf_", None)
        if value_key is None:
            value_key = str(value)
        return (id(owner), self.dps, value_key)

    def record_error(self, label: str, error: Any) -> None:
        previous = self.quadrature_errors.get(label)
        if previous is None or error > previous:
            self.quadrature_errors[label] = error


def _quad(mp, workspace: MPWorkspace, label: str, function, points, options):
    kwargs = {"error": True, "method": getattr(options, "mp_method", "gauss-legendre")}
    maxdegree = getattr(options, "mp_maxdegree", None)
    if maxdegree is not None:
        kwargs["maxdegree"] = int(maxdegree)
    value, error = mp.quad(function, points, **kwargs)
    workspace.record_error(label, error)
    return value


def _basis_parameters(basis, mp):
    times = [to_mpf(value, mp) for value in basis._times_source]
    p = to_mpf(basis._p_source, mp)
    T = None if basis._T_source is None else to_mpf(basis._T_source, mp)
    omega0 = to_mpf(basis._omega0_source, mp)
    return times, p, T, omega0


def basis_B(basis, t, omega, mp):
    _, _, T, _ = _basis_parameters(basis, mp)
    t = to_mpf(t, mp)
    omega = to_mpf(omega, mp)
    if basis.periodic:
        return mp.exp(-omega * t) + mp.exp(-omega * (T - t))
    return mp.exp(-omega * t)


def basis_phi(basis, t, omega, mp):
    _, p, _, _ = _basis_parameters(basis, mp)
    omega = to_mpf(omega, mp)
    return mp.power(omega, p) * basis_B(basis, t, omega, mp)


def _exp_integral(power, alpha, omega0, mp):
    if alpha <= 0:
        raise ValueError("exponential decay parameter alpha must be positive")
    a = power + 1
    return mp.gammainc(a, alpha * omega0, mp.inf) / mp.power(alpha, a)


def basis_A(basis, mp):
    times, p, T, omega0 = _basis_parameters(basis, mp)
    power = 2 * p
    matrix = mp.matrix(len(times), len(times))
    for i, ti in enumerate(times):
        for j in range(i, len(times)):
            tj = times[j]
            if basis.periodic:
                alphas = (ti + tj, ti + T - tj, T - ti + tj, 2 * T - ti - tj)
            else:
                alphas = (ti + tj,)
            value = mp.fsum(
                _exp_integral(power, alpha, omega0, mp) for alpha in alphas
            )
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def basis_R(basis, mp):
    times, p, T, omega0 = _basis_parameters(basis, mp)
    values = []
    for t in times:
        terms = [_exp_integral(p, t, omega0, mp)]
        if basis.periodic:
            terms.append(_exp_integral(p, T - t, omega0, mp))
        values.append(mp.fsum(terms))
    return mp.matrix(values)


def reconstructed_kernel(basis, q, omega, mp):
    q_values = [q[i] for i in range(q.rows)] if is_mp_matrix(q) else mp_values(q, mp)
    times, p, T, _ = _basis_parameters(basis, mp)
    if len(q_values) != len(times):
        raise ValueError("q has incompatible shape")

    def scalar(value):
        w = to_mpf(value, mp)
        omega_power = mp.power(w, p)
        values = []
        for t in times:
            B = mp.exp(-w * t)
            if basis.periodic:
                B += mp.exp(-w * (T - t))
            values.append(omega_power * B)
        return mp.fdot(q_values, values)

    if is_mp_matrix(omega):
        return mp.matrix([scalar(omega[i]) for i in range(omega.rows)])
    if isinstance(omega, (list, tuple)) or getattr(omega, "ndim", 0) == 1:
        return mp.matrix([scalar(value) for value in source_vector(omega)])
    return scalar(omega)


def _stable_sigmoid(x, mp):
    """Return (1+tanh(x))/2 without cancellation in either tail."""
    if x >= 0:
        e = mp.exp(-2 * x)
        return 1 / (1 + e)
    e = mp.exp(2 * x)
    return e / (1 + e)


def _smooth_window_difference(a, b, mp):
    """Stable ``smooth_step(a)-smooth_step(b)`` for a >= b."""
    return mp.sinh(a - b) / (2 * mp.cosh(a) * mp.cosh(b))


def _target_sources(target):
    return getattr(target, "_mp_sources", {})


def _high_energy_parameters(target, mp):
    src = _target_sources(target)
    return {
        "t0": to_mpf(src.get("t0", target.t0), mp),
        "am_tau": to_mpf(src.get("am_tau", target.am_tau), mp),
        "delta": None
        if getattr(target, "delta", None) is None
        else to_mpf(src.get("delta", target.delta), mp),
        "p": to_mpf(src.get("p", target.p), mp),
        "T": None if target.T is None else to_mpf(src.get("T", target.T), mp),
    }


def _window_bounds_from_source(source, a_inv, mp):
    if source is None:
        return None
    if isinstance(source, str):
        key = source.upper()
        if key in {"FULL", "TOTAL", "NONE"}:
            return None
        t0 = mp.mpf("0.4") * a_inv / mp.mpf(HBARC_MEV_FM)
        t1 = mp.mpf("1.0") * a_inv / mp.mpf(HBARC_MEV_FM)
        if key in {"ID", "INT", "INTERMEDIATE", "WIN", "WINDOW"}:
            return t0, t1
        if key in {"SD", "SHORT", "SHORT_DISTANCE"}:
            return mp.ninf, t0
        if key in {"LD", "LONG", "LONG_DISTANCE"}:
            return t1, mp.inf
        raise ValueError("unknown Euclidean window")
    values = tuple(source)
    return tuple(to_mpf(value, mp) for value in values)


def _hvp_parameters(target, mp, workspace: MPWorkspace):
    cached = workspace.hvp_parameters.get(id(target))
    if cached is not None:
        return cached
    src = _target_sources(target)
    a_inv_src = src.get("a_lat_inv_MeV", target.a_lat_inv_MeV)
    a_inv = None if a_inv_src is None else to_mpf(a_inv_src, mp)

    m_mu_src = src.get("m_mu_MeV", target.m_mu_MeV)
    if m_mu_src == target.m_mu_MeV and float(target.m_mu_MeV) == float(MUON_MASS_MEV):
        m_mu = mp.mpf(MUON_MASS_MEV)
    else:
        m_mu = to_mpf(m_mu_src, mp)

    am_mu_src = src.get("am_mu")
    am_mu = m_mu / a_inv if am_mu_src is None else to_mpf(am_mu_src, mp)

    alpha_src = src.get("alpha", target.alpha)
    if alpha_src == target.alpha and float(target.alpha) == 1.0 / float(
        ALPHA_EM_DENOMINATOR
    ):
        alpha = 1 / mp.mpf(ALPHA_EM_DENOMINATOR)
    else:
        alpha = to_mpf(alpha_src, mp)

    window_source = src.get("euclidean_window", target.euclidean_window)
    window = _window_bounds_from_source(window_source, a_inv, mp)
    window_delta_src = src.get("window_delta")
    if window is None:
        window_delta = (
            None if window_delta_src is None else to_mpf(window_delta_src, mp)
        )
    elif window_delta_src is None:
        window_delta = mp.mpf("0.15") * a_inv / mp.mpf(HBARC_MEV_FM)
    else:
        window_delta = to_mpf(window_delta_src, mp)

    result = {
        "am_tau": to_mpf(src.get("am_tau", target.am_tau), mp),
        "delta": None
        if target.delta is None
        else to_mpf(src.get("delta", target.delta), mp),
        "p": to_mpf(src.get("p", target.p), mp),
        "am_mu": am_mu,
        "alpha": alpha,
        "window": window,
        "window_delta": window_delta,
    }
    workspace.hvp_parameters[id(target)] = result
    return result


def _integrated_weights(target, mp):
    specification = getattr(target, "_mp_weight_spec", None)
    if specification is None:
        return [to_mpf(value, mp) for value in target._weight_sources]
    times, weights, apply_quadrature = specification
    values = [to_mpf(value, mp) for value in weights]
    if apply_quadrature:
        quadrature = mp_trapezoid_weights(times, mp)
        values = [q * value for q, value in zip(quadrature, values)]
    return values


def _window_theta_t(target, t, mp, workspace: MPWorkspace):
    params = _hvp_parameters(target, mp, workspace)
    window = params["window"]
    if window is None:
        return mp.one
    delta = params["window_delta"]
    left, right = window
    t = to_mpf(t, mp)
    if mp.isinf(left) and left < 0 and mp.isinf(right) and right > 0:
        return mp.one
    if mp.isinf(left) and left < 0:
        return _stable_sigmoid((right - t) / delta, mp)
    if mp.isinf(right) and right > 0:
        return _stable_sigmoid((t - left) / delta, mp)
    a = (t - left) / delta
    b = (t - right) / delta
    return _smooth_window_difference(a, b, mp)


def _hvp_kernel_K(target, omega, mp, workspace: MPWorkspace):
    params = _hvp_parameters(target, mp, workspace)
    omega = to_mpf(omega, mp)
    if omega <= 0 or not mp.isfinite(omega):
        return mp.zero
    # K depends only on omega/am_mu, so smooth and sharp target objects can
    # share these values during a scan-grid evaluation.
    key = (
        "hvp_kernel",
        workspace.dps,
        getattr(params["am_mu"], "_mpf_", str(params["am_mu"])),
        getattr(omega, "_mpf_", str(omega)),
    )
    cached = workspace.hvp_kernel.get(key)
    if cached is not None:
        return cached
    r = mp.power(omega / params["am_mu"], 2)

    # Closed form of int_0^1 dx x^2(1-x)/(x^2+r(1-x)).  The r>4 branch
    # uses a cancellation-free conformal variable.  For small x, evaluate the
    # log1p remainder directly as its alternating series; evaluating three
    # nearly equal terms separately becomes inaccurate at the extreme nodes
    # of the outer omega quadrature.
    with mp.extradps(15):
        if r > 4:
            beta = mp.sqrt(1 - 4 / r)
            x = 4 / (r * (1 + beta) ** 2)
            if x < mp.mpf("0.075"):
                remainder = mp.zero
                power = x**3
                sign = 1
                n = 3
                while True:
                    term = sign * power / n
                    updated = remainder + term
                    if updated == remainder:
                        break
                    remainder = updated
                    power *= x
                    sign = -sign
                    n += 1
            else:
                remainder = mp.log1p(x) - x + x * x / 2
            value = (
                x * x * (2 - x * x) / 2
                + (1 + x) ** 2 * (1 + x**-2) * remainder
                + (1 + x) * x * x * mp.log(x) / (1 - x)
            )
        elif r == 4:
            value = 8 * mp.log(2) - mp.mpf("5.5")
        else:
            h = mp.sqrt(r * (4 - r))
            reciprocal_integral = 2 * (
                mp.atan((2 - r) / h) + mp.atan(r / h)
            ) / h
            log_coefficient = r * (2 - r) / 2
            reciprocal_coefficient = -r * (r * r - 4 * r + 2) / 2
            value = (
                mp.mpf("0.5")
                - r
                - log_coefficient * mp.log(r)
                + reciprocal_coefficient * reciprocal_integral
            )
    value = +value
    workspace.hvp_kernel[key] = value
    return value


def _bernecker_meyer_inner(target, t, mp, workspace: MPWorkspace):
    """Evaluate the s-prime integral with its exact analytic solution.

    For ``u = m_mu*t``, this implementation's inner integral ``J(t)`` is
    related to the TMR kernel of arXiv:1705.01775 by
    ``J(t) = m_mu**2*f_tilde(t)/pi**2``.  Equations (55) and (57) give the
    Meijer-G result.  Its exact Bessel-K/modified-Struve-L identity is used in
    the central region, with analytic small- and large-distance expansions
    only where their remainder is below the active working precision.
    """
    key = workspace.key(target, t)
    cached = workspace.bm_inner.get(key)
    if cached is not None:
        return cached
    t = to_mpf(t, mp)
    if t <= 0 or not mp.isfinite(t):
        return mp.zero
    m = _hvp_parameters(target, mp, workspace)["am_mu"]
    u = t * m

    # Use analytic limiting representations only once their conservative
    # omitted-term estimates are below the current working precision.  This
    # prevents special-function failures at the extreme nodes generated by an
    # infinite-interval quadrature without imposing a fixed numerical cutoff.
    log_u = mp.log(u)
    u2 = u * u
    short_value = (
        u2**2 / 9
        + u2**3 * (120 * log_u + 120 * mp.euler - 169) / 5400
        + u2**4 * (210 * log_u + 210 * mp.euler - 401) / 88200
        + u2**5 * (360 * log_u + 360 * mp.euler - 787) / 2916000
        + u2**6 * (3080 * log_u + 3080 * mp.euler - 7353) / 768398400
    )
    short_remainder_bound = 10 * u**14 * (abs(log_u) + 1)

    large_value = (
        2 * u2
        - 4 * mp.pi * u
        + 4 * (-1 + 4 * mp.euler + 4 * log_u)
        + 8 / u2
    )
    large_remainder_bound = (
        4 * mp.sqrt(mp.pi / u) * mp.exp(-2 * u) * (1 + 1 / u)
    )

    relative_goal = mp.eps / 10
    if u < 1 and short_remainder_bound <= relative_goal * abs(short_value):
        value = short_value
    elif u > 1 and large_remainder_bound <= relative_goal * abs(large_value):
        value = large_value
    else:
        # Individual terms scale as u**-2 while the result scales as u**4
        # near the origin.  Guard digits protect those six powers of
        # cancellation in the exact central-region representation.
        cancellation_digits = max(0, int(mp.ceil(-6 * mp.log10(u))))
        with mp.extradps(max(20, cancellation_digits + 10)):
            z = 2 * u
            k0 = mp.besselk(0, z)
            k1 = mp.besselk(1, z)
            closed_form_g = -4 * u * (
                k1
                + mp.pi
                * u
                * (k0 * mp.struvel(-1, z) + k1 * mp.struvel(0, z))
            )
            value = 2 * (
                -2
                + 8 * mp.euler
                + 4 / (u * u)
                + u * u
                - 8 * k1 / u
                + 8 * mp.log(u)
                + closed_form_g
            )
        value = +value
    workspace.bm_inner[key] = value
    return value


def _time_points(target, mp, workspace: MPWorkspace):
    window = _hvp_parameters(target, mp, workspace)["window"]
    points = [mp.zero]
    if window is not None:
        points.extend(value for value in window if mp.isfinite(value) and value > 0)
    points = sorted(set(points))
    points.append(mp.inf)
    return points


def tilde_theta(target, omega, mp, workspace: MPWorkspace):
    params = _hvp_parameters(target, mp, workspace)
    if params["window"] is None:
        return mp.one
    key = workspace.key(target, omega)
    cached = workspace.tilde_theta.get(key)
    if cached is not None:
        return cached
    omega = to_mpf(omega, mp)
    if omega <= 0 or not mp.isfinite(omega):
        return mp.zero
    kernel = _hvp_kernel_K(target, omega, mp, workspace)
    khat = 3 * mp.power(omega / params["am_mu"], 2) * kernel
    if khat <= 0 or not mp.isfinite(khat):
        return mp.zero

    integral = _window_laplace_transform(target, omega, mp, workspace)
    value = (
        3 * mp.power(omega, 5) * integral / (8 * mp.power(params["am_mu"], 4) * khat)
    )
    workspace.tilde_theta[key] = value
    return value


def _window_laplace_transform(target, omega, mp, workspace: MPWorkspace):
    key = workspace.key(target, omega)
    cached = workspace.window_laplace.get(key)
    if cached is not None:
        return cached

    def integrand(t):
        return (
            _window_theta_t(target, t, mp, workspace)
            * mp.exp(-omega * t)
            * _bernecker_meyer_inner(target, t, mp, workspace)
        )

    value = _quad(
        mp,
        workspace,
        "energy_space_window",
        integrand,
        _time_points(target, mp, workspace),
        target.window_quad_options,
    )
    workspace.window_laplace[key] = value
    return value


def target_eval(target, omega, mp, workspace: MPWorkspace):
    from .targets import (
        HighEnergyTarget,
        IntegratedHVPTarget,
        IntegratedTarget,
        SharpHighEnergyTarget,
    )

    omega = to_mpf(omega, mp)
    if isinstance(target, HighEnergyTarget):
        params = _high_energy_parameters(target, mp)
        theta = _stable_sigmoid((omega - params["am_tau"]) / params["delta"], mp)
        B = mp.exp(-omega * params["t0"])
        if target.periodic:
            B += mp.exp(-omega * (params["T"] - params["t0"]))
        return theta * mp.power(omega, params["p"]) * B
    if isinstance(target, SharpHighEnergyTarget):
        params = _high_energy_parameters(target, mp)
        if omega < params["am_tau"]:
            return mp.zero
        B = mp.exp(-omega * params["t0"])
        if target.periodic:
            B += mp.exp(-omega * (params["T"] - params["t0"]))
        return mp.power(omega, params["p"]) * B
    if isinstance(target, IntegratedTarget):
        weights = _integrated_weights(target, mp)
        return mp.fsum(
            weight * target_eval(component, omega, mp, workspace)
            for weight, component in zip(weights, target.targets)
        )
    if isinstance(target, IntegratedHVPTarget):
        if omega <= 0 or not mp.isfinite(omega):
            return mp.zero
        params = _hvp_parameters(target, mp, workspace)
        if target.sharp:
            theta = mp.one if omega >= params["am_tau"] else mp.zero
        else:
            theta = _stable_sigmoid((omega - params["am_tau"]) / params["delta"], mp)
        if params["window"] is not None:
            # K(s) cancels analytically between T_2^HVP and tilde-Theta.  Using
            # the reduced form avoids two large/small factors and an otherwise
            # unnecessary numerical K(s) quadrature.
            laplace = _window_laplace_transform(target, omega, mp, workspace)
            return (
                params["alpha"] ** 2
                * mp.power(omega, params["p"])
                * laplace
                * theta
                / params["am_mu"] ** 2
            )
        full_t2 = (
            8
            * params["alpha"] ** 2
            * _hvp_kernel_K(target, omega, mp, workspace)
            / omega
        )
        return mp.power(omega, params["p"] - 2) * full_t2 * theta

    custom = type(target).__dict__.get("eval_mpmath")
    if custom is not None:
        return custom(target, omega, mp=mp, workspace=workspace)
    raise TypeError(
        f"{type(target).__name__} does not provide an mpmath target evaluation"
    )


def target_values(target, omega, mp, workspace: MPWorkspace):
    if is_mp_matrix(omega):
        return mp.matrix(
            [target_eval(target, omega[i], mp, workspace) for i in range(omega.rows)]
        )
    if isinstance(omega, (list, tuple)) or getattr(omega, "ndim", 0) == 1:
        return mp.matrix(
            [
                target_eval(target, value, mp, workspace)
                for value in source_vector(omega)
            ]
        )
    return target_eval(target, omega, mp, workspace)


def _thresholds(target, mp, workspace: MPWorkspace):
    from .targets import (
        HighEnergyTarget,
        IntegratedHVPTarget,
        IntegratedTarget,
        SharpHighEnergyTarget,
    )

    if isinstance(target, (HighEnergyTarget, SharpHighEnergyTarget)):
        params = _high_energy_parameters(target, mp)
        values = [params["am_tau"]]
        if isinstance(target, HighEnergyTarget):
            values.extend(
                [
                    params["am_tau"] - 6 * params["delta"],
                    params["am_tau"] + 6 * params["delta"],
                ]
            )
        return values
    if isinstance(target, IntegratedHVPTarget):
        params = _hvp_parameters(target, mp, workspace)
        values = [params["am_tau"]]
        if not target.sharp:
            values.extend(
                [
                    params["am_tau"] - 6 * params["delta"],
                    params["am_tau"] + 6 * params["delta"],
                ]
            )
        return values
    if isinstance(target, IntegratedTarget):
        values = []
        for component in target.targets:
            values.extend(_thresholds(component, mp, workspace))
        return values
    return []


def _omega_points(target, lower, mp, workspace: MPWorkspace):
    points = [to_mpf(lower, mp)]
    points.extend(
        value
        for value in _thresholds(target, mp, workspace)
        if value > points[0] and mp.isfinite(value)
    )
    points = sorted(set(points))
    points.append(mp.inf)
    return points


def _area_diverges(target, mp, workspace: MPWorkspace) -> bool:
    from .targets import IntegratedHVPTarget

    if not isinstance(target, IntegratedHVPTarget):
        return False
    params = _hvp_parameters(target, mp, workspace)
    if params["p"] < 4:
        return False
    if params["window"] is None:
        return True
    # A smooth tanh/logistic window has nonzero t=0 leakage.  No numerical
    # cutoff is applied in the arbitrary-precision path.
    return _window_theta_t(target, mp.zero, mp, workspace) != 0


def target_area(target, omega0, options, mp, workspace: MPWorkspace):
    if _area_diverges(target, mp, workspace):
        from .targets import NonIntegrableTargetAreaError

        raise NonIntegrableTargetAreaError(
            "the smooth-window target has a mathematically divergent unweighted area; "
            "use area_constraint=False/'auto' or an explicitly compact-support window"
        )
    return _quad(
        mp,
        workspace,
        "target_area",
        lambda omega: target_eval(target, omega, mp, workspace),
        _omega_points(target, omega0, mp, workspace),
        options,
    )


def target_norm_squared(target, omega0, options, mp, workspace: MPWorkspace):
    """Return ``int_omega0^inf target(omega)^2 d omega`` once per preparation."""
    return _quad(
        mp,
        workspace,
        "target_norm",
        lambda omega: target_eval(target, omega, mp, workspace) ** 2,
        _omega_points(target, omega0, mp, workspace),
        options,
    )


def target_b_vector(target, basis, options, mp, workspace: MPWorkspace):
    times, p, T, omega0 = _basis_parameters(basis, mp)
    points = _omega_points(target, omega0, mp, workspace)
    values = []
    for index, t in enumerate(times):
        def integrand(omega, tt=t):
            B = mp.exp(-omega * tt)
            if basis.periodic:
                B += mp.exp(-omega * (T - tt))
            return mp.power(omega, p) * B * target_eval(
                target, omega, mp, workspace
            )

        value = _quad(
            mp,
            workspace,
            f"b_vector[{index}]",
            integrand,
            points,
            options,
        )
        values.append(value)
    return mp.matrix(values)


def kernel_mismatch_integrals(target, basis, q, options, mp, workspace: MPWorkspace):
    _, _, _, omega0 = _basis_parameters(basis, mp)
    points = _omega_points(target, omega0, mp, workspace)

    def difference_squared(omega):
        difference = reconstructed_kernel(basis, q, omega, mp) - target_eval(
            target, omega, mp, workspace
        )
        return difference * difference

    mismatch = _quad(
        mp, workspace, "kernel_mismatch", difference_squared, points, options
    )
    norm = _quad(
        mp,
        workspace,
        "target_norm",
        lambda omega: target_eval(target, omega, mp, workspace) ** 2,
        points,
        options,
    )
    mismatch_abs = mp.sqrt(max(mismatch, mp.zero))
    floor = mp.power(10, -(mp.dps - 5))
    mismatch_rel = mismatch_abs / max(mp.sqrt(max(norm, mp.zero)), floor)
    return mismatch_abs, mismatch_rel


def selected_data_matrices(data, selected_times, mp):
    mean_source, covariance_source = data.sliced_sources(selected_times)
    return mp_vector(mean_source, mp), mp_matrix(covariance_source, mp)


def matrix_quadratic(vector, matrix, mp):
    return (vector.T * matrix * vector)[0]


def matrix_trace(matrix, mp):
    return mp.fsum(matrix[i, i] for i in range(matrix.rows))


def frobenius_norm(matrix, mp):
    return mp.sqrt(
        mp.fsum(
            abs(matrix[i, j]) ** 2
            for i in range(matrix.rows)
            for j in range(matrix.cols)
        )
    )


def identity(size, mp):
    return mp.eye(size)
