"""Analytical solution for the Sod shock tube problem.

The implementation follows the derivations presented in

* E. F. Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*,
  Springer, 3rd edition, 2009, Chapter 4.

The solution features a left rarefaction (or shock), a contact discontinuity,
and a right shock (or rarefaction). The classical Sod initial states yield a
rarefaction to the left, contact at ``u_*``, and a shock to the right.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from riemann_ml.core.euler1d import EPSILON, StatePrim

__all__ = ["sod_exact_profile"]


def _sound_speed(state: StatePrim, gamma: float) -> float:
    return math.sqrt(max(gamma * state.pressure / state.density, EPSILON))


def _wave_function(p: float, state: StatePrim, gamma: float) -> float:
    """Return the Hugoniot/Prandtl-Meyer function f_i(p)."""
    rho_i = state.density
    p_i = state.pressure
    c_i = _sound_speed(state, gamma)

    if p > p_i:  # shock
        A = 2.0 / ((gamma + 1.0) * rho_i)
        B = (gamma - 1.0) / (gamma + 1.0) * p_i
        sqrt_term = math.sqrt(A / (p + B))
        return (p - p_i) * sqrt_term

    exponent = (gamma - 1.0) / (2.0 * gamma)
    return (2.0 * c_i / (gamma - 1.0)) * ((p / p_i) ** exponent - 1.0)


def _wave_function_derivative(p: float, state: StatePrim, gamma: float) -> float:
    rho_i = state.density
    p_i = state.pressure
    c_i = _sound_speed(state, gamma)

    if p > p_i:  # shock
        A = 2.0 / ((gamma + 1.0) * rho_i)
        B = (gamma - 1.0) / (gamma + 1.0) * p_i
        sqrt_term = math.sqrt(A / (p + B))
        return sqrt_term * (1.0 - 0.5 * (p - p_i) / (p + B))

    exponent = -(gamma + 1.0) / (2.0 * gamma)
    return (1.0 / (rho_i * c_i)) * (p / p_i) ** exponent


def _initial_pressure_guess(left: StatePrim, right: StatePrim, gamma: float) -> float:
    """PVRS (pressure-velocity Riemann solver) guess."""
    c_l = _sound_speed(left, gamma)
    c_r = _sound_speed(right, gamma)
    p_avg = 0.5 * (left.pressure + right.pressure)
    u_diff = right.velocity - left.velocity
    p_pv = p_avg - 0.125 * u_diff * (left.density + right.density) * (c_l + c_r)
    return max(p_pv, EPSILON)


def _solve_pressure_star(left: StatePrim, right: StatePrim, gamma: float) -> float:
    p = _initial_pressure_guess(left, right, gamma)
    for _ in range(20):
        f = (
            _wave_function(p, left, gamma)
            + _wave_function(p, right, gamma)
            + (right.velocity - left.velocity)
        )
        df = _wave_function_derivative(p, left, gamma) + _wave_function_derivative(
            p, right, gamma
        )
        dp = -f / df
        p = max(p + dp, EPSILON)
        if abs(dp) < 1e-10:
            break
    return p


def _star_density(p_star: float, state: StatePrim, gamma: float) -> float:
    rho = state.density
    p = state.pressure
    if p_star > p:  # shock
        numerator = p_star / p + (gamma - 1.0) / (gamma + 1.0)
        denominator = (gamma - 1.0) / (gamma + 1.0) * p_star / p + 1.0
        return rho * numerator / denominator
    # rarefaction
    return rho * (p_star / p) ** (1.0 / gamma)


def _sample_left(
    xi: np.ndarray,
    p_star: float,
    u_star: float,
    left: StatePrim,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_l = left.density
    u_l = left.velocity
    p_l = left.pressure
    c_l = _sound_speed(left, gamma)

    if p_star > p_l:  # shock
        shock_speed = u_l - c_l * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * (p_star / p_l - 1.0) + 1.0
        )
        rho_star = _star_density(p_star, left, gamma)

        rho = np.where(xi <= shock_speed, rho_l, rho_star)
        u = np.where(xi <= shock_speed, u_l, u_star)
        p = np.where(xi <= shock_speed, p_l, p_star)
        return rho, u, p

    # rarefaction
    head = u_l - c_l
    c_star = c_l * (p_star / p_l) ** ((gamma - 1.0) / (2.0 * gamma))
    tail = u_star - c_star

    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    mask_left_state = xi <= head
    mask_fan = (xi > head) & (xi < tail)
    mask_star = xi >= tail

    rho[mask_left_state] = rho_l
    u[mask_left_state] = u_l
    p[mask_left_state] = p_l

    if np.any(mask_fan):
        term = (2.0 / (gamma + 1.0)) + ((gamma - 1.0) / ((gamma + 1.0) * c_l)) * (
            u_l - xi[mask_fan]
        )
        term = np.maximum(term, EPSILON)
        rho[mask_fan] = rho_l * term ** (2.0 / (gamma - 1.0))
        u[mask_fan] = (2.0 / (gamma + 1.0)) * (
            c_l + 0.5 * (gamma - 1.0) * u_l + xi[mask_fan]
        )
        p[mask_fan] = p_l * term ** (2.0 * gamma / (gamma - 1.0))

    rho_star = _star_density(p_star, left, gamma)
    rho[mask_star] = rho_star
    u[mask_star] = u_star
    p[mask_star] = p_star

    return rho, u, p


def _sample_right(
    xi: np.ndarray,
    p_star: float,
    u_star: float,
    right: StatePrim,
    gamma: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho_r = right.density
    u_r = right.velocity
    p_r = right.pressure
    c_r = _sound_speed(right, gamma)

    if p_star > p_r:  # shock
        shock_speed = u_r + c_r * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * (p_star / p_r - 1.0) + 1.0
        )
        rho_star = _star_density(p_star, right, gamma)

        rho = np.where(xi >= shock_speed, rho_r, rho_star)
        u = np.where(xi >= shock_speed, u_r, u_star)
        p = np.where(xi >= shock_speed, p_r, p_star)
        return rho, u, p

    # rarefaction
    head = u_r + c_r
    c_star = c_r * (p_star / p_r) ** ((gamma - 1.0) / (2.0 * gamma))
    tail = u_star + c_star

    rho = np.empty_like(xi)
    u = np.empty_like(xi)
    p = np.empty_like(xi)

    mask_right_state = xi >= head
    mask_fan = (xi > tail) & (xi < head)
    mask_star = xi <= tail

    rho[mask_right_state] = rho_r
    u[mask_right_state] = u_r
    p[mask_right_state] = p_r

    if np.any(mask_fan):
        term = (2.0 / (gamma + 1.0)) - ((gamma - 1.0) / ((gamma + 1.0) * c_r)) * (
            u_r - xi[mask_fan]
        )
        term = np.maximum(term, EPSILON)
        rho[mask_fan] = rho_r * term ** (2.0 / (gamma - 1.0))
        u[mask_fan] = (2.0 / (gamma + 1.0)) * (
            -c_r + 0.5 * (gamma - 1.0) * u_r + xi[mask_fan]
        )
        p[mask_fan] = p_r * term ** (2.0 * gamma / (gamma - 1.0))

    rho_star = _star_density(p_star, right, gamma)
    rho[mask_star] = rho_star
    u[mask_star] = u_star
    p[mask_star] = p_star
    return rho, u, p


def sod_exact_profile(
    x: np.ndarray,
    t: float,
    left_state: StatePrim,
    right_state: StatePrim,
    gamma: float = 1.4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the analytical Sod solution at positions ``x`` and time ``t``.

    Parameters
    ----------
    x:
        Spatial coordinates relative to the initial discontinuity (i.e., the
        interface is at ``x = 0``).
    t:
        Time at which the solution is sampled. For ``t <= 0`` the initial
        states are returned.
    left_state, right_state:
        Primitive states on the left and right of the discontinuity.
    gamma:
        Ratio of specific heats (``γ``).
    """
    x = np.asarray(x, dtype=np.float64)
    if t <= 0.0:
        rho = np.where(x <= 0.0, left_state.density, right_state.density)
        u = np.where(x <= 0.0, left_state.velocity, right_state.velocity)
        p = np.where(x <= 0.0, left_state.pressure, right_state.pressure)
        return rho, u, p

    p_star = _solve_pressure_star(left_state, right_state, gamma)
    u_star = 0.5 * (
        left_state.velocity
        + right_state.velocity
        + _wave_function(p_star, right_state, gamma)
        - _wave_function(p_star, left_state, gamma)
    )

    xi = x / t
    rho_left, u_left, p_left = _sample_left(xi, p_star, u_star, left_state, gamma)
    rho_right, u_right, p_right = _sample_right(xi, p_star, u_star, right_state, gamma)

    mask_left_of_contact = xi <= u_star
    rho = np.where(mask_left_of_contact, rho_left, rho_right)
    u = np.where(mask_left_of_contact, u_left, u_right)
    p = np.where(mask_left_of_contact, p_left, p_right)
    return rho, u, p
