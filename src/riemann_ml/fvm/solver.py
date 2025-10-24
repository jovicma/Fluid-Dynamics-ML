"""Finite-volume solver for the 1-D Euler equations using the Rusanov scheme."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from riemann_ml.core.euler1d import EPSILON, StateCons, StatePrim, prim_to_cons

__all__ = [
    "SimulationHistoryEntry",
    "initialize_grid",
    "riemann_flux_rusanov",
    "advance_one_step",
    "simulate",
    "extend_state",
    "max_wave_speed",
]


@dataclass(frozen=True)
class SimulationHistoryEntry:
    """Container for storing snapshots of the conservative state."""

    time: float
    state: np.ndarray


def initialize_grid(
    num_cells: int, domain: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[np.ndarray, float]:
    """Create a uniform 1-D Cartesian grid.

    Parameters
    ----------
    num_cells:
        Number of control volumes.
    domain:
        Tuple ``(x_min, x_max)`` describing the spatial domain.

    Returns
    -------
    Tuple[np.ndarray, float]
        Cell-center coordinates and the grid spacing ``dx``.
    """
    if num_cells <= 0:
        raise ValueError("num_cells must be positive.")
    x_min, x_max = domain
    if x_max <= x_min:
        raise ValueError("Invalid spatial domain. Expected x_max > x_min.")
    dx = (x_max - x_min) / num_cells
    centers = x_min + (np.arange(num_cells, dtype=np.float64) + 0.5) * dx
    return centers, dx


def _as_conservative_array(
    state: StateCons | Sequence[float] | np.ndarray,
) -> np.ndarray:
    if isinstance(state, StateCons):
        return np.array([state.density, state.momentum, state.energy], dtype=np.float64)
    array = np.asarray(state, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError("State array must have at least one dimension.")
    return array


def _primitive_variables(
    cons_state: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(cons_state, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] != 3:
        raise ValueError("Conservative state must have length 3.")
    density = np.clip(arr[:, 0], EPSILON, None)
    momentum = arr[:, 1]
    energy = np.clip(arr[:, 2], EPSILON, None)
    velocity = momentum / density
    kinetic = 0.5 * momentum**2 / density
    internal = np.maximum(energy - kinetic, EPSILON)
    pressure = (gamma - 1.0) * internal
    return density, velocity, pressure, energy


def _flux_from_state(cons_state: np.ndarray, gamma: float) -> np.ndarray:
    density, velocity, pressure, energy = _primitive_variables(cons_state, gamma)
    flux = np.zeros_like(cons_state, dtype=np.float64)
    flux[:, 0] = cons_state[:, 1]
    flux[:, 1] = cons_state[:, 1] * velocity + pressure
    flux[:, 2] = velocity * (energy + pressure)
    return flux


def riemann_flux_rusanov(
    left: StateCons | Sequence[float] | np.ndarray,
    right: StateCons | Sequence[float] | np.ndarray,
    gamma: float = 1.4,
) -> np.ndarray:
    """Compute the Rusanov (local Lax-Friedrichs) flux at interfaces."""
    left_arr = _as_conservative_array(left)
    right_arr = _as_conservative_array(right)
    left_arr = np.asarray(left_arr, dtype=np.float64)
    right_arr = np.asarray(right_arr, dtype=np.float64)
    original_shape = None
    if left_arr.ndim == 1:
        original_shape = left_arr.shape
        left_arr = left_arr[None, :]
    if right_arr.ndim == 1:
        right_arr = right_arr[None, :]
    if left_arr.shape != right_arr.shape:
        raise ValueError("Left and right states must share the same shape.")

    flux_left = _flux_from_state(left_arr, gamma)
    flux_right = _flux_from_state(right_arr, gamma)

    rho_l, vel_l, p_l, _ = _primitive_variables(left_arr, gamma)
    rho_r, vel_r, p_r, _ = _primitive_variables(right_arr, gamma)
    c_l = np.sqrt(gamma * p_l / rho_l)
    c_r = np.sqrt(gamma * p_r / rho_r)
    spectral_radius = np.maximum(np.abs(vel_l) + c_l, np.abs(vel_r) + c_r)
    flux = 0.5 * (flux_left + flux_right) - 0.5 * spectral_radius[:, None] * (
        right_arr - left_arr
    )

    if original_shape is not None:
        return flux[0]
    return flux


def extend_state(cons_state: np.ndarray, boundary: str = "outflow") -> np.ndarray:
    boundary = boundary.lower()
    if boundary == "outflow":
        left = cons_state[0][None, :]
        right = cons_state[-1][None, :]
        return np.vstack((left, cons_state, right))
    if boundary == "periodic":
        return np.vstack((cons_state[-1][None, :], cons_state, cons_state[0][None, :]))
    raise ValueError(
        f"Unsupported boundary condition '{boundary}'. Use 'outflow' or 'periodic'."
    )


def max_wave_speed(cons_state: np.ndarray, gamma: float) -> float:
    rho, vel, pressure, _ = _primitive_variables(cons_state, gamma)
    sound_speed = np.sqrt(gamma * pressure / rho)
    return float(np.max(np.abs(vel) + sound_speed))


def _update_state(
    cons_state: np.ndarray, dx: float, dt: float, gamma: float, boundary: str
) -> np.ndarray:
    cons_state = np.asarray(cons_state, dtype=np.float64)
    q_ext = extend_state(cons_state, boundary=boundary)
    left = q_ext[:-1]
    right = q_ext[1:]
    fluxes = riemann_flux_rusanov(left, right, gamma=gamma)
    flux_diff = fluxes[1:] - fluxes[:-1]
    updated = cons_state - (dt / dx) * flux_diff
    updated[:, 0] = np.clip(updated[:, 0], EPSILON, None)
    updated[:, 2] = np.clip(updated[:, 2], EPSILON, None)
    return updated


def advance_one_step(
    cons_state: np.ndarray,
    dx: float,
    cfl: float,
    gamma: float = 1.4,
    boundary: str = "outflow",
) -> Tuple[np.ndarray, float, float]:
    """Advance the conservative state by one explicit time step."""
    if cfl <= 0.0 or cfl >= 1.0:
        raise ValueError("CFL number should typically lie in (0, 1).")
    max_speed = max_wave_speed(cons_state, gamma)
    max_speed = max(max_speed, EPSILON)
    dt = cfl * dx / max_speed
    next_state = _update_state(cons_state, dx, dt, gamma, boundary=boundary)
    return next_state, dt, max_speed


def _prepare_state_array(
    num_cells: int,
    left_cons: StateCons,
    right_cons: StateCons,
    interface: float,
    x: np.ndarray,
) -> np.ndarray:
    cons_array = np.zeros((num_cells, 3), dtype=np.float64)
    left_values = np.array(
        [left_cons.density, left_cons.momentum, left_cons.energy], dtype=np.float64
    )
    right_values = np.array(
        [right_cons.density, right_cons.momentum, right_cons.energy], dtype=np.float64
    )
    mask_left = x <= interface
    cons_array[mask_left] = left_values
    cons_array[~mask_left] = right_values
    return cons_array


def simulate(
    num_cells: int,
    final_time: float,
    cfl: float,
    left_state: StatePrim | dict,
    right_state: StatePrim | dict,
    gamma: float = 1.4,
    interface_position: float = 0.5,
    store_history: bool = False,
    history_stride: int = 1,
    boundary: str = "outflow",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[List[SimulationHistoryEntry]]]:
    """Simulate a Riemann problem for the 1-D Euler equations."""
    if final_time <= 0.0:
        raise ValueError("final_time must be positive.")
    if interface_position <= 0.0 or interface_position >= 1.0:
        raise ValueError("interface_position must lie within the domain (0, 1).")
    if history_stride <= 0:
        raise ValueError("history_stride must be a positive integer.")

    x, dx = initialize_grid(num_cells)
    if not isinstance(left_state, StatePrim):
        left_state = StatePrim(**left_state)
    if not isinstance(right_state, StatePrim):
        right_state = StatePrim(**right_state)
    left_cons = prim_to_cons(left_state, gamma=gamma)
    right_cons = prim_to_cons(right_state, gamma=gamma)
    q = _prepare_state_array(num_cells, left_cons, right_cons, interface_position, x)

    time = 0.0
    times: List[float] = [time]
    history: Optional[List[SimulationHistoryEntry]] = [] if store_history else None
    if store_history and history is not None:
        history.append(SimulationHistoryEntry(time=time, state=q.copy()))

    step = 0
    while time < final_time:
        q_cfl, dt_cfl, _ = advance_one_step(q, dx, cfl, gamma=gamma, boundary=boundary)
        # Adjust the time step to hit final_time exactly.
        dt = dt_cfl
        if time + dt > final_time:
            dt = final_time - time
            if dt <= 0.0:
                break
            q = _update_state(q, dx, dt, gamma, boundary=boundary)
        else:
            q = q_cfl
        time += dt
        times.append(time)

        if store_history and history is not None and ((step + 1) % history_stride == 0):
            history.append(SimulationHistoryEntry(time=time, state=q.copy()))
        step += 1

    return np.asarray(times, dtype=np.float64), x, q, history
