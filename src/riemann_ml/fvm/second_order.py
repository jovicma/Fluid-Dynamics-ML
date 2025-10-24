"""Second-order MUSCL scheme with Rusanov flux for the 1-D Euler equations."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from riemann_ml.core.euler1d import EPSILON, StatePrim, prim_to_cons
from riemann_ml.fvm.solver import (
    SimulationHistoryEntry,
    extend_state,
    initialize_grid,
    max_wave_speed,
    riemann_flux_rusanov,
)

__all__ = [
    "minmod",
    "muscl_rusanov_step",
    "advance_one_step_muscl",
    "simulate_muscl",
]


def minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise Minmod limiter."""
    result = np.zeros_like(a)
    mask = (a * b) > 0.0
    result[mask] = np.where(np.abs(a[mask]) < np.abs(b[mask]), a[mask], b[mask])
    return result


def muscl_rusanov_step(
    cons_state: np.ndarray,
    dx: float,
    dt: float,
    gamma: float = 1.4,
    boundary: str = "outflow",
) -> np.ndarray:
    """Advance one step using MUSCL reconstruction with Rusanov flux."""
    q_ext = extend_state(cons_state, boundary=boundary)
    slopes = np.zeros_like(q_ext)
    dq_minus = q_ext[1:-1] - q_ext[:-2]
    dq_plus = q_ext[2:] - q_ext[1:-1]
    slopes[1:-1] = minmod(dq_minus, dq_plus)

    q_plus = q_ext + 0.5 * slopes  # right state of each cell
    q_minus = q_ext - 0.5 * slopes  # left state of each cell

    left_states = q_plus[:-1]
    right_states = q_minus[1:]
    fluxes = riemann_flux_rusanov(left_states, right_states, gamma=gamma)
    flux_diff = fluxes[1:] - fluxes[:-1]

    updated = cons_state - (dt / dx) * flux_diff
    updated[:, 0] = np.clip(updated[:, 0], EPSILON, None)
    updated[:, 2] = np.clip(updated[:, 2], EPSILON, None)
    return updated


def advance_one_step_muscl(
    cons_state: np.ndarray,
    dx: float,
    cfl: float,
    gamma: float = 1.4,
    boundary: str = "outflow",
) -> Tuple[np.ndarray, float, float]:
    """Wrapper computing adaptive dt for the MUSCL scheme."""
    max_speed = max_wave_speed(cons_state, gamma)
    max_speed = max(max_speed, EPSILON)
    dt = cfl * dx / max_speed
    next_state = muscl_rusanov_step(cons_state, dx, dt, gamma=gamma, boundary=boundary)
    return next_state, dt, max_speed


def simulate_muscl(
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
    """Second-order MUSCL simulation for the Sod problem."""
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
    cons_array = np.zeros((num_cells, 3), dtype=np.float64)
    mask_left = x <= interface_position
    cons_array[mask_left] = np.array(
        [left_cons.density, left_cons.momentum, left_cons.energy], dtype=np.float64
    )
    cons_array[~mask_left] = np.array(
        [right_cons.density, right_cons.momentum, right_cons.energy], dtype=np.float64
    )

    time = 0.0
    times: List[float] = [time]
    history: Optional[List[SimulationHistoryEntry]] = [] if store_history else None
    if store_history and history is not None:
        history.append(SimulationHistoryEntry(time=time, state=cons_array.copy()))

    step = 0
    while time < final_time:
        q_cfl, dt_cfl, _ = advance_one_step_muscl(
            cons_array, dx, cfl, gamma=gamma, boundary=boundary
        )
        dt = dt_cfl
        if time + dt > final_time:
            dt = final_time - time
            if dt <= 0.0:
                break
            cons_array = muscl_rusanov_step(
                cons_array, dx, dt, gamma=gamma, boundary=boundary
            )
        else:
            cons_array = q_cfl
        time += dt
        times.append(time)

        if store_history and history is not None and ((step + 1) % history_stride == 0):
            history.append(SimulationHistoryEntry(time=time, state=cons_array.copy()))
        step += 1

    return np.asarray(times, dtype=np.float64), x, cons_array, history
