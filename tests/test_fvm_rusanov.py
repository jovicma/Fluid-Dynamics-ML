"""Tests for the 1-D Euler finite-volume solver."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.fvm.solver import advance_one_step, initialize_grid, simulate


def _sod_left_right():
    left = StatePrim(density=1.0, velocity=0.0, pressure=1.0)
    right = StatePrim(density=0.125, velocity=0.0, pressure=0.1)
    return left, right


def test_advance_one_step_cfl_and_finiteness():
    left, right = _sod_left_right()
    num_cells = 50
    x, dx = initialize_grid(num_cells)
    interface = 0.5

    from riemann_ml.core.euler1d import prim_to_cons

    left_cons = prim_to_cons(left)
    right_cons = prim_to_cons(right)
    state = np.zeros((num_cells, 3), dtype=np.float64)
    mask = x <= interface
    state[mask] = np.array([left_cons.density, left_cons.momentum, left_cons.energy])
    state[~mask] = np.array(
        [right_cons.density, right_cons.momentum, right_cons.energy]
    )

    cfl = 0.5
    next_state, dt, max_speed = advance_one_step(state, dx, cfl)

    assert np.isfinite(next_state).all()
    assert dt > 0
    assert max_speed > 0
    expected_dt = cfl * dx / max(max_speed, 1e-12)
    assert dt == pytest.approx(expected_dt)


def test_simulate_returns_expected_shapes_and_history():
    left, right = _sod_left_right()
    num_cells = 40
    final_time = 0.05
    cfl = 0.45

    times, x, final_state, history = simulate(
        num_cells=num_cells,
        final_time=final_time,
        cfl=cfl,
        left_state=left,
        right_state=right,
        store_history=True,
        history_stride=2,
    )

    assert times[-1] == pytest.approx(final_time, rel=1e-6)
    assert x.shape == (num_cells,)
    assert final_state.shape == (num_cells, 3)
    assert np.isfinite(final_state).all()
    assert history is not None
    assert len(history) >= 1
    for entry in history:
        assert entry.state.shape == (num_cells, 3)
        assert entry.time <= final_time + 1e-12
