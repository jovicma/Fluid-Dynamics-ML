"""Tests for the finite-volume solver."""

from __future__ import annotations

import numpy as np

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.fvm.solver import simulate


def test_short_time_step_stability():
    left = StatePrim(density=1.0, velocity=0.0, pressure=1.0)
    right = StatePrim(density=0.125, velocity=0.0, pressure=0.1)

    times, x, q, _ = simulate(
        num_cells=32,
        final_time=5e-3,
        cfl=0.4,
        left_state=left,
        right_state=right,
        gamma=1.4,
        store_history=False,
    )

    assert times[-1] > 0.0
    assert x.shape == (32,)
    assert np.isfinite(q).all()
    assert np.all(q[:, 0] > 0.0)
