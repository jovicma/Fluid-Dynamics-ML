"""Tests for primitive-conservative conversions."""

from __future__ import annotations

import numpy as np

from riemann_ml.core.euler1d import StatePrim, cons_to_prim, prim_to_cons


def test_round_trip_conversion_consistency():
    prim = StatePrim(density=1.2, velocity=0.7, pressure=0.9)
    gamma = 1.4

    cons = prim_to_cons(prim, gamma=gamma)
    recovered = cons_to_prim(cons, gamma=gamma)

    assert np.isclose(cons.density, prim.density)
    assert np.isclose(recovered.velocity, prim.velocity)
    assert np.isclose(recovered.pressure, prim.pressure)
