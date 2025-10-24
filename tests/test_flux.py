"""Tests for flux computations and positivity."""

from __future__ import annotations

from riemann_ml.core.euler1d import StatePrim, flux_vector, prim_to_cons


def test_flux_energy_component_positive():
    prim = StatePrim(density=1.0, velocity=1.0, pressure=1.0)
    gamma = 1.4
    cons = prim_to_cons(prim, gamma=gamma)
    _, _, energy_flux = flux_vector(cons, gamma=gamma)
    assert energy_flux > 0.0
