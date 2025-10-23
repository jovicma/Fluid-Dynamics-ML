"""Unit tests for Euler 1D utilities."""

from __future__ import annotations

import math

import pytest

from riemann_ml.core.euler1d import (
    StateCons,
    StatePrim,
    cons_to_prim,
    flux_vector,
    prim_to_cons,
    sound_speed,
)


def test_round_trip_primitive_to_conservative():
    prim = StatePrim(density=1.0, velocity=2.5, pressure=1.2)
    gamma = 1.4

    cons = prim_to_cons(prim, gamma=gamma)
    recovered = cons_to_prim(cons, gamma=gamma)

    assert recovered.density == pytest.approx(prim.density, rel=1e-12)
    assert recovered.velocity == pytest.approx(prim.velocity, rel=1e-12)
    assert recovered.pressure == pytest.approx(prim.pressure, rel=1e-12)


def test_flux_vector_sanity():
    prim = StatePrim(density=1.0, velocity=2.0, pressure=1.0)
    gamma = 1.4
    cons = prim_to_cons(prim, gamma=gamma)

    flux = flux_vector(cons, gamma=gamma)

    expected_mass_flux = prim.density * prim.velocity
    expected_momentum_flux = prim.density * prim.velocity**2 + prim.pressure
    total_energy = cons.energy
    expected_energy_flux = prim.velocity * (total_energy + prim.pressure)

    assert flux[0] == pytest.approx(expected_mass_flux, rel=1e-12)
    assert flux[1] == pytest.approx(expected_momentum_flux, rel=1e-12)
    assert flux[2] == pytest.approx(expected_energy_flux, rel=1e-12)


def test_sound_speed_matches_analytic_expression():
    prim = StatePrim(density=0.8, velocity=0.3, pressure=0.5)
    gamma = 1.4
    cons = prim_to_cons(prim, gamma=gamma)

    speed = sound_speed(cons, gamma=gamma)
    expected = math.sqrt(gamma * prim.pressure / prim.density)

    assert speed == pytest.approx(expected, rel=1e-12)
