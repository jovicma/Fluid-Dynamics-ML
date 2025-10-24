"""Tests for the exact Sod shock-tube solution."""

from __future__ import annotations

import numpy as np
import pytest

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.exact.sod_exact import sod_exact_profile


@pytest.fixture()
def sod_states() -> tuple[StatePrim, StatePrim]:
    left = StatePrim(density=1.0, velocity=0.0, pressure=1.0)
    right = StatePrim(density=0.125, velocity=0.0, pressure=0.1)
    return left, right


def test_pressure_star_matches_reference(sod_states: tuple[StatePrim, StatePrim]):
    left, right = sod_states
    x = np.array([0.0])
    t = 0.2
    rho, u, p = sod_exact_profile(x, t, left, right)
    expected_p = 0.30313  # Known solution (Toro, 2009) near contact
    assert p[0] == pytest.approx(expected_p, rel=1e-3)


def test_wave_structure_is_monotone(sod_states: tuple[StatePrim, StatePrim]):
    left, right = sod_states
    x = np.linspace(-0.2, 0.4, 512)
    t = 0.2
    rho, u, p = sod_exact_profile(x, t, left, right)

    shock_front_idx = np.argmax(np.abs(np.gradient(rho)))
    contact_idx = np.argmax(np.gradient(u))
    assert shock_front_idx > contact_idx, "Shock should appear to the right of the contact."
    rarefaction_width = np.sum(np.abs(np.gradient(p)) < 1e-2)
    assert rarefaction_width > 0
