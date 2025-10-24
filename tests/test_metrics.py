"""Tests for evaluation metrics."""

from __future__ import annotations

import numpy as np

from riemann_ml.eval.metrics import (
    contact_plateau_error,
    relative_l2,
    shock_location_error,
)


def test_relative_l2_zero_for_identical_vectors():
    vec = np.array([1.0, 2.0, 3.0])
    assert relative_l2(vec, vec) == 0.0


def test_shock_location_error_detects_shift():
    x = np.linspace(0.0, 1.0, 64)
    target = np.where(x < 0.5, 1.0, 0.1)
    shifted = np.where(x < 0.55, 1.0, 0.1)
    err = shock_location_error(x, shifted, target)
    assert err > 0.0


def test_contact_plateau_error_small_for_similar_profiles():
    x = np.linspace(0.0, 1.0, 64)
    base = np.where(x < 0.6, 1.0, 0.2)
    perturbed = base + 0.01
    err = contact_plateau_error(x, perturbed, base)
    assert err < 0.05
