"""Sanity checks for the PINN model."""

from __future__ import annotations

import tensorflow as tf

from riemann_ml.ml.pinn.model import PINN


def test_pinn_residual_shapes():
    config = {"hidden_layers": 2, "hidden_size": 8}
    model = PINN(config)
    x = tf.random.uniform((5, 1), dtype=tf.float32)
    t = tf.random.uniform((5, 1), dtype=tf.float32)
    residual = model.compute_residual(x, t, gamma=1.4)
    assert residual.shape == (5, 3)


def test_pinn_forward_returns_conservative_state():
    config = {"hidden_layers": 2, "hidden_size": 8}
    model = PINN(config)
    x = tf.zeros((4, 1), dtype=tf.float32)
    t = tf.zeros((4, 1), dtype=tf.float32)
    preds = model.predict_conservative(x, t, training=False)
    assert preds.shape == (4, 3)
