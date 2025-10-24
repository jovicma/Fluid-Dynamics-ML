"""Tests for the FNO model forward pass."""

from __future__ import annotations

import torch

from riemann_ml.ml.fno.model import FNO1DModel


def test_fno_forward_shape():
    config = {
        "in_channels": 4,
        "out_channels": 3,
        "n_modes": 8,
        "hidden_channels": 16,
        "n_layers": 2,
    }
    model = FNO1DModel(config)
    x = torch.randn(2, 4, 64)
    y = model(x)
    assert y.shape == (2, 3, 64)
