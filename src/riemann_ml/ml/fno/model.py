"""Fourier Neural Operator model definitions."""

from __future__ import annotations

from typing import Dict

import torch
from neuralop.models.fno import FNO


class FNO1DModel(torch.nn.Module):
    """Wrapper around neuraloperator's FNO for 1-D problems."""

    def __init__(self, config: Dict) -> None:
        super().__init__()
        n_modes = int(config.get("n_modes", 16))
        hidden_channels = int(config.get("hidden_channels", 64))
        n_layers = int(config.get("n_layers", 4))
        in_channels = int(config.get("in_channels", 4))
        out_channels = int(config.get("out_channels", 3))

        self.fno = FNO(
            n_modes=(n_modes,),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fno(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["FNO1DModel"]
