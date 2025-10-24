"""Plotting helpers for 1-D Euler simulations."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_profiles"]


def plot_profiles(
    x: Sequence[float],
    density: Sequence[float],
    velocity: Sequence[float],
    pressure: Sequence[float],
    title: str,
    savepath: str | Path | None,
) -> Path | None:
    """Plot density, velocity, and pressure profiles along the spatial domain."""
    x_arr = np.asarray(x, dtype=np.float64)
    rho = np.asarray(density, dtype=np.float64)
    vel = np.asarray(velocity, dtype=np.float64)
    pres = np.asarray(pressure, dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x_arr, rho, label=r"$\rho$")
    axes[0].set_ylabel("Density")
    axes[0].grid(True)

    axes[1].plot(x_arr, vel, label=r"$u$", color="tab:orange")
    axes[1].set_ylabel("Velocity")
    axes[1].grid(True)

    axes[2].plot(x_arr, pres, label=r"$p$", color="tab:green")
    axes[2].set_ylabel("Pressure")
    axes[2].set_xlabel("x")
    axes[2].grid(True)

    fig.suptitle(title)
    fig.tight_layout()

    output_path: Path | None = None
    if savepath is not None:
        output_path = Path(savepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return output_path
