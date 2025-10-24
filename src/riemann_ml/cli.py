"""Command-line interface for the riemann-ml project."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.exact import sod_exact_profile
from riemann_ml.fvm import solver as fvm_solver
from riemann_ml.utils.plotting import plot_profiles

app = typer.Typer(help="Riemann-ML command-line interface.")

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_CONFIG = "fvm"


def _load_config(name: str = DEFAULT_CONFIG):
    """Load a Hydra configuration located under the package config directory."""
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name=name)
    return cfg


def _state_from_cfg(cfg_section) -> StatePrim:
    container = OmegaConf.to_container(cfg_section, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Configuration section must be convertible to dict.")
    return StatePrim(**container)


@app.command("show-config")
def show_config(name: str = typer.Option(DEFAULT_CONFIG, "--name", "-n", help="Configuration name to display.")) -> None:
    """Print the resolved Hydra configuration as YAML."""
    cfg = _load_config(name)
    typer.echo(OmegaConf.to_yaml(cfg))


@app.command("simulate-fvm")
def simulate_fvm(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Configuration used for the run."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional path to save an NPZ snapshot."),
    history: bool = typer.Option(False, "--history", help="Store intermediate states according to stride."),
    history_stride: int = typer.Option(1, "--history-stride", help="Stride for history snapshots.", min=1),
) -> None:
    """Run the finite-volume solver with parameters pulled from the config."""
    cfg = _load_config(config)
    num_cells = int(cfg.fvm.num_cells)
    final_time = float(cfg.fvm.final_time)
    cfl = float(cfg.fvm.cfl)
    gamma = float(cfg.fvm.gamma)
    interface = float(cfg.fvm.interface_position)

    left_state = _state_from_cfg(cfg.fvm.left_state)
    right_state = _state_from_cfg(cfg.fvm.right_state)

    times, x, q, history_data = fvm_solver.simulate(
        num_cells=num_cells,
        final_time=final_time,
        cfl=cfl,
        left_state=left_state,
        right_state=right_state,
        gamma=gamma,
        interface_position=interface,
        store_history=history,
        history_stride=history_stride,
    )

    typer.echo(f"Finished simulation at t={times[-1]:.6f} with {len(times) - 1} steps.")

    if output is not None:
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "times": times,
            "x": x,
            "q": q,
            "gamma": gamma,
        }
        if history and history_data:
            data["history_times"] = np.array([entry.time for entry in history_data], dtype=np.float64)
            data["history_states"] = np.stack([entry.state for entry in history_data])
        np.savez(output, **data)
        typer.echo(f"Saved results to {output}")


@app.command("plot-sod")
def plot_sod(
    config: str = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="Configuration used for simulation/plots."),
    output_dir: Path = typer.Option(Path("data/artifacts/cli_sod"), "--output-dir", "-o", help="Directory to store plots."),
) -> None:
    """Simulate the Sod problem and write FVM vs. exact comparison plots."""
    cfg = _load_config(config)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    left_state = _state_from_cfg(cfg.fvm.left_state)
    right_state = _state_from_cfg(cfg.fvm.right_state)

    times, x, q, _ = fvm_solver.simulate(
        num_cells=int(cfg.fvm.num_cells),
        final_time=float(cfg.fvm.final_time),
        cfl=float(cfg.fvm.cfl),
        left_state=left_state,
        right_state=right_state,
        gamma=float(cfg.fvm.gamma),
        interface_position=float(cfg.fvm.interface_position),
        store_history=False,
    )

    rho = q[:, 0]
    momentum = q[:, 1]
    energy = q[:, 2]
    velocity = momentum / np.clip(rho, 1e-12, None)
    pressure = (float(cfg.fvm.gamma) - 1.0) * np.maximum(energy - 0.5 * momentum**2 / np.clip(rho, 1e-12, None), 1e-12)

    fvm_plot = output_dir / "fvm_profiles.png"
    plot_profiles(
        x=x,
        density=rho,
        velocity=velocity,
        pressure=pressure,
        title=f"Sod shock tube - t = {times[-1]:.3f}",
        savepath=fvm_plot,
    )

    rho_exact, u_exact, p_exact = sod_exact_profile(
        x - float(cfg.fvm.interface_position),
        float(cfg.fvm.final_time),
        left_state=left_state,
        right_state=right_state,
        gamma=float(cfg.fvm.gamma),
    )

    comparison_path = output_dir / "fvm_vs_exact.png"

    plt.figure(figsize=(10, 9))
    plt.subplot(3, 1, 1)
    plt.plot(x, rho, label="FVM")
    plt.plot(x, rho_exact, "--", label="Exact")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(x, velocity, label="FVM")
    plt.plot(x, u_exact, "--", label="Exact")
    plt.ylabel("Velocity")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(x, pressure, label="FVM")
    plt.plot(x, p_exact, "--", label="Exact")
    plt.ylabel("Pressure")
    plt.xlabel("x")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(comparison_path, dpi=150)
    plt.close()

    typer.echo(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    app()
