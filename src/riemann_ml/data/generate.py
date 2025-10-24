"""Dataset generation utilities for Riemann problems."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import h5py
import numpy as np
import typer
from omegaconf import OmegaConf
from scipy.stats import qmc
from tqdm.auto import tqdm

from riemann_ml.core.euler1d import EPSILON, StatePrim
from riemann_ml.fvm import solver as fvm_solver
from riemann_ml.utils.repro import save_config, save_environment, set_global_seeds

__all__ = ["RiemannInitialCondition", "sample_riemann_ic", "solve_and_store"]


@dataclass(frozen=True)
class RiemannInitialCondition:
    """Primitive initial conditions for a 1-D Riemann problem."""

    rho_left: float
    p_left: float
    rho_right: float
    p_right: float
    u_left: float = 0.0
    u_right: float = 0.0

    def to_states(self) -> tuple[StatePrim, StatePrim]:
        """Convert to left/right primitive states."""
        left = StatePrim(
            density=self.rho_left, velocity=self.u_left, pressure=self.p_left
        )
        right = StatePrim(
            density=self.rho_right, velocity=self.u_right, pressure=self.p_right
        )
        return left, right


def _ensure_condition(
    ic: RiemannInitialCondition | Mapping[str, float]
) -> RiemannInitialCondition:
    if isinstance(ic, RiemannInitialCondition):
        return ic
    if isinstance(ic, Mapping):
        return RiemannInitialCondition(**ic)
    msg = "Initial condition must be a RiemannInitialCondition or mapping."
    raise TypeError(msg)


def sample_riemann_ic(
    num_samples: int,
    ranges: Mapping[str, tuple[float, float]],
    seed: Optional[int] = None,
    method: str = "lhs",
) -> list[RiemannInitialCondition]:
    """Sample Riemann initial conditions."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    keys = ("rho_left", "p_left", "rho_right", "p_right")
    defaults = {
        "rho_left": (0.8, 1.2),
        "p_left": (0.8, 1.2),
        "rho_right": (0.1, 0.5),
        "p_right": (0.05, 0.5),
    }

    bounds = []
    for key in keys:
        lower, upper = ranges.get(key, defaults[key])
        if upper <= lower:
            raise ValueError(f"Invalid range for {key}.")
        bounds.append((lower, upper))
    bounds_arr = np.asarray(bounds, dtype=np.float64)

    if method.lower() == "lhs":
        sampler = qmc.LatinHypercube(d=len(keys), seed=seed)
        samples_unit = sampler.random(num_samples)
    else:
        rng = np.random.default_rng(seed)
        samples_unit = rng.random((num_samples, len(keys)))

    lower = bounds_arr[:, 0]
    upper = bounds_arr[:, 1]
    samples = qmc.scale(samples_unit, lower, upper)

    ic_list = [
        RiemannInitialCondition(
            rho_left=float(sample[0]),
            p_left=float(sample[1]),
            rho_right=float(sample[2]),
            p_right=float(sample[3]),
        )
        for sample in samples
    ]
    return ic_list


def _conservative_to_primitive(
    q: np.ndarray, gamma: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.clip(q[:, 0], EPSILON, None)
    momentum = q[:, 1]
    energy = np.clip(q[:, 2], EPSILON, None)
    velocity = momentum / np.clip(rho, EPSILON, None)
    pressure = (gamma - 1.0) * np.maximum(
        energy - 0.5 * momentum**2 / np.clip(rho, EPSILON, None), EPSILON
    )
    return rho, velocity, pressure


def solve_and_store(
    ic_batch: Sequence[RiemannInitialCondition | Mapping[str, float]],
    num_cells: int,
    cfl: float,
    final_time: float,
    out_path: Path | str,
    x_grid: Optional[Sequence[float]] = None,
    gamma: float = 1.4,
    interface_position: float = 0.5,
    show_progress: bool = False,
) -> Path:
    """Solve a batch of Riemann problems and store results in an HDF5 file."""
    if num_cells <= 0:
        raise ValueError("num_cells must be positive.")
    if final_time <= 0.0:
        raise ValueError("final_time must be positive.")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_samples = len(ic_batch)
    if num_samples == 0:
        raise ValueError("ic_batch must contain at least one initial condition.")

    rho_data = np.empty((num_samples, num_cells), dtype=np.float32)
    vel_data = np.empty_like(rho_data)
    pres_data = np.empty_like(rho_data)

    rho_left_list = np.empty(num_samples, dtype=np.float32)
    p_left_list = np.empty(num_samples, dtype=np.float32)
    rho_right_list = np.empty(num_samples, dtype=np.float32)
    p_right_list = np.empty(num_samples, dtype=np.float32)

    x_reference: Optional[np.ndarray] = None

    iterator: Iterable[tuple[int, RiemannInitialCondition | Mapping[str, float]]] = (
        enumerate(ic_batch)
    )
    if show_progress:
        iterator = enumerate(tqdm(ic_batch, desc="Generating dataset"))

    for idx, ic in iterator:
        cond = _ensure_condition(ic)
        left_state, right_state = cond.to_states()

        times, x, q, _ = fvm_solver.simulate(
            num_cells=num_cells,
            final_time=final_time,
            cfl=cfl,
            left_state=left_state,
            right_state=right_state,
            gamma=gamma,
            interface_position=interface_position,
            store_history=False,
        )

        rho, vel, pres = _conservative_to_primitive(q, gamma=gamma)
        rho_data[idx] = rho.astype(np.float32)
        vel_data[idx] = vel.astype(np.float32)
        pres_data[idx] = pres.astype(np.float32)

        rho_left_list[idx] = cond.rho_left
        p_left_list[idx] = cond.p_left
        rho_right_list[idx] = cond.rho_right
        p_right_list[idx] = cond.p_right

        if x_reference is None:
            x_reference = x.astype(np.float32)
            if x_grid is not None:
                x_grid_arr = np.asarray(x_grid, dtype=np.float32)
                if x_grid_arr.shape != x_reference.shape or not np.allclose(
                    x_grid_arr, x_reference
                ):
                    raise ValueError("Provided x_grid does not match solver grid.")
        else:
            if not np.allclose(x_reference, x):
                raise RuntimeError("Solver returned inconsistent spatial grids.")

    if x_reference is None:
        raise RuntimeError("No simulations were executed.")

    with h5py.File(out_path, "w") as h5f:
        h5f.attrs["gamma"] = gamma
        h5f.attrs["cfl"] = cfl
        h5f.attrs["final_time"] = final_time
        h5f.attrs["interface_position"] = interface_position
        h5f.attrs["num_samples"] = num_samples
        h5f.attrs["num_cells"] = num_cells

        h5f.create_dataset("x", data=x_reference, compression="gzip")
        h5f.create_dataset("rho", data=rho_data, compression="gzip")
        h5f.create_dataset("velocity", data=vel_data, compression="gzip")
        h5f.create_dataset("pressure", data=pres_data, compression="gzip")

        init_group = h5f.create_group("initial_conditions")
        init_group.create_dataset("rho_left", data=rho_left_list, compression="gzip")
        init_group.create_dataset("p_left", data=p_left_list, compression="gzip")
        init_group.create_dataset("rho_right", data=rho_right_list, compression="gzip")
        init_group.create_dataset("p_right", data=p_right_list, compression="gzip")

    return out_path


def main(
    num_samples: int = typer.Option(2000, help="Number of Riemann problems to sample."),
    out_path: Path = typer.Option(
        Path("data/processed/sod_like.h5"), help="Output dataset path."
    ),
    cells: int = typer.Option(512, help="Number of spatial cells."),
    cfl: float = typer.Option(0.5, help="CFL number."),
    final_time: float = typer.Option(0.2, help="Final simulation time."),
    gamma: float = typer.Option(1.4, help="Ratio of specific heats."),
    seed: Optional[int] = typer.Option(42, help="Random seed."),
    rho_left_min: float = typer.Option(0.8, help="Lower bound for left density."),
    rho_left_max: float = typer.Option(1.2, help="Upper bound for left density."),
    p_left_min: float = typer.Option(0.8, help="Lower bound for left pressure."),
    p_left_max: float = typer.Option(1.2, help="Upper bound for left pressure."),
    rho_right_min: float = typer.Option(0.05, help="Lower bound for right density."),
    rho_right_max: float = typer.Option(0.4, help="Upper bound for right density."),
    p_right_min: float = typer.Option(0.05, help="Lower bound for right pressure."),
    p_right_max: float = typer.Option(0.4, help="Upper bound for right pressure."),
) -> None:
    """Command-line entry point for dataset generation."""
    ranges = {
        "rho_left": (rho_left_min, rho_left_max),
        "p_left": (p_left_min, p_left_max),
        "rho_right": (rho_right_min, rho_right_max),
        "p_right": (p_right_min, p_right_max),
    }
    set_global_seeds(seed or 0)
    samples = sample_riemann_ic(num_samples, ranges, seed=seed, method="lhs")
    solve_and_store(
        ic_batch=samples,
        num_cells=cells,
        cfl=cfl,
        final_time=final_time,
        out_path=out_path,
        gamma=gamma,
        show_progress=True,
    )
    output_dir = Path(out_path).resolve().parent
    save_environment(output_dir)
    cfg = OmegaConf.create(
        {
            "num_samples": num_samples,
            "cells": cells,
            "cfl": cfl,
            "final_time": final_time,
            "gamma": gamma,
            "ranges": ranges,
            "seed": seed,
            "dataset_path": str(out_path),
        }
    )
    save_config(cfg, output_dir)


if __name__ == "__main__":
    typer.run(main)
