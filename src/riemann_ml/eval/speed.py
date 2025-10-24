"""Utilities to benchmark inference time of different solvers/models."""

from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np
import tensorflow as tf
import torch

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.fvm.solver import simulate
from riemann_ml.ml.pinn.model import PINN, conservative_to_primitive

__all__ = ["time_fvm", "time_pinn", "time_fno", "benchmark_inference"]


def _average_time(fn: Callable[[], None], repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


def time_fvm(
    left_state: StatePrim,
    right_state: StatePrim,
    num_cells: int,
    final_time: float,
    cfl: float,
    gamma: float = 1.4,
    repeats: int = 3,
) -> float:
    """Benchmark FVM simulation time."""

    def runner() -> None:
        simulate(
            num_cells=num_cells,
            final_time=final_time,
            cfl=cfl,
            left_state=left_state,
            right_state=right_state,
            gamma=gamma,
            store_history=False,
        )

    return _average_time(runner, repeats=repeats)


def time_pinn(
    model: PINN,
    x: np.ndarray,
    t: np.ndarray,
    gamma: float,
    repeats: int = 20,
) -> float:
    """Benchmark PINN inference time at coordinates ``(x, t)``."""
    x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
    t_tf = tf.convert_to_tensor(t.reshape(-1, 1), dtype=tf.float32)

    def runner() -> None:
        preds = model.predict_conservative(x_tf, t_tf, training=False)
        conservative_to_primitive(preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], gamma)

    return _average_time(runner, repeats=repeats)


def time_fno(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    repeats: int = 20,
) -> float:
    """Benchmark FNO forward pass time."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    inputs = input_tensor.to(device)

    def runner() -> None:
        with torch.no_grad():
            model(inputs)

    return _average_time(runner, repeats=repeats)


def benchmark_inference(
    left_state: StatePrim,
    right_state: StatePrim,
    fvm_kwargs: Dict,
    pinn: Optional[PINN] = None,
    pinn_input: Optional[Dict[str, np.ndarray]] = None,
    fno: Optional[torch.nn.Module] = None,
    fno_input: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Convenience wrapper returning a dictionary of average runtimes."""
    results: Dict[str, float] = {}
    results["fvm"] = time_fvm(left_state, right_state, **fvm_kwargs)
    if pinn is not None and pinn_input is not None:
        results["pinn"] = time_pinn(
            pinn, pinn_input["x"], pinn_input["t"], pinn_input["gamma"]
        )
    if fno is not None and fno_input is not None:
        results["fno"] = time_fno(fno, fno_input)
    return results
