"""Evaluation metrics for 1-D Riemann predictions."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def relative_l2(pred: Sequence[float], target: Sequence[float], eps: float = 1e-12) -> float:
    """Compute relative L2 error between prediction and target."""
    pred_arr = np.asarray(pred, dtype=np.float64)
    target_arr = np.asarray(target, dtype=np.float64)
    num = np.linalg.norm(pred_arr - target_arr)
    denom = np.linalg.norm(target_arr)
    return float(num / (denom + eps))


def _feature_location(x: np.ndarray, values: np.ndarray, ignore_index: int | None = None) -> float:
    grad = np.abs(np.gradient(values, x))
    if ignore_index is not None:
        grad = grad.copy()
        grad[ignore_index] = -np.inf
    idx = int(np.argmax(grad))
    return float(x[idx]), idx


def shock_location_error(
    x: Sequence[float],
    pred_density: Sequence[float],
    target_density: Sequence[float],
) -> float:
    """Estimate shock position via maximum density gradient and compare."""
    x_arr = np.asarray(x, dtype=np.float64)
    pred = np.asarray(pred_density, dtype=np.float64)
    target = np.asarray(target_density, dtype=np.float64)
    loc_pred, _ = _feature_location(x_arr, pred)
    loc_target, _ = _feature_location(x_arr, target)
    return abs(loc_pred - loc_target)


def contact_plateau_error(
    x: Sequence[float],
    pred_density: Sequence[float],
    target_density: Sequence[float],
    window: int = 5,
) -> float:
    """Compare plateau density around contact discontinuity."""
    x_arr = np.asarray(x, dtype=np.float64)
    pred = np.asarray(pred_density, dtype=np.float64)
    target = np.asarray(target_density, dtype=np.float64)

    _, shock_idx = _feature_location(x_arr, target)
    _, contact_idx = _feature_location(x_arr, target, ignore_index=shock_idx)

    def window_mean(arr: np.ndarray, center: int) -> float:
        start = max(0, center - window)
        end = min(arr.shape[0], center + window + 1)
        return float(np.mean(arr[start:end]))

    pred_mean = window_mean(pred, contact_idx)
    target_mean = window_mean(target, contact_idx)
    return abs(pred_mean - target_mean)


__all__ = ["relative_l2", "shock_location_error", "contact_plateau_error"]
