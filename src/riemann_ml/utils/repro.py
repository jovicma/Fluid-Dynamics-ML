"""Utilities for reproducibility and environment tracking."""

from __future__ import annotations

import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Set seeds for python, numpy, TensorFlow, and PyTorch (if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # pragma: no cover
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover
        tf = None
    if tf is not None:
        tf.random.set_seed(seed)


def save_environment(output_dir: Path) -> None:
    """Persist pip freeze output to ENVIRONMENT.txt in the target directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = output_dir / "ENVIRONMENT.txt"
    with env_path.open("w", encoding="utf-8") as fh:
        subprocess.run(
            ["pip", "freeze"],
            stdout=fh,
            check=False,
            text=True,
        )


def save_config(
    config: Any, output_dir: Path, filename: str = "configs_used.yaml"
) -> None:
    """Serialize a Hydra/OmegaConf config to the given output directory."""
    from omegaconf import OmegaConf

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = output_dir / filename
    OmegaConf.save(config=config, f=cfg_path)


__all__ = ["set_global_seeds", "save_environment", "save_config"]
