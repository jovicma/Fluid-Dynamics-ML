"""Dataset utilities for training 1-D FNO models on Riemann datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetMetadata:
    num_samples: int
    num_cells: int
    gamma: float
    interface_position: float
    x: np.ndarray


def _read_metadata(path: Path) -> DatasetMetadata:
    with h5py.File(path, "r") as h5f:
        num_samples = h5f["rho"].shape[0]
        num_cells = h5f["rho"].shape[1]
        gamma = float(h5f.attrs.get("gamma", 1.4))
        interface_position = float(h5f.attrs.get("interface_position", 0.5))
        x = h5f["x"][:].astype(np.float32)
    return DatasetMetadata(
        num_samples=num_samples,
        num_cells=num_cells,
        gamma=gamma,
        interface_position=interface_position,
        x=x,
    )


class RiemannH5Dataset(Dataset):
    """Torch dataset reading Riemann solutions stored in HDF5."""

    def __init__(
        self,
        path: Path | str,
        indices: Optional[Iterable[int]] = None,
        add_coordinates: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.path}")

        meta = _read_metadata(self.path)
        self.metadata = meta

        if indices is None:
            self.indices = np.arange(meta.num_samples, dtype=np.int64)
        else:
            self.indices = np.array(list(indices), dtype=np.int64)
        self.add_coordinates = add_coordinates

    def __len__(self) -> int:
        return int(self.indices.size)

    def _initial_state(
        self, rho_left: float, p_left: float, rho_right: float, p_right: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mask = self.metadata.x <= self.metadata.interface_position
        rho_init = np.where(mask, rho_left, rho_right).astype(np.float32)
        vel_init = np.zeros_like(rho_init, dtype=np.float32)
        p_init = np.where(mask, p_left, p_right).astype(np.float32)
        return rho_init, vel_init, p_init

    def __getitem__(self, item: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = int(self.indices[item])
        with h5py.File(self.path, "r") as h5f:
            rho = h5f["rho"][idx].astype(np.float32)
            velocity = h5f["velocity"][idx].astype(np.float32)
            pressure = h5f["pressure"][idx].astype(np.float32)

            ic_group = h5f["initial_conditions"]
            rho_left = float(ic_group["rho_left"][idx])
            p_left = float(ic_group["p_left"][idx])
            rho_right = float(ic_group["rho_right"][idx])
            p_right = float(ic_group["p_right"][idx])

        rho_init, vel_init, p_init = self._initial_state(
            rho_left, p_left, rho_right, p_right
        )

        inputs = [rho_init, vel_init, p_init]
        if self.add_coordinates:
            inputs.append(self.metadata.x)

        input_tensor = torch.from_numpy(np.stack(inputs, axis=0))
        target_tensor = torch.from_numpy(np.stack([rho, velocity, pressure], axis=0))
        return input_tensor, target_tensor


__all__ = ["RiemannH5Dataset", "DatasetMetadata", "_read_metadata"]
