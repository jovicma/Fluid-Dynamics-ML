"""Training script for 1-D Fourier Neural Operator on Riemann datasets."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from riemann_ml.ml.fno.dataset import RiemannH5Dataset, _read_metadata
from riemann_ml.ml.fno.model import FNO1DModel

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _to_dict(cfg: DictConfig) -> Dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _relative_l2(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    diff = pred - target
    numerator = torch.linalg.vector_norm(diff.reshape(diff.shape[0], -1), dim=1)
    denominator = torch.linalg.vector_norm(target.reshape(target.shape[0], -1), dim=1)
    return torch.mean(numerator / (denominator + eps))


def _prepare_dataloaders(cfg: Dict, metadata) -> Tuple[DataLoader, DataLoader, np.ndarray, Path]:
    data_cfg = cfg["data"]
    path = Path(data_cfg["path"])
    val_fraction = float(data_cfg.get("val_fraction", 0.1))
    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(data_cfg.get("num_workers", 0))

    base_dataset = RiemannH5Dataset(path, add_coordinates=data_cfg.get("add_coordinates", True))
    all_indices = np.arange(len(base_dataset))
    rng = np.random.default_rng(cfg["seed"])
    rng.shuffle(all_indices)
    val_size = max(1, int(math.ceil(len(base_dataset) * val_fraction)))
    val_indices = all_indices[:val_size]
    train_indices = all_indices[val_size:]
    if len(train_indices) == 0:
        raise RuntimeError("Validation fraction too large; no training samples remaining.")

    train_dataset = RiemannH5Dataset(path, indices=train_indices, add_coordinates=data_cfg.get("add_coordinates", True))
    val_dataset = RiemannH5Dataset(path, indices=val_indices, add_coordinates=data_cfg.get("add_coordinates", True))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, val_indices, path


def _ensure_dirs(cfg: Dict) -> Tuple[Path, Path, Path]:
    log_dir = Path(cfg["logging"]["log_dir"])
    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    output_dir = Path(cfg["logging"]["output_dir"])
    for directory in (log_dir, ckpt_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return log_dir, ckpt_dir, output_dir


def evaluate(model: torch.nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs)
            loss = _relative_l2(preds, targets)
            losses.append(float(loss.item()))
    if not losses:
        return float("nan")
    return float(np.mean(losses))


def _save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int, path: Path) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def _plot_samples(
    model: torch.nn.Module,
    dataset: RiemannH5Dataset,
    sample_indices: np.ndarray,
    metadata,
    device: torch.device,
    output_dir: Path,
    epoch: int,
) -> None:
    model.eval()
    x_axis = metadata.x
    for i, idx in enumerate(sample_indices):
        inputs, targets = dataset[idx]
        inputs = inputs.unsqueeze(0).to(device)
        with torch.no_grad():
            preds = model(inputs).cpu().numpy()[0]
        targets_np = targets.numpy()

        rho_pred, u_pred, p_pred = preds
        rho_true, u_true, p_true = targets_np

        fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
        axes[0].plot(x_axis, rho_true, label="True")
        axes[0].plot(x_axis, rho_pred, "--", label="Pred")
        axes[0].set_ylabel("Density")
        axes[0].grid(True)

        axes[1].plot(x_axis, u_true, label="True")
        axes[1].plot(x_axis, u_pred, "--", label="Pred")
        axes[1].set_ylabel("Velocity")
        axes[1].grid(True)

        axes[2].plot(x_axis, p_true, label="True")
        axes[2].plot(x_axis, p_pred, "--", label="Pred")
        axes[2].set_ylabel("Pressure")
        axes[2].set_xlabel("x")
        axes[2].grid(True)

        axes[0].legend()
        fig.suptitle(f"FNO predictions (epoch {epoch}, sample {idx})")
        fig.tight_layout()
        fig.savefig(output_dir / f"fno_sample_{i}_epoch_{epoch}.png", dpi=150)
        plt.close(fig)


def train(config_name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg: DictConfig = compose(config_name=config_name)

    cfg_dict = _to_dict(cfg)
    _set_seed(cfg_dict["seed"])

    metadata = _read_metadata(Path(cfg_dict["data"]["path"]))
    train_loader, val_loader, val_indices, dataset_path = _prepare_dataloaders(cfg_dict, metadata)
    log_dir, ckpt_dir, output_dir = _ensure_dirs(cfg_dict)

    device = torch.device(cfg_dict["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    model = FNO1DModel(cfg_dict["model"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg_dict["training"]["learning_rate"],
        weight_decay=cfg_dict["training"].get("weight_decay", 0.0),
    )

    writer = SummaryWriter(log_dir=str(log_dir))
    epochs = int(cfg_dict["training"]["epochs"])
    log_interval = int(cfg_dict["training"].get("log_interval", 1))
    checkpoint_interval = int(cfg_dict["training"].get("checkpoint_interval", 5))
    eval_interval = int(cfg_dict["training"].get("eval_interval", 5))

    loss_history = []
    train_dataset_full = RiemannH5Dataset(dataset_path, add_coordinates=cfg_dict["data"].get("add_coordinates", True))

    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_losses = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = _relative_l2(preds, targets)
            loss.backward()
            optimizer.step()
            running_losses.append(float(loss.item()))

        train_loss = float(np.mean(running_losses)) if running_losses else float("nan")
        val_loss = evaluate(model, val_loader, device)

        loss_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "elapsed_s": time.time() - start_time,
            }
        )

        if epoch % log_interval == 0 or epoch == 1:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)

        if epoch % checkpoint_interval == 0 or epoch == epochs:
            ckpt_path = ckpt_dir / f"fno_epoch_{epoch}.pt"
            _save_checkpoint(model, optimizer, epoch, ckpt_path)

        if epoch % eval_interval == 0 or epoch == epochs:
            num_samples = int(cfg_dict["evaluation"].get("num_samples", 3))
            chosen_indices = val_indices[: num_samples]
            _plot_samples(model, train_dataset_full, chosen_indices, metadata, device, output_dir, epoch)
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "num_parameters": model.num_parameters(),
            }
            with (output_dir / f"metrics_epoch_{epoch}.json").open("w", encoding="utf-8") as fout:
                json.dump(metrics, fout, indent=2)

    writer.close()
    with (output_dir / "loss_history.json").open("w", encoding="utf-8") as fout:
        json.dump(loss_history, fout, indent=2)


def main(config: str = "fno") -> None:
    train(config_name=config)


if __name__ == "__main__":
    main()
