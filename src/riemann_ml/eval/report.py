"""Unified evaluation and reporting for riemann-ml models."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import h5py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from omegaconf import OmegaConf

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.eval.metrics import contact_plateau_error, relative_l2, shock_location_error
from riemann_ml.exact.sod_exact import sod_exact_profile
from riemann_ml.fvm.solver import simulate
from riemann_ml.ml.fno.dataset import RiemannH5Dataset, _read_metadata
from riemann_ml.ml.fno.model import FNO1DModel
from riemann_ml.ml.pinn.model import PINN, conservative_to_primitive as tf_cons_to_prim

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load_config(name: str) -> Dict:
    cfg = OmegaConf.load(CONFIG_DIR / f"{name}.yaml")
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _latest_checkpoint(directory: Path, pattern: str) -> Path:
    candidates = sorted(directory.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints matching {pattern} in {directory}")
    return candidates[-1]


def _load_pinn(cfg: Dict, checkpoint_dir: Path) -> PINN:
    model = PINN(cfg["model"])
    model(tf.zeros((1, 2), dtype=tf.float32))
    latest = tf.train.latest_checkpoint(str(checkpoint_dir))
    if latest is None:
        raise FileNotFoundError(f"No TensorFlow checkpoints in {checkpoint_dir}")
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(latest).expect_partial()
    return model


def _load_fno(cfg: Dict, checkpoint_path: Path, device: torch.device) -> FNO1DModel:
    model = FNO1DModel(cfg["model"])
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = state.get("model_state", state)
    state_dict.pop("_metadata", None)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _pinn_predict(model: PINN, x: np.ndarray, t: float, gamma: float) -> Dict[str, np.ndarray]:
    x_tf = tf.convert_to_tensor(x.reshape(-1, 1), dtype=tf.float32)
    t_tf = tf.convert_to_tensor(np.full_like(x_tf.numpy(), t), dtype=tf.float32)
    preds = model.predict_conservative(x_tf, t_tf, training=False)
    rho, u, p = tf_cons_to_prim(preds[:, 0:1], preds[:, 1:2], preds[:, 2:3], gamma)
    return {
        "rho": rho.numpy().flatten(),
        "u": u.numpy().flatten(),
        "p": p.numpy().flatten(),
    }


def _fvm_predict(
    left: StatePrim,
    right: StatePrim,
    num_cells: int,
    final_time: float,
    cfl: float,
    gamma: float,
) -> Dict[str, np.ndarray]:
    _, x, q, _ = simulate(
        num_cells=num_cells,
        final_time=final_time,
        cfl=cfl,
        left_state=left,
        right_state=right,
        gamma=gamma,
        store_history=False,
    )
    rho = np.clip(q[:, 0], 1e-12, None)
    momentum = q[:, 1]
    energy = np.clip(q[:, 2], 1e-12, None)
    velocity = momentum / rho
    pressure = (gamma - 1.0) * np.maximum(energy - 0.5 * rho * velocity**2, 1e-12)
    return {
        "rho": rho,
        "u": velocity,
        "p": pressure,
        "x": x,
    }


def _fno_predict(model: FNO1DModel, inp: torch.Tensor, device: torch.device) -> Dict[str, np.ndarray]:
    inp = inp.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(inp).cpu().numpy()[0]
    return {
        "rho": pred[0],
        "u": pred[1],
        "p": pred[2],
    }


def initial_profile(metadata, rho_left: float, p_left: float, rho_right: float, p_right: float) -> Dict[str, np.ndarray]:
    mask = metadata.x <= metadata.interface_position
    rho = np.where(mask, rho_left, rho_right).astype(np.float32)
    velocity = np.zeros_like(rho, dtype=np.float32)
    pressure = np.where(mask, p_left, p_right).astype(np.float32)
    return {"rho": rho, "u": velocity, "p": pressure}


def to_fno_input(initial: Dict[str, np.ndarray], metadata) -> torch.Tensor:
    channels = [
        initial["rho"],
        initial["u"],
        initial["p"],
        metadata.x,
    ]
    arr = np.stack(channels, axis=0).astype(np.float32)
    return torch.from_numpy(arr)


def compute_metrics(x: np.ndarray, reference: Dict[str, np.ndarray], prediction: Dict[str, np.ndarray]) -> Dict[str, float]:
    metrics = OrderedDict()
    metrics["relative_l2_rho"] = relative_l2(prediction["rho"], reference["rho"])
    metrics["relative_l2_u"] = relative_l2(prediction["u"], reference["u"])
    metrics["relative_l2_p"] = relative_l2(prediction["p"], reference["p"])
    metrics["shock_location_error"] = shock_location_error(x, prediction["rho"], reference["rho"])
    metrics["contact_plateau_error"] = contact_plateau_error(x, prediction["rho"], reference["rho"])
    return metrics


def plot_comparison(x: np.ndarray, reference: Dict[str, np.ndarray], predictions: Mapping[str, Dict[str, np.ndarray]], title: str, path: Path) -> None:
    fields = [("rho", "Density"), ("u", "Velocity"), ("p", "Pressure")]
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    for ax, (field_key, field_label) in zip(axes, fields):
        ax.plot(x, reference[field_key], label="Reference", linewidth=2)
        for name, pred in predictions.items():
            ax.plot(x, pred[field_key], linestyle="--", label=name)
        ax.set_ylabel(field_label)
        ax.grid(True)
    axes[-1].set_xlabel("x")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_sod(
    cfg_pinn: Dict,
    cfg_fno: Dict,
    pinn_model: PINN,
    fno_model: FNO1DModel,
    metadata,
    output_dir: Path,
) -> Dict[str, Dict[str, float]]:
    gamma = cfg_pinn["gamma"]
    final_time = cfg_pinn["domain"]["t_max"]
    cfl = 0.5
    left_cfg = cfg_pinn["sod"]["left"]
    right_cfg = cfg_pinn["sod"]["right"]
    left_state = StatePrim(**left_cfg)
    right_state = StatePrim(**right_cfg)

    x = metadata.x
    rho_exact, u_exact, p_exact = sod_exact_profile(
        x - cfg_pinn["sod"]["interface"],
        final_time,
        left_state=left_state,
        right_state=right_state,
        gamma=gamma,
    )
    reference = {"rho": rho_exact, "u": u_exact, "p": p_exact}

    fvm = _fvm_predict(left_state, right_state, metadata.num_cells, final_time, cfl=cfl, gamma=gamma)
    pinn = _pinn_predict(pinn_model, x, final_time, gamma=gamma)
    init_profile = initial_profile(metadata, left_cfg["density"], left_cfg["pressure"], right_cfg["density"], right_cfg["pressure"])
    fno_input = to_fno_input(init_profile, metadata)
    device = next(fno_model.parameters()).device
    fno_pred = _fno_predict(fno_model, fno_input, device)

    predictions = OrderedDict(
        [
            ("FVM", {"rho": fvm["rho"], "u": fvm["u"], "p": fvm["p"]}),
            ("PINN", pinn),
            ("FNO", fno_pred),
        ]
    )

    metrics = {name: compute_metrics(x, reference, pred) for name, pred in predictions.items()}
    plot_comparison(x, reference, predictions, "Sod shock tube comparison", output_dir / "sod_comparison.png")
    with (output_dir / "sod_metrics.json").open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2)
    return metrics


def evaluate_random_dataset_samples(
    dataset: RiemannH5Dataset,
    cfg_pinn: Dict,
    pinn_model: PINN,
    fno_model: FNO1DModel,
    metadata,
    output_dir: Path,
    num_samples: int = 5,
) -> List[Dict[str, object]]:
    rng = np.random.default_rng(cfg_pinn["seed"])
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    device = next(fno_model.parameters()).device
    final_time = cfg_pinn["domain"]["t_max"]
    gamma = cfg_pinn["gamma"]
    results: List[Dict[str, object]] = []

    with h5py.File(str(dataset.path), "r") as h5f:  # type: ignore[arg-type]
        rho_left_all = h5f["initial_conditions"]["rho_left"][:]
        p_left_all = h5f["initial_conditions"]["p_left"][:]
        rho_right_all = h5f["initial_conditions"]["rho_right"][:]
        p_right_all = h5f["initial_conditions"]["p_right"][:]

    for sample_id, idx in enumerate(indices):
        inputs, target = dataset[idx]
        reference = {"rho": target.numpy()[0], "u": target.numpy()[1], "p": target.numpy()[2]}
        x = metadata.x

        rho_left = float(rho_left_all[idx])
        p_left = float(p_left_all[idx])
        rho_right = float(rho_right_all[idx])
        p_right = float(p_right_all[idx])

        left_state = StatePrim(density=rho_left, velocity=0.0, pressure=p_left)
        right_state = StatePrim(density=rho_right, velocity=0.0, pressure=p_right)
        fvm_pred = _fvm_predict(left_state, right_state, metadata.num_cells, final_time, cfl=0.5, gamma=gamma)

        pinn_pred = _pinn_predict(pinn_model, x, final_time, gamma=gamma)
        fno_pred = _fno_predict(fno_model, inputs, device)

        predictions = OrderedDict(
            [
                ("FVM", {"rho": fvm_pred["rho"], "u": fvm_pred["u"], "p": fvm_pred["p"]}),
                ("PINN", pinn_pred),
                ("FNO", fno_pred),
            ]
        )

        metrics = {name: compute_metrics(x, reference, pred) for name, pred in predictions.items()}
        results.append({"index": int(idx), "metrics": metrics})

        plot_comparison(
            x,
            reference,
            predictions,
            f"Dataset sample {idx}",
            output_dir / f"dataset_sample_{sample_id}.png",
        )

    with (output_dir / "dataset_metrics.json").open("w", encoding="utf-8") as fout:
        json.dump(results, fout, indent=2)
    return results


def main(
    dataset_path: Path = Path("data/processed/sod_like.h5"),
    pinn_checkpoint_dir: Path = Path("data/artifacts/pinn/checkpoints"),
    fno_checkpoint_path: Optional[Path] = None,
    output_dir: Path = Path("data/artifacts/eval"),
    num_random: int = 5,
) -> None:
    cfg_pinn = _load_config("pinn")
    cfg_fno = _load_config("fno")

    metadata = _read_metadata(dataset_path)
    dataset = RiemannH5Dataset(dataset_path)

    pinn_model = _load_pinn(cfg_pinn, pinn_checkpoint_dir)
    if fno_checkpoint_path is None:
        fno_checkpoint_path = _latest_checkpoint(Path(cfg_fno["logging"]["checkpoint_dir"]), "fno_epoch_*.pt")
    device = torch.device(cfg_fno["training"].get("device", "cpu"))
    fno_model = _load_fno(cfg_fno, fno_checkpoint_path, device)

    out_dir = _ensure_dir(output_dir)
    sod_dir = _ensure_dir(out_dir / "sod")
    dataset_dir = _ensure_dir(out_dir / "dataset")

    evaluate_sod(cfg_pinn, cfg_fno, pinn_model, fno_model, metadata, sod_dir)
    evaluate_random_dataset_samples(dataset, cfg_pinn, pinn_model, fno_model, metadata, dataset_dir, num_samples=num_random)


if __name__ == "__main__":
    main()
