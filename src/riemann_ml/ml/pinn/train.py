"""Training routine for the 1-D Euler PINN."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import typer
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from riemann_ml.core.euler1d import EPSILON, StatePrim
from riemann_ml.exact.sod_exact import sod_exact_profile
from riemann_ml.ml.pinn.model import PINN, conservative_to_primitive
from riemann_ml.utils.repro import save_config, save_environment, set_global_seeds

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
app = typer.Typer(help="Train a PINN for the Sod shock-tube problem.")


def _to_dict(cfg: DictConfig) -> Dict:
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _sample_uniform(
    rng: np.random.Generator, n: int, low: float, high: float
) -> np.ndarray:
    return rng.uniform(low=low, high=high, size=(n, 1)).astype(np.float32)


def _conservative_profile_from_states(
    x: np.ndarray, left: StatePrim, right: StatePrim, interface: float, gamma: float
) -> np.ndarray:
    rho_left = left.density
    rho_right = right.density
    vel_left = left.velocity
    vel_right = right.velocity
    p_left = left.pressure
    p_right = right.pressure

    energy_left = p_left / (gamma - 1.0) + 0.5 * rho_left * vel_left**2
    energy_right = p_right / (gamma - 1.0) + 0.5 * rho_right * vel_right**2

    momentum_left = rho_left * vel_left
    momentum_right = rho_right * vel_right

    mask = x <= interface
    rho = np.where(mask, rho_left, rho_right)
    momentum = np.where(mask, momentum_left, momentum_right)
    energy = np.where(mask, energy_left, energy_right)
    stacked = np.stack([rho, momentum, energy], axis=1).astype(np.float32)
    return stacked.reshape(-1, 3)


def _relative_l2(pred: np.ndarray, target: np.ndarray) -> float:
    denom = np.linalg.norm(target)
    if denom < EPSILON:
        return float(np.linalg.norm(pred - target))
    return float(np.linalg.norm(pred - target) / denom)


def _build_model(cfg: Dict) -> PINN:
    model_cfg = cfg["model"]
    model = PINN(model_cfg)
    dummy = tf.zeros((1, 2), dtype=tf.float32)
    model(dummy)
    return model


def _prepare_states(cfg: Dict) -> Tuple[StatePrim, StatePrim]:
    left_cfg = cfg["sod"]["left"]
    right_cfg = cfg["sod"]["right"]
    left_state = StatePrim(
        density=left_cfg["density"],
        velocity=left_cfg["velocity"],
        pressure=left_cfg["pressure"],
    )
    right_state = StatePrim(
        density=right_cfg["density"],
        velocity=right_cfg["velocity"],
        pressure=right_cfg["pressure"],
    )
    return left_state, right_state


def train(config_name: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg: DictConfig = compose(config_name=config_name)

    cfg_dict = _to_dict(cfg)
    set_global_seeds(int(cfg_dict["seed"]))
    rng = np.random.default_rng(cfg_dict["seed"])

    gamma = cfg_dict["gamma"]
    domain = cfg_dict["domain"]
    x_min, x_max = domain["x_min"], domain["x_max"]
    t_min, t_max = domain["t_min"], domain["t_max"]
    interface = cfg_dict["sod"]["interface"]

    left_state, right_state = _prepare_states(cfg_dict)

    log_dir = Path(cfg_dict["logging"]["log_dir"])
    ckpt_dir = Path(cfg_dict["logging"]["checkpoint_dir"])
    output_dir = Path(cfg_dict["logging"]["output_dir"])
    for directory in (log_dir, ckpt_dir, output_dir):
        directory.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir)
    save_environment(output_dir)

    model = _build_model(cfg_dict)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg_dict["training"]["learning_rate"]
    )
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, directory=str(ckpt_dir), max_to_keep=3)
    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)

    writer = tf.summary.create_file_writer(str(log_dir))

    n_pde = cfg_dict["training"]["n_pde"]
    n_ic = cfg_dict["training"]["n_ic"]
    n_bc = cfg_dict["training"]["n_bc"]
    steps = cfg_dict["training"]["max_steps"]
    log_interval = cfg_dict["training"]["log_interval"]
    ckpt_interval = cfg_dict["training"]["checkpoint_interval"]
    eval_interval = cfg_dict["training"]["eval_interval"]

    left_cons = np.array(
        [
            left_state.density,
            left_state.density * left_state.velocity,
            left_state.pressure / (gamma - 1.0)
            + 0.5 * left_state.density * left_state.velocity**2,
        ],
        dtype=np.float32,
    )
    right_cons = np.array(
        [
            right_state.density,
            right_state.density * right_state.velocity,
            right_state.pressure / (gamma - 1.0)
            + 0.5 * right_state.density * right_state.velocity**2,
        ],
        dtype=np.float32,
    )

    loss_history = []

    @tf.function
    def train_step(pde_x, pde_t, ic_x, ic_t, ic_q, bc_x, bc_t, bc_q):
        with tf.GradientTape() as tape:
            residual = model.compute_residual(
                pde_x, pde_t, gamma=tf.constant(gamma, dtype=tf.float32)
            )
            loss_pde = tf.reduce_mean(tf.square(residual))

            pred_ic = model.predict_conservative(ic_x, ic_t, training=True)
            loss_ic = tf.reduce_mean(tf.square(pred_ic - ic_q))

            pred_bc = model.predict_conservative(bc_x, bc_t, training=True)
            loss_bc = tf.reduce_mean(tf.square(pred_bc - bc_q))

            total_loss = loss_pde + loss_ic + loss_bc

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total_loss, loss_pde, loss_ic, loss_bc

    start_time = time.time()

    for step in range(1, steps + 1):
        pde_x = _sample_uniform(rng, n_pde, x_min, x_max)
        pde_t = _sample_uniform(rng, n_pde, t_min, t_max)

        ic_x_np = _sample_uniform(rng, n_ic, x_min, x_max)
        ic_t_np = np.zeros_like(ic_x_np, dtype=np.float32)
        ic_q_np = _conservative_profile_from_states(
            ic_x_np, left_state, right_state, interface, gamma
        )

        bc_t_np = _sample_uniform(rng, n_bc, t_min, t_max)
        half = n_bc // 2
        bc_left_x = np.full((half, 1), x_min, dtype=np.float32)
        bc_right_x = np.full((n_bc - half, 1), x_max, dtype=np.float32)
        bc_x_np = np.vstack([bc_left_x, bc_right_x])
        bc_q_np = np.vstack(
            [
                np.repeat(left_cons[None, :], half, axis=0),
                np.repeat(right_cons[None, :], n_bc - half, axis=0),
            ]
        )

        tensors = [
            tf.convert_to_tensor(pde_x),
            tf.convert_to_tensor(pde_t),
            tf.convert_to_tensor(ic_x_np),
            tf.convert_to_tensor(ic_t_np),
            tf.convert_to_tensor(ic_q_np),
            tf.convert_to_tensor(bc_x_np),
            tf.convert_to_tensor(bc_t_np),
            tf.convert_to_tensor(bc_q_np),
        ]

        total_loss, loss_pde, loss_ic, loss_bc = train_step(*tensors)

        if step % log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            loss_entry = {
                "step": step,
                "loss_total": float(total_loss.numpy()),
                "loss_pde": float(loss_pde.numpy()),
                "loss_ic": float(loss_ic.numpy()),
                "loss_bc": float(loss_bc.numpy()),
                "elapsed_s": elapsed,
            }
            loss_history.append(loss_entry)
            with writer.as_default():
                tf.summary.scalar("loss/total", total_loss, step=step)
                tf.summary.scalar("loss/pde", loss_pde, step=step)
                tf.summary.scalar("loss/ic", loss_ic, step=step)
                tf.summary.scalar("loss/bc", loss_bc, step=step)

        if step % ckpt_interval == 0:
            manager.save(checkpoint_number=step)

        if step % eval_interval == 0:
            evaluate_and_save(model, cfg_dict, output_dir, step)

    manager.save(checkpoint_number=steps)
    evaluate_and_save(model, cfg_dict, output_dir, steps)

    history_path = output_dir / "loss_history.json"
    with history_path.open("w", encoding="utf-8") as fout:
        json.dump(loss_history, fout, indent=2)


def evaluate_and_save(model: PINN, cfg: Dict, output_dir: Path, step: int) -> None:
    gamma = cfg["gamma"]
    domain = cfg["domain"]
    x_min, x_max = domain["x_min"], domain["x_max"]
    t_final = domain["t_max"]
    interface = cfg["sod"]["interface"]
    left_state, right_state = _prepare_states(cfg)

    n_eval = cfg["evaluation"]["n_points"]
    x_eval = np.linspace(x_min, x_max, n_eval, dtype=np.float32).reshape(-1, 1)
    t_eval = np.full_like(x_eval, t_final, dtype=np.float32)

    preds = model.predict_conservative(
        tf.convert_to_tensor(x_eval),
        tf.convert_to_tensor(t_eval),
        training=False,
    )
    rho_pred, mom_pred, energy_pred = tf.split(preds, 3, axis=1)
    rho_pred_np, vel_pred_np, p_pred_np = conservative_to_primitive(
        rho_pred, mom_pred, energy_pred, gamma
    )

    rho_pred_np = rho_pred_np.numpy().flatten()
    vel_pred_np = vel_pred_np.numpy().flatten()
    p_pred_np = p_pred_np.numpy().flatten()

    rho_exact, u_exact, p_exact = sod_exact_profile(
        x_eval.flatten() - interface,
        t_final,
        left_state=left_state,
        right_state=right_state,
        gamma=gamma,
    )

    rel_rho = _relative_l2(rho_pred_np, rho_exact)
    rel_u = _relative_l2(vel_pred_np, u_exact)
    rel_p = _relative_l2(p_pred_np, p_exact)

    metrics = {
        "step": step,
        "relative_l2_rho": rel_rho,
        "relative_l2_u": rel_u,
        "relative_l2_p": rel_p,
    }

    metrics_path = output_dir / f"metrics_step_{step}.json"
    with metrics_path.open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    axes[0].plot(x_eval, rho_exact, label="Exact")
    axes[0].plot(x_eval, rho_pred_np, "--", label="PINN")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(x_eval, u_exact, label="Exact")
    axes[1].plot(x_eval, vel_pred_np, "--", label="PINN")
    axes[1].set_ylabel("Velocity")
    axes[1].grid(True)

    axes[2].plot(x_eval, p_exact, label="Exact")
    axes[2].plot(x_eval, p_pred_np, "--", label="PINN")
    axes[2].set_ylabel("Pressure")
    axes[2].set_xlabel("x")
    axes[2].grid(True)

    fig.suptitle(f"PINN vs Exact at t={t_final:.3f} (step {step})")
    fig.tight_layout()
    fig.savefig(output_dir / f"pinn_vs_exact_step_{step}.png", dpi=150)
    plt.close(fig)


@app.command()
def main(
    config: str = typer.Option(
        "pinn", "--config", "-c", help="Configuration name located in configs/."
    )
) -> None:
    train(config_name=config)


if __name__ == "__main__":
    app()
