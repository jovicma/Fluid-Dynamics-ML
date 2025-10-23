#!/usr/bin/env bash
set -euo pipefail

N=400
CFL=0.5
TF=0.2
OUT_DIR="data/artifacts/fvm_sod"
PLOT_PATH="${OUT_DIR}/profiles.png"
DATA_PATH="${OUT_DIR}/solution.npz"

mkdir -p "${OUT_DIR}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
export N CFL TF OUT_DIR PLOT_PATH DATA_PATH

python - <<'PY'
import os
from pathlib import Path

import numpy as np

from riemann_ml.core.euler1d import StatePrim
from riemann_ml.fvm.solver import simulate
from riemann_ml.utils.plotting import plot_profiles

N = int(os.environ["N"])
CFL = float(os.environ["CFL"])
TF = float(os.environ["TF"])
OUT_DIR = Path(os.environ["OUT_DIR"])
PLOT_PATH = Path(os.environ["PLOT_PATH"])
DATA_PATH = Path(os.environ["DATA_PATH"])
GAMMA = 1.4

left_state = StatePrim(density=1.0, velocity=0.0, pressure=1.0)
right_state = StatePrim(density=0.125, velocity=0.0, pressure=0.1)

times, x, q, _ = simulate(
    num_cells=N,
    final_time=TF,
    cfl=CFL,
    left_state=left_state,
    right_state=right_state,
    gamma=GAMMA,
    store_history=False,
)

rho = q[:, 0]
momentum = q[:, 1]
energy = q[:, 2]
velocity = momentum / np.clip(rho, 1e-12, None)
pressure = (GAMMA - 1.0) * np.maximum(energy - 0.5 * momentum**2 / np.clip(rho, 1e-12, None), 1e-12)

plot_profiles(
    x=x,
    density=rho,
    velocity=velocity,
    pressure=pressure,
    title=f"Sod shock tube - t = {times[-1]:.3f}",
    savepath=PLOT_PATH,
)

np.savez(
    DATA_PATH,
    times=times,
    x=x,
    q=q,
    gamma=GAMMA,
)

print(f"[run_fvm_sod] Saved figure to {PLOT_PATH}")
print(f"[run_fvm_sod] Saved solution data to {DATA_PATH}")
PY
