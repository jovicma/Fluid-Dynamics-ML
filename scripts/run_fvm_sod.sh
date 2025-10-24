#!/usr/bin/env bash
set -euo pipefail

N=400
CFL=0.5
TF=0.2
OUT_DIR="data/artifacts/fvm_sod"
PLOT_PATH="${OUT_DIR}/profiles.png"
DATA_PATH="${OUT_DIR}/solution.npz"

mkdir -p "${OUT_DIR}"
export N CFL TF OUT_DIR PLOT_PATH DATA_PATH

PYTHON_BIN="${PYTHON_BIN:-python}"
if [ -d ".venv" ]; then
  if [ -x ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
  elif [ -x ".venv/Scripts/python.exe" ]; then
    PYTHON_BIN=".venv/Scripts/python.exe"
  fi
fi

"${PYTHON_BIN}" - <<'PY'
import os
import sys
from pathlib import Path

import numpy as np

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from riemann_ml.core.euler1d import StatePrim  # noqa: E402
from riemann_ml.fvm.solver import simulate  # noqa: E402
from riemann_ml.utils.plotting import plot_profiles  # noqa: E402
from riemann_ml.exact.sod_exact import sod_exact_profile  # noqa: E402

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

rho_exact, u_exact, p_exact = sod_exact_profile(
    x - 0.5,
    TF,
    left_state=left_state,
    right_state=right_state,
    gamma=GAMMA,
)

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

comparison_path = OUT_DIR / "fvm_vs_exact.png"
plt.tight_layout()
plt.savefig(comparison_path, dpi=150)
plt.close()

np.savez(
    DATA_PATH,
    times=times,
    x=x,
    q=q,
    gamma=GAMMA,
)

print(f"[run_fvm_sod] Saved figure to {PLOT_PATH}")
print(f"[run_fvm_sod] Saved comparison plot to {comparison_path}")
print(f"[run_fvm_sod] Saved solution data to {DATA_PATH}")
PY
