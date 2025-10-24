#!/usr/bin/env bash
set -euo pipefail

DEMO_DIR="data/artifacts/demo_sod"
PINN_CONFIG="src/riemann_ml/configs/pinn_demo.yaml"
FNO_CONFIG="src/riemann_ml/configs/fno_demo.yaml"
DATASET_PATH="data/processed/demo_sod_small.h5"

mkdir -p "${DEMO_DIR}/fvm" "${DEMO_DIR}/pinn" "${DEMO_DIR}/fno"

# 1) FVM simulation and plots
riemann-ml plot-sod --output-dir "${DEMO_DIR}/fvm"

# 2) Short PINN training
python - "$PINN_CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

target = Path(sys.argv[1])
cfg = OmegaConf.load(Path("src/riemann_ml/configs/pinn.yaml"))
cfg.training.max_steps = 500
cfg.logging.log_dir = "data/artifacts/demo_sod/pinn/logs"
cfg.logging.checkpoint_dir = "data/artifacts/demo_sod/pinn/checkpoints"
cfg.logging.output_dir = "data/artifacts/demo_sod/pinn/outputs"
target.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
python -m riemann_ml.ml.pinn.train --config pinn_demo

# 3) Dataset and short FNO training
NUM_SAMPLES=200 CELLS=256 TF=0.2 CFL=0.5 OUT_PATH="${DATASET_PATH}" ./scripts/gen_dataset.sh
python - "$FNO_CONFIG" "$DATASET_PATH" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

target = Path(sys.argv[1])
dataset_path = sys.argv[2]
cfg = OmegaConf.load(Path("src/riemann_ml/configs/fno.yaml"))
cfg.training.epochs = 5
cfg.training.device = "cpu"
cfg.training.log_interval = 1
cfg.training.checkpoint_interval = 5
cfg.training.eval_interval = 5
cfg.logging.log_dir = "data/artifacts/demo_sod/fno/logs"
cfg.logging.checkpoint_dir = "data/artifacts/demo_sod/fno/checkpoints"
cfg.logging.output_dir = "data/artifacts/demo_sod/fno/outputs"
cfg.data.path = dataset_path
target.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
python -m riemann_ml.ml.fno.train --config fno_demo

# 4) Evaluation (sod + random samples)
python - <<'PY'
from pathlib import Path
from riemann_ml.eval.report import main

main(
    dataset_path=Path("data/processed/demo_sod_small.h5"),
    output_dir=Path("data/artifacts/demo_sod/eval"),
    num_random=3,
)
PY

# Persist environment snapshot
python - <<'PY'
from pathlib import Path
from riemann_ml.utils.repro import save_environment

save_environment(Path("data/artifacts/demo_sod"))
PY

# Cleanup temporary configs
rm -f "${PINN_CONFIG}" "${FNO_CONFIG}"

echo "Demo concluida. Artefatos principais em ${DEMO_DIR}."
