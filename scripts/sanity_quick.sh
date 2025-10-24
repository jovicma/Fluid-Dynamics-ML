#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG_DIR="src/riemann_ml/configs"
SANITY_DIR="data/artifacts/sanity"
PINN_CONFIG="${BASE_CONFIG_DIR}/pinn_sanity.yaml"
FNO_CONFIG="${BASE_CONFIG_DIR}/fno_sanity.yaml"
DATASET_PATH="data/processed/sanity_small.h5"

mkdir -p "${SANITY_DIR}/fvm" "${SANITY_DIR}/pinn" "${SANITY_DIR}/fno"

# 1) FVM simulation for Sod
riemann-ml plot-sod --output-dir "${SANITY_DIR}/fvm"

# 2) Prepare short PINN config and train
python - "$PINN_CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

target = Path(sys.argv[1])
cfg = OmegaConf.load(Path("src/riemann_ml/configs/pinn.yaml"))
cfg.training.max_steps = 500
cfg.logging.log_dir = "data/artifacts/sanity/pinn/logs"
cfg.logging.checkpoint_dir = "data/artifacts/sanity/pinn/checkpoints"
cfg.logging.output_dir = "data/artifacts/sanity/pinn/outputs"
target.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
python -m riemann_ml.ml.pinn.train --config pinn_sanity

# 3) Generate small dataset and train FNO briefly
NUM_SAMPLES=100 CELLS=256 TF=0.2 CFL=0.5 OUT_PATH="${DATASET_PATH}" ./scripts/gen_dataset.sh
python - "$FNO_CONFIG" "$DATASET_PATH" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf

target = Path(sys.argv[1])
dataset_path = sys.argv[2]
cfg = OmegaConf.load(Path("src/riemann_ml/configs/fno.yaml"))
cfg.training.epochs = 2
cfg.training.device = "cpu"
cfg.training.log_interval = 1
cfg.training.checkpoint_interval = 2
cfg.training.eval_interval = 2
cfg.logging.log_dir = "data/artifacts/sanity/fno/logs"
cfg.logging.checkpoint_dir = "data/artifacts/sanity/fno/checkpoints"
cfg.logging.output_dir = "data/artifacts/sanity/fno/outputs"
cfg.data.path = dataset_path
target.write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
PY
python -m riemann_ml.ml.fno.train --config fno_sanity

# 4) Evaluation
./scripts/eval_all.sh

# Persist environment snapshot
python - <<'PY'
from pathlib import Path
from riemann_ml.utils.repro import save_environment

save_environment(Path("data/artifacts/sanity"))
PY

# 5) Cleanup temporary configs
rm -f "${PINN_CONFIG}" "${FNO_CONFIG}"

# 6) Print artifact paths
echo "Artifacts generated under:"
echo "  - ${SANITY_DIR}/fvm"
echo "  - ${SANITY_DIR}/pinn"
echo "  - ${SANITY_DIR}/fno"
echo "  - data/artifacts/eval"
echo "  - ${DATASET_PATH}"
