#!/usr/bin/env bash
set -euo pipefail

BASE_CONFIG_DIR="src/riemann_ml/configs"
SANITY_DIR="data/artifacts/sanity"
PINN_SANITY_CFG="${BASE_CONFIG_DIR}/pinn_sanity.yaml"
FNO_SANITY_CFG="${BASE_CONFIG_DIR}/fno_sanity.yaml"
DATASET_PATH="data/processed/sanity_small.h5"

mkdir -p "${SANITY_DIR}/fvm" "${SANITY_DIR}/pinn" "${SANITY_DIR}/fno"

# 1) FVM simulation for Sod
riemann-ml plot-sod --output-dir "${SANITY_DIR}/fvm"

# 2) Prepare short PINN config and train
python - <<'PY' > "${PINN_SANITY_CFG}"
from pathlib import Path
from omegaconf import OmegaConf

cfg_path = Path("src/riemann_ml/configs/pinn.yaml")
cfg = OmegaConf.load(cfg_path)
cfg.training.max_steps = 500
cfg.logging.log_dir = "data/artifacts/sanity/pinn/logs"
cfg.logging.checkpoint_dir = "data/artifacts/sanity/pinn/checkpoints"
cfg.logging.output_dir = "data/artifacts/sanity/pinn/outputs"
print(OmegaConf.to_yaml(cfg))
PY
python -m riemann_ml.ml.pinn.train --config pinn_sanity

# 3) Generate small dataset and train FNO briefly
NUM_SAMPLES=100 CELLS=256 TF=0.2 CFL=0.5 OUT_PATH="${DATASET_PATH}" ./scripts/gen_dataset.sh
python - <<'PY' > "${FNO_SANITY_CFG}"
from pathlib import Path
from omegaconf import OmegaConf

cfg_path = Path("src/riemann_ml/configs/fno.yaml")
cfg = OmegaConf.load(cfg_path)
cfg.training.epochs = 2
cfg.training.device = "cpu"
cfg.training.log_interval = 1
cfg.training.checkpoint_interval = 2
cfg.training.eval_interval = 2
cfg.logging.log_dir = "data/artifacts/sanity/fno/logs"
cfg.logging.checkpoint_dir = "data/artifacts/sanity/fno/checkpoints"
cfg.logging.output_dir = "data/artifacts/sanity/fno/outputs"
cfg.data.path = "data/processed/sanity_small.h5"
print(OmegaConf.to_yaml(cfg))
PY
python -m riemann_ml.ml.fno.train --config fno_sanity

# 4) Evaluation
./scripts/eval_all.sh

# 5) Cleanup temporary configs
rm -f "${PINN_SANITY_CFG}" "${FNO_SANITY_CFG}"

# 6) Print artifact paths
echo "Artifacts generated under:"
echo "  - ${SANITY_DIR}/fvm"
echo "  - ${SANITY_DIR}/pinn"
echo "  - ${SANITY_DIR}/fno"
echo "  - data/artifacts/eval"
echo "  - ${DATASET_PATH}"
