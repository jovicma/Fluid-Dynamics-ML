#!/usr/bin/env bash
set -euo pipefail

NUM_SAMPLES=${NUM_SAMPLES:-2000}
CELLS=${CELLS:-512}
CFL=${CFL:-0.5}
TF=${TF:-0.2}
SEED=${SEED:-42}
OUT_PATH=${OUT_PATH:-data/processed/sod_like.h5}

python -m riemann_ml.data.generate \
  --num-samples "${NUM_SAMPLES}" \
  --cells "${CELLS}" \
  --cfl "${CFL}" \
  --final-time "${TF}" \
  --seed "${SEED}" \
  --out-path "${OUT_PATH}"
