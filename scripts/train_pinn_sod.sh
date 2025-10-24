#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:-pinn}

python -m riemann_ml.ml.pinn.train --config "${CONFIG_NAME}"
