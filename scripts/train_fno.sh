#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME=${CONFIG_NAME:-fno}

python -m riemann_ml.ml.fno.train --config "${CONFIG_NAME}"
