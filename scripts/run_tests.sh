#!/usr/bin/env bash
set -euo pipefail

if command -v hatch >/dev/null 2>&1; then
  echo "[run_tests] Running tests via hatch."
  hatch run tests
else
  echo "[run_tests] Hatch not found, running pytest directly."
  if [ -d ".venv" ]; then
    ACTIVATE_PATH=".venv/bin/activate"
    if [ ! -f "$ACTIVATE_PATH" ]; then
      ACTIVATE_PATH=".venv/Scripts/activate"
    fi
    if [ -f "$ACTIVATE_PATH" ]; then
      # shellcheck disable=SC1091
      source "$ACTIVATE_PATH"
    else
      echo "[run_tests] Could not locate virtualenv activation script."
    fi
  fi
  export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
  pytest --cov=riemann_ml --cov-report=term-missing "$@"
fi
