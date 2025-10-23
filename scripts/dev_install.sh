#!/usr/bin/env bash
set -euo pipefail

if command -v hatch >/dev/null 2>&1; then
  echo "[dev_install] Using hatch environment."
  hatch env create || hatch env update
  hatch run pip install -r requirements.txt
else
  echo "[dev_install] Hatch not found, falling back to pip + virtualenv."
  if [ ! -d ".venv" ]; then
    python -m venv .venv
  fi
  ACTIVATE_PATH=".venv/bin/activate"
  if [ ! -f "$ACTIVATE_PATH" ]; then
    ACTIVATE_PATH=".venv/Scripts/activate"
  fi
  if [ ! -f "$ACTIVATE_PATH" ]; then
    echo "[dev_install] Could not locate virtualenv activation script."
    exit 1
  fi
  # shellcheck disable=SC1091
  source "$ACTIVATE_PATH"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

if command -v pre-commit >/dev/null 2>&1; then
  echo "[dev_install] Installing pre-commit hooks."
  pre-commit install
else
  echo "[dev_install] pre-commit not found. Install it to enable git hooks."
fi
