#!/usr/bin/env bash
set -euo pipefail

TARGETS=${*:-"src tests"}

if command -v hatch >/dev/null 2>&1; then
  echo "[format] Formatting via hatch."
  hatch run format
else
  echo "[format] Hatch not found, running tools directly."
  if [ -d ".venv" ]; then
    ACTIVATE_PATH=".venv/bin/activate"
    if [ ! -f "$ACTIVATE_PATH" ]; then
      ACTIVATE_PATH=".venv/Scripts/activate"
    fi
    if [ -f "$ACTIVATE_PATH" ]; then
      # shellcheck disable=SC1091
      source "$ACTIVATE_PATH"
    else
      echo "[format] Could not locate virtualenv activation script."
    fi
  fi
  if command -v ruff >/dev/null 2>&1; then
    ruff check "$TARGETS"
    ruff format "$TARGETS"
  fi
  if command -v black >/dev/null 2>&1; then
    black $TARGETS
  fi
fi
