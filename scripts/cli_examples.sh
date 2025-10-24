#!/usr/bin/env bash
set -euo pipefail

echo "# Show configuration"
riemann-ml show-config

echo -e "\n# Run FVM simulation and save results"
riemann-ml simulate-fvm --output data/artifacts/cli_run/solution.npz --history

echo -e "\n# Plot Sod comparison"
riemann-ml plot-sod --output-dir data/artifacts/cli_run/plots
