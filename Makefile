.PHONY: dev test fvm-sod train-pinn train-fno eval

PYTHON := python
VENV_PYTHON ?= .venv/Scripts/python.exe

dev:
	@./scripts/dev_install.sh

test:
	@env PYTHONPATH=src $(PYTHON) -m pytest -q --maxfail=1 --disable-warnings

fvm-sod:
	@$(PYTHON) -m riemann_ml plot-sod --output-dir data/artifacts/cli_sod

train-pinn:
	@$(PYTHON) -m riemann_ml.ml.pinn.train --config pinn

train-fno:
	@$(PYTHON) -m riemann_ml.ml.fno.train --config fno

eval:
	@./scripts/eval_all.sh
