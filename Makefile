# Comandos del proyecto. `make help` lista todo.
SHELL := /bin/bash
BACKEND := backend
FRONTEND := frontend
VENV := $(BACKEND)/.venv
PY := $(VENV)/bin/python
SYMBOL ?= BTC

.PHONY: help setup setup-backend setup-frontend dev backend frontend test test-backend test-frontend lint format cli train-regime walk-forward clean

help: ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: setup-backend setup-frontend ## Instala todas las dependencias
	@test -f .env || (cp .env.example .env && echo "Creado .env a partir de .env.example: completá tus keys a mano.")

setup-backend: ## Crea el entorno de Python e instala el backend
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --allow-existing --python 3.12 $(VENV) && uv pip install --python $(PY) -e "$(BACKEND)[dev]"; \
	else \
		python3 -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,14), "Se necesita Python 3.11 a 3.13 (recomendado: brew install python@3.12 o brew install uv)"' && \
		python3 -m venv $(VENV) && $(PY) -m pip install --upgrade pip && $(PY) -m pip install -e "$(BACKEND)[dev]"; \
	fi

setup-frontend: ## Instala las dependencias del frontend
	cd $(FRONTEND) && npm install

dev: ## Levanta backend y frontend juntos (accesible desde la red local)
	./scripts/dev.sh

backend: ## Solo el backend
	cd $(BACKEND) && .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

frontend: ## Solo el frontend
	cd $(FRONTEND) && npm run dev

test: test-backend test-frontend ## Corre todos los tests

test-backend: ## Tests del backend
	cd $(BACKEND) && .venv/bin/pytest

test-frontend: ## Typecheck y tests del frontend
	cd $(FRONTEND) && npm run typecheck && npm test

lint: ## Lint del backend
	cd $(BACKEND) && .venv/bin/ruff check . && .venv/bin/ruff format --check .

format: ## Formatea el backend
	cd $(BACKEND) && .venv/bin/ruff format . && .venv/bin/ruff check --fix .

cli: ## Datos, régimen, score y plan de una moneda. Ej: make cli SYMBOL=ETH ARGS="--offline"
	cd $(BACKEND) && .venv/bin/python -m scripts.analyze_cli $(SYMBOL) $(ARGS)

# Sin SYMBOL (ni en la línea de comandos ni en el entorno) entrena todas las
# favoritas: el BTC por omisión de arriba no cuenta (origin "file").
train-regime: ## Entrena los modelos de régimen (HMM). Ej: make train-regime SYMBOL=BTC ARGS="--offline"
	cd $(BACKEND) && .venv/bin/python -m scripts.train_regime $(if $(filter file,$(origin SYMBOL)),all,$(SYMBOL)) $(ARGS)

walk-forward: ## Validación histórica (Hito 10)
	cd $(BACKEND) && .venv/bin/python -m scripts.walk_forward $(SYMBOL)

clean: ## Borra cachés y builds (no toca data/ ni .env)
	rm -rf $(FRONTEND)/dist $(BACKEND)/.pytest_cache $(BACKEND)/.ruff_cache
	find $(BACKEND) -name __pycache__ -type d -prune -exec rm -rf {} +
