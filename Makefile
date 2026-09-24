.DEFAULT_GOAL := help
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYTEST  := $(VENV)/bin/pytest
RUFF    := $(VENV)/bin/ruff
MYPY    := $(VENV)/bin/mypy

THRESHOLD ?= 0.65
TOP_K     ?= 20

# Ports come from .env so `make` and `docker compose` never disagree.
# Override per-invocation: `make up CRIBRIX_API_PORT=9000`
-include .env
export
CRIBRIX_API_PORT ?= 8000
CRIBRIX_DB_PORT  ?= 5432
BASE_URL         := http://localhost:$(CRIBRIX_API_PORT)

.PHONY: help
help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

$(VENV): pyproject.toml
	python3 -m venv $(VENV)
	$(PIP) install --quiet --upgrade pip
	$(PIP) install --quiet -e ".[dev]"
	@touch $(VENV)

.PHONY: install
install: $(VENV) ## Create venv and install with dev extras

.PHONY: test
test: $(VENV) ## Run unit tests (no external services required)
	$(PYTEST) tests/unit -q

.PHONY: test-all
test-all: $(VENV) ## Run unit + integration tests (needs a live pgvector)
	CRIBRIX_TEST_DATABASE_URL="postgresql+asyncpg://cribrix:cribrix@localhost:$(CRIBRIX_DB_PORT)/cribrix" \
		$(PYTEST) tests -q

.PHONY: verify
verify: $(VENV) ## Run everything: lint, types, tests, scenarios, eval
	@$(MAKE) --no-print-directory lint
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory scenarios
	@$(MAKE) --no-print-directory eval
	@echo ""
	@echo "  All checks passed."

.PHONY: cov
cov: $(VENV) ## Unit tests with a coverage report
	$(PYTEST) tests/unit --cov=cribrix --cov-report=term-missing --cov-report=html

.PHONY: scenarios
scenarios: $(VENV) ## Run the 3 demo scenarios offline (deterministic)
	$(PY) -m cribrix.evaluation.scenarios

.PHONY: scenarios-live
scenarios-live: $(VENV) ## Run the 3 demo scenarios against real Jev + LLM APIs
	$(PY) -m cribrix.evaluation.scenarios --live

.PHONY: eval
eval: $(VENV) ## Run the evaluation harness (baseline vs. cribrix)
	$(PY) -m cribrix.evaluation.runner --threshold $(THRESHOLD) --top-k $(TOP_K)

.PHONY: eval-sweep
eval-sweep: $(VENV) ## Sweep relevance thresholds to find the calibration point
	@for t in 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do \
		echo "=== threshold $$t ==="; \
		$(PY) -m cribrix.evaluation.runner --threshold $$t | head -11; \
	done

.PHONY: lint
lint: $(VENV) ## Lint and type-check
	$(RUFF) check cribrix tests
	$(RUFF) format --check cribrix tests
	$(MYPY) cribrix

.PHONY: fmt
fmt: $(VENV) ## Auto-format and apply safe lint fixes
	$(RUFF) check --fix cribrix tests
	$(RUFF) format cribrix tests

.PHONY: run
run: $(VENV) ## Run the API locally with hot reload
	$(VENV)/bin/uvicorn cribrix.main:app --reload --port 8000

.PHONY: up
up: ## Start the full Docker stack
	@$(MAKE) --no-print-directory check-ports
	docker compose up --build -d
	@echo ""
	@echo "  API docs -> $(BASE_URL)/docs"
	@echo "  Health   -> $(BASE_URL)/health"
	@echo "  Next     -> make seed"

.PHONY: check-ports
check-ports: ## Warn if the configured ports are in use by something else
	@for p in $(CRIBRIX_API_PORT) $(CRIBRIX_DB_PORT); do \
		if lsof -nP -iTCP:$$p -sTCP:LISTEN >/dev/null 2>&1; then \
			if docker compose ps --format '{{.Ports}}' 2>/dev/null | grep -q ":$$p->"; then \
				continue; \
			fi; \
			owner=$$(lsof -nP -iTCP:$$p -sTCP:LISTEN -Fc 2>/dev/null | grep '^c' | head -1 | cut -c2-); \
			echo ""; \
			echo "  WARNING: port $$p is in use by '$$owner' (not a Cribrix container)."; \
			echo "  Edit CRIBRIX_API_PORT / CRIBRIX_DB_PORT in .env, then re-run."; \
			echo ""; \
		fi; \
	done

.PHONY: ask
ask: $(VENV) ## Ask one question: make ask Q="what is the refund window?"
	@test -n "$(Q)" || (echo 'Usage: make ask Q="your question"'; exit 1)
	@curl -s $(BASE_URL)/query -H 'content-type: application/json' \
		-d '{"query": $(shell printf '%s' '$(Q)' | $(PY) -c "import json,sys; print(json.dumps(sys.stdin.read()))")}' \
		| $(PY) -m cribrix.evaluation.explain

.PHONY: down
down: ## Stop the stack
	docker compose down

.PHONY: clean-volumes
clean-volumes: ## Stop the stack and delete the database volume
	docker compose down -v

.PHONY: logs
logs: ## Tail API logs
	docker compose logs -f api

.PHONY: seed
seed: $(VENV) ## Load the demo corpus and run a guided tour
	$(PY) scripts/seed.py

.PHONY: reseed
reseed: $(VENV) ## Wipe and reload the demo corpus
	$(PY) scripts/seed.py --reset

.PHONY: clean
clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage dist build
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +
